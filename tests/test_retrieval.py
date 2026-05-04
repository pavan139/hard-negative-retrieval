import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hard_negative_retrieval.data import Document, Query, dataset_from_dict, load_dataset, synthetic_dataset
from hard_negative_retrieval.metrics import evaluate_retriever, ranking_metrics
from hard_negative_retrieval.model import DualEncoder, TrainingConfig, contrastive_loss_gradients
from hard_negative_retrieval.retrieval import LexicalRetriever, TokenOverlapRetriever, mine_hard_negatives


class GradientTests(unittest.TestCase):
    def test_query_gradient_matches_central_difference(self):
        query = [0.3, -0.2, 0.4]
        docs = [[0.2, 0.5, -0.1], [-0.4, 0.1, 0.7], [0.6, 0.1, -0.3]]
        _, analytical, _ = contrastive_loss_gradients(query, docs, 0.7)
        for index in range(len(query)):
            plus, minus = query[:], query[:]
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical = (
                contrastive_loss_gradients(plus, docs, 0.7)[0] - contrastive_loss_gradients(minus, docs, 0.7)[0]
            ) / 2e-6
            self.assertAlmostEqual(numerical, analytical[index], places=7)

    def test_document_gradients_match_central_difference(self):
        query, docs = [0.3, -0.2], [[0.2, 0.5], [-0.4, 0.1]]
        _, _, analytical = contrastive_loss_gradients(query, docs, 0.3)
        for doc_index in range(len(docs)):
            for coordinate in range(len(query)):
                plus, minus = copy.deepcopy(docs), copy.deepcopy(docs)
                plus[doc_index][coordinate] += 1e-6
                minus[doc_index][coordinate] -= 1e-6
                numerical = (
                    contrastive_loss_gradients(query, plus, 0.3)[0] - contrastive_loss_gradients(query, minus, 0.3)[0]
                ) / 2e-6
                self.assertAlmostEqual(numerical, analytical[doc_index][coordinate], places=7)

    def test_logsumexp_stability_and_nonfinite_rejection(self):
        loss, _, _ = contrastive_loss_gradients([1000.0], [[1000.0], [-1000.0]], 0.1)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(loss, 0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            contrastive_loss_gradients([math.inf], [[1.0]], 0.2)

    def test_temperature_and_dimensions_are_validated(self):
        for temperature in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                contrastive_loss_gradients([1.0], [[1.0]], temperature)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            contrastive_loss_gradients([1.0], [[1.0, 2.0]], 1)


class MetricTests(unittest.TestCase):
    def test_metrics_with_multiple_relevant_documents(self):
        measured = ranking_metrics(["x", "b", "a"], ("a", "b"), 2)
        self.assertEqual(measured["recall"], 0.5)
        self.assertEqual(measured["mrr"], 0.5)
        expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(measured["ndcg"], expected_ndcg)

    def test_no_hits_and_perfect_ranking(self):
        self.assertEqual(ranking_metrics([], ("a",), 5), {"recall": 0, "mrr": 0, "ndcg": 0})
        self.assertEqual(ranking_metrics(["a", "b"], ("a", "b"), 2), {"recall": 1, "mrr": 1, "ndcg": 1})

    def test_duplicate_results_cannot_inflate_recall(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ranking_metrics(["a", "a"], ("a",), 2)

    def test_empty_labels_and_bad_cutoff_fail(self):
        for relevant, k in (((), 1), (("a",), 0), (("a", "a"), 2), (("a",), True)):
            with self.assertRaises(ValueError):
                ranking_metrics(["a"], relevant, k)


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.dataset = synthetic_dataset()

    def test_deterministic_data_and_split_denominators(self):
        self.assertEqual(self.dataset.fingerprint(), synthetic_dataset().fingerprint())
        self.assertEqual(len(self.dataset.documents), 32)
        self.assertEqual(tuple(map(len, (self.dataset.train, self.dataset.dev, self.dataset.test))), (96, 32, 64))

    def test_normalized_query_overlap_is_rejected(self):
        collision = Query("new-id", "  " + self.dataset.train[0].text.upper() + "  ", self.dataset.train[0].relevant)
        with self.assertRaisesRegex(ValueError, "split_leakage"):
            replace(self.dataset, test=(collision, *self.dataset.test[1:]))

    def test_unknown_relevance_and_duplicate_document_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown_relevant"):
            replace(self.dataset, test=(Query("missing", "a valid new question", ("missing",)),))
        with self.assertRaisesRegex(ValueError, "duplicate_document"):
            replace(self.dataset, documents=(*self.dataset.documents, self.dataset.documents[0]))

    def test_blank_and_oversized_contexts_are_rejected(self):
        for text in ("", " " * 10, "x" * 5001):
            with self.assertRaises(ValueError):
                replace(self.dataset, documents=(Document("bad", text), *self.dataset.documents[1:]))

    def test_json_roundtrip_and_bad_schema(self):
        restored = dataset_from_dict(json.loads(json.dumps(self.dataset.to_dict())))
        self.assertEqual(self.dataset.fingerprint(), restored.fingerprint())
        with self.assertRaisesRegex(ValueError, "schema"):
            dataset_from_dict({"documents": []})

    def test_json_file_reads_are_bounded_and_invalid_json_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text("x" * 20)
            with patch("hard_negative_retrieval.data.MAX_DATASET_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "too_large"):
                    load_dataset(path)
            with self.assertRaisesRegex(ValueError, "invalid_dataset_json"):
                load_dataset(path)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.documents = (Document("a", "apple fruit"), Document("b", "truck vehicle"))
        self.train = (Query("train-apple", "orchard edible", ("a",)), Query("train-truck", "road engine", ("b",)))
        self.config = TrainingConfig(seed=4, epochs=18, dimension=8, learning_rate=0.7)

    def test_training_updates_both_tables_and_reduces_loss(self):
        model = DualEncoder(self.documents, self.config).fit(self.train)
        losses = model.trace["epoch_mean_training_loss"]
        self.assertLess(losses[-1], losses[0])
        self.assertNotEqual(model.query_embeddings, model.document_embeddings)
        self.assertEqual(model.rank("orchard edible", 1)[0][0], "a")
        self.assertEqual(model.rank("road engine", 1)[0][0], "b")

    def test_training_is_deterministic(self):
        first = DualEncoder(self.documents, self.config).fit(self.train)
        second = DualEncoder(self.documents, self.config).fit(self.train)
        self.assertEqual(first.query_embeddings, second.query_embeddings)
        self.assertEqual(first.trace, second.trace)

    def test_heldout_query_tokens_do_not_enter_vocabulary(self):
        model = DualEncoder(self.documents, self.config).fit(self.train)
        self.assertNotIn("heldoutcanary", model.vocabulary)
        self.assertEqual(model.rank("heldoutcanary", 1), [])
        self.assertEqual(model.trace["train_query_count"], 2)

    def test_mining_excludes_all_positives(self):
        docs = (*self.documents, Document("c", "apple apple fruit"))
        miner = LexicalRetriever(docs)
        negatives = mine_hard_negatives(miner, "apple", ("a", "c"), 5, len(docs))
        self.assertEqual(negatives, ["b"])

    def test_no_negative_training_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no_negative"):
            DualEncoder(self.documents, self.config).fit((Query("q", "all relevant", ("a", "b")),))

    def test_duplicate_training_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate_training_query"):
            DualEncoder(self.documents, self.config).fit((self.train[0], self.train[0]))

    def test_compute_budget_accounts_for_negative_candidate_work(self):
        excessive = TrainingConfig(epochs=200, dimension=128, negatives=50)
        many_queries = tuple(Query(f"q-{i}", f"orchard edible {i}", ("a",)) for i in range(100))
        with self.assertRaisesRegex(ValueError, "compute_budget"):
            DualEncoder(self.documents, excessive).fit(many_queries)

    def test_checkpoint_roundtrip_preserves_rank_and_checks_corpus(self):
        model = DualEncoder(self.documents, self.config).fit(self.train)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = DualEncoder.load(path, self.documents)
            self.assertEqual(model.rank("orchard"), restored.rank("orchard"))
            with self.assertRaisesRegex(ValueError, "corpus_mismatch"):
                DualEncoder.load(path, (Document("a", "modified"), self.documents[1]))
            raw = json.loads(path.read_text())
            raw["query_embeddings"][0][0] = math.nan
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "weights"):
                DualEncoder.load(path, self.documents)

    def test_configuration_budgets_are_enforced(self):
        for override in (
            {"epochs": 0},
            {"dimension": 129},
            {"seed": True},
            {"temperature": 0},
            {"learning_rate": math.inf},
        ):
            with self.assertRaises(ValueError):
                TrainingConfig(**override)

    def test_baselines_and_evaluation_denominators(self):
        query = Query("q", "apple", ("a",))
        for baseline in (LexicalRetriever(self.documents), TokenOverlapRetriever(self.documents)):
            measured = evaluate_retriever(baseline, (query,))
            self.assertEqual(measured["query_count"], 1)
            self.assertEqual(measured["relevance_label_count"], 1)
            self.assertEqual(measured["metrics"]["recall@1"], 1)


if __name__ == "__main__":
    unittest.main()
