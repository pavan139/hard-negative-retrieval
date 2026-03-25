import json
import math
from pathlib import Path
import tempfile
import unittest

from hard_negative_retrieval.data import Document
from hard_negative_retrieval.index_benchmark import benchmark, challenging_dataset
from hard_negative_retrieval.late_interaction import IVFIndex, LateInteractionRetriever, maxsim
from hard_negative_retrieval.model import DualEncoder, TrainingConfig


def controlled_encoder():
    """Hand-specified vector fixture, never reported as a trained benchmark."""
    documents = (Document("best", "d1 d2 dneg dneg dneg"), Document("other", "dmid"))
    model = DualEncoder(documents, TrainingConfig(dimension=2, epochs=1))
    model.vocabulary = {word: index for index, word in enumerate(("q1", "q2", "d1", "d2", "dneg", "dmid"))}
    model.query_embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]
    model.document_embeddings = [row[:] for row in model.query_embeddings]
    return model


class MaxSimAndIndexTests(unittest.TestCase):
    def test_maxsim_matches_independent_scalar_math_including_negative_scores(self):
        query = ((1.0, 0.0), (0.0, 1.0))
        documents = ((0.6, 0.8), (-1.0, 0.0))
        self.assertAlmostEqual(maxsim(query, documents), 0.6 + 0.8)
        self.assertEqual(maxsim(((1.0, 0.0),), ((-1.0, 0.0),)), -1.0)
        self.assertAlmostEqual(maxsim(query + query[:1], documents), 2.0)

    def test_exact_reranking_finds_multitoken_match_hidden_by_coarse_mean(self):
        retriever = LateInteractionRetriever(controlled_encoder())
        result = retriever.search("q1 q2", 2)
        self.assertEqual(result.rankings[0], ("best", 2.0))
        self.assertAlmostEqual(result.rankings[1][1], math.sqrt(2))
        self.assertEqual(result.maxsim_token_dot_products, 12)
        index = IVFIndex.build(retriever, clusters=2)
        approximate = index.search("q1 q2", 1, probes=1)
        exhaustive = index.search("q1 q2", 2, probes=2)
        self.assertEqual(approximate.rankings[0][0], "other")
        self.assertEqual(exhaustive.rankings, result.rankings)
        self.assertLess(approximate.maxsim_token_dot_products, result.maxsim_token_dot_products)

    def test_candidate_budget_can_lose_a_match_even_when_all_clusters_are_probed(self):
        retriever = LateInteractionRetriever(controlled_encoder())
        result = IVFIndex.build(retriever, clusters=2).search("q1 q2", 1, probes=2, max_candidates=1)
        self.assertEqual(result.rankings[0][0], "other")
        self.assertEqual(result.candidates_before_budget, 2)
        self.assertEqual(result.candidates_reranked, 1)

    def test_oov_queries_abstain_without_arbitrary_hits(self):
        retriever = LateInteractionRetriever(controlled_encoder())
        self.assertEqual(retriever.search("unknownword").rankings, ())
        self.assertEqual(IVFIndex.build(retriever, clusters=2).search("unknownword").rankings, ())

    def test_probe_and_candidate_budgets_are_strict(self):
        index = IVFIndex.build(LateInteractionRetriever(controlled_encoder()), clusters=2)
        for settings in ({"probes": 0}, {"probes": 3}, {"probes": True}, {"max_candidates": 0}, {"max_candidates": 3}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                index.search("q1 q2", **settings)

    def test_frozen_encoder_does_not_change_after_source_model_is_modified(self):
        model = controlled_encoder()
        retriever = LateInteractionRetriever(model)
        before = retriever.search("q1 q2").rankings
        model.document_embeddings[2] = [-1.0, -1.0]
        self.assertEqual(retriever.search("q1 q2").rankings, before)

    def test_empty_and_nonfinite_vectors_rejected(self):
        for query, document in (((), ((1.0,),)), (((math.nan,),), ((1.0,),)), (((1.0, 0.0),), ((1.0,),))):
            with self.subTest(query=query), self.assertRaises(ValueError):
                maxsim(query, document)

    def test_full_probe_is_exact_and_kmeans_is_reproducible(self):
        data = challenging_dataset()
        model = DualEncoder(data.documents, TrainingConfig(epochs=1, dimension=4)).fit(data.train)
        retriever = LateInteractionRetriever(model)
        one, two = IVFIndex.build(retriever, clusters=4), IVFIndex.build(retriever, clusters=4)
        self.assertEqual(one.centroids, two.centroids)
        self.assertEqual(one.postings, two.postings)
        for query in data.test[::17]:
            self.assertEqual(one.search(query.text, 5, probes=4).rankings, retriever.search(query.text, 5).rankings)

    def test_index_checkpoint_binding_and_partition_validation(self):
        model = controlled_encoder()
        retriever = LateInteractionRetriever(model)
        index = IVFIndex.build(retriever, clusters=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index.save(path)
            restored = IVFIndex.load(path, retriever)
            self.assertEqual(restored.search("q1 q2", probes=2).rankings, index.search("q1 q2", probes=2).rankings)
            model.document_embeddings[0][0] += 0.3
            with self.assertRaisesRegex(ValueError, "mismatch"):
                IVFIndex.load(path, LateInteractionRetriever(model))
            raw = json.loads(path.read_text())
            raw["postings"][0].append(raw["postings"][0][0])
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "partition"):
                IVFIndex.load(path, retriever)

    def test_relation_swapped_documents_are_an_explicit_architectural_failure(self):
        data = challenging_dataset()
        model = DualEncoder(data.documents, TrainingConfig(epochs=1, dimension=4)).fit(data.train)
        retriever = LateInteractionRetriever(model)
        self.assertEqual(
            sorted(retriever.token_vectors["binding-00-a"]), sorted(retriever.token_vectors["binding-00-b"])
        )
        query = retriever.encode("project0 blue teams export tables")
        self.assertEqual(
            maxsim(query, retriever.token_vectors["binding-00-a"]),
            maxsim(query, retriever.token_vectors["binding-00-b"]),
        )

    def test_benchmark_keeps_difficult_slice_and_reports_oracle_recall_separately(self):
        result = benchmark(epochs=1, clusters=4, queries=8)
        self.assertEqual(result["oracle"]["slices"]["relation_binding"]["query_count"], 4)
        full = next(
            row
            for row in result["settings"]
            if row["probes"] == 4 and row["max_candidates"] == result["document_count"]
        )
        self.assertEqual(full["mean_oracle_recall_at_5"], 1.0)
        self.assertLessEqual(result["oracle"]["slices"]["relation_binding"]["ground_truth_recall_at_1"], 0.5)


if __name__ == "__main__":
    unittest.main()
