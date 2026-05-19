"""Numerical, schema, metric and ablation regressions found during review."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from hard_negative_retrieval.__main__ import main
from hard_negative_retrieval.data import Document, Query, dataset_from_dict, synthetic_dataset
from hard_negative_retrieval.metrics import evaluate_retriever
from hard_negative_retrieval.model import DualEncoder, TrainingConfig, contrastive_loss_gradients
from hard_negative_retrieval.retrieval import LexicalRetriever


class RetrievalReviewTests(unittest.TestCase):
    def setUp(self):
        self.documents = (Document("a", "apple orchard"), Document("b", "truck engine"), Document("c", "apple fruit"))
        self.train = (Query("q1", "edible orchard", ("a", "c")), Query("q2", "vehicle engine", ("b",)))

    def test_duplicate_cutoffs_cannot_inflate_metrics_above_one(self):
        with self.assertRaisesRegex(ValueError, "cutoffs"):
            evaluate_retriever(LexicalRetriever(self.documents), (self.train[0],), (1, 1))

    def test_empty_and_malformed_cutoffs_fail_before_retrieval(self):
        for cutoffs in ((), (0,), (True,), ([1],)):
            with self.subTest(cutoffs=cutoffs), self.assertRaises(ValueError):
                evaluate_retriever(LexicalRetriever(self.documents), (self.train[0],), cutoffs)

    def test_finite_inputs_that_overflow_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            contrastive_loss_gradients([1e308], [[1e308], [-1e308]], 0.2)

    def test_boolean_and_huge_numeric_configuration_rejected(self):
        for overrides in (
            {"learning_rate": True},
            {"temperature": True},
            {"learning_rate": 10**1000},
            {"temperature": "warm"},
        ):
            with self.subTest(overrides=str(overrides)[:80]), self.assertRaises(ValueError):
                TrainingConfig(**overrides)

    def test_scalar_relevance_string_is_not_silently_split_into_characters(self):
        raw = {
            "documents": [{"id": "a", "text": "apple orchard"}, {"id": "b", "text": "truck engine"}],
            "train": [{"id": "train", "text": "training orchard", "relevant": "a"}],
            "dev": [{"id": "dev", "text": "development orchard", "relevant": ["a"]}],
            "test": [{"id": "test", "text": "test engine", "relevant": ["b"]}],
        }
        with self.assertRaisesRegex(ValueError, "schema"):
            dataset_from_dict(raw)

    def test_zero_information_training_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty_training"):
            DualEncoder(self.documents).fit((Query("q", "the and of", ("a",)),))

    def test_random_negative_training_is_deterministic_and_excludes_other_positives(self):
        config = TrainingConfig(seed=8, epochs=4, dimension=8, negatives=2, negative_strategy="random")
        one = DualEncoder(self.documents, config).fit(self.train)
        two = DualEncoder(self.documents, config).fit(self.train)
        self.assertEqual(one.query_embeddings, two.query_embeddings)
        # q1 has one available negative, q2 has two; each epoch must use exactly three.
        self.assertEqual(one.trace["negative_pairs_used"], 12)
        self.assertEqual(one.trace["negative_strategy"], "random")

    def test_ablation_keeps_query_update_order_identical(self):
        config = TrainingConfig(seed=8, epochs=4, dimension=8, negatives=2)
        hard = DualEncoder(self.documents, config).fit(self.train)
        random = DualEncoder(self.documents, replace(config, negative_strategy="random")).fit(self.train)
        self.assertEqual(hard.trace["epoch_order_sha256"], random.trace["epoch_order_sha256"])

    def test_checkpoint_trace_shape_and_oversized_integer_weights_rejected(self):
        model = DualEncoder(self.documents, TrainingConfig(epochs=1, dimension=4)).fit(self.train)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            model.save(path)
            original = json.loads(path.read_text())
            for mutated in ({**original, "trace": []}, {**original, "query_embeddings": {}}):
                path.write_text(json.dumps(mutated))
                with self.assertRaises(ValueError):
                    DualEncoder.load(path, self.documents)
            original["query_embeddings"][0][0] = 10**1000
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                DualEncoder.load(path, self.documents)

    def test_nonfinite_rank_scores_are_rejected(self):
        class BadRetriever:
            def rank(self, text, k=10):
                return [("a", math.nan)]

        with self.assertRaisesRegex(ValueError, "ranked_result"):
            evaluate_retriever(BadRetriever(), (self.train[0],))

    def test_case_rankings_reconstruct_reported_metrics(self):
        report = evaluate_retriever(LexicalRetriever(self.documents), self.train, (1, 3))
        self.assertEqual(len(report["cases"]), 2)
        for key, aggregate in report["metrics"].items():
            self.assertAlmostEqual(aggregate, sum(row["metrics"][key] for row in report["cases"]) / 2)

    def test_cli_ablation_and_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = path / "weights.json"
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "evaluate",
                        "--epochs",
                        "1",
                        "--ablation",
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(path / "ablation.json"),
                    ]
                )
                loaded_status = main(
                    ["evaluate", "--load-checkpoint", str(checkpoint), "--output", str(path / "loaded.json")]
                )
            self.assertEqual((status, loaded_status), (0, 0))
            ablation = json.loads((path / "ablation.json").read_text())
            loaded = json.loads((path / "loaded.json").read_text())
            self.assertEqual(
                ablation["negative_strategy_ablation"]["comparison_training"]["negative_strategy"], "random"
            )
            self.assertEqual(ablation["evaluation"], loaded["evaluation"])

    def test_cli_loaded_checkpoint_must_match_training_split(self):
        dataset = synthetic_dataset()
        model = DualEncoder(dataset.documents, TrainingConfig(epochs=1, dimension=4)).fit(dataset.train[:-1])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.json"
            model.save(checkpoint)
            error = io.StringIO()
            with redirect_stderr(error):
                result = main(["evaluate", "--load-checkpoint", str(checkpoint)])
            self.assertEqual(result, 2)
            self.assertIn("training_split_mismatch", error.getvalue())


if __name__ == "__main__":
    unittest.main()
