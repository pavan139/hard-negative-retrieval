"""Reproducible CPU experiments. Errors are explicit; there is no synthetic success fallback."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from .data import load_dataset, synthetic_dataset
from .metrics import evaluate_retriever
from .model import DualEncoder, TrainingConfig
from .retrieval import LexicalRetriever, TokenOverlapRetriever


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    if selected and selected[0] == "index":
        from .index_cli import main as index_main

        return index_main(selected[1:])
    parser = argparse.ArgumentParser(
        description="Train a small dual encoder and compare honest synthetic retrieval metrics."
    )
    parser.add_argument(
        "command", choices=("demo", "evaluate", "index"), help="Use index --help for MaxSim/IVF experiments"
    )
    parser.add_argument("--output", type=Path, help="Write JSON, also printed to stdout.")
    parser.add_argument("--dataset", type=Path, help="Validated JSON corpus and disjoint train/dev/test query splits.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int, help="Override config, range 1..200.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Save actual trained embedding tables as JSON.")
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="Evaluate saved weights without retraining; corpus and training labels must match.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also train the opposite negative strategy with the same seed and training split.",
    )
    args = parser.parse_args(argv)
    try:
        if args.ablation and args.command != "evaluate":
            raise ValueError("ablation_requires_evaluate_command")
        dataset = load_dataset(args.dataset) if args.dataset else synthetic_dataset()
        raw_config = {}
        if args.config:
            with args.config.open("rb") as stream:
                encoded = stream.read(65537)
            if len(encoded) > 65536:
                raise ValueError("config_too_large")
            raw_config = json.loads(encoded)
            if not isinstance(raw_config, dict):
                raise ValueError("invalid_config_schema")
        if args.epochs is not None:
            raw_config["epochs"] = args.epochs
        if args.seed is not None:
            raw_config["seed"] = args.seed
        config = TrainingConfig(**raw_config)
        if args.load_checkpoint:
            if args.ablation or args.epochs is not None or args.seed is not None or args.config:
                raise ValueError("checkpoint_loading_cannot_be_combined_with_training_overrides_or_ablation")
            model = DualEncoder.load(args.load_checkpoint, dataset.documents)
            serialized_train = json.dumps(
                [{"id": q.id, "text": q.text, "relevant": q.relevant} for q in dataset.train], sort_keys=True
            )
            if model.trace["training_query_sha256"] != hashlib.sha256(serialized_train.encode()).hexdigest():
                raise ValueError("checkpoint_training_split_mismatch")
            config = model.config
        else:
            model = DualEncoder(dataset.documents, config).fit(dataset.train)
        lexical = LexicalRetriever(dataset.documents)
        retrievers = {
            "token_overlap": TokenOverlapRetriever(dataset.documents),
            "tfidf_cosine": lexical,
            "trained_dual_encoder": model,
        }
        result = {
            "project": "hard-negative-retrieval",
            "dataset_source": dataset.source,
            "dataset_sha256": dataset.fingerprint(),
            "synthetic": args.dataset is None or dataset.source.startswith("synthetic"),
            "document_count": len(dataset.documents),
            "split_sizes": {split: len(getattr(dataset, split)) for split in ("train", "dev", "test")},
            "training": model.trace,
            "limitations": [
                "Closed-corpus synthetic query-template benchmark, not evidence of pretrained or real-world accuracy.",
                "Training sees corpus text and train queries only; dev/test labels never enter training.",
                "Scores are retrieval-only; no LLM generation is performed.",
            ],
        }
        if args.command == "evaluate":
            result["evaluation"] = {
                split: {
                    name: evaluate_retriever(retriever, getattr(dataset, split))
                    for name, retriever in retrievers.items()
                }
                for split in ("dev", "test")
            }
            if args.ablation:
                other_strategy = "random" if config.negative_strategy == "hard" else "hard"
                comparison = DualEncoder(dataset.documents, replace(config, negative_strategy=other_strategy)).fit(
                    dataset.train
                )
                result["negative_strategy_ablation"] = {
                    "comparison_training": comparison.trace,
                    "comparison_evaluation": {
                        split: evaluate_retriever(comparison, getattr(dataset, split)) for split in ("dev", "test")
                    },
                    "controlled_factors": [
                        "corpus",
                        "training queries and labels",
                        "initialization seed",
                        "embedding dimension",
                        "epochs",
                        "negative count",
                        "learning rate",
                        "temperature",
                        "query update order",
                    ],
                    "limitation": "A single small synthetic run does not establish a general advantage; compare several seeds and independent data before a general claim.",
                }
        else:
            result["examples"] = [
                {
                    "query_id": query.id,
                    "query": query.text,
                    "relevant": query.relevant,
                    "rankings": {
                        name: [
                            {"document_id": doc_id, "score": score} for doc_id, score in retriever.rank(query.text, 3)
                        ]
                        for name, retriever in retrievers.items()
                    },
                }
                for query in dataset.test[:4]
            ]
        if args.checkpoint:
            model.save(args.checkpoint)
        serialized = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        print(serialized)
        return 0
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": "experiment_failed", "detail": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
