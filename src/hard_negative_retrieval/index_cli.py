from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .data import load_dataset
from .index_benchmark import benchmark
from .late_interaction import IVFIndex, LateInteractionRetriever
from .model import DualEncoder


def main(argv=None):
    parser = argparse.ArgumentParser(description="Learned-token MaxSim, spherical IVF and exact candidate reranking")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--epochs", type=int, default=8)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--clusters", type=int, default=8)
    bench.add_argument("--queries", type=int, default=128)
    bench.add_argument("--artifact-directory", type=Path)
    for name in ("build", "search"):
        child = sub.add_parser(name)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--index", type=Path, required=True)
        if name == "build":
            child.add_argument("--clusters", type=int, default=8)
            child.add_argument("--seed", type=int, default=7)
        else:
            child.add_argument("--query", required=True)
            child.add_argument("--probes", type=int, default=1)
            child.add_argument("--max-candidates", type=int)
            child.add_argument("--top-k", type=int, default=5)
    for child in (bench, *[action for action in sub.choices.values() if action is not bench]):
        child.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            result = benchmark(
                epochs=args.epochs,
                seed=args.seed,
                clusters=args.clusters,
                queries=args.queries,
                artifact_directory=args.artifact_directory,
            )
        else:
            data = load_dataset(args.dataset)
            retriever = LateInteractionRetriever(DualEncoder.load(args.checkpoint, data.documents))
            if args.command == "build":
                index = IVFIndex.build(retriever, clusters=args.clusters, seed=args.seed)
                index.save(args.index)
                result = {
                    "index": str(args.index),
                    "encoder_sha256": retriever.encoder_sha256,
                    "training": index.training,
                }
            else:
                index = IVFIndex.load(args.index, retriever)
                result = {
                    "encoder_sha256": retriever.encoder_sha256,
                    **asdict(
                        index.search(args.query, args.top_k, probes=args.probes, max_candidates=args.max_candidates)
                    ),
                }
        encoded = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded)
        return 2 if result.get("full_probe_consistency_passed") is False else 0
    except (ValueError, TypeError, OSError) as error:
        print(json.dumps({"error": "index_experiment_failed", "detail": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
