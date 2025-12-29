"""Controlled retrieval/index experiments including an architectural hard case."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import statistics
import time

from .data import Dataset, Document, Query, synthetic_dataset
from .late_interaction import IVFIndex, LateInteractionRetriever
from .model import DualEncoder, TrainingConfig


def challenging_dataset() -> Dataset:
    base = synthetic_dataset()
    documents, train, dev, test = list(base.documents), list(base.train), list(base.dev), list(base.test)
    for index in range(32):
        project = f"project{index}"
        a, b = f"binding-{index:02d}-a", f"binding-{index:02d}-b"
        documents.extend(
            (
                Document(a, f"{project} permits red teams to export tables and blue teams to import images"),
                Document(b, f"{project} permits blue teams to export tables and red teams to import images"),
            )
        )
        for color, relevant in (("red", a), ("blue", b)):
            train.append(
                Query(
                    f"train-binding-{index}-{color}",
                    f"How {project} allows {color} teams exporting tables",
                    (relevant,),
                )
            )
            dev.append(
                Query(
                    f"dev-binding-{index}-{color}",
                    f"Explain tables export permission for {color} teams under {project}",
                    (relevant,),
                )
            )
            test.append(
                Query(
                    f"test-binding-{index}-{color}",
                    f"Within {project} find the rule allowing {color} teams to export tables",
                    (relevant,),
                )
            )
    return Dataset(
        tuple(documents),
        tuple(train),
        tuple(dev),
        tuple(test),
        source="synthetic-index-v1; alias tasks plus relation-swapped equal-token-multiset hard negatives",
    )


def _mean(values):
    return sum(values) / len(values) if values else None


def benchmark(*, epochs=8, seed=7, clusters=8, queries=128, artifact_directory: Path | None = None) -> dict:
    if type(queries) is not int or not 2 <= queries <= 128:
        raise ValueError("benchmark query count must be in [2,128]")
    data = challenging_dataset()
    config = TrainingConfig(seed=seed, epochs=epochs, dimension=16, negatives=5)
    started = time.perf_counter()
    model = DualEncoder(data.documents, config).fit(data.train)
    training_seconds = time.perf_counter() - started
    retriever = LateInteractionRetriever(model)
    index = IVFIndex.build(retriever, clusters=clusters, seed=seed)
    # Interleave ordinary alias questions and relation-binding failures so a
    # smaller query budget cannot silently remove the difficult slice.
    ordinary = [query for query in data.test if not query.id.startswith("test-binding")]
    binding = [query for query in data.test if query.id.startswith("test-binding")]
    selected = [query for pair in zip(ordinary, binding) for query in pair][:queries]
    k = 5
    oracles = {query.id: retriever.search(query.text, k) for query in selected}
    oracle_rows = [
        {
            "query_id": query.id,
            "slice": "relation_binding" if query.id.startswith("test-binding") else "alias",
            "relevant": list(query.relevant),
            "ranked_ids": [identifier for identifier, _ in oracles[query.id].rankings],
            "ground_truth_hit_at_1": bool(
                oracles[query.id].rankings and oracles[query.id].rankings[0][0] in query.relevant
            ),
            "work": oracles[query.id].work(),
        }
        for query in selected
    ]
    settings = []
    probes_list = sorted({1, min(2, clusters), min(4, clusters), clusters})
    budgets = sorted({min(16, len(data.documents)), len(data.documents)})
    for probes in probes_list:
        for budget in budgets:
            rows = []
            for query in selected:
                result = index.search(query.text, k, probes=probes, max_candidates=budget)
                expected = {identifier for identifier, _ in oracles[query.id].rankings}
                actual = {identifier for identifier, _ in result.rankings}
                rows.append(
                    {
                        "query_id": query.id,
                        "ranked_ids": [identifier for identifier, _ in result.rankings],
                        "oracle_recall_at_5": len(expected & actual) / len(expected) if expected else None,
                        "ground_truth_hit_at_1": bool(result.rankings and result.rankings[0][0] in query.relevant),
                        "work": result.work(),
                    }
                )
            settings.append(
                {
                    "probes": probes,
                    "max_candidates": budget,
                    "query_count": len(rows),
                    "mean_oracle_recall_at_5": _mean(
                        [row["oracle_recall_at_5"] for row in rows if row["oracle_recall_at_5"] is not None]
                    ),
                    "ground_truth_recall_at_1": _mean([row["ground_truth_hit_at_1"] for row in rows]),
                    "mean_candidates_reranked": _mean([row["work"]["candidates_reranked"] for row in rows]),
                    "mean_maxsim_token_dot_products": _mean([row["work"]["maxsim_token_dot_products"] for row in rows]),
                    "mean_coarse_dot_products": _mean([row["work"]["coarse_dot_products"] for row in rows]),
                    "median_query_seconds": statistics.median(row["work"]["query_seconds"] for row in rows),
                    "cases": rows,
                }
            )
    if artifact_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        (artifact_directory / "dataset.json").write_text(json.dumps(data.to_dict(), indent=2) + "\n")
        model.save(artifact_directory / "model.json")
        index.save(artifact_directory / "index.json")
    slices = {}
    for name in ("alias", "relation_binding"):
        rows = [row for row in oracle_rows if row["slice"] == name]
        slices[name] = {
            "query_count": len(rows),
            "ground_truth_recall_at_1": _mean([row["ground_truth_hit_at_1"] for row in rows]),
        }
    full_probe = next(
        setting
        for setting in settings
        if setting["probes"] == clusters and setting["max_candidates"] == len(data.documents)
    )
    return {
        "experiment": "learned_token_maxsim_and_ivf_exact_rerank",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset_sha256": data.fingerprint(),
        "full_probe_consistency_passed": full_probe["mean_oracle_recall_at_5"] == 1.0,
        "document_count": len(data.documents),
        "query_count": len(selected),
        "training_query_count": len(data.train),
        "encoder_training": model.trace,
        "encoder_training_seconds": training_seconds,
        "encoder_sha256": retriever.encoder_sha256,
        "corpus_sha256": retriever.corpus_sha256,
        "index_training": index.training,
        "oracle": {
            "system": "exhaustive_l2_normalized_token_maxsim",
            "ground_truth_recall_at_1": _mean([row["ground_truth_hit_at_1"] for row in oracle_rows]),
            "mean_token_dot_products": _mean([row["work"]["maxsim_token_dot_products"] for row in oracle_rows]),
            "median_query_seconds": statistics.median(row["work"]["query_seconds"] for row in oracle_rows),
            "slices": slices,
            "cases": oracle_rows,
        },
        "settings": settings,
        "limitations": [
            "The existing mean-pooling-trained token tables are reused; no contextual encoder or MaxSim-specific training is claimed.",
            "Relation-swapped documents have identical token multisets: this architecture cannot bind their roles, even under exhaustive search.",
            "Oracle recall measures agreement with exhaustive MaxSim, not factual relevance; ground-truth relevance is reported separately.",
            "Query timings are one local CPU run including Python validation and bookkeeping; work counters are exact token/centroid dot-product counts, not FLOPs or serving SLAs.",
        ],
    }
