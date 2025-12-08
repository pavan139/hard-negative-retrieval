"""Binary-relevance metrics with denominators and explicit cutoffs."""

from __future__ import annotations

import math
from .data import Query
from .retrieval import Retriever, validate_k


def ranking_metrics(ranking: list[str], relevant: tuple[str, ...], k: int) -> dict[str, float]:
    validate_k(k)
    if (
        not isinstance(relevant, tuple)
        or not relevant
        or any(not isinstance(value, str) for value in relevant)
        or len(set(relevant)) != len(relevant)
    ):
        raise ValueError("invalid_relevance_labels")
    if len(set(ranking)) != len(ranking):
        raise ValueError("duplicate_ranked_document")
    selected = ranking[:k]
    relevant_set = set(relevant)
    hits = [i + 1 for i, doc_id in enumerate(selected) if doc_id in relevant_set]
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return {"recall": len(hits) / len(relevant), "mrr": 1 / hits[0] if hits else 0.0, "ndcg": dcg / ideal}


def evaluate_retriever(
    retriever: Retriever, queries: tuple[Query, ...], cutoffs: tuple[int, ...] = (1, 3, 5, 10)
) -> dict:
    if not queries:
        raise ValueError("empty_evaluation")
    if not isinstance(cutoffs, (tuple, list)) or not cutoffs:
        raise ValueError("invalid_cutoffs: require a nonempty sequence of unique cutoffs")
    for cutoff in cutoffs:
        validate_k(cutoff)
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError("invalid_cutoffs: require a nonempty sequence of unique cutoffs")
    totals = {f"{metric}@{k}": 0.0 for k in cutoffs for metric in ("recall", "mrr", "ndcg")}
    failures = []
    cases = []
    for query in queries:
        retrieved = retriever.rank(query.text, max(cutoffs))
        try:
            valid_results = isinstance(retrieved, (list, tuple)) and all(
                isinstance(doc_id, str) and type(score) in (int, float) and math.isfinite(score)
                for doc_id, score in retrieved
            )
        except (OverflowError, TypeError, ValueError):
            valid_results = False
        if not valid_results:
            raise ValueError("invalid_ranked_result")
        ranked = [doc_id for doc_id, _ in retrieved]
        measured = {}
        for k in cutoffs:
            for metric, value in ranking_metrics(ranked, query.relevant, k).items():
                totals[f"{metric}@{k}"] += value
                measured[f"{metric}@{k}"] = value
        cases.append(
            {"query_id": query.id, "relevant": list(query.relevant), "ranked_ids": ranked, "metrics": measured}
        )
        if not ranked or ranked[0] not in query.relevant:
            failures.append(
                {"query_id": query.id, "relevant": query.relevant, "top_prediction": ranked[0] if ranked else None}
            )
    return {
        "query_count": len(queries),
        "relevance_label_count": sum(len(q.relevant) for q in queries),
        "metrics": {key: value / len(queries) for key, value in totals.items()},
        "top1_failures": failures,
        "cases": cases,
    }
