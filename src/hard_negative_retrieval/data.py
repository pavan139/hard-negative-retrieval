"""Validated synthetic data and JSON import. The corpus is shared; query labels are split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

MAX_DOCUMENTS = 1000
MAX_QUERIES = 5000
MAX_TEXT = 5000
MAX_DATASET_BYTES = 5_000_000


def bounded_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise ValueError(f"invalid_{name}: expected 1..{MAX_TEXT} nonblank characters")
    return value


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    documents: tuple[Document, ...]
    train: tuple[Query, ...]
    dev: tuple[Query, ...]
    test: tuple[Query, ...]
    source: str = "user-supplied"

    def __post_init__(self) -> None:
        if not 1 <= len(self.documents) <= MAX_DOCUMENTS:
            raise ValueError("invalid_document_count")
        ids = [bounded_text(doc.id, "document_id") for doc in self.documents]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate_document_id")
        for doc in self.documents:
            bounded_text(doc.text, "document_text")
        queries = (*self.train, *self.dev, *self.test)
        if not all((self.train, self.dev, self.test)) or len(queries) > MAX_QUERIES:
            raise ValueError("invalid_split_sizes")
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        for query in queries:
            bounded_text(query.id, "query_id")
            normalized = " ".join(bounded_text(query.text, "query_text").casefold().split())
            if query.id in seen_ids or normalized in seen_text:
                raise ValueError("query_split_leakage: duplicate query ID or normalized text")
            seen_ids.add(query.id)
            seen_text.add(normalized)
            if (
                not isinstance(query.relevant, tuple)
                or not query.relevant
                or any(not isinstance(doc_id, str) for doc_id in query.relevant)
                or len(set(query.relevant)) != len(query.relevant)
            ):
                raise ValueError("invalid_relevance_labels")
            if not set(query.relevant).issubset(ids):
                raise ValueError("unknown_relevant_document")
        bounded_text(self.source, "source")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def dataset_from_dict(raw: dict) -> Dataset:
    if not isinstance(raw, dict):
        raise ValueError("invalid_dataset: expected object")
    try:
        for split in ("train", "dev", "test"):
            if not isinstance(raw[split], list) or any(
                not isinstance(q, dict) or not isinstance(q.get("relevant"), list) for q in raw[split]
            ):
                raise ValueError("invalid_dataset_schema: split queries and relevance labels must be arrays")
        documents = tuple(Document(**doc) for doc in raw["documents"])
        splits = {
            split: tuple(Query(q["id"], q["text"], tuple(q["relevant"])) for q in raw[split])
            for split in ("train", "dev", "test")
        }
        return Dataset(documents, **splits, source=raw.get("source", "user-supplied"))
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid_dataset_schema") from exc


def load_dataset(path: str | Path) -> Dataset:
    with Path(path).open("rb") as source:
        raw = source.read(MAX_DATASET_BYTES + 1)
    if len(raw) > MAX_DATASET_BYTES:
        raise ValueError("dataset_too_large")
    try:
        parsed = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid_dataset_json") from exc
    return dataset_from_dict(parsed)


def corpus_fingerprint(documents: tuple[Document, ...]) -> str:
    serialized = json.dumps([asdict(d) for d in documents], sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def synthetic_dataset() -> Dataset:
    """Closed-corpus benchmark with disjoint query templates, not a generalization claim."""
    topics = (
        (
            "password",
            "replace your password in the account security panel",
            "forgotten credential",
            "recover account access",
        ),
        ("billing", "download your invoice from the billing payment history", "purchase receipt", "proof of payment"),
        (
            "export",
            "export your records as a csv file from the data export menu",
            "spreadsheet copy",
            "save table locally",
        ),
        (
            "cancel",
            "cancel your subscription from the plan termination page",
            "stop membership",
            "end recurring service",
        ),
        ("invite", "invite a teammate using the workspace member invitation page", "add colleague", "onboard coworker"),
        (
            "webhook",
            "configure your webhook endpoint in the developer callback settings",
            "event notification url",
            "receive automatic callbacks",
        ),
        (
            "quota",
            "increase your storage quota with the capacity upgrade control",
            "disk space limit",
            "more room for files",
        ),
        (
            "region",
            "change your deployment region using the location migration menu",
            "move hosting geography",
            "switch server country",
        ),
    )
    documents: list[Document] = []
    splits: dict[str, list[Query]] = {"train": [], "dev": [], "test": []}
    templates = {
        "train": (
            "How can I handle {alias} for {product}?",
            "Help with {other} on {product}.",
            "For {product}, I need {alias} and {other}.",
        ),
        "dev": ("Explain {alias} within {product} please.",),
        "test": ("Where does {product} support {other}?", "Find instructions about {alias} in {product}."),
    }
    for product in ("atlas", "boreal", "cobalt", "delta"):
        for topic, instruction, alias, other in topics:
            doc_id = f"{product}-{topic}"
            documents.append(
                Document(doc_id, f"{product} guide: {instruction}. These instructions apply only to {product}.")
            )
            for split, patterns in templates.items():
                for i, pattern in enumerate(patterns):
                    splits[split].append(
                        Query(
                            f"{split}-{doc_id}-{i}",
                            pattern.format(product=product, alias=alias, other=other),
                            (doc_id,),
                        )
                    )
    return Dataset(
        tuple(documents),
        **{name: tuple(rows) for name, rows in splits.items()},
        source="synthetic-v1; product x intent; held-out query templates; shared corpus",
    )
