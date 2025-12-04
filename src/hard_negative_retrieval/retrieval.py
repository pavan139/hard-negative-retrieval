"""Inspectable lexical and set-similarity baselines plus positive-safe negative mining."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Protocol

from .data import Document, bounded_text

STOPWORDS = frozenset(
    "a an the is to of from in as your you can how i my for and on with these only apply instructions guide please does where find about within help need handle support explain".split()
)


def tokenize(text: str) -> list[str]:
    bounded_text(text, "text")
    return [token for token in re.findall(r"[a-z0-9]+", text.casefold()) if token not in STOPWORDS][:128]


class Retriever(Protocol):
    def rank(self, text: str, k: int = 10) -> list[tuple[str, float]]: ...


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 1000:
        raise ValueError("invalid_k: expected 1..1000")


class LexicalRetriever:
    """TF-IDF cosine similarity. Corpus statistics never read query relevance labels."""

    def __init__(self, documents: tuple[Document, ...]):
        self.documents = documents
        df = Counter(token for doc in documents for token in set(tokenize(doc.text)))
        self.idf = {token: math.log((1 + len(documents)) / (1 + count)) + 1 for token, count in df.items()}
        self.vectors = {doc.id: self.vector(doc.text) for doc in documents}

    def vector(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        vector = {token: count * self.idf[token] for token, count in counts.items() if token in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    def rank(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        validate_k(k)
        query = self.vector(text)
        scored = [
            (doc.id, sum(weight * self.vectors[doc.id].get(token, 0) for token, weight in query.items()))
            for doc in self.documents
        ]
        return sorted(scored, key=lambda row: (-row[1], row[0]))[:k]


class TokenOverlapRetriever:
    """Jaccard token similarity, a second untrained baseline."""

    def __init__(self, documents: tuple[Document, ...]):
        self.tokens = {doc.id: set(tokenize(doc.text)) for doc in documents}

    def rank(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        validate_k(k)
        query = set(tokenize(text))
        scored = [(doc_id, len(query & tokens) / max(1, len(query | tokens))) for doc_id, tokens in self.tokens.items()]
        return sorted(scored, key=lambda row: (-row[1], row[0]))[:k]


def mine_hard_negatives(
    retriever: Retriever, text: str, relevant: tuple[str, ...], count: int, corpus_size: int
) -> list[str]:
    if not 1 <= count <= 100 or not 1 <= corpus_size <= 1000:
        raise ValueError("invalid_negative_budget")
    relevant_ids = set(relevant)
    # Rank the whole bounded corpus so positives can never silently consume the budget.
    return [doc_id for doc_id, _ in retriever.rank(text, corpus_size) if doc_id not in relevant_ids][:count]
