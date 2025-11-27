"""Frozen learned token encoders, exact MaxSim and spherical k-means IVF search.

The existing dual encoder supplies real learned token tables. This module changes
the scoring/index algorithm; it does not claim that those tables were trained
with a late-interaction objective or that vectors are contextual transformer tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .data import corpus_fingerprint
from .model import DualEncoder, finite_number
from .retrieval import tokenize, validate_k


def _unit(vector):
    norm = math.sqrt(math.fsum(value * value for value in vector))
    return tuple(value / norm for value in vector) if norm else tuple(0.0 for _ in vector)


def _dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def maxsim(query_tokens, document_tokens) -> float:
    """Sum over query tokens of the maximum document-token dot product.

    Each query occurrence contributes once. Callers can pass normalized token
    vectors for cosine MaxSim; an empty side has no valid retrieval score.
    """
    if not query_tokens or not document_tokens:
        raise ValueError("maxsim_requires_nonempty_token_vectors")
    dimension = len(query_tokens[0])
    if not 1 <= dimension <= 128 or len(query_tokens) > 128 or len(document_tokens) > 128:
        raise ValueError("maxsim_shape_budget")
    if any(
        len(vector) != dimension or not all(finite_number(value) and abs(value) <= 1000 for value in vector)
        for vector in (*query_tokens, *document_tokens)
    ):
        raise ValueError("invalid_maxsim_vectors")
    return math.fsum(max(_dot(query, document) for document in document_tokens) for query in query_tokens)


@dataclass(frozen=True)
class SearchResult:
    rankings: tuple[tuple[str, float], ...]
    candidates_reranked: int
    maxsim_token_dot_products: int
    coarse_dot_products: int
    query_seconds: float
    probed_clusters: tuple[int, ...] = ()
    candidates_before_budget: int = 0

    def work(self):
        return {
            "candidates_reranked": self.candidates_reranked,
            "maxsim_token_dot_products": self.maxsim_token_dot_products,
            "coarse_dot_products": self.coarse_dot_products,
            "query_seconds": self.query_seconds,
            "probed_clusters": list(self.probed_clusters),
            "candidates_before_budget": self.candidates_before_budget,
        }


class LateInteractionRetriever:
    """Immutable copied encoder state, exact token interaction and deterministic ties."""

    def __init__(self, model: DualEncoder):
        if not model.vocabulary or not model.query_embeddings or not model.document_embeddings:
            raise ValueError("model_not_fitted")
        self.documents = tuple(model.documents)
        self.dimension = model.config.dimension
        self.vocabulary = dict(model.vocabulary)
        self.query_table = tuple(_unit(row) for row in model.query_embeddings)
        self.document_table = tuple(_unit(row) for row in model.document_embeddings)
        self.corpus_sha256 = corpus_fingerprint(self.documents)
        source_state = {
            "vocabulary": model.vocabulary,
            "query_embeddings": model.query_embeddings,
            "document_embeddings": model.document_embeddings,
            "dimension": model.config.dimension,
        }
        self.source_model_sha256 = hashlib.sha256(json.dumps(source_state, sort_keys=True).encode()).hexdigest()
        state = {
            "corpus_sha256": self.corpus_sha256,
            "source_model_sha256": self.source_model_sha256,
            "vocabulary": self.vocabulary,
            "query_table": self.query_table,
            "document_table": self.document_table,
            "scoring": "l2_normalized_token_maxsim_v1",
        }
        self.encoder_sha256 = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        self.token_vectors = {doc.id: self.encode(doc.text, query=False) for doc in self.documents}
        if any(not vectors for vectors in self.token_vectors.values()):
            raise ValueError("indexed_document_has_no_known_tokens")
        self.coarse_vectors = {identifier: self.coarse(vectors) for identifier, vectors in self.token_vectors.items()}

    def encode(self, text, *, query=True):
        table = self.query_table if query else self.document_table
        return tuple(
            table[self.vocabulary[token]]
            for token in tokenize(text)
            if token in self.vocabulary and any(table[self.vocabulary[token]])
        )

    def coarse(self, vectors):
        return (
            _unit(
                [
                    math.fsum(vector[coordinate] for vector in vectors) / len(vectors)
                    for coordinate in range(self.dimension)
                ]
            )
            if vectors
            else tuple(0.0 for _ in range(self.dimension))
        )

    def search(self, text, k=10, *, candidates=None) -> SearchResult:
        validate_k(k)
        started = time.perf_counter()
        query = self.encode(text)
        if not query:
            return SearchResult((), 0, 0, 0, time.perf_counter() - started)
        selected = sorted(self.token_vectors) if candidates is None else list(candidates)
        if len(selected) != len(set(selected)) or any(identifier not in self.token_vectors for identifier in selected):
            raise ValueError("invalid_candidate_ids")
        work = sum(len(query) * len(self.token_vectors[identifier]) for identifier in selected)
        if work * self.dimension > 20_000_000:
            raise ValueError("maxsim_query_compute_budget_exceeded")
        scored = [(identifier, maxsim(query, self.token_vectors[identifier])) for identifier in selected]
        scored.sort(key=lambda row: (-row[1], row[0]))
        return SearchResult(
            tuple(scored[:k]),
            len(selected),
            work,
            0,
            time.perf_counter() - started,
            candidates_before_budget=len(selected),
        )

    def rank(self, text, k=10):
        return list(self.search(text, k).rankings)


class IVFIndex:
    """Spherical k-means document partition; exact MaxSim within probed candidates."""

    def __init__(self, retriever, centroids, postings, *, training):
        self.retriever = retriever
        self.centroids = tuple(tuple(vector) for vector in centroids)
        self.postings = tuple(tuple(bucket) for bucket in postings)
        self.training = training

    @classmethod
    def build(cls, retriever: LateInteractionRetriever, *, clusters=8, iterations=20, seed=7):
        count = len(retriever.documents)
        if type(clusters) is not int or not 1 <= clusters <= min(64, count):
            raise ValueError("cluster_count_out_of_range")
        if type(iterations) is not int or not 1 <= iterations <= 100 or type(seed) is not int:
            raise ValueError("invalid_index_training_budget")
        estimated_pairs = count * (clusters * (clusters - 1) // 2 + clusters * (iterations + 1))
        if estimated_pairs * retriever.dimension > 100_000_000:
            raise ValueError("index_training_compute_budget_exceeded")
        started = time.perf_counter()
        identifiers = sorted(retriever.coarse_vectors)
        vectors = [retriever.coarse_vectors[identifier] for identifier in identifiers]
        rng = random.Random(seed)
        chosen = [rng.randrange(count)]
        # k-means++ over unit vectors. Degenerate duplicate vectors are allowed;
        # empty buckets are explicit and do not lose any documents.
        while len(chosen) < clusters:
            weights = [
                max(0.0, min(2 - 2 * _dot(vector, vectors[index]) for index in chosen)) if i not in chosen else 0.0
                for i, vector in enumerate(vectors)
            ]
            total = math.fsum(weights)
            if not total:
                selected = next(index for index in range(count) if index not in chosen)
            else:
                draw = rng.random() * total
                selected = next(index for index in range(count) if index not in chosen)
                cumulative = 0.0
                for index, weight in enumerate(weights):
                    cumulative += weight
                    if weight > 0 and draw < cumulative:
                        selected = index
                        break
            chosen.append(selected)
        centroids = [vectors[index] for index in chosen]
        previous = None
        assignment_work = 0
        for iteration in range(iterations):
            assignments = [
                max(range(clusters), key=lambda index: (_dot(vector, centroids[index]), -index)) for vector in vectors
            ]
            assignment_work += count * clusters
            if assignments == previous:
                break
            previous = assignments
            updated = []
            for cluster in range(clusters):
                members = [vectors[index] for index, label in enumerate(assignments) if label == cluster]
                updated.append(retriever.coarse(members) if members else centroids[cluster])
            centroids = updated
        # Final assignments use the exact centroids persisted by the index.
        assignments = [
            max(range(clusters), key=lambda index: (_dot(vector, centroids[index]), -index)) for vector in vectors
        ]
        assignment_work += count * clusters
        postings = [
            [identifier for identifier, label in zip(identifiers, assignments) if label == cluster]
            for cluster in range(clusters)
        ]
        return cls(
            retriever,
            centroids,
            postings,
            training={
                "algorithm": "spherical_kmeans++",
                "seed": seed,
                "iterations_executed": iteration + 1,
                "assignment_dot_products": assignment_work,
                "seeding_dot_products": sum(index * (count - index) for index in range(1, clusters)),
                "build_seconds": time.perf_counter() - started,
                "cluster_sizes": list(map(len, postings)),
            },
        )

    def search(self, text, k=10, *, probes=1, max_candidates=None) -> SearchResult:
        validate_k(k)
        count = len(self.retriever.documents)
        max_candidates = count if max_candidates is None else max_candidates
        if type(probes) is not int or not 1 <= probes <= len(self.centroids):
            raise ValueError("probe_budget_out_of_range")
        if type(max_candidates) is not int or not 1 <= max_candidates <= count:
            raise ValueError("candidate_budget_out_of_range")
        started = time.perf_counter()
        tokens = self.retriever.encode(text)
        if not tokens:
            return SearchResult((), 0, 0, 0, time.perf_counter() - started)
        vector = self.retriever.coarse(tokens)
        clusters = sorted(range(len(self.centroids)), key=lambda index: (-_dot(vector, self.centroids[index]), index))[
            :probes
        ]
        pool = [identifier for cluster in clusters for identifier in self.postings[cluster]]
        # Budget truncation is based on coarse similarity rather than ID order.
        pool.sort(key=lambda identifier: (-_dot(vector, self.retriever.coarse_vectors[identifier]), identifier))
        result = self.retriever.search(text, k, candidates=pool[:max_candidates])
        return SearchResult(
            result.rankings,
            result.candidates_reranked,
            result.maxsim_token_dot_products,
            len(self.centroids) + len(pool),
            time.perf_counter() - started,
            tuple(clusters),
            len(pool),
        )

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "encoder_sha256": self.retriever.encoder_sha256,
                    "corpus_sha256": self.retriever.corpus_sha256,
                    "centroids": self.centroids,
                    "postings": self.postings,
                    "training": self.training,
                },
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path, retriever):
        with Path(path).open("rb") as stream:
            encoded = stream.read(10_000_001)
        if len(encoded) > 10_000_000:
            raise ValueError("index_artifact_too_large")
        try:
            data = json.loads(encoded)
            if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
                raise ValueError("invalid_index_schema")
            if (
                data.get("encoder_sha256") != retriever.encoder_sha256
                or data.get("corpus_sha256") != retriever.corpus_sha256
            ):
                raise ValueError("index_encoder_or_corpus_mismatch")
            centroids, postings = data["centroids"], data["postings"]
            if (
                not isinstance(centroids, list)
                or not 1 <= len(centroids) <= min(64, len(retriever.documents))
                or len(centroids) != len(postings)
            ):
                raise ValueError("invalid_index_cluster_shape")
            for vector in centroids:
                if (
                    not isinstance(vector, list)
                    or len(vector) != retriever.dimension
                    or not all(finite_number(value) and abs(value) <= 1.000001 for value in vector)
                ):
                    raise ValueError("invalid_index_centroid")
                norm = math.fsum(value * value for value in vector)
                if not (abs(norm) <= 1e-12 or abs(norm - 1.0) <= 1e-6):
                    raise ValueError("index_centroid_must_be_unit_or_zero")
            if not isinstance(postings, list) or any(
                not isinstance(bucket, list) or any(not isinstance(identifier, str) for identifier in bucket)
                for bucket in postings
            ):
                raise ValueError("invalid_index_postings")
            identifiers = [identifier for bucket in postings for identifier in bucket]
            if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(retriever.token_vectors):
                raise ValueError("index_postings_must_partition_corpus")
            for cluster, bucket in enumerate(postings):
                for identifier in bucket:
                    nearest = max(
                        range(len(centroids)),
                        key=lambda index: (_dot(retriever.coarse_vectors[identifier], centroids[index]), -index),
                    )
                    if nearest != cluster:
                        raise ValueError("index_posting_assignment_disagrees_with_centroids")
            if not isinstance(data.get("training"), dict):
                raise ValueError("invalid_index_training_metadata")
            return cls(retriever, centroids, postings, training=data["training"])
        except (AttributeError, KeyError, TypeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("invalid_index_schema") from error
