"""A trainable mean-pooling dual encoder with exact InfoNCE gradients.

Both token-embedding tables are optimized. This is real gradient training, not
random embeddings or a call to a hidden hosted model. It intentionally uses only
the Python standard library to keep the CPU experiment transparent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random

from .data import Document, Query, bounded_text, corpus_fingerprint
from .retrieval import LexicalRetriever, mine_hard_negatives, tokenize, validate_k


def finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def contrastive_loss_gradients(
    query: list[float], candidates: list[list[float]], temperature: float
) -> tuple[float, list[float], list[list[float]]]:
    """Positive is candidate zero; return exact derivatives of -log softmax(s)[0]."""
    if not candidates or not finite_number(temperature) or temperature <= 0:
        raise ValueError("invalid_contrastive_inputs")
    if not query or any(len(doc) != len(query) for doc in candidates):
        raise ValueError("inconsistent_embedding_dimensions")
    if not all(finite_number(value) for vector in [query, *candidates] for value in vector):
        raise ValueError("nonfinite_embedding")
    scores = [dot(query, doc) / temperature for doc in candidates]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("nonfinite_contrastive_score")
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    normalizer = sum(exponentials)
    loss = maximum + math.log(normalizer) - scores[0]
    coefficients = [(value / normalizer - (1 if i == 0 else 0)) / temperature for i, value in enumerate(exponentials)]
    query_gradient = [
        sum(coefficient * doc[j] for coefficient, doc in zip(coefficients, candidates)) for j in range(len(query))
    ]
    doc_gradients = [[coefficient * value for value in query] for coefficient in coefficients]
    if not math.isfinite(loss) or not all(
        math.isfinite(value) for vector in [query_gradient, *doc_gradients] for value in vector
    ):
        raise ValueError("nonfinite_contrastive_result")
    return loss, query_gradient, doc_gradients


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    dimension: int = 24
    epochs: int = 35
    learning_rate: float = 0.7
    negatives: int = 5
    temperature: float = 0.2
    negative_strategy: str = "hard"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("invalid_seed")
        for key, minimum, maximum in (("dimension", 2, 128), ("epochs", 1, 200), ("negatives", 1, 50)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"invalid_{key}")
        if not finite_number(self.learning_rate) or not 0 < self.learning_rate <= 2:
            raise ValueError("invalid_learning_rate")
        if not finite_number(self.temperature) or not 0.01 <= self.temperature <= 10:
            raise ValueError("invalid_temperature")
        if self.negative_strategy not in ("hard", "random"):
            raise ValueError("invalid_negative_strategy")


class DualEncoder:
    def __init__(self, documents: tuple[Document, ...], config: TrainingConfig | None = None):
        if not documents or len(documents) > 1000 or len({doc.id for doc in documents}) != len(documents):
            raise ValueError("invalid_corpus")
        self.documents = documents
        self.config = config or TrainingConfig()
        self.vocabulary: dict[str, int] = {}
        self.query_embeddings: list[list[float]] = []
        self.document_embeddings: list[list[float]] = []
        self.trace: dict = {}
        self._doc_vectors: dict[str, list[float]] = {}
        self._doc_features: dict[str, list[tuple[int, float]]] = {}

    def _features(self, text: str) -> list[tuple[int, float]]:
        counts = Counter(token for token in tokenize(text) if token in self.vocabulary)
        total = sum(counts.values()) or 1
        return [(self.vocabulary[token], count / total) for token, count in sorted(counts.items())]

    def _encode(self, features: list[tuple[int, float]], table: list[list[float]]) -> list[float]:
        vector = [0.0] * self.config.dimension
        for index, weight in features:
            for j, value in enumerate(table[index]):
                vector[j] += weight * value
        return vector

    def _refresh_documents(self) -> None:
        self._doc_vectors = {
            doc_id: self._encode(features, self.document_embeddings) for doc_id, features in self._doc_features.items()
        }

    def rank(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        validate_k(k)
        if not self.vocabulary:
            raise ValueError("model_not_fitted")
        features = self._features(text)
        if not features:
            # Unknown tokens produce no invented evidence and no arbitrary top hit.
            return []
        query = self._encode(features, self.query_embeddings)
        return sorted(
            ((doc_id, dot(query, vector)) for doc_id, vector in self._doc_vectors.items()),
            key=lambda row: (-row[1], row[0]),
        )[:k]

    def _update(self, table: list[list[float]], features: list[tuple[int, float]], gradient: list[float]) -> None:
        norm = math.sqrt(dot(gradient, gradient))
        factor = min(1.0, 5.0 / max(norm, 1e-12))
        for index, weight in features:
            for j in range(self.config.dimension):
                table[index][j] -= self.config.learning_rate * (
                    weight * gradient[j] * factor + 0.00001 * table[index][j]
                )

    def fit(self, train: tuple[Query, ...]) -> "DualEncoder":
        """Only train queries enter vocabulary, hard mining, gradients, or fingerprints."""
        if not train or len(train) > 5000:
            raise ValueError("invalid_training_size")
        for query in train:
            bounded_text(query.id, "query_id")
            bounded_text(query.text, "query_text")
            if not tokenize(query.text):
                raise ValueError("empty_training_query_tokens")
        if len({query.id for query in train}) != len(train):
            raise ValueError("duplicate_training_query_id")
        doc_ids = {doc.id for doc in self.documents}
        if any(
            not isinstance(query.relevant, tuple)
            or not query.relevant
            or len(set(query.relevant)) != len(query.relevant)
            or not set(query.relevant).issubset(doc_ids)
            for query in train
        ):
            raise ValueError("invalid_training_labels")
        if len(doc_ids) < 2 or any(set(q.relevant) == doc_ids for q in train):
            raise ValueError("no_negative_documents")
        if self.config.epochs * len(train) * self.config.dimension > 20_000_000:
            raise ValueError("training_budget_exceeded")
        work_estimate = (
            self.config.epochs
            * len(train)
            * self.config.dimension
            * (len(self.documents) + (self.config.negatives + 1) * 128)
        )
        if work_estimate > 100_000_000:
            raise ValueError("training_compute_budget_exceeded")
        vocabulary = sorted(
            {
                token
                for text in [*(d.text for d in self.documents), *(q.text for q in train)]
                for token in tokenize(text)
            }
        )
        if not vocabulary or len(vocabulary) > 10000:
            raise ValueError("invalid_vocabulary_size")
        self.vocabulary = {token: i for i, token in enumerate(vocabulary)}
        rng = random.Random(self.config.seed)
        negative_rng = random.Random(self.config.seed ^ 0x5F3759DF)
        initial = [[rng.gauss(0, 0.35) for _ in range(self.config.dimension)] for _ in vocabulary]
        # Same initialization gives exact-token alignment; tables train independently.
        self.query_embeddings = [row[:] for row in initial]
        self.document_embeddings = [row[:] for row in initial]
        self._doc_features = {doc.id: self._features(doc.text) for doc in self.documents}
        features = {query.id: self._features(query.text) for query in train}
        lexical = LexicalRetriever(self.documents)
        self._refresh_documents()
        losses = []
        epoch_order_hashes = []
        mined_count = 0
        for epoch in range(self.config.epochs):
            miner = lexical if epoch == 0 else self
            # Freeze rankings once per epoch; all embedding updates still use current weights.
            if self.config.negative_strategy == "random":
                negatives = {}
                for query in train:
                    eligible = sorted(doc_ids - set(query.relevant))
                    negatives[query.id] = negative_rng.sample(eligible, min(self.config.negatives, len(eligible)))
            else:
                negatives = {
                    query.id: mine_hard_negatives(
                        miner, query.text, query.relevant, self.config.negatives, len(self.documents)
                    )
                    for query in train
                }
            order = list(train)
            rng.shuffle(order)
            epoch_order_hashes.append(hashlib.sha256(json.dumps([query.id for query in order]).encode()).hexdigest())
            loss_sum = 0.0
            for query in order:
                positive = query.relevant[epoch % len(query.relevant)]
                candidates = [positive, *negatives[query.id]]
                qfeatures = features[query.id]
                qvector = self._encode(qfeatures, self.query_embeddings)
                dvectors = [self._encode(self._doc_features[doc_id], self.document_embeddings) for doc_id in candidates]
                loss, qgrad, dgrads = contrastive_loss_gradients(qvector, dvectors, self.config.temperature)
                if not math.isfinite(loss):
                    raise ValueError("training_nonfinite_loss")
                self._update(self.query_embeddings, qfeatures, qgrad)
                for doc_id, gradient in zip(candidates, dgrads):
                    self._update(self.document_embeddings, self._doc_features[doc_id], gradient)
                loss_sum += loss
                mined_count += len(candidates) - 1
            self._refresh_documents()
            losses.append(loss_sum / len(train))
        serialized_train = json.dumps(
            [{"id": q.id, "text": q.text, "relevant": q.relevant} for q in train], sort_keys=True
        )
        self.trace = {
            "algorithm": "mean-pooling dual encoder; InfoNCE SGD",
            "negative_strategy": self.config.negative_strategy,
            "mining": "lexical then model-mined"
            if self.config.negative_strategy == "hard"
            else "uniform without replacement, excluding all positives",
            "train_query_count": len(train),
            "training_query_sha256": hashlib.sha256(serialized_train.encode()).hexdigest(),
            "vocabulary_size": len(vocabulary),
            "negative_pairs_used": mined_count,
            "epoch_mean_training_loss": losses,
            "epoch_order_sha256": epoch_order_hashes,
            "config": asdict(self.config),
        }
        return self

    def save(self, path: str | Path) -> None:
        if not self.vocabulary:
            raise ValueError("model_not_fitted")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "version": 1,
                    "corpus_sha256": corpus_fingerprint(self.documents),
                    "config": asdict(self.config),
                    "vocabulary": self.vocabulary,
                    "query_embeddings": self.query_embeddings,
                    "document_embeddings": self.document_embeddings,
                    "trace": self.trace,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path, documents: tuple[Document, ...]) -> "DualEncoder":
        path = Path(path)
        with path.open("rb") as source:
            encoded = source.read(50_000_001)
        if len(encoded) > 50_000_000:
            raise ValueError("checkpoint_too_large")
        try:
            raw = json.loads(encoded)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid_checkpoint_json") from exc
        if not isinstance(raw, dict):
            raise ValueError("invalid_checkpoint_schema")
        if (
            type(raw.get("version")) is not int
            or raw.get("version") != 1
            or raw.get("corpus_sha256") != corpus_fingerprint(documents)
        ):
            raise ValueError("checkpoint_corpus_mismatch")
        try:
            model = cls(documents, TrainingConfig(**raw["config"]))
            vocabulary = raw["vocabulary"]
            if (
                not isinstance(vocabulary, dict)
                or not 1 <= len(vocabulary) <= 10000
                or any(
                    not isinstance(token, str) or not isinstance(index, int) or isinstance(index, bool)
                    for token, index in vocabulary.items()
                )
                or set(vocabulary.values()) != set(range(len(vocabulary)))
            ):
                raise ValueError("invalid_checkpoint_vocabulary")
            for table in (raw["query_embeddings"], raw["document_embeddings"]):
                if (
                    not isinstance(table, list)
                    or len(table) != len(vocabulary)
                    or any(
                        not isinstance(row, list)
                        or len(row) != model.config.dimension
                        or not all(finite_number(value) and abs(value) <= 1000 for value in row)
                        for row in table
                    )
                ):
                    raise ValueError("invalid_checkpoint_weights")
            trace = raw["trace"]
            if (
                not isinstance(trace, dict)
                or not isinstance(trace.get("training_query_sha256"), str)
                or len(trace["training_query_sha256"]) != 64
            ):
                raise ValueError("invalid_checkpoint_trace")
            model.vocabulary = vocabulary
            model.query_embeddings = raw["query_embeddings"]
            model.document_embeddings = raw["document_embeddings"]
            model.trace = trace
            model._doc_features = {doc.id: model._features(doc.text) for doc in documents}
            model._refresh_documents()
            return model
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid_checkpoint_schema") from exc


class SentenceTransformerRetriever:
    """Optional real pretrained adapter. Installing/using it may download model weights."""

    def __init__(self, documents: tuple[Document, ...], model_name: str):
        from sentence_transformers import SentenceTransformer  # Lazy optional dependency.

        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([doc.text for doc in documents], normalize_embeddings=True)

    def rank(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        validate_k(k)
        tokenize(text)  # Apply the same input length contract.
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return sorted(
            ((doc.id, float(embedding @ vector)) for doc, embedding in zip(self.documents, self.embeddings)),
            key=lambda row: (-row[1], row[0]),
        )[:k]
