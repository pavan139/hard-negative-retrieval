"""Small, real contrastive retrieval experiments; no hosted model required."""

from .data import Dataset, Document, Query, synthetic_dataset
from .model import DualEncoder, TrainingConfig
from .late_interaction import IVFIndex, LateInteractionRetriever

__all__ = [
    "Dataset",
    "Document",
    "Query",
    "synthetic_dataset",
    "DualEncoder",
    "TrainingConfig",
    "LateInteractionRetriever",
    "IVFIndex",
]
