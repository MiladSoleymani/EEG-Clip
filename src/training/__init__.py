"""Training modules for EEG-CLIP"""

from .trainer_4class import EEGCLIPTrainer4Class
from .trainer_batch import EEGCLIPTrainerBatch

__all__ = [
    "EEGCLIPTrainer4Class",
    "EEGCLIPTrainerBatch",
]
