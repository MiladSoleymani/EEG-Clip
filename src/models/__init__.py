"""Model modules for EEG-CLIP"""

from .atcnet import ATCNetDirectEmbedding
from .text_encoder import FrozenTextEncoder
from .eeg_clip_4class import EEG_CLIP_4Class
from .eeg_clip_batch import EEG_CLIP_Batch

__all__ = [
    "ATCNetDirectEmbedding",
    "FrozenTextEncoder",
    "EEG_CLIP_4Class",
    "EEG_CLIP_Batch",
]
