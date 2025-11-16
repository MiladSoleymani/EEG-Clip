"""Data loading modules for EEG-CLIP"""

from .dataset import MultiSubjectECoGDataset
from .caption_generator import CaptionGenerator, ClassTemplates

__all__ = [
    "MultiSubjectECoGDataset",
    "CaptionGenerator",
    "ClassTemplates",
]
