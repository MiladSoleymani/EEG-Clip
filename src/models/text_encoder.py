"""
Frozen text encoder using pre-trained sentence transformers.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class FrozenTextEncoder(nn.Module):
    """
    Frozen pre-trained text encoder.

    Uses sentence-transformers models that are completely frozen during training.
    Only the EEG encoder is trained to align with the frozen text embeddings.

    Args:
        model_name: Name of the sentence-transformer model
            Options: "all-MiniLM-L6-v2" (384-dim), "all-mpnet-base-v2" (768-dim)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super(FrozenTextEncoder, self).__init__()

        print(f"\nLoading FROZEN text encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # FREEZE all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        # Determine output dimension based on model
        if 'mpnet' in model_name:
            self.output_dim = 768
        else:
            self.output_dim = 384

        print(f"✓ Text encoder FROZEN")
        print(f"✓ Output dimension: {self.output_dim}")

    def forward(self, texts):
        """
        Encode text to embeddings.

        Args:
            texts: List of strings or single string

        Returns:
            embeddings: Normalized text embeddings (batch_size, output_dim)
        """
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        return embeddings

    def train(self, mode=True):
        """Override train() to keep encoder in eval mode"""
        return self

    @property
    def device(self):
        """Get device of the model"""
        return next(self.encoder.parameters()).device
