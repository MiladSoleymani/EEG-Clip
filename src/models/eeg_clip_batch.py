"""
EEG-CLIP model with batch-wise contrastive learning.
Compares each EEG against all text captions in the batch.
"""

import torch
import torch.nn as nn

from .atcnet import ATCNetDirectEmbedding
from .text_encoder import FrozenTextEncoder


class EEG_CLIP_Batch(nn.Module):
    """
    EEG-CLIP with batch-wise contrastive learning.

    Standard CLIP approach: compares each EEG against ALL text captions in batch.
    Uses symmetric contrastive loss (EEG→Text + Text→EEG).

    Args:
        num_electrodes: Number of EEG electrodes
        chunk_size: Time points per sample
        embedding_dim: Dimension of embeddings
        temperature: Temperature parameter for scaling logits
        text_model: Name of text encoder model
        atcnet_config: Configuration for ATCNet architecture
    """

    def __init__(
        self,
        num_electrodes: int = 50,
        chunk_size: int = 1001,
        embedding_dim: int = 384,
        temperature: float = 0.07,
        text_model: str = 'all-MiniLM-L6-v2',
        atcnet_config: dict = None
    ):
        super(EEG_CLIP_Batch, self).__init__()

        # ATCNet configuration
        if atcnet_config is None:
            atcnet_config = {
                'num_windows': 3,
                'conv_pool_size': 7,
                'F1': 16,
                'D': 2,
                'tcn_kernel_size': 4,
                'tcn_depth': 2
            }

        # EEG encoder - outputs embeddings directly
        self.eeg_encoder = ATCNetDirectEmbedding(
            in_channels=1,
            embedding_dim=embedding_dim,
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            **atcnet_config
        )

        # Frozen text encoder
        self.text_encoder = FrozenTextEncoder(model_name=text_model)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(temperature))

        print("\n" + "="*80)
        print("EEG-CLIP Batch-wise Contrastive Model")
        print("="*80)
        print(f"Architecture:")
        print(f"  EEG Encoder: ATCNet (Direct) → {embedding_dim} dim (TRAINABLE)")
        print(f"  Text Encoder: {text_model} → {embedding_dim} dim (FROZEN)")
        print(f"  Temperature: {temperature} (learnable)")
        print(f"\nTraining Strategy:")
        print(f"  Each EEG compared against ALL batch text captions")
        print(f"  Symmetric loss: EEG→Text + Text→EEG")
        print(f"  Random baseline: 1/batch_size (e.g., 3.125% for batch=32)")
        print("="*80)

    def forward(self, eeg_data, text_captions):
        """
        Forward pass.

        Args:
            eeg_data: (batch_size, 1, num_electrodes, time_points)
            text_captions: List of strings (batch_size captions)

        Returns:
            logits_eeg_to_text: (batch_size, batch_size) - EEG→Text similarities
            logits_text_to_eeg: (batch_size, batch_size) - Text→EEG similarities
        """
        # Encode EEG
        eeg_embeddings = self.eeg_encoder(eeg_data)  # (batch_size, 384)

        # Encode text
        text_embeddings = self.text_encoder(text_captions)  # (batch_size, 384)

        # Compute similarity matrix
        # (batch_size, 384) @ (384, batch_size) = (batch_size, batch_size)
        logits = torch.matmul(eeg_embeddings, text_embeddings.t()) / self.temperature

        # For symmetric loss, we need both directions
        logits_eeg_to_text = logits  # Each row: one EEG vs all texts
        logits_text_to_eeg = logits.t()  # Each row: one text vs all EEGs

        return logits_eeg_to_text, logits_text_to_eeg

    def get_eeg_embeddings(self, eeg_data):
        """
        Get EEG embeddings without computing logits.

        Args:
            eeg_data: (batch_size, 1, num_electrodes, time_points)

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.eeg_encoder(eeg_data)

    def get_text_embeddings(self, text_captions):
        """
        Get text embeddings.

        Args:
            text_captions: List of strings

        Returns:
            embeddings: (len(text_captions), embedding_dim)
        """
        return self.text_encoder(text_captions)
