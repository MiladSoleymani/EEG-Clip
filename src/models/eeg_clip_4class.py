"""
EEG-CLIP model with 4 fixed class templates.
Pre-computes text embeddings for 4 classes at initialization.
"""

import torch
import torch.nn as nn

from .atcnet import ATCNetDirectEmbedding
from .text_encoder import FrozenTextEncoder


class EEG_CLIP_4Class(nn.Module):
    """
    EEG-CLIP with 4 fixed class templates.

    Compares each EEG against 4 pre-computed text embeddings.
    More efficient than batch-wise approach as text embeddings are computed once.

    Args:
        num_classes: Number of classes (fixed at 4)
        num_electrodes: Number of EEG electrodes
        chunk_size: Time points per sample
        embedding_dim: Dimension of embeddings
        temperature: Temperature parameter for scaling logits
        text_model: Name of text encoder model
        class_templates: List of 4 class descriptions
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_electrodes: int = 50,
        chunk_size: int = 1001,
        embedding_dim: int = 384,
        temperature: float = 0.07,
        text_model: str = 'all-MiniLM-L6-v2',
        class_templates: list = None,
        atcnet_config: dict = None
    ):
        super(EEG_CLIP_4Class, self).__init__()

        assert num_classes == 4, "This model is designed for 4 classes"

        # Default class templates
        if class_templates is None:
            class_templates = [
                "imagining left hand movement",
                "imagining right hand movement",
                "imagining foot movement",
                "no motor imagery"
            ]

        # ATCNet configuration
        if atcnet_config is None:
            atcnet_config = {}

        # Set default ATCNet parameters
        atcnet_params = {
            'in_channels': 1,
            'embedding_dim': embedding_dim,
            'num_electrodes': num_electrodes,
            'chunk_size': chunk_size,
            'num_windows': 3,
            'conv_pool_size': 7,
            'F1': 16,
            'D': 2,
            'tcn_kernel_size': 4,
            'tcn_depth': 2
        }

        # Update with user-provided config
        atcnet_params.update(atcnet_config)

        # EEG encoder - outputs embeddings directly
        self.eeg_encoder = ATCNetDirectEmbedding(**atcnet_params)

        # Frozen text encoder
        self.text_encoder = FrozenTextEncoder(model_name=text_model)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Store class templates
        self.class_templates = class_templates

        # Pre-compute text embeddings for 4 classes (FROZEN)
        self._precompute_class_embeddings()

        print("\n" + "="*80)
        print("EEG-CLIP 4-Class Model - Direct Embedding")
        print("="*80)
        print(f"Architecture:")
        print(f"  EEG Encoder: ATCNet (Direct) → {embedding_dim} dim (TRAINABLE)")
        print(f"  Text Encoder: {text_model} → {embedding_dim} dim (FROZEN)")
        print(f"  Class embeddings: Pre-computed (4 fixed vectors)")
        print(f"  Temperature: {temperature} (learnable)")
        print(f"\nClass Templates:")
        for i, template in enumerate(class_templates):
            print(f"  Class {i}: '{template}'")
        print("="*80)

    def _precompute_class_embeddings(self):
        """
        Pre-compute text embeddings for all 4 classes.
        These are FIXED and never change during training.
        """
        print("\nPre-computing class embeddings...")

        with torch.no_grad():
            class_embeddings = self.text_encoder(self.class_templates)  # (4, 384)

        # Register as buffer (not a parameter, moves with model to GPU/CPU)
        self.register_buffer('class_embeddings', class_embeddings)

        print(f"✓ Class embeddings computed: {self.class_embeddings.shape}")
        print(f"  These embeddings are FIXED (never updated during training)")

    def forward(self, eeg_data, labels=None):
        """
        Forward pass.

        Args:
            eeg_data: (batch_size, 1, num_electrodes, time_points)
            labels: (batch_size,) - class labels (0-3), optional

        Returns:
            logits: (batch_size, 4) - similarity scores for each class
        """
        # Encode EEG - direct embedding output
        eeg_embeddings = self.eeg_encoder(eeg_data)  # (batch_size, 384)

        # Compare against 4 class embeddings
        # class_embeddings is (4, 384)
        # eeg_embeddings is (batch_size, 384)
        # Result: (batch_size, 4)
        logits = torch.matmul(eeg_embeddings, self.class_embeddings.t()) / self.temperature

        return logits

    def get_eeg_embeddings(self, eeg_data):
        """
        Get EEG embeddings without computing logits.

        Args:
            eeg_data: (batch_size, 1, num_electrodes, time_points)

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.eeg_encoder(eeg_data)

    def get_text_embeddings(self):
        """
        Get the pre-computed text embeddings.

        Returns:
            embeddings: (4, embedding_dim)
        """
        return self.class_embeddings
