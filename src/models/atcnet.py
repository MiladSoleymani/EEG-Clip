"""
ATCNet (Attention-based Temporal Convolutional Network) for EEG encoding.
Outputs direct embeddings without classification bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATCNetDirectEmbedding(nn.Module):
    """
    ATCNet model that outputs embeddings directly.

    Architecture:
        Conv Block → Multi-head Self-Attention → TCN → Feature Projection → Embedding

    Args:
        in_channels: Number of input channels (default: 1)
        embedding_dim: Dimension of output embeddings (default: 384)
        num_windows: Number of attention windows (default: 3)
        num_electrodes: Number of EEG electrodes (default: 50)
        conv_pool_size: Pooling size in conv block (default: 7)
        F1: Number of temporal filters (default: 16)
        D: Depth multiplier for spatial filters (default: 2)
        tcn_kernel_size: Kernel size for TCN (default: 4)
        tcn_depth: Number of TCN blocks (default: 2)
        chunk_size: Time points per sample (default: 1001)
    """

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 384,
        num_windows: int = 3,
        num_electrodes: int = 50,
        conv_pool_size: int = 7,
        F1: int = 16,
        D: int = 2,
        tcn_kernel_size: int = 4,
        tcn_depth: int = 2,
        chunk_size: int = 1001
    ):
        super(ATCNetDirectEmbedding, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_windows = num_windows
        self.num_electrodes = num_electrodes
        self.pool_size = conv_pool_size
        self.F1 = F1
        self.D = D
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_depth = tcn_depth
        self.chunk_size = chunk_size
        F2 = F1 * D

        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, F1, (1, int(chunk_size/2+1)),
                      stride=1, padding='same', bias=False),
            nn.BatchNorm2d(F1, False),
            nn.Conv2d(F1, F2, (num_electrodes, 1), padding=0, groups=F1),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(0.1),
            nn.Conv2d(F2, F2, (1, 16), bias=False, padding='same'),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_size)),
            nn.Dropout2d(0.1)
        )

        # Build attention and TCN blocks dynamically
        self.__build_model()

    def __build_model(self):
        """Dynamically build attention and TCN layers"""
        with torch.no_grad():
            # Dummy forward to calculate dimensions
            x = torch.zeros(2, self.in_channels, self.num_electrodes, self.chunk_size)
            x = self.conv_block(x)
            x = x[:, :, -1, :]
            x = x.permute(0, 2, 1)
            self.__chan_dim, self.__embed_dim = x.shape[1:]
            self.win_len = self.__chan_dim - self.num_windows + 1

            # Build attention and TCN for each window
            for i in range(self.num_windows):
                st = i
                end = x.shape[1] - self.num_windows + i + 1
                x2 = x[:, st:end, :]

                # Multi-head self-attention
                self.__add_msa(i)
                x2_ = self.get_submodule("msa"+str(i))(x2, x2, x2)[0]
                self.__add_msa_drop(i)
                x2_ = self.get_submodule("msa_drop"+str(i))(x2)
                x2 = torch.add(x2, x2_)

                # TCN blocks
                for j in range(self.tcn_depth):
                    self.__add_tcn((i+1)*j, x2.shape[1])
                    out = self.get_submodule("tcn"+str((i+1)*j))(x2)
                    if x2.shape[1] != out.shape[1]:
                        self.__add_recov(i)
                        x2 = self.get_submodule("re"+str(i))(x2)
                    x2 = torch.add(x2, out)
                    x2 = nn.ELU()(x2)

                x2 = x2[:, -1, :]
                self.__dense_dim = x2.shape[-1]

                # Feature projection for this window
                self.__add_feature_projection(i)
                x2 = self.get_submodule("feature_proj"+str(i))(x2)

            # Calculate concatenated feature dimension
            self.__concat_feature_dim = x2.shape[-1] * self.num_windows

            # Add final projection to embedding dimension
            self.__add_final_projection()

    def __add_msa(self, index: int):
        """Add multi-head self-attention"""
        self.add_module(
            'msa'+str(index),
            nn.MultiheadAttention(
                embed_dim=self.__embed_dim,
                num_heads=2,
                batch_first=True
            )
        )

    def __add_msa_drop(self, index: int):
        """Add dropout after attention"""
        self.add_module('msa_drop'+str(index), nn.Dropout(0.3))

    def __add_tcn(self, index: int, num_electrodes: int):
        """Add TCN block"""
        self.add_module(
            'tcn'+str(index),
            nn.Sequential(
                nn.Conv1d(num_electrodes, 32, self.tcn_kernel_size, padding='same'),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Conv1d(32, 32, self.tcn_kernel_size, padding='same'),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3)
            )
        )

    def __add_recov(self, index: int):
        """Add recovery layer for dimension matching"""
        self.add_module(
            're'+str(index),
            nn.Conv1d(self.win_len, 32, 4, padding='same')
        )

    def __add_feature_projection(self, index: int):
        """Add feature projection for each window"""
        intermediate_dim = 256
        self.add_module(
            'feature_proj'+str(index),
            nn.Sequential(
                nn.Linear(self.__dense_dim, intermediate_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        )

    def __add_final_projection(self):
        """Add final projection to embedding dimension"""
        self.final_projection = nn.Sequential(
            nn.Linear(self.__concat_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, self.embedding_dim)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, in_channels, num_electrodes, chunk_size)

        Returns:
            embeddings: Normalized embeddings (batch_size, embedding_dim)
        """
        # Convolutional block
        x = self.conv_block(x)
        x = x[:, :, -1, :]
        x = x.permute(0, 2, 1)

        # Process each window
        window_features = []
        for i in range(self.num_windows):
            st = i
            end = x.shape[1] - self.num_windows + i + 1
            x2 = x[:, st:end, :]

            # Attention
            x2_ = self.get_submodule("msa"+str(i))(x2, x2, x2)[0]
            x2_ = self.get_submodule("msa_drop"+str(i))(x2)
            x2 = torch.add(x2, x2_)

            # TCN
            for j in range(self.tcn_depth):
                out = self.get_submodule("tcn"+str((i+1)*j))(x2)
                if x2.shape[1] != out.shape[1]:
                    x2 = self.get_submodule("re"+str(i))(x2)
                x2 = torch.add(x2, out)
                x2 = nn.ELU()(x2)

            # Feature projection
            x2 = x2[:, -1, :]
            x2 = self.get_submodule("feature_proj"+str(i))(x2)
            window_features.append(x2)

        # Concatenate window features
        concat_features = torch.cat(window_features, dim=1)

        # Final projection to embedding dimension
        embeddings = self.final_projection(concat_features)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
