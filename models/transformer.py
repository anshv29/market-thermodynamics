"""
Temporal Transformer — Market Regime Forecaster
================================================
Takes a sequence of latent market states z(t-T)...z(t)
and predicts:
  1. z(t+1)     — next latent state (regression)
  2. r(t+1)     — next regime label (classification)

Architecture:
  - Patch-based tokenization (5-day patches)
  - 4-layer transformer encoder
  - Dual output heads (regression + classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Groups consecutive days into patches before feeding to transformer.
    Reduces noise — 5-day patches are more stable than daily tokens.

    Example: 60-day sequence → 12 patches of 5 days each
    """
    def __init__(self, latent_dim: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(latent_dim * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, latent_dim)
        B, T, D = x.shape
        n_patches = T // self.patch_size

        # Reshape into patches
        x = x[:, :n_patches * self.patch_size, :]
        x = x.reshape(B, n_patches, self.patch_size * D)

        # Project to model dimension
        return self.projection(x)


class PositionalEncoding(nn.Module):
    """
    Adds positional information to patch embeddings.
    Without this, the transformer doesn't know the order of patches.
    """
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MarketTransformer(nn.Module):
    """
    Transformer for regime forecasting.

    Input:  sequence of z(t) vectors over last SEQ_LEN days
    Output: predicted z(t+1) + regime probabilities P(r(t+1))
    """

    def __init__(self,
                 latent_dim:  int = 16,
                 n_regimes:   int = 6,
                 seq_len:     int = 60,
                 patch_size:  int = 5,
                 d_model:     int = 128,
                 n_heads:     int = 8,
                 n_layers:    int = 4,
                 d_ff:        int = 256,
                 dropout:     float = 0.1):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_len    = seq_len
        self.patch_size = patch_size
        self.n_patches  = seq_len // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(latent_dim, patch_size, d_model)
        self.pos_encode  = PositionalEncoding(d_model, max_len=self.n_patches + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # CLS token — aggregates sequence information
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output heads
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_regimes),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, latent_dim)
        Returns:
          z_pred:       (batch, latent_dim) — predicted next latent state
          regime_logits:(batch, n_regimes)  — raw scores for regime classification
        """
        B = x.shape[0]

        # Patch embedding
        patches = self.patch_embed(x)              # (B, n_patches, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        patches = torch.cat([cls, patches], dim=1) # (B, n_patches+1, d_model)

        # Positional encoding
        patches = self.pos_encode(patches)

        # Transformer
        encoded = self.transformer(patches)        # (B, n_patches+1, d_model)

        # Use CLS token output for predictions
        cls_out = encoded[:, 0, :]                 # (B, d_model)

        z_pred        = self.regression_head(cls_out)
        regime_logits = self.classification_head(cls_out)

        return z_pred, regime_logits