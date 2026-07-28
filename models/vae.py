"""
Variational Autoencoder — Market State Encoder
===============================================
Compresses daily market snapshots into a 16-dimensional
latent vector z(t) that captures the essential thermodynamic
state of the market.

Input:  full feature vector (physics + aggregated GNN embeddings)
Output: z(t) — 16-dim latent market fingerprint

Architecture:
  Encoder: input_dim → 128 → 64 → [mu, logvar] (16-dim each)
  Decoder: 16 → 64 → 128 → input_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketVAE(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 16, beta: float = 1.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta       = beta  # KL weight (beta-VAE)

        # ── Encoder ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Latent distribution parameters
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # ── Decoder ──────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x: torch.Tensor):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization trick:
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        Makes sampling differentiable so we can backprop through it.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # at inference, just use the mean

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z)
        return x_recon, mu, logvar, z

    def vae_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor):
        """
        Total VAE loss = Reconstruction loss + beta * KL divergence

        Reconstruction loss: how well the decoder recreates the input
        KL divergence: forces the latent space to be smooth and Gaussian
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss