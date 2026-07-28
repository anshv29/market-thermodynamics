"""
Graph Neural Network — Stock Interaction Model
===============================================
Models the market as a graph where:
  - Nodes = individual stocks
  - Edges = correlation-based interaction potential Φ_ij
  - Node features = physics features (PE, acceleration, vol)
  - Task = predict next-day volatility for each stock

Architecture: 2-layer Graph Attention Network (GATConv)
Output: 32-dimensional embedding per stock per day
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class MarketGNN(nn.Module):
    """
    Two-layer Graph Attention Network.

    Layer 1: Input features → 64 hidden dims (8 attention heads)
    Layer 2: 64 hidden dims → 32 embedding dims (1 attention head)

    The attention mechanism learns which neighboring stocks
    matter most for each stock's prediction.
    """

    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 64,
                 embedding_dim: int = 32,
                 heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()

        self.dropout = dropout

        # Layer 1: node_feature_dim → hidden_dim (with multi-head attention)
        self.conv1 = GATConv(
            in_channels=node_feature_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,   # concatenate heads → hidden_dim total
        )

        # Layer 2: hidden_dim → embedding_dim (single head)
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=embedding_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        # Prediction head: embedding → next-day volatility
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2 — this is the embedding
        embedding = self.conv2(x, edge_index)
        embedding = F.elu(embedding)

        # Prediction
        out = self.predictor(embedding)

        return embedding, out.squeeze(-1)


def build_graph(node_features: torch.Tensor,
                correlation_matrix: torch.Tensor,
                threshold: float = 0.3) -> Data:
    """
    Build a PyTorch Geometric graph for one day.

    Args:
        node_features      — (N, F) tensor of stock features
        correlation_matrix — (N, N) tensor of pairwise correlations
        threshold          — minimum |correlation| to create an edge

    Returns:
        PyTorch Geometric Data object
    """
    N = node_features.shape[0]

    # Find all pairs above the correlation threshold
    edge_sources = []
    edge_targets = []
    edge_weights = []

    corr = correlation_matrix.numpy() if hasattr(correlation_matrix, 'numpy') else correlation_matrix

    for i in range(N):
        for j in range(i + 1, N):
            weight = abs(float(corr[i, j]))
            if weight >= threshold:
                # Add both directions (undirected graph)
                edge_sources.extend([i, j])
                edge_targets.extend([j, i])
                edge_weights.extend([weight, weight])

    if len(edge_sources) == 0:
        # Fallback: connect each node to itself if no edges found
        edge_sources = list(range(N))
        edge_targets = list(range(N))
        edge_weights = [1.0] * N

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr  = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
    )