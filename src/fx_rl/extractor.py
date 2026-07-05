"""GAT feature extractor for Stable-Baselines3.

Turns the Dict observation ``{'nodes': (B, N, 2), 'edges': (B, E, 5)}`` into a
flat ``(B, features_dim)`` embedding that TD3's actor and critic consume.  This is
the thesis's core: the policy reasons over a *graph* of the currency market via
graph attention, rather than a flat feature vector.

Architecture (config-driven via ``ModelConfig``)::

    GATConv(node_feats -> gat_hidden, heads=H, edge_dim=edge_feats) -> LeakyReLU
    GATConv(gat_hidden*H -> gat_out, heads=1, edge_dim=edge_feats)  -> LeakyReLU
    flatten (B, N*gat_out) -> Linear -> ReLU -> LayerNorm -> (B, features_dim)

Unchanged in spirit from the notebook; tidied so layer sizes come from config and
the fully-connected edge topology is built once and registered as a buffer (so it
moves to GPU with the module).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv


class FxGraphExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        gat_hidden: int = 16,
        gat_heads: int = 4,
        gat_out: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        self.n_nodes = observation_space.spaces["nodes"].shape[0]
        self.n_edges = observation_space.spaces["edges"].shape[0]
        n_node_feats = observation_space.spaces["nodes"].shape[1]
        n_edge_feats = observation_space.spaces["edges"].shape[1]
        self.gat_out = gat_out

        self.gat1 = GATConv(
            in_channels=n_node_feats,
            out_channels=gat_hidden,
            heads=gat_heads,
            edge_dim=n_edge_feats,
            add_self_loops=False,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=gat_hidden * gat_heads,
            out_channels=gat_out,
            heads=1,
            edge_dim=n_edge_feats,
            add_self_loops=False,
        )

        # Fully-connected directed topology (no self-loops), built once.
        src, dst = [], []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self.register_buffer("base_edge_index", torch.tensor([src, dst], dtype=torch.long))

        self.linear = nn.Sequential(
            nn.Linear(self.n_nodes * gat_out, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        nodes = observations["nodes"]  # (B, N, node_feats)
        edges = observations["edges"]  # (B, E, edge_feats)
        B = nodes.shape[0]

        x = nodes.reshape(B * self.n_nodes, -1)
        edge_attr = edges.reshape(B * self.n_edges, -1)
        edge_index = torch.cat(
            [self.base_edge_index + b * self.n_nodes for b in range(B)], dim=1
        )

        x = F.leaky_relu(self.gat1(x, edge_index, edge_attr))
        x = F.leaky_relu(self.gat2(x, edge_index, edge_attr))
        x = x.reshape(B, self.n_nodes * self.gat_out)
        return self.linear(x)
