# ==========================================
# FILE: model_def.py  (REVISED — no quantum)
# ==========================================
#
# What changed from the original:
#   REMOVED  — PennyLane quantum circuit (VQC, 7 qubits)
#   REMOVED  — HybridQuantumGNN class
#   REMOVED  — pre_net, q_layer, post_net (quantum pipeline)
#   ADDED    — SpatialGNN class (GATv2 only, cleaner architecture)
#   ADDED    — input_proj linear layer (replaces quantum pre/post processing)
#   ADDED    — skip connection (residual path, replaces classical_bypass)
#   CHANGED  — in_dim: 7 → 10 (9 base features + 1 XGBoost risk score appended)
#
# How it fits the pipeline:
#   XGBoost Risk Model  →  raw_risk_score (scalar, per shop)
#   Append to features  →  node feature matrix [n_shops, 10]
#   SpatialGNN.forward  →  spatial_risk_score (scalar, per shop)
#                           (propagated through road network graph)
#
# Edge attributes encode road type:
#   1.0 = National Highway   (high attention weight)
#   0.6 = State Highway
#   0.3 = Rural / Dirt Road  (low attention weight)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class SpatialGNN(nn.Module):
    """
    Graph Attention Network for spatial demand propagation
    across the Jangaon village road network.

    Nodes  = retailer shops / villages
    Edges  = road connections between them
    Edge attr = road type weight (highway → higher attention)

    The model answers: "given what's happening in neighboring
    villages (weighted by road connectivity), what is the true
    risk for this shop — beyond what its own features say?"
    """

    def __init__(self, in_dim: int = 10, hidden_dim: int = 64):
        """
        Args:
            in_dim:     Number of node features.
                        = 9 base features + 1 XGBoost risk score = 10
            hidden_dim: Internal representation size. 64 is a good default
                        for a ~100-node village graph.
        """
        super().__init__()

        # --- 1. Input Projection ---
        # Maps raw node features (10) to the hidden space GATv2 works in.
        # Equivalent role to the old classical_bypass + post_net combined.
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # --- 2. Graph Attention Layer 1 ---
        # 4 independent attention heads — each learns a different
        # "perspective" on which neighbors matter most.
        # concat=True stacks head outputs → hidden_dim * 4 channels.
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=4,
            concat=True,
            edge_dim=1,      # one edge feature: road type weight
            dropout=0.1,
        )

        # --- 3. Graph Attention Layer 2 ---
        # Aggregates the multi-head neighborhood representation
        # into a single scalar risk score per node.
        # concat=False averages the single head output.
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * 4,
            out_channels=1,
            heads=1,
            concat=False,
            edge_dim=1,
            dropout=0.1,
        )

        # --- 4. Skip Connection ---
        # Adds a residual path from the projected input directly
        # to the output. Prevents gradient vanishing on small graphs
        # and stabilises training. Weight = 0.1 keeps it as a
        # "corrective signal" rather than dominating the graph output.
        self.skip = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Args:
            data: torch_geometric.data.Data with:
                  data.x          — node features [n_nodes, in_dim]
                  data.edge_index — edge connectivity [2, n_edges]
                  data.edge_attr  — road type weights [n_edges, 1+]

        Returns:
            Tensor [n_nodes, 1] — spatial risk score per shop, sigmoid-normalised
        """
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        # Road type weight is the first (and currently only) edge attribute
        edge_weight = ea[:, 0].unsqueeze(-1)  # shape [n_edges, 1]

        # Project input features to hidden space
        x_proj = F.elu(self.input_proj(x))             # [n_nodes, hidden_dim]

        # Graph attention pass 1: aggregate immediate neighbors
        x = F.elu(self.gat1(x_proj, ei, edge_weight))  # [n_nodes, hidden_dim*4]

        # Graph attention pass 2: compress to scalar
        x_out = self.gat2(x, ei, edge_weight)           # [n_nodes, 1]

        # Add skip connection (residual from projected input)
        x_skip = self.skip(x_proj)                      # [n_nodes, 1]

        # Sigmoid → normalise to [0, 1] range (probability of stockout risk)
        return torch.sigmoid(x_out + 0.1 * x_skip)
