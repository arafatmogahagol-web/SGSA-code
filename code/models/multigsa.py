"""MultiGSA-style baseline adapted to the multi-turbine tensor layout.

Structural adaptation of the MultiGSA idea (multi-scale time processing plus
learned graphs), not a reproduction of the original paper's datasets,
architecture settings or training recipe. ``MultiGSA_adapted`` is reported under
the shared data protocol of this study.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from layers import SimAM1D


class LearnedGraph(nn.Module):
    """Row-normalized affinity from two learned embeddings (as in SGSA)."""

    def __init__(self, count: int, rank: int = 16):
        super().__init__()
        self.left = nn.Parameter(torch.empty(count, rank))
        self.right = nn.Parameter(torch.empty(rank, count))
        nn.init.xavier_uniform_(self.left)
        nn.init.xavier_uniform_(self.right)

    def forward(self) -> torch.Tensor:
        logits = F.elu(self.left @ self.right)
        return torch.softmax(logits, dim=-1)


class MultiGSAAdapted(nn.Module):
    """Multi-scale temporal encoders, learned time/turbine graphs, SimAM weighting."""

    def __init__(self, seq_len: int, num_nodes: int, input_dim: int, pred_len: int,
                 d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        if seq_len % 4:
            raise ValueError('seq_len must be divisible by four for the 1/2 and 1/4 scales')
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.scale_fusion = nn.Linear(3 * d_model, d_model)
        self.time_graph = LearnedGraph(seq_len)
        self.turbine_graph = LearnedGraph(num_nodes)
        self.time_norm = nn.LayerNorm(d_model)
        self.fuse_gate = nn.Linear(2 * d_model, d_model)
        self.simam = SimAM1D()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.time_projection = nn.Linear(seq_len, pred_len)
        self.output_projection = nn.Linear(d_model, 1)
        self.latest_adj = None
        self.fusion_residual_scale = 1.0

    @staticmethod
    def _pool_to_length(x: torch.Tensor, factor: int, out_len: int) -> torch.Tensor:
        # x: [B, T, N, D]
        b, t, n, d = x.shape
        z = x.permute(0, 2, 3, 1).reshape(b, n * d, t)
        z = F.avg_pool1d(z, kernel_size=factor, stride=factor)
        z = F.interpolate(z, size=out_len, mode='linear', align_corners=False)
        return z.reshape(b, n, d, out_len).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, F]
        h = self.input_proj(x)
        h2 = self._pool_to_length(h, 2, self.seq_len)
        h4 = self._pool_to_length(h, 4, self.seq_len)
        multi = self.time_norm(F.gelu(self.scale_fusion(torch.cat([h, h2, h4], dim=-1))))
        time_adj = self.time_graph()
        h_time = torch.einsum('ts,bsnd->btnd', time_adj, multi)
        node_adj = self.turbine_graph()
        self.latest_adj = node_adj.detach()
        h_node = torch.einsum('ij,btjd->btid', node_adj, multi)
        gate = torch.sigmoid(self.fuse_gate(torch.cat([h_time, h_node], dim=-1)))
        h = gate * h_time + (1.0 - gate) * h_node
        h = self.simam(h)
        h = self.norm(h + self.fusion_residual_scale * self.dropout(multi))
        return self._forecast(h)

    def _forecast(self, h: torch.Tensor) -> torch.Tensor:
        # Direct multi-step projection, independently for each turbine and hidden channel.
        z = self.time_projection(h.permute(0, 2, 3, 1))  # [B, N, D, P]
        z = self.output_projection(z.permute(0, 3, 1, 2))
        return z.squeeze(-1)
