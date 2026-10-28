"""Canonical SGSA implementation shared by both experiment protocols.

This module is the single source of truth for the proposed model. The copies that
previously existed under ``experiments/*/code`` (``sgsa.py``, ``model.py``,
``sgsa_layers.py``, ``layers.py``) are now thin re-export shims so that the two
protocols cannot drift apart.

Architecture (paper Sections 3.2-3.4):

1. ``input_proj``: per-timestep linear map F -> d.
2. Static turbine graph: ``S = softmax_row(ELU(EL @ ER))`` with
   ``EL in R^(N x r)`` and ``ER in R^(r x N)``; S is shared by every historical
   step and every batch element (no per-window graph construction).
3. Spatial aggregation ``A[b,t,i,:] = sum_j S[i,j] H[b,t,j,:]`` followed by the
   scaled self-feature residual
   ``Hs = LN_s(Dropout(GELU(A Ws + bs)) + rho * H0)`` with rho fixed at 0.25.
4. Learnable positional encoding and layer normalization along the hidden axis:
   ``Z = LN_t(Hs + P_pos)``.
5. Temporal SimAM weighting computed only from feature statistics along the
   history axis, then the last step is mapped directly to P futures by a
   turbine-shared MLP head.

Variable names keep the historical checkpoint keys, so ``state_dict`` files
produced by the released runs still load with ``strict=True``.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SimAM1D(nn.Module):
    """Parameter-free temporal SimAM weighting (paper Eq. 4).

    With the mean ``mu`` and the population variance ``v`` taken along the
    historical axis, the implemented weights are

        e = (Z - mu)^2 / (4 * (v + eps)) + 0.5
        a = sigmoid(e)
        H_a = a * Z

    The ``+ 0.5`` offset is part of the SimAM energy definition and keeps the
    sigmoid centred, so ``a`` is ~0.5 for features at their window mean. There is
    no learnable parameter here.
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        energy = (x - mean).square() / (4.0 * (var + self.eps)) + 0.5
        return x * torch.sigmoid(energy)


class SGSA(nn.Module):
    """Static graph with statistical attention.

    Defaults reproduce the SDWPF protocol (N=134, F=16). The Hill of Towie
    protocol passes its own N and F explicitly; no dataset-specific value is
    inferred from the data.
    """

    def __init__(self, seq_len=96, num_nodes=134, input_dim=16,
                 pred_len=48, d_model=64, dropout=0.1, graph_rank=16, rho=0.25):
        super().__init__()
        self.seq_len, self.num_nodes = seq_len, num_nodes
        self.input_dim, self.d_model, self.rho = input_dim, d_model, rho
        self.input_proj = nn.Linear(input_dim, d_model)
        self.node_left = nn.Parameter(torch.empty(num_nodes, graph_rank))
        self.node_right = nn.Parameter(torch.empty(graph_rank, num_nodes))
        nn.init.xavier_uniform_(self.node_left)
        nn.init.xavier_uniform_(self.node_right)
        self.spatial_proj = nn.Linear(d_model, d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, 1, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.filter = SimAM1D()
        self.forecast = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d_model, pred_len))
        self.latest_adj = None

    def forward(self, x):
        b, length, nodes, features = x.shape
        if (length, nodes, features) != (self.seq_len, self.num_nodes, self.input_dim):
            raise ValueError('Unexpected input dimensions')
        h = self.input_proj(x)
        s = torch.softmax(F.elu(self.node_left @ self.node_right), dim=-1)
        a = s[None].expand(b, -1, -1)
        self.latest_adj = a.detach()
        spatial = torch.einsum('bij,btjd->btid', a, h)
        h = self.spatial_norm(self.rho * h + self.dropout(F.gelu(self.spatial_proj(spatial))))
        z = (h + self.position).permute(0, 2, 1, 3).reshape(b * nodes, length, self.d_model)
        z = self.temporal_norm(z)
        h = z.reshape(b, nodes, length, self.d_model).permute(0, 2, 1, 3)
        return self.forecast(self.filter(h)[:, -1]).permute(0, 2, 1)


def load_historical_checkpoint(model, state):
    """Drop only the two proven-unused Q/K matrices; all active keys load strictly."""
    unused = {'query.weight', 'key.weight'}
    extra = set(state) - set(model.state_dict())
    if extra != unused:
        raise ValueError(f'Unexpected historical checkpoint keys: {extra}')
    model.load_state_dict({k: v for k, v in state.items() if k not in unused}, strict=True)
    return sorted(unused)
