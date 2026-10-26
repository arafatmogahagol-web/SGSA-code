"""Temporal SimAM used by the audited historical SDWPF implementation."""
import torch
from torch import nn

class SimAM1D(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        energy = (x - mean).square() / (4.0 * (var + self.eps)) + 0.5
        return x * torch.sigmoid(energy)
