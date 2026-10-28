"""Temporal SimAM: parameter-free statistical attention (paper Eq. 4).

With the mean ``mu`` and the population variance ``v`` taken along the
historical axis, the implemented weights are

    e   = (Z - mu)^2 / (4 * (v + eps)) + 0.5
    a   = sigmoid(e)
    H_a = a * Z

The ``+ 0.5`` offset is part of the SimAM energy definition and keeps the
sigmoid centred, so ``a`` is about 0.5 for a feature sitting at its window mean.
There is no learnable parameter here.
"""
import torch
from torch import nn


class SimAM1D(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: [batch, time, turbine, hidden]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        energy = (x - mean).square() / (4.0 * (var + self.eps)) + 0.5
        return x * torch.sigmoid(energy)
