"""Single-series baselines sharing weights across turbines: LSTM and Transformer.

Both treat each turbine as an independent univariate series with a shared
parameter set, then map the last hidden state to the whole horizon. No calendar
covariates are used for any model in this study.
"""
import torch
from torch import nn


class Temporal(nn.Module):
    def __init__(self, kind, cfg):
        super().__init__()
        self.kind = kind
        seq_len, input_dim, d_model, pred_len, dropout = (
            cfg['seq_len'], cfg['input_dim'], cfg['d_model'], cfg['pred_len'], cfg['dropout'])
        self.proj = nn.Linear(input_dim, d_model)
        if kind == 'LSTM':
            self.encoder = nn.LSTM(d_model, d_model, 2, batch_first=True, dropout=dropout)
        else:
            self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(self.position, std=.02)
            layer = nn.TransformerEncoderLayer(d_model, 4, 2 * d_model, dropout,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def forward(self, x):
        b, t, n, f = x.shape
        z = self.proj(x.permute(0, 2, 1, 3).reshape(b * n, t, f))
        z = self.encoder(z)[0] if self.kind == 'LSTM' else self.encoder(z + self.position)
        return self.head(self.norm(z[:, -1])).reshape(b, n, -1).permute(0, 2, 1)
