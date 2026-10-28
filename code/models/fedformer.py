"""FEDformer adapter around the unmodified vendor implementation."""
import sys

import torch
from torch import nn
from types import SimpleNamespace

from models._vendor import VENDOR, module


class FEDformerAdapter(nn.Module):
    """Series-wise FEDformer on the shared masked-MSE objective.

    ``n_heads`` is fixed at eight because the official ``FourierBlock`` hard-codes
    that head count internally. All time-mark channels are zero: no model in this
    study receives calendar covariates.
    """

    def __init__(self, cfg):
        super().__init__()
        # The vendor file imports ``layers.*`` and ``utils.masking`` relative to
        # its own root, so that root has to be importable.
        sys.path.insert(0, str(VENDOR / 'FEDformer'))
        official = module('official_fedformer', VENDOR / 'FEDformer/models/FEDformer.py')
        config = SimpleNamespace(version='Fourier', mode_select='random', modes=32,
                                 seq_len=cfg['seq_len'], label_len=cfg['pred_len'],
                                 pred_len=cfg['pred_len'], output_attention=False,
                                 moving_avg=25, enc_in=cfg['input_dim'], dec_in=cfg['input_dim'],
                                 c_out=cfg['input_dim'], d_model=cfg['d_model'], embed='timeF',
                                 freq='h', dropout=cfg['dropout'], n_heads=8,
                                 d_ff=2 * cfg['d_model'], e_layers=2, d_layers=1, activation='gelu')
        self.net = official.Model(config)
        self.seq_len, self.pred_len = cfg['seq_len'], cfg['pred_len']
        self.input_dim = cfg['input_dim']

    def forward(self, x):
        b, t, n, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        mark = z.new_zeros((b * n, t, 4))
        dec = torch.cat([z[:, -self.pred_len:], z.new_zeros((b * n, self.pred_len, f))], dim=1)
        y = self.net(z, mark, dec, mark)
        return y[:, :, -1].reshape(b, n, -1).permute(0, 2, 1)
