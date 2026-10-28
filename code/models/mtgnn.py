"""MTGNN adapter around the unmodified vendor implementation."""
import sys

import torch
from torch import nn

from models._vendor import VENDOR


class MTGNNAdapter(nn.Module):
    """Official ``gtnet`` fed with the [B, N, F, T] layout it expects.

    Graph learning uses the vendor's subgraph sampling; its top-k tie-breaking
    draws random numbers even in eval mode, which is why evaluation runs under a
    fixed RNG stream (see ``exp/evaluator.py``).
    """

    def __init__(self, cfg, device='cuda'):
        super().__init__()
        sys.path.insert(0, str(VENDOR / 'MTGNN'))
        from net import gtnet
        self.net = gtnet(True, True, 2, cfg['num_nodes'], torch.device(device),
                         dropout=cfg['dropout'], subgraph_size=20, node_dim=40,
                         dilation_exponential=2, conv_channels=64, residual_channels=64,
                         skip_channels=64, end_channels=128, seq_length=cfg['seq_len'],
                         in_dim=cfg['input_dim'], out_dim=cfg['pred_len'], layers=3,
                         propalpha=.05, tanhalpha=3, layer_norm_affline=True)

    def forward(self, x):
        return self.net(x.permute(0, 3, 2, 1)).squeeze(-1)
