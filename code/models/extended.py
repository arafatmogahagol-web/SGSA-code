"""Author implementations behind small, documented adapters.

Vendor source is never edited on disk. Two adapters load a patched copy of an
official file because of API changes in current PyTorch:

* GraphWaveNet: the 2019 code calls ``nn.Conv1d`` with a two-dimensional kernel on
  a four-dimensional input; ``Conv2d`` is the intended operation.
* (see ``models/extra.py`` for the StemGNN FFT replacements.)

Everything else is imported as-is.
"""
import sys

import torch
from torch import nn
from types import SimpleNamespace

from models._vendor import VENDOR, module, package, patched_source

NAMES = ['GraphWaveNet', 'AGCRN', 'STID_no_calendar', 'DLinear', 'PatchTST']


class GraphWaveNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        path = VENDOR / 'GraphWaveNet/model.py'
        official = patched_source('official_graphwavenet_compat', path,
                                  [('nn.Conv1d(', 'nn.Conv2d(', 3)])
        self.net = official.gwnet(torch.device('cpu'), cfg['num_nodes'], dropout=cfg['dropout'],
                                  supports=None, gcn_bool=True, addaptadj=True,
                                  in_dim=cfg['input_dim'], out_dim=cfg['pred_len'],
                                  residual_channels=64, dilation_channels=64, skip_channels=64,
                                  end_channels=128, kernel_size=2, blocks=4, layers=5)
        assert self.net.receptive_field >= cfg['seq_len'], 'sequence shorter than the receptive field'

    def forward(self, x):
        y = self.net(x.permute(0, 3, 2, 1))
        assert y.shape[-1] == 1, y.shape
        return y.squeeze(-1)


class AGCRN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # The original imports are ``model.AGCRNCell`` / ``model.AGCN``.
        package('model', VENDOR / 'AGCRN/model')
        official = module('official_agcrn', VENDOR / 'AGCRN/model/AGCRN.py')
        config = SimpleNamespace(num_nodes=cfg['num_nodes'], input_dim=cfg['input_dim'],
                                 rnn_units=64, output_dim=1, horizon=cfg['pred_len'],
                                 num_layers=2, default_graph=True, embed_dim=10, cheb_k=2)
        self.net = official.AGCRN(config)
        # The author's Run.py initializes all parameters, including the empty pools.
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        return self.net(x, None, teacher_forcing_ratio=0.).squeeze(-1)


class STID(nn.Module):
    """STID with node identity only; no time-of-day or day-of-week inputs."""

    def __init__(self, cfg):
        super().__init__()
        module('official_stid_arch', VENDOR / 'STID/stid/arch/__init__.py', True)
        arch = module('official_stid_arch.stid_arch', VENDOR / 'STID/stid/arch/stid_arch.py')
        self.net = arch.STID(num_nodes=cfg['num_nodes'], node_dim=32, input_len=cfg['seq_len'],
                             input_dim=cfg['input_dim'], embed_dim=64, output_len=cfg['pred_len'],
                             num_layer=3, temp_dim_tid=32, temp_dim_diw=32, time_of_day_size=48,
                             day_of_week_size=7, if_T_i_D=False, if_D_i_W=False, if_node=True)
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.p = cfg['dropout']

    def forward(self, x):
        return self.net(x, None, 0, 0, self.training).squeeze(-1)


class DLinear(nn.Module):
    """Channel-independent DLinear; the target channel is selected for scoring."""

    def __init__(self, cfg):
        super().__init__()
        official = module('official_dlinear', VENDOR / 'DLinear/models/DLinear.py')
        self.net = official.Model(SimpleNamespace(seq_len=cfg['seq_len'], pred_len=cfg['pred_len'],
                                                 individual=False, enc_in=cfg['input_dim']))
        self.pred_len = cfg['pred_len']

    def forward(self, x):
        b, t, n, c = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        return self.net(z)[..., -1].reshape(b, n, self.pred_len).permute(0, 2, 1)


class PatchTST(nn.Module):
    """Channel-independent PatchTST with RevIN; the target channel is scored."""

    def __init__(self, cfg):
        super().__init__()
        sys.path.insert(0, str(VENDOR / 'PatchTST/PatchTST_supervised'))
        official = module('official_patchtst',
                          VENDOR / 'PatchTST/PatchTST_supervised/models/PatchTST.py')
        config = SimpleNamespace(enc_in=cfg['input_dim'], seq_len=cfg['seq_len'],
                                 pred_len=cfg['pred_len'], e_layers=2, n_heads=4, d_model=64,
                                 d_ff=128, dropout=cfg['dropout'], fc_dropout=cfg['dropout'],
                                 head_dropout=cfg['dropout'], individual=False, patch_len=16,
                                 stride=8, padding_patch='end', revin=True, affine=True,
                                 subtract_last=False, decomposition=False, kernel_size=25)
        self.net = official.Model(config, attn_dropout=cfg['dropout'])
        self.pred_len = cfg['pred_len']

    def forward(self, x):
        b, t, n, c = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        return self.net(z)[..., -1].reshape(b, n, self.pred_len).permute(0, 2, 1)
