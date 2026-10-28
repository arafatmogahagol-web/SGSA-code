"""PDFormer and StemGNN adapters (author code, loaded from ``vendor/``)."""
import math

import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

from models._vendor import VENDOR, package, patched_source


def _num_batches(cfg):
    if 'num_batches' in cfg and cfg['num_batches']:
        return int(cfg['num_batches'])
    windows = cfg.get('expected_windows', {}).get('train')
    if not windows:
        return 1
    return int(math.ceil(windows / cfg['batch_size']))


class PDFormer(nn.Module):
    """PDFormer with training-only graph priors (no coordinates, no test data).

    The author's loss registry imports scikit-learn for unused R2 helpers; the
    benchmark applies its own masked MSE, so that single import is replaced by
    ``None`` and no vendor loss function is ever called.
    """

    def __init__(self, cfg, priors):
        super().__init__()
        package('libcity', VENDOR / 'PDFormer/libcity')
        package('libcity.model', VENDOR / 'PDFormer/libcity/model')
        path = VENDOR / 'PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py'
        official = patched_source(
            'official_pdformer', path,
            [('from libcity.model import loss', 'loss = None  # external benchmark masked MSE only')])
        config = dict(dataset=f"{cfg['dataset']}_train_graph", embed_dim=64, skip_dim=256,
                      lape_dim=8, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, enc_depth=6,
                      mlp_ratio=4, drop=cfg['dropout'], attn_drop=cfg['dropout'],
                      drop_path=cfg['dropout'], input_window=cfg['seq_len'],
                      output_window=cfg['pred_len'], output_dim=1, add_time_in_day=False,
                      add_day_in_week=False, device=torch.device('cpu'),
                      use_curriculum_learning=False, max_epoch=cfg['epochs'],
                      far_mask_delta=3, dtw_delta=5)
        features = dict(num_nodes=cfg['num_nodes'], feature_dim=cfg['input_dim'], ext_dim=0,
                        num_batches=_num_batches(cfg), scaler=None,
                        dtw_matrix=priors['dtw'], adj_mx=priors['adjacency'],
                        sd_mx=1 - priors['correlation'],
                        sh_mx=torch.from_numpy(np.ascontiguousarray(priors['hops'])),
                        pattern_keys=priors['pattern_keys'])
        self.net = official.PDFormer(config, features)
        for key in ['geo_mask', 'sem_mask', 'pattern_keys']:
            value = getattr(self.net, key)
            delattr(self.net, key)
            self.net.register_buffer(key, value)
        self.register_buffer('lap_pe', torch.from_numpy(priors['lap_pe']))

    def forward(self, x):
        # The pattern extractor uses its first channel as the target power.
        x = torch.cat([x[..., -1:], x[..., :-1]], dim=-1)
        return self.net({'X': x}, self.lap_pe).squeeze(-1)


class StemGNN(nn.Module):
    """StemGNN on the historical power channel only.

    The upstream file calls the removed ``torch.rfft``/``torch.irfft`` APIs. They
    are replaced by equivalent two-sided ``torch.fft`` calls that preserve both
    real and imaginary components with the original normalization.
    """

    def __init__(self, cfg):
        super().__init__()
        path = VENDOR / 'StemGNN/models/base_model.py'
        official = patched_source(
            'official_stemgnn_compat', path,
            [('torch.rfft(input, 1, onesided=False)',
              'torch.view_as_real(torch.fft.fft(input, dim=-1, norm="backward"))'),
             ('torch.irfft(time_step_as_inner, 1, onesided=False)',
              'torch.fft.ifft(torch.view_as_complex(time_step_as_inner.contiguous()), '
              'dim=-1, norm="backward").real')])
        self.net = official.Model(units=cfg['num_nodes'], stack_cnt=2,
                                  time_step=cfg['seq_len'], multi_layer=5,
                                  horizon=cfg['pred_len'], dropout_rate=cfg['dropout'],
                                  device='cpu')

    def forward(self, x):
        return self.net(x[..., -1])[0]
