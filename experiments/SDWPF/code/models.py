"""Thirteen benchmark methods with explicit adapters to official implementations.

``build(name, cfg)`` receives the frozen protocol configuration so that the
proposed model and its multi-turbine adaptation are instantiated from
``config.json`` rather than from duplicated literals. The single-turbine vendor
adapters were measured only in the 134-turbine / 16-variable / 96->48 setting;
``build`` refuses any other shape instead of silently reshaping inputs.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgsa import SGSA
from multigsa_adapted import MultiGSAAdapted
from sgsa_utils.checkpoints import frequency_indices

VENDOR = Path(__file__).resolve().parents[3] / 'vendor'
NAMES = ['SGSA', 'MultiGSA_adapted', 'LSTM', 'Transformer', 'MTGNN', 'FEDformer',
         'GraphWaveNet', 'AGCRN', 'STID_no_calendar', 'DLinear', 'PatchTST',
         'PDFormer_train_graph', 'StemGNN_power_only']

# Shape the vendor adapters below were written and measured for.
FIXED_ADAPTER_SHAPE = dict(seq_len=96, num_nodes=134, input_dim=16, pred_len=48)
CFG_DEFAULTS = dict(seq_len=96, num_nodes=134, input_dim=16, pred_len=48,
                    d_model=64, dropout=0.1, graph_rank=16, rho=0.25)


def _cfg(cfg):
    merged = dict(CFG_DEFAULTS)
    merged.update({k: v for k, v in (cfg or {}).items() if k in CFG_DEFAULTS})
    return merged


def _require_fixed_shape(cfg, name):
    for key, value in FIXED_ADAPTER_SHAPE.items():
        if cfg[key] != value:
            raise ValueError(f'{name} adapter only supports {key}={value}, got {cfg[key]}')


class Temporal(nn.Module):
    """Single-turbine LSTM / Transformer baseline sharing weights across turbines."""

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


class MTGNNAdapter(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        _require_fixed_shape(cfg, 'MTGNN')
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


class FEDformerAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        _require_fixed_shape(cfg, 'FEDformer')
        sys.path.insert(0, str(VENDOR / 'FEDformer'))
        spec = importlib.util.spec_from_file_location(
            'official_fedformer', VENDOR / 'FEDformer/models/FEDformer.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Eight heads required by official FourierBlock (hard-coded internal head count).
        config = SimpleNamespace(version='Fourier', mode_select='random', modes=32,
                                 seq_len=cfg['seq_len'], label_len=cfg['pred_len'],
                                 pred_len=cfg['pred_len'], output_attention=False,
                                 moving_avg=25, enc_in=cfg['input_dim'], dec_in=cfg['input_dim'],
                                 c_out=cfg['input_dim'], d_model=cfg['d_model'], embed='timeF',
                                 freq='h', dropout=cfg['dropout'], n_heads=8,
                                 d_ff=2 * cfg['d_model'], e_layers=2, d_layers=1, activation='gelu')
        self.net = mod.Model(config)

    def forward(self, x):
        b, t, n, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        # No extra calendar covariates for any model. All time-mark channels are zero.
        mark = z.new_zeros((b * n, t, 4))
        dec = torch.cat([z[:, -48:], z.new_zeros((b * n, 48, f))], dim=1)
        y = self.net(z, mark, dec, mark)
        return y[:, :, -1].reshape(b, n, -1).permute(0, 2, 1)


def build(name, cfg=None, device='cuda'):
    """Instantiate one benchmark architecture.

    ``device`` only matters for the MTGNN adapter, which needs to construct its
    internal tensors on the same device it will run on (CPU smoke checks use
    ``device='cpu'``).
    """
    cfg = _cfg(cfg)
    if name in ['FEDformer', 'PatchTST']:
        from types import ModuleType
        for key in list(sys.modules):
            if key == 'layers' or key.startswith('layers.'):
                sys.modules.pop(key)
        package = ModuleType('layers')
        package.__path__ = [str(VENDOR / ('FEDformer/layers' if name == 'FEDformer'
                                          else 'PatchTST/PatchTST_supervised/layers'))]
        sys.modules['layers'] = package
    if name == 'SGSA':
        return SGSA(cfg['seq_len'], cfg['num_nodes'], cfg['input_dim'], cfg['pred_len'],
                    cfg['d_model'], cfg['dropout'], cfg['graph_rank'], cfg['rho'])
    if name == 'MultiGSA_adapted':
        return MultiGSAAdapted(cfg['seq_len'], cfg['num_nodes'], cfg['input_dim'],
                               cfg['pred_len'], cfg['d_model'], cfg['dropout'])
    if name in ['LSTM', 'Transformer']:
        return Temporal(name, cfg)
    if name == 'MTGNN':
        return MTGNNAdapter(cfg, device=device)
    if name == 'FEDformer':
        return FEDformerAdapter(cfg)
    if name in ['GraphWaveNet', 'AGCRN', 'STID_no_calendar', 'DLinear', 'PatchTST']:
        _require_fixed_shape(cfg, name)
        import adapters_extended
        return adapters_extended.build(name)
    if name in ['PDFormer_train_graph', 'StemGNN_power_only']:
        _require_fixed_shape(cfg, name)
        import adapters_extra
        return adapters_extra.build(name)
    raise ValueError(name)


def parameter_counts(model):
    """Report complex coefficients as two real scalars, as stated in the paper."""
    params = list(model.parameters())
    return dict(tensor_elements=sum(p.numel() for p in params),
                real_scalar_parameters=sum(p.numel() * (2 if p.is_complex() else 1) for p in params),
                parameter_bytes=sum(p.numel() * p.element_size() for p in params),
                trainable_real_scalars=sum(p.numel() * (2 if p.is_complex() else 1)
                                           for p in params if p.requires_grad))


__all__ = ['NAMES', 'build', 'parameter_counts', 'frequency_indices']
