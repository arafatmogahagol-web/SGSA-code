"""Model registry for the 13 compared architectures.

``build(name, cfg, device, priors)`` is the only entry point. Every shape comes
from the dataset configuration, so the same registry serves both protocols;
``priors`` is only needed by PDFormer, whose graph/DTW/pattern inputs are fitted
on the training partition and stored beside the dataset.

Two vendor models (FEDformer, PatchTST) import their own top-level package named
``layers``, which is also the name of this repository's own layer package. The
vendor package is installed only while the vendor model is being constructed and
the repository package is restored afterwards, so neither shadows the other.
"""
import sys
from types import ModuleType

from models._vendor import VENDOR

NAMES = ['SGSA', 'MultiGSA_adapted', 'LSTM', 'Transformer', 'MTGNN', 'FEDformer',
         'GraphWaveNet', 'AGCRN', 'STID_no_calendar', 'DLinear', 'PatchTST',
         'PDFormer_train_graph', 'StemGNN_power_only']

LAYER_PACKAGE = {'FEDformer': 'FEDformer/layers',
                 'PatchTST': 'PatchTST/PatchTST_supervised/layers'}

# Vendor packages whose names collide with this repository's own top-level
# packages (``layers`` here, ``utils`` here and in the official FEDformer).
VENDOR_PACKAGES = {'FEDformer': {'layers': 'FEDformer/layers', 'utils': 'FEDformer/utils'},
                   'PatchTST': {'layers': 'PatchTST/PatchTST_supervised/layers'}}


def _install_vendor_packages(name):
    """Temporarily put the vendor's own ``layers``/``utils`` packages in place."""
    saved = {}
    for package_name, relative in VENDOR_PACKAGES[name].items():
        saved[package_name] = sys.modules.pop(package_name, None)
        vendor = ModuleType(package_name)
        vendor.__path__ = [str(VENDOR / relative)]
        sys.modules[package_name] = vendor
    return saved


def _restore_vendor_packages(saved):
    for package_name, module in saved.items():
        if module is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = module


def _construct(name, cfg, device, priors):
    if name == 'SGSA':
        from SGSA import SGSA
        return SGSA(cfg['seq_len'], cfg['num_nodes'], cfg['input_dim'], cfg['pred_len'],
                    cfg['d_model'], cfg['dropout'], cfg['graph_rank'], cfg['rho'])
    if name == 'MultiGSA_adapted':
        from models.multigsa import MultiGSAAdapted
        return MultiGSAAdapted(cfg['seq_len'], cfg['num_nodes'], cfg['input_dim'],
                               cfg['pred_len'], cfg['d_model'], cfg['dropout'])
    if name in ['LSTM', 'Transformer']:
        from models.temporal import Temporal
        return Temporal(name, cfg)
    if name == 'MTGNN':
        from models.mtgnn import MTGNNAdapter
        return MTGNNAdapter(cfg, device=device)
    if name == 'FEDformer':
        from models.fedformer import FEDformerAdapter
        return FEDformerAdapter(cfg)
    if name in ['GraphWaveNet', 'AGCRN', 'STID_no_calendar', 'DLinear', 'PatchTST']:
        from models import extended
        cls = {'GraphWaveNet': extended.GraphWaveNet, 'AGCRN': extended.AGCRN,
               'STID_no_calendar': extended.STID, 'DLinear': extended.DLinear,
               'PatchTST': extended.PatchTST}[name]
        return cls(cfg)
    if name == 'PDFormer_train_graph':
        if priors is None:
            raise ValueError('PDFormer needs the training-only prior file '
                             '(run `python run.py prepare --dataset <name>`)')
        from models.extra import PDFormer
        return PDFormer(cfg, priors)
    if name == 'StemGNN_power_only':
        from models.extra import StemGNN
        return StemGNN(cfg)
    raise ValueError(f'Unknown model: {name}')


def build(name, cfg, device='cuda', priors=None):
    if name not in NAMES:
        raise ValueError(f'Unknown model: {name}')
    if name in VENDOR_PACKAGES:
        saved = _install_vendor_packages(name)
        try:
            return _construct(name, cfg, device, priors)
        finally:
            _restore_vendor_packages(saved)
    return _construct(name, cfg, device, priors)


__all__ = ['NAMES', 'build']
