"""Dataset registry.

Each dataset lives in its own folder under ``datasets/`` and owns a ``config.json``
plus its data and prior files. This module maps a dataset name to the code that
knows how to load it, so the shared framework contains no dataset constants.
"""
import importlib
import json
from pathlib import Path

import numpy as np

DATASETS = ['SDWPF', 'Hill21']
DATASET_DIR = Path(__file__).resolve().parents[1] / 'datasets'
MODULES = {'SDWPF': 'sdwpf', 'Hill21': 'hill21'}
WINDOW_CLASSES = {'SDWPF': 'SDWPFWindows', 'Hill21': 'Hill21Windows'}


def dataset_root(name):
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {name!r}; choose from {DATASETS}')
    return DATASET_DIR / name


def load_config(name):
    return json.loads((dataset_root(name) / 'config.json').read_text(encoding='utf-8'))


def module(name):
    """The dataset module (``data_provider.sdwpf`` / ``data_provider.hill21``)."""
    return importlib.import_module('data_provider.' + MODULES[name])


def load(name, cfg=None):
    """Load a prepared dataset and its configuration."""
    cfg = cfg or load_config(name)
    return cfg, module(name).load(cfg, dataset_root(name))


def windows(name, data, partition, cfg, augmentation=False, severity=None, seed=None):
    """Windows of one partition, using that dataset's own corruption stream."""
    dataset_cls = getattr(module(name), WINDOW_CLASSES[name])
    return dataset_cls(data, data.window_starts[partition], cfg,
                       augmentation=augmentation, severity=severity, seed=seed)


def load_priors(name, cfg=None):
    """Training-only graph/DTW/pattern priors, or ``None`` when not prepared."""
    cfg = cfg or load_config(name)
    path = dataset_root(name) / cfg['priors']
    if not path.exists():
        return None
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


__all__ = ['DATASETS', 'dataset_root', 'load_config', 'load', 'windows', 'load_priors', 'module']
