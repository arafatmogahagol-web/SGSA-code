"""SDWPF (134 turbines, 16 variables) preprocessing for the processed cache.

The cache is a single ``data_20000.npz`` with a 30-minute time grid. Everything
that touches statistics is estimated on the training range only:

* the variable-wise mean/std used for z-scoring,
* the variable-wise median used to impute missing historical inputs,
* the target's mean/std used to standardize the labels.

Missing labels are never imputed: they stay ``False`` in the mask and are
excluded from both the loss and the metrics.
"""
from pathlib import Path

import numpy as np

from data_provider.common import PreparedData, Windows, apply_outliers, apply_packet_loss, window_starts

NAMES = 'SDWPF'


def check_split(split, total_steps):
    """Validate the protocol split and return ``{name: (start, end)}``."""
    if not isinstance(split, dict) or 'train' not in split or 'test' not in split:
        raise ValueError('split must define at least train and test ranges')
    ranges = {}
    for name, bounds in split.items():
        start, end = int(bounds[0]), int(bounds[1])
        if not 0 <= start < end <= total_steps:
            raise ValueError(f'Split {name!r} out of bounds: {(start, end)}')
        ranges[name] = (start, end)
    if ranges['train'][0] != 0:
        raise ValueError('The training range must start at index 0')
    ordered = [ranges[n] for n in ('train', 'val', 'unused', 'test') if n in ranges]
    for (_, end), (next_start, _) in zip(ordered, ordered[1:]):
        if end > next_start:
            raise ValueError('Split ranges must not overlap')
    return ranges


def data_path(cfg, root):
    return Path(root) / cfg['data']


def load(cfg, root):
    """Read the cache and return a :class:`PreparedData` instance."""
    with np.load(data_path(cfg, root), allow_pickle=False) as archive:
        values = archive['values'].astype(np.float32)
        reference_values = (archive['normalization_values'].astype(np.float32)
                            if 'normalization_values' in archive else values)
        target_values = (archive['target_values'].astype(np.float32)
                         if 'target_values' in archive else reference_values)
        turbine_ids = archive['turbine_ids'].copy()
        timestamps = archive['timestamps'].copy()
        feature_names = archive['feature_names'].astype(str).tolist()
        target_name = str(archive['target_name'].item())

    if ((values.shape[1], values.shape[2], len(feature_names))
            != (cfg['num_nodes'], cfg['input_dim'], cfg['input_dim'])):
        raise ValueError('Cache shape does not match the configured nodes/features')
    if target_name not in feature_names:
        raise ValueError(f'Target {target_name!r} is not in feature_names')

    split = check_split(cfg['split'], values.shape[0])
    train_start, train_end = split['train']
    train_values = reference_values[train_start:train_end]

    feature_median = np.nanmedian(train_values, axis=(0, 1)).astype(np.float32)
    feature_median = np.nan_to_num(feature_median, nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(values)
    filled = np.where(finite, values, feature_median[None, None, :])
    feature_mean = np.nan_to_num(np.nanmean(train_values, axis=(0, 1)).astype(np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)
    feature_std = np.nan_to_num(np.nanstd(train_values, axis=(0, 1)).astype(np.float32),
                                nan=1.0, posinf=1.0, neginf=1.0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    x_scaled = ((filled - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)

    target_index = feature_names.index(target_name)
    raw_target = target_values[:, :, target_index]
    mask = np.isfinite(raw_target)
    target_mean = float(feature_mean[target_index])
    target_std = float(feature_std[target_index])
    y_scaled = np.where(mask, (raw_target - target_mean) / target_std, 0.0).astype(np.float32)

    data = PreparedData(
        name=NAMES, x=x_scaled, y=y_scaled, mask=mask, finite_inputs=finite,
        feature_mean=feature_mean, feature_std=feature_std, feature_median=feature_median,
        target_mean=target_mean, target_std=target_std,
        feature_names=feature_names, target_name=target_name,
        timestamps=timestamps, turbine_ids=turbine_ids, split=split)
    # The protocol keeps every candidate window; windows without a single valid
    # label cannot contribute to a masked metric but are still enumerated.
    for partition in ('train', 'val', 'test'):
        data.window_starts[partition] = window_starts(data, partition, cfg, drop_empty=False)
    return data


class SDWPFWindows(Windows):
    """Corruption stream of the SDWPF protocol.

    Deterministic per ``(seed, epoch, index)`` and identical for every method
    trained with the same strategy and seed.
    """

    def _rng(self, index):
        return np.random.default_rng(int(self.seed) + 1000003 * int(self.epoch) + 104729 * int(index))

    def _corrupt(self, rng, x, origin, level):
        packets = apply_packet_loss(rng, x, self.data.median_std, level)
        if level == 2:
            finite = self.data.finite_inputs[origin - self.seq_len:origin]
            apply_outliers(rng, x, packets, allowed=finite)
