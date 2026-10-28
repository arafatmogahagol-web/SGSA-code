"""Chronological SDWPF preprocessing with one explicit protocol split.

The half-open index ranges below are the split reported in the paper
(Table 2). They are defined once here, mirrored in
``experiments/SDWPF/config.json``, and asserted to agree by the benchmark, so
the helper functions and the paper protocol cannot drift apart.

Leakage rules implemented here:

* the variable-wise mean/std used for z-scoring and the variable-wise median
  used for imputation are computed from the training range only;
* the target array is never imputed - a missing label stays missing in
  ``y_mask`` and is excluded from both the loss and the metrics;
* feature arrays outside the training range only ever consume statistics
  estimated inside the training range.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Paper Table 2: training / validation / unused / test index ranges.
PROTOCOL_SPLIT = {'train': (0, 14000), 'val': (14000, 15000),
                  'unused': (15000, 16000), 'test': (16000, 20000)}


@dataclass
class PreparedData:
    values: np.ndarray
    x_scaled: np.ndarray
    y_scaled: np.ndarray
    y_mask: np.ndarray
    turbine_ids: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    target_mean: float
    target_std: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_median: np.ndarray
    split: dict


class ForecastWindows(Dataset):
    """Clean (uncorrupted) windows evaluated with stride 1 by default."""

    def __init__(self, data: PreparedData, starts, seq_len: int, pred_len: int):
        self.data = data
        self.starts = starts
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        forecast_start = self.starts[index]
        x0 = forecast_start - self.seq_len
        x1 = forecast_start
        y1 = forecast_start + self.pred_len
        return (
            torch.from_numpy(self.data.x_scaled[x0:x1]),
            torch.from_numpy(self.data.y_scaled[x1:y1]),
            torch.from_numpy(self.data.y_mask[x1:y1]),
            torch.tensor(forecast_start, dtype=torch.int64),
        )


def check_split(split, total_steps: int) -> dict:
    """Validate the protocol split and return it as ``{name: (start, end)}``."""
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


def load_and_prepare(path: str | Path, split=None) -> PreparedData:
    with np.load(path, allow_pickle=False) as archive:
        values = archive['values'].astype(np.float32)
        reference_values = (archive['normalization_values'].astype(np.float32)
                            if 'normalization_values' in archive else values)
        target_values = (archive['target_values'].astype(np.float32)
                         if 'target_values' in archive else reference_values)
        turbine_ids = archive['turbine_ids'].copy()
        timestamps = archive['timestamps'].copy()
        feature_names = archive['feature_names'].astype(str).tolist()
        target_name = str(archive['target_name'].item())

    if target_name not in feature_names:
        raise ValueError(f'Target {target_name!r} is not in feature_names')

    ranges = check_split(PROTOCOL_SPLIT if split is None else split, values.shape[0])
    train_start, train_end = ranges['train']
    train_values = reference_values[train_start:train_end]

    feature_median = np.nanmedian(train_values, axis=(0, 1)).astype(np.float32)
    feature_median = np.nan_to_num(feature_median, nan=0.0, posinf=0.0, neginf=0.0)
    filled = np.where(np.isfinite(values), values, feature_median[None, None, :])
    feature_mean = np.nanmean(train_values, axis=(0, 1)).astype(np.float32)
    feature_std = np.nanstd(train_values, axis=(0, 1)).astype(np.float32)
    feature_mean = np.nan_to_num(feature_mean, nan=0.0, posinf=0.0, neginf=0.0)
    feature_std = np.nan_to_num(feature_std, nan=1.0, posinf=1.0, neginf=1.0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    x_scaled = ((filled - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)

    target_index = feature_names.index(target_name)
    raw_target = target_values[:, :, target_index]
    y_mask = np.isfinite(raw_target)
    target_mean = float(feature_mean[target_index])
    target_std = float(feature_std[target_index])
    y_scaled = np.where(y_mask, (raw_target - target_mean) / target_std, 0.0).astype(np.float32)

    return PreparedData(
        # Retain clean reference values for physical-unit diagnostics even when inputs are corrupted.
        values=target_values,
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        y_mask=y_mask.astype(np.float32),
        turbine_ids=turbine_ids,
        timestamps=timestamps,
        feature_names=feature_names,
        target_mean=target_mean,
        target_std=target_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_median=feature_median,
        split=ranges,
    )


def make_splits(data: PreparedData, seq_len: int, pred_len: int,
                train_stride: int = 4, evaluation_stride: int = 24):
    """Clean windows for the three protocol partitions.

    The ranges and strides are the ones used for the reported results. The
    benchmark itself builds its own ``Windows`` objects from ``data.split``
    because it also needs the corruption settings; the two paths therefore
    share a single definition of the split.
    """
    windows = {}
    for name in ('train', 'val', 'test'):
        start, end = data.split[name]
        stride = train_stride if name == 'train' else evaluation_stride
        # A forecast origin must have a complete history, including at train start.
        starts = list(range(max(start, seq_len), end - pred_len + 1, stride))
        if not starts:
            raise ValueError(f'Not enough timestamps for partition {name!r}')
        windows[name] = ForecastWindows(data, starts, seq_len, pred_len)
    return windows['train'], windows['val'], windows['test']
