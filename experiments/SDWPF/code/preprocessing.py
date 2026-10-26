from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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
    train_end: int
    val_end: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_median: np.ndarray


class ForecastWindows(Dataset):
    def __init__(self, data: PreparedData, starts: list[int], seq_len: int, pred_len: int):
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


def load_and_prepare(path: str | Path, train_ratio: float = 0.70, val_ratio: float = 0.15) -> PreparedData:
    with np.load(path, allow_pickle=False) as archive:
        values = archive["values"].astype(np.float32)
        reference_values = (archive["normalization_values"].astype(np.float32)
                           if "normalization_values" in archive else values)
        target_values = (archive["target_values"].astype(np.float32)
                         if "target_values" in archive else reference_values)
        turbine_ids = archive["turbine_ids"].copy()
        timestamps = archive["timestamps"].copy()
        feature_names = archive["feature_names"].astype(str).tolist()
        target_name = str(archive["target_name"].item())

    if target_name not in feature_names:
        raise ValueError(f"Target {target_name!r} is not in feature_names")
    t_steps = values.shape[0]
    train_end = int(t_steps * train_ratio)
    val_end = int(t_steps * (train_ratio + val_ratio))
    if train_end <= 0 or val_end >= t_steps:
        raise ValueError("Invalid chronological split")

    train_values = reference_values[:train_end]
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
        train_end=train_end,
        val_end=val_end,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_median=feature_median,
    )


def make_splits(data: PreparedData, seq_len: int, pred_len: int):
    train_starts = list(range(seq_len, data.train_end - pred_len + 1))
    val_starts = list(range(data.train_end, data.val_end - pred_len + 1))
    test_starts = list(range(data.val_end, len(data.timestamps) - pred_len + 1))
    if min(map(len, (train_starts, val_starts, test_starts))) == 0:
        raise ValueError("Not enough timestamps for the selected sequence and horizon lengths")
    return (
        ForecastWindows(data, train_starts, seq_len, pred_len),
        ForecastWindows(data, val_starts, seq_len, pred_len),
        ForecastWindows(data, test_starts, seq_len, pred_len),
    )


