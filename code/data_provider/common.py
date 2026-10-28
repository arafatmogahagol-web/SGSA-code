"""Dataset-independent pieces: the prepared-data container, window slicing,
masked-metric bookkeeping and the shared corruption levels.

Corruption levels (paper Section 4.1 / 5.1):

* 0 - clean
* 1 - light: 5% of the [time, turbine] packets replaced by the training median
* 2 - heavy: 20% packet loss plus 5% of the remaining entries replaced by +-8
  standard deviations

Only historical inputs are perturbed. Targets are never modified, and a missing
target stays missing in ``mask`` whatever the corruption level is. The random
stream that turns a window index into a corruption pattern is dataset specific
and lives in the two dataset modules.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LEVEL_SHARE = (0.5, 0.3, 0.2)
LIGHT_LOSS, HEAVY_LOSS = 0.05, 0.2
OUTLIER_SHARE, OUTLIER_MAGNITUDE = 0.05, 8.0


@dataclass
class PreparedData:
    """One dataset after preprocessing, with the protocol split attached."""

    name: str
    x: np.ndarray                 # [T, N, F] standardized inputs, medians filled
    y: np.ndarray                 # [T, N] standardized target (0 where masked)
    mask: np.ndarray              # [T, N] valid label mask
    finite_inputs: np.ndarray     # [T, N, F] finiteness before imputation
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_median: np.ndarray
    target_mean: float
    target_std: float
    feature_names: list
    target_name: str
    timestamps: np.ndarray
    turbine_ids: np.ndarray
    split: dict
    window_starts: dict = field(default_factory=dict)

    @property
    def median_std(self):
        """Training median expressed on the standardized scale."""
        return ((self.feature_median - self.feature_mean) / self.feature_std).astype('float32')

    def __len__(self):
        return self.x.shape[0]


def window_starts(data, partition, cfg, drop_empty=False):
    """First forecast origins of one partition, with the protocol strides.

    A window needs a full history, so an origin is never earlier than
    ``seq_len``. ``drop_empty`` removes windows whose target slice has no valid
    label at all (used where the protocol counts only usable windows).
    """
    start, end = data.split[partition]
    stride = cfg['train_stride'] if partition == 'train' else cfg['evaluation_stride']
    first = max(start, cfg['seq_len'])
    starts = np.arange(first, end - cfg['pred_len'] + 1, stride, dtype='int64')
    if drop_empty:
        starts = np.array([t for t in starts
                           if data.mask[t:t + cfg['pred_len']].any()], dtype='int64')
    return starts


class Windows(Dataset):
    """Sliding windows of one partition, optionally corrupted on the input side."""

    def __init__(self, data, starts, cfg, augmentation=False, severity=None, seed=None):
        self.data = data
        self.starts = np.asarray(starts, dtype='int64')
        self.cfg = cfg
        self.seq_len, self.pred_len = cfg['seq_len'], cfg['pred_len']
        self.augmentation, self.severity = augmentation, severity
        self.seed = cfg['augmentation_seed'] if seed is None else seed
        self.epoch = 0

    def __len__(self):
        return len(self.starts)

    # ---- dataset hooks -------------------------------------------------
    def _rng(self, index):
        raise NotImplementedError

    def _pick_level(self, rng, index):
        if self.severity is not None:
            return int(self.severity)
        if not self.augmentation:
            return 0
        return int(rng.choice(3, p=list(LEVEL_SHARE)))

    def _corrupt(self, rng, x, origin, level):
        raise NotImplementedError

    # ---- shared behaviour ---------------------------------------------
    def __getitem__(self, index):
        origin = int(self.starts[index])
        x = self.data.x[origin - self.seq_len:origin].copy()
        rng = self._rng(index)
        level = self._pick_level(rng, index)
        if level:
            self._corrupt(rng, x, origin, level)
        return (torch.from_numpy(x),
                torch.from_numpy(self.data.y[origin:origin + self.pred_len]),
                torch.from_numpy(self.data.mask[origin:origin + self.pred_len]),
                origin, level)


def apply_packet_loss(rng, x, median_std, level):
    """Replace whole [time, turbine] packets by the training median."""
    packets = rng.random(x.shape[:2]) < (LIGHT_LOSS if level == 1 else HEAVY_LOSS)
    x[packets] = median_std
    return packets


def apply_outliers(rng, x, packets, allowed=None):
    """Replace 5% of the remaining entries by +-8 standard deviations."""
    outliers = (rng.random(x.shape) < OUTLIER_SHARE) & ~packets[..., None]
    if allowed is not None:
        outliers &= allowed
    x[outliers] = rng.choice([-OUTLIER_MAGNITUDE, OUTLIER_MAGNITUDE], size=int(outliers.sum()))
    return outliers


def loader(dataset, shuffle=False, seed=0, batch_size=32, device=None):
    """Batch loader with an explicit, reproducible order.

    Passing the same ``seed`` on every epoch reproduces the same batch order on
    every epoch; the released SDWPF runs did exactly that (the per-run
    ``protocol.json`` records it). Pass ``seed + epoch`` to reshuffle.
    """
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device is not None and str(device).startswith('cuda'),
                      generator=generator)


def priors_path(cfg, root):
    return Path(root) / cfg['priors']
