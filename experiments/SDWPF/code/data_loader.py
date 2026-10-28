"""Historical chronological windows and deterministic input-only corruption.

Two independent random streams matter here and both are explicit:

* the *batch order* is controlled by the ``torch.Generator`` passed to the
  ``DataLoader`` (see ``loader``);
* the *corruption pattern* of a training window is a pure function of
  ``(base_seed, epoch, window_index)``, so the same window receives the same
  corruption on the same epoch and every method sees an identical corruption
  schedule for a given strategy and seed.

Corruption is applied to historical inputs only. Targets and the valid-label
mask are returned untouched, and missing entries are replaced by the
training-segment median expressed in standardized units.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_and_prepare  # noqa: F401  (re-exported for callers)

# Corruption levels: 0 = clean, 1 = light packet loss, 2 = heavy (paper Sec. 4.1).
LEVEL_SHARE = (0.5, 0.3, 0.2)
LIGHT_LOSS, HEAVY_LOSS = 0.05, 0.2
OUTLIER_SHARE, OUTLIER_MAGNITUDE = 0.05, 8.0


class Windows(Dataset):
    """Sliding windows over one chronological partition of the processed cache."""

    def __init__(self, data, start, end, seed=2024, augmentation=False, stride=24,
                 severity=None, seq_len=96, pred_len=48):
        if end > len(data.timestamps):
            raise ValueError('History/target crosses data bounds')
        if end - start < pred_len:
            raise ValueError('Partition is shorter than the prediction horizon')
        self.data = data
        self.seq_len, self.pred_len = seq_len, pred_len
        # The first forecast origin of a partition needs a full history window;
        # for the training partition (which starts at index 0) that is seq_len.
        first = max(start, seq_len)
        self.starts = list(range(first, end - pred_len + 1, stride))
        self.seed, self.augmentation, self.severity, self.epoch = seed, augmentation, severity, 0
        self.median = ((data.feature_median - data.feature_mean) / data.feature_std).astype('float32')

    @classmethod
    def for_partition(cls, data, partition, **kwargs):
        """Build windows from the protocol split recorded in ``PreparedData``."""
        start, end = data.split[partition]
        return cls(data, start, end, **kwargs)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        s = self.starts[index]
        x = self.data.x_scaled[s - self.seq_len:s].copy()
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + 104729 * index)
        level = (self.severity if self.severity is not None
                 else int(rng.choice(3, p=list(LEVEL_SHARE))) if self.augmentation else 0)
        if level:
            packets = rng.random(x.shape[:2]) < (LIGHT_LOSS if level == 1 else HEAVY_LOSS)
            x[packets] = self.median
            if level == 2:
                finite = np.isfinite(self.data.values[s - self.seq_len:s])
                outliers = (rng.random(x.shape) < OUTLIER_SHARE) & finite & ~packets[..., None]
                x[outliers] = rng.choice([-OUTLIER_MAGNITUDE, OUTLIER_MAGNITUDE],
                                         size=int(outliers.sum()))
        return (torch.from_numpy(x),
                torch.from_numpy(self.data.y_scaled[s:s + self.pred_len]),
                torch.from_numpy(self.data.y_mask[s:s + self.pred_len]),
                s, level)


def loader(ds, shuffle=False, seed=2024):
    """Batch loader with an explicit, reproducible order.

    Passing the same ``seed`` on every epoch reproduces the same batch order on
    every epoch. The released SDWPF runs did exactly that (the archived
    protocol records ``shuffle='fixed order per epoch'``); pass ``seed + epoch``
    to reshuffle each epoch instead.
    """
    return DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=0,
                      pin_memory=True, generator=torch.Generator().manual_seed(seed))
