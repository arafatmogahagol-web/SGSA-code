"""Hill of Towie (21 turbines, 7 variables) preprocessing.

Two steps, in this order:

1. ``prepare`` reads the public 2020 archive and writes ``panel.npz``: native
   10-minute records get a conservative availability lag, are binned into
   right-closed 30-minute means and are split chronologically.
2. ``load`` reads ``panel.npz`` and attaches the protocol split.

Historical inputs are forward-filled and then imputed with the training median;
labels are never filled. All statistics come from the training range.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.common import (PreparedData, Windows, apply_outliers, apply_packet_loss,
                                 window_starts)

NAMES = 'Hill21'

SOURCE_COLUMNS = ['wtc_AcWindSp_mean', 'wtc_GenRpm_mean', 'wtc_MainSRpm_mean',
                  'wtc_PitcPosA_mean', 'wtc_NacelPos_mean']


def data_path(cfg, root):
    return Path(root) / cfg['data']


def prepare(cfg, root, archive):
    """Build ``panel.npz`` from the public archive (no training, no tuning)."""
    raw_cfg = cfg['raw']
    blocks = []
    with zipfile.ZipFile(archive) as z:
        for month in raw_cfg['months']:
            base = pd.read_csv(z.open(f'tblSCTurbine_2020_{month:02d}.csv'),
                               usecols=['TimeStamp', 'StationId'] + SOURCE_COLUMNS)
            power = pd.read_csv(z.open(f'tblSCTurGrid_2020_{month:02d}.csv'),
                                usecols=['TimeStamp', 'StationId', 'wtc_ActPower_mean'])
            if (base.duplicated(['TimeStamp', 'StationId']).any()
                    or power.duplicated(['TimeStamp', 'StationId']).any()):
                raise ValueError('Duplicate time/node records')
            blocks.append(base.merge(power, on=['TimeStamp', 'StationId'], how='outer',
                                     validate='one_to_one'))
    frame = pd.concat(blocks, ignore_index=True)
    frame['TimeStamp'] = pd.to_datetime(frame.TimeStamp)
    ids = np.sort(frame.StationId.unique())
    if len(ids) != cfg['num_nodes']:
        raise ValueError(f"Expected {cfg['num_nodes']} turbines, got {len(ids)}")

    # Source timestamps denote interval ends. The protocol adds a conservative
    # availability lag before assigning records to right-closed 30-minute bins.
    frame['available'] = frame.TimeStamp + pd.Timedelta(minutes=raw_cfg['availability_lag_minutes'])
    frame['bin'] = frame.available.dt.ceil('30min')
    frame['yaw_sin'] = np.sin(np.deg2rad(frame['wtc_NacelPos_mean']))
    frame['yaw_cos'] = np.cos(np.deg2rad(frame['wtc_NacelPos_mean']))
    features = SOURCE_COLUMNS[:4] + ['yaw_sin', 'yaw_cos', 'wtc_ActPower_mean']
    if features != cfg['features']:
        raise ValueError('Feature order differs from the configuration')

    grid = pd.date_range(frame.bin.min(), frame.bin.max(), freq='30min')
    raw = np.stack([frame.groupby(['bin', 'StationId'])[c].mean().unstack('StationId')
                    .reindex(index=grid, columns=ids).to_numpy() for c in features], -1
                   ).astype('float32')
    total = len(raw)
    b1 = int(raw_cfg['train_ratio'] * total)
    b2 = int((raw_cfg['train_ratio'] + raw_cfg['val_ratio']) * total)
    coverage = np.isfinite(raw[:b1, :, -1]).mean(0)
    if coverage.min() < 0.7:
        raise ValueError(f'Insufficient training label coverage: {coverage.tolist()}')
    mean = np.nanmean(raw[:b1], axis=(0, 1))
    std = np.maximum(np.nanstd(raw[:b1], axis=(0, 1)), 1e-5)
    median = np.nanmedian(raw[:b1], axis=(0, 1))
    if not np.isfinite(mean).all():
        raise ValueError('Invalid training statistics')

    x = raw.copy()
    for n in range(len(ids)):
        x[:, n, :] = pd.DataFrame(x[:, n, :]).ffill().to_numpy()
    x = ((np.where(np.isfinite(x), x, median) - mean) / std).astype('float32')
    y = (raw[:, :, -1] - mean[-1]) / std[-1]
    mask = np.isfinite(y)
    y = np.nan_to_num(y).astype('float32')

    starts = {}
    counts = {}
    for name, a, b, step in [('train', cfg['seq_len'], b1, cfg['train_stride']),
                             ('validation', b1, b2, cfg['evaluation_stride']),
                             ('test', b2, total, cfg['evaluation_stride'])]:
        values = np.array([t for t in range(a, b - cfg['pred_len'] + 1, step)
                           if mask[t:t + cfg['pred_len']].any()], dtype='int64')
        starts[name + '_starts'] = values
        counts[name] = len(values)
        if len(values) < 20:
            raise ValueError('Too few evaluation windows')

    Path(data_path(cfg, root)).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path(cfg, root), x=x, y=y, mask=mask, raw=raw, mean=mean, std=std, median=median,
        timestamps=grid.asi8, turbine_ids=ids, **starts)

    valid_test = int(sum(mask[t:t + cfg['pred_len']].sum() for t in starts['test_starts']))
    audit = dict(dataset=f"{cfg['dataset']} {raw_cfg['period']}",
                 source_doi=raw_cfg['source_doi'], nodes=len(ids), turbine_ids=ids.tolist(),
                 features=features,
                 power_target='wtc_ActPower_mean measured active power in kW; not PowerRef',
                 cadence_minutes=30, history_hours=cfg['seq_len'] // 2,
                 forecast_hours=cfg['pred_len'] // 2, boundaries=[0, b1, b2, total],
                 boundary_dates=[str(grid[b1]), str(grid[b2])], windows=counts,
                 valid_test_points=valid_test,
                 training_power_coverage=[float(c) for c in coverage],
                 target_mask='finite measured labels only; never forward-fill targets into loss',
                 inputs='causal forward fill then training median',
                 normalization='training-only mean/std over time and turbines',
                 availability=(f'source end timestamp plus protocol-specific '
                               f"{raw_cfg['availability_lag_minutes']}min lag, "
                               f'then right-closed 30min means'),
                 preprocessing_only=True)
    (Path(root) / cfg['audit']).write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


def load(cfg, root):
    with np.load(data_path(cfg, root)) as archive:
        arrays = {k: archive[k] for k in archive.files}
    split = {'train': (int(cfg['split']['train'][0]), int(cfg['split']['train'][1])),
             'val': (int(cfg['split']['val'][0]), int(cfg['split']['val'][1])),
             'test': (int(cfg['split']['test'][0]), int(cfg['split']['test'][1]))}
    data = PreparedData(
        name=NAMES, x=arrays['x'], y=arrays['y'], mask=arrays['mask'].astype(bool),
        finite_inputs=np.isfinite(arrays['raw']),
        feature_mean=arrays['mean'], feature_std=arrays['std'],
        feature_median=arrays['median'],
        target_mean=float(arrays['mean'][-1]), target_std=float(arrays['std'][-1]),
        feature_names=list(cfg['features']), target_name=cfg['features'][-1],
        timestamps=arrays['timestamps'], turbine_ids=arrays['turbine_ids'], split=split)
    # Only windows with at least one valid label are enumerated for this protocol.
    for partition in ('train', 'val', 'test'):
        data.window_starts[partition] = window_starts(data, partition, cfg, drop_empty=True)
    return data


class Hill21Windows(Windows):
    """Corruption stream of the Hill of Towie protocol.

    The stream is keyed by the forecast origin (not by the position in the window
    list) and adds a fixed offset for the explicitly corrupted evaluation set, so
    clean and corrupted passes use independent, reproducible patterns.
    """

    def _rng(self, index):
        origin = int(self.starts[index])
        offset = 701 if self.severity is not None else 0
        seed = (2024 + origin * 1009 + int(self.epoch) * 1000003 + offset) % 2 ** 32
        return np.random.default_rng(seed)

    def _pick_level(self, rng, index):
        if self.severity is not None:
            return int(self.severity)
        if not self.augmentation:
            return 0
        u = rng.random()
        return 0 if u < .5 else (1 if u < .8 else 2)

    def _corrupt(self, rng, x, origin, level):
        packets = apply_packet_loss(rng, x, self.data.median_std, level)
        if level == 2:
            apply_outliers(rng, x, packets)
