"""Figures produced by the benchmark (loss curves, bars, adjacency, example curve)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.' + ext), dpi=dpi)
    plt.close(fig)


def plot_loss(history, path, title=''):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [row['epoch'] for row in history]
    ax.plot(epochs, [row['train_mse'] for row in history], label='Train')
    ax.plot(epochs, [row['val_mse'] for row in history], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Standardized masked MSE', title=title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_prediction_example(arr, folder, pred_len):
    """One deterministic example: first window and turbine with a valid label."""
    mask = arr['mask'].astype(bool)
    good = np.argwhere(mask.any(axis=1))
    if not len(good):
        return
    i, n = map(int, good[0])
    target = np.where(mask[i, :, n], arr['target'][i, :, n], np.nan)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(1, pred_len + 1), target, label='Target')
    ax.plot(np.arange(1, pred_len + 1), arr['prediction'][i, :, n], label='Prediction')
    ax.set(xlabel='Horizon', ylabel='Standardized power',
           title=f"Forecast start={int(arr['starts'][i])}, turbine index={n}")
    ax.legend()
    fig.tight_layout()
    _save(fig, Path(folder) / 'prediction_example')


def plot_adjacency(adjacency, path, title='Static turbine graph'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(adjacency, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set(xlabel='Source turbine', ylabel='Receiving turbine', title=title)
    fig.tight_layout()
    _save(fig, path)


def plot_mae_bars(frame, path, title):
    """Grouped MAE bars with the sample standard deviation across seeds."""
    conditions = [c for c in ['clean', 'corrupted'] if c in set(frame.condition)]
    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 4), squeeze=False)
    for ax, condition in zip(axes[0], conditions):
        subset = frame[frame.condition == condition].groupby('model').mae.agg(['mean', 'std'])
        ax.bar(subset.index, subset['mean'], yerr=subset['std'].fillna(0), capsize=3)
        ax.set(title=f'{condition} test', ylabel='Standardized MAE')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def plot_parameter_vs_mae(frame, parameters, path):
    """Scatter of parameter count against mean MAE (one point per model)."""
    clean = frame[frame.condition == 'clean'].groupby('model').mae.mean()
    models = [m for m in clean.index if m in parameters]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([parameters[m] for m in models], [clean[m] for m in models])
    for m in models:
        ax.annotate(m, (parameters[m], clean[m]), fontsize=7,
                    xytext=(4, 3), textcoords='offset points')
    ax.set(xscale='log', xlabel='Parameters (real scalars)', ylabel='Standardized MAE (clean)')
    fig.tight_layout()
    _save(fig, path)
