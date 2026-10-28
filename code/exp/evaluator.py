"""Masked evaluation and the artifacts written next to every scored run."""
import contextlib
from pathlib import Path

import numpy as np
import torch

from data_provider.common import loader


def _fixed_stream(enabled):
    """Context manager that pins the evaluation RNG stream.

    MTGNN's official graph top-k breaks ties with random numbers even in eval
    mode, so every method is scored under the same fixed stream and the training
    RNG state is restored afterwards.
    """
    if not enabled:
        return contextlib.nullcontext()
    devices = [0] if torch.cuda.is_available() else []
    return torch.random.fork_rng(devices=devices)


@torch.no_grad()
def evaluate(model, dataset, cfg, micro=None, save=None, method='clean'):
    """Mean absolute error, RMSE and MSE over valid labels only.

    Errors are accumulated in float64 to reduce summation error. Model predictions
    can still vary with device, batching and nondeterministic kernels.
    """
    device = next(model.parameters()).device
    micro = int(micro or cfg['batch_size'])
    model.eval()
    absolute = squared = count = 0
    arrays = {k: [] for k in ['prediction', 'target', 'mask', 'starts']}

    stream = _fixed_stream(cfg.get('eval_fixed_rng', False))
    with stream:
        if cfg.get('eval_fixed_rng', False):
            torch.manual_seed(cfg['corruption_seed'])
            torch.cuda.manual_seed_all(cfg['corruption_seed'])
        for x, y, m, starts, _ in loader(dataset, shuffle=False, seed=0,
                                         batch_size=cfg['batch_size'], device=device):
            pred = torch.cat([model(x[i:i + micro].to(device)).cpu()
                              for i in range(0, len(x), micro)])
            if not torch.isfinite(pred).all():
                raise RuntimeError('Nonfinite predictions')
            error = (pred.double() - y.double())[m.bool()]
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
            if save is not None:
                for key, value in zip(arrays, [pred, y, m, starts]):
                    arrays[key].append(value.numpy())

    if not count:
        raise RuntimeError('Empty target mask')
    metrics = dict(mae=absolute / count, rmse=(squared / count) ** .5, mse=squared / count,
                   valid_points=int(count), windows=len(dataset))
    if save is None:
        return metrics

    stacked = {k: np.concatenate(v) for k, v in arrays.items()}
    np.savez_compressed(str(save) + '.npz', turbine_ids=dataset.data.turbine_ids, **stacked)
    write_artifacts(stacked, Path(save).parent, cfg, method)
    return metrics, stacked


def write_artifacts(arr, folder, cfg, method):
    """Horizon table, per-window table and one example forecast curve."""
    from utils.io import csvwrite
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prediction = arr['prediction'].astype('float64')
    target = arr['target'].astype('float64')
    mask = arr['mask'].astype(bool)
    error = prediction - target

    rows = []
    for h in range(cfg['pred_len']):
        e = error[:, h][mask[:, h]]
        rows.append(dict(horizon=h + 1, mae=float(np.abs(e).mean()),
                         rmse=float(np.sqrt(np.square(e).mean())), valid_points=len(e)))
    csvwrite(folder / f'{method}_horizon.csv', rows)

    rows = []
    for i, origin in enumerate(arr['starts']):
        e = error[i][mask[i]]
        rows.append(dict(start=int(origin), valid_points=len(e),
                         mae=float(np.abs(e).mean()) if len(e) else '',
                         mse=float(np.square(e).mean()) if len(e) else '',
                         sse=float(np.square(e).sum()), sae=float(np.abs(e).sum())))
    csvwrite(folder / f'{method}_per_window.csv', rows)

    from utils.plotting import plot_prediction_example
    plot_prediction_example(arr, folder, cfg['pred_len'])
