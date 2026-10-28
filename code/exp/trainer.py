"""One training run: build, train, select on the validation split, score the test split.

Behaviour that is fixed by the paper's protocol and therefore hard-coded here:

* the loss is the masked MSE, i.e. squared error summed over valid labels divided
  by the number of valid labels in the effective batch (no padding target ever
  enters the loss);
* one optimizer step always covers ``cfg['batch_size']`` windows, split into
  microbatches small enough for the device (gradient accumulation);
* gradient clipping, early stopping and checkpoint selection use the metric named
  by ``cfg['selection_key']`` (SDWPF: validation MSE, Hill21: validation MAE);
* the best checkpoint is chosen on the validation partition only.
"""
import math
import time
from pathlib import Path

import torch
from torch import nn

from data_provider import load_priors, windows
from data_provider.common import loader
from exp.evaluator import evaluate
from models import build
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.io import atomic_save, csvwrite, dump, sha256
from utils.params import parameter_counts
from utils.plotting import plot_adjacency, plot_loss
from utils.seeding import restore_rng, rng_state, seed_all


def _now():
    return time.time()


def train_run(dataset, name, strategy, seed, cfg, data, dirs, device,
              micro=None, reshuffle=False, priors=None):
    """Train (or resume) one ``(dataset, model, strategy, seed)`` run and score it."""
    if not cfg.get('training_authorized', False):
        raise RuntimeError('Configuration is a draft: training is not authorized')
    run_dir = Path(dirs['checkpoints']) / name / strategy / str(seed)
    out_dir = Path(dirs['results']) / name / strategy / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'complete.json').exists():
        return {'status': 'already_complete', 'run': str(out_dir)}

    seed_all(seed)
    data_cfg = dict(cfg, dataset=dataset)
    model = build(name, data_cfg, device=str(device), priors=priors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'])
    micro = int(micro or cfg.get('micro_batch') or cfg['batch_size'])
    train_windows = windows(dataset, data, 'train', cfg,
                            augmentation=strategy == 'noise', seed=cfg['augmentation_seed'])
    val_windows = windows(dataset, data, 'val', cfg,
                          augmentation=strategy == 'noise', seed=cfg['validation_noise_seed'])
    steps = math.ceil(len(train_windows) / cfg['batch_size'])

    dump(out_dir / 'config.json', dict(cfg, dataset=dataset, model=name, strategy=strategy,
                                       seed=seed, micro_batch=micro,
                                       shuffle_each_epoch=reshuffle,
                                       parameters=parameter_counts(model)))
    history, best, stale, start_epoch = [], float('inf'), 0, 1
    if (run_dir / 'last.pt').exists():
        state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        history, best, stale = state['history'], state['best'], state['stale']
        start_epoch = state['epoch'] + 1
        restore_rng(state['rng'])

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        if stale >= cfg['patience']:
            break
        model.train()
        train_windows.epoch = epoch
        total = labels = 0.
        begin = _now()
        order_seed = seed + epoch if reshuffle else seed
        for step, (x, y, m, _, _) in enumerate(
                loader(train_windows, shuffle=True, seed=order_seed,
                       batch_size=cfg['batch_size'], device=device), 1):
            denominator = m.sum().item()
            if not denominator:
                continue
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(x), micro):
                target = y[i:i + micro].to(device)
                mask = m[i:i + micro].to(device).bool()
                prediction = model(x[i:i + micro].to(device))
                error = (prediction - target)[mask]
                # Divide by the valid-label count of the whole effective batch, so
                # accumulating over microbatches yields the true batch mean.
                loss = error.square().sum() / denominator
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                total += error.detach().square().sum().item()
                labels += error.numel()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                     error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 10 == 0:
                dump(Path(dirs['results']) / 'current_training.json',
                     dict(dataset=dataset, model=name, strategy=strategy, seed=seed,
                          epoch=epoch, step=step, steps=steps, time=_now()))

        validation = evaluate(model, val_windows, cfg, micro=micro)
        record = dict(epoch=epoch, train_mse=total / labels, val_mse=validation['mse'],
                      val_mae=validation['mae'], val_rmse=validation['rmse'],
                      seconds=_now() - begin)
        history.append(record)
        csvwrite(out_dir / 'history.csv', history)
        print(f'{dataset} {name} {strategy} {seed} {record}', flush=True)

        score = record[cfg['selection_key']]
        if score < best:
            best, stale = score, 0
            save_checkpoint(model, run_dir / 'best.pt')
        else:
            stale += 1
        atomic_save(run_dir / 'last.pt',
                    dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=epoch, history=history, best=best, stale=stale,
                         rng=rng_state()))
        plot_loss(history, out_dir / 'loss', title=f'{name} {strategy} seed {seed}')

    load_checkpoint(model, run_dir / 'best.pt')
    dump(out_dir / 'checkpoint.json', dict(sha256=sha256(run_dir / 'best.pt'),
                                           best_validation=best,
                                           selection=cfg['selection_key'],
                                           epochs_run=len(history),
                                           micro_batch=micro))
    expected = cfg['expected_valid_test_points']
    for method, severity in [('clean', None), ('corrupted', 2)]:
        folder = out_dir / method
        folder.mkdir(parents=True, exist_ok=True)
        dataset_eval = windows(dataset, data, 'test', cfg, severity=severity,
                               seed=cfg['corruption_seed'])
        metrics, _ = evaluate(model, dataset_eval, cfg, micro=micro, save=folder / method,
                              method=method)
        if metrics['valid_points'] != expected:
            raise RuntimeError(f'Valid test points drifted: {metrics["valid_points"]} != {expected}')
        dump(folder / 'metrics.json', dict(dataset=dataset, model=name, strategy=strategy,
                                           seed=seed, condition=method, **metrics))

    if name == 'SGSA':
        adjacency = model.static_graph().detach().cpu().numpy()
        import numpy as np
        np.save(out_dir / 'static_adjacency.npy', adjacency)
        plot_adjacency(adjacency, out_dir / 'adjacency')

    dump(out_dir / 'complete.json', dict(status='complete', time=_now()))
    return {'status': 'complete', 'run': str(out_dir), 'epochs': len(history),
            'best_validation': best}
