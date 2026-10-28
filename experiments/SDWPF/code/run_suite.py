"""Restartable 13-model CUDA benchmark for the SDWPF protocol.

Every protocol constant is read from ``experiments/SDWPF/config.json``; the
values that the paper reports as fixed are additionally asserted against the
frozen release values below, so a silent edit cannot change what was measured.
Each optimizer step covers 32 windows through gradient accumulation.

Actions
-------
preflight  : hash the data, dump the protocol, find a feasible microbatch and
             smoke-test every architecture (no training).
train      : train one (model, strategy, seed) run, then score it.
queue      : run the 13 models x 2 strategies x 3 seeds in sequence.
benchmark  : measure parameter count, inference latency and peak memory with
             the protocol reported in the paper (batch 32, 20 warm-up
             iterations, 100 timed repetitions, median per checkpoint).
report     : aggregate every finished run into CSV/markdown summaries.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_and_prepare, Windows, loader
from models import NAMES, build, parameter_counts, frequency_indices
from preprocessing import PROTOCOL_SPLIT
from sgsa_utils.checkpoints import load_checkpoint, save_checkpoint, validate_run_config

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs'
PAPER = ROOT / 'outputs'
DATA = ROOT / 'data' / 'data_20000.npz'
CONFIG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))

# Frozen release protocol (paper Tables 2 and 4). config.json must agree.
FROZEN = dict(seq_len=96, pred_len=48, num_nodes=134, input_dim=16, d_model=64,
              graph_rank=16, rho=0.25, dropout=0.1, batch_size=32, epochs=30,
              patience=5, learning_rate=0.0005, weight_decay=0.0001,
              gradient_clip=1.0, selection='validation MSE', metric_scale='training-only z-score',
              augmentation_seed=2024, validation_noise_seed=502024, corruption_seed=910024,
              seeds=[2024, 2025, 2026], train_stride=4, evaluation_stride=24,
              split={'train': [0, 14000], 'val': [14000, 15000],
                     'unused': [15000, 16000], 'test': [16000, 20000]})
for key, expected in FROZEN.items():
    if key not in CONFIG or CONFIG[key] != expected:
        raise SystemExit(f'config.json disagrees with the frozen protocol on {key!r}: '
                         f'{CONFIG.get(key)!r} != {expected!r}')
# The preprocessing helper and the benchmark must describe the same split.
if {k: list(v) for k, v in PROTOCOL_SPLIT.items()} != CONFIG['split']:
    raise SystemExit('preprocessing.PROTOCOL_SPLIT disagrees with config.json split')

EXPECTED_VALID_TEST_POINTS = 966746          # paper Table 2
EXPECTED_WINDOWS = dict(train=3465, val=40, test=165)  # paper Table 2

CFG = dict(
    epochs=CONFIG['epochs'], patience=CONFIG['patience'], batch_size=CONFIG['batch_size'],
    learning_rate=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'],
    gradient_clip=CONFIG.get('gradient_clip', 1.0), seeds=CONFIG['seeds'],
    strategies=['clean', 'noise'], models=NAMES,
    seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'], d_model=CONFIG['d_model'],
    num_nodes=CONFIG['num_nodes'], input_dim=CONFIG['input_dim'],
    graph_rank=CONFIG['graph_rank'], rho=CONFIG['rho'], dropout=CONFIG['dropout'],
    split=CONFIG['split'], train_stride=CONFIG['train_stride'],
    evaluation_stride=CONFIG['evaluation_stride'],
    loss='masked MSE', selection=CONFIG.get('selection', 'validation MSE'),
    augmentation_seed=CONFIG['augmentation_seed'],
    validation_noise_seed=CONFIG['validation_noise_seed'],
    corruption_seed=CONFIG['corruption_seed'],
    noise='50% clean / 30% 5% packet loss / 20% (20% packet loss + 5% replacement +/-8 z outliers)',
    test_corruption='20% packet loss + 5% replacement +/-8 z outliers, inputs only',
    scale=CONFIG.get('metric_scale', 'training-only z-score') + ' (MAE/RMSE are not physical units)',
    precision='float32/complex64, no AMP, no TF32',
    train_shuffle='fixed batch order per epoch (generator re-seeded with the run seed)',
    tuning=('Single frozen configuration. rho was compared at 0.5 and 0.25 on the '
            'development test panel before being fixed at 0.25; the rho=0.5 runs are '
            'archived and not reported. No further per-model test-driven tuning.'),
    warning=('Evaluation panel inspected during development: a development test panel, '
             'not an untouched confirmatory holdout.'))


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def csvwrite(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_setup():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_data():
    d = load_and_prepare(DATA, CFG['split'])
    assert d.x_scaled.shape == (20000, 134, 16) and d.feature_names[-1] == 'Patv'
    return d


def split_data(data, strategy):
    """Training/validation windows for the 3465/40 protocol split."""
    train = Windows.for_partition(data, 'train', seed=CFG['augmentation_seed'],
                                  augmentation=strategy == 'noise', stride=CFG['train_stride'],
                                  seq_len=CFG['seq_len'], pred_len=CFG['pred_len'])
    val = Windows.for_partition(data, 'val', seed=CFG['validation_noise_seed'],
                                augmentation=strategy == 'noise', stride=CFG['evaluation_stride'],
                                seq_len=CFG['seq_len'], pred_len=CFG['pred_len'])
    return train, val


def test_data(data, severity):
    """Fixed test windows; ``severity=2`` is the heavy corruption condition (NC)."""
    return Windows.for_partition(data, 'test', seed=CFG['corruption_seed'], augmentation=False,
                                 stride=CFG['evaluation_stride'], severity=severity,
                                 seq_len=CFG['seq_len'], pred_len=CFG['pred_len'])


def atomic_save(path, state):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['py'])
    np.random.set_state(s['np'])
    torch.set_rng_state(s['cpu'])
    torch.cuda.set_rng_state_all(s['cuda'])


@torch.no_grad()
def evaluate(model, ds, micro, save=None):
    model.eval()
    absolute = squared = count = 0.
    arrays = {k: [] for k in ['prediction', 'target', 'mask', 'starts']}
    # MTGNN's official graph top-k adds random tie-breaking noise even in eval.
    # A fixed evaluation stream makes every method see identical tie-breaking;
    # fork_rng restores the training RNG state afterwards.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(CFG['corruption_seed'])
        torch.cuda.manual_seed_all(CFG['corruption_seed'])
        for x, y, m, starts, _ in loader(ds):
            pred = torch.cat([model(x[i:i + micro].cuda()).cpu() for i in range(0, len(x), micro)])
            if not torch.isfinite(pred).all():
                raise RuntimeError('Nonfinite predictions')
            e = (pred.double() - y.double())[m.bool()]
            absolute += e.abs().sum().item()
            squared += e.square().sum().item()
            count += e.numel()
            if save:
                for k, v in zip(arrays, [pred, y, m, starts]):
                    arrays[k].append(v.numpy())
    if not count:
        raise RuntimeError('Empty target mask')
    stats = dict(mae=absolute / count, rmse=(squared / count) ** .5,
                 mse=squared / count, valid_points=int(count))
    if save:
        arr = {k: np.concatenate(v) for k, v in arrays.items()}
        np.savez_compressed(save, **arr, turbine_ids=ds.data.turbine_ids)
        return stats, arr
    return stats


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['epoch'] for r in history], [r['train_mse'] for r in history], label='Train')
    ax.plot([r['epoch'] for r in history], [r['val_mse'] for r in history], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Standardized masked MSE')
    ax.legend()
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(str(path) + '.' + ext, dpi=300)
    plt.close(fig)


def write_metrics(arr, folder):
    pred, y, m = (arr['prediction'].astype('float64'), arr['target'].astype('float64'),
                  arr['mask'].astype(bool))
    err = pred - y
    rows = []
    for h in range(CFG['pred_len']):
        e = err[:, h][m[:, h]]
        rows.append(dict(horizon=h + 1, mae=float(np.abs(e).mean()),
                         rmse=float(np.sqrt(np.square(e).mean())), valid_points=len(e)))
    csvwrite(folder / 'horizon_metrics.csv', rows)
    rows = []
    for i, s in enumerate(arr['starts']):
        e = err[i][m[i]]
        rows.append(dict(start=int(s), valid_points=len(e),
                         mae=float(np.abs(e).mean()) if len(e) else '',
                         sse=float(np.square(e).sum()), sae=float(np.abs(e).sum())))
    csvwrite(folder / 'per_window.csv', rows)
    # First valid window and first turbine with valid labels, deterministic selection.
    good = np.argwhere(m.any(axis=1))
    i, n = map(int, good[0])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.where(m[i, :, n], y[i, :, n], np.nan), label='Target')
    ax.plot(pred[i, :, n], label='Prediction')
    ax.set(xlabel='Horizon', ylabel='Standardized power',
           title=f'Forecast start={arr["starts"][i]}, turbine index={n}')
    ax.legend()
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(folder / f'prediction_example.{ext}', dpi=300)
    plt.close(fig)


def preflight():
    cuda_setup()
    PAPER.mkdir(parents=True, exist_ok=True)
    data = load_data()
    train, val = split_data(data, 'noise')
    test_starts = test_data(data, None).starts
    assert (len(train), len(val), len(test_starts)) == (
        EXPECTED_WINDOWS['train'], EXPECTED_WINDOWS['val'], EXPECTED_WINDOWS['test']), 'Window count drift'
    np.savez_compressed(PAPER / 'indices_and_scaling.npz', train_starts=train.starts,
                        val_starts=val.starts, test_starts=test_starts,
                        mean=data.feature_mean, std=data.feature_std,
                        median=data.feature_median, turbine_ids=data.turbine_ids,
                        feature_names=data.feature_names, timestamps=data.timestamps)
    dump(PAPER / 'protocol.json', dict(CFG, data_sha256=sha(DATA),
                                       gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    source = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*.py') if 'runs' not in p.parts}
    dump(PAPER / 'source_hashes.json', source)
    x, y, m, _, _ = next(iter(loader(train)))
    results = []
    for name in NAMES:
        passed = False
        for micro in [32, 16, 8, 4, 2, 1]:
            seed_all(2024)
            model = build(name, CFG).cuda()
            model.train()
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                pred = model(x[:micro].cuda())
                mask = m[:micro].cuda().bool()
                assert pred.shape == (micro, CFG['pred_len'], CFG['num_nodes'])
                loss = (pred - y[:micro].cuda())[mask].square().mean()
                loss.backward()
                torch.cuda.synchronize()
                assert torch.isfinite(loss)
                for p in model.parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all()
                backward_seconds = time.perf_counter() - start
                reachable = sum(p.numel() * (2 if p.is_complex() else 1)
                                for p in model.parameters() if p.grad is not None)
                unused = [n for n, p in model.named_parameters() if p.grad is None]
                model.zero_grad(set_to_none=True)
                model.eval()
                times = []
                with torch.inference_mode():
                    for j in range(8):
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        model(x[:micro].cuda())
                        torch.cuda.synchronize()
                        if j >= 3:
                            times.append(time.perf_counter() - t)
                row = dict(model=name, **parameter_counts(model),
                           autograd_reachable_real_scalars=reachable, micro_batch=micro,
                           one_forward_backward_seconds=backward_seconds,
                           exploratory_inference_seconds=float(np.median(times)),
                           smoke_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                           status='passed')
                # These probe timings are engineering smoke measurements, not the
                # paper's latency protocol; use the `benchmark` action for that.
                results.append(row)
                dump(PAPER / f'architecture_{name}.json',
                     dict(architecture=str(model), frequency_indices=frequency_indices(model),
                          parameters_without_gradient_in_probe=unused))
                print(row, flush=True)
                passed = True
            except torch.cuda.OutOfMemoryError:
                print(name, 'OOM at microbatch', micro, flush=True)
            finally:
                del model
                torch.cuda.empty_cache()
            if passed:
                break
        if not passed:
            raise RuntimeError('Cannot fit ' + name)
        csvwrite(PAPER / 'preflight.csv', results)
    dump(PAPER / 'preflight.json', results)


def train_one(name, strategy, seed, reshuffle_each_epoch=False):
    cuda_setup()
    seed_all(seed)
    micro = next(r['micro_batch'] for r in json.loads((PAPER / 'preflight.json').read_text())
                 if r['model'] == name)
    run = WORK / name / strategy / str(seed)
    out = PAPER / name / strategy / str(seed)
    run.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(CFG, model=name, strategy=strategy, seed=seed,
                    micro_batch=micro, reshuffle_each_epoch=reshuffle_each_epoch)
    validate_run_config(out / 'config.json', identity,
                        [run / 'last.pt', run / 'best.pt', out / 'complete.json'])
    if (out / 'complete.json').exists():
        return
    data = load_data()
    train, val = split_data(data, strategy)
    steps_per_epoch = math.ceil(len(train) / CFG['batch_size'])
    model = build(name, CFG).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'],
                            weight_decay=CFG['weight_decay'])
    initial_indices = frequency_indices(model)
    dump(out / 'config.json', dict(CFG, model=name, strategy=strategy, seed=seed,
                                   micro_batch=micro, reshuffle_each_epoch=reshuffle_each_epoch,
                                   parameters=parameter_counts(model),
                                   frequency_indices=initial_indices))
    history = []
    best = float('inf')
    stale = 0
    start_epoch = 1
    if (run / 'last.pt').exists():
        state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        assert state['frequency_indices'] == initial_indices
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        history = state['history']
        best = state['best']
        stale = state['stale']
        start_epoch = state['epoch'] + 1
        restore_rng(state['rng'])
    for epoch in range(start_epoch, CFG['epochs'] + 1):
        if stale >= CFG['patience']:
            break
        model.train()
        train.epoch = epoch
        total = count = 0.
        begin = time.perf_counter()
        # Batch order: fixed per epoch for the released runs; per-epoch reshuffling
        # is available but was not used for the reported numbers.
        order_seed = seed + epoch if reshuffle_each_epoch else seed
        for step, (x, y, m, _, _) in enumerate(loader(train, True, order_seed), 1):
            denom = m.sum().item()
            if not denom:
                continue
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(x), micro):
                target = y[i:i + micro].cuda()
                mask = m[i:i + micro].cuda().bool()
                pred = model(x[i:i + micro].cuda())
                err = (pred - target)[mask]
                loss = err.square().sum() / denom
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite train loss')
                loss.backward()
                total += err.detach().square().sum().item()
                count += err.numel()
            nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            if step == 1 or step % 10 == 0:
                dump(PAPER / 'current_training.json',
                     dict(model=name, strategy=strategy, seed=seed, epoch=epoch, step=step,
                          steps=steps_per_epoch, time=time.time()))
        vm = evaluate(model, val, micro)
        record = dict(epoch=epoch, train_mse=total / count, val_mse=vm['mse'],
                      val_mae=vm['mae'], val_rmse=vm['rmse'],
                      seconds=time.perf_counter() - begin)
        history.append(record)
        csvwrite(out / 'history.csv', history)
        print(name, strategy, seed, record, flush=True)
        if vm['mse'] < best:
            best = vm['mse']
            stale = 0
            save_checkpoint(model, run / 'best.pt')
        else:
            stale += 1
        atomic_save(run / 'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                         epoch=epoch, history=history, best=best, stale=stale,
                                         rng=rng_state(), frequency_indices=initial_indices))
        plot_loss(history, out / 'loss')
    load_checkpoint(model, run / 'best.pt')
    dump(out / 'checkpoint.json', dict(sha256=sha(run / 'best.pt'), best_val_mse=best,
                                       epochs_run=len(history),
                                       frequency_indices=initial_indices))
    for cond, severity in [('clean', None), ('corrupted', 2)]:
        folder = out / cond
        folder.mkdir(exist_ok=True)
        ds = test_data(data, severity)
        metrics, arr = evaluate(model, ds, micro, run / f'{cond}_predictions.npz')
        assert metrics['valid_points'] == EXPECTED_VALID_TEST_POINTS, 'Valid test point drift'
        dump(folder / 'metrics.json', dict(model=name, strategy=strategy, seed=seed,
                                           condition=cond, **metrics))
        write_metrics(arr, folder)
    if name == 'SGSA':
        adj = torch.softmax(torch.nn.functional.elu(model.node_left @ model.node_right), -1).detach().cpu().numpy()
        np.save(out / 'static_adjacency.npy', adj)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(adj, cmap='viridis')
        fig.colorbar(im, ax=ax)
        ax.set(xlabel='Source turbine', ylabel='Receiving turbine')
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'adjacency.{ext}', dpi=300)
        plt.close(fig)
    dump(out / 'complete.json', dict(status='complete', time=time.time()))


def benchmark(model_names=None, batch_size=32, warmup=20, repeats=100, seeds=None):
    """Paper Section 4.3 protocol: batch 32, 20 warm-up, 100 timed repetitions.

    Each clean-training checkpoint is timed separately and the per-checkpoint
    medians are averaged over the three seeds. Peak memory is PyTorch's peak
    allocated memory during the forward pass. Parameter counts are real-valued
    scalars (a complex parameter counts as two).
    """
    cuda_setup()
    data = load_data()
    ds = test_data(data, None)
    origin = int(ds.starts[0])
    sample = torch.from_numpy(data.x_scaled[origin - CFG['seq_len']:origin][None]).cuda()
    seeds = seeds or CFG['seeds']
    rows = []
    for name in model_names or NAMES:
        checkpoints = [WORK / name / 'clean' / str(s) / 'best.pt' for s in seeds]
        missing = [str(p) for p in checkpoints if not p.exists()]
        if missing:
            print(f'{name}: skipped, missing {missing}', flush=True)
            continue
        per_seed = []
        for seed, ckpt in zip(seeds, checkpoints):
            model = build(name, CFG).cuda()
            # Strict load through the shared loader: FEDformer's frequency index lists
            # are part of checkpoint identity and are restored from the hash-bound
            # sidecar instead of being re-sampled.
            load_checkpoint(model, ckpt)
            model.eval()
            counts = parameter_counts(model)
            x = sample.repeat(batch_size, 1, 1, 1)
            times = []
            with torch.inference_mode():
                for _ in range(warmup):
                    model(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(repeats):
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    model(x)
                    b.record()
                    torch.cuda.synchronize()
                    times.append(a.elapsed_time(b))
                peak = torch.cuda.max_memory_allocated() / 2 ** 20
            per_seed.append(dict(seed=seed, checkpoint_sha256=sha(ckpt),
                                 latency_ms=float(np.median(times)),
                                 peak_allocated_MiB=float(peak), **counts))
            del model, x
            torch.cuda.empty_cache()
        row = dict(model=name, batch_size=batch_size, warmup=warmup, repeats=repeats,
                   seeds=seeds,
                   parameters=int(np.mean([r['real_scalar_parameters'] for r in per_seed])),
                   latency_ms=float(np.mean([r['latency_ms'] for r in per_seed])),
                   latency_ms_per_seed=[r['latency_ms'] for r in per_seed],
                   peak_allocated_MiB=float(np.mean([r['peak_allocated_MiB'] for r in per_seed])),
                   peak_allocated_MiB_per_seed=[r['peak_allocated_MiB'] for r in per_seed],
                   gpu=torch.cuda.get_device_name())
        rows.append(row)
        dump(PAPER / 'efficiency' / f'{name}.json', dict(row, per_seed=per_seed))
        print(row, flush=True)
    if rows:
        csvwrite(PAPER / 'efficiency_summary.csv',
                 [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows])
        dump(PAPER / 'efficiency_summary.json', rows)


def _acquire_queue_lock(path):
    """Cross-platform exclusive lock; released by the OS when the process exits.

    ``fcntl`` is unavailable on Windows, so a one-byte ``msvcrt`` region lock is
    used there. Both variants are advisory locks released by the operating
    system, which means a crashed queue process cannot leave a stale lock.
    """
    Path(path).touch(exist_ok=True)
    handle = open(path, 'r+')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            handle.close()
        except OSError:
            pass
        raise SystemExit(f'Another queue process already holds {path}')
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f'pid={os.getpid()}\n')
        handle.flush()
    except OSError:
        pass
    return handle


def queue(reshuffle_each_epoch=False):
    ROOT.mkdir(exist_ok=True)
    lock = _acquire_queue_lock(ROOT / 'queue.lock')
    try:
        # Check the immutable source manifest before starting/restarting the queue.
        for rel, digest in json.loads((PAPER / 'source_hashes.json').read_text()).items():
            assert sha(ROOT / rel) == digest, 'Source changed after preflight: ' + rel
        errors = []
        for strategy in ['clean', 'noise']:
            for seed in CFG['seeds']:
                for name in NAMES:
                    out = PAPER / name / strategy / str(seed)
                    if (out / 'complete.json').exists():
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    dump(PAPER / 'queue_status.json',
                         dict(stage='running', model=name, strategy=strategy, seed=seed,
                              completed=len(list(PAPER.glob('*/*/*/complete.json'))),
                              total=len(NAMES) * len(CFG['seeds']) * 2, errors=errors))
                    log = out / 'train.log'
                    command = [sys.executable, '-u', __file__, 'train', '--model', name,
                               '--strategy', strategy, '--seed', str(seed)]
                    if reshuffle_each_epoch:
                        command.append('--reshuffle-each-epoch')
                    with log.open('a') as f:
                        rc = subprocess.call(command, stdout=f, stderr=subprocess.STDOUT)
                    if rc:
                        error = dict(model=name, strategy=strategy, seed=seed, exit_code=rc)
                        errors.append(error)
                        dump(out / 'failure.json', error)
                    report()
        dump(PAPER / 'queue_status.json',
             dict(stage='complete' if not errors else 'finished_with_errors',
                  completed=len(list(PAPER.glob('*/*/*/complete.json'))),
                  total=len(NAMES) * len(CFG['seeds']) * 2, errors=errors))
        report()
    finally:
        lock.close()


def report():
    import pandas as pd
    rows = [json.loads(p.read_text()) for p in PAPER.glob('*/*/*/*/metrics.json')]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(PAPER / 'all_metrics.csv', index=False)
    summary = df.groupby(['model', 'strategy', 'condition'])[['mae', 'rmse']].agg(['mean', 'std', 'count'])
    summary.to_csv(PAPER / 'summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (s, c) in zip(axes.flat, [(s, c) for s in ['clean', 'noise'] for c in ['clean', 'corrupted']]):
        v = df[(df.strategy == s) & (df.condition == c)].groupby('model').mae.agg(['mean', 'std', 'count'])
        if len(v):
            ax.bar(v.index, v['mean'], yerr=v['std'].fillna(0), capsize=3)
        ax.set(title=f'{s} training -> {c} test', ylabel='Standardized MAE')
        ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(PAPER / f'comparison.{ext}', dpi=300)
    plt.close(fig)
    complete = list(PAPER.glob('*/*/*/complete.json'))
    text = ('# SGSA unified 30-epoch benchmark\n\nCompleted trainings: ' + str(len(complete))
            + '/' + str(len(NAMES) * len(CFG['seeds']) * 2) + '.\n\n' + summary.to_string())
    text += ('\n\nMeans with fewer than 3 seeds are provisional. Single frozen configuration, '
             'no per-model test-driven tuning (rho was fixed after comparing 0.5 and 0.25 on the '
             'development panel; rho=0.5 runs are archived and not reported). The evaluation panel '
             'was inspected during development and is not an untouched confirmatory holdout. '
             'Metrics are on the training-only z-score scale, not physical units. '
             'MultiGSA_adapted is a documented structural adaptation, not an exact reproduction of '
             'the original paper. LSTM/Transformer/FEDformer share weights across turbines; '
             'MTGNN/SGSA/MultiGSA_adapted aggregate turbines. No additional calendar inputs. '
             'Complex FEDformer parameters count as two real scalars. No significance claim is made '
             'from three-seed means alone.\n')
    (PAPER / 'report.md').write_text(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=['preflight', 'train', 'queue', 'benchmark', 'report'])
    p.add_argument('--model', choices=NAMES)
    p.add_argument('--models', choices=NAMES, nargs='+')
    p.add_argument('--strategy', choices=['clean', 'noise'])
    p.add_argument('--seed', type=int)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--reshuffle-each-epoch', action='store_true',
                   help='reshuffle the training batch order every epoch (not used for the paper)')
    a = p.parse_args()
    if a.action == 'preflight':
        preflight()
    elif a.action == 'train':
        train_one(a.model, a.strategy, a.seed, a.reshuffle_each_epoch)
    elif a.action == 'queue':
        queue(a.reshuffle_each_epoch)
    elif a.action == 'benchmark':
        benchmark(a.models or ([a.model] if a.model else None), a.batch_size,
                  a.warmup, a.repeats, a.seeds)
    else:
        report()
