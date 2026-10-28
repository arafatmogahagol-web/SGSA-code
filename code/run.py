"""Single entry point for both wind-farm protocols.

    python run.py prepare   --dataset SDWPF
    python run.py preflight --dataset SDWPF
    python run.py train     --dataset SDWPF --model SGSA --strategy clean --seed 2024
    python run.py queue     --dataset SDWPF
    python run.py evaluate  --dataset Hill21 --model SGSA --checkpoint ckpt.pt --device cpu
    python run.py benchmark --dataset SDWPF --models SGSA MultiGSA_adapted
    python run.py report    --dataset SDWPF

Every protocol constant is read from ``datasets/<name>/config.json``; nothing in
``models/``, ``data_provider/`` or ``exp/`` hard-codes a dataset shape.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import numpy as np
import torch

from data_provider import DATASETS, dataset_root, load, load_config, load_priors, module, windows
from data_provider import priors as priors_builder
from data_provider.common import loader
from exp import benchmark as benchmark_module
from exp import evaluator, reporter, trainer
from models import NAMES, build
from utils.checkpoints import load_checkpoint
from utils.io import dump, sha256
from utils.params import autograd_reachable_real_scalars, parameter_counts


def dirs_for(dataset):
    return dict(code=CODE,
                checkpoints=CODE / 'checkpoints' / dataset,
                results=CODE / 'results' / dataset,
                dataset=dataset_root(dataset))


def check_protocol(dataset, cfg, data):
    """Fail loudly if the prepared data disagrees with the configured protocol."""
    for partition in ('train', 'val', 'test'):
        expected = cfg['expected_windows'][partition]
        actual = len(data.window_starts[partition])
        if actual != expected:
            raise SystemExit(f'{dataset}: {partition} window count {actual} != {expected}')
    test = windows(dataset, data, 'test', cfg, severity=None, seed=cfg['corruption_seed'])
    valid = int(sum(int(data.mask[origin:origin + cfg['pred_len']].sum()) for origin in test.starts))
    if valid != cfg['expected_valid_test_points']:
        raise SystemExit(f'{dataset}: valid test points {valid} != {cfg["expected_valid_test_points"]}')
    if (data.x.shape[1], data.x.shape[2]) != (cfg['num_nodes'], cfg['input_dim']):
        raise SystemExit(f'{dataset}: data shape does not match num_nodes/input_dim')
    return valid


def source_hashes():
    return {str(path.relative_to(CODE)): sha256(path)
            for path in CODE.rglob('*.py') if 'checkpoints' not in path.parts
            and 'results' not in path.parts and '__pycache__' not in path.parts}


# --------------------------------------------------------------------------- prepare
def prepare(args):
    cfg = load_config(args.dataset)
    root = dataset_root(args.dataset)
    if args.dataset == 'Hill21':
        panel = root / cfg['data']
        if args.zip:
            module('Hill21').prepare(cfg, root, args.zip)
        elif not panel.exists():
            raise SystemExit(f'{panel} is missing; pass --zip /path/to/2020.zip')
    data = module(args.dataset).load(cfg, root)
    if args.dataset == 'SDWPF':
        pass
    valid = check_protocol(args.dataset, cfg, data)
    audit = priors_builder.build(cfg, data, root)
    print(json.dumps(dict(dataset=args.dataset, valid_test_points=valid,
                          windows={k: len(v) for k, v in data.window_starts.items()},
                          priors=audit['prior_sha256']), indent=2))


# ------------------------------------------------------------------------- preflight
def preflight(args):
    cfg = load_config(args.dataset)
    dirs = dirs_for(args.dataset)
    cfg, data = load(args.dataset)
    results = Path(dirs['results'])
    results.mkdir(parents=True, exist_ok=True)
    valid = check_protocol(args.dataset, cfg, data)
    dump(results / 'protocol.json', dict(cfg, dataset=args.dataset,
                                         data_sha256=sha256(Path(dirs['dataset']) / cfg['data']),
                                         valid_test_points=valid,
                                         gpu=(torch.cuda.get_device_name()
                                              if torch.cuda.is_available() else 'cpu'),
                                         torch=torch.__version__))
    dump(results / 'source_hashes.json', source_hashes())
    print(f'{args.dataset}: windows and valid-label counts match the protocol', flush=True)
    if not torch.cuda.is_available():
        print('CUDA unavailable: skipping the microbatch probe', flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = windows(args.dataset, data, 'train', cfg, augmentation=True, seed=cfg['augmentation_seed'])
    x, y, m, _, _ = next(iter(loader(train, shuffle=False, batch_size=cfg['batch_size'],
                                     device='cuda')))
    priors = load_priors(args.dataset)
    probe = []
    for name in NAMES:
        passed = False
        for micro in [cfg['batch_size'], 16, 8, 4, 2, 1]:
            try:
                model = build(name, dict(cfg, dataset=args.dataset), priors=priors).cuda()
                model.train()
                torch.cuda.reset_peak_memory_stats()
                prediction = model(x[:micro].cuda())
                mask = m[:micro].cuda().bool()
                assert prediction.shape == (micro, cfg['pred_len'], cfg['num_nodes'])
                loss = (prediction - y[:micro].cuda())[mask].square().mean()
                loss.backward()
                assert torch.isfinite(loss)
                row = dict(model=name, micro_batch=micro,
                           autograd_reachable_real_scalars=autograd_reachable_real_scalars(model),
                           parameters=parameter_counts(model),
                           smoke_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                           status='passed')
                probe.append(row)
                print(row, flush=True)
                passed = True
            except torch.cuda.OutOfMemoryError:
                print(name, 'OOM at microbatch', micro, flush=True)
            finally:
                torch.cuda.empty_cache()
            if passed:
                break
        if not passed:
            raise RuntimeError('Cannot fit ' + name)
    dump(results / 'preflight.json', probe)


# ----------------------------------------------------------------------------- train
def train(args):
    cfg, data = load(args.dataset)
    dirs = dirs_for(args.dataset)
    micro = args.micro
    if micro is None:
        probe = Path(dirs['results']) / 'preflight.json'
        if probe.exists():
            records = json.loads(probe.read_text(encoding='utf-8'))
            micro = next(row['micro_batch'] for row in records if row['model'] == args.model)
    result = trainer.train_run(args.dataset, args.model, args.strategy, args.seed, cfg, data,
                               dirs, _device(args.device), micro=micro,
                               reshuffle=args.reshuffle_each_epoch,
                               priors=load_priors(args.dataset, cfg))
    print(json.dumps(result, indent=2))


# ----------------------------------------------------------------------------- queue
def _acquire_lock(path):
    """Exclusive, OS-released lock: ``msvcrt`` on Windows, ``flock`` elsewhere."""
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


def queue(args):
    cfg = load_config(args.dataset)
    dirs = dirs_for(args.dataset)
    results = Path(dirs['results'])
    results.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(results / 'queue.lock')
    manifest = results / 'source_hashes.json'
    try:
        if manifest.exists():
            for relative, digest in json.loads(manifest.read_text(encoding='utf-8')).items():
                if sha256(CODE / relative) != digest:
                    raise SystemExit('Source changed after preflight: ' + relative)
        errors = []
        for strategy in cfg['strategies']:
            for seed in cfg['seeds']:
                for name in NAMES:
                    out = results / name / strategy / str(seed)
                    if (out / 'complete.json').exists():
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    dump(results / 'queue_status.json',
                         dict(dataset=args.dataset, stage='running', model=name,
                              strategy=strategy, seed=seed,
                              completed=len(list(results.glob('*/*/*/complete.json'))),
                              total=len(NAMES) * len(cfg['strategies']) * len(cfg['seeds']),
                              errors=errors))
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'train',
                               '--dataset', args.dataset, '--model', name,
                               '--strategy', strategy, '--seed', str(seed),
                               '--device', args.device]
                    if args.reshuffle_each_epoch:
                        command.append('--reshuffle-each-epoch')
                    with (out / 'train.log').open('a') as log:
                        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
                    if code:
                        error = dict(model=name, strategy=strategy, seed=seed, exit_code=code)
                        errors.append(error)
                        dump(out / 'failure.json', error)
                    reporter.report(args.dataset, cfg, dirs)
        dump(results / 'queue_status.json',
             dict(dataset=args.dataset,
                  stage='complete' if not errors else 'finished_with_errors',
                  completed=len(list(results.glob('*/*/*/complete.json'))),
                  total=len(NAMES) * len(cfg['strategies']) * len(cfg['seeds']),
                  errors=errors))
        reporter.report(args.dataset, cfg, dirs)
    finally:
        lock.close()


# -------------------------------------------------------------------------- evaluate
def evaluate(args):
    cfg, data = load(args.dataset)
    device = _device(args.device)
    model = build(args.model, dict(cfg, dataset=args.dataset), device=str(device),
                  priors=load_priors(args.dataset, cfg)).to(device)
    load_checkpoint(model, args.checkpoint)
    severity = None if args.condition == 'clean' else 2
    dataset = windows(args.dataset, data, 'test', cfg, severity=severity,
                      seed=cfg['corruption_seed'])
    out = Path(args.output or (CODE / 'results' / args.dataset / 'evaluate'
                               / f'{args.model}_{args.condition}'))
    out.mkdir(parents=True, exist_ok=True)
    metrics = evaluator.evaluate(model, dataset, cfg, save=out / 'predictions',
                                 method=args.condition)[0]
    dump(out / 'metrics.json', dict(dataset=args.dataset, model=args.model,
                                    condition=args.condition, **metrics))
    print(json.dumps(metrics, indent=2))


# ------------------------------------------------------------------------- benchmark
def benchmark(args):
    cfg, data = load(args.dataset)
    rows = benchmark_module.benchmark(args.dataset, cfg, data, dirs_for(args.dataset),
                                      model_names=args.models, warmup=args.warmup,
                                      repeats=args.repeats, batches=args.batch,
                                      seeds=args.seeds, priors=load_priors(args.dataset, cfg))
    print(json.dumps([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows],
                     indent=2))


# ---------------------------------------------------------------------------- report
def report(args):
    cfg = load_config(args.dataset)
    summary = reporter.report(args.dataset, cfg, dirs_for(args.dataset))
    if summary is None:
        print(f'{args.dataset}: no finished runs to aggregate')
    else:
        print(f'Wrote {Path(dirs_for(args.dataset)["results"]) / "report.md"}')


def _device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; use --device cpu for frozen inference')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)

    def common(p, dataset_required=True):
        p.add_argument('--dataset', choices=DATASETS, required=dataset_required)
        return p

    p = common(sub.add_parser('prepare', help='build the dataset cache and training-only priors'))
    p.add_argument('--zip', help='Hill of Towie 2020 archive')
    p.set_defaults(func=prepare)

    p = common(sub.add_parser('preflight', help='verify protocol counts and find a feasible microbatch'))
    p.set_defaults(func=preflight)

    p = common(sub.add_parser('train', help='train one (model, strategy, seed) run'))
    p.add_argument('--model', choices=NAMES, required=True)
    p.add_argument('--strategy', choices=['clean', 'noise'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--micro', type=int, help='windows per forward pass (default: preflight result)')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--reshuffle-each-epoch', action='store_true')
    p.set_defaults(func=train)

    p = common(sub.add_parser('queue', help='run every model x strategy x seed'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--reshuffle-each-epoch', action='store_true')
    p.set_defaults(func=queue)

    p = common(sub.add_parser('evaluate', help='score an existing checkpoint'))
    p.add_argument('--model', choices=NAMES, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--condition', choices=['clean', 'corrupted'], default='clean')
    p.add_argument('--output')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    p.set_defaults(func=evaluate)

    p = common(sub.add_parser('benchmark', help='measure parameters, latency and peak memory'))
    p.add_argument('--models', choices=NAMES, nargs='+')
    p.add_argument('--warmup', type=int)
    p.add_argument('--repeats', type=int)
    p.add_argument('--batch', type=int, nargs='+')
    p.add_argument('--seeds', type=int, nargs='+')
    p.set_defaults(func=benchmark)

    p = common(sub.add_parser('report', help='aggregate finished runs'))
    p.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
