"""Inference-cost benchmark with the protocol reported in the paper.

SDWPF (Section 4.3): batch 32, 20 warm-up iterations, 100 timed repetitions,
median per checkpoint, then the mean over the three clean checkpoints.
Hill of Towie (Section 5.3): batch 1 and 32, 10 warm-up iterations, 30 timed
repetitions. The protocol of each dataset is stored in its ``config.json``.

FP32 only, AMP and TF32 disabled, host-to-device transfer excluded, and parameter
counts reported as real-valued scalars (a complex parameter counts as two).
"""
from pathlib import Path

import numpy as np
import torch

from data_provider import load_priors, windows
from models import NAMES, build
from utils.checkpoints import load_checkpoint
from utils.io import csvwrite, dump, sha256
from utils.params import parameter_counts


def _timed_forward(model, x, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return float(np.median(times)), float(peak)


def benchmark(dataset, cfg, data, dirs, model_names=None, warmup=None, repeats=None,
              batches=None, seeds=None, priors=None):
    protocol = cfg.get('benchmark', {})
    warmup = int(warmup if warmup is not None else protocol.get('warmup', 20))
    repeats = int(repeats if repeats is not None else protocol.get('repeats', 100))
    batches = list(batches or protocol.get('batches', [32]))
    seeds = list(seeds or cfg['seeds'])
    if not torch.cuda.is_available():
        raise RuntimeError('Inference benchmarking requires CUDA')

    test = windows(dataset, data, 'test', cfg, severity=None, seed=cfg['corruption_seed'])
    origin = int(test.starts[0])
    sample = torch.from_numpy(data.x[origin - cfg['seq_len']:origin][None]).cuda()
    rows = []
    for name in (model_names or NAMES):
        checkpoints = [Path(dirs['checkpoints']) / name / 'clean' / str(seed) / 'best.pt'
                       for seed in seeds]
        missing = [str(p) for p in checkpoints if not p.exists()]
        if missing:
            print(f'{name}: skipped, missing {missing}', flush=True)
            continue
        records = []
        for batch in batches:
            per_seed = []
            for seed, checkpoint in zip(seeds, checkpoints):
                model = build(name, dict(cfg, dataset=dataset),
                              device='cuda', priors=priors).cuda()
                # Strict load: FEDformer frequency lists come from the hash-bound
                # sidecar instead of being re-sampled.
                load_checkpoint(model, checkpoint)
                model.eval()
                x = sample.repeat(batch, 1, 1, 1)
                latency, peak = _timed_forward(model, x, warmup, repeats)
                per_seed.append(dict(seed=seed, checkpoint_sha256=sha256(checkpoint),
                                     latency_ms=latency, peak_allocated_MiB=peak,
                                     **parameter_counts(model)))
                del model, x
                torch.cuda.empty_cache()
            record = dict(dataset=dataset, model=name, batch_size=batch, warmup=warmup,
                          repeats=repeats, seeds=seeds,
                          parameters=int(np.mean([r['real_scalar_parameters'] for r in per_seed])),
                          latency_ms=float(np.mean([r['latency_ms'] for r in per_seed])),
                          latency_ms_per_seed=[r['latency_ms'] for r in per_seed],
                          peak_allocated_MiB=float(np.mean([r['peak_allocated_MiB'] for r in per_seed])),
                          peak_allocated_MiB_per_seed=[r['peak_allocated_MiB'] for r in per_seed],
                          gpu=torch.cuda.get_device_name(), per_seed=per_seed)
            records.append(record)
            rows.append(record)
            print(record, flush=True)
        folder = Path(dirs['results']) / 'efficiency'
        dump(folder / f'{name}.json', records)
    if rows:
        folder = Path(dirs['results']) / 'efficiency'
        csvwrite(folder / 'efficiency_summary.csv',
                 [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
        dump(folder / 'efficiency_summary.json', rows)
        report = ['# Inference cost', '',
                  f'Protocol: warm-up {warmup}, {repeats} timed repetitions, median per '
                  f'checkpoint, mean over {len(seeds)} clean checkpoints. FP32, no AMP/TF32.',
                  '', '| model | batch | parameters | latency (ms) | peak memory (MiB) |',
                  '|---|---|---|---|---|']
        for row in rows:
            report.append(f"| {row['model']} | {row['batch_size']} | {row['parameters']:,} | "
                          f"{row['latency_ms']:.3f} | {row['peak_allocated_MiB']:.2f} |")
        (folder / 'efficiency.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return rows
