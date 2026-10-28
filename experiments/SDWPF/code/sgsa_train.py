"""Standalone single-model SGSA training entry. No training starts on import.

This is a convenience path for one (seed, strategy) pair; the reported numbers
come from ``run_suite.py``, which additionally fixes the microbatch through
preflight. Configuration is read from ``experiments/SDWPF/config.json`` so this
script cannot silently use a different protocol.

Fresh initialization differs from the old inherited classes' RNG consumption.
Converted historical checkpoints are the audited exact-reproduction route.
"""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from model import SGSA
from data_loader import load_and_prepare, Windows
from sgsa_evaluate import evaluate
from sgsa_utils.checkpoints import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, choices=[2024, 2025, 2026], required=True)
    p.add_argument('--strategy', choices=['clean', 'noise'], required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg = json.loads(Path(a.config).read_text())
    if not cfg.get('training_authorized', False):
        raise RuntimeError('Protocol draft: training is not yet authorized')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    data = load_and_prepare(a.data, cfg['split'])
    if data.x_scaled.shape != (20000, 134, 16):
        raise ValueError('Unexpected dataset')
    model = SGSA(cfg['seq_len'], cfg['num_nodes'], cfg['input_dim'], cfg['pred_len'],
                 cfg['d_model'], cfg['dropout'], cfg['graph_rank'], cfg['rho']).cuda()
    train = Windows.for_partition(data, 'train', seed=2024,
                                  augmentation=a.strategy == 'noise',
                                  stride=cfg['train_stride'],
                                  seq_len=cfg['seq_len'], pred_len=cfg['pred_len'])
    val = Windows.for_partition(data, 'val', seed=502024,
                                augmentation=a.strategy == 'noise',
                                stride=cfg['evaluation_stride'],
                                seq_len=cfg['seq_len'], pred_len=cfg['pred_len'])
    np.savez_compressed(out / 'preprocessing_and_indices.npz', mean=data.feature_mean,
                        std=data.feature_std, median=data.feature_median,
                        train_starts=train.starts, val_starts=val.starts,
                        turbine_ids=data.turbine_ids)
    (out / 'config.json').write_text(json.dumps(dict(cfg, seed=a.seed, strategy=a.strategy), indent=2))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    # Fixed batch order per epoch, matching the released benchmark.
    from data_loader import loader
    best = float('inf')
    stale = 0
    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        train.epoch = epoch + 1
        total = count = 0
        for x, y, m, _, _ in loader(train, True, a.seed):
            x, y, m = x.cuda(), y.cuda(), m.cuda().bool()
            if not m.any():
                continue
            opt.zero_grad(set_to_none=True)
            e = (model(x) - y)[m]
            loss = e.square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('gradient_clip', 1.0))
            opt.step()
            total += e.detach().square().sum().item()
            count += m.sum().item()
        if not count:
            raise RuntimeError('No valid training labels')
        score = evaluate(model, val)['mse']
        history.append(dict(epoch=epoch + 1, train_mse=total / count, val_mse=score))
        with (out / 'history.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        if score < best:
            best = score
            stale = 0
            save_checkpoint(model, out / 'best.pt')
        else:
            stale += 1
        print(history[-1], flush=True)
        if stale >= cfg['patience']:
            break
    (out / 'complete.json').write_text(json.dumps({'best_val_mse': best, 'test_evaluated': False}))


if __name__ == '__main__':
    main()
