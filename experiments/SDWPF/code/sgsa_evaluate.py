"""Evaluate a converted checkpoint on the frozen historical test protocol.

The split, stride and corruption setting come from
``experiments/SDWPF/config.json`` so that a standalone evaluation cannot drift
from the benchmark. Metrics are computed on the training-only z-score scale with
missing labels excluded.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from model import SGSA
from data_loader import load_and_prepare, Windows, loader
from sgsa_utils.checkpoints import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


@torch.inference_mode()
def evaluate(model, dataset, save=None):
    model.eval()
    absolute = squared = count = 0.
    arrays = {k: [] for k in ['prediction', 'target', 'mask', 'starts']}
    for x, y, m, starts, _ in loader(dataset):
        pred = model(x.cuda()).cpu()
        if not torch.isfinite(pred).all():
            raise ValueError('Nonfinite predictions')
        valid = m.bool()
        e = (pred.double() - y.double())[valid]
        absolute += e.abs().sum().item()
        squared += e.square().sum().item()
        count += valid.sum().item()
        if save:
            for key, value in zip(arrays, [pred, y, m, starts]):
                arrays[key].append(value.numpy())
    if not count:
        raise ValueError('No valid targets')
    if save:
        np.savez_compressed(save, **{k: np.concatenate(v) for k, v in arrays.items()},
                            turbine_ids=dataset.data.turbine_ids)
    return dict(mae=absolute / count, mse=squared / count, rmse=(squared / count) ** .5,
                valid_points=int(count))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--condition', choices=['clean', 'corrupted'], default='clean')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    data = load_and_prepare(a.data, CFG['split'])
    if data.x_scaled.shape != (20000, 134, 16):
        raise ValueError('Protocol requires the audited 20000-point array')
    model = SGSA(CFG['seq_len'], CFG['num_nodes'], CFG['input_dim'], CFG['pred_len'],
                 CFG['d_model'], CFG['dropout'], CFG['graph_rank'], CFG['rho']).cuda()
    checkpoint_record = load_checkpoint(model, a.checkpoint)
    ds = Windows.for_partition(data, 'test', seed=CFG['corruption_seed'],
                               augmentation=False, stride=CFG['evaluation_stride'],
                               severity=None if a.condition == 'clean' else 2,
                               seq_len=CFG['seq_len'], pred_len=CFG['pred_len'])
    result = evaluate(model, ds, out / 'predictions.npz')
    if result['valid_points'] != 966746:
        raise ValueError('Valid test point drift from paper Table 2')
    result['checkpoint'] = checkpoint_record
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(result)
