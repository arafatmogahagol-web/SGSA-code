"""Training-only graph priors used by the graph-based baselines.

Everything here is estimated on the training partition only: the correlation
graph, the symmetric top-k adjacency, the transition matrix, shortest-path hops,
the Laplacian eigenvectors and the DTW profiles of the mean 48-slot power shape,
plus a fixed three-step pattern dictionary. No validation or test array is read.

PDFormer consumes these priors; SGSA does not (its graph is learned end to end).
"""
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path

PATTERN_COUNT = 16
PATTERN_ITERS = 30
PATTERN_SEED = 2024
TOP_K = 10
POLES = 8


def build(cfg, data, root, audit_name='prior_audit.json'):
    train_end = data.split['train'][1]
    power = data.x[:train_end, :, -1].astype('float64')
    nodes = power.shape[1]

    # Historical missing values already use training medians, as in the benchmark.
    corr = np.nan_to_num(np.corrcoef(power.T), nan=0.)
    np.fill_diagonal(corr, 1.)
    affinity = np.abs(corr)
    np.fill_diagonal(affinity, 0.)
    adjacency = np.zeros((nodes, nodes))
    for i in range(nodes):
        ind = np.argsort(-affinity[i], kind='stable')[:TOP_K]
        adjacency[i, ind] = affinity[i, ind]
    adjacency = np.maximum(adjacency, adjacency.T)

    transition = adjacency + np.eye(nodes)
    transition /= transition.sum(axis=1, keepdims=True)
    hops = shortest_path((adjacency > 0).astype(float), directed=False, unweighted=True)
    degree = adjacency.sum(1)
    inv = 1 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(nodes) - inv[:, None] * adjacency * inv[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    poles = min(POLES, nodes - 1)
    lap_pe = vectors[:, 1:poles + 1]
    for j in range(poles):
        if lap_pe[np.argmax(np.abs(lap_pe[:, j])), j] < 0:
            lap_pe[:, j] *= -1

    # Exact DTW on mean 48-slot profiles computed only from the training slice.
    profile = np.stack([power[k::48].mean(0) for k in range(48)], axis=1)
    dtw = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in range(i):
            d = np.full((49, 49), np.inf)
            d[0, 0] = 0
            for a in range(1, 49):
                for b in range(1, 49):
                    d[a, b] = (profile[i, a - 1] - profile[j, b - 1]) ** 2 + min(
                        d[a - 1, b], d[a, b - 1], d[a - 1, b - 1])
            dtw[i, j] = dtw[j, i] = np.sqrt(d[-1, -1])

    # Three-step power patterns; predeclared k=16, seed 2024 and 30 Lloyd steps.
    patterns = np.stack([power[:-2], power[1:-1], power[2:]], axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(PATTERN_SEED)
    sample = patterns[rng.choice(len(patterns), min(20000, len(patterns)), replace=False)]
    centers = sample[rng.choice(len(sample), PATTERN_COUNT, replace=False)].copy()
    for _ in range(PATTERN_ITERS):
        assignment = ((sample[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        for k in range(PATTERN_COUNT):
            if (assignment == k).any():
                centers[k] = sample[assignment == k].mean(0)

    out = Path(root) / cfg['priors']
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, transition=transition.astype('float32'), adjacency=adjacency.astype('float32'),
        correlation=corr.astype('float32'), hops=hops.astype('float32'),
        lap_pe=lap_pe.astype('float32'), dtw=dtw.astype('float32'),
        pattern_keys=centers[..., None].astype('float32'))

    from utils.io import sha256
    audit = dict(
        source=cfg['data'], source_sha256=sha256(Path(root) / cfg['data']),
        fit_partition=list(data.split['train']), validation_used=False, test_used=False,
        graph=f'abs training power correlation top{TOP_K} symmetric',
        dtw='exact DTW of training-only mean 48-slot profiles',
        patterns=f'3-step training segments, {PATTERN_COUNT} centers, '
                 f'seed{PATTERN_SEED},{PATTERN_ITERS} Lloyd iterations',
        prior_sha256=sha256(out))
    (Path(root) / audit_name).write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit
