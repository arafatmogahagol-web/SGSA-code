# Implementation notes

## One implementation of everything

`code/SGSA.py` is the only implementation of the proposed model; `code/layers/simam.py` is
the only implementation of the statistical attention (paper Eq. 4):

```
e   = (Z - mu)^2 / (4 (v + eps)) + 0.5      eps = 1e-4, v = population variance
a   = sigmoid(e)
H_a = a * Z
```

`code/models/` holds one implementation per baseline in a single registry, and both datasets
instantiate them from their own `config.json`. There is no per-dataset copy of a model or of
the training step.

Parameter names are unchanged from the released checkpoints (`input_proj`, `node_left`,
`node_right`, `spatial_proj`, `spatial_norm`, `position`, `temporal_norm`, `filter`,
`forecast`), so `state_dict` files still load with `strict=True`.

## Vendor packages

`vendor/` sits next to `code/` and is never edited on disk. Two vendors ship top-level
packages whose names collide with this repository (`layers`, and `utils` in the case of
FEDformer). `models/__init__.py` installs those vendor packages only while the vendor model
is being constructed and restores this repository's packages afterwards, so neither side
shadows the other. Loaded vendor modules are cached under private names
(`official_fedformer`, `official_agcrn`, ...).

## Explicit source patches

Two vendor files need a documented, asserted edit instead of an on-disk change:

- GraphWaveNet: the 2019 code calls `nn.Conv1d` with a two-dimensional kernel on a
  four-dimensional input; the adapter replaces all three occurrences with `nn.Conv2d` (the
  intended operation on current PyTorch) and asserts the count.
- StemGNN: `torch.rfft`/`torch.irfft` were removed from PyTorch. They are replaced by
  two-sided `torch.fft` calls that keep both real and imaginary components and the original
  normalization.

`models/_vendor.py` fails loudly if the upstream text changes, so a patch can never apply
silently to a different file.

## Baseline adaptations

- `MultiGSA_adapted` is a multi-turbine structural adaptation, not a reproduction of its
  paper's datasets, architecture settings or training recipe.
- STID uses node identity but no time-of-day/day-of-week inputs.
- DLinear and PatchTST operate turbine-wise with shared parameters; the target power channel
  is selected for evaluation.
- PDFormer uses training-only power correlations, DTW profiles, Laplacian positions and
  pattern priors; no geographic coordinates and no test-fitted graph.
- StemGNN uses the historical power channel only, and its upstream file is fetched
  separately (`tools/fetch_stemgnn.py`) because no redistribution license was established.
- FEDformer frequency lists are part of checkpoint identity even though upstream does not
  register them as `state_dict` buffers; the shared loader restores them from a hash-bound
  sidecar. Six original Hill21 recovery records are included.

## Reporting conventions

- MAE and RMSE are computed on the training-only z-score scale, not in kW.
- Parameter counts are real-valued scalars: a complex parameter counts as two.
  `utils/params.py` reports both conventions.
- Loss and metrics are masked by the valid-label mask; padding a missing label with zero in
  storage never makes it a valid target.
- Errors are accumulated in float64, so reported values do not depend on batch size.
- No significance test is claimed from three-seed means alone.

## Configuration provenance

`code/datasets/*/config.json` is the single source of truth, including rho. Each file states
how its hyperparameters were chosen during development, including the fact that the
evaluation panels were inspected before the final configuration was frozen.

This release reorganises the tree under `code/`, merges the duplicated per-dataset model
files into one registry, and routes the protocol through the configuration. It does not
modify the SGSA computational graph and does not retrospectively retune the experiments.
