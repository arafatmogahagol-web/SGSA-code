# Implementation adaptations

## Single implementation of the proposed model

The model now has exactly one implementation, `sgsa_utils/model.py`. The files
`experiments/*/code/sgsa.py`, `model.py`, `sgsa_layers.py`, `layers.py` and
`sgsa_model.py` are import shims that re-export it, so the two protocols cannot
drift apart. The computational graph, variable names and `state_dict` keys are
unchanged, so historical checkpoints still load with `strict=True`.

Implemented statistical attention (paper Eq. 4), computed along the historical
axis with no learnable parameter:

```
e = (Z - μ)² / (4 (v + ε)) + 0.5      ε = 1e-4, v = population variance
a = sigmoid(e)
H_a = a ⊙ Z
```

## Baselines

- `MultiGSA_adapted` is a multi-turbine structural adaptation, not an exact
  reproduction of its paper's datasets, architecture settings or training recipe.
  It reuses the same parameter-free statistical attention class for consistency.
- STID uses node identity but no extra time-of-day/day-of-week inputs.
- DLinear and PatchTST operate turbine-wise with shared parameters; the target power
  channel is selected for evaluation.
- PDFormer uses training-only power correlations, DTW profiles, graph positions and
  pattern priors; no geographic coordinates or test-fitted graph is claimed.
- StemGNN uses the historical power channel only. Its removed `torch.rfft/irfft`
  operations are replaced in the adapter by equivalent two-sided `torch.fft`
  transforms. Fetch the original file separately; its redistribution license was not
  established.
- GraphWaveNet's historical four-dimensional convolution input is handled with
  `Conv2d` in the adapter on current PyTorch.
- FEDformer frequency lists are part of checkpoint identity even though upstream does
  not register them as `state_dict` buffers. The shared loader restores them before
  inference and checks the weight hash. Six original Hill21 recovery records are
  included.
- All third-party directories retain their own notices. Original repository paths in
  the provenance manifest have been reduced to relative source names.

## Reporting conventions

- MAE and RMSE are computed on the training-only z-score scale, not in kW.
- Parameter counts are real-valued scalars: a complex parameter counts as two.
  `models.parameter_counts` reports both conventions.
- Loss and metrics are masked by the valid-label mask; padding a missing label with
  zero in storage never makes it a valid target.
- No significance test is claimed from three-seed means alone; the paper reports the
  mean and the sample standard deviation.

## Configuration provenance

`experiments/SDWPF/config.json` and `experiments/Hill21/config.json` are the single
source of truth for the reported configuration, including ρ. Both files state how the
hyperparameters were chosen during development, including the fact that the
evaluation panels were inspected before the final configuration was frozen.

This release corrects packaging, import, duplication and configuration-wiring
problems. It does not modify the SGSA computational graph or retrospectively retune
the experiments.
