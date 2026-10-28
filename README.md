# SGSA: Static Graph with Statistical Attention

Research code for lightweight multi-turbine wind power forecasting. SGSA combines a
learned static turbine graph, a scaled self-feature residual (`rho=0.25`), learnable
positional encoding, temporal SimAM weighting, and a shared direct prediction head.
History=96, horizon=48, hidden width=64, graph rank=16. Each dataset is trained
separately.

The proposed model has a single implementation, `sgsa_utils/model.py`. The module names
that earlier scripts used (`sgsa.py`, `model.py`, `sgsa_layers.py`, `layers.py`,
`sgsa_model.py`) are thin re-export shims, so the two protocols cannot diverge.

## Repository layout
- `experiments/SDWPF/`: 134 turbines, 16 variables; configuration and all benchmark adapters.
- `experiments/Hill21/`: Hill of Towie, 21 turbines, 7 variables; January–March 2020.
- `sgsa_utils/model.py`: the SGSA architecture, temporal SimAM, historical checkpoint loader.
- `sgsa_utils/checkpoints.py`: strict checkpoint loading and hash-bound Fourier-index metadata.
- `docs/PAPER_CODE_MAPPING.md`: every paper claim mapped to the file that implements it.
- `docs/DATA_PROTOCOLS.md`: the two protocols and their differences.
- `docs/IMPLEMENTATION_NOTES.md`: baseline adaptations and reporting conventions.
- `docs/DEVELOPMENT_HISTORY.md`: prior exploratory use of the evaluation panels.
- `vendor/`: licensed third-party implementations with their original notices.

## Installation
Use a separate Python environment. GPU training uses NVIDIA CUDA; install a matching
PyTorch build, then run `pip install -r requirements.txt`. The original measurements used
an RTX 3080 Ti. Consult `docs/DATA_PROTOCOLS.md` for the different dataset protocols.
CPU execution supports frozen evaluation and the integrity checks only.

## Data preparation
Raw data and checkpoints are distributed separately from source code.

**Hill of Towie:** obtain the 2020 archive from https://doi.org/10.5281/zenodo.14870023 .
The six monthly CSVs used here are `tblSCTurbine` and `tblSCTurGrid` for January–March
2020. They may remain in the full-year ZIP or in a ZIP containing only those six files.

```sh
python experiments/Hill21/code/prepare_data.py /path/to/2020.zip
python experiments/Hill21/code/prepare_priors.py
python experiments/Hill21/code/worker.py SGSA
```

The worker uses fixed settings from `experiments/Hill21/config.json`, three seeds and
clean/noise training. No hyperparameter search runs as part of preprocessing or the
worker. Substitute another model name, including `FEDformer`, to run that benchmark. To
evaluate existing checkpoints without training:

```sh
python experiments/Hill21/code/worker.py FEDformer --evaluate-only --device cpu
```

Place checkpoints and their run files under
`experiments/Hill21/runs/<model>/<clean|noise>/<seed>/`. New checkpoints include
`best.pt.metadata.json`. Six original FEDformer checkpoints can also load with the supplied
SHA256-matched records in `checkpoint_metadata/`.

**SDWPF:** source https://doi.org/10.1038/s41597-024-03427-5 . Place the audited processed
cache at `experiments/SDWPF/data/data_20000.npz`. The release identifies its hash and
protocol; it does not substitute an unverified raw-to-cache recipe. Then:

```sh
python experiments/SDWPF/code/prepare_priors.py
python experiments/SDWPF/code/run_suite.py preflight
python experiments/SDWPF/code/run_suite.py train --model SGSA --strategy clean --seed 2024
python experiments/SDWPF/code/run_suite.py queue          # 13 models x 2 strategies x 3 seeds
python experiments/SDWPF/code/run_suite.py benchmark --model SGSA   # paper Table 4 protocol
```

Preflight discovers a feasible microbatch while preserving 32 windows per optimizer step
through gradient accumulation, and refuses to continue if the window counts or the
configuration disagree with the frozen protocol. The queue covers 13 methods and 78
training runs. Extended model adapters retain their documented differences from the
original papers. For StemGNN only, first run `python tools/fetch_stemgnn.py`; its upstream
source is fetched locally and is not redistributed here.

## Verification without training
```sh
python tools/verify_sources.py     # packaged file hashes + Python syntax
python tools/smoke_checks.py       # CPU forward checks, parameter counts vs the paper
```
`verify_sources.py` checks the packaged hashes and parses every Python file without
importing models or starting training. `smoke_checks.py` builds every architecture on CPU,
compares the parameter counts with the values printed in Tables 4 and 7, confirms that both
`config.json` files match the recorded splits, and calls `--help` on each entry point; it
writes `provenance/smoke_checks.json`. Four adapters need inputs that are not
redistributed (the training-only prior file for PDFormer and the separately fetched StemGNN
source) and are reported as skipped. After an intentional source edit, regenerate the hash
manifest with `python tools/update_manifest.py`.

## Disclosures
Both evaluation panels were inspected during development, and both configuration files
record how the hyperparameters were chosen:

- `experiments/SDWPF/config.json`: `rho` was compared at 0.5 and 0.25 on the development
  test panel before being fixed at 0.25; the 0.5 runs are archived and are not reported.
  The test period `[16000,20000)` is a development evaluation panel, not an untouched
  confirmatory holdout.
- `experiments/Hill21/config.json`: the learning rate and dropout were selected from a
  six-candidate grid with seed 2024 by validation MAE; the processed test period was also
  inspected during development.
- All reported metrics are on the training-only z-score scale, not in kW.
- `MultiGSA_adapted` is a documented structural adaptation, not a reproduction of the
  original paper's configuration.
- The released SDWPF runs use a fixed batch order on every epoch; see
  `docs/DATA_PROTOCOLS.md`.

## Reproducibility
See `REPRODUCIBILITY.md` and `docs/PAPER_CODE_MAPPING.md`. Both experiments use
chronological partitions, training-only normalization, finite-target masks, and
input-only artificial corruption. Their preprocessing details and training settings
differ. This repository does not claim that the evaluated panels are untouched
confirmatory test sets.

## Licensing
No additional open-source license has currently been granted for repository-owned code.
