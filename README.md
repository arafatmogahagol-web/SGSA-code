# SGSA: Static Graph with Statistical Attention

Research code for lightweight multi-turbine wind power forecasting. SGSA combines a
learned static turbine graph, a scaled self-feature residual (`rho=0.25`), learnable
positional encoding, temporal SimAM weighting, and a shared direct prediction head.
History=96, horizon=48, hidden width=64, graph rank=16. Each dataset is trained
separately.

The proposed model has a single implementation, `code/SGSA.py`. One framework serves both
datasets: shapes and training settings come from `code/datasets/<name>/config.json`, so
no dataset constant is hard-coded in the model, the data pipeline or the trainer.

## Layout

```
code/
├── SGSA.py                 proposed model (static graph + scaled residual + SimAM head)
├── run.py                  single entry point: prepare / preflight / train / queue /
│                           evaluate / benchmark / report
├── layers/simam.py         parameter-free temporal attention (paper Eq. 4)
├── models/                 13 compared architectures, one registry (models/__init__.py)
│   ├── multigsa.py         MultiGSA_adapted (structural adaptation)
│   ├── temporal.py         LSTM, Transformer (weights shared across turbines)
│   ├── mtgnn.py            MTGNN
│   ├── fedformer.py        FEDformer
│   ├── extended.py         GraphWaveNet, AGCRN, STID, DLinear, PatchTST
│   └── extra.py            PDFormer, StemGNN
├── data_provider/          per-dataset preprocessing + windows + corruption
│   ├── common.py           prepared-data container, window slicing, corruption levels
│   ├── sdwpf.py            SDWPF cache loader (134 turbines, 16 variables)
│   ├── hill21.py           Hill of Towie raw-to-panel builder (21 turbines, 7 variables)
│   └── priors.py           training-only graph/DTW/pattern priors
├── exp/                    trainer, evaluator, guards, cost benchmark, reporter
├── utils/                  checkpoint identity, seeding, plotting, parameter counting
├── datasets/
│   ├── SDWPF/              config.json + data/, train_priors.npz (both untracked)
│   └── Hill21/             config.json + data audit + FEDformer recovery records
├── checkpoints/            trained weights, ignored by Git
└── results/                metrics, tables and figures, ignored by Git
```

Every dataset keeps its own folder: `code/datasets/SDWPF` and `code/datasets/Hill21` each
hold the configuration, the audit record and the (untracked) data cache. Supporting
records: `docs/EXPERIMENT_REVIEW_20261028.md` (protocol review against the manuscript) and
`docs/PAPER_CODE_MAPPING.md` (claim-by-claim map).

## Installation
Use a separate Python environment. GPU training uses NVIDIA CUDA; install a matching
PyTorch build, then run `pip install -r requirements.txt`. The original measurements used
an RTX 3080 Ti. CPU execution supports frozen evaluation and the integrity checks only.

## Usage

```sh
# 0) data: see below, then build the training-only priors
python code/run.py prepare   --dataset Hill21 --zip /path/to/2020.zip
python code/run.py prepare   --dataset SDWPF

# 1) verify the protocol counts and pick a feasible microbatch
python code/run.py preflight --dataset SDWPF

# 2) one run, or the full grid (13 models x 2 strategies x 3 seeds)
python code/run.py train     --dataset SDWPF --model SGSA --strategy clean --seed 2024
python code/run.py queue     --dataset SDWPF

# 3) score a checkpoint, or re-measure parameters / latency / peak memory
python code/run.py evaluate  --dataset Hill21 --model SGSA --checkpoint best.pt --device cpu
python code/run.py benchmark --dataset SDWPF

# 4) aggregate everything that finished
python code/run.py report    --dataset SDWPF
```

All commands read `code/datasets/<name>/config.json`; `preflight` fails if the prepared
data, the window counts or the valid-label count disagree with it.

## Data preparation
Raw data and checkpoints are distributed separately from source code.

**Hill of Towie:** obtain the 2020 archive from https://doi.org/10.5281/zenodo.14870023 .
The six monthly CSVs used here are `tblSCTurbine` and `tblSCTurGrid` for January–March
2020. They may remain in the full-year ZIP or in a ZIP containing only those six files.
`run.py prepare` writes `code/datasets/Hill21/data/panel.npz` plus `data_audit.json` and
the training-only priors.

**SDWPF:** source https://doi.org/10.1038/s41597-024-03427-5 . Place the audited processed
cache at `code/datasets/SDWPF/data/data_20000.npz`. The release identifies its hash and
protocol; it does not substitute an unverified raw-to-cache recipe. `run.py prepare` then
builds the training-only priors.

Checkpoints are written to `code/checkpoints/<dataset>/<model>/<strategy>/<seed>/best.pt`
with a `best.pt.metadata.json` sidecar that binds the weight hash to FEDformer's frequency
index lists. The six original Hill21 recovery records stay in
`code/datasets/Hill21/checkpoint_metadata/`. Extended model adapters retain their
documented differences from the original papers. For StemGNN only, first run
`python tools/fetch_stemgnn.py`; its upstream source is fetched locally and is not
redistributed here.

## Verification without training
```sh
python tools/verify_sources.py         # packaged file hashes + Python syntax
python tools/smoke_checks.py           # CPU forward checks, parameter counts vs the paper
python tools/test_integrity_guards.py  # checkpoint rho, run reuse and window origins
```
`verify_sources.py` checks the packaged hashes and parses every Python file without
importing models or starting training. `smoke_checks.py` builds every architecture on CPU,
compares the parameter counts with the values printed in Tables 4 and 7, cross-checks both
dataset configurations against their audit records, runs the integrity guards and pings each
entry point; it writes `provenance/smoke_checks.json`. Four adapters need inputs that are
not redistributed (the training-only prior file for PDFormer and the separately fetched
StemGNN source) and are reported as skipped. After an intentional source edit, regenerate
the hash manifest with `python tools/update_manifest.py`.

## Checkpoint and run identity
New checkpoints carry a hash-bound sidecar that records the frequency index lists and the
residual coefficient, so loading weights trained with a different `rho` fails even when all
tensor names and shapes match. Plain historical `state_dict` files remain loadable, but they
do not encode `rho`; the metadata record marks whether it was verified. Before a run
directory is reused or resumed, its stored configuration is compared with the current one,
including the shuffle choice, and a mismatch raises instead of silently mixing results.
`python tools/test_integrity_guards.py` exercises these protections and checks that the
window helper reproduces the paper's origins and counts.

## Disclosures
Both evaluation panels were inspected during development, and both dataset configurations
record how the hyperparameters were chosen:

- `code/datasets/SDWPF/config.json`: `rho` was compared at 0.5 and 0.25 on the development
  test panel before being fixed at 0.25; the 0.5 runs are archived and are not reported.
  The test period `[16000,20000)` is a development evaluation panel, not an untouched
  confirmatory holdout.
- `code/datasets/Hill21/config.json`: the learning rate and dropout were selected from a
  six-candidate grid with seed 2024 by validation MAE; the processed test period was also
  inspected during development.
- All reported metrics are on the training-only z-score scale, not in kW.
- `MultiGSA_adapted` is a documented structural adaptation, not a reproduction of the
  original paper's configuration.
- The released SDWPF runs used a fixed batch order on every epoch
  (`shuffle_each_epoch=false` in the configuration); Hill21 reshuffles every epoch.

## Reproducibility
See `REPRODUCIBILITY.md` and `docs/PAPER_CODE_MAPPING.md`. Both experiments use
chronological partitions, training-only normalization, finite-target masks, and
input-only artificial corruption. Their preprocessing details and training settings
differ. This repository does not claim that the evaluated panels are untouched
confirmatory test sets.

## Licensing
No additional open-source license has currently been granted for repository-owned code.
