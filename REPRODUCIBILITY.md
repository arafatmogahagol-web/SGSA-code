# Reproducibility status

## What this repository fixes and what it does not
This release moves the source under `code/` in the layout used by the reference repository
(`SGSA.py`, `run.py`, `layers/`, `models/`, `data_provider/`, `exp/`, `utils/`, `datasets/`,
`checkpoints/`, `results/`), keeps the two datasets in separate folders, and replaces the
per-dataset model copies with one registry driven by configuration. It also adds the
inference-cost benchmark that the paper describes. It does not change the SGSA computational
graph and does not retrospectively retune any experiment. The model still has 23,216
parameters on SDWPF and 19,024 on Hill of Towie, matching Tables 4 and 7.

## Configuration as the single source of truth
`code/datasets/SDWPF/config.json` and `code/datasets/Hill21/config.json` hold every reported
setting, including ρ. `code/run.py preflight` aborts if the prepared data disagrees with the
configuration (window counts, valid-label count, shape), and `tools/smoke_checks.py`
cross-checks the configurations against the stored audit records.

## Development panels
Both evaluation panels were inspected during development. The SDWPF residual coefficient ρ
was compared at 0.5 and 0.25 on the processed test panel before the configuration was frozen
at 0.25; the ρ=0.5 runs are archived and are not reported. The Hill21 learning rate and
dropout were chosen from a six-candidate grid by validation MAE with seed 2024. Both
`config.json` files state this, and `docs/DEVELOPMENT_HISTORY.md` records the earlier
exploratory work. The evaluated panels are therefore development evaluation panels, not
untouched confirmatory holdouts; no reported number depends on test labels being used as
model inputs, and graph priors plus normalization statistics are fitted on the training
range only.

## FEDformer
The Hill21 loading issue is resolved in the published source: training writes hash-bound
Fourier metadata beside every best checkpoint; evaluation restores it strictly. Six
SHA256-matched original recovery records are under
`code/datasets/Hill21/checkpoint_metadata/`. A Fourier model without matching metadata raises
an error instead of silently sampling different modes. Non-Fourier checkpoints still load as
plain `state_dict` files.

The historical correction used frozen CPU inference and validation replay, not new training.
The original GPU latency/memory measurements remain separate. Source training defaults still
require CUDA; `run.py evaluate --device cpu` supports checkpoint inspection and prediction.

## Data and experimental scope
Only the 134-turbine SDWPF and 21-turbine Hill of Towie protocols are implemented. See
`docs/DATA_PROTOCOLS.md` for differences, and `docs/DEVELOPMENT_HISTORY.md` for prior
exploratory use of the evaluation panels. No test labels are used as model inputs. Graph
priors and normalization are fitted on training data only.

## Random initialization and outputs
Three seeds are reported separately. Exact historical results require the original caches
and checkpoints, including Fourier metadata. Fresh training from the current model class can
consume random numbers differently from an inherited development class; matching a seed alone
does not promise bit-identical historical scores. The released SDWPF runs also used one fixed
batch order on every epoch (`shuffle_each_epoch` in the configuration);
`run.py train --reshuffle-each-epoch` changes the trajectory and was not used for the
reported numbers.

## Efficiency
SDWPF: 20 warm-up iterations and 100 timed repetitions (`run.py benchmark`). Hill21: 10
warm-up iterations and 30 timed repetitions at batch 1 and 32. GPU forward timing excludes
host-to-device transfer. Compare within each stated protocol; parameter count alone is not a
claim of minimum latency or memory.

## Release checks
Results of source, inference, checkpoint and cached-data checks are saved in
`provenance/release_checks.json`. These checks are not a new full GPU training replication.
Two regenerable checks cover the current tree:

- `python tools/verify_sources.py` re-checks every packaged file hash and parses all Python
  files;
- `python tools/smoke_checks.py` builds every architecture on CPU, compares parameter counts
  with the values printed in the paper, cross-checks both configurations against their audit
  records and pings each entry point, writing `provenance/smoke_checks.json`.

`python tools/update_manifest.py` regenerates the hash manifest after an intentional edit.

## Checkpoint and run identity
New checkpoints include a hash-bound residual coefficient in their metadata, so loading
weights trained with a different rho fails even when all tensor names and shapes match.
Plain historical state dictionaries do not encode rho; they remain compatible, but their
coefficient still requires the original experiment records. Source edits cannot recover
missing historical metadata. Existing run configurations are checked before training reuse,
including the SDWPF shuffle choice. Use a separate run directory for changed settings.

`python tools/test_integrity_guards.py` exercises these protections and verifies that the
window helper produces the paper's origins and counts. The review against the supplied
manuscript is recorded in `docs/EXPERIMENT_REVIEW_20261028.md`.
