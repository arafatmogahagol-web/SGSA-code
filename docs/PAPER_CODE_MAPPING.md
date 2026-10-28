# Paper ↔ code map

Every number and protocol statement in the experimental sections of the paper maps
to a specific file. This table is meant to be read next to the manuscript so that a
reader can check any claim without rerunning training. Where a claim cannot be
reproduced from the repository alone, that limitation is stated.

Legend: `config.json` = `experiments/SDWPF/config.json`,
`H21/config.json` = `experiments/Hill21/config.json`.

## Model (Sections 3.2–3.4)

| Paper item | Code | Note |
|---|---|---|
| Eq. (1) input/output shapes `B×T×N×F → B×P×N` | `sgsa_utils/model.py::SGSA.forward` | shape is validated on entry |
| Eq. (2) `S = softmax_row(ELU(E_L E_R))`, shared over time | `sgsa_utils/model.py::SGSA.forward` (`node_left`, `node_right`, `graph_rank=16`) | one `S` per forward pass, broadcast over batch and time |
| Eq. (2) aggregation `A[b,t,i,:] = Σ_j S[i,j] H[b,t,j,:]` | same file, `torch.einsum('bij,btjd->btid', a, h)` | |
| Eq. (3) scaled self-feature residual, `ρ = 0.25` | same file, `self.rho * h + dropout(gelu(spatial_proj(spatial)))` then `LayerNorm` | ρ comes from `config.json` |
| Eq. (3) positional encoding + `LN_t` | same file, `self.position`, `self.temporal_norm` | |
| Eq. (4) temporal SimAM weighting | `sgsa_utils/model.py::SimAM1D` | implemented energy: `e = (Z-μ)²/(4(v+ε)) + 0.5`, `a = σ(e)`, `H_a = a ⊙ Z` (ε = 1e-4, no learnable parameters) |
| Eq. (5) shared direct multi-step head | `sgsa_utils/model.py::SGSA.forward` (`self.forecast`) | one `d→d→P` MLP shared by all turbines |
| Parameter counts 23,216 / 19,024 | `SGSA(96,134,16,48,64,.1,16,.25)` / `SGSA(96,21,7,48,64,.1,16,.25)` | matches Tables 4 and 7 exactly |

## Experiment I — SDWPF (Section 4)

| Paper item | Code | Reproduce / verify |
|---|---|---|
| Processed cache 134×16×20,000, target `Patv` | `run_suite.py::load_data` (shape assertion) | requires `data/data_20000.npz`; SHA256 recorded in `docs/DATA_PROTOCOLS.md` |
| Table 2 split `[0,14000) / [14000,15000) / [15000,16000) / [16000,20000)` | `config.json` → `preprocessing.PROTOCOL_SPLIT` (asserted equal) | `python tools/verify_sources.py` then `run_suite.py preflight` |
| Table 2 window counts 3,465 / 40 / 165 / 153 | `data_loader.Windows` + `run_suite.py::preflight` assertion | preflight raises if the counts differ |
| Table 2 valid test points 966,746 | `run_suite.py::train_one` assertion after scoring | asserted for every run |
| Training-only z-score and training-median imputation | `preprocessing.load_and_prepare` | statistics computed from `split['train']` only |
| Labels never imputed; masked loss/metrics | `preprocessing.load_and_prepare` (`y_mask`) + `run_suite.py::evaluate` | |
| Stride 4 / 24 | `config.json` (`train_stride`, `evaluation_stride`) | |
| Optimizer, learning rate, weight decay, clipping, patience, epochs | `config.json` + `run_suite.py::train_one` | |
| Seeds 2024 / 2025 / 2026, mean ± sample std | `config.json` (`seeds`) + `run_suite.py::report` | `report` groups per model/strategy/condition |
| Corruption 50 % clean / 30 % light / 20 % heavy, 5 % and 20 % packet loss, ±8 outliers on inputs only | `data_loader.py` (`LEVEL_SHARE`, `LIGHT_LOSS`, `HEAVY_LOSS`, `OUTLIER_SHARE`) | heavy corruption is the test condition (`severity=2`) |
| Table 3 accuracy numbers | `outputs/<model>/<strategy>/<seed>/<condition>/metrics.json` | produced by `train` runs; aggregated by `report` |
| Table 4 parameters, latency, peak memory | `run_suite.py benchmark` | batch 32, 20 warm-up iterations, 100 timed repetitions, median per checkpoint, averaged over the three clean checkpoints |
| Fig. 4 error bars | `outputs/summary.csv` | sample std over the three seeds |
| Figs. 5–8 curves, adjacency | `write_metrics`, `outputs/<model>.../loss.*`, `static_adjacency.npy` | |

## Experiment II — Hill of Towie (Section 5)

| Paper item | Code | Reproduce / verify |
|---|---|---|
| 21 turbines × 7 variables, 4,369 points, 30-min grid, 10-min availability lag | `H21/code/prepare_data.py` | requires the public 2020 archive (Zenodo DOI in `README.md`) |
| Split `[0,3058) / [3058,3713) / [3713,4369)` | `H21/config.json` (`split`) + `data_audit.json` | boundaries come from the same audit file |
| Windows 729 / 26 / 26, forward fill then training-median imputation | `prepare_data.py` | window counts written to `data_audit.json` |
| 26,187 valid test points | `H21/code/training_core.py::EXPECTED_VALID_TEST_POINTS`, asserted in `worker.py` | asserted on every evaluation |
| Table 5 training settings | `H21/config.json` | |
| Table 6 accuracy numbers | `outputs/<model>_metrics.csv` | |
| Table 7 parameters, latency, peak memory | `H21/code/worker.py` efficiency block | batch 32, 10 warm-up iterations, 30 timed repetitions, batch 1 also recorded |
| Complex parameters as two real scalars | `training_core.real_scalars`, `worker.py` | applies only to FEDformer |

## Disclosures that are deliberately in the code and not hidden

* `config.json` → `hyperparameter_provenance`: ρ was compared at 0.5 and 0.25 on the
  development test panel before being fixed at 0.25; the 0.5 runs are archived and
  are not reported.
* `config.json` → `development_panel_note`: the test period was inspected during
  development and is a development evaluation panel, not a confirmatory holdout.
  The same statement is repeated in `README.md`, `REPRODUCIBILITY.md`,
  `docs/DEVELOPMENT_HISTORY.md` and in the per-run `protocol.json`.
* `run_suite.py` records `train_shuffle` in every `protocol.json`: the released runs
  used a fixed batch order on every epoch; `--reshuffle-each-epoch` switches to a
  per-epoch order and was not used for the reported numbers.
* `docs/IMPLEMENTATION_NOTES.md` lists every baseline adaptation, and
  `MultiGSA_adapted` is labelled a structural adaptation, not a reproduction of the
  original paper's configuration.

## Limits of repository-only reproduction

* Raw datasets, processed caches and trained checkpoints are distributed separately
  (see `README.md`); the repository contains source code, protocol records and audit
  hashes only.
* Bit-identical historical scores are not promised: fresh training consumes random
  numbers differently from the original development classes even for the same seed
  (see `REPRODUCIBILITY.md`).
* GPU wall-clock numbers depend on the measurement host. The protocol (batch size,
  warm-up, repetitions, median per checkpoint) is reproducible; the absolute
  milliseconds are not a claim about other hardware.
