# Paper ↔ code map

Every number and protocol statement in the experimental sections of the paper maps to a
specific file. This table is meant to be read next to the manuscript so that a reader can
check any claim without rerunning training. Where a claim cannot be reproduced from the
repository alone, that limitation is stated.

Legend: `config.json` = `code/datasets/SDWPF/config.json`, `H21/config.json` =
`code/datasets/Hill21/config.json`, and `run.py` = `code/run.py`.

## Model (Sections 3.2–3.4)

| Paper item | Code | Note |
|---|---|---|
| Eq. (1) input/output shapes `B×T×N×F → B×P×N` | `code/SGSA.py::SGSA.forward` | shape validated on entry |
| Eq. (2) `S = softmax_row(ELU(E_L E_R))`, shared over time | `code/SGSA.py::SGSA.static_graph` | one `S` per forward pass, broadcast over batch and time |
| Eq. (2) aggregation `A[b,t,i,:] = Σ_j S[i,j] H[b,t,j,:]` | `code/SGSA.py`, `torch.einsum('bij,btjd->btid', a, h)` | |
| Eq. (3) scaled self-feature residual, `ρ = 0.25` | `code/SGSA.py`, `self.rho * h + dropout(gelu(spatial_proj(spatial)))` then `LayerNorm` | ρ comes from `config.json` |
| Eq. (3) positional encoding + `LN_t` | `code/SGSA.py`, `self.position`, `self.temporal_norm` | |
| Eq. (4) temporal SimAM weighting | `code/layers/simam.py::SimAM1D` | implemented energy `e = (Z-μ)²/(4(v+ε)) + 0.5`, `a = σ(e)`, `H_a = a ⊙ Z` (ε = 1e-4, no learnable parameter) |
| Eq. (5) shared direct multi-step head | `code/SGSA.py`, `self.forecast` | one `d→d→P` MLP shared by all turbines |
| Parameter counts 23,216 / 19,024 | `SGSA(96,134,16,48,64,.1,16,.25)` and `SGSA(96,21,7,48,64,.1,16,.25)` | matches Tables 4 and 7 exactly |

## Experiment I — SDWPF (Section 4)

| Paper item | Code | Reproduce / verify |
|---|---|---|
| Processed cache 134×16×20,000, target `Patv` | `code/data_provider/sdwpf.py::load` (shape assertion) | needs `code/datasets/SDWPF/data/data_20000.npz`; SHA256 recorded in `docs/DATA_PROTOCOLS.md` |
| Table 2 split `[0,14000) / [14000,15000) / [15000,16000) / [16000,20000)` | `config.json` (`split`) | asserted by `run.py preflight` |
| Table 2 window counts 3,465 / 40 / 165 and 966,746 valid points | `code/data_provider/common.py::window_starts` + `run.py::check_protocol` | preflight raises on any drift |
| Training-only z-score and training-median imputation | `code/data_provider/sdwpf.py::load` | statistics computed from `split['train']` only |
| Labels never imputed; masked loss/metrics | `code/data_provider/sdwpf.py` (`mask`) + `code/exp/evaluator.py` | |
| Stride 4 / 24 | `config.json` (`train_stride`, `evaluation_stride`) | |
| Optimizer, learning rate, weight decay, clipping, patience, epochs | `config.json` + `code/exp/trainer.py` | |
| Seeds 2024 / 2025 / 2026, mean ± sample std | `config.json` (`seeds`) + `code/exp/reporter.py` | `report` groups per model/strategy/condition |
| Corruption 50 % clean / 30 % light / 20 % heavy, 5 % and 20 % packet loss, ±8 outliers on inputs only | `code/data_provider/common.py` + `sdwpf.py::SDWPFWindows` | heavy corruption is the evaluation condition (`severity=2`) |
| Table 3 accuracy numbers | `code/results/SDWPF/<model>/<strategy>/<seed>/<condition>/metrics.json` | written by `run.py train`; aggregated by `run.py report` |
| Table 4 parameters, latency, peak memory | `run.py benchmark` (`code/exp/benchmark.py`) | batch 32, 20 warm-up iterations, 100 timed repetitions, median per checkpoint, mean over the three clean checkpoints |
| Fig. 4 error bars | `code/results/SDWPF/summary.csv` + `mae_by_model.*` | sample std over the three seeds |
| Figs. 5–8 curves, adjacency | `code/results/SDWPF/<model>/<strategy>/<seed>/{loss,adjacency,prediction_example}.*` | |

## Experiment II — Hill of Towie (Section 5)

| Paper item | Code | Reproduce / verify |
|---|---|---|
| 21 turbines × 7 variables, 4,369 points, 30-min grid, 10-min availability lag | `code/data_provider/hill21.py::prepare` | needs the public 2020 archive (Zenodo DOI in `README.md`) |
| Split `[0,3058) / [3058,3713) / [3713,4369)` | `H21/config.json` + `code/datasets/Hill21/data_audit.json` | `tools/smoke_checks.py` cross-checks them |
| Windows 729 / 26 / 26, forward fill then training-median imputation | `code/data_provider/hill21.py` | counts written to `data_audit.json` and asserted by preflight |
| 26,187 valid test points | `H21/config.json` (`expected_valid_test_points`), asserted in `run.py`/trainer | checked on every evaluation |
| Table 5 training settings | `H21/config.json` | |
| Table 6 accuracy numbers | `code/results/Hill21/<model>/<strategy>/<seed>/<condition>/metrics.json` | |
| Table 7 parameters, latency, peak memory | `run.py benchmark` | batch 1 and 32, 10 warm-up iterations, 30 timed repetitions |
| Complex parameters as two real scalars | `code/utils/params.py` | applies only to FEDformer |

## Disclosures that are deliberately in the code and not hidden

* `config.json` → `hyperparameter_provenance`: ρ was compared at 0.5 and 0.25 on the
  development test panel before being fixed at 0.25; the 0.5 runs are archived and are not
  reported.
* `config.json` → `development_panel_note`: the test period was inspected during development
  and is a development evaluation panel, not a confirmatory holdout. The same statement is
  repeated in `README.md`, `REPRODUCIBILITY.md`, `docs/DEVELOPMENT_HISTORY.md` and in the
  per-run `protocol.json`.
* `config.json` → `shuffle_each_epoch`: SDWPF used one fixed batch order on every epoch;
  Hill21 reshuffled. `run.py train --reshuffle-each-epoch` switches the former and was not
  used for the reported numbers.
* `docs/IMPLEMENTATION_NOTES.md` lists every baseline adaptation, and `MultiGSA_adapted` is
  labelled a structural adaptation, not a reproduction of the original paper's
  configuration.

## Limits of repository-only reproduction

* Raw datasets, processed caches and trained checkpoints are distributed separately (see
  `README.md`); the repository contains source code, protocol records and audit hashes only.
* Bit-identical historical scores are not promised: fresh training consumes random numbers
  differently from the original development classes even for the same seed (see
  `REPRODUCIBILITY.md`).
* GPU wall-clock numbers depend on the measurement host. The protocol (batch size, warm-up,
  repetitions, median per checkpoint) is reproducible; the absolute milliseconds are not a
  claim about other hardware.
