# Reproducibility status

## FEDformer
The Hill21 loading issue is resolved in the published source: training writes hash-bound Fourier metadata beside every best checkpoint; evaluation restores it strictly. Six SHA256-matched original recovery records are under experiments/Hill21/checkpoint_metadata/. A Fourier model without matching metadata raises an error instead of silently sampling different modes. Non-Fourier checkpoints still load as plain state_dict files.

The historical correction used frozen CPU inference and validation replay, not new training. The original GPU latency/memory measurements remain separate. Source training defaults still require CUDA; --evaluate-only --device cpu supports checkpoint inspection and prediction.

## Data and experimental scope
Only the134-turbine SDWPF and21-turbine Hill of Towie protocols are implemented. See docs/DATA_PROTOCOLS.md for differences, and docs/DEVELOPMENT_HISTORY.md for prior exploratory use of test periods. No test labels are used as model inputs. Graph priors and normalization are fitted on training data only.

## Random initialization and outputs
Three seeds are reported separately. Exact historical results require the original caches and checkpoints, including Fourier metadata. Fresh training from a cleaned model class can consume random numbers differently from an inherited development class; matching a seed alone does not promise bit-identical historical scores.

## Efficiency
SDWPF:20 warmup/100 repetitions. Hill21:10 warmup/30 repetitions in the existing worker process. GPU forward timing excludes host-to-device transfer. Compare within each stated protocol; parameter count alone is not a claim of minimum latency or memory.

## Release checks
Results of source, inference,checkpoint and cached-data checks are saved in provenance/release_checks.json. These checks are not a new full GPU training replication.
