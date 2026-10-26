# Reproducibility and implementation audit

## Frozen scope
Only SDWPF (134 turbines) and Hill of Towie (21 turbines) are included. SGSA rho=0.25, d=64. Main tables report three separate seeds (2024/2025/2026), not the best seed. Data masks exclude missing target values. Artificial corruption modifies inputs only.

## FEDformer replay
The official Fourier layers keep random frequency indices in ordinary Python lists rather than state_dict buffers. The historical Hill21 worker rebuilt a model before loading weights without restoring these lists. Its stored test accuracy is therefore not used as a valid main comparison; raw numbers remain in the evidence archive. This package blocks this Hill21 entry rather than repeating the problem. Re-evaluation requires the original training indices or a separately authorized retraining; a fresh model with arbitrary frequency lists is not an equivalent checkpoint.

The SDWPF six-model runner reloads the best state into the same model instance and saves/checks frequency-index metadata for resume. The Hill21 issue must not be indiscriminately applied to SDWPF results.

## Historical protocol
The SDWPF split is train [0,14000), validation [14000,15000), unused [15000,16000), test [16000,20000). Hill21 uses calendar-ordered 70/15/15 splitting. Both are previously inspected development test panels, not newly sealed confirmatory holdouts. Differences between datasets must not be attributed solely to turbine count.

## Efficiency
SDWPF uses 20 warmup/100 timed repeats; Hill21 uses 10/30. Results are GPU forward timings excluding transfer. Hill21 peak allocation is measured in the existing worker process. Compare methods within each stated protocol; parameter count alone does not imply minimum memory or latency.

## Packaging changes
Absolute machine paths were made relative, dependency folders consolidated, an invalid adapter dictionary produced during packaging was corrected, and the unsafe Hill21 FEDformer entry was blocked. Architecture/training settings were not tuned. Source parsing was inspected; no new training or full end-to-end replication was executed.


## Recovery completed 2026-10-22
All six Hill21 FEDformer original frequency maps were recovered from FEDformer.retry.log and independently replayed with the original NumPy seeds. Original validation MAE was reproduced (maximum absolute gap < 3.3e-8) on CPU PyTorch2.6.0. All original checkpoint hashes and test labels/masks/indices matched. Recovered predictions and metrics are in results/exp2/fedformer_frequency_recovery_20261022. The unsafe historical worker remains blocked; use the recovery script/metadata for frozen evaluation. No retraining was performed. Prior pending-accuracy notes above describe the superseded pre-recovery status.
