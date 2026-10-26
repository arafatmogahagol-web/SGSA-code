# Development history and interpretation

The public execution path prepares data, trains fixed configurations and selects checkpoints with validation metrics. It does not implement an answer-based oracle search.

Before the final Hill21 benchmark, SGSA and MultiGSA_adapted were explored on a six-candidate learning-rate/dropout grid using seed2024. The saved validation winner was learning_rate0.001/dropout0.1. The processed test period was also inspected during development. SDWPF underwent multiple architecture/configuration explorations. Consequently, the existing test periods are development comparisons, not newly sealed confirmatory holdouts.

An old preprocessing metadata field named oracle described that development history rather than a preprocessing operation. Its original text is retained in provenance/development_history.json. Data preprocessing metadata now documents only data transformations. Removing a stale field does not erase the development history or establish statistical independence.

Hill21 FEDformer frequency metadata was recovered on2026-10-22 from original training logs. All six checkpoint hashes matched; original best validation MAE was reproduced with a maximum absolute gap3.267825154917858e-8. Recovered test labels,masks,origins and turbine IDs matched the original panel. This was frozen re-evaluation, not unfreezing or retraining.
