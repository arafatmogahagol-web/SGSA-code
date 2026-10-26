# SGSA: Static Graph–Statistical Attention

Research code for lightweight multi-turbine wind power forecasting. SGSA combines a learned static turbine graph, a scaled self-feature residual (rho=0.25), learnable positional encoding, temporal SimAM weighting, and a shared direct prediction head. History=96, horizon=48, hidden width=64, graph rank=16. Each dataset is trained separately.

## Repository layout
- `experiments/SDWPF/`: 134 turbines, 16 variables; configuration and all benchmark adapters.
- `experiments/Hill21/`: Hill of Towie,21 turbines,7 variables; January–March2020.
- `sgsa_utils/checkpoints.py`: strict checkpoint loading and hash-bound Fourier-index metadata.
- `vendor/`: licensed third-party implementations with their original notices.
- `docs/`: data protocols, implementation adaptations and development history.

## Installation
Use a separate Python environment. GPU training uses NVIDIA CUDA; install a matching PyTorch build, then run `pip install -r requirements.txt`. The original measurements used an RTX3080Ti. Consult `docs/DATA_PROTOCOLS.md` for the different dataset protocols.

## Data preparation
Raw data and checkpoints are distributed separately from source code.

**Hill of Towie:** obtain the2020 archive from https://doi.org/10.5281/zenodo.14870023 . The six monthly CSVs used here are tblSCTurbine and tblSCTurGrid for January–March2020. They may remain in the full-year ZIP or in a ZIP containing only those six files.

```sh
python experiments/Hill21/code/prepare_data.py /path/to/2020.zip
python experiments/Hill21/code/prepare_priors.py
python experiments/Hill21/code/worker.py SGSA
```

The worker uses fixed settings from config.json, three seeds and clean/noise training. No hyperparameter search runs as part of preprocessing or the worker. Substitute another model name, including FEDformer, to run that benchmark. To evaluate existing checkpoints without training:

```sh
python experiments/Hill21/code/worker.py FEDformer --evaluate-only --device cpu
```

Place checkpoints and their run files under `experiments/Hill21/runs/<model>/<clean|noise>/<seed>/`. New checkpoints include `best.pt.metadata.json`. Six original FEDformer checkpoints can also load with the supplied SHA256-matched records in checkpoint_metadata/. Do not load an arbitrary FEDformer state_dict with newly randomized frequency modes.

**SDWPF:** source https://doi.org/10.1038/s41597-024-03427-5 . Place the audited processed cache at `experiments/SDWPF/data/data_20000.npz`. The release identifies its hash and protocol; it does not substitute an unverified raw-to-cache recipe. Then:

```sh
python experiments/SDWPF/code/prepare_priors.py
python experiments/SDWPF/code/run_suite.py preflight
python experiments/SDWPF/code/run_suite.py train --model SGSA --strategy clean --seed 2024
```

Preflight discovers a feasible microbatch while preserving32 windows per optimizer step through gradient accumulation. The queue covers13 methods and78 training runs. Extended model adapters retain their documented differences from the original papers. For StemGNN only, first run `python tools/fetch_stemgnn.py`; its upstream source is fetched locally and is not redistributed here.

## Reproducibility
See REPRODUCIBILITY.md. Both experiments use chronological partitions, training-only normalization, finite-target masks, and input-only artificial corruption. Their preprocessing details and training settings differ. Existing data splits were inspected during development; this repository does not claim they are untouched confirmatory test sets.

## Licensing
No additional open-source license has currently been granted for repository-owned code. Public visibility alone is not a blanket permission to reuse it. Third-party components retain the licenses listed in THIRD_PARTY_NOTICES.md; preserve those notices. Dataset licenses are separate from code licenses. No third-party implementation is presented as original SGSA work.
