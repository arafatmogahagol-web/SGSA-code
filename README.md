# SGSA: two-dataset research package

SGSA uses a learned static turbine graph, a self-feature residual (rho=0.25), position encoding, temporal SimAM statistics and a direct multi-step prediction head. Both datasets use d=64, graph rank=16, history=96 and horizon=48. Datasets are trained separately.

## Layout
- `experiments/SDWPF`: 134 turbines, 16 variables; historical protocol and adapted baselines.
- `experiments/Hill21`: 21 turbines, 7 variables; January–March 2020.
- `vendor`: author implementations with their existing license files.
- `provenance`: source records and packaged-file hashes.

## Environment and data
Use Linux with NVIDIA CUDA and a compatible PyTorch installation. Install PyTorch for the installed CUDA driver, then `pip install -r requirements.txt`. These are dependency names, not a claim that every future version is compatible. The source is portable, but no new training was performed while assembling this release.

Raw data and checkpoints are not embedded in this code package. SDWPF source: https://doi.org/10.1038/s41597-024-03427-5 . Hill of Towie: https://doi.org/10.5281/zenodo.14870023 . Respect the original data licenses.

SDWPF requires the audited historical `data_20000.npz` cache at `experiments/SDWPF/data/data_20000.npz`. Do not silently regenerate it using a different downsampling or feature order. Its split and preprocessing are in config.json and data_loader.py. Hill21 can be prepared from the 2020 source ZIP containing tblSCTurbine and tblSCTurGrid monthly CSVs:

```sh
python experiments/Hill21/code/prepare_data.py /path/to/2020.zip
python experiments/Hill21/code/prepare_priors.py
python experiments/Hill21/code/worker.py SGSA
```

The Hill21 worker trains three seeds with clean/noise training and evaluates clean/corrupted inputs. Substitute `MultiGSA_adapted`, `AGCRN`, etc. for SGSA. FEDformer is blocked pending the audit described below.

For SDWPF, run `code/run_suite.py preflight` from the experiment folder before `train --model SGSA --strategy clean --seed 2024`. The original six-model runner and the separate extended/extra adapter files are preserved; this release does not pretend the six-model queue automatically covers every adapter. `sgsa_train.py` and `sgsa_evaluate.py` provide explicit SGSA entry points; use `--help` for required file paths. Training and inference outputs go into dataset-local directories.

## Reproducibility status
Read REPRODUCIBILITY.md before comparing results. The paper reports existing audited caches; moving files to this repository is not a new replication. Model-specific adapters are part of the experimental method, not exact reproductions of every original paper. Do not overwrite existing outputs.

## Licenses
Third-party implementations retain upstream notices. No blanket relicensing is asserted. Choose a license for original SGSA code before public release and retain all upstream licenses and citations.
