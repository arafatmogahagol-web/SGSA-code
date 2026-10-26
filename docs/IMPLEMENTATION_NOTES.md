# Implementation adaptations

- MultiGSA_adapted is a multi-turbine adaptation, not an exact reproduction of its paper's original datasets and architecture settings.
- STID uses node identity but no extra time-of-day/day-of-week inputs.
- DLinear and PatchTST operate turbine-wise with shared parameters; the target power channel is selected for evaluation.
- PDFormer uses training-only power correlations, DTW profiles, graph positions and pattern priors; no geographic coordinates or test-fitted graph is claimed.
- StemGNN uses the historical power channel only. Its removed torch.rfft/irfft operations are replaced in the adapter by equivalent two-sided torch.fft transforms. Fetch the original file separately; its redistribution license was not established.
- GraphWaveNet's historical four-dimensional convolution input is handled with Conv2d in the adapter on current PyTorch.
- FEDformer frequency lists are part of checkpoint identity even though upstream does not register them as state_dict buffers. The shared loader restores them before inference and checks the weight hash. Six original Hill21 recovery records are included.
- All third-party directories retain their own notices. Original repository paths in the provenance manifest have been reduced to relative source names.

This release corrects packaging/import/checkpoint-loading problems. It does not modify the SGSA computational graph or retrospectively retune the experiments.
