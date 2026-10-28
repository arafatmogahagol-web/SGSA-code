# Data and training protocols

`code/datasets/SDWPF/config.json` and `code/datasets/Hill21/config.json` are the single
source of truth for the reported settings. `code/run.py` refuses to start when the prepared
data disagrees with a configuration, so the paper, the configuration and the executed code
cannot silently diverge.

| Item | SDWPF | Hill of Towie |
|---|---|---|
| Turbines/features | 134/16 | 21/7 |
| Processed points | 20,000 | 4,369 |
| Training indices | [0,14000) | [0,3058) |
| Validation indices | [14000,15000) | [3058,3713) |
| Test indices | [16000,20000) | [3713,4369) |
| Unused interval | [15000,16000) | none |
| Input missing values | Training median | Causal forward-fill, then training median |
| Learning rate/dropout | 0.0005/0.1 | 0.001/0.1 |
| Maximum epochs/patience | 30/5 | 50/8 |
| Checkpoint selection | Validation MSE | Validation MAE |
| Train/evaluation stride | 4/24 | 4/24 |
| History/horizon | 96/48 | 96/48 |
| Windows (train/val/test) | 3,465/40/165 | 729/26/26 |
| Valid test points | 966,746 | 26,187 |

Both use 30-minute processed grids, training-only variable mean/std across time and
turbines, masked MSE training, batch 32, AdamW (weight decay 1e-4), gradient clipping 1,
and seeds 2024/2025/2026. At origin t, inputs are [t−96,t); every target label in
[t,t+48) lies within its assigned partition. A missing target is never counted in a loss
or metric: filling missing target storage with 0 does not make it a valid label, the mask
is authoritative.

## Window enumeration

`data_provider.common.window_starts` derives the origins from the configured split and
strides: a window needs a full history, so the first origin is `max(start, seq_len)`.
SDWPF enumerates every candidate test window (165, of which 153 carry at least one valid
label); Hill of Towie enumerates only windows with at least one valid label, which is how
729/26/26 are obtained. `run.py preflight` compares both counts and the valid-label total
against the configuration.

## Batch order

SDWPF keeps one fixed batch order on every epoch (`"shuffle_each_epoch": false`); Hill21
reshuffles with `seed + epoch` (`"shuffle_each_epoch": true`). The flag documents what the
reported runs used; `run.py train --reshuffle-each-epoch` switches SDWPF to a per-epoch
order, which was not used for the paper and changes the trajectory.

## Hill of Towie details

Official native timestamps are UTC and denote the END of each 10-minute interval. This
experiment adds a conservative 10-minute availability lag, then assigns records to
right-closed 30-minute mean bins. The extra lag is a protocol choice, not a claim that the
source timestamps denote interval starts. Keep this choice to reproduce the original cache.

Features, in order: `wtc_AcWindSp_mean`, `wtc_GenRpm_mean`, `wtc_MainSRpm_mean`,
`wtc_PitcPosA_mean`, `yaw_sin`, `yaw_cos`, `wtc_ActPower_mean`. Power is measured active
power, not a power setpoint. Label masks are generated from unfilled measured power;
historical inputs alone are forward-filled. Statistics and graph priors use only the
training range. `code/run.py prepare --dataset Hill21 --zip ...` writes the panel and an
audit record (`code/datasets/Hill21/data_audit.json`); it never trains a model or chooses a
configuration.

## SDWPF cache identity

SHA256: `c3b9ce258774d422ff7938d4157eb859e8724ae0960dda348d293e34af0c9a84`.
Shape `[20000,134,16]`, target `Patv`. A different cache, feature order or sampling recipe
is not automatically equivalent. `preflight` records the SHA256 of the cache it actually
read in `code/results/SDWPF/protocol.json`.

## Artificial input corruption

Clean and noise-trained models share targets and masks. Noise training samples 50 % clean,
30 % light (5 % packet loss) and 20 % heavy (20 % packet loss plus 5 %
eight-standard-deviation replacements). Packet loss uses the training median in
standardized units. The corrupted evaluation set uses the fixed heavy configuration.

The random stream that maps a window to its corruption pattern is dataset specific
(`SDWPFWindows` keys it by seed/epoch/index, `Hill21Windows` keys it by forecast origin) and
is identical for every method trained with the same strategy and seed; the exact random
arrays are not claimed to be identical across datasets.
