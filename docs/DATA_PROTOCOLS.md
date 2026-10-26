# Data and training protocols

| Item | SDWPF | Hill of Towie |
|---|---|---|
| Turbines/features |134/16|21/7|
| Processed points |20000|4369|
| Training indices |[0,14000)|[0,3058)|
| Validation indices |[14000,15000)|[3058,3713)|
| Test indices |[16000,20000)|[3713,4369)|
| Unused interval |[15000,16000)|None|
| Input missing values |Training median|Causal forward-fill, then training median|
| Learning rate/dropout |0.0005/0.1|0.001/0.1|
| Maximum epochs/patience |30/5|50/8|
| Checkpoint selection |Validation MSE|Validation MAE|
| Train/evaluation stride |4/24|4/24|
| History/horizon |96/48|96/48|

Both use30-minute processed grids, training-only variable mean/std across time and turbines, masked MSE training, batch32, AdamW(weight decay1e-4), gradient clipping1, and seeds2024/2025/2026. At origin t, inputs are[t-96,t); every target label in[t,t+48) lies within its assigned partition. A missing target is never counted in a loss or metric. Filling missing target-array storage with0 does not make it a valid label; the mask is authoritative.

## Hill of Towie details
Official native timestamps are UTC and denote the END of each10-minute interval. This experiment adds a conservative10-minute availability lag, then assigns records to right-closed30-minute mean bins. The extra lag is a protocol choice, not a claim that the source timestamps denote interval starts. Keep this choice to reproduce the original cache.

Features, in order: wtc_AcWindSp_mean,wtc_GenRpm_mean,wtc_MainSRpm_mean,wtc_PitcPosA_mean,yaw_sin,yaw_cos,wtc_ActPower_mean. Power is measured active power, not a power setpoint. Label masks are generated from unfilled measured power; historical inputs alone are forward-filled. Statistics and graph priors use only the training range. Training,validation,test window counts are729,26,26; test has26187 valid prediction points. prepare_data.py never trains a model or chooses a configuration.

## SDWPF cache identity
SHA256:c3b9ce258774d422ff7938d4157eb859e8724ae0960dda348d293e34af0c9a84 . Shape[20000,134,16],targetPatv. There are153 valid test windows and966746 valid prediction points. A different cache, feature order or sampling recipe is not automatically equivalent. The benchmark's explicit split overrides generic helper defaults; do not use preprocessing.make_splits defaults as the paper protocol.

## Artificial input corruption
Clean/noise models share targets and masks. Noise training samples50%clean,30%light(5%packet loss),20%heavy(20%packet loss plus5%8-standard-deviation replacements). Packet loss uses the training median in standardized units. Corrupted test uses the fixed heavy configuration. Dataset-specific RNG implementations are deterministic and shared across methods within a dataset; the exact random arrays are not claimed to be identical across datasets.
