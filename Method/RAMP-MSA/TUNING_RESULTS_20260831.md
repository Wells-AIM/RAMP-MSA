# RAMP-MSA missing-modality tuning report

Date: 2026-08-31  
Seed: 1111  
Protocol: deterministic exact-rate missingness; MOSI 0.2, MOSEI 0.1.

All tuning candidates were selected using validation `Acc2_non0` (higher is
better), with MAE and Corr used as secondary diagnostics. Test evaluation was
disabled for every tuning run. Only the validation-locked final configuration
for each dataset was evaluated on the test split.

## MOSI validation tuning

### Backbone (memory disabled)

| Candidate | Acc2-nonzero | MAE | Corr |
|---|---:|---:|---:|
| original parameters, Acc2 selection | **0.7870** | **1.0015** | **0.6252** |
| lr 1e-4 | 0.7639 | 1.0557 | 0.6122 |
| lr 1e-4, dropout 0.3 | 0.7639 | 1.1578 | 0.5424 |
| lr 3e-4 | 0.7778 | 1.0249 | 0.6176 |
| dropout 0.1 | **0.7870** | 1.0581 | 0.6239 |
| SmoothL1 beta 1.0 | 0.7824 | 1.0301 | 0.6136 |

### Procedural memory

| Candidate | Acc2-nonzero | MAE | Corr | Mean gate |
|---|---:|---:|---:|---:|
| memory disabled | 0.7870 | **1.0015** | **0.6252** | 0.000 |
| default memory | **0.7917** | 1.0338 | 0.6212 | 0.321 |
| weak memory | **0.7917** | 1.0379 | 0.6208 | 0.513 |
| static weak memory | **0.7917** | 1.0291 | 0.6212 | 0.479 |

The validation-locked choice was static weak memory: policy strength 0.05,
policy loss weight 0.05, and online memory updates disabled.

## MOSEI validation tuning

### Backbone (memory disabled)

| Candidate | Acc2-nonzero | MAE | Corr |
|---|---:|---:|---:|
| original parameters, Acc2 selection | 0.8303 | **0.5491** | **0.6904** |
| lr 1e-4 | 0.8261 | 0.5609 | 0.6825 |
| dropout 0.1 | **0.8338** | 0.5747 | 0.6897 |

### Procedural memory with dropout 0.1

| Candidate | Acc2-nonzero | MAE | Corr | Mean gate |
|---|---:|---:|---:|---:|
| memory disabled | **0.8338** | 0.5747 | **0.6897** | 0.000 |
| default memory | 0.8289 | 0.5622 | 0.6895 | 0.161 |
| weak memory | 0.8296 | 0.5780 | 0.6835 | 0.176 |
| static weak memory | 0.8296 | **0.5586** | 0.6877 | 0.213 |

The validation-locked choice was dropout 0.1 with procedural memory disabled.

## Final locked test results

| Dataset / selected model | Acc2-nonzero | F1-nonzero | MAE | Corr | Acc7 | Acc5 |
|---|---:|---:|---:|---:|---:|---:|
| MOSI static weak memory | 0.7256 | 0.7269 | 1.1022 | 0.5836 | 0.3309 | 0.3921 |
| MOSEI dropout 0.1, no memory | 0.8327 | 0.8314 | 0.5934 | 0.7073 | 0.5143 | 0.5299 |

For MOSI, tuning raised the previous full-RAMP test Acc2 from 0.6936 to
0.7256, but the final memory correction had negative mean loss gain
(-0.000744) and did not change binary predictions relative to its own base
branch. The improvement primarily came from checkpoint selection and weaker,
static memory training, not from inference-time memory correction.

For MOSEI, memory did not pass validation selection. The tuned no-memory model
raised Acc2 from 0.8310 to 0.8327 and Corr from 0.6945 to 0.7073, while MAE
increased from 0.5871 to 0.5934.

## Locked artifacts

- `configs/mosi_missing_tuned.yaml`
- `configs/mosei_missing_tuned.yaml`
- `runs/tune_mosi_mem_static_weak/best.pt`
- `runs/tune_mosi_mem_static_weak/selected_test_metrics.json`
- `runs/tune_mosei_drop01/best.pt`
- `runs/tune_mosei_drop01/selected_test_metrics.json`
