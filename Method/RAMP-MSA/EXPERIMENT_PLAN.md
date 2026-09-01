# RAMP-MSA Experiment Plan

## Stage A — Falsify the core mechanism quickly

Run MOSI with 5 seeds for:

1. Base (`memory.enabled=false`)
2. Full RAMP
3. Static initialized memory (`memory.online_update=false`)
4. No novelty (`write_novelty_power=0`)
5. No stability (`write_stability_power=0`)
6. Fast writer (`writer_momentum=0`)

Primary decision criterion: full RAMP must improve the high-interaction subset consistently, not only average MAE.

## Stage B — Benchmark table

### MOSI / MOSEI

Report MAE, Corr, Acc-7, Acc-5, Acc-3, Acc-2, F1 under the same feature/split protocol as the compared methods.

### IEMOCAP

Report WA/Accuracy, Macro-F1 or the exact metrics used by the chosen split protocol. State class definitions and session split explicitly.

## Stage C — Required paper ablations

- no memory
- class-prototype memory baseline
- raw-example kNN retrieval baseline
- MLP/hypernetwork policy predictor instead of external memory
- static procedural memory
- no paired utility credit
- FIFO replacement
- no hardness
- no novelty
- no stability
- no redundancy term
- fast writer vs EMA writer
- different number of regimes
- different memory capacity
- different top-R / top-K retrieval
- policy action space: unimodal-only 3-D vs pairwise 6-D vs full 7-D

The class-prototype, raw-kNN and hypernetwork baselines are especially important for reviewer differentiation. They are not yet implemented in this first package and should be added after the main model is validated.

## Stage D — Robustness

For each modality independently:

- complete removal
- Gaussian corruption at several strengths
- random temporal dropout

Use the same trained model; do not retrain for each corruption unless explicitly evaluating adaptation.

## Stage E — Low-resource

Train with 10%, 25%, 50%, 100% of the training set using identical validation/test splits. The memory hypothesis predicts stronger relative gains in limited-data / long-tail conditions.

## Stage F — Interaction diagnostics

- low/mid/high Möbius interaction difficulty
- route/regime usage distribution
- memory utility histogram
- policy component histograms `[T,A,V,TA,TV,AV,TAV]`
- retrieval entropy vs memory gain
- qualitative cases where base and RAMP disagree

## Stage G — Statistical protocol

Use at least 5 random seeds for MOSI. Report mean ± std and paired significance where appropriate. MOSI is small enough that single-seed improvements are not convincing.

## Stage H — Efficiency

Report:

- trainable parameters
- memory size in KB/MB
- training throughput (oracle policy adds training-only cost)
- inference throughput

At inference RAMP does not compute oracle gradients; it only computes the seven coalition states, memory retrieval and a residual correction.
