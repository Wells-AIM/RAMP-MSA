# CaReFlow + Emotion Error Recycling (Direction 2.5)

Official CaReFlow code (CVPR 2026, commit 5c9f9c7; copied from `Method/CaReFlow-D3`, which adds the
training-only Direction-3 distillation hooks). Direction 2.5 of the ICLR2026 method summary:

> After a first training stage a model has stable error patterns (positive -> neutral, weak negative -> neutral ...).
> Store them in an Emotion Confusion Bank (x, y, y_hat, confidence) and re-train on them, including under
> injected disturbances, so that the model learns to correct its own mistakes.

## Method

**Stage 1 — cross-fitted Emotion Confusion Bank.** A model's error on its own training samples is ~0 (training
loss goes to ~0), so the stable error pattern is measured out of fold: the TRAIN split is cut into K=5 folds,
CaReFlow is trained on 4 folds (dev split still selects the epoch) and predicts the held-out fold
(`--oof_fold k`). `tools/build_error_bank.py` merges the folds:

| field | meaning |
|---|---|
| `pred`, `confidence=|pred|` | out-of-fold prediction, distance to the polarity boundary |
| `etype` | 0 correct · 1 intensity error (right polarity, \|e\|≥1) · 2 polarity flip, low margin · 3 polarity flip, high confidence |
| `severity` | min(1, \|e\|/2 + 0.5·flip) |
| `in_bank` | etype>0, or a low-margin (\|pred\|<0.2) correct sample |

**Stage 2 — targeted retraining** (`er.py`, all terms training-only, inference unchanged):

1. bank re-weighting `w = 1 + α·severity` (normalised to batch mean 1) — `--er_weight`
2. confusion margin `relu(m − sign(y)·f(x))` on stage-1 polarity flips — `--er_margin`, `--er_margin_m`
3. error injection: bank samples get a perturbed view — audio degradation, visual occlusion, text masking,
   modality dropout, temporal jitter — that must still be predicted correctly — `--er_perturb`, `--er_pert_kinds`

RQ3 baselines: `--focal_gamma` (focal-style regression weighting), `--ohem_frac` (online hard example mining),
`--er_pert_all` (the same perturbations on random instead of bank samples). With no Direction-2.5 flag the
training path is exactly the official one.

## Protocol

Official command and hyper-parameters (as `cf_none` in `ALP-MSA/exps/queue_careflow_d3.txt`), official
val-loss epoch selection, seeds 1111..5555 paired with `cf_none` (`ALP-MSA/tools/paired.py`).

```bash
cd /media/disk3/muxy/Method/ALP-MSA
python tools/grid.py ../CaReFlow-ER/exps/stage1_oof.txt --seeds 1111          # 5 folds
python ../CaReFlow-ER/tools/build_error_bank.py /media/disk3/muxy/Dataset/error_bank/mosi_cf_oof5_s1111.npz \
       runs/er_oof_f*/seed_1111/oof.npz
python tools/grid.py ../CaReFlow-ER/exps/stage2_rq3.txt --seeds 1111,2222,3333,4444,5555
python tools/paired.py runs cf_none er_focal er_ohem er_bank er_bankm er_full er_randpert
```
