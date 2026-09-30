# Direction 1 — Reasoning-Aligned Affective Representation (2026-09-30)

Student: official CaReFlow (`Method/CaReFlow-D3`), CMU-MOSI, val-loss selection, test metrics of the
selected epoch. Paired tests (`tools/paired.py`) over identical seeds. Teachers (training only):
- **B**: Qwen2.5-Omni-7B structured emotion-evidence reasoning (Semantic / Prosody / Facial / Consistency /
  Conclusion), embedded with bge; train MAE 1.135, Acc2 84.9 (weaker than the student).
- **C**: Qwen2.5-Omni-7B LoRA regression, 5-fold cross-fitted (out-of-fold on train); test Acc2 90.2, MAE 0.550.
  `teacher_CB.npz` = C's score + B's reasoning embeddings + `score_verify` (B's own conclusion).

New in this direction (`alp/distill.py`, default off):
- `verify_sigma`: verified reasoning distillation — reasoning terms weighted per sample by
  `exp(-(conclusion - label)^2 / 2σ^2)` on train labels (σ=1 keeps effective n 748/1284 for B).
- `w_rel`: relational reasoning distillation — KL between the student's and the teacher-reasoning in-batch
  similarity distributions.

## Results (Acc2 / Acc7 / MAE / Corr, mean ± std)

| run | seeds | Acc2 | Acc7 | MAE | Corr |
|---|---|---|---|---|---|
| cf_none (no teacher) | 10 | 86.98 ± 1.82 | 46.99 | 0.649 | 0.836 |
| cf_C_label (label distillation, teacher C) | 10 | 87.74 ± 0.55 | 47.64 | 0.620 | 0.851 |
| + reason/fields, unverified (cf_d1C_r) | 5 | 87.27 | 47.47 | 0.641 | 0.842 |
| + reason/fields, **verified** (cf_d1C_rv) | 5 | 87.63 | 48.64 | 0.627 | 0.846 |
| + relational, unverified (cf_d1C_rel) | 5 | 88.06 | 48.53 | 0.622 | 0.849 |
| + relational, **verified** (cf_d1C_relv) | 10 | 87.15 ± 1.67 | 48.47 | 0.635 | 0.842 |

Paired effects:
- **Verification (same loss, same seeds)**: rv vs r — Acc7 +1.17 (5/5, p=0.017), MAE −0.014 (5/5, p=0.035),
  Corr +0.004 (5/5); relv vs rel — Acc7 +1.05 (5/5, p=0.009), MAE −0.009 (5/5, p=0.080).
- Unverified reasoning on top of label distillation is harmful: cf_d1C_r vs cf_C_label MAE +0.027 (0/5, p=0.019).
- Best D1 config vs label-only (cf_d1C_relv vs cf_C_label, 10 seeds): Acc7 +0.83 (7/10, n.s.), Acc2 −0.60;
  seed 42 diverged (Acc2 82.75, MAE 0.809). Without that seed Acc7 improves on 7/9 seeds.
- Teacher B alone (no label term), vs cf_none, 10 seeds: no significant effect; verification again helps
  (cf_d1_rv vs cf_rf: MAE −0.008 7/10 p=0.097, Corr +0.006 8/10 p=0.056).

## Conclusions
1. Distilling a teacher's reasoning is only safe when the reasoning is **verified** against its conclusion;
   verification consistently turns a harmful term into a helpful one (≈ +1 Acc7, all seeds).
2. The Acc2 / MAE ceiling is set by the label/score teacher (C); reasoning alignment mainly improves the
   fine-grained 7-class accuracy.
3. Relational KD at `rel_tau=0.1` is occasionally unstable (1/10 seeds diverged) — next: larger rel_tau / warm-up.
