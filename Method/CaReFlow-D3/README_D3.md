# CaReFlow + Direction 3 (reasoning distillation as a training-only plug-in)

Official CaReFlow code (https://github.com/TmacMai/CaReFlow, commit 5c9f9c7, CVPR 2026) with one addition:
training-only distillation of a multimodal LLM teacher's emotion-evidence reasoning (see `Method/ALP-MSA`,
`tools/teacher/`). Inference is unchanged (no teacher at test time).

Changes vs. the official code
- `model_reflow_new.py`: optional `return_aux=True` returns the fused feature and the pooled T/A/V vectors.
- `train_reflow_new.py`: `--teacher_path/--w_label/--w_reason/--w_fields/--tau/--distill_lr/--result_json`;
  teacher targets are attached to TRAIN samples only; the heads use a forked RNG so data order / dropout of a
  seed are identical with and without distillation. With all weights 0 the code path equals the official one.
- `distill.py`: copied from `ALP-MSA/alp/distill.py`.
- `map_ids.py`: CaReFlow segment id `vid[k]` -> MMSA id `vid$_$j` (video + transcript + label, fuzzy fallback);
  MOSI coverage 1269/1281 train, no cross-split matches.

Data: `datasets/mosi.pkl` -> `/media/disk3/muxy/Dataset/careflow_mosi_aligned.pkl` (sha256 5c3cc6ab...,
same file as the official-protocol reproduction). Runs are launched via `ALP-MSA/exps/queue_careflow_d3.txt`.
