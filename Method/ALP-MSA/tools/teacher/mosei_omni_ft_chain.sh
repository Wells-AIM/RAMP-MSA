#!/bin/bash
# MOSEI fine-tuned Omni teacher (3-fold cross-fitting, text, 2 epochs) -> MOSEI teacher_C -> ALMT distillation.
# Starts after the Direction-1 session's queue (logs_d1b.txt: D1B_ALL_DONE) and when no grid.py is running.
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/media/disk3/muxy/envs/omni/bin/python
D=/media/disk3/muxy/Dataset/MSA_raw_feats/MOSEI
OUT=$D/teacher/omni_ft
LIGHT=$D/mosei_light.pkl
mkdir -p $OUT
until grep -q "D1B_ALL_DONE" logs_d1b.txt 2>/dev/null; do sleep 120; done
while pgrep -f "^/media/disk3/muxy/envs/HME/bin/python tools/grid.py" >/dev/null; do sleep 60; done   # anchored: the tmux server cmdline also contains "tools/grid.py"
echo "$(date +%T) MOSEI fine-tuning (3 folds)"
for f in 0 1 2; do CUDA_VISIBLE_DEVICES=$f $PY tools/teacher/finetune_omni.py --pkl $LIGHT --fold $f --nfolds 3 --epochs 2 \
    --mode text --out_dir $OUT > $OUT/text_fold$f.log 2>&1 & done
wait
grep -h "held:\|valid:\|test:" $OUT/text_fold*.log
$PY tools/teacher/merge_folds.py --dir $OUT --mode text --nfolds 3 --base $D/teacher/teacher_B.npz --pkl $LIGHT --out $D/teacher/teacher_C.npz
echo "$(date +%T) MOSEI distillation with teacher C"
/media/disk3/muxy/envs/HME/bin/python tools/grid.py exps/queue_teacherC_mosei.txt --seeds 1111,2222,3333 --per_gpu 2 --min_free_mb 10000
echo "$(date +%T) MOSEI FT CHAIN DONE"
