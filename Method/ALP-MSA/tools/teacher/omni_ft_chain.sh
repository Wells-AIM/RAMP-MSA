#!/bin/bash
# Step 4: fine-tuned Omni teacher (5-fold cross-fitting, text mode) -> teacher_C.npz -> distillation queue.
# Waits for the strong-teacher chain and for any grid.py so it has all three GPUs.
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/media/disk3/muxy/envs/omni/bin/python
OUT=/media/disk3/muxy/Dataset/MSA_raw_feats/MOSI/teacher/omni_ft
T=/media/disk3/muxy/Dataset/MSA_raw_feats/MOSI/teacher
PKL=/media/disk3/muxy/Dataset/MSA_unaligned/MOSI/unaligned_50.pkl
MODE=${MODE:-text}
mkdir -p $OUT
while tmux has-session -t strong_chain 2>/dev/null || tmux has-session -t mosei_chain 2>/dev/null \
      || pgrep -f "tools/grid.py" >/dev/null; do sleep 60; done
echo "$(date +%T) fine-tuning Omni teacher, mode=$MODE"
for f in 0 1 2; do CUDA_VISIBLE_DEVICES=$f $PY tools/teacher/finetune_omni.py --fold $f --nfolds 5 --mode $MODE --out_dir $OUT > $OUT/${MODE}_fold$f.log 2>&1 & done
wait
for f in 3 4; do CUDA_VISIBLE_DEVICES=$((f-3)) $PY tools/teacher/finetune_omni.py --fold $f --nfolds 5 --mode $MODE --out_dir $OUT > $OUT/${MODE}_fold$f.log 2>&1 & done
wait
grep -h "held:\|valid:\|test:" $OUT/${MODE}_fold*.log
$PY tools/teacher/merge_folds.py --dir $OUT --mode $MODE --nfolds 5 --base $T/teacher_B.npz --pkl $PKL --out $T/teacher_C.npz
echo "$(date +%T) distillation with teacher C"
/media/disk3/muxy/envs/HME/bin/python tools/grid.py exps/queue_teacherC.txt --seeds 1111,2222,3333,4444,5555 --per_gpu 2 --min_free_mb 10000
echo "$(date +%T) MOSEI distillation (teacher B)"
/media/disk3/muxy/envs/HME/bin/python tools/grid.py exps/queue_d3_mosei_later.txt --seeds 1111,2222,3333 --per_gpu 2 --min_free_mb 10000
echo "$(date +%T) OMNI FT CHAIN DONE"
