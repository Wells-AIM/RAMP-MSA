#!/bin/bash
# MOSEI fine-tuned teacher: re-run the collapsed folds 0 and 2 with a lower lr (fold 1 kept), re-merge,
# then run the MOSEI teacher-C distillation queue. Old fold outputs are kept as *_v1.
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/media/disk3/muxy/envs/omni/bin/python
D=/media/disk3/muxy/Dataset/MSA_raw_feats/MOSEI
OUT=$D/teacher/omni_ft
LIGHT=$D/mosei_light.pkl
for f in 0 2; do
  mv $OUT/text_fold$f.log $OUT/text_fold${f}_v1.log 2>/dev/null
  mv $OUT/text_fold${f}of3.npz $OUT/text_fold${f}of3_v1.npz 2>/dev/null
  mv $OUT/text_fold${f}of3.json $OUT/text_fold${f}of3_v1.json 2>/dev/null
done
echo "$(date +%T) re-running MOSEI folds 0,2 (lr 1e-4, head_lr 5e-4, 3 epochs) on GPUs $GPUS"
set -- $GPUS
CUDA_VISIBLE_DEVICES=$1 $PY tools/teacher/finetune_omni.py --pkl $LIGHT --fold 0 --nfolds 3 --epochs 3 --lr 1e-4 --head_lr 5e-4 \
    --mode text --out_dir $OUT > $OUT/text_fold0.log 2>&1 &
CUDA_VISIBLE_DEVICES=$2 $PY tools/teacher/finetune_omni.py --pkl $LIGHT --fold 2 --nfolds 3 --epochs 3 --lr 1e-4 --head_lr 5e-4 \
    --mode text --out_dir $OUT > $OUT/text_fold2.log 2>&1 &
wait
grep -h "held:\|valid:\|test:" $OUT/text_fold*.log
$PY tools/teacher/merge_folds.py --dir $OUT --mode text --nfolds 3 --base $D/teacher/teacher_B.npz --pkl $LIGHT --out $D/teacher/teacher_C.npz
echo "$(date +%T) MOSEI distillation with teacher C (v2)"
/media/disk3/muxy/envs/HME/bin/python tools/grid.py exps/queue_teacherC_mosei.txt --seeds 1111,2222,3333 --per_gpu 2 --min_free_mb 10000
echo "$(date +%T) MOSEI RERUN CHAIN DONE"
