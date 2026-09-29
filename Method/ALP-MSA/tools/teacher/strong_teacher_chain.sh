#!/bin/bash
# After the MOSEI chain: stronger text teachers (Qwen2.5-32B/72B-Instruct-AWQ) on MOSI, semantic task, + eval.
cd "$(dirname "$0")/../.."
M=/media/disk3/muxy/Dataset/pretrained_models
T=/media/disk3/muxy/Dataset/MSA_raw_feats/MOSI/teacher
PKL=/media/disk3/muxy/Dataset/MSA_unaligned/MOSI/unaligned_50.pkl
VL=/media/disk3/muxy/envs/vllm/bin/python
until grep -q "OK Qwen/Qwen2.5-72B-Instruct-AWQ" $M/download2.log 2>/dev/null; do sleep 60; done
while tmux has-session -t mosei_chain 2>/dev/null || pgrep -f "tools/grid.py" >/dev/null; do sleep 60; done
echo "$(date +%T) strong teachers"
C="--pkl $PKL --decoded /media/disk3/muxy/Dataset/MSA_raw_feats/MOSI/decoded --task semantic --max_new_tokens 256 --chunk 256"
$VL tools/teacher/run_teacher_vllm.py --model $M/Qwen2.5-32B-Instruct-AWQ --gpu 0 --out $T/semantic_q32.jsonl $C > $T/semantic_q32.log 2>&1 &
$VL tools/teacher/run_teacher_vllm.py --model $M/Qwen2.5-72B-Instruct-AWQ --gpu 1,2 --tp 2 --gpu_util 0.92 --out $T/semantic_q72.jsonl $C > $T/semantic_q72.log 2>&1 &
wait
/media/disk3/muxy/envs/omni/bin/python tools/teacher/eval_teacher.py --pkl $PKL --jsonl $T/semantic_omni.jsonl $T/semantic_q32.jsonl $T/semantic_q72.jsonl
echo "$(date +%T) STRONG DONE"
