#!/bin/bash
# Waits for the MOSI teacher runs, builds teacher A/B vectors, then launches the Direction 3 queue.
cd "$(dirname "$0")/../.."
T=/media/disk3/muxy/Dataset/MSA_raw_feats/MOSI/teacher
M=/media/disk3/muxy/Dataset/pretrained_models
PKL=/media/disk3/muxy/Dataset/MSA_unaligned/MOSI/unaligned_50.pkl
while tmux has-session -t t_whole 2>/dev/null || tmux has-session -t t_fac 2>/dev/null; do sleep 30; done
wc -l $T/*.jsonl
/media/disk3/muxy/envs/omni/bin/python tools/teacher/eval_teacher.py --pkl $PKL --jsonl $T/semantic_omni.jsonl $T/prosody_et.jsonl $T/facial_omni.jsonl $T/whole_et.jsonl $T/whole_omni.jsonl
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | cut -d, -f1)
/media/disk3/muxy/envs/omni/bin/python tools/teacher/embed_reasoning.py --encoder $M/bge-base-en-v1.5 --gpu $G --score_task semantic \
    --jsonl $T/whole_et.jsonl $T/semantic_omni.jsonl $T/prosody_et.jsonl $T/facial_omni.jsonl --out $T/teacher_A.npz
/media/disk3/muxy/envs/omni/bin/python tools/teacher/embed_reasoning.py --encoder $M/bge-base-en-v1.5 --gpu $G --score_task semantic \
    --jsonl $T/whole_omni.jsonl $T/semantic_omni.jsonl $T/prosody_et.jsonl $T/facial_omni.jsonl --out $T/teacher_B.npz
/media/disk3/muxy/envs/HME/bin/python tools/grid.py exps/queue_d3.txt --seeds 1111,2222,3333,4444,5555 --per_gpu 2 --min_free_mb 10000
