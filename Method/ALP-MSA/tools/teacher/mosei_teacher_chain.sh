#!/bin/bash
# MOSEI Direction-3 teacher pipeline, fully unattended:
#   light pkl (ids/text/labels) -> wait for Raw.zip -> decode clips -> wait for GPU jobs ->
#   vLLM teachers (whole_omni / semantic / prosody / facial) -> teacher_B vectors -> MOSEI distillation queue
cd "$(dirname "$0")/../.."
D=/media/disk3/muxy/Dataset/MSA_raw_feats/MOSEI
T=$D/teacher
M=/media/disk3/muxy/Dataset/pretrained_models
LIGHT=$D/mosei_light.pkl
mkdir -p $T
HME=/media/disk3/muxy/envs/HME/bin/python
VL=/media/disk3/muxy/envs/vllm/bin/python

echo "$(date +%T) light pkl"
[ -f $LIGHT ] || $HME - <<EOF
import pickle
d = pickle.load(open('/media/disk3/muxy/Dataset/MSA_unaligned/MOSEI/unaligned_50.pkl', 'rb'))
light = {s: {k: d[s][k] for k in ('id', 'raw_text', 'regression_labels')} for s in d}
pickle.dump(light, open('$LIGHT', 'wb'))
print({s: len(light[s]['id']) for s in light})
EOF

echo "$(date +%T) waiting for Raw.zip"
until grep -q "EXIT=" $D/dl.log 2>/dev/null; do sleep 60; done
grep "EXIT=0" $D/dl.log || { echo "download failed"; exit 1; }
ls -l $D/Raw.zip

echo "$(date +%T) decoding"
/media/disk3/muxy/envs/RoboTwin/bin/python tools/raw/decode_raw.py --pkl $LIGHT --zip $D/Raw.zip \
    --out $D/decoded --workers 16 > $D/decode.log 2>&1
tail -2 $D/decode.log

echo "$(date +%T) waiting for training jobs to finish"
while pgrep -f "^$HME tools/grid.py" >/dev/null; do sleep 60; done

echo "$(date +%T) teachers"
C="--pkl $LIGHT --decoded $D/decoded --chunk 256"
$VL tools/teacher/run_teacher_vllm.py --model $M/Qwen2.5-Omni-7B --task whole    --gpu 0 --max_new_tokens 400 --out $T/whole_omni.jsonl    $C > $T/whole_omni.log 2>&1 &
( $VL tools/teacher/run_teacher_vllm.py --model $M/Qwen2.5-Omni-7B --task semantic --gpu 1 --max_new_tokens 200 --out $T/semantic_omni.jsonl $C > $T/semantic_omni.log 2>&1
  $VL tools/teacher/run_teacher_vllm.py --model $M/EmotionThinker   --task prosody  --gpu 1 --max_new_tokens 320 --out $T/prosody_et.jsonl    $C > $T/prosody_et.log 2>&1 ) &
$VL tools/teacher/run_teacher_vllm.py --model $M/Qwen2.5-Omni-7B --task facial   --gpu 2 --max_new_tokens 256 --out $T/facial_omni.jsonl  $C > $T/facial_omni.log 2>&1 &
wait
wc -l $T/*.jsonl

echo "$(date +%T) eval + vectors"
/media/disk3/muxy/envs/omni/bin/python tools/teacher/eval_teacher.py --pkl $LIGHT \
    --jsonl $T/semantic_omni.jsonl $T/prosody_et.jsonl $T/facial_omni.jsonl $T/whole_omni.jsonl
/media/disk3/muxy/envs/omni/bin/python tools/teacher/embed_reasoning.py --encoder $M/bge-base-en-v1.5 --gpu 0 \
    --score_task semantic --jsonl $T/whole_omni.jsonl $T/semantic_omni.jsonl $T/prosody_et.jsonl $T/facial_omni.jsonl \
    --out $T/teacher_B.npz

echo "$(date +%T) MOSEI distillation queue"
$HME tools/grid.py exps/queue_d3_mosei.txt --seeds 1111,2222,3333 --per_gpu 2 --min_free_mb 10000
echo "$(date +%T) CHAIN DONE"
