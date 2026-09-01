#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/disk3/muxy/QA/PDER_MOSI_MOSEI_20260829}"
RESULT_ROOT="${RESULT_ROOT:-/media/disk3/muxy/QA/experiments/pder_mosi_mosei_generalization_20260829}"
PYTHON="${PYTHON:-/home/muxingyu/.conda/envs/HME/bin/python}"
DATA_ROOT="/media/disk3/muxy/Dataset/Method/Method"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/pilot_full_ramped"
if [[ ! -f "${RESULT_ROOT}/pilot_full_ramped/ARCHIVED" ]]; then
  cp -a "${RESULT_ROOT}/mosi/full" "${RESULT_ROOT}/pilot_full_ramped/mosi_full"
  cp -a "${RESULT_ROOT}/mosei/full" "${RESULT_ROOT}/pilot_full_ramped/mosei_full"
  date -Iseconds >"${RESULT_ROOT}/pilot_full_ramped/ARCHIVED"
fi

run_full() {
  local dataset="$1" seed="$2" gpu="$3" epochs="$4" min_epochs="$5" patience="$6"
  local output_dir="${RESULT_ROOT}/${dataset}/full/seed_${seed}"
  local log_path="${RESULT_ROOT}/logs/${dataset}_full_strength_seed_${seed}_gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. PYTHONUNBUFFERED=1 "${PYTHON}" \
    scripts/run_sentiment_generalization.py \
    --dataset "${dataset}" \
    --data "${DATA_ROOT}/aligned_${dataset}_ramp_bert.pkl" \
    --variant full \
    --seed "${seed}" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --epochs "${epochs}" \
    --min-epochs "${min_epochs}" \
    --patience "${patience}" \
    --batch-size 96 \
    --lr 3e-4 \
    --refinement-start-epoch 0 \
    --refinement-ramp-epochs 0 \
    >"${log_path}" 2>&1
}

worker() {
  local seed="$1" gpu="$2"
  run_full mosi "${seed}" "${gpu}" 30 12 8
  run_full mosei "${seed}" "${gpu}" 12 1 4
}

worker 1701 0 & pid0=$!
worker 1702 1 & pid1=$!
worker 1703 2 & pid2=$!
wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
printf '{"status":"complete","refinement_start_epoch":0,"refinement_ramp_epochs":0,"finished_at":"%s"}\n' \
  "$(date -Iseconds)" >"${RESULT_ROOT}/FULL_STRENGTH_COMPLETE.json"
