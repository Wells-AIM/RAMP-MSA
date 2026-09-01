#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/disk3/muxy/QA/PDER_MOSI_MOSEI_20260829}"
RESULT_ROOT="${RESULT_ROOT:-/media/disk3/muxy/QA/experiments/pder_mosi_mosei_generalization_20260829}"
PYTHON="${PYTHON:-/home/muxingyu/.conda/envs/HME/bin/python}"
DATA="/media/disk3/muxy/Dataset/Method/Method/aligned_mosi_ramp_bert.pkl"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_ROOT}/logs"

worker() {
  local seed="$1"
  local gpu="$2"
  for variant in full g_only; do
    local output_dir="${RESULT_ROOT}/mosi/${variant}/seed_${seed}"
    local log_path="${RESULT_ROOT}/logs/mosi_stable_${variant}_seed_${seed}_gpu${gpu}.log"
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. PYTHONUNBUFFERED=1 "${PYTHON}" \
      scripts/run_sentiment_generalization.py \
      --dataset mosi \
      --data "${DATA}" \
      --variant "${variant}" \
      --seed "${seed}" \
      --output-dir "${output_dir}" \
      --device cuda:0 \
      --epochs 30 \
      --min-epochs 12 \
      --patience 8 \
      --batch-size 96 \
      --lr 3e-4 \
      >"${log_path}" 2>&1
  done
}

pilot_dir="${RESULT_ROOT}/pilot_mosi_short_schedule"
mkdir -p "${pilot_dir}"
if [[ ! -f "${pilot_dir}/ARCHIVED" ]]; then
  cp -a "${RESULT_ROOT}/mosi" "${pilot_dir}/mosi"
  date -Iseconds >"${pilot_dir}/ARCHIVED"
fi

worker 1701 0 & pid0=$!
worker 1702 1 & pid1=$!
worker 1703 2 & pid2=$!
wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
printf '{"status":"complete","protocol":"30 epochs, min 12, patience 8","finished_at":"%s"}\n' \
  "$(date -Iseconds)" >"${RESULT_ROOT}/MOSI_STABLE_COMPLETE.json"
