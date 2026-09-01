#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/disk3/muxy/QA/PDER_MOSI_MOSEI_20260829}"
RESULT_ROOT="${RESULT_ROOT:-/media/disk3/muxy/QA/experiments/pder_mosi_mosei_generalization_20260829}"
PYTHON="${PYTHON:-/home/muxingyu/.conda/envs/HME/bin/python}"
DATA_ROOT="${DATA_ROOT:-/media/disk3/muxy/Dataset/Method/Method}"

mkdir -p "${RESULT_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_job() {
  local dataset="$1"
  local variant="$2"
  local seed="$3"
  local gpu="$4"
  local data_path="${DATA_ROOT}/aligned_${dataset}_ramp_bert.pkl"
  local output_dir="${RESULT_ROOT}/${dataset}/${variant}/seed_${seed}"
  local log_path="${RESULT_ROOT}/logs/${dataset}_${variant}_seed_${seed}_gpu${gpu}.log"

  if [[ -f "${output_dir}/COMPLETE.json" ]]; then
    printf 'SKIP complete %s %s seed=%s\n' "${dataset}" "${variant}" "${seed}"
    return 0
  fi
  mkdir -p "${output_dir}"
  printf 'START %s %s seed=%s gpu=%s %s\n' "${dataset}" "${variant}" "${seed}" "${gpu}" "$(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. PYTHONUNBUFFERED=1 "${PYTHON}" \
    scripts/run_sentiment_generalization.py \
    --dataset "${dataset}" \
    --data "${data_path}" \
    --variant "${variant}" \
    --seed "${seed}" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --epochs 12 \
    --patience 4 \
    --batch-size 96 \
    --lr 3e-4 \
    >"${log_path}" 2>&1
  printf 'DONE %s %s seed=%s gpu=%s %s\n' "${dataset}" "${variant}" "${seed}" "${gpu}" "$(date -Iseconds)"
}

worker() {
  local seed="$1"
  local gpu="$2"
  run_job mosi full "${seed}" "${gpu}"
  run_job mosi g_only "${seed}" "${gpu}"
  run_job mosei full "${seed}" "${gpu}"
  run_job mosei g_only "${seed}" "${gpu}"
  printf 'WORKER_COMPLETE seed=%s gpu=%s %s\n' "${seed}" "${gpu}" "$(date -Iseconds)"
}

worker 1701 0 >"${RESULT_ROOT}/logs/worker_gpu0.log" 2>&1 &
pid0=$!
worker 1702 1 >"${RESULT_ROOT}/logs/worker_gpu1.log" 2>&1 &
pid1=$!
worker 1703 2 >"${RESULT_ROOT}/logs/worker_gpu2.log" 2>&1 &
pid2=$!

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
wait "${pid2}" || status=1
if [[ "${status}" -ne 0 ]]; then
  printf '{"status":"failed","finished_at":"%s"}\n' "$(date -Iseconds)" >"${RESULT_ROOT}/GRID_FAILED.json"
  exit "${status}"
fi
printf '{"status":"complete","finished_at":"%s"}\n' "$(date -Iseconds)" >"${RESULT_ROOT}/GRID_COMPLETE.json"
