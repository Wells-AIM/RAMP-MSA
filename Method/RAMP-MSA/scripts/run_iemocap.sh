#!/usr/bin/env bash
set -e
DATA=${1:?Usage: scripts/run_iemocap.sh /path/to/IEMOCAP/features.pkl}
python train.py --config configs/iemocap.yaml --data-path "$DATA"
