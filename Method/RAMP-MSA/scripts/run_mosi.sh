#!/usr/bin/env bash
set -e
DATA=${1:?Usage: scripts/run_mosi.sh /path/to/MOSI/Processed/unaligned_50.pkl}
python train.py --config configs/mosi.yaml --data-path "$DATA"
