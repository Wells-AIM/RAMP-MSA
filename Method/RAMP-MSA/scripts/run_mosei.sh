#!/usr/bin/env bash
set -e
DATA=${1:?Usage: scripts/run_mosei.sh /path/to/MOSEI/Processed/unaligned_50.pkl}
python train.py --config configs/mosei.yaml --data-path "$DATA"
