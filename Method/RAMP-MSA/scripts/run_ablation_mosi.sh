#!/usr/bin/env bash
set -e
DATA=${1:?Usage: scripts/run_ablation_mosi.sh /path/to/MOSI/Processed/unaligned_50.pkl}
python train.py --config configs/mosi.yaml --data-path "$DATA" --output runs/mosi_full
python train.py --config configs/mosi.yaml --data-path "$DATA" --output runs/mosi_no_memory --set memory.enabled=false
python train.py --config configs/mosi.yaml --data-path "$DATA" --output runs/mosi_static --set memory.online_update=false
python train.py --config configs/mosi.yaml --data-path "$DATA" --output runs/mosi_no_novelty --set memory.write_novelty_power=0.0
python train.py --config configs/mosi.yaml --data-path "$DATA" --output runs/mosi_fast_writer --set memory.writer_momentum=0.0
