#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ramp_msa.config import apply_overrides, load_config, save_config
from ramp_msa.data import build_dataloaders
from ramp_msa.model import RAMPModel
from ramp_msa.trainer import train
from ramp_msa.utils import count_parameters, ensure_dir, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train RAMP-MSA")
    p.add_argument("--config", required=True, help="YAML config")
    p.add_argument("--data-path", default=None, help="Override data.path")
    p.add_argument("--output", default=None, help="Override experiment.output_dir")
    p.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    p.add_argument("--set", action="append", default=[], help="Nested override, e.g. --set memory.enabled=false")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.data_path is not None:
        cfg.data.path = args.data_path
        cfg.data.synthetic = False
    if args.output is not None:
        cfg.experiment.output_dir = args.output

    set_seed(int(cfg.experiment.seed), deterministic=bool(cfg.experiment.get("deterministic", False)))
    device_str = args.device or cfg.experiment.get("device", "cuda")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    loaders, info = build_dataloaders(cfg)
    print(f"Dataset dims: text={info.text_dim}, audio={info.audio_dim}, vision={info.vision_dim}")
    print(f"Splits: train={info.num_train}, valid={info.num_valid}, test={info.num_test}")

    model = RAMPModel(info.text_dim, info.audio_dim, info.vision_dim, cfg).to(device)
    params = count_parameters(model)
    print(f"Parameters: {params['trainable']:,} trainable / {params['total']:,} total")
    print(f"Memory capacity: {model.memory.capacity} slots, policy_dim={model.POLICY_DIM}")

    out_dir = ensure_dir(cfg.experiment.output_dir)
    save_config(cfg, out_dir / "resolved_config.yaml")
    result = train(model, loaders, device, cfg, out_dir)
    print(f"Done. Best epoch: {result['best_epoch']}. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
