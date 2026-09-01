#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from ramp_msa.config import apply_overrides, load_config
from ramp_msa.data import build_dataloaders
from ramp_msa.model import RAMPModel


POLICY_NAMES = ["T", "A", "V", "TA", "TV", "AV", "TAV"]


def main():
    p = argparse.ArgumentParser(description="Inspect learned RAMP procedural memory")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.data_path:
        cfg.data.path = args.data_path
        cfg.data.synthetic = False
    loaders, info = build_dataloaders(cfg)
    device = torch.device(args.device)
    model = RAMPModel(info.text_dim, info.audio_dim, info.vision_dim, cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    mem = model.memory

    report = {"global": mem.stats(), "regimes": []}
    for c in range(mem.num_regimes):
        idx = torch.where(mem.valid[c])[0]
        if idx.numel() == 0:
            report["regimes"].append({"regime": c, "valid_slots": 0})
            continue
        values = mem.values[c, idx].detach().cpu()
        util = mem.utility[c, idx].detach().cpu()
        weighted = torch.softmax(util, dim=0)[:, None] * values
        report["regimes"].append({
            "regime": c,
            "valid_slots": int(idx.numel()),
            "mean_utility": float(util.mean()),
            "max_utility": float(util.max()),
            "mean_policy": {name: float(values[:, j].mean()) for j, name in enumerate(POLICY_NAMES)},
            "utility_weighted_policy": {name: float(weighted[:, j].sum()) for j, name in enumerate(POLICY_NAMES)},
        })
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
