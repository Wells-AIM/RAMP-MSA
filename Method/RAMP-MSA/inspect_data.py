#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def describe(x):
    try:
        arr = np.asarray(x)
        return f"type={type(x).__name__}, shape={arr.shape}, dtype={arr.dtype}"
    except Exception:
        return f"type={type(x).__name__}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    args = p.parse_args()
    path = Path(args.path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("root keys:", list(data.keys()) if isinstance(data, dict) else type(data))
    if isinstance(data, dict):
        for split_name, split in data.items():
            if not isinstance(split, dict):
                continue
            print(f"\n[{split_name}] keys={list(split.keys())}")
            for k, v in split.items():
                if hasattr(v, "__len__") and len(v) > 0:
                    print(f"  {k:24s} len={len(v):6d} first: {describe(v[0])}")
                else:
                    print(f"  {k:24s} {describe(v)}")


if __name__ == "__main__":
    main()
