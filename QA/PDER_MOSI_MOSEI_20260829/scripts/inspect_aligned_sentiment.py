#!/usr/bin/env python3
"""Audit aligned MOSI/MOSEI pickle payloads without modifying them."""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
from typing import Any

import numpy as np


def describe(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape"):
        array = np.asarray(value)
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.array([])
        return {
            "type": type(value).__name__,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first": describe(value[0]) if value else None,
        }
    return {"type": type(value).__name__, "repr": repr(value)[:120]}


def summarize(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    report: dict[str, Any] = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "top_level_type": type(payload).__name__,
        "splits": {},
    }
    all_ids: dict[str, set[str]] = {}
    split_names = ("train", "valid", "test") if "valid" in payload else ("train", "dev", "test")
    for split_name in split_names:
        samples = payload[split_name]
        if isinstance(samples, dict):
            labels = np.asarray(samples["regression_labels"], dtype=np.float64).reshape(-1)
            ids = [str(value) for value in samples["id"]]
            first_sample = {key: describe(value[0]) for key, value in samples.items()}
            count = len(labels)
        else:
            labels = np.asarray(
                [float(np.asarray(sample[1]).reshape(-1)[0]) for sample in samples],
                dtype=np.float64,
            )
            ids = [str(sample[2]) for sample in samples]
            first_sample = [describe(value) for value in samples[0]]
            count = len(samples)
        all_ids[split_name] = set(ids)
        rounded_counts = collections.Counter(f"{value:.6g}" for value in labels)
        split_report = {
            "count": count,
            "first_sample": first_sample,
            "label_min": float(labels.min()),
            "label_max": float(labels.max()),
            "label_mean": float(labels.mean()),
            "negative": int((labels < 0).sum()),
            "zero": int((labels == 0).sum()),
            "positive": int((labels > 0).sum()),
            "unique_ids": len(set(ids)),
            "duplicate_ids": len(ids) - len(set(ids)),
            "common_labels": rounded_counts.most_common(12),
        }
        report["splits"][split_name] = split_report

    validation_name = "valid" if "valid" in all_ids else "dev"
    report["id_overlap"] = {
        "train_dev": len(all_ids["train"] & all_ids[validation_name]),
        "train_test": len(all_ids["train"] & all_ids["test"]),
        "dev_test": len(all_ids[validation_name] & all_ids["test"]),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.paths], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
