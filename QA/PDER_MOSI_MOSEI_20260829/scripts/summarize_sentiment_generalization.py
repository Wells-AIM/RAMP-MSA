#!/usr/bin/env python3
"""Aggregate three-seed MOSI/MOSEI generalization results and draw a paper figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DATASETS = ("mosi", "mosei")
VARIANTS = ("g_only", "full")
SEEDS = (1701, 1702, 1703)
METRICS = ("acc2_nonzero", "f1_macro", "f1_positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    return parser.parse_args()


def load_runs(root: Path) -> list[dict]:
    rows = []
    missing = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                path = root / dataset / variant / f"seed_{seed}" / "result.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                row = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
                row.update({metric: result["test"][metric] for metric in METRICS})
                rows.append(row)
    if missing:
        raise FileNotFoundError("Missing runs:\n" + "\n".join(missing))
    return rows


def values(rows: list[dict], dataset: str, variant: str, metric: str) -> np.ndarray:
    selected = sorted(
        (row for row in rows if row["dataset"] == dataset and row["variant"] == variant),
        key=lambda row: row["seed"],
    )
    return np.asarray([row[metric] for row in selected], dtype=np.float64)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for dataset in DATASETS:
        for metric in METRICS:
            baseline = values(rows, dataset, "g_only", metric)
            full = values(rows, dataset, "full", metric)
            delta = full - baseline
            t_result = stats.ttest_rel(full, baseline)
            try:
                wilcoxon_p = float(stats.wilcoxon(full, baseline).pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            summary.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "g_only_mean": float(baseline.mean()),
                    "g_only_std": float(baseline.std(ddof=1)),
                    "full_mean": float(full.mean()),
                    "full_std": float(full.std(ddof=1)),
                    "paired_delta_mean": float(delta.mean()),
                    "paired_delta_std": float(delta.std(ddof=1)),
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_p": wilcoxon_p,
                    "full_better_seeds": int((delta > 0).sum()),
                    "n_seeds": int(len(delta)),
                }
            )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict]) -> None:
    lines = [
        "# MOSI/MOSEI Generalization",
        "",
        "Official train/validation/test splits; zero-sentiment samples are excluded consistently. Values are mean +/- sample SD over three paired seeds.",
        "",
        "| Dataset | Metric | G-only | Full PDER | Paired delta | p (paired t) | Seeds improved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset'].upper()} | {row['metric']} | "
            f"{row['g_only_mean']:.4f} +/- {row['g_only_std']:.4f} | "
            f"{row['full_mean']:.4f} +/- {row['full_std']:.4f} | "
            f"{row['paired_delta_mean']:+.4f} | {row['paired_t_p']:.4g} | "
            f"{row['full_better_seeds']}/{row['n_seeds']} |"
        )
    lines.extend(
        [
            "",
            "Statistical tests are exploratory because n=3 paired seeds; effect direction and seed consistency are emphasized.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_figure(root: Path, rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"g_only": "#999999", "full": "#0072B2"}
    labels = {"g_only": "G-only", "full": "Full PDER"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)
    for panel_index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        x = np.arange(2)
        width = 0.34
        for variant_index, variant in enumerate(VARIANTS):
            offset = (variant_index - 0.5) * width
            metric_values = [values(rows, dataset, variant, metric) for metric in ("acc2_nonzero", "f1_macro")]
            means = [value.mean() for value in metric_values]
            stds = [value.std(ddof=1) for value in metric_values]
            axis.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=2.5,
                color=colors[variant],
                edgecolor="black",
                linewidth=0.5,
                label=labels[variant],
                zorder=2,
            )
            for metric_index, seed_values in enumerate(metric_values):
                jitter = np.linspace(-0.04, 0.04, len(seed_values))
                axis.scatter(
                    np.full(len(seed_values), x[metric_index] + offset) + jitter,
                    seed_values,
                    s=13,
                    color="white",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
        axis.set_title(dataset.upper())
        axis.set_xticks(x, ["Acc-2", "Macro F1"])
        axis.set_ylim(0.45, 1.0)
        axis.set_ylabel("Test score")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(-0.13, 1.04, chr(ord("A") + panel_index), transform=axis.transAxes, fontweight="bold", fontsize=10)
    axes[0].legend(frameon=False, loc="lower right")
    fig.savefig(root / "pder_mosi_mosei_generalization.png", dpi=400, bbox_inches="tight")
    fig.savefig(root / "pder_mosi_mosei_generalization.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.result_root)
    rows = load_runs(root)
    summary = summarize(rows)
    write_csv(root / "seed_results.csv", rows)
    write_csv(root / "three_seed_summary.csv", summary)
    (root / "three_seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(root / "RESULTS.md", summary)
    draw_figure(root, rows)
    (root / "ANALYSIS_COMPLETE.json").write_text(
        json.dumps({"runs": len(rows), "summary_rows": len(summary)}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
