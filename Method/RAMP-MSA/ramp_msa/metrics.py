from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .utils import pearson_corr


def regression_metrics(pred, target) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(pred - target)))
    corr = pearson_corr(pred, target)

    # Common MOSI/MOSEI conventions.
    acc7 = float(np.mean(np.clip(np.rint(pred), -3, 3) == np.clip(np.rint(target), -3, 3)))
    acc5 = float(np.mean(np.clip(np.rint(pred), -2, 2) == np.clip(np.rint(target), -2, 2)))
    acc3 = float(np.mean(np.sign(np.clip(np.rint(pred), -1, 1)) == np.sign(np.clip(np.rint(target), -1, 1))))

    all_true = (target >= 0).astype(np.int64)
    all_pred = (pred >= 0).astype(np.int64)
    acc2_all = float(accuracy_score(all_true, all_pred))
    f1_all = float(f1_score(all_true, all_pred, average="weighted", zero_division=0))

    non0 = target != 0
    if non0.any():
        nz_true = (target[non0] > 0).astype(np.int64)
        nz_pred = (pred[non0] > 0).astype(np.int64)
        acc2_non0 = float(accuracy_score(nz_true, nz_pred))
        f1_non0 = float(f1_score(nz_true, nz_pred, average="weighted", zero_division=0))
    else:
        acc2_non0, f1_non0 = 0.0, 0.0

    return {
        "MAE": mae,
        "Corr": corr,
        "Acc7": acc7,
        "Acc5": acc5,
        "Acc3": acc3,
        "Acc2_all": acc2_all,
        "F1_all": f1_all,
        "Acc2_non0": acc2_non0,
        "F1_non0": f1_non0,
    }


def classification_metrics(pred, target) -> Dict[str, float]:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    return {
        "Acc": float(accuracy_score(target, pred)),
        "MacroF1": float(f1_score(target, pred, average="macro", zero_division=0)),
        "WeightedF1": float(f1_score(target, pred, average="weighted", zero_division=0)),
    }
