"""Standard MOSI/MOSEI regression metrics (MMSA definitions)."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _multiclass_acc(p, t):
    return float(np.mean(np.round(p) == np.round(t)))


def mosi_metrics(pred, true):
    p, t = np.asarray(pred).reshape(-1), np.asarray(true).reshape(-1)
    nz = t != 0
    res = {
        'Has0_acc_2': accuracy_score(t >= 0, p >= 0),
        'Has0_F1': f1_score(t >= 0, p >= 0, average='weighted'),
        'Non0_acc_2': accuracy_score(t[nz] > 0, p[nz] > 0),
        'Non0_F1': f1_score(t[nz] > 0, p[nz] > 0, average='weighted'),
        'Acc_5': _multiclass_acc(np.clip(p, -2, 2), np.clip(t, -2, 2)),
        'Acc_7': _multiclass_acc(np.clip(p, -3, 3), np.clip(t, -3, 3)),
        'MAE': float(np.mean(np.abs(p - t))),
        'Corr': float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else 0.0,
    }
    return {k: round(float(v), 4) for k, v in res.items()}
