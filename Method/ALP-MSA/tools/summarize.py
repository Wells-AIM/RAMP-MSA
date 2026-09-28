"""Aggregate runs/<name>/seed_*/result.json into mean +- std tables (test @ val-selected epoch)."""

import glob
import json
import os
import sys

import numpy as np

KEYS = ['Non0_acc_2', 'Non0_F1', 'Has0_acc_2', 'Acc_7', 'Acc_5', 'MAE', 'Corr']


def main(root='runs', pattern='*'):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'seed_*', 'result.json')))]
        if not res:
            continue
        te = {k: np.array([r['selected']['test'][k] for r in res]) for k in KEYS}
        va_mae = np.array([r['selected']['valid']['MAE'] for r in res])
        rows.append((os.path.basename(d), len(res), va_mae, te))
    hdr = f'{"run":34s} {"n":>2s} {"valMAE":>7s} ' + ' '.join(f'{k:>14s}' for k in KEYS)
    print(hdr)
    for name, n, va, te in rows:
        cells = []
        for k in KEYS:
            s = 1 if k in ('MAE', 'Corr') else 100
            cells.append(f'{te[k].mean()*s:7.2f}±{te[k].std(ddof=1 if n > 1 else 0)*s:5.2f}'
                         if k not in ('MAE', 'Corr') else f'{te[k].mean():7.4f}±{te[k].std(ddof=1 if n > 1 else 0):.3f}')
        print(f'{name:34s} {n:2d} {va.mean():7.4f} ' + ' '.join(f'{c:>14s}' for c in cells))


if __name__ == '__main__':
    main(*sys.argv[1:])
