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

    # frame-rate robustness (A/V resampled along time at the selected epoch)
    rob = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'seed_*', 'result.json')))]
        res = [r for r in res if 'test_resampled' in r['selected']]
        if res:
            fs = list(res[0]['selected']['test_resampled'])
            acc = {f: np.mean([r['selected']['test_resampled'][f]['Non0_acc_2'] for r in res]) * 100 for f in fs}
            mae = {f: np.mean([r['selected']['test_resampled'][f]['MAE'] for r in res]) for f in fs}
            base = np.mean([r['selected']['test']['Non0_acc_2'] for r in res]) * 100
            rob.append((os.path.basename(d), len(res), base, acc, mae))
    if rob:
        fs = list(rob[0][3])
        print('\nframe-rate robustness: Non0_acc_2 (MAE) of test A/V resampled by factor')
        print(f'{"run":34s} {"n":>2s} {"x1.0":>7s} ' + ' '.join(f'{"x"+f:>15s}' for f in fs))
        for name, n, base, acc, mae in rob:
            print(f'{name:34s} {n:2d} {base:7.2f} ' + ' '.join(f'{acc[f]:6.2f} ({mae[f]:.3f})' for f in fs))


if __name__ == '__main__':
    main(*sys.argv[1:])
