"""Mean +- std of val-selected test metrics for runs whose result.json only has the core metrics
(e.g. CaReFlow-D3 runs). usage: python tools/summarize_simple.py runs NAME [NAME ...]"""

import glob
import json
import os
import sys

import numpy as np

KEYS = [('Non0_acc_2', 100), ('Non0_F1', 100), ('Acc_7', 100), ('MAE', 1), ('Corr', 1)]


def main(root, *names):
    print(f'{"run":14s} {"n":>2s} ' + ' '.join(f'{k:>16s}' for k, _ in KEYS) + '  seeds')
    for name in names:
        files = sorted(glob.glob(os.path.join(root, name, 'seed_*', 'result.json')))
        rs = [json.load(open(f)) for f in files]
        if not rs:
            continue
        cells = []
        for k, s in KEYS:
            v = np.array([r['selected']['test'][k] for r in rs]) * s
            cells.append(f'{v.mean():8.3f}±{v.std(ddof=1) if len(v) > 1 else 0:6.3f}')
        print(f'{name:14s} {len(rs):2d} ' + ' '.join(f'{c:>16s}' for c in cells) + '  ' +
              ','.join(str(r['seed']) for r in rs))


if __name__ == '__main__':
    main(*sys.argv[1:])
