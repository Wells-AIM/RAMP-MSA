"""Seed-paired comparison of runs against a reference run (test metrics at the val-selected epoch).

usage: python tools/paired.py runs REF_RUN RUN [RUN ...]
Prints mean difference (run - ref), how many seeds improve, and a paired t-test p-value.
"""

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

KEYS = [('Non0_acc_2', 100, +1), ('Acc_7', 100, +1), ('MAE', 1, -1), ('Corr', 1, +1)]


def load(root, name):
    out = {}
    for f in glob.glob(os.path.join(root, name, 'seed_*', 'result.json')):
        r = json.load(open(f))
        out[r['seed']] = r['selected']['test']
    return out


def main(root, ref, *names):
    base = load(root, ref)
    print(f'reference {ref}: {len(base)} seeds')
    print(f'{"run":22s} {"n":>2s} ' + ' '.join(f'{k:>24s}' for k, _, _ in KEYS))
    for name in names:
        cur = load(root, name)
        seeds = sorted(set(base) & set(cur))
        cells = []
        for k, scale, sign in KEYS:
            d = np.array([cur[s][k] - base[s][k] for s in seeds]) * scale
            better = int(np.sum(sign * d > 0))
            p = stats.ttest_1samp(d, 0.0).pvalue if len(d) > 1 and d.std() > 0 else float('nan')
            cells.append(f'{d.mean():+7.3f} ({better}/{len(d)}) p={p:.3f}')
        print(f'{name:22s} {len(seeds):2d} ' + ' '.join(f'{c:>24s}' for c in cells))


if __name__ == '__main__':
    main(*sys.argv[1:])
