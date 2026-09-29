"""How good is the teacher on its own? Per-task score accuracy vs. ground truth (per split).

usage: python eval_teacher.py --pkl MOSI/unaligned_50.pkl --jsonl t1.jsonl t2.jsonl ... [--show 3]
"""

import argparse
import json
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--jsonl', nargs='+', required=True)
    ap.add_argument('--show', type=int, default=0)
    opt = ap.parse_args()

    data = pickle.load(open(opt.pkl, 'rb'))
    lab = {str(i): float(y) for s in data for i, y in zip(data[s]['id'], data[s]['regression_labels'])}
    for path in opt.jsonl:
        rs = [json.loads(line) for line in open(path)]
        parsed = [r for r in rs if r['score'] is not None]
        print(f'{path.split("/")[-1]:24s} n={len(rs)} parsed={len(parsed) / max(1, len(rs)):.2f}')
        for split in ('train', 'valid', 'test'):
            ok = [r for r in parsed if r['split'] == split and lab[r['id']] != 0]
            if len(ok) < 3:
                continue
            p = np.array([r['score'] for r in ok])
            y = np.array([lab[r['id']] for r in ok])
            corr = np.corrcoef(p, y)[0, 1] if p.std() > 0 else 0.0
            print(f'    {split:5s} n={len(ok):4d} acc2(non0)={np.mean((p > 0) == (y > 0)):.3f} '
                  f'corr={corr:.3f} MAE={np.mean(np.abs(p - y)):.3f} score std={p.std():.2f}')
        for r in rs[:opt.show]:
            print(f'  -- {r["id"]} label={lab[r["id"]]:+.1f} teacher={r["score"]}\n     {r["raw"][:400]!r}')


if __name__ == '__main__':
    main()
