"""
Merge stage-1 out-of-fold predictions into an Emotion Confusion Bank (Direction 2.5).

usage: python tools/build_error_bank.py OUT.npz RUN_DIR/fold*/seed_*/oof.npz ...

Per train sample (indexed by position in the CaReFlow train split):
  label, pred (out-of-fold), confidence = |pred| (distance to the polarity boundary)
  etype  0 correct | 1 intensity (right polarity, |e| >= 1) | 2 flip, low margin | 3 flip, high confidence
  severity = min(1, |e|/2 + 0.5*flip)
  in_bank  = etype > 0, or a low-margin correct sample (|pred| < --low_margin, y != 0)
"""

import argparse
import collections

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er import ERROR_TYPES   # noqa: E402

NAMES = {-3: 'strong neg', -2: 'neg', -1: 'weak neg', 0: 'neutral', 1: 'weak pos', 2: 'pos', 3: 'strong pos'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('oof', nargs='+')
    ap.add_argument('--conf', type=float, default=0.5, help='|pred| above which a polarity flip is high-confidence')
    ap.add_argument('--intensity', type=float, default=1.0)
    ap.add_argument('--low_margin', type=float, default=0.2)
    opt = ap.parse_args()

    parts = [np.load(f) for f in opt.oof]
    idx = np.concatenate([p['index'] for p in parts])
    assert len(set(idx.tolist())) == len(idx), 'overlapping folds'
    n = int(idx.max()) + 1
    assert len(idx) == n, f'folds cover {len(idx)} of {n} train samples'
    pred, label = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for p in parts:
        pred[p['index']] = p['pred']
        label[p['index']] = p['label']

    err = np.abs(pred - label)
    flip = (label != 0) & ((pred >= 0) != (label >= 0))
    conf = np.abs(pred)
    etype = np.zeros(n, np.int64)
    etype[(~flip) & (err >= opt.intensity)] = 1
    etype[flip & (conf < opt.conf)] = 2
    etype[flip & (conf >= opt.conf)] = 3
    severity = np.minimum(1.0, err / 2 + 0.5 * flip).astype(np.float32)
    in_bank = (etype > 0) | ((conf < opt.low_margin) & (label != 0))

    np.savez(opt.out, label=label, pred=pred, err=err, confidence=conf, flip=flip, etype=etype,
             severity=severity, in_bank=in_bank)

    nz = label != 0
    print(f'{n} train samples from {len(parts)} folds | OOF Acc2(non0) {1 - flip[nz].mean():.4f} '
          f'| MAE {err.mean():.4f} | Corr {np.corrcoef(pred, label)[0, 1]:.4f}')
    for i, t in enumerate(ERROR_TYPES):
        print(f'  {t:16s} {int((etype == i).sum()):5d}')
    print(f'  in bank          {int(in_bank.sum()):5d}  (mean severity {severity[in_bank].mean():.3f})')
    yt = np.clip(np.round(label), -3, 3).astype(int)
    yp = np.clip(np.round(pred), -3, 3).astype(int)
    conf_pairs = collections.Counter((a, b) for a, b in zip(yt, yp) if a != b)
    print('most frequent 7-class confusions (true -> predicted):')
    for (a, b), c in conf_pairs.most_common(8):
        print(f'  {NAMES[a]:>10s} -> {NAMES[b]:<10s} {c:4d}  ({c / max(1, (yt == a).sum()):.0%} of {NAMES[a]})')


if __name__ == '__main__':
    main()
