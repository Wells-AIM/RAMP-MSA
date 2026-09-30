"""
Direction 1: attach the reasoning teacher's own conclusion as `score_verify` to another teacher npz.

teacher_C.npz carries the fine-tuned (cross-fitted) score for label distillation, but its reasoning
embeddings (z_all, z_<field>) come from the reasoning teacher B.  Verified reasoning distillation must
check each chain against ITS OWN conclusion, so we copy B's score as score_verify (rows matched by id).

usage: python add_verify_score.py --base teacher_C.npz --verify teacher_B.npz --out teacher_CB.npz
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--verify', required=True)
    ap.add_argument('--out', required=True)
    opt = ap.parse_args()
    base, ver = np.load(opt.base, allow_pickle=True), np.load(opt.verify, allow_pickle=True)
    row = {str(i): k for k, i in enumerate(ver['ids'])}
    idx = np.array([row.get(str(i), -1) for i in base['ids']])
    sv = np.where(idx >= 0, ver['score'][np.maximum(idx, 0)], np.nan).astype(np.float32)
    out = {k: base[k] for k in base.files}
    out['score_verify'] = sv
    np.savez(opt.out, **out)
    print(f'{opt.out}: score_verify for {np.isfinite(sv).mean():.3f} of {len(sv)} rows')


if __name__ == '__main__':
    main()
