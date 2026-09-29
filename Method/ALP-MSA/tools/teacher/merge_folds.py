"""
Merge K cross-fitted fine-tuned-teacher folds (finetune_omni.py) into one teacher npz for distillation.

  train utterances : prediction / hidden state from the fold model that did NOT train on them (out-of-fold)
  valid / test     : average over the K fold models (analysis only; students never use valid/test targets)

The output starts from an existing reasoning-teacher npz (e.g. teacher_B.npz: evidence-text embeddings) and
  - replaces `score` by the fine-tuned teacher prediction (label distillation from a stronger teacher),
  - adds `z_hid` (L2-normalised final hidden state, float16) + `valid_hid`.

usage: python merge_folds.py --dir .../omni_ft --mode text --nfolds 5 --base teacher_B.npz --pkl MOSI.pkl --out teacher_C.npz
"""

import argparse
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--mode', default='text')
    ap.add_argument('--nfolds', type=int, default=5)
    ap.add_argument('--base', required=True)
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--out', required=True)
    opt = ap.parse_args()

    pred, hid, n = {}, {}, {}
    for k in range(opt.nfolds):
        f = np.load(f'{opt.dir}/{opt.mode}_fold{k}of{opt.nfolds}.npz')
        for i, p, h in zip(f['held_ids'], f['held_pred'], f['held_hid']):        # out-of-fold train
            pred[str(i)], hid[str(i)] = float(p), h.astype(np.float32)
        for s in ('valid', 'test'):                                             # fold average
            for i, p, h in zip(f[f'{s}_ids'], f[f'{s}_pred'], f[f'{s}_hid']):
                i = str(i)
                pred[i] = pred.get(i, 0.0) + float(p) / opt.nfolds
                hid[i] = hid.get(i, 0.0) + h.astype(np.float32) / opt.nfolds

    data = pickle.load(open(opt.pkl, 'rb'))
    for s in ('train', 'valid', 'test'):
        ids = [str(i) for i in data[s]['id']]
        y = np.array(data[s]['regression_labels'], np.float32)
        ok = np.array([i in pred for i in ids])
        p = np.array([pred.get(i, np.nan) for i in ids], np.float32)
        nz = ok & (y != 0)
        print(f'fine-tuned teacher ({opt.mode}) {s:5s}: covered {ok.mean():.3f}  MAE {np.nanmean(np.abs(p - y)):.4f}  '
              f'corr {np.corrcoef(p[ok], y[ok])[0, 1]:.4f}  acc2(non0) {np.mean((p[nz] > 0) == (y[nz] > 0)):.4f}  '
              f'acc7 {np.mean(np.round(np.clip(p[ok], -3, 3)) == np.round(np.clip(y[ok], -3, 3))):.4f}')

    base = dict(np.load(opt.base))
    ids = [str(i) for i in base['ids']]
    base['score'] = np.array([pred.get(i, np.nan) for i in ids], np.float32)
    dim = next(iter(hid.values())).shape[0]
    z = np.zeros((len(ids), dim), np.float32)
    valid = np.zeros(len(ids), bool)
    for r, i in enumerate(ids):
        if i in hid:
            v = hid[i]
            z[r], valid[r] = v / (np.linalg.norm(v) + 1e-8), True
    base['z_hid'], base['valid_hid'] = z.astype(np.float16), valid
    np.savez(opt.out, **base)
    print('saved', opt.out, 'z_hid', z.shape, 'coverage', valid.mean())


if __name__ == '__main__':
    main()
