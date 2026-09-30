"""
Are the samples in the Emotion Confusion Bank genuine model errors or unreliable labels?

Independent evidence: the fine-tuned, 5-fold cross-fitted Omni-7B teacher (Direction 3, teacher_C.npz;
its train-split scores are also out-of-fold). For each bank error type we compare
  * polarity agreement of the teacher with the LABEL vs with the CaReFlow OOF prediction
  * |teacher - label| vs |teacher - oof|
usage: python tools/label_audit.py BANK.npz TEACHER.npz datasets/mosi.pkl datasets/mosi_id_map.json
"""

import json
import pickle
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from er import ERROR_TYPES   # noqa: E402

bank = np.load(sys.argv[1])
teacher = np.load(sys.argv[2])
train = pickle.load(open(sys.argv[3], 'rb'))['train']
id_map = json.load(open(sys.argv[4]))

row = {str(i): k for k, i in enumerate(teacher['ids'])}
t_idx = np.array([row.get(id_map.get(seg, ''), -1) for (_, _, seg) in train])
has = (t_idx >= 0) & np.isfinite(teacher['score'][np.maximum(t_idx, 0)])
ts = teacher['score'][np.maximum(t_idx, 0)].astype(np.float64)
y, p, et = bank['label'], bank['pred'], bank['etype']
assert np.allclose(y, np.array([np.asarray(lab).reshape(-1)[0] for (_, lab, _) in train])), 'bank not aligned with the pickle train split'
print(f'teacher score available for {has.mean():.1%} of {len(y)} train samples')
nz = has & (y != 0)
print(f'teacher OOF on train: Acc2(non0) {np.mean((ts[nz] >= 0) == (y[nz] >= 0)):.4f}, '
      f'MAE {np.mean(np.abs(ts[has] - y[has])):.4f}; CaReFlow OOF Acc2 {np.mean((p[nz] >= 0) == (y[nz] >= 0)):.4f}')

print(f'\n{"group":16s} {"n":>5s} {"|y|":>6s} {"T sign=label":>13s} {"T sign=OOF":>11s} '
      f'{"|T-label|":>10s} {"|T-OOF|":>8s}')
groups = [(t, et == i) for i, t in enumerate(ERROR_TYPES)] + [('all bank', bank['in_bank'].astype(bool))]
for name, g in groups:
    g = g & has & (y != 0)
    if g.sum() == 0:
        continue
    print(f'{name:16s} {g.sum():5d} {np.abs(y[g]).mean():6.2f} {np.mean((ts[g] >= 0) == (y[g] >= 0)):13.1%} '
          f'{np.mean((ts[g] >= 0) == (p[g] >= 0)):11.1%} {np.abs(ts[g] - y[g]).mean():10.3f} {np.abs(ts[g] - p[g]).mean():8.3f}')
flip = bank['flip'].astype(bool) & has
print(f'\npolarity flips: teacher sides with the label on {np.mean((ts[flip] >= 0) == (y[flip] >= 0)):.1%}, '
      f'with CaReFlow OOF on {np.mean((ts[flip] >= 0) == (p[flip] >= 0)):.1%} of {flip.sum()} samples')
print('examples (label, CaReFlow OOF, teacher, words) for high-confidence flips:')
hc = np.where((et == 3) & has)[0]
for i in hc[np.argsort(-np.abs(p[hc] - y[hc]))][:8]:
    words = ' '.join(train[i][0][0])[:90]
    print(f'  y={y[i]:+.2f} oof={p[i]:+.2f} T={ts[i]:+.2f} | {words}')
