"""
Audit of the Direction-2.5 implementation (CPU only).

  1. bank <-> train-split alignment (labels by index)
  2. polarity-flip samples: |y| distribution vs the margin m (does the margin contradict the label?)
  3. magnitude of the injected perturbations after batch_minmax (noise / occlusion vs feature spread)

usage: CUDA_VISIBLE_DEVICES= python tools/diag_er.py BANK.npz
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)
bank_path = sys.argv[1]
sys.argv = [sys.argv[0], '--dataset', 'mosi', '--model',
            '/media/disk3/muxy/Method/RAMP-MSA/experiment_versions/embedding_swap_20260918/models/deberta-v3-base',
            '--train_batch_size', '32']
import train_reflow_new as T   # noqa: E402  (parses the argv above)
import er                      # noqa: E402

train_loader, _, _, _ = T.set_up_data_loader()
ds = train_loader.dataset
labels = ds.tensors[3].view(-1).numpy()
z = np.load(bank_path)

print('== 1. alignment')
print('train samples', len(labels), '| bank rows', len(z['label']),
      '| max |bank label - train label|', float(np.abs(z['label'] - labels).max()))

print('== 2. polarity flips vs margin')
flip = z['flip'].astype(bool)
ay = np.abs(labels[flip])
print(f'flips {flip.sum()} | |y| quantiles 10/25/50/75/90%: {np.round(np.quantile(ay, [.1, .25, .5, .75, .9]), 3)}')
for m in (0.2, 0.3, 0.5):
    print(f'  margin m={m}: {np.mean(ay < m):.0%} of flip samples have |y| < m (margin demands |pred| beyond the label)')
print('  oof pred of flips: |pred| median', np.round(np.median(np.abs(z['pred'][flip])), 3))

print('== 3. perturbation magnitude after batch_minmax (5 train batches)')
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
stats = {'a_std': [], 'v_std': [], 'a_range': [], 'v_range': []}
for i, batch in enumerate(train_loader):
    if i == 5:
        break
    ids, vis, aco, y, mask, idx = batch
    vis, aco = T.batch_minmax(vis), T.batch_minmax(aco)
    valid = mask.bool()
    a, v = aco[valid], vis[valid]                     # (N_valid_tokens, D)
    stats['a_std'].append(a.std(0).median().item())
    stats['v_std'].append(v.std(0).median().item())
    stats['a_range'].append((a.max() - a.min()).item())
    stats['v_range'].append((v.max() - v.min()).item())
for k, v in stats.items():
    print(f'  {k}: {np.round(v, 4)}')
print(f'  audio noise std 0.1 vs median per-dim audio std {np.median(stats["a_std"]):.4f} '
      f'-> SNR(std) {np.median(stats["a_std"]) / 0.1:.3f}')

print('== 4. legacy vs mild perturbation on one batch (one kind at a time)')
ids, vis, aco, y, mask, idx = next(iter(train_loader))
vis, aco = T.batch_minmax(vis), T.batch_minmax(aco)
valid = mask.bool()
a_std = aco[valid].std(0)
mask_id = T.get_tokenizer(T.args.model).mask_token_id
for name, kw in (('legacy', dict(noise_rel=None, occ=0.3, text_mask=0.15)),
                 ('mild', dict(noise_rel=0.3, occ=0.15, text_mask=0.10))):
    out = []
    for kind in ('audio', 'visual', 'text'):
        g = torch.Generator().manual_seed(0)
        p_ids, p_vis, p_aco = er.perturb(ids, mask, vis, aco, (kind,), mask_id, g, **kw)
        if kind == 'audio':
            snr = (a_std / (p_aco - aco)[valid].std(0).clamp(min=1e-8)).median().item()
            out.append(f'audio SNR(std) {snr:.2f}')
        elif kind == 'visual':
            out.append(f'visual frames occluded {((p_vis != vis).any(-1) & valid).sum().item() / valid.sum().item():.1%}')
        else:
            out.append(f'text tokens masked {((p_ids != ids) & valid).sum().item() / valid.sum().item():.1%}')
    print(f'  {name:6s}: ' + ' | '.join(out))
m = torch.tensor([True, True]); yy = torch.tensor([0.2, -1.0]); pp = torch.tensor([0.25, -0.1])
print('  margin (m=0.3) on y=[0.2,-1.0], pred=[0.25,-0.1]: fixed',
      er.margin_loss(pp, yy, m, 0.3).item(), '| clamped', er.margin_loss(pp, yy, m, 0.3, clamp=True).item())
