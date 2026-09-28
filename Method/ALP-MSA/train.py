"""
Train / evaluate ALP-Net under a strict protocol:
  * model selection uses the VALIDATION split only (key: cfg.base.key_eval);
  * the reported test metrics are those of the single selected epoch
    (no per-metric best-epoch picking on the test set, unlike the ALMT repo);
  * test metrics of every epoch are logged for curves only.

usage:
  python train.py --config configs/mosi.yaml --seed 1111 --name almt_repro \
      --set model.audio_mode=dynamic model.vision_mode=dynamic
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml

from alp.data import build_loaders
from alp.metrics import mosi_metrics
from alp.models.alp_net import ALPNet
from alp.utils import dict_to_ns, apply_overrides


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, device, optimizer=None, grad_clip=0.0, event_weight=0.0,
              warp=None, warp_prob=0.5, cons_weight=0.0):
    """warp=(lo, hi): train-time A/V temporal resampling by a per-sample factor ~ U(lo, hi).
    cons_weight > 0: two views (original + warped) per batch, both supervised, plus a
    granularity-consistency term ||f(x) - f(warp(x))||^2; otherwise each sample is warped
    with probability warp_prob (plain augmentation)."""
    train = optimizer is not None
    model.train(train)
    preds, trues, tot, n = [], [], 0.0, 0
    ev_tot = 0.0
    mse = torch.nn.functional.mse_loss
    for batch in loader:
        v, a, t = batch['vision'].to(device), batch['audio'].to(device), batch['text'].to(device)
        vm, am = batch['vision_mask'].to(device), batch['audio_mask'].to(device)
        y = batch['label'].to(device)
        with torch.set_grad_enabled(train):
            extra = 0.0
            if train and warp is not None:
                f = torch.empty(y.size(0)).uniform_(*warp).tolist()
                if cons_weight <= 0:
                    keep = torch.rand(y.size(0)) >= warp_prob
                    f = [1.0 if k else fi for k, fi in zip(keep.tolist(), f)]
                v2, vm2 = resample_time(v, vm, f)
                a2, am2 = resample_time(a, am, f)
                if cons_weight > 0:
                    out, aux = model(v, a, t, vm, am, return_aux=True)
                    out2 = model(v2, a2, t, vm2, am2)
                    extra = 0.5 * mse(out2, y) + cons_weight * mse(out, out2)
                    loss = mse(out, y)
                    loss_main = 0.5 * loss
                else:
                    out, aux = model(v2, a2, t, vm2, am2, return_aux=True)
                    loss = loss_main = mse(out, y)
            else:
                out, aux = model(v, a, t, vm, am, return_aux=True)
                loss = loss_main = mse(out, y)
            ev = sum(aux[k]['event_loss'] for k in aux if 'event_loss' in aux[k])
            ev_tot += float(ev) * y.size(0)
        if train:
            optimizer.zero_grad(set_to_none=True)
            total = loss_main + extra + (event_weight * ev if event_weight > 0 else 0.0)
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
        preds.append(out.detach().cpu())
        trues.append(y.cpu())
    res = mosi_metrics(torch.cat(preds).numpy(), torch.cat(trues).numpy())
    res['loss'] = round(tot / n, 5)
    res['event_loss'] = round(ev_tot / n, 5)
    return res, torch.cat(preds).numpy().reshape(-1)


def resample_time(x, mask, factor):
    """Linearly resample the valid part of every A/V sequence to round(L*factor) frames
    (simulates a different frame rate / speaking-speed); padding stays zero."""
    B, L, D = x.shape
    out = torch.zeros_like(x)
    new_mask = torch.zeros_like(mask)
    lens = mask.sum(1).clamp(min=1).tolist()
    factors = factor if isinstance(factor, (list, tuple)) else [factor] * B
    for i in range(B):
        n = int(lens[i])
        m = max(1, min(L, int(round(n * factors[i]))))
        if n > 1:
            seq = x[i, :n].t().unsqueeze(0)                               # (1, D, n)
            out[i, :m] = torch.nn.functional.interpolate(seq, size=m, mode='linear',
                                                         align_corners=True).squeeze(0).t()
        else:
            out[i, :m] = x[i, :1].expand(m, D)
        new_mask[i, :m] = True
    return out, new_mask


def zero_modality(x, mask):
    """Remove a modality: all-zero features, a single valid frame (keeps masking well-defined)."""
    m = torch.zeros_like(mask)
    m[:, 0] = True
    return torch.zeros_like(x), m


@torch.no_grad()
def ablation_eval(model, loader, device):
    """Test metrics with audio / vision / both removed at test time (reliance on A/V)."""
    model.eval()
    res = {}
    for name in ('no_audio', 'no_vision', 'no_av'):
        preds, trues = [], []
        for batch in loader:
            v, a, t = batch['vision'].to(device), batch['audio'].to(device), batch['text'].to(device)
            vm, am = batch['vision_mask'].to(device), batch['audio_mask'].to(device)
            if name in ('no_audio', 'no_av'):
                a, am = zero_modality(a, am)
            if name in ('no_vision', 'no_av'):
                v, vm = zero_modality(v, vm)
            preds.append(model(v, a, t, vm, am).cpu())
            trues.append(batch['label'])
        res[name] = mosi_metrics(torch.cat(preds).numpy(), torch.cat(trues).numpy())
    return res


@torch.no_grad()
def robust_eval(model, loader, device, factors):
    model.eval()
    res = {}
    for f in factors:
        preds, trues = [], []
        for batch in loader:
            v, a, t = batch['vision'].to(device), batch['audio'].to(device), batch['text'].to(device)
            vm, am = batch['vision_mask'].to(device), batch['audio_mask'].to(device)
            v, vm = resample_time(v, vm, f)
            a, am = resample_time(a, am, f)
            preds.append(model(v, a, t, vm, am).cpu())
            trues.append(batch['label'])
        res[str(f)] = mosi_metrics(torch.cat(preds).numpy(), torch.cat(trues).numpy())
    return res


def better(key, new, best):
    if best is None:
        return True
    lower = key in ('MAE', 'loss')
    return new < best if lower else new > best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=1111)
    ap.add_argument('--name', default='debug')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--epochs', type=int, default=None)
    opt = ap.parse_args()

    with open(opt.config) as f:
        raw = yaml.safe_load(f)
    raw = apply_overrides(raw, opt.set)
    if opt.epochs is not None:
        raw['base']['n_epochs'] = opt.epochs
    cfg = dict_to_ns(raw)

    out_dir = os.path.join(cfg.base.run_root, opt.name, f'seed_{opt.seed}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(raw, f)

    setup_seed(opt.seed)
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')
    loaders = build_loaders(cfg, opt.seed)
    model = ALPNet(cfg).to(device)
    n_params = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith('bert.'))
    print(f'non-BERT params: {n_params/1e6:.3f}M', flush=True)

    bert_params = [p for n_, p in model.named_parameters() if n_.startswith('bert.')]
    other_params = [p for n_, p in model.named_parameters() if not n_.startswith('bert.')]
    bert_lr = getattr(cfg.base, 'bert_lr', None) or cfg.base.lr
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': bert_lr},
        {'params': other_params, 'lr': cfg.base.lr},
    ], weight_decay=cfg.base.weight_decay)

    E = cfg.base.n_epochs
    warm = max(1, int(0.1 * E))

    def lr_lambda(ep):  # linear warm-up then cosine (ALMT schedule, per epoch)
        if ep < warm:
            return (ep + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (ep - warm) / max(1, 0.9 * E))))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    key = cfg.base.key_eval
    best, best_ep, history = None, -1, []
    t0 = time.time()
    for ep in range(1, E + 1):
        warp = getattr(cfg.base, 'time_warp', None)
        tr, _ = run_epoch(model, loaders['train'], device, optimizer, cfg.base.grad_clip,
                          event_weight=cfg.model.event_weight,
                          warp=tuple(warp) if warp else None,
                          warp_prob=getattr(cfg.base, 'warp_prob', 0.5),
                          cons_weight=getattr(cfg.base, 'warp_consistency', 0.0))
        va, _ = run_epoch(model, loaders['valid'], device)
        te, te_pred = run_epoch(model, loaders['test'], device)
        sched.step()
        history.append({'epoch': ep, 'train': tr, 'valid': va, 'test': te})
        if better(key, va[key], best):
            best, best_ep = va[key], ep
            sel = {'epoch': ep, 'valid': va, 'test': te}
            factors = getattr(cfg.base, 'robust_eval', None) or []
            if factors:
                sel['test_resampled'] = robust_eval(model, loaders['test'], device, factors)
            if getattr(cfg.base, 'ablate_eval', False):
                sel['test_ablated'] = ablation_eval(model, loaders['test'], device)
            np.save(os.path.join(out_dir, 'test_pred_best.npy'), te_pred)
            if cfg.base.save_ckpt:
                torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
        if ep % cfg.base.log_every == 0 or ep == E:
            print(f'ep {ep:3d} | tr loss {tr["loss"]:.4f} | va MAE {va["MAE"]:.4f} acc2 {va["Non0_acc_2"]:.4f} '
                  f'| te MAE {te["MAE"]:.4f} acc2 {te["Non0_acc_2"]:.4f} acc7 {te["Acc_7"]:.4f} '
                  f'| best ep {best_ep} | {time.time()-t0:.0f}s', flush=True)

    result = {'name': opt.name, 'seed': opt.seed, 'key_eval': key, 'selected': sel,
              'overrides': opt.set, 'n_params_nonbert': n_params, 'time_s': round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    print('SELECTED', json.dumps(sel['test']), 'epoch', sel['epoch'], flush=True)


if __name__ == '__main__':
    main()
