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


def run_epoch(model, loader, device, optimizer=None, grad_clip=0.0):
    train = optimizer is not None
    model.train(train)
    preds, trues, tot, n = [], [], 0.0, 0
    for batch in loader:
        v, a, t = batch['vision'].to(device), batch['audio'].to(device), batch['text'].to(device)
        vm, am = batch['vision_mask'].to(device), batch['audio_mask'].to(device)
        y = batch['label'].to(device)
        with torch.set_grad_enabled(train):
            out = model(v, a, t, vm, am)
            loss = torch.nn.functional.mse_loss(out, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
        preds.append(out.detach().cpu())
        trues.append(y.cpu())
    res = mosi_metrics(torch.cat(preds).numpy(), torch.cat(trues).numpy())
    res['loss'] = round(tot / n, 5)
    return res, torch.cat(preds).numpy().reshape(-1)


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
        tr, _ = run_epoch(model, loaders['train'], device, optimizer, cfg.base.grad_clip)
        va, _ = run_epoch(model, loaders['valid'], device)
        te, te_pred = run_epoch(model, loaders['test'], device)
        sched.step()
        history.append({'epoch': ep, 'train': tr, 'valid': va, 'test': te})
        if better(key, va[key], best):
            best, best_ep = va[key], ep
            sel = {'epoch': ep, 'valid': va, 'test': te}
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
