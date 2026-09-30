"""
Emotion Error Recycling (Direction 2.5 of the ICLR2026 method summary).

Stage 1 (cross-fitted Emotion Confusion Bank)
    A model has ~zero error on the samples it was trained on, so its *stable* error
    patterns are measured out-of-fold: the train split is cut into K folds, a model is
    trained on K-1 folds (dev split still used for epoch selection) and predicts the
    held-out fold.  tools/build_error_bank.py merges the folds into
        (index, y, y_hat, confidence, error type, severity)   for every train sample.

Stage 2 (targeted retraining, this module)
    * bank re-weighting    w_i = 1 + alpha * severity_i        (batch-normalised to mean 1)
    * confusion margin     relu(m - sign(y_i) * f(x_i)) on the samples whose polarity the
                           stage-1 model flipped (explicit correction of the confusion direction)
    * error injection      bank samples get a perturbed view (audio degradation, visual
                           occlusion, text masking, modality dropout, temporal jitter) that must
                           still be predicted correctly
Baselines for RQ3: focal-style regression weighting |e|^gamma and online hard example mining.
All terms are training-only; inference is unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F

ERROR_TYPES = ('correct', 'intensity', 'flip_low_margin', 'flip_high_conf')


class ErrorBank:
    def __init__(self, path, n_train, device, min_severity=0.0):
        z = np.load(path)
        assert len(z['label']) == n_train, f'bank has {len(z["label"])} rows, train split has {n_train}'
        self.sev = torch.as_tensor(z['severity'], dtype=torch.float32, device=device)
        self.pred = torch.as_tensor(z['pred'], dtype=torch.float32, device=device)   # out-of-fold prediction
        self.flip = torch.as_tensor(z['flip'], dtype=torch.bool, device=device)
        self.in_bank = torch.as_tensor(z['in_bank'], dtype=torch.bool, device=device) & (self.sev >= min_severity)
        types = z['etype']
        print('error bank: ' + ', '.join(f'{t}={int((types == i).sum())}' for i, t in enumerate(ERROR_TYPES))
              + f' | in bank {int(self.in_bank.sum())}/{n_train}', flush=True)


def sample_weights(err, index, bank, args):
    """Per-sample loss weights (mean 1 over the batch). err: detached |f(x)-y| of the batch."""
    w = torch.ones_like(err)
    if bank is not None and args.er_weight != 0:   # alpha < 0 down-weights the bank (min weight 1+alpha)
        w = w + args.er_weight * bank.sev[index] * bank.in_bank[index].float()
    if args.focal_gamma > 0:                      # baseline: focal-style regression weighting
        w = w * (err + 1e-3) ** args.focal_gamma
    return w / w.mean()


def soft_target(y, index, bank, args):
    """Error-aware label refinement: move the target of (bank) samples towards the cross-fitted
    out-of-fold prediction, y' = y + lambda * (y_oof - y).  The stage-1 error is read as evidence
    that the label is ambiguous/noisy rather than as a sample to be fitted harder."""
    if bank is None or args.er_soft <= 0:
        return y
    on = bank.in_bank[index] if args.er_soft_on == 'bank' else torch.ones_like(y, dtype=torch.bool)
    # control: 'zero' shrinks the same samples towards 0 by the same lambda (no per-sample error information)
    ref = bank.pred[index] if args.er_soft_ref == 'oof' else torch.zeros_like(y)
    return y + args.er_soft * (ref - y) * on.float()


def weighted_mse(pred, y, index, bank, args):
    y = soft_target(y, index, bank, args)
    e2 = (pred - y) ** 2
    if args.ohem_frac > 0:                        # baseline: online hard example mining
        k = max(1, int(round(args.ohem_frac * e2.numel())))
        return e2.topk(k).values.mean()
    w = sample_weights(e2.detach().sqrt(), index, bank, args)
    return (w * e2).mean()


def margin_loss(pred, y, mask, m, clamp=False):
    """Push the prediction to the correct side of the polarity boundary by margin m.
    clamp=True uses m_i = min(m, |y_i|): the margin never asks for more than the label itself
    (with a fixed m, 21% of MOSI stage-1 flips have |y| < 0.3 and the margin contradicts MSE)."""
    keep = mask & (y != 0)
    if not keep.any():
        return pred.sum() * 0.0
    yk = y[keep]
    mk = torch.minimum(torch.full_like(yk, m), yk.abs()) if clamp else m
    return F.relu(mk - torch.sign(yk) * pred[keep]).mean()


# ---------------------------------------------------------------------- #
# error injection (inputs are already batch-min-max normalised to [0, 1])
# ---------------------------------------------------------------------- #
def perturb(input_ids, input_mask, visual, acoustic, kinds, mask_id, gen,
            noise_rel=None, occ=0.3, text_mask=0.15):
    """Apply one randomly chosen perturbation per sample. Returns new tensors (inputs untouched).

    noise_rel=None: legacy absolute noise std 0.1 (after batch_minmax the per-dim std of MOSI audio is
    ~0.106, i.e. SNR ~ 1: the audio is replaced rather than degraded).  noise_rel=r: noise std =
    r * per-dim std of the valid frames in the batch (r=0.3 -> SNR ~ 3)."""
    ids, vis, aco = input_ids.clone(), visual.clone(), acoustic.clone()
    B, L = ids.shape
    valid = input_mask.bool().clone()
    lens = valid.sum(1)
    valid[torch.arange(B), 0] = False                          # [CLS]
    valid[torch.arange(B), (lens - 1).clamp(min=0)] = False    # [SEP]
    if noise_rel is None:
        a_scale = torch.full((acoustic.size(-1),), 0.1, device=acoustic.device)
    else:
        a_scale = noise_rel * acoustic[valid].std(0) if valid.any() else torch.zeros(acoustic.size(-1),
                                                                                   device=acoustic.device)
    choice = torch.randint(len(kinds), (B,), generator=gen, device=ids.device)
    rand = torch.rand(B, L, generator=gen, device=ids.device)
    # "blank" frame = the [CLS] position, which is an all-zero feature vector before min-max scaling
    blank_a, blank_v = acoustic[:, :1], visual[:, :1]
    for b in range(B):
        kind = kinds[int(choice[b])]
        v = valid[b].unsqueeze(-1)
        if kind == 'audio':        # degradation: additive noise on the valid frames
            noise = torch.randn(aco[b].shape, generator=gen, device=aco.device) * a_scale
            aco[b] = aco[b] + noise * v
        elif kind == 'visual':     # occlusion: blank a fraction `occ` of the frames
            o = ((rand[b] < occ) & valid[b]).unsqueeze(-1)
            vis[b] = torch.where(o, blank_v[b], vis[b])
        elif kind == 'text':       # masking: a fraction `text_mask` of the word pieces -> [MASK]
            tm = (rand[b] < text_mask) & valid[b]
            ids[b][tm] = mask_id
        elif kind == 'drop':       # modality dropout: remove audio or vision entirely
            if rand[b, 0] < 0.5:
                aco[b] = torch.where(v, blank_a[b], aco[b])
            else:
                vis[b] = torch.where(v, blank_v[b], vis[b])
        elif kind == 'jitter':     # temporal jitter: shift A/V by 1-2 steps against the words
            s = 1 + int(rand[b, 1] < 0.5)
            s = s if rand[b, 2] < 0.5 else -s
            aco[b] = torch.where(v, torch.roll(aco[b], s, 0), aco[b])
            vis[b] = torch.where(v, torch.roll(vis[b], s, 0), vis[b])
        else:
            raise ValueError(kind)
    return ids, vis, aco
