"""
Direction 3: distil teacher emotion-evidence reasoning into the tri-modal student (training only).

  label     : MSE(student prediction, teacher sentiment score)              (logit/label distillation)
  reason    : InfoNCE(fused student feature, embedding of the whole reasoning chain)
              L = -log exp(sim(z_M, z_R)/tau) / sum_j exp(sim(z_M, z_R^j)/tau)   (doc eq. L_reason)
  fields    : per-modality evidence alignment, text tokens <-> Semantic, audio tokens <-> Prosody,
              vision tokens <-> Facial (same InfoNCE, one term per modality)

Direction 1 (Reasoning-Aligned Affective Representation) extensions:
  verify_sigma > 0 : verified reasoning distillation -- reason / fields / rel terms are weighted per
                     sample by exp(-(teacher score - label)^2 / 2 sigma^2) (train labels only), so
                     reasoning that reached a wrong conclusion is not distilled
  rel              : relational reasoning distillation (in-batch similarity structure, KL)

The teacher only provides targets for training-split utterances; nothing teacher-related is used at
validation / test time.
"""

import torch
import torch.nn.functional as F
from torch import nn

FIELD_OF = {'l': 'semantic', 'a': 'prosody', 'v': 'facial'}


def _wmean(x, w):
    return x.mean() if w is None else (x * w).sum() / w.sum().clamp(min=1e-6)


def info_nce(a, b, valid, tau, weight=None):
    """Symmetric in-batch InfoNCE between rows of a and b, restricted to rows with valid targets.
    weight (optional, per row): Direction 1 verified distillation -- rows whose teacher reasoning
    reached a wrong conclusion contribute less (weights are detached targets, not learned)."""
    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return a.sum() * 0.0
    a, b = F.normalize(a[idx], dim=-1), F.normalize(b[idx], dim=-1)
    logits = a @ b.t() / tau
    target = torch.arange(len(idx), device=a.device)
    per_row = 0.5 * (F.cross_entropy(logits, target, reduction='none')
                     + F.cross_entropy(logits.t(), target, reduction='none'))
    return _wmean(per_row, None if weight is None else weight[idx])


def relational_kd(s, t, valid, tau_s, tau_t, weight=None):
    """Direction 1 relational reasoning distillation: the student's in-batch similarity distribution
    should match the teacher-reasoning similarity distribution (which utterances are reasoned about
    alike), KL(p_teacher || p_student) per row, self-similarity excluded.  t is the frozen teacher
    embedding, so only the geometry of the reasoning space is transferred, not instance identity."""
    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() < 3:
        return s.sum() * 0.0
    s, t = F.normalize(s[idx], dim=-1), F.normalize(t[idx], dim=-1)
    eye = torch.eye(len(idx), dtype=torch.bool, device=s.device)
    ls = (s @ s.t() / tau_s).masked_fill(eye, float('-inf'))
    lt = (t @ t.t() / tau_t).masked_fill(eye, float('-inf'))
    p_t = F.softmax(lt, -1)
    kl = (p_t * (torch.log(p_t.clamp(min=1e-12)) - F.log_softmax(ls, -1).masked_fill(eye, 0.0))).sum(-1)
    return _wmean(kl, None if weight is None else weight[idx])


def verify_weight(teacher, sigma):
    """w_i = exp(-(s_teacher - y)^2 / 2 sigma^2) on TRAIN samples (labels are available there);
    0 where the teacher gave no parsable score.  sigma <= 0 disables verification (w = None)."""
    if sigma <= 0 or 'y' not in teacher:
        return None
    y = teacher['y'].view(-1).float()
    # score_verify (if given) = conclusion of the distilled reasoning chain; otherwise the teacher score
    key = 'score_verify' if 'score_verify' in teacher else 'score'
    w = torch.exp(-(teacher[key].float() - y) ** 2 / (2 * sigma ** 2))
    return torch.where(teacher['has_' + key], w, torch.zeros_like(w)).detach()


class DistillHeads(nn.Module):
    def __init__(self, feat_dim, token_dim, teacher_dim, proj_dim=256, hid_dim=3584):
        super().__init__()
        def mlp(i):
            return nn.Sequential(nn.Linear(i, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.s_all, self.t_all = mlp(feat_dim), nn.Linear(teacher_dim, proj_dim)
        self.s_mod = nn.ModuleDict({m: mlp(token_dim) for m in FIELD_OF})
        self.t_mod = nn.ModuleDict({m: nn.Linear(teacher_dim, proj_dim) for m in FIELD_OF})
        # fine-tuned strong teacher: final hidden state of the cross-fitted Omni regressor (z_hid)
        self.s_hid, self.t_hid = mlp(feat_dim), nn.Linear(hid_dim, proj_dim)

    def forward(self, out, aux, teacher, cfg):
        """teacher: dict of tensors for the batch (score, has_score, z_all, has_all, z_<field>, has_<field>)."""
        losses = {}
        w = verify_weight(teacher, getattr(cfg, 'verify_sigma', 0.0))
        if cfg.w_label > 0:
            m = teacher['has_score']
            losses['label'] = F.mse_loss(out[m].view(-1), teacher['score'][m]) if m.any() else out.sum() * 0.0
        if cfg.w_reason > 0:
            losses['reason'] = info_nce(self.s_all(aux['fused']), self.t_all(teacher['z_all']),
                                        teacher['has_all'], cfg.tau, w)
        if cfg.w_fields > 0:
            losses['fields'] = sum(info_nce(self.s_mod[m](aux['pooled'][m]), self.t_mod[m](teacher[f'z_{f}']),
                                            teacher[f'has_{f}'], cfg.tau, w) for m, f in FIELD_OF.items()) / 3
        w_rel = getattr(cfg, 'w_rel', 0.0)
        if w_rel > 0:
            losses['rel'] = relational_kd(self.s_all(aux['fused']), teacher['z_all'], teacher['has_all'],
                                          getattr(cfg, 'rel_tau', 0.1), getattr(cfg, 'rel_tau', 0.1), w)
        w_hid = getattr(cfg, 'w_hid', 0.0)
        if w_hid > 0 and 'z_hid' in teacher:
            losses['hid'] = info_nce(self.s_hid(aux['fused']), self.t_hid(teacher['z_hid'].float()),
                                     teacher['has_hid'], cfg.tau)
        total = (cfg.w_label * losses.get('label', 0.0) + cfg.w_reason * losses.get('reason', 0.0)
                 + cfg.w_fields * losses.get('fields', 0.0) + w_rel * losses.get('rel', 0.0)
                 + w_hid * losses.get('hid', 0.0))
        return total, {k: float(v) for k, v in losses.items()}
