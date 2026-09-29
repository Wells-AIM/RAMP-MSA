"""
Direction 3: distil teacher emotion-evidence reasoning into the tri-modal student (training only).

  label     : MSE(student prediction, teacher sentiment score)              (logit/label distillation)
  reason    : InfoNCE(fused student feature, embedding of the whole reasoning chain)
              L = -log exp(sim(z_M, z_R)/tau) / sum_j exp(sim(z_M, z_R^j)/tau)   (doc eq. L_reason)
  fields    : per-modality evidence alignment, text tokens <-> Semantic, audio tokens <-> Prosody,
              vision tokens <-> Facial (same InfoNCE, one term per modality)

The teacher only provides targets for training-split utterances; nothing teacher-related is used at
validation / test time.
"""

import torch
import torch.nn.functional as F
from torch import nn

FIELD_OF = {'l': 'semantic', 'a': 'prosody', 'v': 'facial'}


def info_nce(a, b, valid, tau):
    """Symmetric in-batch InfoNCE between rows of a and b, restricted to rows with valid targets."""
    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return a.sum() * 0.0
    a, b = F.normalize(a[idx], dim=-1), F.normalize(b[idx], dim=-1)
    logits = a @ b.t() / tau
    target = torch.arange(len(idx), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


class DistillHeads(nn.Module):
    def __init__(self, feat_dim, token_dim, teacher_dim, proj_dim=256):
        super().__init__()
        def mlp(i):
            return nn.Sequential(nn.Linear(i, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.s_all, self.t_all = mlp(feat_dim), nn.Linear(teacher_dim, proj_dim)
        self.s_mod = nn.ModuleDict({m: mlp(token_dim) for m in FIELD_OF})
        self.t_mod = nn.ModuleDict({m: nn.Linear(teacher_dim, proj_dim) for m in FIELD_OF})

    def forward(self, out, aux, teacher, cfg):
        """teacher: dict of tensors for the batch (score, has_score, z_all, has_all, z_<field>, has_<field>)."""
        losses = {}
        if cfg.w_label > 0:
            m = teacher['has_score']
            losses['label'] = F.mse_loss(out[m].view(-1), teacher['score'][m]) if m.any() else out.sum() * 0.0
        if cfg.w_reason > 0:
            losses['reason'] = info_nce(self.s_all(aux['fused']), self.t_all(teacher['z_all']),
                                        teacher['has_all'], cfg.tau)
        if cfg.w_fields > 0:
            losses['fields'] = sum(info_nce(self.s_mod[m](aux['pooled'][m]), self.t_mod[m](teacher[f'z_{f}']),
                                            teacher[f'has_{f}'], cfg.tau) for m, f in FIELD_OF.items()) / 3
        total = (cfg.w_label * losses.get('label', 0.0) + cfg.w_reason * losses.get('reason', 0.0)
                 + cfg.w_fields * losses.get('fields', 0.0))
        return total, {k: float(v) for k, v in losses.items()}
