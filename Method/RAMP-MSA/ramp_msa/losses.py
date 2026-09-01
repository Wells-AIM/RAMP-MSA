from __future__ import annotations

import torch
import torch.nn.functional as F


def task_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor, task_type: str, smooth_l1_beta: float = 1.0) -> torch.Tensor:
    if task_type == "regression":
        pred = logits.squeeze(-1)
        return F.smooth_l1_loss(pred, labels.float(), reduction="none", beta=smooth_l1_beta)
    return F.cross_entropy(logits, labels.long(), reduction="none")


def cosine_policy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, p=2, dim=-1, eps=1e-8)
    target_n = F.normalize(target.detach(), p=2, dim=-1, eps=1e-8)
    valid = target.detach().norm(dim=-1) > 1e-8
    if not valid.any():
        return pred.sum() * 0.0
    return (1.0 - (pred_n[valid] * target_n[valid]).sum(dim=-1)).mean()


def route_balance_loss(route_probs: torch.Tensor) -> torch.Tensor:
    if route_probs is None or route_probs.numel() == 0:
        return torch.tensor(0.0, device=route_probs.device if route_probs is not None else "cpu")
    mean_usage = route_probs.mean(dim=0).clamp_min(1e-8)
    c = mean_usage.numel()
    return (mean_usage * torch.log(mean_usage * c)).sum()
