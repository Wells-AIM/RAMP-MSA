from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrievalOutput:
    policy: torch.Tensor
    route_probs: torch.Tensor
    slot_attn: torch.Tensor
    entropy: torch.Tensor
    max_similarity: torch.Tensor


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


@torch.no_grad()
def spherical_kmeans(x: torch.Tensor, k: int, iters: int = 30, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small pure-PyTorch spherical k-means used only for memory initialization."""
    if x.ndim != 2:
        raise ValueError("x must be [N,D]")
    n = x.size(0)
    if n < k:
        raise ValueError(f"Need at least k={k} warmup samples, got N={n}")
    x = _normalize(x)
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)

    # Farthest-point flavored initialization is more stable than pure random.
    first = int(torch.randint(n, (1,), generator=g, device=x.device).item())
    centers = [x[first]]
    min_dist = 1.0 - (x @ centers[0].unsqueeze(-1)).squeeze(-1)
    for _ in range(1, k):
        idx = int(torch.argmax(min_dist).item())
        centers.append(x[idx])
        d = 1.0 - (x @ centers[-1].unsqueeze(-1)).squeeze(-1)
        min_dist = torch.minimum(min_dist, d)
    centers = torch.stack(centers, dim=0)

    assignment = torch.full((n,), -1, dtype=torch.long, device=x.device)
    for _ in range(iters):
        sim = x @ centers.t()
        new_assignment = sim.argmax(dim=-1)
        if torch.equal(new_assignment, assignment):
            assignment = new_assignment
            break
        assignment = new_assignment
        new_centers = []
        max_sim = sim.max(dim=-1).values
        for c in range(k):
            idx = torch.where(assignment == c)[0]
            if idx.numel() == 0:
                # Recover empty clusters with the least covered point.
                refill = torch.argmin(max_sim)
                new_centers.append(x[refill])
            else:
                new_centers.append(_normalize(x[idx].mean(dim=0, keepdim=True))[0])
        centers = torch.stack(new_centers, dim=0)
    return assignment, centers


@torch.no_grad()
def diverse_select(keys: torch.Tensor, scores: torch.Tensor, m: int) -> torch.Tensor:
    """Greedy k-center selection, seeded by the highest-scoring sample."""
    n = keys.size(0)
    if n <= m:
        return torch.arange(n, device=keys.device)
    keys = _normalize(keys)
    scores = scores.float()
    smin, smax = scores.min(), scores.max()
    score01 = (scores - smin) / (smax - smin + 1e-8)
    first = int(torch.argmax(score01).item())
    selected = [first]
    min_dist = 1.0 - keys @ keys[first].unsqueeze(-1)
    min_dist = min_dist.squeeze(-1)
    for _ in range(1, m):
        # Diversity is dominant; hardness only breaks near-ties.
        criterion = min_dist + 0.10 * score01
        criterion[selected] = -1e9
        idx = int(torch.argmax(criterion).item())
        selected.append(idx)
        dist = 1.0 - (keys @ keys[idx].unsqueeze(-1)).squeeze(-1)
        min_dist = torch.minimum(min_dist, dist)
    return torch.tensor(selected, dtype=torch.long, device=keys.device)


class ProceduralMemory(nn.Module):
    """Interaction-addressed key-policy memory with utility-aware consolidation.

    Key   = when a fusion skill is useful (interaction state embedding)
    Value = how to adjust fusion (7-D Möbius interaction policy)
    Utility = historical paired loss reduction credited through retrieval attention
    """

    def __init__(
        self,
        num_regimes: int,
        slots_per_regime: int,
        key_dim: int,
        policy_dim: int = 7,
        top_regimes: int = 2,
        top_slots: int = 16,
        route_temperature: float = 0.12,
        slot_temperature: float = 0.08,
        utility_momentum: float = 0.95,
        stats_momentum: float = 0.95,
        merge_similarity: float = 0.90,
        merge_momentum: float = 0.20,
        redundancy_weight: float = 0.25,
        min_updates: int = 1,
        max_updates: int = 4,
        plasticity_bias: float = -0.5,
        plasticity_hardness: float = 1.0,
        plasticity_novelty: float = 2.0,
        plasticity_bad_gain: float = 1.0,
    ):
        super().__init__()
        self.num_regimes = int(num_regimes)
        self.slots_per_regime = int(slots_per_regime)
        self.key_dim = int(key_dim)
        self.policy_dim = int(policy_dim)
        self.top_regimes = min(int(top_regimes), self.num_regimes)
        self.top_slots = int(top_slots)
        self.route_temperature = float(route_temperature)
        self.slot_temperature = float(slot_temperature)
        self.utility_momentum = float(utility_momentum)
        self.stats_momentum = float(stats_momentum)
        self.merge_similarity = float(merge_similarity)
        self.merge_momentum = float(merge_momentum)
        self.redundancy_weight = float(redundancy_weight)
        self.min_updates = int(min_updates)
        self.max_updates = int(max_updates)
        self.plasticity_bias = float(plasticity_bias)
        self.plasticity_hardness = float(plasticity_hardness)
        self.plasticity_novelty = float(plasticity_novelty)
        self.plasticity_bad_gain = float(plasticity_bad_gain)

        c, m, d, p = self.num_regimes, self.slots_per_regime, self.key_dim, self.policy_dim
        self.register_buffer("keys", torch.zeros(c, m, d))
        self.register_buffer("values", torch.zeros(c, m, p))
        self.register_buffer("utility", torch.zeros(c, m))
        self.register_buffer("valid", torch.zeros(c, m, dtype=torch.bool))
        self.register_buffer("cores", torch.zeros(c, d))
        self.register_buffer("regime_hardness_ema", torch.ones(c))
        self.register_buffer("regime_novelty_ema", torch.full((c,), 0.5))
        self.register_buffer("regime_gain_ema", torch.zeros(c))
        self.register_buffer("global_hardness_ema", torch.tensor(1.0))
        self.register_buffer("num_consolidations", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @property
    def capacity(self) -> int:
        return self.num_regimes * self.slots_per_regime

    def ready(self) -> bool:
        return bool(self.initialized.item()) and bool(self.valid.any().item())

    @torch.no_grad()
    def initialize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        hardness: Optional[torch.Tensor] = None,
        seed: int = 0,
        kmeans_iters: int = 30,
    ) -> Dict[str, int]:
        device = self.keys.device
        keys = _normalize(keys.detach().to(device))
        values = values.detach().to(device)
        if hardness is None:
            hardness = torch.ones(keys.size(0), device=device)
        else:
            hardness = hardness.detach().float().to(device)

        assignment, centers = spherical_kmeans(keys, self.num_regimes, iters=kmeans_iters, seed=seed)
        self.keys.zero_()
        self.values.zero_()
        self.utility.zero_()
        self.valid.zero_()
        self.cores.copy_(centers)

        counts = {}
        for c in range(self.num_regimes):
            idx = torch.where(assignment == c)[0]
            if idx.numel() == 0:
                counts[str(c)] = 0
                continue
            local = diverse_select(keys[idx], hardness[idx], self.slots_per_regime)
            chosen = idx[local]
            n = chosen.numel()
            self.keys[c, :n].copy_(keys[chosen])
            self.values[c, :n].copy_(values[chosen])
            self.valid[c, :n] = True
            # Give difficult warmup experiences a small initial utility prior.
            h = hardness[chosen]
            h = h / (hardness.mean() + 1e-8)
            self.utility[c, :n].copy_(0.05 * torch.tanh(h))
            counts[str(c)] = int(n)

        self._recompute_cores()
        self.global_hardness_ema.copy_(hardness.mean().clamp_min(1e-6))
        self.initialized.fill_(True)
        return counts

    @torch.no_grad()
    def _recompute_cores(self) -> None:
        for c in range(self.num_regimes):
            idx = torch.where(self.valid[c])[0]
            if idx.numel() > 0:
                self.cores[c].copy_(_normalize(self.keys[c, idx].mean(dim=0, keepdim=True))[0])

    def retrieve(self, q: torch.Tensor) -> RetrievalOutput:
        b = q.size(0)
        if not self.ready():
            zero_policy = torch.zeros(b, self.policy_dim, device=q.device, dtype=q.dtype)
            route = torch.full((b, self.num_regimes), 1.0 / self.num_regimes, device=q.device, dtype=q.dtype)
            attn = torch.zeros(b, self.num_regimes, self.slots_per_regime, device=q.device, dtype=q.dtype)
            return RetrievalOutput(zero_policy, route, attn, torch.zeros(b, device=q.device), torch.zeros(b, device=q.device))

        qn = _normalize(q)
        cores = _normalize(self.cores)
        core_sim = qn @ cores.t()  # [B,C]
        route_logits = core_sim / self.route_temperature
        if self.top_regimes < self.num_regimes:
            top = torch.topk(route_logits, k=self.top_regimes, dim=-1).indices
            keep = torch.zeros_like(route_logits, dtype=torch.bool)
            keep.scatter_(1, top, True)
            route_logits = route_logits.masked_fill(~keep, torch.finfo(route_logits.dtype).min)
        route_probs = torch.softmax(route_logits, dim=-1)

        slot_sim = torch.einsum("bd,cmd->bcm", qn, _normalize(self.keys))
        valid = self.valid.unsqueeze(0).expand(b, -1, -1)
        logits = slot_sim / self.slot_temperature + torch.log(route_probs.clamp_min(1e-8)).unsqueeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        flat = logits.flatten(1)
        flat_valid = valid.flatten(1)

        k = min(self.top_slots, int(self.valid.sum().item()))
        if k > 0 and k < flat.size(1):
            top = torch.topk(flat, k=k, dim=-1).indices
            keep = torch.zeros_like(flat_valid)
            keep.scatter_(1, top, True)
            flat = flat.masked_fill(~keep, torch.finfo(flat.dtype).min)
            flat_valid = flat_valid & keep

        flat_attn = torch.softmax(flat, dim=-1)
        flat_attn = flat_attn * flat_valid.to(flat_attn.dtype)
        flat_attn = flat_attn / flat_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slot_attn = flat_attn.reshape(b, self.num_regimes, self.slots_per_regime)
        policy = torch.einsum("bcm,cmp->bp", slot_attn, self.values)
        entropy = -(flat_attn.clamp_min(1e-8) * torch.log(flat_attn.clamp_min(1e-8))).sum(dim=-1)
        max_similarity = slot_sim.masked_fill(~valid, -1.0).flatten(1).max(dim=-1).values
        return RetrievalOutput(policy, route_probs, slot_attn, entropy, max_similarity)

    @torch.no_grad()
    def candidate_novelty(self, candidate_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_keys = _normalize(candidate_keys)
        core_sim = candidate_keys @ _normalize(self.cores).t()
        regime = core_sim.argmax(dim=-1)
        novelty = torch.ones(candidate_keys.size(0), device=candidate_keys.device)
        max_sim = torch.zeros_like(novelty)
        for c in range(self.num_regimes):
            idx = torch.where(regime == c)[0]
            if idx.numel() == 0:
                continue
            slots = torch.where(self.valid[c])[0]
            if slots.numel() == 0:
                novelty[idx] = 1.0
                max_sim[idx] = 0.0
                continue
            sim = candidate_keys[idx] @ _normalize(self.keys[c, slots]).t()
            ms = sim.max(dim=-1).values
            max_sim[idx] = ms
            novelty[idx] = (1.0 - ms).clamp_min(0.0)
        return novelty, regime, max_sim

    @torch.no_grad()
    def credit(self, slot_attn: torch.Tensor, gain: torch.Tensor, route_probs: Optional[torch.Tensor] = None) -> None:
        if not self.ready():
            return
        gain = gain.detach().float()
        attn = slot_attn.detach().float()
        mass = attn.sum(dim=0)
        credited = (attn * gain[:, None, None]).sum(dim=0) / mass.clamp_min(1e-8)
        touched = mass > 1e-6
        self.utility[touched] = (
            self.utility_momentum * self.utility[touched]
            + (1.0 - self.utility_momentum) * credited[touched]
        )
        if route_probs is not None:
            rp = route_probs.detach().float()
            r_mass = rp.sum(dim=0)
            r_gain = (rp * gain[:, None]).sum(dim=0) / r_mass.clamp_min(1e-8)
            self.regime_gain_ema.mul_(self.stats_momentum).add_(r_gain, alpha=1.0 - self.stats_momentum)

    @torch.no_grad()
    def _update_stats(
        self,
        regime: torch.Tensor,
        hardness: torch.Tensor,
        novelty: torch.Tensor,
        gains: torch.Tensor,
    ) -> None:
        h = hardness.detach().float()
        n = novelty.detach().float()
        g = gains.detach().float()
        self.global_hardness_ema.mul_(self.stats_momentum).add_(h.mean(), alpha=1.0 - self.stats_momentum)
        for c in range(self.num_regimes):
            idx = torch.where(regime == c)[0]
            if idx.numel() == 0:
                continue
            self.regime_hardness_ema[c].mul_(self.stats_momentum).add_(h[idx].mean(), alpha=1.0 - self.stats_momentum)
            self.regime_novelty_ema[c].mul_(self.stats_momentum).add_(n[idx].mean(), alpha=1.0 - self.stats_momentum)
            self.regime_gain_ema[c].mul_(self.stats_momentum).add_(g[idx].mean(), alpha=1.0 - self.stats_momentum)

    @torch.no_grad()
    def _budget(self, c: int) -> int:
        rel_h = self.regime_hardness_ema[c] / self.global_hardness_ema.clamp_min(1e-6) - 1.0
        nov = self.regime_novelty_ema[c] - 0.25
        bad_gain = torch.tanh(-self.regime_gain_ema[c] / self.global_hardness_ema.clamp_min(1e-6))
        logit = (
            self.plasticity_bias
            + self.plasticity_hardness * rel_h
            + self.plasticity_novelty * nov
            + self.plasticity_bad_gain * bad_gain
        )
        frac = torch.sigmoid(logit).item()
        budget = round(self.min_updates + frac * (self.max_updates - self.min_updates))
        return int(max(self.min_updates, min(self.max_updates, budget)))

    @torch.no_grad()
    def _slot_redundancy(self, c: int) -> torch.Tensor:
        valid_idx = torch.where(self.valid[c])[0]
        redundancy = torch.zeros(self.slots_per_regime, device=self.keys.device)
        if valid_idx.numel() <= 1:
            return redundancy
        k = _normalize(self.keys[c, valid_idx])
        sim = k @ k.t()
        sim.fill_diagonal_(-1.0)
        r = sim.max(dim=-1).values.clamp_min(0.0)
        redundancy[valid_idx] = r
        return redundancy

    @torch.no_grad()
    def consolidate(
        self,
        candidate_keys: torch.Tensor,
        candidate_values: torch.Tensor,
        hardness: torch.Tensor,
        stability: torch.Tensor,
        gains: torch.Tensor,
        hardness_power: float = 1.0,
        novelty_power: float = 1.0,
        stability_power: float = 1.0,
    ) -> Dict[str, float]:
        if not self.ready() or candidate_keys.numel() == 0:
            return {"writes": 0.0, "merges": 0.0, "mean_novelty": 0.0}

        ck = _normalize(candidate_keys.detach())
        cv = candidate_values.detach()
        h = hardness.detach().float().clamp_min(1e-8)
        st = stability.detach().float().clamp(0.0, 1.0)
        g = gains.detach().float()
        novelty, regime, max_sim = self.candidate_novelty(ck)
        self._update_stats(regime, h, novelty, g)

        h_norm = (h / (h.mean() + 1e-8)).clamp(0.1, 10.0)
        write_score = (
            h_norm.pow(hardness_power)
            * (novelty + 1e-3).pow(novelty_power)
            * (st + 1e-3).pow(stability_power)
        )

        writes = 0
        merges = 0
        for c in range(self.num_regimes):
            idx = torch.where(regime == c)[0]
            if idx.numel() == 0:
                continue
            budget = min(self._budget(c), idx.numel())
            chosen = idx[torch.topk(write_score[idx], k=budget, largest=True).indices]
            for i in chosen:
                valid_idx = torch.where(self.valid[c])[0]
                if valid_idx.numel() == 0:
                    slot = 0
                    self.keys[c, slot].copy_(ck[i])
                    self.values[c, slot].copy_(cv[i])
                    self.utility[c, slot] = max(float(g[i].item()), 0.0)
                    self.valid[c, slot] = True
                    writes += 1
                    continue

                sims = ck[i] @ _normalize(self.keys[c, valid_idx]).t()
                best_local = int(torch.argmax(sims).item())
                best_slot = int(valid_idx[best_local].item())
                if float(sims[best_local].item()) >= self.merge_similarity:
                    m = self.merge_momentum
                    merged_key = _normalize(((1.0 - m) * self.keys[c, best_slot] + m * ck[i]).unsqueeze(0))[0]
                    self.keys[c, best_slot].copy_(merged_key)
                    self.values[c, best_slot].mul_(1.0 - m).add_(cv[i], alpha=m)
                    self.utility[c, best_slot] = (
                        self.utility_momentum * self.utility[c, best_slot]
                        + (1.0 - self.utility_momentum) * g[i]
                    )
                    merges += 1
                    continue

                invalid = torch.where(~self.valid[c])[0]
                if invalid.numel() > 0:
                    slot = int(invalid[0].item())
                else:
                    redundancy = self._slot_redundancy(c)
                    retention = self.utility[c] - self.redundancy_weight * redundancy
                    retention = retention.masked_fill(~self.valid[c], float("inf"))
                    slot = int(torch.argmin(retention).item())
                self.keys[c, slot].copy_(ck[i])
                self.values[c, slot].copy_(cv[i])
                self.utility[c, slot] = max(float(g[i].item()), 0.0)
                self.valid[c, slot] = True
                writes += 1

        self._recompute_cores()
        self.num_consolidations.add_(1)
        return {
            "writes": float(writes),
            "merges": float(merges),
            "mean_novelty": float(novelty.mean().item()),
            "mean_write_score": float(write_score.mean().item()),
        }

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        valid_n = int(self.valid.sum().item())
        return {
            "valid_slots": float(valid_n),
            "capacity": float(self.capacity),
            "mean_utility": float(self.utility[self.valid].mean().item()) if valid_n else 0.0,
            "mean_regime_gain": float(self.regime_gain_ema.mean().item()),
            "mean_regime_novelty": float(self.regime_novelty_ema.mean().item()),
            "consolidations": float(self.num_consolidations.item()),
        }
