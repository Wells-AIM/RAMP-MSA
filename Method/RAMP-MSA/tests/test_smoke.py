from pathlib import Path

import torch

from ramp_msa.config import load_config
from ramp_msa.data import build_dataloaders
from ramp_msa.model import RAMPModel
from ramp_msa.losses import task_loss_per_sample
from ramp_msa.trainer import _apply_missingness, deterministic_missing_matrix


def test_forward_and_memory_init():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "synthetic_regression.yaml")
    cfg.data.synthetic_cfg.n_train = 32
    cfg.data.synthetic_cfg.n_valid = 8
    cfg.data.synthetic_cfg.n_test = 8
    cfg.train.batch_size = 8
    cfg.model.d_model = 32
    cfg.model.interaction_dim = 32
    cfg.memory.key_dim = 32
    cfg.model.nhead = 4
    cfg.memory.num_regimes = 4
    cfg.memory.slots_per_regime = 4

    loaders, info = build_dataloaders(cfg)
    model = RAMPModel(info.text_dim, info.audio_dim, info.vision_dim, cfg)
    batch = next(iter(loaders["train"]))
    out = model(batch, use_memory=False)
    assert out["final_logits"].shape[0] == batch["label"].shape[0]
    assert out["components"].shape[1] == 7
    recon = out["components"].sum(dim=1)
    assert torch.allclose(recon, out["h_base"], atol=1e-4, rtol=1e-4)

    policy, stability = model.oracle_policy(out, batch["label"], view_dropout=0.1)
    hard = task_loss_per_sample(out["base_logits"], batch["label"], "regression")
    model.memory.initialize(out["slow_key"], policy, hard, seed=1, kmeans_iters=5)
    assert model.memory.ready()
    out2 = model(batch, use_memory=True)
    assert out2["retrieved_policy"].shape[-1] == 7
    assert stability.min() >= 0 and stability.max() <= 1


def test_deterministic_missingness_and_input_leakage():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "synthetic_regression.yaml")
    cfg.data.synthetic_cfg.n_train = 24
    cfg.data.synthetic_cfg.n_valid = 8
    cfg.data.synthetic_cfg.n_test = 8
    cfg.train.batch_size = 8
    cfg.model.d_model = 32
    cfg.model.interaction_dim = 32
    cfg.memory.key_dim = 32
    cfg.model.nhead = 4
    cfg.memory.num_regimes = 4
    cfg.memory.slots_per_regime = 4

    first = deterministic_missing_matrix(24, 0.2, "mosi", "train", 5576, 0)
    repeat = deterministic_missing_matrix(24, 0.2, "mosi", "train", 5576, 0)
    next_epoch = deterministic_missing_matrix(24, 0.2, "mosi", "train", 5576, 1)
    assert torch.equal(torch.from_numpy(first), torch.from_numpy(repeat))
    assert not torch.equal(torch.from_numpy(first), torch.from_numpy(next_epoch))
    assert int((first == 0).sum()) == round(24 * 3 * 0.2)
    assert (first.sum(axis=1) >= 1).all()

    loaders, info = build_dataloaders(cfg)
    model = RAMPModel(info.text_dim, info.audio_dim, info.vision_dim, cfg).eval()
    clean = next(iter(loaders["train"]))
    table = torch.ones(len(loaders["train"].dataset), 3, dtype=torch.bool)
    table[clean["index"], 1] = False
    changed = dict(clean)
    changed["audio"] = torch.randn_like(clean["audio"]) * 1000
    masked_clean = _apply_missingness(clean, table)
    masked_changed = _apply_missingness(changed, table)
    with torch.no_grad():
        clean_out = model(masked_clean, use_memory=False)["final_logits"]
        changed_out = model(masked_changed, use_memory=False)["final_logits"]
    assert torch.equal(clean_out, changed_out)
