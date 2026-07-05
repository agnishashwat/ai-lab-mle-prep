"""Day 5 pytest -- eval + ablation harness.

Run:
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_eval_ablation.py -q
"""
from __future__ import annotations

import math

import torch

from train import TinyLM, set_seed
from eval_ablation import (
    RunConfig,
    evaluate_loss,
    run_with_val,
    overfit_one_batch,
    ablation,
)


def test_evaluate_loss_hygiene():
    cfg = RunConfig()
    set_seed(cfg.seed)
    model = TinyLM(cfg.V, cfg.d)
    model.train()
    g = torch.Generator().manual_seed(7)
    v = evaluate_loss(model, g, cfg, iters=5)
    assert isinstance(v, float)
    assert 0.0 < v < 10.0
    assert model.training                                   # mode restored
    assert all(p.grad is None for p in model.parameters())  # no grad leak


def test_run_with_val_learns_and_logs():
    cfg = RunConfig(steps=200, eval_every=50)
    out = run_with_val(cfg)
    assert len(out["train_curve"]) > 0 and len(out["val_curve"]) > 0
    assert out["final_val"] < 0.5 * math.log(cfg.V)


def test_run_with_val_reproducible():
    cfg = RunConfig(steps=150, eval_every=50)
    a = run_with_val(cfg)
    b = run_with_val(cfg)
    assert a["train_curve"] == b["train_curve"]
    assert a["val_curve"] == b["val_curve"]


def test_overfit_one_batch():
    """The marquee sanity test: data->loss->optimizer wired right => loss -> ~0."""
    cfg = RunConfig()
    losses = overfit_one_batch(cfg, lr=1e-2, steps=300)
    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-2, f"pipeline can't overfit one batch (final {losses[-1]:.3e})"


def test_ablation_shape_and_optimum():
    # clipping disabled so the too-high LR arm can actually diverge (see eval_ablation.py note)
    base = RunConfig(steps=200, eval_every=50, grad_clip=1e9)
    table = ablation(base, "base_lr", [3e2, 3e-3, 3e-5], seeds=[0, 1])
    assert len(table) == 3
    for row in table:
        assert set(row) >= {"value", "mean", "min", "max", "seeds"}
        assert len(row["seeds"]) == 2
        assert row["min"] <= row["mean"] <= row["max"]
    by_val = {round(r["value"], 8): r["mean"] for r in table}
    assert by_val[round(3e-3, 8)] < by_val[round(3e2, 8)]
    assert by_val[round(3e-3, 8)] < by_val[round(3e-5, 8)]


def test_overfit_gap_observable():
    """Small fixed train pool => memorization => train << val (generalization gap)."""
    cfg = RunConfig(steps=400, eval_every=50, train_pool=2, label_noise=True, base_lr=3e-3)
    out = run_with_val(cfg)
    final_train = out["train_curve"][-1][1]
    final_val = out["val_curve"][-1][1]
    assert final_train < final_val - 0.3
