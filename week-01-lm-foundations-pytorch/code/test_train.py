"""pytest checks for Day 4 minimal training loop.

Run:
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_train.py -q
"""
from __future__ import annotations

import math

import torch

from train import TinyLM, TrainConfig, evaluate, lr_at_step, set_seed, train


def test_lr_schedule_shape():
    W, Tot, base, mn = 10, 100, 1.0, 0.1
    assert 0 < lr_at_step(0, W, Tot, base, mn) < base
    assert lr_at_step(1, W, Tot, base, mn) > lr_at_step(0, W, Tot, base, mn)  # warmup rising
    assert abs(lr_at_step(W, W, Tot, base, mn) - base) < 1e-6                 # peak at base
    assert mn < lr_at_step(W + (Tot - W) // 2, W, Tot, base, mn) < base       # cosine mid
    assert abs(lr_at_step(Tot, W, Tot, base, mn) - mn) < 1e-6                 # floor at min_lr


def test_reproducible():
    cfg = TrainConfig(steps=120, eval_every=40)
    assert train(cfg)["loss_history"] == train(cfg)["loss_history"]


def test_learns():
    cfg = TrainConfig(steps=300)
    out = train(cfg)
    final = sum(out["loss_history"][-10:]) / 10
    assert final < 0.5 * math.log(cfg.V)


def test_eval_restores_mode_and_no_grads():
    cfg = TrainConfig(steps=1)
    set_seed(cfg.seed)
    model = TinyLM(cfg.V, cfg.d)
    model.train()
    gen = torch.Generator().manual_seed(123)
    evaluate(model, gen, cfg)
    assert model.training
    assert all(p.grad is None for p in model.parameters())


def test_lr_history_wired():
    cfg = TrainConfig(steps=50, warmup_steps=10)
    out = train(cfg)
    expected0 = lr_at_step(0, cfg.warmup_steps, cfg.steps, cfg.base_lr, cfg.min_lr)
    assert len(out["lr_history"]) == cfg.steps
    assert abs(out["lr_history"][0] - expected0) < 1e-9
