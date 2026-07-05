"""Day 5 -- evaluation + ablation discipline on top of the Day-4 loop.

We reuse the Day-4 machinery (TinyLM, make_batch, lr_at_step) and add the *eval
layer*: a train/val split, a run that logs BOTH curves, an ablation sweep across
seeds reporting mean +/- range, and the canonical "overfit one batch" sanity test.

You implement the four functions marked TODO. Everything else is done.

Self-check (quick):
    ./.venv/bin/python ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/eval_ablation.py

Full check (pytest):
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_eval_ablation.py -q
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from train import TinyLM, make_batch, lr_at_step, set_seed


# ==================================================================================
# Provided: config + a fixed finite train pool so overfitting is observable.
# ==================================================================================
@dataclass
class RunConfig:
    V: int = 16
    d: int = 32
    B: int = 32
    T: int = 16
    steps: int = 300
    warmup_steps: int = 30
    base_lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 1
    eval_every: int = 25
    eval_iters: int = 10
    train_pool: int = 0        # 0 => infinite fresh train batches; >0 => fixed pool of this many batches (enables overfit)
    label_noise: bool = False  # if True, the pool's labels are RANDOM (memorizable but non-generalizing) => real overfit gap
    seed: int = 0
    device: str = "cpu"


def _make_pool(cfg: RunConfig, gen: torch.Generator) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize a fixed list of `train_pool` (x, y) batches the model will see on repeat.

    With label_noise, replace the true successor labels with RANDOM ones: the model can
    still memorize this finite pool (train loss -> 0) but it has learned noise, so a fresh
    (true-successor) val set stays high. That divergence IS overfitting.
    """
    pool = []
    for _ in range(cfg.train_pool):
        x, y = make_batch(gen, cfg.B, cfg.T, cfg.V, cfg.device)
        if cfg.label_noise:
            y = torch.randint(0, cfg.V, (cfg.B, cfg.T), generator=gen, device=cfg.device)
        pool.append((x, y))
    return pool


# ==================================================================================
# TODO 1 -- evaluate_loss: mean cross-entropy over `iters` batches drawn from `gen`.
#   Requirements (same hygiene as Day 4):
#     - save the model's current mode; call model.eval(); restore on exit.
#     - run under torch.no_grad().
#     - CE over F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)); average over iters.
#     - return a python float.
# ==================================================================================
@torch.no_grad()
def evaluate_loss(model: nn.Module, gen: torch.Generator, cfg: RunConfig, iters: int) -> float:
    was_training = model.training
    model.eval()
    loss = 0
    for _ in range(iters):
        x, y = make_batch(gen, cfg.B, cfg.T, cfg.V, cfg.device)
        loss += F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1)).item()
    model.train(was_training)
    return loss/iters



# ==================================================================================
# TODO 2 -- run_with_val: train while logging BOTH train and val curves.
#   Build model/opt exactly like Day-4 train() (seed first, AdamW, per-step LR schedule,
#   grad-accum with backward INSIDE the accum loop, clip, step, zero_grad).
#   Data:
#     - one train Generator seeded off cfg.seed, one val Generator seeded off cfg.seed+1.
#     - if cfg.train_pool > 0: build a fixed pool once and cycle through it
#       (batch = pool[step % len(pool)]); else draw fresh make_batch each micro-step.
#   Every cfg.eval_every steps (and once at the very end) record BOTH:
#       train-loss-at-that-step and evaluate_loss(model, val_gen, cfg, cfg.eval_iters).
#   Return dict:
#     "train_curve": list[(step, train_loss)]
#     "val_curve":   list[(step, val_loss)]
#     "final_val":   float   # last recorded val loss
#   Reproducibility: same cfg -> identical curves.
# ==================================================================================
def run_with_val(cfg: RunConfig) -> dict:
    set_seed(cfg.seed)
    gen_train = torch.Generator(cfg.device).manual_seed(cfg.seed)
    gen_val = torch.Generator(cfg.device).manual_seed(cfg.seed+1)
    model = TinyLM(cfg.V, cfg.d).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    res = {"train_curve":[], "val_curve":[], "final_val": 0}

    if cfg.train_pool > 0:
        pool = _make_pool(cfg, gen_train)
    else: pool = None

    idx = 0
    for step in range(cfg.steps):
        lr = lr_at_step(step, cfg.warmup_steps, cfg.steps, cfg.base_lr, cfg.min_lr)
        for pg in opt.param_groups:
            pg['lr']=lr
        loss_total = 0

        for _ in range(cfg.grad_accum):
            if pool is None:
                x, y  = make_batch(gen_train, cfg.B, cfg.T, cfg.V, cfg.device)
            else:
                x, y = pool[idx%len(pool)]
            idx += 1
            loss = F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1))/cfg.grad_accum
            loss_total += loss.item()
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad()
        if (step+1)%cfg.eval_every==0:
            res["train_curve"].append((idx-1, loss_total))
            res["val_curve"].append((idx-1, evaluate_loss(model, gen_val, cfg, cfg.eval_iters)))
    res["final_val"]= evaluate_loss(model, gen_val, cfg, cfg.eval_iters)
    return res



# ==================================================================================
# TODO 3 -- overfit_one_batch: the canonical pipeline sanity test.
#   Train on ONE fixed batch for `steps`, no eval, no LR schedule needed (constant lr ok).
#   Return the list of per-step train losses. A correct data->loss->optimizer wiring
#   drives this to ~0. Keep it simple: set_seed, TinyLM, AdamW(lr), one (x,y), loop:
#   forward, CE, backward, clip(optional), step, zero_grad, record loss.
# ==================================================================================
def overfit_one_batch(cfg: RunConfig, lr: float = 1e-2, steps: int = 200) -> list[float]:
    set_seed(cfg.seed)
    gen_train = torch.Generator(cfg.device).manual_seed(cfg.seed)
    model = TinyLM(cfg.V, cfg.d).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    train_curve = []

    x, y = make_batch(gen_train, cfg.B, cfg.T, cfg.V, cfg.device)

    for step in range(steps):
        loss_total = 0
        for _ in range(cfg.grad_accum):
            loss = F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1))/cfg.grad_accum
            loss_total += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad()
        
        train_curve.append(loss_total)
    return train_curve

# ==================================================================================
# TODO 4 -- ablation: sweep ONE variable across a list of values, each over `seeds`,
#   holding everything else fixed. Return, per value, the mean and range of final_val.
#   Args:
#     base: RunConfig (the controls)
#     var:  str  -- attribute name to vary (e.g. "base_lr")
#     values: list -- the levels to try
#     seeds: list[int] -- run each level under each seed
#   Return: list of dicts, one per value, each:
#     {"value": v, "mean": float, "min": float, "max": float, "seeds": [final_val,...]}
#   Use dataclasses.replace to make a per-cell config with the varied attr + seed set.
# ==================================================================================
def ablation(base: RunConfig, var: str, values: list, seeds: list[int]) -> list[dict]:
    res = []
    for val in values:
        dict_val = {}
        dict_val['value'] = val
        val_scores = []
        for seed in seeds:
            cfg_ablation = replace(base, seed=seed, **{var: val})
            val_scores.append(run_with_val(cfg_ablation)["final_val"])
        dict_val['mean'] = np.mean(val_scores)
        dict_val['min'] = np.min(val_scores)
        dict_val['max'] = np.max(val_scores)
        dict_val['seeds'] = val_scores
        res.append(dict_val)
    return res


        




# ==================================================================================
# Quick self-check harness -- do not edit.
# ==================================================================================
def _check(name: str, ok: bool) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main() -> None:
    results = []

    # 1. evaluate_loss: sane range, restores mode, leaves no grads.
    try:
        cfg = RunConfig()
        set_seed(cfg.seed)
        model = TinyLM(cfg.V, cfg.d)
        model.train()
        g = torch.Generator().manual_seed(7)
        v = evaluate_loss(model, g, cfg, iters=5)
        ok = (
            isinstance(v, float)
            and 0.0 < v < 10.0
            and model.training                                  # restored
            and all(p.grad is None for p in model.parameters()) # no grads
        )
        results.append(_check("evaluate_loss (float, restores mode, no grads)", ok))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"evaluate_loss (raised: {e})", False))

    # 2. run_with_val: logs both curves, learns, reproducible.
    try:
        cfg = RunConfig(steps=200, eval_every=50)
        out = run_with_val(cfg)
        out2 = run_with_val(cfg)
        has_curves = len(out["train_curve"]) > 0 and len(out["val_curve"]) > 0
        learned = out["final_val"] < 0.5 * math.log(cfg.V)
        reproducible = out["val_curve"] == out2["val_curve"]
        results.append(_check("run_with_val (both curves, learns, reproducible)",
                              has_curves and learned and reproducible))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"run_with_val (raised: {e})", False))

    # 3. overfit_one_batch: collapses toward zero.
    try:
        cfg = RunConfig()
        losses = overfit_one_batch(cfg, lr=1e-2, steps=200)
        ok = losses[-1] < 1e-2 and losses[-1] < losses[0]
        results.append(_check(f"overfit_one_batch (final {losses[-1]:.2e} -> ~0)", ok))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"overfit_one_batch (raised: {e})", False))

    # 4. ablation: mid LR should beat both a too-high and a too-low LR.
    #    NOTE: clipping is disabled here -- with grad_clip on, Adam tames even LR=30 on
    #    this toy task, so the "too-high diverges" arm only appears once clipping is off.
    try:
        base = RunConfig(steps=200, eval_every=50, grad_clip=1e9)
        table = ablation(base, "base_lr", [3e2, 3e-3, 3e-5], seeds=[0, 1])
        by_val = {round(r["value"], 8): r["mean"] for r in table}
        mid = by_val[round(3e-3, 8)]
        hi = by_val[round(3e2, 8)]
        lo = by_val[round(3e-5, 8)]
        results.append(_check(f"ablation (mid {mid:.3f} < hi {hi:.3f} & < lo {lo:.3f})",
                              mid < hi and mid < lo))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"ablation (raised: {e})", False))

    # 5. overfitting is observable: small fixed pool -> train << val at the end.
    try:
        cfg = RunConfig(steps=400, eval_every=50, train_pool=2, label_noise=True, base_lr=3e-3)
        out = run_with_val(cfg)
        final_train = out["train_curve"][-1][1]
        final_val = out["val_curve"][-1][1]
        results.append(_check(f"overfit gap (train {final_train:.3f} << val {final_val:.3f})",
                              final_train < final_val - 0.3))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"overfit gap (raised: {e})", False))

    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
