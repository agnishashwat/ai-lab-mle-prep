"""Day 4 -- a minimal, correct, reproducible training loop.

The MODEL is deliberately tiny; the LOOP is the object of study. You implement the
three functions marked TODO. Everything else (toy task, model, harness) is done.

Toy task: "successor" -- token[t+1] = (token[t] + 1) mod V. A tiny embedding+linear
model can drive cross-entropy to ~0, so a correct loop shows a clean, reproducible
loss drop; a buggy loop (missing zero_grad, wrong schedule, eval leaking grads) shows.

Self-check (quick, no pytest needed):
    ./.venv/bin/python ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/train.py

Full check (pytest):
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_train.py -q
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================================
# Provided: reproducibility, toy data, tiny model. (Read these, don't edit.)
# ==================================================================================
def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a CPU run."""
    random.seed(seed)
    torch.manual_seed(seed)


def make_batch(
    gen: torch.Generator, B: int, T: int, V: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Successor task. Returns (x, y), each (B, T) long.
        x[:, t]   = random start unrolled by +1 mod V
        y[:, t]   = (x[:, t] + 1) mod V   == the next token
    Draw the start token with `gen` so runs are reproducible.
    """
    start = torch.randint(0, V, (B, 1), generator=gen, device=device)
    steps = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    x = (start + steps) % V                                # (B, T)
    y = (x + 1) % V
    return x, y


class TinyLM(nn.Module):
    """Embedding -> Linear. Enough capacity to learn the successor map."""

    def __init__(self, V: int, d: int = 32):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.head = nn.Linear(d, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, V)
        return self.head(self.emb(x))


@dataclass
class TrainConfig:
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
    eval_every: int = 50
    eval_iters: int = 10
    seed: int = 0
    device: str = "cpu"
    log: bool = False


# ==================================================================================
# TODO 1 -- Learning-rate schedule: linear warmup, then cosine decay.
#   step in [0, warmup_steps):  linear 0 -> base_lr.  Use (step+1)/warmup_steps so
#                               step 0 is a small nonzero lr and lr == base_lr at the
#                               end of warmup.
#   step in [warmup_steps, total_steps): cosine from base_lr down to min_lr, i.e.
#       progress = (step - warmup_steps) / (total_steps - warmup_steps)   in [0, 1]
#       lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi * progress))
#   step >= total_steps: return min_lr.
#   Return a plain float.
# ==================================================================================
def lr_at_step(
    step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float
) -> float:
    if step < warmup_steps: return (step+1)/warmup_steps * base_lr
    progress = torch.clip(torch.tensor(step - warmup_steps)/(total_steps - warmup_steps), min=0, max=1)
    return min_lr + 0.5*(base_lr-min_lr)*(1+torch.cos(torch.pi * progress)).item()
    


# ==================================================================================
# TODO 2 -- Evaluation. Average cross-entropy over `eval_iters` fresh batches.
#   Requirements the harness checks:
#     - put the model in eval mode, and restore its ORIGINAL mode on exit
#     - run under torch.no_grad() (no graph, no .grad populated)
#     - return a python float (mean loss over the eval batches)
#   Use F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).
# ==================================================================================
@torch.no_grad()  # you may keep or replace this; the point is: no grads during eval
def evaluate(model: nn.Module, gen: torch.Generator, cfg: TrainConfig) -> float:
    was_training = model.training
    sum = 0
    model.eval()
    for iter in range(cfg.eval_iters):
        x, y_eval = make_batch(gen, cfg.B, cfg.T, cfg.V)    # [B, T]
        logits = model(x)                                   # [B, T, V]   
        sum += F.cross_entropy(logits.reshape(-1, cfg.V), y_eval.reshape(-1))
    model.train(was_training)
    return sum.item()/cfg.eval_iters





# ==================================================================================
# TODO 3 -- The training loop. Return a dict with:
#     "loss_history": list[float]   -- train loss per OPTIMIZER step
#     "lr_history":   list[float]   -- lr used at each OPTIMIZER step
#     "eval_history": list[tuple[int, float]]  -- (step, eval_loss) at eval points
#
#   Do it in this order, and get the invariants right:
#     1. set_seed(cfg.seed) FIRST (before model/opt creation) for reproducibility.
#     2. build TinyLM(cfg.V, cfg.d), move to device, model.train().
#     3. optimizer = AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay).
#        (lr is overwritten each step via the schedule -- see below.)
#     4. one Generator for train data, a SEPARATE one for eval (so eval batches are
#        fixed and don't perturb the train stream): seed them off cfg.seed.
#     5. for each optimizer step:
#          - set every param_group's 'lr' to lr_at_step(step, ...).
#          - accumulate over cfg.grad_accum micro-batches:
#                logits = model(x); loss = CE(...); (loss / grad_accum).backward()
#          - clip_grad_norm_(params, cfg.grad_clip)
#          - optimizer.step(); optimizer.zero_grad(set_to_none=True)
#          - record the (unscaled) mean loss for this step and the lr.
#          - every cfg.eval_every steps (and at the end), call evaluate(...) and
#            record it. Remember evaluate() must leave the model back in train mode.
#
#   Keep it CPU/fp32 (no autocast needed for the harness). Reproducibility is the
#   headline test: two calls with the same cfg must give identical loss_history.
# ==================================================================================
def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    model = TinyLM(cfg.V, cfg.d).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    gen_train = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    gen_eval = torch.Generator(device=cfg.device).manual_seed(cfg.seed+1)
    result = {
        'loss_history': [], 'lr_history': [], 'eval_history': []
        }
    
    for step in range(cfg.steps):
        lr = lr_at_step(step, cfg.warmup_steps, cfg.steps, cfg.base_lr, cfg.min_lr)
        for pg in opt.param_groups:
            pg['lr'] = lr
        result['lr_history'].append(lr)
        step_loss = 0
        for iter in range(cfg.grad_accum):
            x, y = make_batch(gen_train, cfg.B, cfg.T, cfg.V)    # [B, T]
            logits = model(x)                                   # [B, T, V]   
            loss = F.cross_entropy(logits.reshape(-1, cfg.V), y.reshape(-1))/cfg.grad_accum
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad()
        result["loss_history"].append(step_loss)
        if (step+1) % cfg.eval_every == 0:
            result["eval_history"].append((step, evaluate(model, gen_eval, cfg)))
    
    result["eval_history"].append((step, evaluate(model, gen_eval, cfg)))
    return result


# ==================================================================================
# Quick self-check harness -- do not edit.
# ==================================================================================
def _check(name: str, ok: bool) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main() -> None:
    results = []

    # 1. LR schedule shape: warmup up, cosine down, endpoints correct.
    try:
        W, Tot, base, mn = 10, 100, 1.0, 0.1
        lr0 = lr_at_step(0, W, Tot, base, mn)
        lr_peak = lr_at_step(W, W, Tot, base, mn)          # end of warmup ~ base
        lr_mid = lr_at_step(W + (Tot - W) // 2, W, Tot, base, mn)
        lr_end = lr_at_step(Tot, W, Tot, base, mn)         # >= total -> min_lr
        warmup_up = lr_at_step(1, W, Tot, base, mn) > lr0
        ok = (
            0 < lr0 < base
            and abs(lr_peak - base) < 1e-6
            and mn < lr_mid < base
            and abs(lr_end - mn) < 1e-6
            and warmup_up
        )
        results.append(_check("lr_at_step (warmup up, cosine to min_lr)", ok))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"lr_at_step (raised: {e})", False))

    # 2. Reproducibility: same cfg -> identical loss history.
    try:
        cfg = TrainConfig(steps=120, eval_every=40)
        h1 = train(cfg)["loss_history"]
        h2 = train(cfg)["loss_history"]
        results.append(_check("reproducible (same seed == identical losses)", h1 == h2))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"reproducible (raised: {e})", False))

    # 3. It actually learns: final train loss well below the uniform baseline ln(V).
    try:
        cfg = TrainConfig(steps=300)
        out = train(cfg)
        baseline = math.log(cfg.V)
        final = sum(out["loss_history"][-10:]) / 10
        results.append(
            _check(f"learns (final {final:.3f} << ln V {baseline:.3f})", final < 0.5 * baseline)
        )
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"learns (raised: {e})", False))

    # 4. eval leaves model in train mode and populates no grads.
    try:
        cfg = TrainConfig(steps=1)
        set_seed(cfg.seed)
        model = TinyLM(cfg.V, cfg.d)
        model.train()
        gen = torch.Generator().manual_seed(123)
        _ = evaluate(model, gen, cfg)
        no_grads = all(p.grad is None for p in model.parameters())
        results.append(_check("evaluate (restores train mode + no grads)", model.training and no_grads))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"evaluate (raised: {e})", False))

    # 5. lr_history matches the schedule the loop claims to use.
    try:
        cfg = TrainConfig(steps=50, warmup_steps=10)
        out = train(cfg)
        expected0 = lr_at_step(0, cfg.warmup_steps, cfg.steps, cfg.base_lr, cfg.min_lr)
        ok = abs(out["lr_history"][0] - expected0) < 1e-9 and len(out["lr_history"]) == cfg.steps
        results.append(_check("lr_history wired to schedule", ok))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"lr_history (raised: {e})", False))

    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
