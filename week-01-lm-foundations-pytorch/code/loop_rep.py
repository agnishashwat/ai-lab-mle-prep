"""Day 4 -- COLD-START REP. Write the loop from memory, timed.

Rules of the rep:
  - Do NOT open train.py. Write lr_at_step / evaluate / train from the spec below.
  - Target: < 12 min for a green run. Then DELETE your three functions and do it again.
  - The harness is strict: it fails on the classic silent bugs (flat warmup, double
    softmax, missing grad-accum scaling, eval leaking grads / wrong mode).

Run:
    ./.venv/bin/python ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/loop_rep.py

SPEC (all you get):
  lr_at_step(step, warmup, total, base, min) -> float
      linear warmup base*(step+1)/warmup for step<warmup; else cosine base->min over
      the remaining steps; clamp to min at/after total.
  evaluate(model, gen, cfg) -> float
      mean CE over cfg.eval_iters batches; no_grad; eval() then RESTORE prior mode.
  train(cfg) -> {"loss_history":[float], "lr_history":[float], "eval_history":[(int,float)]}
      seed first; AdamW(params, base_lr, weight_decay); per step: set lr into param_groups,
      accumulate grad_accum micro-batches (loss/grad_accum, backward inside), clip, step,
      zero_grad; eval every cfg.eval_every steps and once at the end.
  Data: make_batch(gen,B,T,V) successor task. CE wants RAW logits, flattened to (-1,V).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_batch(gen, B, T, V, device="cpu"):
    start = torch.randint(0, V, (B, 1), generator=gen, device=device)
    x = (start + torch.arange(T, device=device).unsqueeze(0)) % V
    return x, (x + 1) % V


class TinyLM(nn.Module):
    def __init__(self, V, d=32):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.head = nn.Linear(d, V)

    def forward(self, x):
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


# ============================ WRITE THESE THREE ============================
def lr_at_step(step, warmup_steps, total_steps, base_lr, min_lr) -> float:
    if step < warmup_steps:
        return base_lr*(step+1)/warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min(max(min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi * progress)), 0), 1)
     


def evaluate(model, gen, cfg) -> float:
    was_training = model.training
    model.eval()
    CE_loss = 0
    for _ in range(cfg.eval_iters):
        x, y = make_batch(gen, cfg.B, cfg.T, cfg.V, cfg.device)
        y_model = model(x)
        CE_loss = CE_loss + F.cross_entropy(y_model.reshape(-1, cfg.V), y.reshape(-1)).item()
    model.train(was_training)
    return CE_loss/cfg.eval_iters




def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    model = TinyLM(cfg.V, cfg.d).to(cfg.device)
    gen_train = torch.Generator(cfg.device).manual_seed(cfg.seed)
    gen_eval = torch.Generator(cfg.device).manual_seed(cfg.seed + 1)
    opt = torch.optim.AdamW(params = model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    res = {"loss_history":[], "lr_history":[], "eval_history":[]}
    for step in range(cfg.steps):
        lr = lr_at_step(step, cfg.warmup_steps, cfg.steps, cfg.base_lr, cfg.min_lr)
        res['lr_history'].append(lr)
        for pg in opt.param_groups:
            pg['lr']=lr
        sum=0
        for _ in range(cfg.grad_accum):
            x,y=make_batch(gen_train,cfg.B, cfg.T,cfg.V, cfg.device)
            loss=F.cross_entropy(model(x).reshape(-1,cfg.V),y.reshape(-1))/cfg.grad_accum
            sum+=loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad()
        res['loss_history'].append(sum)
        if (step+1)%cfg.eval_every==0:
            res["eval_history"].append((step, evaluate(model,gen_eval,cfg)))
    res["eval_history"].append((cfg.steps-1, evaluate(model,gen_eval,cfg)))
    return res


    raise NotImplementedError
# ==========================================================================


def _check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    r = []

    # 1. schedule with REAL base_lr (catches the flat-warmup / missing *base_lr bug)
    try:
        W, Tot, base, mn = 30, 300, 3e-3, 3e-4
        ok = (
            0 < lr_at_step(0, W, Tot, base, mn) < base
            and lr_at_step(1, W, Tot, base, mn) > lr_at_step(0, W, Tot, base, mn)
            and abs(lr_at_step(W, W, Tot, base, mn) - base) < 1e-6      # warmup ends AT base
            and mn < lr_at_step(W + (Tot - W) // 2, W, Tot, base, mn) < base
            and abs(lr_at_step(Tot, W, Tot, base, mn) - mn) < 1e-6
        )
        r.append(_check("lr_at_step (real base_lr; warmup ends at base)", ok))
    except Exception as e:  # noqa: BLE001
        r.append(_check(f"lr_at_step (raised: {e})", False))

    # 2. reproducibility (float histories, exact equality)
    try:
        cfg = TrainConfig(steps=120)
        r.append(_check("reproducible", train(cfg)["loss_history"] == train(cfg)["loss_history"]))
    except Exception as e:  # noqa: BLE001
        r.append(_check(f"reproducible (raised: {e})", False))

    # 3. learns HARD (strict threshold -> double-softmax / missing-scaling plateaus fail this)
    try:
        out = train(TrainConfig(steps=300))
        final = sum(out["loss_history"][-10:]) / 10
        r.append(_check(f"learns strict (final {final:.3f} < 0.10)", final < 0.10))
    except Exception as e:  # noqa: BLE001
        r.append(_check(f"learns (raised: {e})", False))

    # 4. eval hygiene: float return, train-mode restored, no grads populated
    try:
        cfg = TrainConfig(steps=1)
        set_seed(cfg.seed)
        m = TinyLM(cfg.V, cfg.d)
        m.train()
        v = evaluate(m, torch.Generator().manual_seed(1), cfg)
        r.append(_check(
            "evaluate (float + restores train + no grads)",
            isinstance(v, float) and m.training and all(p.grad is None for p in m.parameters()),
        ))
    except Exception as e:  # noqa: BLE001
        r.append(_check(f"evaluate (raised: {e})", False))

    # 5. eval_history is [(int,float)] and fires on the right count
    try:
        cfg = TrainConfig(steps=100, eval_every=50)
        eh = train(cfg)["eval_history"]
        shape_ok = all(isinstance(s, int) and isinstance(l, float) for s, l in eh)
        # evals at steps 49 and 99, plus one final -> at least 2 boundary evals
        r.append(_check("eval_history typed + non-empty", shape_ok and len(eh) >= 2))
    except Exception as e:  # noqa: BLE001
        r.append(_check(f"eval_history (raised: {e})", False))

    # 6. grad-accum scaling: accum=4,B=8 must still learn (missing /grad_accum -> too-hot LR)
    try:
        out = train(TrainConfig(steps=300, grad_accum=4, B=8))
        final = sum(out["loss_history"][-10:]) / 10
        r.append(_check(f"grad-accum scaling (accum=4,B=8 -> final {final:.3f} < 0.30)", final < 0.30))
    except Exception as e:  # noqa: BLE001
        r.append(_check(f"grad-accum (raised: {e})", False))

    print(f"\n{sum(r)}/{len(r)} passed")


if __name__ == "__main__":
    main()
