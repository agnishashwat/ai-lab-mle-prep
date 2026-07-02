# Deliverable - Week 01 Day 04: Minimal Transformer Training Loop

## What I Built / Derived

`code/train.py` — a minimal, correct, reproducible training loop (+ `test_train.py`, 5/5).
The model is deliberately trivial (`TinyLM` = embedding -> linear) on a "successor" toy
task (`token[t+1] = (token[t]+1) mod V`); the **loop** is the artifact:

- `lr_at_step` — linear warmup then cosine decay to `min_lr`.
- `evaluate` — averages CE over N batches under `no_grad`, saves/restores model mode.
- `train` — seed-first, per-step LR override, gradient accumulation, grad clip, AdamW,
  periodic eval, and full loss/lr/eval history.

Trace (steps=300, V=16, so the uniform baseline is ln 16 ≈ 2.773):

- train loss @ [0,10,30,60,120,300] = [2.816, 2.667, 1.649, 0.339, 0.056, 0.022]
- lr @ [0,15,30,150,300] = [1e-4, 1.6e-3, 3e-3(peak), 1.9e-3, 3e-4(floor)]
- reproducibility: two calls with the same cfg produce **bit-identical** loss_history.

## Core Takeaway

A training loop is five invariants in a fixed order: **(1) seed before any model/optim
construction, (2) set the scheduled LR into every param_group, (3) accumulate grads over
micro-batches with `backward` inside the accum loop, (4) clip then `step` then `zero_grad`
outside it, (5) eval under `eval()` + `no_grad()` and restore mode.** The model can be
anything; getting these five right is what "implement a training loop" is actually testing.

## Key Equations

- Warmup then cosine:
  `lr(t) = base * (t+1)/warmup` for `t < warmup`;
  else `min + 0.5*(base-min)*(1 + cos(pi * progress))`, `progress = (t-warmup)/(total-warmup)`.
- Effective batch size = `micro_batch x grad_accum x world_size`.
- Grad-accum scaling: divide each micro-loss by `grad_accum` so summed grads = mean over the
  full effective batch ("sum of means" -> "mean of means").
- AdamW decoupled decay: `theta -= eta * m_hat/(sqrt(v_hat)+eps)  -  eta * lambda * theta`
  (decay applied directly, NOT through the `1/sqrt(v)` preconditioner like Adam's L2).
- LR<->batch coupling: `Var(g_B) = sigma^2 / B` -> bigger batch = less noise = bigger step.
  Linear rule `eta ~ B` (Goyal), sqrt rule `eta ~ sqrt(B)` (constant update variance),
  valid up to a **critical batch size**.

## Implementation Notes

- CE takes **raw logits**, never `softmax(logits)` (it does log_softmax internally) — the
  double-softmax bug: no crash, loss just plateaus high / underfits.
- Flatten for CE: `logits.reshape(-1, V)`, `targets.reshape(-1)` as class ids (no unsqueeze).
- LR schedule is inert at construction (`AdamW(..., lr=base)`) because it's overwritten every
  step — but it becomes `initial_lr`/`base_lr` if you swap in `torch.optim.lr_scheduler`.
- Reproducibility needs the **data Generators seeded explicitly** (fresh `torch.Generator()`
  starts from system entropy); a separate eval generator keeps eval from perturbing the train
  stream. `set_seed` only covers the global RNG used for model init.

## Failure Modes / Debugging Notes

- Missing `zero_grad` -> grads accumulate across *steps* -> effective LR balloons -> diverge.
  Misplaced (inside accum loop) -> only last micro-batch counts.
- Eval without `no_grad()` -> activation graph kept (OOM/slow) + `.grad` can leak into the
  next train step. Eval without `eval()` -> dropout stays on (noisy metric) + BatchNorm uses
  batch stats and corrupts its running stats.
- No grad clip -> one spike -> oversized update -> NaN. (NaN@step0 = config; NaN after stable
  steps = something grew -> check grad-norm history first.)
- No seeding -> non-reproducible, can't A/B ablations or reproduce a divergence.

## Interview Answer

Prompt:

> Write a training loop from scratch. What do people most commonly get wrong, and why does
> each one bite?

Answer:

Skeleton: `set_seed`; build model + AdamW; `model.train()`; per step: set scheduled LR into
param_groups, then for each of `grad_accum` micro-batches `loss = CE(logits, y)/grad_accum;
loss.backward()`, then `clip_grad_norm_`, `opt.step()`, `opt.zero_grad(set_to_none=True)`;
periodically `evaluate()` under `eval()`+`no_grad()`. Common tells that lose points: (1)
missing/misplaced `zero_grad` (grads accumulate across steps -> divergence); (2) eval without
`eval()`+`no_grad()` — two separate bugs: memory blow-up + grad leak from `no_grad`, and
dropout/BatchNorm misbehavior from `eval()`; (3) no seeding (non-reproducible); (4) no grad
clip (a single spike NaNs the run); (5) feeding `softmax(logits)` to `cross_entropy` (double
softmax — silent, just underfits). Senior signal = naming the *silent* ones (grad leak,
double softmax) and the mechanism, not just the crashers.

## Reusable Artifact

- Path: `code/train.py` (self-check: run the file; pytest: `test_train.py`)
- How to reuse: the seed of the Week-1 "tiny LM harness." Day 5 extends it with eval/ablations
  and (natural next step) an AMP + GradScaler path reusing the Day-3 mixed-precision muscle.
