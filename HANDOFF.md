# Handoff — ai-lab-mle-prep

Living state doc for cross-chat continuity. The user starts a **new chat per day**; this
file (plus `progress.md` and each day's `session.md`) carries context forward. The
`ai-lab-mle-prep-day` skill reads this at the start of a session and updates it at the end.

## Current position

- **Week 01 — COMPLETE** (Days 01-05 done). LM Foundations & PyTorch Fluency.
- **Next up: Week 02 Day 01 — Transformers + generation** (decoder-only mini-implementation).

## Next session

Start with the prompt: **"Start Week 2 Day 1"** (or **"Generate Week 2 sessions"** first if the
`week-02-.../day-*/session.md` files don't exist yet — check `curriculum.md` for the block list).

Week 2 theme: transformers + generation → decoder-only mini-implementation. Likely arc: attention
from scratch (QKV, scaled dot-product, causal mask), multi-head, positional encodings, the
decoder block, then a KV-cache + sampling (greedy/temp/top-k/top-p) generation loop. Reuse the
Week-1 harness (`train.py` loop, `eval_ablation.py`) to train/eval the mini-transformer.
Quiz Week-1 cold at the start: `∂CE/∂z = p − e_y`; loop invariants w/ backward INSIDE accum
(memory not math); eval hygiene two-bugs; bf16 memory (16 B/param); masked-mean static form.

## Completed

- **W1 D1 — Role Map + Baseline Diagnostic.** Built role map + competency matrix; stood up
  the lab env (`requirements.txt`, `smoke_test.py`); completed `fluency_probe.py` 5/5
  (softmax, cross_entropy, masked_mean, sgd_step, + 3-D `seq_token_logprobs` stretch).
  Baseline: ~20 min, slow on `gather` + masking. Deliverable written; the derivation of
  `∂CE/∂z` was covered as a preview.
- **W1 D2 — Token-Level Language Modeling.** Built `code/token_lm.py` (log_softmax via
  logsumexp, cross_entropy, perplexity, analytic `ce_grad`) + `test_token_lm.py` (6/6 pass,
  incl. autograd AND finite-difference gradient checks). Derived `∂CE/∂z = p − e_y` cleanly on
  the whiteboard (fixed an `e^{z_k}` index slip and a gather-vs-scatter bug live). Quiz 3.5/4,
  drill 0.9/1.
- **W1 D3 — Autograd / Shapes / Mixed Precision.** Built `code/autograd_shapes_amp.py` +
  test (5/5 self-check, 6/6 pytest): masked-mean drills, **manual backward** through
  `Linear→ReLU→Linear→CE` (matched autograd first try), fp16 overflow repro + bf16-autocast fix.
  Derived bf16 memory = **16 bytes/param = 112 GB for 7B (same as fp32)**; savings are activations +
  throughput. Quiz 2.5/3, checkpoints 1.5/3, interview drills ~0.65. Live-caught 3 latent code bugs
  (in-place input mutation, CPU-hardcoded zero guard, and a `-inf`-then-sum masking bug).
- **W1 D4 — Minimal Training Loop.** Built `code/train.py` + `test_train.py` (5/5 self-check,
  5/5 pytest): `lr_at_step` (warmup→cosine), `evaluate` (no_grad + mode save/restore), and the
  full `train` loop (seed-first, per-step LR, grad-accum, clip, AdamW, periodic eval, histories).
  Toy successor task drives CE 2.82→0.022; two runs bit-identical. Quiz D3 ≈4.1/5 (masked-mean
  static form still the soft spot). Loop coding was the hard part — needed heavy scaffolding and
  hit every classic bug live: `AdamW(model)` not `.parameters()`, flat warmup (missing `*base_lr`),
  inverted eval `%`, backward-inside-accum on a summed tensor, missing grad-accum `/N`, and the
  **double-softmax** (feeding `softmax` to `F.cross_entropy`). Drill ≈3.6/5 — under-answered the
  eval-hygiene split. This is his weakest area so far; wants more loop reps.

- **W1 D5 — Eval + Ablation Discipline.** Built `code/eval_ablation.py` + test (5/5 self-check,
  6/6 pytest): `evaluate_loss` (hygiene), `run_with_val` (dual train/val curves + `train_pool`/
  `label_noise` to make overfit observable), `overfit_one_batch` (2.82→0.0016), `ablation`
  (seed-averaged one-var sweep, LR U-shape: 3e-5→2.08, 3e-3→0.06, 3e2→2.92 w/ clip off). All four
  of his impls were correct first-try on logic; the 3 failures were the scaffold's flawed task
  assumptions, which became the day's best lessons: (1) `cross_entropy` K-dim shape bug is SILENT
  when `T==V` (treats time axis as class axis); (2) a noiseless consistent rule CAN'T overfit
  (memorize==generalize) → need label noise; (3) grad clipping tames Adam so hard LR-divergence
  only shows at LR≈300 with clip off. Quiz D4 4.2/5 (regressed: said backward AFTER accum loop —
  must be INSIDE, reason=memory; missed clip in the invariant list). Drills D1 0.85 / D2 0.7
  (led with capacity/data instead of "verify eval first"). Also wrote week-summary + retrospective.

## Learner profile (quick facts)

- Strong: PhD-level math/optimization, senior ML eng, real post-training experience
  (SFT/DPO/GRPO, retrieval, personalization). RL intuition is genuinely good.
- **Top gaps:** (1) ML systems / memory-cost reasoning; (2) making RL depth
  interview-crisp and derivable on demand.
- Prior work to reference in examples: Amazon personalized prompt-augmentation (RAG:
  "pet food" → "pet food for cats"), DPO to beat a Sonnet teacher via HF.
- Teaching style that works: derive from first principles, quiz from prior day, give code
  stubs + self-check harness (don't pre-solve), connect to his projects and later weeks.

## Conventions

- Run code: `./.venv/bin/python ai-lab-mle-prep/<path>` from `/Users/sshivam/1point3acres`
  (torch 2.12, Python 3.14, pytest installed).
- **Auto-commit + push is authorized** (`agnishashwat` identity; see
  `.cursor/rules/auto-commit.mdc`). Confirm only for destructive git ops or secrets.
- Each day ends by writing that day's `deliverable.md` and updating `progress.md` + this file.

## Running facts to reinforce (spaced repetition)

- LM loss: `−(1/BT) Σ log p_θ(y_{b,t} | x_{b,<t})`; one-hot label collapses the vocab sum.
- `∂CE/∂z = softmax(z) − onehot(y)` = "prediction − target"; sums to 0 (mass is moved).
- Stable softmax: subtract row-max, `keepdim=True` on both max and sum.
- Token log-prob idiom (works for `(N,V)` and `(B,T,V)`):
  `log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)`.
- Masked mean denominator is `mask.sum()`, not `numel()` (matters for variable-length seqs).
- Optimizer memory: params + grads + **2 Adam moments**; 7B fp32 ≈ 28+28+56 ≈ 112 GB
  before activations. (D1 gap — reproduced cold on D2. Adam `m`=1st moment, `v`=2nd moment.)
- `log_softmax(z) = z − logsumexp(z)`; NEVER `softmax(z).log()` (underflow → −inf).
- gather = *read* value at index (loss reads log p at target); scatter = *write* value at index
  (build the one-hot for the gradient). `scatter` scalar form uses `value=`, not `src=`.
- Mean-CE gradient carries a **1/N**: per-position grad is `p − e_y`, averaged over N positions.
- Perplexity = exp(CE) = effective branching factor. Floor 1, uniform baseline = V,
  **unbounded above** (PPL > V ⇒ worse than random). PPL 30 ⇔ CE ≈ 3.4 nats.
- CE vs KL: identical gradient (H(q) const in θ), but KL is the right *metric* for soft targets
  (distillation/label smoothing) — it reads 0 at a perfect match, CE bottoms out at H(q).
- Mixed-precision (bf16 Adam) = **16 bytes/param** (fp32 master 4 + fp32 m 4 + fp32 v 4 + bf16 W 2 +
  bf16 grad 2) = **112 GB for 7B, identical to fp32**. Savings = activations (~½) + ~2× matmul throughput,
  NOT optimizer/weight state. (D3 slip: forgot the 2nd bf16 grad copy → 98 vs 112.)
- fp16: exponent 5 bits (range ~6e-5..65504), training-killer = gradient **underflow** → loss scaling /
  `GradScaler`. bf16: exponent 8 bits (fp32 range), mantissa 7 → no loss scaling needed.
- Two numerics safeguards: max-subtraction ⇒ no `exp` overflow; `z−logsumexp(z)` ⇒ no log-of-zero underflow.
- Masking: `masked_fill(-inf)` **before** max/softmax (within reduction axis); multiply/boolean-index
  **after** (exclude positions from a mean). **Never `-inf` a value you then sum.**
- `detach` scopes by tensor (DPO reference logprobs); `no_grad` scopes by code region (inference,
  optimizer step). `no_grad` on the branch you optimize ⇒ silent no-learning.
- `.grad` populated only on **leaf** tensors; backward **accumulates** (`+=`) → need `zero_grad`; graph
  freed after `.backward()` unless `retain_graph=True`.
- `view` = zero-copy, raises on non-contiguous; `reshape` = copy-fallback. `x[bool_mask]` flattens to 1-D,
  data-dependent (bad for compile); `(x*mask).sum()/mask.sum()` is the static-shape form.
- Interview debug habit: **NaN at step 0 = config bug; NaN after stable steps = something grew** (check
  grad norms first, `set_detect_anomaly`, then matched fix).
- Training-loop five invariants + order: **seed→(model/opt)→ per step: set LR into param_groups →
  accumulate (`backward` INSIDE accum loop) → clip → step → zero_grad → eval under `eval()`+`no_grad()`.**
- `F.cross_entropy` eats **raw logits** (does log_softmax internally). Feeding `softmax(logits)` =
  double-softmax = SILENT underfit (no crash). Flatten `(-1,V)` + class-id targets, no unsqueeze.
- Grad-accum: divide each micro-loss by `grad_accum` (backward SUMS grads → "sum of means" must
  become "mean of means"). Effective batch = micro × accum × world_size.
- Eval hygiene = TWO bugs: `no_grad` off → activation graph kept (OOM) + `.grad` leaks into next
  train step; `eval()` off → dropout stays on (noisy metric) + BatchNorm uses/corrupts running stats.
- LR override lives in `opt.param_groups[i]['lr']` (optimizer reads it live each `step()`); the
  constructor `lr=` is inert in a hand-rolled loop but becomes `base_lr` under `lr_scheduler`.
- Reproducibility needs **data `torch.Generator`s seeded explicitly** (fresh Generator = system
  entropy); `set_seed`/`torch.manual_seed` only covers global RNG (model init).
- AdamW vs Adam: Adam folds L2 `λθ` into `g` so it passes through `1/√v̂` (uneven decay). AdamW
  applies `−ηλθ` directly after the adaptive step → uniform decay. Warmup guards the step-0 spike
  (Adam's `v̂` is garbage early); cosine anneals to settle. LR↔batch: `Var(g_B)=σ²/B` → linear rule
  `η∝B` (Goyal) / sqrt rule `η∝√B`, up to a critical batch size.
- `F.cross_entropy` K-dim form: input `(N,C,d₁…)`, target `(N,d₁…)`. Passing `(B,T,V)` vs `(B,T)`
  does NOT crash when `T==V` — silently treats time axis as classes → garbage loss. ALWAYS
  `reshape(-1,V)`/`reshape(-1)`. Square shapes hide shape bugs.
- Overfitting needs NON-generalizing signal: a noiseless, globally-consistent rule can't overfit
  (memorize == generalize). To see a train≪val gap you must inject label noise into a finite pool.
- Grad clipping (norm cap) makes Adam nearly unkillable on easy tasks: LR≤30 still converged with
  `grad_clip=1.0`; divergence only at LR≈300 clip-off, NaN by ~1000. Adam step is ~lr-bounded.
- Ablation: change ONE var, FIX the compute budget, ≥3 seeds, report mean ± spread; claim a result
  only if the effect beats seed noise (non-overlapping intervals). One seed can lie.
- Debug reflex: VERIFY THE EVAL PIPELINE before diagnosing overfitting — a broken eval mimics
  catastrophic overfit (train 0.05 / val 5.5 was a shape bug, not the model). Senior tell.
- `dataclasses.replace(base, **{var: val}, seed=s)` for a dynamic-key config edit; `replace(base,
  var=val)` fails (passes literal kwarg `var`).
