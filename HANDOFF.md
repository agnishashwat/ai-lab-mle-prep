# Handoff — ai-lab-mle-prep

Living state doc for cross-chat continuity. The user starts a **new chat per day**; this
file (plus `progress.md` and each day's `session.md`) carries context forward. The
`ai-lab-mle-prep-day` skill reads this at the start of a session and updates it at the end.

## Current position

- **Week 01** — LM Foundations & PyTorch Fluency
- **Next up: Day 04 — Minimal Training Loop** (Days 01-03 complete)

## Next session

Start with the prompt: **"Start Week 1 Day 4"**.

Day 4 plan (`week-01-lm-foundations-pytorch/day-04-.../session.md`): a minimal end-to-end
training loop that ties together D2 (cross-entropy) + D3 (autograd + AMP). Expect: dataset →
forward → loss → `backward` → optimizer step → eval, with correct `zero_grad`, an AMP/`GradScaler`
path, and grad-norm logging (reinforce the "NaN after stable steps = something grew" debug habit).
Quiz D3 cold: `loss.backward()` narrative, the 16-bytes/param memory anchor, and masked-fill-before-max
vs multiply-after.

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
