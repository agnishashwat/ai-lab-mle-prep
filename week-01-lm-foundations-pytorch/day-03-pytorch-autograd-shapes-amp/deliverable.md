# Deliverable — Week 01 Day 03: Autograd, Shapes & Mixed Precision

## What I Built / Derived

- `code/autograd_shapes_amp.py` — 5 drills, all green (self-check 5/5, pytest 6/6):
  1. `masked_max_logit_mean` — broadcast bias + masked mean over real positions.
  2. `masked_mean` — correct denominator (`mask.sum()`), all-`False` guard (no NaN).
  3. `linear_ce_grad_wrt_W` — **manual backward** through `x→Linear→ReLU→Linear→CE`,
     matched to autograd.
  4. `fp16_overflow` — repro `exp(fp16)` → `+inf`.
  5. `bf16_stable_softmax` — the fix: max-subtraction under `torch.autocast(bf16)`, finite + normalized.
- `code/test_autograd_shapes_amp.py` — pytest mirror.
- Hand-derived the backprop chain and the bf16 mixed-precision memory number for 7B.

## Core Takeaways

- **Manual backward, `x→Linear→ReLU→Linear→CE`:** `g_z = (softmax(z)−onehot)/N` →
  `g_h = g_z @ W2` → `g_a = g_h * (h>0)` → `dW = g_a.T @ x` (lands on `(H,D_in)`).
  Shape-matching at each step is both the correctness check and the interview signal.
- **Mixed-precision memory is NOT smaller.** bf16 Adam = **16 bytes/param** (fp32 master 4 +
  fp32 m 4 + fp32 v 4 + bf16 weights 2 + bf16 grads 2) = **112 GB for 7B — identical to fp32.**
  Savings come from **activations** (~halved) and **~2× matmul throughput**, not optimizer/weight state.
- **Two numerics safeguards, don't conflate:** max-subtraction ⇒ no `exp` *overflow*;
  `z − logsumexp(z)` form ⇒ no log-of-zero *underflow*. fp16's *training*-killer is gradient
  **underflow** (< ~6e-5 → 0), fixed by **loss scaling / `GradScaler`**; bf16 avoids it via wide exponent range.
- **Masking has two jobs, two tools:** `masked_fill(-inf)` *before* a max/softmax (mask within the
  reduction axis); multiply-or-boolean-index *after* (exclude whole positions from a mean).
  **Never `-inf` a value you're going to sum.** (Cost me a live `-inf` bug today.)
- **Graph scoping:** `detach()` scopes by *tensor* (cut one value out, e.g. DPO reference logprobs);
  `no_grad()` scopes by *code region* (inference, the optimizer step). `no_grad` around the branch
  you're optimizing ⇒ no gradient ⇒ silent no-learning (or a `requires_grad` error).
- **`view` vs `reshape`:** `view` is zero-copy but raises on non-contiguous layout; `reshape`
  falls back to a copy. `x[bool_mask]` flattens to 1-D and is data-dependent (bad for `torch.compile`);
  `(x*mask).sum()/mask.sum()` is the static-shape production form.

## Interview Angle

- "Walk me through `loss.backward()`": seed grad 1 on the scalar loss → reverse-topo traversal →
  local VJPs with cached activations → accumulate (`+=`) into **leaf** `.grad` → graph freed.
- "fp16 NaN at step ~50, debug it": **NaN at step 0 = config bug; NaN after stable steps = something
  grew.** Process: localize in time/space (grad-norm per step, `set_detect_anomaly`) → rank hypotheses
  (grad spike/overflow, climbing grad norm, bad batch, LR) → cheap checks → **matched** fix
  (GradScaler / grad clip / LR), bf16 as robust default — not the reflex first move.

## Scores

- Recall quiz (D2): 2.5/3 (overflow-vs-underflow swap; "sum over positions" reflex).
- Concept checkpoints: 1.5/3 (`no_grad`-vs-`detach` on the optimized branch; `view`/`reshape` mechanism).
- bf16 memory drill: 0.8/1 (missed the second bf16 grad copy → 98 vs correct 112 GB).
- Coding: 5/5 (manual backward correct first try; 3 latent bugs caught in review: in-place mutation,
  CPU-hardcoded zero guard, `-inf`-then-sum).
- Interview drills: ~0.65 (systems facts present; need to package as a debugging *narrative*).

## Next

Day 4 — minimal end-to-end training loop (tie today's autograd + AMP into a real step).
