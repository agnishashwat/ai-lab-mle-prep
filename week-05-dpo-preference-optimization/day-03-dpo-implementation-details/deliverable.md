# Deliverable — Week 05 Day 03: DPO Implementation Details

**Artifact:** `week-05-dpo-preference-optimization/code/dpo_trainer.py` — seven stages,
**51 checks**, built on Week 2's `TinyGPT`, clean under `-W error::RuntimeWarning`.

## Core Takeaway

> A DPO implementation is simple only after five contracts are made explicit.

The objective is one `logsigmoid`, but a correct run also needs: causal sequence logprobs
reduced at the right level; a frozen reference evaluated through an exactly reproducible
preprocessing path; a stable cDPO loss with the correct finite optimum; an SFT checkpoint
cloned into both policy and reference roles; and diagnostics that record marginal likelihood,
length, and KL rather than only pair accuracy.

The pairwise loss cannot report what it does not control. Rising accuracy and falling loss can
coexist with both sequence logprobs falling, length exploitation, or movement into behavior that
never appears in the fixed preference dataset.

## What I Built

`dpo_trainer.py` contains:

1. Causal, masked, fp32 `sequence_logprob_sum` and `sequence_logprob_mean`.
2. Bitwise-exact frozen-reference logprob caching.
3. One stable loss covering DPO, cDPO, and length-normalized DPO.
4. Chosen-completion SFT warmup.
5. A TinyGPT DPO loop with step-0 and post-update snapshots of:
   chosen logprob, rejected logprob, log-ratio margin, implicit-reward magnitude,
   preference accuracy, exact subset KL, and expected generated length.
6. Controlled checks for the length prior, bf16 accumulation error, cDPO's finite optimum,
   and the corrected `β` / label-smoothing boundaries.

## Key Equations

### Sequence logprob

```text
log π_θ(y|x) = Σ_t m_t log softmax(z_t)[y_t]
```

`m_t` includes completion tokens and EOS, and excludes prompt and padding. Logits at position
`t` predict the target at `t+1`, so the logits and token ids are shifted exactly once.

### DPO preference logit

```text
chosen_logratio   = log π_θ(y_w|x) − log π_ref(y_w|x)
rejected_logratio = log π_θ(y_l|x) − log π_ref(y_l|x)

Δ̂ = chosen_logratio − rejected_logratio
M  = β Δ̂
```

### Conservative DPO

```text
L_cDPO(M)
  = −(1−ε) log σ(M) − ε log σ(−M)

dL/dΔ̂  = β[σ(βΔ̂) − (1−ε)]
d²L/dΔ̂² = β² σ(βΔ̂)σ(−βΔ̂) > 0

M*  = log((1−ε)/ε)
Δ̂* = log((1−ε)/ε) / β
```

cDPO therefore does not merely slow a diverging margin. Past `Δ̂*`, the gradient reverses it.
At `ε=0`, the finite optimum disappears. At `ε=0.5`, the target is a tie: the loss is minimized
at zero margin and is not constant.

### Sum versus mean

For current average per-token drift `d̄_L` and a new token with drift `d_new`:

```text
sum:   lengthen iff d_new > 0
mean:  lengthen iff d_new > d̄_L
```

The normalization is a length prior, not cosmetic scaling.

## Measurements

### cDPO's loss-side cap is Day 1's data-side cap divided by β

At `β=0.3`, the analytic optimum had gradient below `3e-17` and positive curvature:

| ε | `M* = log((1−ε)/ε)` | `Δ̂* = M*/β` |
|---:|---:|---:|
| 0.05 | 2.944439 | **9.814797** |
| 0.10 | 2.197225 | **7.324082** |
| 0.20 | 1.386294 | **4.620981** |
| 0.35 | 0.619039 | **2.063464** |

This is the same cap Day 1 measured from noisy labels, now imposed by the loss.

### Reference caching

| quantity | measured result |
|---|---:|
| reference forwards during preprocessing | **2** — chosen and rejected |
| cache storage | **2 fp32 scalars/pair = 8 bytes/pair** |
| cached vs freshly computed logprobs | **bitwise exact** |
| reference-model calls during training | **0 by API construction** |

The trainer has no `model_ref` argument. Caching is exact because the model is frozen and the
completions are fixed; it ceases to be valid if tokenization, chat template, truncation, EOS,
masking, checkpoint, precision, or attention implementation changes.

### Length prior

On a controlled fixture with constant per-token drift `+0.2`:

| reduction | measured margin slope versus length |
|---|---:|
| sum | **+0.200000** |
| mean | **0.000000** |

A marginal drift of `+0.10` passes the sum rule (`>0`) and fails the mean rule when the running
average is `+0.20`.

### Numerical precision

At `T=256`, bf16 log-softmax/reduction differed from the fp32 path by **1.026245**. The
representative preference signal `β·margin` was **0.050000**, so accumulation error was
**20.525× larger than the signal**.

The stable extreme-logit probes were:

```text
fp32 z = +40: log(sigmoid(z)) becomes exactly 0;
              logsigmoid(z) preserves a small negative value.

fp32 z = −100: −log(sigmoid(z)) becomes +∞;
               −logsigmoid(z) remains exactly 100 to tolerance.
```

### SFT initialization

Chosen-completion SFT produced:

```text
token NLL:                 2.071305 → 0.396558
chosen mean logprob/token: −2.081233 → −0.474107
```

Cloning the warmed checkpoint into **both** roles gave exactly zero initial chosen and rejected
log-ratios and initial DPO NLL `+log 2 = 0.693147`.

### TinyGPT DPO run

The full run (`β=0.2`, `ε=0.1`, 24 updates) produced:

| metric | step 0 | step 24 |
|---|---:|---:|
| DPO loss | 0.693147 | **0.549205** |
| chosen logprob | −6.76001 | **−4.54599** |
| rejected logprob | −5.95584 | **−8.07871** |
| log-ratio margin | 0.0000 | **4.3369** |
| mean `|r̂|` | 0.0000 | **0.7751** |
| exact subset KL | 0.00000 | **0.92182** |
| expected length | 2.1227 | **2.3867** |

On the smaller fast fixture, chosen logprob instead fell by **−0.59155** while rejected fell by
**−2.06221** and the margin still rose. The sign changed across sampled fixtures; it is not a
reliable diagnostic for `μ ≠ π_ref`. The stable mismatch diagnostic remains Day 2's KL comparison
at matched margin/loss.

## Implementation Notes

### Correct causal alignment

```python
logits = model(input_ids)[:, :-1]
targets = input_ids[:, 1:]
token_logps = F.log_softmax(logits, dim=-1, dtype=torch.float32).gather(
    -1, targets.unsqueeze(-1)
).squeeze(-1)
sequence_logps = token_logps.masked_fill(~completion_mask, 0.0).sum(-1)
```

The first implementation correctly shifted and gathered but returned `(B,T−1)`: the time
reduction was missing. The contract is one scalar per sequence, shape `(B,)`.

### SFT mutates one checkpoint; the caller creates the two roles

```text
pretrained checkpoint
  → SFT on chosen completions
  → warmed checkpoint
      ├─ deep copy, frozen = π_ref
      └─ deep copy, trainable = initial π_θ
```

Returning two models from `sft_warmup` would mix optimization with pipeline ownership. The helper
mutates its model and returns only its loss history.

### Separate optimization from logging

```text
optimization:
  forward with graph → loss.backward() → optimizer.step()

snapshot:
  fresh eval/no_grad forward → Python-float diagnostics
```

Step 0 is measured before training. A record labelled step `k` is measured after update `k`.
The final record is recomputed, never produced by relabelling a stale dictionary.

## Corrected Claims

1. **The no-op boundary is `β→0`, not uniformly `β→∞`.** The gradient with respect to the
   log-ratio margin carries an outer factor `β`.
2. **`ε=0.5` does not make the loss constant.** It makes zero margin the unique target.
3. **The chosen-logprob sign does not diagnose data/reference mismatch.** It can change with
   fixture and training phase. Compare KL at matched margin/loss.
4. **Naive `log(sigmoid(z))` has two dtype-dependent failures.** Positive saturation erases
   small loss; negative underflow turns finite NLL into infinity.
5. **At `π_θ=π_ref`, the initial log-ratio is exactly zero even on low-probability data.**
   What can be large is the ratio required by the labels after optimization.

## Bugs and What They Generalize To

1. **Missing NLL minus sign — third consecutive day.** The first cDPO expression was a
   log-likelihood. The asymmetric probe is `M=0 ⇒ NLL=+log 2`, never `−log 2`.
2. **Missing time reduction.** Correct `(B,T−1)` token values are not yet `(B,)` sequence values.
   Only the explicit reduction crosses that boundary.
3. **`finally: return None`.** A return in `finally` overrides successful returns and suppresses
   exceptions. `finally` is for restoration, not a fallback result.
4. **Discarded functional return.** Calling `policy_logps.reduced(...)` without assignment changes
   nothing; the frozen dataclass is not mutated.
5. **Integer mean in PyTorch.** `(margin > 0)` must be cast to float before `.mean()`. NumPy silently
   promotes integer reductions; PyTorch requires the dtype choice explicitly.
6. **Operator precedence in logging.** `step+1 % k` parses as `step + (1 % k)`. More importantly,
   pre-update records cannot be appended as post-update metrics.
7. **Absolute policy logprob is not implicit reward.** The correct object is
   `β·(log π_θ−log π_ref)`, not `β·|log π_θ|`.

## Interview Answer

Prompt:

> Your DPO run has rising preference accuracy and falling loss, but sampled outputs are getting
> worse. What do you inspect?

Answer:

1. Split the pair margin into chosen and rejected policy logprobs.
2. Compare `M=βΔ̂` with the noise-calibrated cap `log((1−ε)/ε)`; accuracy saturates too early.
3. Audit generated length, EOS rate, margin versus length, and length-controlled quality.
4. Identify the data generator `μ`, confirm that the same SFT checkpoint seeded policy and
   reference, and measure exact or sampled `KL(π_θ‖π_ref)` at matched margin/loss.
5. Fix the diagnosed mechanism: early-stop first; then choose cDPO, length normalization, a chosen
   NLL term, a rebuilt SFT/reference checkpoint, or cleaned preference data.

The compact line is:

> Accuracy and loss see pair ordering. I instrument marginals, cap, length, and KL because those
> expose what the pairwise objective cannot see.

## Session Record

| block | score |
|---|---:|
| cold recall | **3.9 / 6 (65%)** |
| D1–D2 recall quiz | **8 / 14 (57%)** |
| cDPO derivation | **3 / 4** |
| interview drills | **4 / 9**, then both retrieval resets complete |
| coding | **51 / 51 checks** |

The learner wrote the core sequence-logprob, loss, cache, and SFT bodies with hints and review.
The final optimization/snapshot separation in `train` was refactored for him at his request.

## Reusable Artifact

- **Path:** `week-05-dpo-preference-optimization/code/dpo_trainer.py`
- **Run all:** `.venv/bin/python -W error::RuntimeWarning week-05-dpo-preference-optimization/code/dpo_trainer.py`
- **Run one stage:** append `logprobs`, `cache`, `loss`, `length`, `precision`, `sft`, or `train`
- **Reuse on Day 4:** compare IPO, KTO, ORPO, and SimPO against the same three axes:
  finite margin, reference/data requirements, and length reduction.
