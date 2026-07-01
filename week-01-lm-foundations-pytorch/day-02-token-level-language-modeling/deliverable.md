# Deliverable - Week 01 Day 02: Token-Level Language Modeling

## What I Built / Derived

- From-scratch, gradient-checked token-LM primitives in `code/token_lm.py`:
  `log_softmax` (max-subtraction / logsumexp form), `cross_entropy` (log-softmax + gather),
  `perplexity` (`exp(mean CE)`), and `ce_grad_analytic` (the analytic `(p - onehot)/N`).
- `code/test_token_lm.py`: 6 pytest checks incl. numerical stability on ~1000-scale logits,
  uniform-PPL-equals-V, and **two independent gradient checks** (autograd + finite differences).
- Whiteboard derivation of `∂CE/∂z = softmax(z) - onehot(y)`, done cleanly with the correct
  `e^{z_k}` index in the quotient rule.

## Core Takeaway

An LM is `T` coupled `V`-way classifiers sharing weights. Training = MLE = minimizing the mean
NLL of the true next token under teacher forcing. With a one-hot target, cross-entropy collapses
to `-log p_y`, and its gradient w.r.t. the logits is exactly "prediction minus target,"
`p - e_y`, which sums to zero (mass is redistributed, never created).

## Key Equations

- `CE = -(1/N) Σ log softmax(z_t)[y_t]`
- `perplexity = exp(CE)`  — the *effective branching factor*
- `∂CE/∂z = softmax(z) - onehot(y)` (per position); mean-CE gradient carries a `1/N`
- `log_softmax(z) = z - logsumexp(z)`, with `logsumexp` computed after subtracting `max(z)`

## Implementation Notes

- **gather vs scatter**: loss *reads* `log p` at the target (`gather`); the gradient *writes* a 1
  at the target to build the one-hot (`scatter`). Reached for `gather` first — wrong tool.
- `scatter` scalar overload wants `value=1.0`, not `src=1.0` (`src` must be a tensor).
- The `1/N` factor: per-position grad is `p - e_y`, but mean-CE averages over N, so the
  autograd gradient is `(p - e_y)/N`. Most common place to drop a factor.
- Stability: use `z - logsumexp(z)`, never `softmax(z).log()` (underflow → `-inf`).

## Failure Modes / Debugging Notes

- Naive `log(softmax)` underflows on large/confident logits; the max trick is mandatory.
- Perplexity is **unbounded above**: PPL > V means worse-than-random. V is the uniform baseline,
  not a ceiling. Floor is 1.
- Teacher forcing → exposure bias: train conditions on gold prefixes, inference on the model's
  own (possibly wrong) outputs — a distribution shift never seen at train time.

## Interview Answer

Prompt:

> Derive the gradient of softmax cross-entropy w.r.t. the logits, and explain perplexity
> (what does PPL = 30 mean, and what are its bounds?).

Answer:

`L = -log p_y`, `p = softmax(z)`. Chain rule: `∂L/∂z_k = -(1/p_y) ∂p_y/∂z_k`. Quotient rule on
`e^{z_y}/Σ e^{z_j}` (denominator derivative brings down `e^{z_k}`) gives
`∂p_y/∂z_k = p_y δ_{yk} - p_y p_k`, so `∂L/∂z_k = p_k - δ_{yk}`, i.e. `∂L/∂z = p - e_y`
("prediction − target," sums to 0). Perplexity `= exp(CE)` is the effective branching factor:
PPL 30 ≈ as uncertain as a uniform choice among 30 tokens (CE ≈ 3.4 nats). Floor 1 (perfect);
uniform model = V; PPL can exceed V (confidently wrong), so it's unbounded above.

## Reusable Artifact

- Path: `week-01-lm-foundations-pytorch/code/token_lm.py` (+ `test_token_lm.py`)
- How to reuse: import `cross_entropy` / `perplexity` / `log_softmax` as reference
  implementations; `ce_grad_analytic` is a template for hand-derived backward passes in later
  weeks (policy-gradient / DPO / GRPO losses).
