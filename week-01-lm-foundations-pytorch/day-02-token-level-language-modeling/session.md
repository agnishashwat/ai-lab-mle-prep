# Week 01 Day 02 — Token-Level Language Modeling

## Objective

Derive the chain logits -> logprobs -> cross-entropy -> perplexity from first principles, and be able to reproduce the softmax-cross-entropy gradient on a whiteboard.

## Prerequisites

- Softmax, log rules, basic probability
- Day 1 environment (torch, pytest)

## Required Reading

- nanoGPT loss computation (the `F.cross_entropy` call + how logits/targets are shaped) — or any compact cross-entropy / perplexity refresher
- Optional: a short note on label smoothing

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 1)

1. Name the 5 competency axes and your weakest.
2. Your one-line positioning statement.
3. Warm-up: define softmax and explain why it is shift-invariant.

### 0:10-0:35 - Concept Explanation (First Principles)

- Autoregressive factorization: `p(x_1..x_T) = Π_t p(x_t | x_<t)`.
- The model emits logits `z ∈ R^V` per position -> softmax -> categorical over the vocab.
- Training = maximum likelihood = minimize the negative log-likelihood of the true next token (teacher forcing).
- Cross-entropy vs NLL vs KL; where the "label distribution" comes from (one-hot target).

### 0:35-1:00 - Derivation

Derive cleanly:

```text
CE          = -(1/N) Σ_t log softmax(z_t)[y_t]
perplexity  = exp(CE)                      # per-token
∂CE/∂z      = softmax(z) - onehot(y)
```

Show the gradient result step by step (the classic ask). Note that a uniform model over `V` tokens has perplexity exactly `V`.

### 1:00-1:45 - Coding Task

In `code/`, implement from scratch and unit-test against torch:

- `log_softmax(z)` with the max-subtraction trick (numerical stability)
- `cross_entropy(logits, targets)` matching `F.cross_entropy` to ~1e-6
- `perplexity(logits, targets)`
- Verify `∂CE/∂z == softmax(z) - onehot(y)` via autograd (and/or finite differences)

Add a `pytest` test file.

### 1:45-1:55 - Interview Drill

Out loud: "Derive the gradient of softmax cross-entropy," and "A model has perplexity 30 — what does that mean, and what are the floor and ceiling?"

### 1:55-2:00 - Deliverable & Next Session

`deliverable.md`: the module, the tests, and the gradient derivation. Next: Day 3 (autograd / shapes / mixed precision).

## Checkpoint Questions

1. Why subtract the max inside log-softmax?
2. Why is `∂CE/∂z = softmax(z) - onehot(y)` so clean?
3. What perplexity does a uniform distribution over `V` give, and why?
4. How do teacher forcing and exposure bias relate to this objective?
5. How does padding/masking change the per-token average loss?

## Frontier-Lab Interview Angle

"Derive the softmax-CE gradient" and "explain perplexity / read this loss curve" are staple whiteboard and ML-coding asks. Being fast and exact here signals implementation fluency.
