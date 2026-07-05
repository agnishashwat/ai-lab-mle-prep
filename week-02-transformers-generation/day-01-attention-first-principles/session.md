# Week 02 Day 01 — Attention from First Principles

## Objective

Derive scaled dot-product attention (Q/K/V) from scratch, understand *why* the scale is
\(1/\sqrt{d_k}\), implement causal masking correctly, and know the three positional-encoding
families (sinusoidal, learned, RoPE) well enough to argue trade-offs. Build the `attention`
primitive that the rest of Week 2's decoder will stand on.

## Prerequisites

- Week 1: stable softmax (max-subtraction, `keepdim`), masking discipline (`masked_fill(-inf)`
  *before* softmax, never `-inf` a value you then sum), tensor-shape fluency, `F.cross_entropy`.
- Matrix calculus comfort (you have it): Jacobians, quadratic forms.

## Required Reading

- "Attention Is All You Need" §3.2 (Scaled Dot-Product + Multi-Head) — just that section.
- Optional: the RoPE paper's §3 (rotary embeddings) — skim the rotation intuition only.

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Week 1)

1. `∂CE/∂z` for softmax+cross-entropy, and why it sums to zero.
2. The masking rule: when do you use `masked_fill(-inf)` vs multiply-by-mask, and the one absolute "never"?
3. Loop invariants + order, with the exact placement of `.backward()` under grad-accum and *why*.
4. bf16 Adam memory per param, and where the savings actually come from.

### 0:10-0:35 - Concept Explanation (First Principles)

Attention as *content-based soft dictionary lookup*: queries \(Q\), keys \(K\), values \(V\).
Score = \(QK^\top\) (similarity), normalize with softmax over keys, use it to average values.
Full form: \(\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V\).
Why the \(1/\sqrt{d_k}\): with unit-variance independent \(q,k\) entries, \(q\cdot k\) has variance
\(d_k\); unscaled scores grow with \(d_k\), pushing softmax into saturation (near one-hot, tiny
gradients). Dividing by \(\sqrt{d_k}\) restores unit-variance logits. Causal mask \(M\): additive
\(-\infty\) upper-triangular so position \(t\) can't attend to \(>t\) (autoregressive property).
Multi-head: split \(d\) into \(h\) subspaces, attend independently, concat — lets the model attend
to multiple relation types at once. Positional info: attention is permutation-equivariant, so
order must be injected (sinusoidal absolute / learned absolute / RoPE relative).

### 0:35-1:00 - Derivation / Paper Deconstruction

Derive on the whiteboard: (a) the variance argument for \(1/\sqrt{d_k}\) explicitly
(\(\mathrm{Var}(q\cdot k)=d_k\sigma^4\)); (b) the softmax Jacobian
\(\partial\,\text{softmax}_i/\partial z_j = s_i(\delta_{ij}-s_j)\) and what it implies for
saturated attention (vanishing gradient when a row is near one-hot); (c) why causal masking must
be additive-\(-\infty\) *inside* the softmax, not a post-hoc zeroing (renormalization). Move the
full derivation to `derivation.md` if it runs long.

### 1:00-1:45 - Coding / Experiment Task

Create `week-02-transformers-generation/code/attention.py` with stubs + a self-check harness
(same pattern as Week 1). Implement:

- `scaled_dot_product_attention(q, k, v, mask=None)` — shapes `(B, H, T, d_k)`; return output +
  attention weights. Stable softmax, additive mask.
- `causal_mask(T)` — the `(T, T)` additive `-inf` upper-triangular mask.
- `multi_head_attention` — a small `nn.Module`: project to Q/K/V, reshape to heads, attend, concat,
  output projection.

Self-check assertions to include: (1) attention weights are a valid distribution (rows sum to 1,
non-negative); (2) causal mask ⇒ strictly lower-triangular weights (no future leakage); (3) output
shape preserved; (4) a permutation of *values* permutes outputs identically (sanity); (5) numerical
stability at large `d_k` (no NaN, scaled version has smaller logit variance than unscaled).

### 1:45-1:55 - Interview Drill

"Why divide by \(\sqrt{d_k}\)?" and "How does causal masking work, and what breaks if you apply it
after the softmax?" — answer crisply with the variance and renormalization arguments.

### 1:55-2:00 - Deliverable & Next Session

Fill `deliverable.md` (attention primitive + the scale derivation). Next: assemble the full
transformer block (residual + LN + MLP) and study numerical stability (pre-LN vs post-LN).

## Checkpoint Questions

1. Derive \(\mathrm{Var}(q\cdot k)\) and explain the exact failure mode of dropping the scale.
2. Why must the causal mask be additive `-inf` *before* softmax rather than zeroing weights after?
3. What does the softmax Jacobian tell you about gradients through a saturated (near one-hot) attention row?
4. Multi-head vs single-head of the same total width: what does multi-head buy you?
5. Sinusoidal vs learned vs RoPE positional encodings — one-line trade-off for each.

## Frontier-Lab Interview Angle

Attention internals are the single most common transformer whiteboard question. Expect "implement
attention," "why the scale," "where's the causal mask," and shape-debugging on `(B, H, T, d_k)`.
RoPE and long-context extrapolation are hot follow-ups. This connects to your retrieval background:
attention *is* soft, differentiable retrieval over the sequence.
