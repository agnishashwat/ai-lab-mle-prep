# Deliverable - Week 02 Day 01: Attention from First Principles

## What I Built / Derived

- Reviewed my existing attention core in `anthropic_attention/` (SDPA with stable softmax +
  additive mask, `causal_mask`, multi-head split/merge, einsum & matmul parity, `F.sdpa` match) —
  already correct, so today focused on derivations + the missing positional-encoding piece.
- Built `code/positional_encoding.py` from scratch (6/6 self-checks PASS):
  - `sinusoidal_positional_encoding(T, D)` — sin on even dims, cos on odd, via strided-slice interleave.
  - RoPE: `rope_cos_sin` (geometric-frequency cos/sin cache), `_rotate_half`, `apply_rope`.
  - Harness verifies the key property: `<RoPE(q,m), RoPE(k,n)>` is constant along each
    constant-offset diagonal (spread 1.4e-6) ⇒ it truly encodes only `m−n` (relative position).
- Derived on paper: `Var(q·k)=d_k σ⁴`; softmax Jacobian `diag(s)−ssᵀ`; saturation → zero Jacobian.

## Core Takeaway

Attention is soft, differentiable retrieval: `softmax(QKᵀ/√d_k + M)V` is a content-based lookup
where softmax replaces a non-differentiable `argmax` over keys. Three structural facts generate
everything: the score is a dot product (→ the `1/√d_k` scale), it's normalized by softmax (→ the
saturation/gradient story), and it's permutation-equivariant (→ positional encodings are mandatory).

## Key Equations

- `Attn(Q,K,V) = softmax(QKᵀ/√d_k + M) V`
- `Var(q·k) = Σ_i Var(q_i k_i) = d_k σ⁴` (zero-mean, independent). Scale by `1/√d_k` ⇒ std is
  `d_k`-independent. (Dividing by `d_k` over-shrinks: variance `∝ 1/d_k` → uniform attention.)
- Softmax Jacobian: `∂s_i/∂z_j = s_i(δ_ij − s_j)`, matrix form `diag(s) − ssᵀ`.
  At a one-hot `s=e_k`, `diag(s)=ssᵀ` ⇒ Jacobian = 0 ⇒ no gradient through a saturated row.
- RoPE freq ladder: `θ_i = base^(−2i/D)`, geometric from ~1 (fast/local) to ~1/base (slow/global).

## Implementation Notes

- Sinusoidal interleave without a loop: `m[:, 0::2] = sin(angles)`, `m[:, 1::2] = cos(angles)`
  (strided slices are writable views of shape `(T, D/2)`). `math.log(base)` not `torch.log`
  (base is a Python float); positions must be float for `torch.outer`.
- RoPE convention consistency is the #1 silent bug: `_rotate_half` uses interleaved pairs
  (`x[...,0::2]`/`x[...,1::2]`), so `rope_cos_sin` must `repeat_interleave(angles, 2)` to match.
- RoPE is applied to Q and K only (not V), after the head split, before the scores.
- `x * cos` broadcasts `(T,D)` against `(...,T,D)` automatically — no explicit `broadcast_to` needed.

## Failure Modes / Debugging Notes

- Large `d_k` without the scale → saturated softmax → vanishing gradients (a *trainability* bug, not
  just numerics). This is the causal chain: big logit variance → near one-hot rows → zero Jacobian.
- Causal mask applied *after* softmax → rows sum to <1 (unnormalized); only fix is re-normalize,
  which just reimplements masking-before the hard way.
- RoPE past training length → low-frequency hands hit unseen angles → OOD attention scores. Fix:
  NTK-aware / YaRN base-rescaling to slow the hands and extend context without retraining.

## Interview Answer

Prompt:

> Why divide by √d_k, and how does causal masking work (what breaks if applied after softmax)?

Answer:

With unit-variance independent entries, `q·k` has variance `d_k`, so unscaled logits grow with
`d_k` and push softmax toward one-hot, where the Jacobian `diag(s)−ssᵀ → 0` and gradients vanish.
Dividing by `√d_k` normalizes the *standard deviation* so logit scale is `d_k`-independent (dividing
by `d_k` would over-shrink to near-uniform). Causal masking adds `−∞` to future positions **before**
softmax so they contribute 0 to the denominator and rows stay normalized; it also lets you train all
positions in parallel in one pass. Applied *after* softmax, the future tokens' mass stays in the
denominator so rows sum to <1 — you'd have to re-normalize, which is just masking-before done
expensively.

## Reusable Artifact

- Path: `code/positional_encoding.py` (sinusoidal + RoPE, with self-check harness)
- Reuse: `cos, sin = rope_cos_sin(T, D); q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)`
  applied to `(B, H, T, D)` Q/K before SDPA. Feeds directly into Day 2's transformer block.
