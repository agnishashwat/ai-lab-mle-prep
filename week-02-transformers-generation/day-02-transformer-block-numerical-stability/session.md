# Week 02 Day 02 — Transformer Block & Numerical Stability

## Objective

Assemble the full decoder block (residual streams, LayerNorm, MLP, attention) and understand the
numerical-stability decisions that make deep transformers trainable: **pre-LN vs post-LN**, residual
scaling, LayerNorm vs RMSNorm, GELU/SwiGLU, and where fp16/bf16 bite. Produce a working
`TransformerBlock` and a tiny decoder-only `TinyGPT` that trains on the Week-1 harness.

## Prerequisites

- Day 1 attention primitives (`scaled_dot_product_attention`, `multi_head_attention`).
- Week 1: `train.py` loop + `eval_ablation.py` (we reuse them to train the block), mixed-precision
  failure modes (fp16 underflow → GradScaler; bf16 range).

## Required Reading

- "Attention Is All You Need" §3.1 + §5.4 (block structure, residual dropout) — or the GPT-2
  architecture note (pre-LN). Keep to one.
- Optional: the RMSNorm paper abstract + eq (1-2) only.

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 1)

1. Why \(1/\sqrt{d_k}\); state the variance argument in one sentence.
2. Additive `-inf` causal mask — why before softmax, not after?
3. What does multi-head buy over single-head of equal total width?
4. Softmax Jacobian and the saturated-attention gradient implication.

### 0:10-0:35 - Concept Explanation (First Principles)

The **residual stream** as the backbone: each sublayer *reads* from and *writes to* a shared
additive stream, `x = x + sublayer(norm(x))`. Two sublayers per block: causal MHA, then a
position-wise MLP (`d → 4d → d` with GELU/SwiGLU). **Pre-LN** (`x + f(LN(x))`) vs **post-LN**
(`LN(x + f(x))`): pre-LN keeps a clean identity path so gradients flow to the input undistorted →
trains deep nets without warmup-sensitivity; post-LN (original paper) has better final quality but
needs careful warmup/init or it diverges. **LayerNorm vs RMSNorm**: RMSNorm drops the mean-centering
and bias (just scales by RMS) — cheaper, ~same quality, now standard (LLaMA). Weight init:
scale output projections by \(1/\sqrt{2L}\) so residual variance doesn't blow up with depth.

### 0:35-1:00 - Derivation / Paper Deconstruction

Derive: (a) why the residual identity path makes \(\partial \text{loss}/\partial x_\ell\) contain a
`+1` term (gradient highway → no vanishing with depth); (b) LayerNorm's Jacobian and why it removes
the "scale" degree of freedom that would otherwise let activations drift; (c) show pre-LN bounds the
input to each sublayer regardless of depth, whereas post-LN's normalized *output* lets the residual
grow. Note where bf16 matters: LN statistics and softmax should accumulate in fp32.

### 1:00-1:45 - Coding / Experiment Task

Extend `code/` with `transformer_block.py` + self-check/pytest:

- `RMSNorm` (or LayerNorm) module.
- `MLP` (`d → 4d → d`, GELU).
- `TransformerBlock` — pre-LN: `x = x + attn(norm1(x), causal); x = x + mlp(norm2(x))`.
- `TinyGPT` — token embed + positional embed + N blocks + final norm + tied/untied LM head → logits `(B,T,V)`.

Then **train `TinyGPT` on the Week-1 successor task** via `train.py`'s loop (or a thin adapter) and
confirm CE drops well below `ln V`. Self-check assertions: shapes `(B,T,V)`; causal property (logits
at `t` unchanged when you edit tokens `>t` — the marquee test); pre-LN block preserves stream shape;
overfit-one-batch collapses to ~0 (reuse Day-5 sanity test on the real model).

### 1:45-1:55 - Interview Drill

"Pre-LN vs post-LN — which and why?" and "Your 24-layer transformer's loss is NaN at step 0. Walk
the config-bug checklist" (init scale, LR/warmup, LN placement, fp16 vs bf16).

### 1:55-2:00 - Deliverable & Next Session

`deliverable.md`: the block diagram + pre/post-LN gradient argument + a trained-TinyGPT loss curve.
Next: decoding strategies (temperature, top-k/p, beam) on top of this trained model.

## Checkpoint Questions

1. Write the pre-LN block equations and explain the gradient-highway `+1`.
2. Pre-LN vs post-LN: gradient flow, warmup sensitivity, final-quality trade-off.
3. RMSNorm vs LayerNorm — what's dropped and why is it fine?
4. Why scale residual/output projections by \(1/\sqrt{2L}\)?
5. Which ops must stay fp32 under bf16 autocast, and why?

## Frontier-Lab Interview Angle

"Build a transformer block" and "debug a diverging deep transformer" are staple exercises. The
pre/post-LN + init-scaling story signals you've actually trained deep models, not just read about
them. The NaN-at-step-0 vs NaN-after-stable-steps distinction (Week-1 habit) shines here.
