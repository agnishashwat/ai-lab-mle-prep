# Deliverable - Week 02 Day 02: Transformer Block & Numerical Stability

## What I Built / Derived

`code/transformer_block.py` (+ `test_transformer_block.py`, 6/6 pytest, 7/7 self-check) — the full
decoder-only stack on a residual stream, built on the Day-1 attention core:

- **`RMSNorm`** — normalize by RMS over the feature dim (fp32 reduction), learnable gain, no
  mean-centering / no bias.
- **`MLP`** — position-wise `d → 4d → d` with GELU (the "think locally" sublayer).
- **`TransformerBlock`** — **pre-LN**: `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`.
- **`TinyGPT`** — token embed + learned positional embed + N blocks + final RMSNorm + LM head →
  logits `(B,T,V)`; with the GPT-2 `1/sqrt(2L)` output-projection rescale wired into `__init__`.

Marquee test passes: editing tokens after position `t` leaves logits at `≤ t` **bit-identical**
(`max|Δ|_past = 0.00e+00`) while future logits change — end-to-end causality through the stack.
Overfit-one-batch drives CE `→ 0.0010` (`<< ln V = 2.773`), confirming data→loss→optimizer wiring.

## Core Takeaway

A decoder-only transformer is a **residual stream** with sublayers that *read → compute → add back*.
Attention is the only cross-position mixing; the MLP is per-position. The design that makes deep
stacks trainable is **pre-LN**: putting the norm *inside* each sublayer leaves the residual path a
clean identity, so the backward pass carries a standalone `+1` (a gradient highway) that no product
of small Jacobians can kill. Post-LN puts an LN Jacobian *on* the highway at every layer → those
`2L` factors compound → warmup-sensitive divergence. That is why everything modern (GPT-2, LLaMA)
is pre-LN.

## Key Equations

- Pre-LN block: `x = x + Attn(Norm(x))`, then `x = x + MLP(Norm(x))`.
- Gradient highway: `∂L/∂x_ℓ = ∂L/∂x_L · Π_{k≥ℓ} (I + ∂f_k/∂x_k)` → expands to `I + Σ ∂f_k + …`;
  the standalone `I` survives even if every `∂f_k ≈ 0`.
- RMSNorm: `y = γ ⊙ x / sqrt(mean(x²) + ε)` (no mean subtraction, no bias).
- LayerNorm Jacobian: `∂x̂/∂x = (1/σ)(I − (1/d)11ᵀ − (1/d)x̂x̂ᵀ)` — projects out the mean
  direction and the radial (scale) direction ⇒ LN removes the scale degree of freedom.
- Depth init: scale each sublayer's output projection (`Wo`, `fc2`) by `1/sqrt(2L)` so residual
  variance stays `O(1)` across `L` layers.

## Implementation Notes

- **RMSNorm reduces in fp32** (`dtype=float32` on the mean-square): a normalization is
  "reduce-then-divide", so the statistic must be precise. Contrast a matmul, which accumulates in
  fp32 internally and rounds once — safe in bf16.
- **Silent arg bug caught:** `TransformerBlock(d, n_heads, mult)` — passing `n_heads` for `mult`
  passed only because `n_heads == mult == 4`. Same class as the Week-1 `T==V` cross-entropy bug: a
  wrong argument hiding behind two equal numbers. Fixed to `cfg.mult`.
- Named submodules (`fc1`/`fc2`, `Wo`) over `nn.Sequential` — needed for the `1/sqrt(2L)` rescale
  and any surgical weight access (LoRA, tying, freezing).
- `torch.arange(T, device=idx.device)` for the positional index (GPU-safety; harmless on CPU).

## Failure Modes / Debugging Notes

- **Which ops stay fp32 under bf16 autocast:** LayerNorm/RMSNorm statistics, softmax, and the final
  logits→CE (logsumexp). Rule: matmul = accumulate-then-round-once (safe); normalization =
  round-then-divide (unsafe).
- **Post-LN divergence** = the compounding LN-Jacobian product on the residual path; the fix people
  reach for is warmup + careful init, or just switch to pre-LN.
- **Residual variance blow-up with depth** if output projections aren't rescaled → `1/sqrt(2L)`.

## Interview Answer

Prompt:

> "You train a 24-layer transformer and the loss is NaN at step 0. Walk me through debugging."

Answer:

NaN *at step 0* means the first forward/backward already produced non-finite values — no weight has
updated yet, so it's a **config bug, not training dynamics**; that immediately rules out the
learning rate (which only bites after `optimizer.step()`, i.e. step 1+). I'd bisect
forward → loss → backward. Highest-probability suspects, in order: (1) the **loss/mask** — a fully
masked attention row gives softmax over all `−∞` → 0/0 → NaN; check logits/loss finiteness on batch
0. (2) **fp16 overflow** — a deep residual stream exceeds fp16's 65504 → Inf → NaN; switch to bf16
(fp32 range) or fp16 + GradScaler (note bf16 fixes overflow, not underflow-to-zero, which is a
different, non-NaN failure). (3) **init/normalization** — unscaled output projections make residual
variance grow ~`2L`, or `eps=0` in LN → divide-by-tiny; apply the `1/sqrt(2L)` rescale and nonzero
`eps`. (4) **bad data/labels** — non-finite inputs or a target id ≥ vocab. The senior tell is the
temporal contrast: step-0 NaN ⇒ forward/loss config; step-500-then-NaN ⇒ "something grew," so I'd
flip to grad-norm inspection and `torch.autograd.set_detect_anomaly`.

## Reusable Artifact

- Path: `code/transformer_block.py`, `code/test_transformer_block.py`
- How to reuse: `TinyGPT(GPTConfig(...))` is a real decoder-only LM (forward `(B,T)→(B,T,V)`),
  drop-in for the Week-1 loop. The causal marquee test and overfit-one-batch adapter are reusable
  sanity checks for any new attention/model change. Day 3 (decoding strategies) generates from this
  trained model.
