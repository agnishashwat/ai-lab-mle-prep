# Week 02 Summary - Transformer Internals & Efficient Generation

## The 5-Minute Version

A decoder-only transformer is a **residual stream** `x = x + f(Norm(x))` alternating two sublayers:
**causal multi-head attention** (the only cross-position mixing — soft differentiable retrieval
`softmax(QKᵀ/√d_k + M)V`) and a **position-wise MLP** (`d→4d→d`, per-token compute). Attention is
permutation-equivariant, so we inject order with **positional encodings** (sinusoidal =
parameter-free/any-length; learned = expressive but hard-capped; RoPE = relative via rotations,
extrapolates best). Stability comes from **pre-LN** (residual stays an identity → gradient-highway
`+1` that survives deep stacks), **RMSNorm**, fp32 reductions, and the `1/√(2L)` output-proj rescale.
We train with **teacher forcing** (one parallel pass, causal mask makes it valid) and generate
**auto-regressively**. Generation quality is a **decoding** choice: search (greedy/beam) maximizes
likelihood → great for low-entropy tasks (MT/code) but degenerates (repetition attractor) on open
text, where **temperature + nucleus** sampling matches human-text entropy. Generation *speed* comes
from the **KV cache**: past K/V never change under causality, so cache them → decode drops O(T²)→O(T),
total O(N³)→O(N²). At scale the cache dominates memory (`2·L·n_kv·d_head·T·B·bytes`; 7B fp16 = 512
KiB/tok → 64 GiB @ 8k×16, dwarfing the 14 GB weights), and **decode is memory-bandwidth-bound**
(≈1 op/byte) so batching (amortize weight streaming), GQA/MQA, quantization, and speculative decoding
are the throughput levers. The week closed with a **debug lab**: a four-contract map + invariant
regression tests that catch the whole bug taxonomy.

## Key Concepts (crisp)

- **Residual stream + two sublayers:** attention = cross-position mixing; MLP = per-position.
- **`1/√d_k` scaling:** keeps logit STD `d_k`-independent; dividing by `d_k` over-shrinks → uniform.
- **Softmax saturation = vanishing gradient:** big unscaled logits → near one-hot → Jacobian `diag(s)−ssᵀ → 0`.
- **Pre-LN gradient highway** vs post-LN compounding `J_LN` product (warmup-sensitive).
- **RoPE relative-position:** `<rope(q,m), rope(k,n)>` depends only on `m−n`; best extrapolation.
- **Decoding axis:** search maximizes likelihood (bland/repetitive on open text); sampling matches entropy.
- **Top-p (nucleus) is adaptive** (smallest set with cumprob ≥ p); top-k is a fixed budget that fails both ways.
- **KV cache:** cache K/V (reused every step), never Q (used once); prefill = compute-bound, decode = BW-bound.
- **Four debugging contracts:** causal / shape / numeric / train-inference — every bug is one violation.

## Key Equations

- Attention: `softmax(QKᵀ/√d_k + M) V`; `Var(q·k) = d_k σ⁴`.
- Softmax Jacobian: `diag(s) − ssᵀ`; = 0 at a one-hot (saturation).
- LN Jacobian: `(1/σ)(I − (1/d)11ᵀ − (1/d)x̂x̂ᵀ)` (projects out mean + radial dirs).
- Temperature: `softmax(z/T)`; entropy monotone `0 → ln V`.
- Beam length-norm: `(1/Lᵅ) Σ log p_t`, `α ≈ 0.6–0.7`.
- KV-cache bytes: `2 · L · n_kv · d_head · T · B · dtype_bytes` (512 KiB/tok for 7B fp16 MHA).

## What I Implemented

- D1 `positional_encoding.py`: sinusoidal + RoPE (relative-offset invariance proven to 1.4e-6).
- D2 `transformer_block.py`: RMSNorm, MLP, pre-LN block, TinyGPT with `1/√(2L)` rescale (causal test bit-exact).
- D3 `decoding.py`: temperature, top-k, top-p (scatter-back), sample, generate, beam (length-normalized).
- D4 `kv_cache.py`: cache-aware attention (offset causal mask), prefill+decode `generate_cached`
  (token-identical to naive, 4.15x decode speedup), `kv_cache_bytes`.
- D5 `debug_lab.py`: six invariant regression tests + planted-bug teeth check; `work.ipynb` demo.

## Failure Modes I Now Understand

- Fully-masked softmax row → 0/0 → NaN (prevent: every row keeps ≥1 key).
- fp16 overflow → Inf → NaN (bf16 / fp32 softmax); LN ε=0 division.
- CE K-dim shape trap: silent when `T==V` — always `reshape(-1, V)`.
- Silent-arg bug when two config dims are equal (`n_heads==mult`, `T==V`) — vary a value to expose.
- KV-cache: `q@k` not `q@k_all` (silent prefill, crashes decode); `arange(S)` not `pos_offset+arange(S)`
  (repeats the prefix); uncropped context overruns the learned-pos table.
- **Test-design failure:** an invariant test is only as sensitive as the input makes it (the
  successor task is blind to attention bugs → use a random model for the cache-equality test).
- Neural text degeneration (repetition attractor) — expected behavior, not a bug; fix = temp+nucleus.

## Open Questions / To Revisit

- Numerics checklist cold (attention-NaN fastest-first) — still the soft spot.
- Arithmetic intensity / roofline phrasing; BW-tied levers (quant, speculative decode) as *mechanisms*.
- RoPE base-rescaling (NTK/YaRN) for context extension — conceptual only so far.
- GQA/MQA + PagedAttention internals (capacity vs bandwidth) beyond the memory formula.
