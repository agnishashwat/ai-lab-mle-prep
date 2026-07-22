# Deliverable - Week 02 Day 04: KV Cache, Batching & Inference Throughput

## What I Built / Derived

`code/kv_cache.py` (+ `test_kv_cache.py`, 6/6 pytest, 13/13 self-check) — a KV-cache-accelerated
decode path that reuses the exact Day-2 `TinyGPT` weights, plus the inference memory calculator:

- **`sdpa_cached`** — cache-aware attention: compute `q,k,v` for the NEW tokens only, `cat` past
  `k,v` along the time dim, score `q @ k_all`, and apply an **offset causal mask**
  (`torch.tril(ones(S, P+S), diagonal=pos_offset)` → keep iff `key_j <= pos_offset + query_i`).
  One formula covers prefill (`P=0` → lower-triangular) and decode (`S=1` → keep all).
- **`block_forward_cached` / `gpt_forward_cached`** — thread a per-layer `(k,v)` cache through the
  pre-LN block and the model; positions are `pos_offset + arange(S)` (learned-pos-table correctness).
- **`generate_cached`** — **prefill** the whole prompt once (`pos_offset=0`), then **decode** one
  token per step feeding ONLY `next_tok` with `pos_offset = len-1`.
- **`kv_cache_bytes(cfg, seq_len, batch, dtype_bytes, n_kv_heads)`** — the memory formula.

## Core Takeaway

Past tokens' K/V never change under causal attention, so cache them: attention per step drops
`O(T²) → O(T)`, total generation `O(N³) → O(N²)`. Inference splits into **prefill**
(one parallel pass over the prompt, compute-bound) and **decode** (one token at a time,
**memory-bandwidth bound** — you stream all weights + KV per token to emit one token). The whole
serving stack (batching, continuous batching, PagedAttention, GQA/quantization) exists to manage
the KV-cache memory and the weight-streaming bandwidth that this split exposes.

## Key Results

- **Correctness anchor:** cached greedy is **token-identical** to the uncached Day-3 `generate()`;
  prefill logits match a full forward at **0.00e+00**; incremental decode logits match a full
  recompute at every step (`atol 1e-4`).
- **Cache grows by exactly 1 per decode step**; trained-successor cached greedy counts up `[6,7,8,9,10]`.
- **Speedup:** `4.15x` faster than uncached at `N=200` on a 6-layer/`d=256` model (uncached
  1111 ms → cached 268 ms).

## Key Equations

- **KV-cache size:** `bytes = 2 · L · n_kv · d_head · T · B · dtype_bytes`
  (2 = K and V; `n_kv` is what GQA/MQA shrink; `d_head = d/n_heads`).
- **7B example** (`L=32, n_kv=32, d_head=128, fp16`): **512 KiB / token** → `2 GiB` at `T=4096, B=1`;
  `64 GiB` at `T=8192, B=16` (dwarfs the `14 GB` of fp16 weights). GQA `n_kv=8` → `4×` smaller.
- **Arithmetic intensity** at decode ≈ `1 op/byte` (move ~all weights, do ~one token of math) →
  memory-bound; batching raises FLOPs at ~constant bytes → intensity ↑ `~B×` → climb toward compute-bound.

## Implementation Notes

- Score `q @ k_all` (ALL cached+new keys), not `q @ k` — the latter passes prefill silently (P=0)
  then crashes at the first decode step (`S` vs `P+S` mismatch). (Live bug this session.)
- **Prefill/decode invariant:** the model sees each token exactly once — prefill feeds `(B, L)` at
  `pos_offset=0`; each decode step feeds `(B, 1)` at `pos_offset = len-1`. Feeding the whole running
  context while ALSO passing the past cache double-counts and blows up positions. (Live bug.)
- Learned absolute pos-table → decode positions MUST be `pos_offset + arange(S)`, not `arange(S)`.
- `cat` past+new K/V on the time axis (`dim=-2`/`2`), never the default `dim=0`.
- Build the causal mask on `x.device` (CPU-default mask errors on GPU — a silent-until-GPU bug).

## Failure Modes / Debugging Notes

- KV cache is **per-sequence and grows with context×batch**; at long context it dominates memory
  over the (fixed) weights → the real limit on how large a batch you can serve.
- Cropping the context for the learned pos-table is trivial in the uncached loop but non-trivial
  with a cache (you must evict oldest entries) — scoped out here via `assert L+max_new ≤ cfg.T`.

## Interview Answer

Prompt:

> "Estimate KV-cache memory for a 7B model at 8k context, batch 16 — and why is decode
> memory-bandwidth bound?"

Answer:

`bytes = 2 · L · n_kv · d_head · T · B · dtype_bytes = 2·32·32·128·8192·16·2 = 2^36 = 64 GiB`
(fp16, MHA) — which dwarfs the ~14 GB of fp16 weights, so KV cache is the binding constraint;
GQA `n_kv=8` cuts it 4× to 16 GB. Decode is memory-bandwidth bound because you stream all weights
(+ growing KV) from HBM per token but do only ~one token of arithmetic → arithmetic intensity ≈ 1
op/byte, far below the GPU's FLOP:bandwidth ratio, so the GPU idles on memory. The fix is to raise
intensity: batch to amortize weight-streaming across sequences, continuous-batch to keep the batch
full, quantize weights to move fewer bytes, and speculative-decode to emit multiple tokens per
weight sweep; GQA/MQA both shrink KV and free room for bigger batches.

## Reusable Artifact

- Path: `code/kv_cache.py`, `code/test_kv_cache.py`
- How to reuse: `generate_cached(model, prompt, N, greedy=...)` for fast decode on any `TinyGPT`;
  `kv_cache_bytes(cfg, seq_len, batch, dtype_bytes, n_kv_heads)` for capacity planning (MHA vs GQA).
