# Week 02 Day 04 — KV Cache, Batching & Inference Throughput

## Objective

Make autoregressive generation efficient and reason about its cost. Implement a **KV cache** (turn
\(O(T^2)\) per-token recompute into \(O(T)\)), derive the **KV-cache memory formula**, and understand
the prefill-vs-decode split, continuous batching, and the memory-bandwidth bound that governs
LLM inference. Directly attacks your top gap: ML-systems / memory-cost reasoning.

## Prerequisites

- Day 3 `generate()` loop and Day 2 `TinyGPT`.
- Week 1 memory accounting (params + optimizer states; bf16 = 16 B/param for training).

## Required Reading

- A concise KV-cache + PagedAttention (vLLM) explainer — the memory + batching section only.
- Optional: a note on arithmetic intensity / roofline (memory-bound vs compute-bound).

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Days 1-3)

1. Top-k vs top-p; which is adaptive.
2. Causal mask placement and the causal-property test.
3. Temperature limits and what it rescales.
4. Pre-LN gradient-highway argument.

### 0:10-0:35 - Concept Explanation (First Principles)

Naive generation recomputes attention over the whole prefix every step → \(O(T^2 d)\) total. But K
and V for past tokens **never change** (causal), so cache them: each new token computes its Q, attends
to cached K/V, and appends its own K/V. Two phases: **prefill** (process the whole prompt in one
parallel pass, compute-bound, fills the cache) and **decode** (one token at a time, **memory-bandwidth
bound** — you stream all weights + KV per token to produce one token, so throughput is limited by HBM
bandwidth, not FLOPs). This is why **batching** is the key inference lever: batching amortizes the
weight-streaming cost across many sequences. **Continuous batching** (vLLM) swaps finished sequences
out and new ones in each step instead of waiting for the whole batch; **PagedAttention** stores the KV
cache in non-contiguous "pages" to kill fragmentation. **GQA/MQA** shrink the KV cache by sharing
K/V heads across query heads.

### 0:35-1:00 - Derivation / Paper Deconstruction

Derive the **KV-cache size**:
\[
\text{bytes} = 2 \times L \times n_{kv} \times d_{head} \times T_{seq} \times B \times \text{dtype\_bytes}
\]
(2 = K and V; \(L\) layers; \(n_{kv}\) KV heads). Plug a 7B-class example (e.g. \(L=32, n_{kv}=32,
d_{head}=128, T=4096, B=1\), fp16) → GB, and show it *grows linearly with context and batch* — often
dominating memory at long context. Contrast MHA vs GQA (\(n_{kv}\ll n_q\)) savings. Then the roofline
sketch: decode arithmetic intensity is ~1 op/byte → memory-bound → batching raises intensity.

### 1:00-1:45 - Coding / Experiment Task

`code/kv_cache.py` + self-check/pytest:

- Add optional `past_kv` to attention/block/`TinyGPT.forward` so a single-token step reuses the cache.
- `generate_cached(model, prompt, max_new_tokens, ...)` — prefill then incremental decode.
- A `kv_cache_bytes(cfg)` calculator function.

Self-check assertions: **cached generation produces token-identical output to the uncached Day-3
`generate()`** (the correctness anchor); cache length grows by 1 per step; a timing check shows cached
is faster for longer sequences; `kv_cache_bytes` matches a hand-computed example. Optional: measure
tokens/sec vs batch size and plot.

### 1:45-1:55 - Interview Drill

"Estimate the KV-cache memory for a 7B model at 8k context, batch 16." and "Why is decode
memory-bandwidth bound, and what does that imply for how you'd raise throughput?"

### 1:55-2:00 - Deliverable & Next Session

`deliverable.md`: KV-cache implementation + the memory formula + a tokens/sec-vs-batch note. Next:
the Week-2 debug lab — diagnose bad generations, NaNs, shape and masking bugs (capstone of the week).

## Checkpoint Questions

1. Why can K/V be cached but not Q? What exactly is stored?
2. Write the KV-cache memory formula and identify what makes it explode.
3. Prefill vs decode: which is compute-bound, which is memory-bound, and why?
4. Why does batching help inference throughput so much during decode?
5. How do GQA/MQA and PagedAttention each reduce KV-cache pressure?

## Frontier-Lab Interview Angle

Inference-systems and memory math are a *direct* frontier-lab MLE differentiator and your flagged
gap. "Estimate the memory / throughput" is a very common systems question; being fluent in the
prefill/decode + KV-cache + batching story is high signal. Ties to Week 8 (training/inference systems).
