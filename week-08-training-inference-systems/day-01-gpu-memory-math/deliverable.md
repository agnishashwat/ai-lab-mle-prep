# Deliverable - Week 08 Day 01: GPU Memory Math

## What I Built / Derived

Built `code/gpu_memory.py`, a first-principles calculator for:

- named model-state components;
- arbitrary dense tensor storage;
- MHA/GQA/MQA KV-cache storage;
- aggregate KV token capacity after fixed allocations;
- transparent saved-hidden-state and naive-attention activation scenarios.

The accompanying test suite pins the 7B worked examples and scaling laws.

## Core Takeaway

A model fits only if its **peak live bytes** fit. A 7B BF16 model has 14 GB
(about 13.0 GiB) of weights, but classic mixed-precision Adam requires 16
bytes/parameter, or 112 GB (104.3 GiB), before activations. Inference must add
the persistent KV cache and runtime headroom; training must add saved
activations, temporary buffers, communication buffers, and runtime headroom.
Activation memory is an implementation-and-liveness property, not a constant
of the architecture.

## Key Equations

```text
model states = P * (weight B + gradient B + master-weight B
                    + optimizer-state count * optimizer-state B)

dense tensor bytes = product(shape) * element bytes * copies

KV bytes = 2 * layers * KV heads * head dimension
             * batch * sequence length * element bytes

KV token slots = floor((device capacity - weights - reserve)
                       / KV bytes per token)

global batch = data-parallel workers * microbatch * accumulation steps
```

## Implementation Notes

- `head_dim = hidden_size / query_heads`, while cache storage uses
  `kv_heads`; this is what models GQA and MQA correctly.
- For the 32-layer, 32-KV-head, head-dimension-128 example, BF16 KV storage is
  exactly 512 KiB/token: 2 GiB at batch 1 and 4,096 tokens, or 64 GiB at batch
  16 and 8,192 tokens.
- A simplified 24 GiB budget with 13 GiB weights and a 2 GiB reserve leaves
  18,432 aggregate token slots, or 4,608 tokens per sequence at batch 4.
- At `B=8, T=4096, H=4096`, one saved BF16 `[B,T,H]` tensor per layer is
  8 GiB across 32 layers. Fully materialized attention matrices would be
  256 GiB; doubling context quadruples that term.

## Failure Modes / Debugging Notes

- Mixing decimal GB with binary GiB can erase an already-small safety margin.
- Parameter count is not a KV-cache dimension; adding it makes the estimate
  billions of times too large.
- Using query heads instead of KV heads overestimates GQA/MQA caches.
- A serving KV cache should not be added separately to a training activation
  estimate; training K/V tensors are activations.
- FlashAttention, checkpointing, fusion, precision, padding, and tensor
  lifetimes all change peak activation memory.

## Interview Answer

Prompt:

> A 7B BF16 model is only 14 GB, so it should fit comfortably on a 16 GiB GPU
> for inference. Is that correct?

Answer:

Not comfortably at batch 1 and 4k context under these assumptions. The weights
occupy about 13.0 GiB, and the BF16 MHA KV cache occupies another 2 GiB, leaving
only about 1 GiB for runtime state, kernel workspaces, temporary activations,
and allocator fragmentation. Shorter context, fewer KV heads, or quantized
weights can change the conclusion, so the answer must state its assumptions.

## Reusable Artifact

- Path: `code/gpu_memory.py`
- Checks: `code/test_gpu_memory.py` (9 passed)
- How to reuse: instantiate `TransformerSpec`, choose explicit precision and
  optimizer assumptions, then calculate fixed model state, variable KV or
  activation storage, and a safety reserve separately.
