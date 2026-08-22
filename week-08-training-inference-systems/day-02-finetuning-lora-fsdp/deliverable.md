# Deliverable - Week 08 Day 02: Fine-Tuning Systems

## What I Built / Derived

Built `code/finetuning_memory.py`, a component-wise estimator for:

- LoRA parameter counts over named, repeated linear targets;
- full-FT, LoRA, and idealized QLoRA persistent model state;
- ideal per-rank ZeRO-0/1/2/3 state, including an explicit frozen-base policy;
- a simplified gradient-checkpointing curve over every integer interval.

The staged harness contains 31 checks. It pins the scoped 7B examples, uneven
shard rounding, validation contracts, and the full discrete checkpoint optimum.

## Core Takeaway

Fine-tuning memory methods are not interchangeable. LoRA reduces trainable
state; QLoRA also compresses the frozen base; ZeRO/FSDP shards state that would
otherwise be replicated; and gradient checkpointing reduces saved activation
liveness. Before choosing one, identify the binding bucket. Every estimate must
separate persistent from peak memory because activations, transient parameter
materialization, communication buffers, kernel workspaces, and runtime
headroom can still determine whether the job fits.

## Key Equations

```text
LoRA update:
W' = W0 + (alpha/r)BA
A in R^(r x d_in), B in R^(d_out x r), rank(BA) <= r

LoRA parameters per targeted matrix:
A = r(d_in + d_out)

classic trainable state:
2-byte weight + 2-byte gradient + 4-byte master + 8-byte Adam moments
= 16 bytes/trainable parameter

full FT:  16P
LoRA:      2P + 16A
QLoRA:   0.5P + 16A     (idealized 4-bit frozen-base lower bound)

ZeRO-0: 16P
ZeRO-1: P(4 + 12/N)
ZeRO-2: P(2 + 14/N)
ZeRO-3: 16P/N           (persistent full-FT state, not peak memory)

checkpoint chain model:
slots(k) = ceil(L/k) + k
continuous optimum near k = sqrt(L)
```

## Implementation Notes

- Rank-16 LoRA on `q_proj` and `v_proj` in 32 hidden-size-4096 layers has
  **8,388,608 adapter parameters**. Its classic trainable state is exactly
  **0.125 GiB**.
- Scoped 7B persistent totals are **104.308 GiB full FT**, **13.164 GiB
  LoRA**, and **3.385 GiB idealized QLoRA**.
- At world size eight, full-FT ZeRO-0/1/2/3 persistent state is
  **104.308 / 35.856 / 24.447 / 13.039 GiB per rank**.
- ZeRO stage semantics and frozen-base placement are separate inputs. Sharding
  only the LoRA trainable state leaves the 14 GB frozen base replicated.
- Four-bit storage uses ceiling division on total bits so odd parameter counts
  cannot produce fractional bytes.
- For `L=32`, the discrete checkpoint model has a flat optimum:
  **12 slots at every `k` from 4 through 8**. Returning only
  `round(sqrt(L))` loses that information.

## Failure Modes / Debugging Notes

- A trainable-fraction estimate that omits repeated target count can be wrong by
  the number of layers; the first LoRA implementation returned 262,144 instead
  of 8,388,608.
- Correct arithmetic can still violate the API contract. The first method
  estimator passed all numerical checks but accepted full FT with adapters and
  LoRA with zero adapters.
- Stage-specific branching left ZeRO-0 fields unbound. Assigning each component
  once from a `partition(value, enabled)` policy made stage 0 correct by
  construction and exposed the exact sharding rule.
- QLoRA does not make activations 4-bit. A long-context OOM can persist after a
  large frozen-weight saving.
- `16P/N` omits FSDP all-gather materialization, possible prefetch, communication
  buffers, activations, temporaries, and runtime headroom.
- Both LoRA factors must not start at zero. Random `A`, zero `B` makes `B` move
  first; zero `A`, random `B` is also algebraically viable and makes `A` move
  first. Only the both-zero initialization stalls both gradients.

## Interview Answer

Prompt:

> How do you choose between LoRA, QLoRA, FSDP/ZeRO, and gradient checkpointing?

Answer:

First decompose peak memory into persistent model state, saved activations,
temporary and communication buffers, transient parameter materialization, and
runtime headroom. Use LoRA when trainable gradients and optimizer state bind;
use QLoRA when the replicated frozen base also binds; use ZeRO/FSDP when model
state must be distributed while preserving full fine-tuning flexibility; and
use checkpointing when saved activations bind. These techniques compose because
they affect different buckets. I would quantify the selected bucket under named
dtype, target-rank, world-size, sequence, microbatch, and kernel assumptions,
then state the price: low-rank update constraints, quantization/dequantization,
communication/materialization, or recomputation.

## Reusable Artifact

- Path: `week-08-training-inference-systems/code/finetuning_memory.py`
- Checks: four staged sections, **31 passed**
- How to reuse: create named `LinearTarget` families, calculate the base and
  adapter state, apply an explicit sharding policy, and add activation/runtime
  estimates separately rather than hiding them in one total.
