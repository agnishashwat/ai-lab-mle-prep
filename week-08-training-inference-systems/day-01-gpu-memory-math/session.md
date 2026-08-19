# Week 08 Day 01 - GPU Memory Math: count peak live bytes

## Objective

Estimate whether a training or inference workload fits before launching it. Separate
parameters, gradients, optimizer states, saved activations, temporary buffers, KV cache, and
runtime overhead instead of hiding them inside one rule of thumb.

The thesis:

> A model fits only when its peak live bytes fit. Parameter bytes are one term, not the answer.

## Prerequisites

- Week 02 Day 04: KV-cache layout and prefill versus decode.
- `shared/numbers-card.md`, Week 1-2: 16 bytes/parameter and the 512 KiB/token result.
- Tensor shape accounting: number of elements times bytes per element.

## Required Reading

- Hugging Face, "Model training anatomy", memory-anatomy section.
- Optional: FlashAttention paper, IO-awareness motivation and memory-complexity discussion.

## Memory model

Use binary units for device capacity and report assumptions:

```text
1 KiB = 1024 bytes
1 MiB = 1024 KiB
1 GiB = 1024 MiB

training peak ~= model states + saved activations + temporary buffers
                 + communication buffers + runtime/allocator headroom

inference peak ~= weights + KV cache + temporary activations/workspaces
                  + runtime/allocator headroom
```

These are accounting identities, not universal constants. Peak liveness depends on the
implementation, kernel choice, parallelism strategy, and tensor lifetime.

## 2-Hour Plan

### 0:00-0:10 - Recall quiz

1. A 7B BF16 model, weights only: `7e9 * 2 = 14 GB = 13.0 GiB`.
2. Classic mixed-precision Adam model states:

   ```text
   BF16 weights          2 bytes/parameter
   BF16 gradients        2 bytes/parameter
   FP32 master weights   4 bytes/parameter
   FP32 Adam m and v     8 bytes/parameter
   total                16 bytes/parameter
   ```

   For 7B: `112 GB = 104.3 GiB`, before activations.

3. KV cache:

   ```text
   bytes = 2 * L * n_kv * d_head * B * T * element_bytes
           ^
           K and V
   ```

   For `L=32, n_kv=32, d_head=128, B=1, T=4096, BF16`: exactly `2 GiB`.

### 0:10-0:35 - First-principles accounting

#### Model states do not scale with batch or sequence

The 16-byte rule is an assumption bundle. Some BF16 optimizers omit the FP32 master copy;
8-bit optimizers change moment storage; LoRA changes which parameters need gradients and
optimizer states. Always list components before giving the total.

#### Activations are shapes plus lifetimes

One dense tensor costs:

```text
bytes = product(shape) * element_bytes
```

At `B=8, T=4096, H=4096`, one BF16 `[B,T,H]` tensor is `256 MiB`. Saving one
such tensor at each of 32 layers is already `8 GiB`; real backward passes may save several
tensors per layer.

A naively materialized attention matrix has shape `[B,n_heads,T,T]`. Doubling `T` therefore:

- doubles residual-stream-style `[B,T,H]` storage;
- quadruples materialized attention-score storage;
- doubles inference KV-cache storage.

FlashAttention avoids storing the full `T*T` matrix. It reduces peak memory but does not turn
the arithmetic complexity of exact attention into linear time.

#### Training K/V is not a serving KV cache

Teacher-forced training processes the sequence together. Its K and V tensors are saved
activations needed by backward, not a persistent incremental-decode cache. Do not add both an
activation estimate and a serving KV-cache estimate to the same training budget.

### 0:35-1:00 - Capacity derivations

#### Inference token budget

First calculate the architecture's per-token cache:

```text
kv_bytes_per_token = 2 * L * n_kv * d_head * element_bytes
```

Then:

```text
usable_for_kv = device_capacity - weights - reserve
token_slots   = floor(usable_for_kv / kv_bytes_per_token)
```

For a 24 GiB device, 13 GiB of weights, a 2 GiB reserve, and 512 KiB/token:

```text
usable_for_kv = 9 GiB
token_slots   = 9 * 1024 MiB / 0.5 MiB = 18,432
```

At equal lengths, batch 4 permits `18,432 / 4 = 4,608` tokens per sequence. In a paged
runtime the useful quantity is aggregate live tokens, `sum_i T_i`, rather than
`batch_size * max(T_i)`.

#### Global batch versus peak activation memory

```text
global_batch = data_parallel_workers * microbatch_per_worker * accumulation_steps
```

For global batch 128, 8 workers, and microbatch 2, accumulation is 8. Reducing microbatch to
1 roughly halves batch-linear activation memory and doubles accumulation to 16; model-state
memory does not change.

Gradient checkpointing similarly changes saved activation liveness, not model-state bytes. It
trades memory for recomputation.

### 1:00-1:15 - Interview drill

Answer cold, with four slots: conclusion, arithmetic, missing buckets, scope/caveat.

1. "A 7B BF16 model is only 14 GB, so it fits comfortably on a 16 GB GPU. Correct?"
2. "Can a 7B model be fully fine-tuned with mixed-precision Adam on one 80 GiB GPU?"
3. "Why can two implementations of the same model and batch have different peak memory?"

Write the final answer into `interview-bank/systems-qa.md`. Do not claim that KV cache is a
separate full-sequence training allocation.

### 1:15-1:55 - Coding task

Implement the four TODOs in `code/gpu_memory.py`:

1. named model-state accounting;
2. dense tensor storage;
3. MHA/GQA/MQA KV-cache storage;
4. aggregate KV token-slot capacity.

Run:

```bash
python3 -m pytest week-08-training-inference-systems/code/test_gpu_memory.py -q
```

The checks pin the worked examples and two important scaling laws. After they pass, add one
real model scenario with its exact architecture and state every precision assumption.

### 1:55-2:00 - Deliverable and next session

Fill `deliverable.md` with the calculator, the 7B worked examples, and one sentence explaining
why an activation estimate must name the implementation. Add only stable measured values to
`shared/numbers-card.md`.

## Checkpoint questions

1. Why is `14 GB` not `14 GiB`, and when does that distinction matter?
2. Which model-state components disappear for inference?
3. Why does GQA reduce KV memory without changing the query-head count?
4. Which terms scale linearly and quadratically with sequence length?
5. What does gradient accumulation reduce, and what does it leave unchanged?

## Frontier-lab interview angle

Do not answer a fit question with one memorized bytes-per-parameter number. State precision,
optimizer, trainable fraction, microbatch, sequence length, attention implementation, KV-head
count, parallelism/sharding, and safety reserve. Then identify the likely bottleneck and the
first lever you would change.

## Session result - complete

- The learner derived the 7B weight, mixed-precision Adam, KV-cache, context-scaling,
  capacity, and gradient-accumulation examples.
- A unit slip (`512 KiB` read as `0.5 GiB`) was repaired to `0.5 MiB`, producing
  18,432 aggregate token slots and 4,608 tokens per sequence at batch 4.
- The learner implemented all four calculator TODOs. The model-state path initially omitted
  the parameter-count multiplier; the KV path initially added parameter count as an extra
  dimension. Both were corrected by checking units and tensor shape.
- `code/test_gpu_memory.py`: **9 passed**.
- All three interview drills were completed and recorded in `interview-bank/systems-qa.md`.
