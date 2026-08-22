# Week 08 Day 02 - Fine-Tuning Systems: Which Bytes Move, Which Bytes Shard

## Objective

Given a model, adapter configuration, optimizer layout, world size, and activation policy,
estimate the persistent per-rank memory and identify the peak-memory terms the estimate omits.
Compare full fine-tuning, LoRA, QLoRA, ZeRO/FSDP, and gradient checkpointing without treating
them as interchangeable tricks.

The thesis:

> The techniques attack different memory buckets: LoRA reduces trainable state, QLoRA
> compresses frozen weights, ZeRO/FSDP shards replicated state, and checkpointing reduces saved
> activations. A fit claim must say which bucket remains and what is transiently materialized.

## Prerequisites

- Week 08 Day 01: peak live bytes and the classic mixed-precision Adam layout.
- Week 02: linear-layer shapes and named projections inside a transformer block.
- Week 04-05: a frozen reference can be the base model with the adapter disabled.

## Required Reading

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", Sections 3-4.
- Dettmers et al., "QLoRA", Sections 2-3.
- Rajbhandari et al., "ZeRO", the three partitioning stages.
- Optional: PyTorch FSDP documentation on `FULL_SHARD` and activation checkpointing.

## Memory Model

Keep four independent questions separate:

1. **What is trainable?** Full parameters or a low-rank adapter.
2. **How is the frozen base stored?** BF16/FP16 or quantized.
3. **What is replicated per rank?** Parameters, gradients, and optimizer state may be sharded.
4. **Which activations stay live?** Checkpointing changes saved tensors, not model state.

The formulas below use the Day 01 classic layout for each trainable parameter:

```text
BF16 weight + BF16 gradient + FP32 master weight + two FP32 Adam moments
= 2 + 2 + 4 + 8
= 16 bytes/trainable parameter
```

These are explicit assumptions, not library guarantees. Real jobs also need activations,
temporary dequantization/kernel buffers, communication buffers, allocator headroom, and the
largest parameter materialization performed by a sharded wrapper.

## 2-Hour Plan

### 0:00-0:10 - Cold Recall

Read `shared/numbers-card.md` out loud first. Then answer without notes.

1. For a 7B model under the Day 01 classic mixed-precision Adam assumptions, fill:

   ```text
   [bytes/parameter] -> [decimal GB] -> [binary GiB]
   components: [___] + [___] + [___] + [___]
   ```

2. For the Day 01 32-layer MHA model at BF16, fill:

   ```text
   [KiB per cached token] -> [GiB at B=1, T=4096]
   dimensions in the equation: [___]
   ```

3. A run has global batch 128, eight data-parallel workers, and microbatch two. Fill:

   ```text
   accumulation steps = [a number]
   after microbatch becomes one = [a number]
   reduced memory bucket = [one bucket]
   unchanged memory bucket = [one bucket]
   ```

### 0:10-0:35 - First-Principles Comparison

#### Full fine-tuning

For `P` base parameters, every parameter can move. Under the stated optimizer layout:

```text
persistent model state = 16P bytes
```

This provides the least constrained update but pays gradient and optimizer storage for the
whole model, plus distributed gradient communication.

#### LoRA: reduce the trainable state

For a frozen linear map:

```text
W₀ ∈ R^(d_out x d_in)
A  ∈ R^(r x d_in)
B  ∈ R^(d_out x r)

W' = W₀ + (α/r)BA
rank(BA) <= r
```

The adapter has:

```text
adapter parameters = r(d_in + d_out)
full matrix         = d_in*d_out
fraction for square d x d matrix = 2r/d
```

`B` is commonly initialized to zero so the initial adapter update is exactly zero. `A` is
initialized randomly, and `α/r` separates update scale from rank.

For a 32-layer, hidden-size-4096 model targeting only `q_proj` and `v_proj` at rank 16:

```text
adapter parameters
= 32 layers * 2 matrices/layer * 16 * (4096 + 4096)
= 8,388,608
```

That is about 0.12% of a 7B base. Under BF16 frozen weights and the 16-byte adapter-state
assumption:

```text
LoRA persistent state = 2P + 16A
```

For this scoped example it is about 13.16 GiB: 13.04 GiB of frozen base weights plus exactly
0.125 GiB of adapter state.

LoRA does **not** make the base disappear, quantize it, or eliminate the need to backpropagate
through the network to train earlier adapters. It can reduce some autograd state associated
with frozen weight gradients, but sequence-length-dependent activation memory remains.

#### QLoRA: compress the frozen base

QLoRA keeps the low-rank adapters trainable while storing the frozen base in 4-bit form and
dequantizing blocks into the compute dtype for matrix multiplication. NF4 is the data type;
double quantization compresses quantization constants; paged optimizers handle memory spikes.
These are three separate mechanisms.

An idealized 4-bit lower-bound estimate is:

```text
QLoRA persistent state ~= 0.5P + 16A bytes
```

For the same scoped 7B example this is about 3.38 GiB. Real storage is higher because scales,
quantization metadata, packing, temporary dequantization buffers, and runtime overhead still
exist. Activations and adapter computation are not 4-bit.

#### ZeRO and FSDP: shard replicated state

Let `N` be the data-parallel world size. With 2-byte weights, 2-byte gradients, 4-byte master
weights, and 8 bytes of Adam moments, the ideal persistent per-rank lower bounds are:

| Strategy | Replicated | Sharded | Per-rank bytes |
|---|---|---|---:|
| DDP / ZeRO-0 | everything | nothing | `16P` |
| ZeRO-1 | weights, gradients | master weights, moments | `P(4 + 12/N)` |
| ZeRO-2 | weights | gradients, master weights, moments | `P(2 + 14/N)` |
| ZeRO-3 / FSDP full shard | no full persistent state | weights, gradients, optimizer state | `16P/N` |

For 7B parameters and `N=8`, those ideal persistent values are approximately:

```text
DDP     104.31 GiB/rank
ZeRO-1   35.86 GiB/rank
ZeRO-2   24.45 GiB/rank
ZeRO-3   13.04 GiB/rank
```

The ZeRO-3/FSDP number is not the peak. A rank all-gathers the current wrapped unit for compute,
then re-shards it; prefetching may overlap more than one unit. Add the largest simultaneous
materialization, communication buffers, activations, temporaries, and headroom.

LoRA and sharding are orthogonal. Ordinary data-parallel LoRA replicates the frozen base and
only makes the adapter's trainable state small. Sharding the tiny adapter state buys little;
sharding the frozen base can still matter when the base itself does not fit.

#### Gradient checkpointing: reduce saved activations

For a chain of `L` layers checkpointed every `k` layers, a useful shape model is:

```text
peak activation slots ~= ceil(L/k) + k
```

The first term is saved segment boundaries; the second is the currently recomputed segment.
The minimum is near `k = sqrt(L)`, changing the asymptotic storage from `O(L)` to `O(sqrt(L))`
under this simplified chain model.

Checkpointing recomputes forward work during backward. It does not reduce parameter,
gradient, optimizer, or frozen-base storage. Stateful layers, dropout RNG handling, in-place
operations, and wrapper boundaries can make a naive checkpointing setup wrong or slow.

### 0:35-1:00 - Derivations and Decision Table

#### Derivation 1: the adapter fraction

For one square `d x d` matrix:

```text
LoRA/full = r(d+d)/d^2 = 2r/d
```

At `d=4096, r=16`, one targeted matrix has 131,072 adapter parameters instead of 16,777,216:
0.78125% as many. This ratio applies only to targeted matrices. The whole-model trainable
fraction also depends on how many projections are targeted and whether embeddings, norms, or
biases are trainable.

#### Derivation 2: sharding is a weighted sum, not "divide memory by N"

For ZeRO-2:

```text
per rank
= replicated weights + sharded gradients + sharded master/moments
= 2P + (2P + 4P + 8P)/N
= P(2 + 14/N)
```

Only ZeRO-3 shards the persistent parameter copy. Even then, peak memory does not divide
perfectly by `N` because compute needs transiently materialized parameters.

#### Derivation 3: why checkpoint spacing is near sqrt(L)

Ignoring integer effects:

```text
f(k) = L/k + k
f'(k) = -L/k^2 + 1
f'(k) = 0  =>  k = sqrt(L)
```

This derivation concerns saved activation slots for a simple chain. It is not a universal
byte estimate for arbitrary graphs or a promise about wall-clock speed.

#### Decision table

| Constraint | First lever | What it buys | What remains |
|---|---|---|---|
| Optimizer/gradient state dominates | LoRA or ZeRO-1/2/3 | Fewer trainable states or sharded states | Base weights, activations |
| Frozen base does not fit | QLoRA or parameter sharding | Smaller or distributed base storage | Activations, transient gathers |
| Long sequence/microbatch dominates | Checkpointing, smaller microbatch, efficient attention | Lower activation peak | Model state, more compute/steps |
| Need maximum update flexibility | Full FT with FSDP/ZeRO-3 | Unconstrained parameter update | Communication and high total state |
| Need many cheap task variants | LoRA adapters | Small checkpoints and base reuse | Shared base dependency |

### 1:00-1:15 - Interview Drill

Answer cold and out loud. Print the slots first, then fill them. Write the learner's final
answers into `interview-bank/systems-qa.md`; model answers are diffs, not substitutes.

1. "Estimate memory to fine-tune a 7B model with full FT, LoRA, and QLoRA."

   ```text
   [full-FT GiB under named assumptions]
   [LoRA GiB under named target/rank assumptions]
   [QLoRA idealized GiB under named quantization assumptions]
   [the dominant bucket none of these numbers includes]
   [one scope caveat]
   ```

2. "What do the ZeRO stages shard, and why is ZeRO-3/FSDP not simply peak memory divided by
   world size?"

   ```text
   [ZeRO-1 shards ...]
   [ZeRO-2 additionally shards ...]
   [ZeRO-3 additionally shards ...]
   [one transient peak term]
   [one communication/throughput tradeoff]
   ```

3. "A QLoRA run fits at short context but OOMs when sequence length doubles. Diagnose it."

   ```text
   [most likely growing memory bucket]
   [its sequence-length scaling under the named attention implementation]
   [first lever and why]
   [what QLoRA changed]
   [what QLoRA did not change]
   ```

### 1:15-1:55 - Coding Task

Build `code/finetuning_memory.py`, a staged, assumption-explicit estimator with four TODOs:

1. count LoRA parameters from targeted matrix shapes and rank;
2. decompose full-FT, LoRA, and idealized QLoRA persistent model state;
3. compute ideal per-rank ZeRO-0/1/2/3 state by component rather than dividing the total;
4. estimate checkpointed activation slots and find the best integer interval.

The checks must pin the 8,388,608-parameter adapter, the 7B worked examples, the four sharding
formulas, and the fact that checkpointing changes only the activation scenario. Every result
must retain named components so an apparently correct total cannot hide a wrong bucket.

Run each staged check as soon as its TODO is implemented. Do not add activation bytes to a
persistent-state estimate unless the activation implementation and tensor liveness are named.

### 1:55-2:00 - Deliverable and Next Session

Fill `deliverable.md` with:

- the four-bucket thesis;
- the scoped 7B full/LoRA/QLoRA numbers;
- the ZeRO stage table and the transient-materialization caveat;
- one debugging story where a technique attacked the wrong memory bucket.

Add only measured, assumption-labeled values to `shared/numbers-card.md`. Day 03 starts from
the communication introduced here: collectives, data/tensor/pipeline parallelism, and failure
modes.

## Checkpoint Questions

1. Why does LoRA reduce optimizer memory much more than activation memory?
2. What does QLoRA quantize, and in what dtype does useful training computation occur?
3. Why does ZeRO-2 leave a `2P` per-rank floor under BF16 weights?
4. What transient allocation is missing from the ideal ZeRO-3/FSDP `16P/N` expression?
5. Which memory terms does gradient checkpointing leave unchanged, and what does it spend?

## Frontier-Lab Interview Angle

Do not answer "use QLoRA" or "use FSDP" before naming the binding bucket. Start with a
component-wise estimate, distinguish persistent from peak memory, state the quality and
throughput tradeoff, and identify the first measurement that would falsify the diagnosis.

## Session Result - Complete

- Cold recall scored **8.75/11 (79.5%)**. All arithmetic was correct; the lost
  points were omitted dimensions or named memory buckets.
- The learner derived the scoped LoRA/QLoRA counts, all three ZeRO formulas,
  and the checkpoint recomputation trade. He also caught a defective
  initialization implication: zero `A` with random `B` can learn just as random
  `A` with zero `B` can; only setting both factors to zero stalls both gradients.
- All three mandatory interview drills were completed and recorded in
  `interview-bank/systems-qa.md`.
- `code/finetuning_memory.py` passes **31/31 staged checks**. TODOs 1 and 4 are
  learner-written after review. TODO 2's final implementation is tutor-written
  at the learner's request. The learner implemented TODO 3's stage semantics;
  its final declarative refactor is tutor-written at his request.
- The reference run corrected the continuous shortcut: for `L=32`,
  `ceil(L/k)+k` has a flat integer optimum at **every `k` from 4 through 8**,
  all with 12 slots. A single rounded `sqrt(L)` answer is incomplete.
