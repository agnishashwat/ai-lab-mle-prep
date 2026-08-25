# Week 08 Day 03 - Distributed Training: Which Bytes Cross Which Link

## Objective

Given a model, cluster topology, process mesh, and microbatch schedule:

1. state what each collective consumes and produces;
2. estimate ideal per-rank communication volume and latency;
3. distinguish data, tensor, and pipeline parallelism by the axis each partitions;
4. identify the synchronization point, memory trade, and dominant failure mode of each.

The thesis:

> Parallelism only helps when the partitioned work outweighs the bytes and bubbles placed on
> the critical path. "More GPUs" is not a scaling argument.

## Prerequisites

- Week 08 Day 01: `global_batch = DP * microbatch * accumulation_steps`.
- Week 08 Day 02: DDP replication, ZeRO/FSDP sharding, transient parameter materialization.
- Transformer linear-layer shapes and the distinction between parameters and activations.

## Required Reading

- PyTorch distributed collectives documentation: `all_reduce`, `all_gather_into_tensor`, and
  `reduce_scatter_tensor`.
- Shoeybi et al., "Megatron-LM", model-parallel transformer sections.
- Optional: Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters",
  the 3D parallelism and pipeline-scheduling sections.

## Assumptions and Vocabulary

- A **rank** is one process participating in a process group.
- **World size** `N` is the number of ranks in that group, not necessarily every GPU in the job.
- A **collective** must be called by every rank in a group in a compatible order, with compatible
  shapes and dtypes.
- `M` below is the logical full tensor size in bytes. The ring formulas assume equal chunks,
  homogeneous links, no contention, and full overlap of symmetric sends and receives.
- `BW` is effective one-direction bytes/second available to this collective, not an advertised
  aggregate fabric number.
- The latency-bandwidth model is:

  ```text
  collective time ~= rounds * per-round latency + bytes sent per rank / BW
  ```

Real NCCL implementations select among ring, tree, and topology-aware algorithms. The formulas
here are transparent baselines, not universal profiler predictions.

## 2-Hour Plan

### 0:00-0:10 - Cold Recall

Read `shared/numbers-card.md` out loud first. Then answer without notes or scrollback.

1. A 7B full fine-tune uses the Day 01 `2 + 2 + 4 + 8` byte layout at world size eight.

   ```text
   ZeRO-2 persistent state per rank [GiB, 3 dp]: [___]
   replicated ZeRO-2 component [name and bytes/parameter]: [___]
   ZeRO-3 persistent state per rank [GiB, 3 dp]: [___]
   one peak-memory term missing from ZeRO-3's 16P/N [named allocation]: [___]
   ```

2. An FSDP-wrapped unit owns only a parameter shard while idle.

   ```text
   collective before compute [name]: [___]
   its per-rank input -> output [shard/full]: [___]
   collective after gradient computation [name]: [___]
   its per-rank input -> output [full/shard]: [___]
   ```

3. Global batch is 256, data-parallel size is eight, and microbatch per rank is four.

   ```text
   accumulation steps [integer]: [___]
   accumulation after DP doubles to 16, holding global batch and microbatch fixed: [___]
   model-state memory per DDP rank after doubling DP [increase/decrease/unchanged]: [___]
   ```

Score all eleven slots. Arithmetic without the named component or allocation does not earn the
corresponding mechanism point.

### 0:10-0:35 - First-Principles Collectives and Parallelism

#### The five operations to keep separate

| Operation | Per-rank input | Per-rank output | Typical training use |
|---|---|---|---|
| Broadcast | root has `M` | every rank has `M` | initialization or metadata |
| All-gather | one `M/N` shard | full `M` tensor | materialize sharded parameters |
| Reduce-scatter | full `M` contribution | one reduced `M/N` shard | aggregate and shard gradients |
| All-reduce | full `M` contribution | full reduced `M` tensor | synchronize replicated gradients |
| All-to-all | partitioned `M` | differently partitioned `M` | route tokens in expert parallelism |

For a ring with `N` ranks and a tensor of `M` bytes:

```text
chunk bytes                  = M/N
all-gather sent/rank         = (N - 1)M/N
reduce-scatter sent/rank     = (N - 1)M/N
all-reduce sent/rank         = 2(N - 1)M/N
all-reduce                   = reduce-scatter + all-gather
```

The receive volume is the same under this symmetric ring model. Do not double the bandwidth term
again if send and receive are simultaneous and `BW` already represents the limiting direction.

A reduction also has a numerical contract. Some APIs produce a sum; frameworks often divide by
world size to form an average. Dividing in both the loss and the synchronization hook silently
shrinks the gradient by another factor of `N`.

#### DDP: partition examples, replicate the model

Each rank runs the full model on a different microbatch. Backward produces a full local gradient;
an all-reduce gives every rank the same reduced gradient before the optimizer step.

```text
compute/rank              ~= total compute / DP
persistent model state    ~= replicated
gradient traffic/rank     ~= 2(N - 1)G/N
global batch              = DP * microbatch * accumulation
```

DDP is attractive when the model state fits per rank and enough compute lies between gradient
bucket synchronizations. It is limited by large gradient traffic, small buckets, stragglers, and
global-batch constraints. Gradient accumulation can defer synchronization with `no_sync`; it does
not reduce the final synchronized gradient's logical size.

#### FSDP/ZeRO-3: shard state, materialize just in time

For one wrapped unit:

1. all-gather parameter shards before compute;
2. compute with the full unit;
3. optionally re-shard after forward;
4. re-all-gather for backward if it was re-sharded;
5. reduce-scatter the full gradient contributions into gradient shards.

Let `a` be the number of parameter all-gathers per step and let
`c = (N - 1)/N`. The ideal sent bytes per rank are:

```text
DDP:         2cG
FSDP:        acW + cG
```

If `W = G` and `a = 1`, the leading byte count equals DDP even though the timing and memory
liveness differ. If forward parameters are re-sharded and backward needs a second gather, `a = 2`
and the ideal count becomes `3cW`, 1.5 times DDP's `2cW`. Activation checkpoint recomputation,
wrapping boundaries, and prefetch can change the observed gather count.

Prefetch can hide communication under compute, but it may make two full units live together.
That is the direct link between Day 02's transient-materialization caveat and today's collectives.

#### Tensor parallelism: partition work inside each layer

For `Y = XW`, with `X` shaped `[tokens, d_in]` and `W` shaped `[d_in, d_out]`:

- **Column parallel:** split `W` along `d_out`. Each rank produces different output features.
  The output may remain partitioned or be all-gathered.
- **Row parallel:** split `W` along `d_in` and split `X` compatibly. Each rank produces a partial
  sum for every output feature; those partials need a reduction.

Transformer implementations pair placements so one layer's partitioned output feeds the next
without an unnecessary gather. Even so, tensor parallelism puts collectives inside the repeated
layer body. It is latency-sensitive and normally belongs inside the fastest topology domain,
such as the GPUs connected by NVLink within one node.

Tensor parallelism reduces per-rank parameter and matrix-multiply work, but it does not mean every
activation divides by `TP`. Replicated residuals, layer norms, logits, communication buffers, and
temporary kernels remain implementation-dependent.

#### Pipeline parallelism: partition layers

Pipeline stages own contiguous or otherwise assigned layer groups. Adjacent stages exchange
activations in forward and activation gradients in backward. Microbatches keep multiple stages
busy, but the pipeline must fill and drain.

In an ideal padded-slot model with `S` stages and `m` microbatches:

```text
total slots                = m + S - 1
bubble fraction            = (S - 1) / (m + S - 1)
bubble-only efficiency     = m / (m + S - 1)
```

This is an explicit simplified model. GPipe, 1F1B, interleaving, recomputation, and unequal
forward/backward times change the exact schedule. The invariant remains: too few microbatches,
an imbalanced slow stage, or a slow boundary link leaves expensive devices idle.

#### 3D parallelism and topology

```text
world size = DP * TP * PP
```

- A **TP group** holds DP and PP coordinates fixed and varies the tensor coordinate.
- A **PP group** holds DP and TP fixed and varies the pipeline coordinate.
- A **DP group** holds PP and TP fixed and varies the data coordinate.

Typical placement keeps high-frequency TP collectives inside a node, places PP boundaries across
links that can carry activation traffic, and uses DP across replicas. That is a starting heuristic,
not a substitute for measuring the actual tensor sizes and fabric.

### 0:35-1:00 - Derivations and Failure Analysis

#### Derivation 1: ring all-reduce volume

Split `M` into `N` equal chunks. Reduce-scatter takes `N - 1` rounds, sending one `M/N` chunk per
round. All-gather does the same:

```text
reduce-scatter bytes/rank = (N - 1)M/N
all-gather bytes/rank     = (N - 1)M/N
all-reduce bytes/rank     = 2(N - 1)M/N
```

For `M = 1 GiB` and `N = 8`:

```text
chunk                      = 128 MiB
all-gather                 = 7 * 128 MiB = 896 MiB/rank
reduce-scatter             = 896 MiB/rank
all-reduce                 = 1,792 MiB/rank = 1.75 GiB/rank
```

As `N` grows, the all-reduce byte volume approaches `2M`; it does not approach zero. Strong
scaling reduces compute per rank while leaving a communication floor.

#### Derivation 2: strong-scaling efficiency

For a fixed global workload:

```text
T_step(N) ~= T_compute(1)/N
             + T_exposed_communication(N)
             + T_pipeline_bubble(N)
             + T_imbalance(N)

strong-scaling efficiency = T_step(1) / (N * T_step(N))
```

Only communication left on the critical path belongs in the exposed term. A profiler can show
high total collective time while useful overlap keeps the step fast, or low nominal traffic while
many tiny latency-bound collectives serialize the step.

#### Derivation 3: pipeline bubbles and imbalance compose

Let each stage's useful work per microbatch be `t_i`, and pad every slot to the bottleneck stage
`t_max`. Then:

```text
balance efficiency = mean(t_i) / max(t_i)
hardware utilization
  = bubble-only efficiency * balance efficiency
  = [m/(m + S - 1)] * [mean(t_i)/max(t_i)]
```

For four equal stages and eight microbatches, bubble-only efficiency is `8/11 = 72.73%`. Increasing
to 32 microbatches gives `32/35 = 91.43%`, but smaller microbatches may reduce kernel efficiency
and increase launch/communication overhead.

For stage times `[1, 1, 1, 2]`, balance efficiency is `1.25/2 = 62.5%`. With eight microbatches,
the padded-slot utilization is only `(8/11) * 0.625 = 45.45%`.

#### Failure modes to diagnose before changing the model

| Symptom | Likely mechanism | First evidence |
|---|---|---|
| Job hangs at one collective | rank skipped/reordered a call; shape mismatch; failed rank | per-rank traces and last collective sequence number |
| More GPUs make the step slower | compute shrank below exposed latency/traffic | compute/collective overlap timeline |
| Periodic long steps | input or host straggler; allocator/GC; network contention | per-rank step-time distribution |
| One pipeline stage is always busy | partition or token-count imbalance | stage service times and queue depth |
| OOM only with prefetch | overlapping full-unit materializations | peak live tensors around all-gather |
| Loss changes with world size | wrong reduction or batch semantics | gradient norm before/after reduction |
| Data-parallel quality regression | duplicated/missing samples or changed global batch | sampler indices and effective examples/update |
| Multi-node TP collapses throughput | high-frequency layer collectives on slow links | process-group placement and per-layer collective time |

A synchronous job advances at the slowest rank. Means hide the rank that determines wall-clock
time; report maxima or rank distributions for data, compute, and communication phases.

### 1:00-1:15 - Interview Drill

Answer cold and out loud. Print the slots first, then fill them. Write the learner's final answers
into `interview-bank/systems-qa.md`; model answers are diffs, not substitutes.

1. "Explain all-reduce, all-gather, and reduce-scatter, then connect them to DDP and FSDP."

   ```text
   all-reduce input -> output and DDP use: [___]
   all-gather input -> output and FSDP use: [___]
   reduce-scatter input -> output and FSDP use: [___]
   ring all-reduce bytes/rank: [___]
   one peak/throughput caveat: [___]
   ```

2. "A replica fits only after TP=8. You have 64 GPUs arranged as eight 8-GPU nodes. Propose an
   initial DP/TP/PP mesh and defend the placement."

   ```text
   DP * TP * PP = 64 proposal: [___]
   highest-frequency collective and placement: [___]
   global-batch consequence: [___]
   condition that would force PP above one: [___]
   first profiler measurement that could falsify the plan: [___]
   ```

3. "A fixed-workload training step takes 1.0 s on eight GPUs and 0.42 s on 32 GPUs. Diagnose the
   scaling result rather than saying the network is slow."

   ```text
   speedup from 8 -> 32 [x, 3 dp]: [___]
   relative scaling efficiency [%, 1 dp]: [___]
   four-way ideal step time [seconds]: [___]
   first timing decomposition to inspect: [___]
   one communication fix and one non-communication hypothesis: [___]
   ```

### 1:15-1:55 - Coding Task

Build `code/distributed_training.py`, a staged, assumption-explicit lab with four TODOs:

1. compute ring all-gather, reduce-scatter, and all-reduce rounds, chunks, bytes/rank, and ideal
   latency-bandwidth time;
2. map a flat rank into DP/PP/TP coordinates and construct its three process groups;
3. compare DDP gradient all-reduce with FSDP parameter all-gathers plus gradient reduce-scatter,
   keeping the number of parameter materializations explicit;
4. compute pipeline bubbles, stage-balance efficiency, and their composed utilization.

Run one stage at a time:

```bash
python3 week-08-training-inference-systems/code/distributed_training.py collectives
python3 week-08-training-inference-systems/code/distributed_training.py mesh
python3 week-08-training-inference-systems/code/distributed_training.py data_parallel
python3 week-08-training-inference-systems/code/distributed_training.py pipeline
```

The checks pin exact `N=8, M=1 GiB` ring volumes, a `DP=2, PP=2, TP=4` rank map, the DDP/FSDP
traffic comparison, and the `8/11` pipeline-efficiency result. Keep sent bytes distinct from
logical tensor size and state every scheduling assumption.

### 1:55-2:00 - Deliverable and Next Session

Fill `deliverable.md` with:

- the collective input/output table and ring formulas;
- a DP/TP/PP decision table with memory, communication, and topology tradeoffs;
- one DDP-versus-FSDP byte calculation;
- one pipeline bubble/imbalance calculation;
- a debugging checklist that starts from per-rank evidence.

Add only stable, assumption-labeled measurements to `shared/numbers-card.md`. Day 04 uses this
same critical-path reasoning for continuous batching, paged KV cache, and serving throughput.

## Checkpoint Questions

1. Why does ring all-reduce approach `2M` sent bytes per rank rather than `2M/N`?
2. Under what resharding assumption can FSDP move more bytes than DDP despite using less memory?
3. Why is tensor parallelism normally kept inside the fastest topology domain?
4. How do microbatch count and stage imbalance independently reduce pipeline utilization?
5. Why can average rank time look healthy while synchronous step time is poor?

## Frontier-Lab Interview Angle

Do not answer a scaling question with only a framework name. State the partition axis, process
group, tensor shape, collective, per-rank bytes, synchronization point, topology, and overlap
assumption. Then identify the rank or stage on the critical path and the measurement that would
disprove the proposed bottleneck.

## Session Result - Complete

- Cold recall scored **8/11 (72.7%)**. The arithmetic was strong; the misses
  were multiplying ZeRO-3's per-parameter result by all 7B parameters and
  distinguishing all-gather/reduce-scatter from plain gather/scatter.
- The learner completed the `N=4, M=400 MiB` collective exercise exactly:
  100 MiB chunks, 300 MiB/rank for each one-phase collective, and
  600 MiB/rank for all-reduce.
- The mandatory interview drill was completed and recorded in
  `interview-bank/systems-qa.md`. The scored answers total **10/15** across the
  three entries. Q1 and Q3's precision repairs were tutor-supplied at the
  learner's request; Q2 used prompting and a selected profiler criterion.
- `code/distributed_training.py` passes **29/29 checks**. TODO 1 began
  learner-authored; its traffic semantics and final construction are
  tutor-written at request. TODO 2 began learner-authored; its validation and
  process-group construction are tutor-written at request. TODOs 3 and 4 are
  tutor-written at request.
- The central debugging correction was that collective names describe logical
  input/output placement, not one-way network traffic. Every rank sends and
  receives in the symmetric ring because it both forwards shards others need
  and receives shards or partial reductions needed for its own output.
- Stable results: `N=8, M=1 GiB` ring all-reduce sends **1.75 GiB/rank**;
  one-gather FSDP matches DDP's leading bytes at `W=G`, while two-gather FSDP
  sends **2.625 GiB/rank**; four equal pipeline stages at `m=8` reach
  **72.73%** bubble-only efficiency, falling to **45.45%** with stage times
  `[1,1,1,2]`.
