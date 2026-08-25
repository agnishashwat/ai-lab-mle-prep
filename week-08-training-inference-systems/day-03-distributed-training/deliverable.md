# Deliverable - Week 08 Day 03: Distributed Training Basics

## What I Built / Derived

Built `code/distributed_training.py`, a four-stage, assumption-explicit lab for:

- ring all-gather, reduce-scatter, and all-reduce traffic and ideal time;
- flat-rank mapping into DP/PP/TP coordinates and process groups;
- DDP versus FSDP per-step communication with explicit parameter gather count;
- pipeline bubbles, stage imbalance, and composed hardware utilization.

The staged harness passes **29/29 checks**.

## Core Takeaway

Distributed training partitions a named axis, not "the model" generically:
data parallelism partitions examples, tensor parallelism partitions work inside
a layer, and pipeline parallelism partitions layers. A useful scaling argument
must identify the process group, tensor, collective, per-rank bytes,
synchronization point, topology, and overlap. More GPUs help only while the
partitioned compute outweighs exposed communication, pipeline bubbles, and the
slowest rank.

## Collective Contracts

| Collective | Per-rank input | Per-rank output | Training use |
|---|---:|---:|---|
| All-gather | one `M/N` shard | full `M` | materialize FSDP parameters |
| Reduce-scatter | full `M` contribution | one reduced `M/N` shard | aggregate and shard FSDP gradients |
| All-reduce | full `M` contribution | full reduced `M` | synchronize DDP gradients |

All ranks both send and receive in the symmetric ring algorithm. Gather/scatter
describe logical placement, not one-way network participation.

## Key Equations

```text
ring chunk = M/N

all-gather or reduce-scatter sent bytes/rank = (N - 1)M/N
all-reduce sent bytes/rank                  = 2(N - 1)M/N

collective time
  ~= rounds * per-round latency + sent bytes/rank / effective BW

DDP traffic/rank  = 2cG
FSDP traffic/rank = acW + cG
where c = (N - 1)/N and a = parameter all-gathers/step

world size = DP * TP * PP
global batch = DP * microbatch/replica * accumulation

pipeline slots = m + S - 1
bubble fraction = (S - 1)/(m + S - 1)
balance efficiency = mean(stage time)/max(stage time)
utilization = [m/(m + S - 1)] * balance efficiency
```

## Parallelism Decision Table

| Axis | Partitions | Main communication | Memory effect | Placement pressure |
|---|---|---|---|---|
| DP | examples | gradient all-reduce | replicated state under DDP | can span nodes if overlap is effective |
| FSDP | examples plus state placement | parameter all-gather, gradient reduce-scatter | shards persistent state | wrapping/prefetch bound transient peaks |
| TP | matrices/work inside layers | repeated layer reductions/gathers | shards selected weights/work | keep on fastest links |
| PP | layer stack | boundary activations and gradients | each stage owns fewer layers | balance stages and control bubbles |

## Worked Results

For an equal-chunk ring with `N=8` and `M=1 GiB`:

```text
chunk = 128 MiB
all-gather or reduce-scatter = 896 MiB sent/rank in 7 rounds
all-reduce = 1.75 GiB sent/rank in 14 rounds

at 5 us/round and effective 100e9 bytes/s:
ideal all-reduce time = 18.860482 ms
```

For `W=G=1 GiB` at `N=8`:

```text
DDP gradient all-reduce                   = 1.750 GiB/rank
FSDP one gather + one reduce-scatter      = 1.750 GiB/rank
FSDP two gathers + one reduce-scatter     = 2.625 GiB/rank
```

For four pipeline stages:

```text
8 equal-time microbatches: efficiency = 8/11 = 72.73%
32 equal-time microbatches: efficiency = 32/35 = 91.43%

stage times [1,1,1,2], m=8:
balance efficiency = 62.5%
composed utilization = (8/11) * 0.625 = 45.45%
```

## Implementation Notes

- The first ring implementation treated all-gather as receive-only and
  reduce-scatter as send-only. The correction was to separate logical
  input/output placement from the bidirectional forwarding performed by every
  ring rank.
- The rank layout is DP-major, PP-middle, TP-minor:
  `rank = ((dp * PP) + pp) * TP + tp`. Process groups contain flat ranks, not
  `(axis size, coordinate)` metadata.
- DDP and one-gather FSDP can have equal leading byte counts when `W=G`, despite
  very different persistent memory and communication timing.
- The pipeline result is a padded-slot model. It is not an exact GPipe, 1F1B,
  or interleaved schedule.

## Failure Modes / Debugging Notes

- A collective hang often means one rank skipped or reordered a call, used a
  mismatched shape/dtype, or failed earlier.
- Average rank timing hides the straggler that sets synchronous wall time.
- Total collective kernel time is not exposed communication time; inspect
  overlap and the final synchronization tail.
- FSDP prefetch can improve overlap while causing multiple full wrapped units
  to be live and triggering OOM.
- Loss or quality changing with world size can come from double averaging,
  changed global batch, or duplicated/missing distributed samples.
- Multi-node TP can collapse throughput because collectives occur inside every
  layer.

## Interview Answer

Prompt:

> How do you choose and diagnose data, tensor, and pipeline parallelism?

Answer:

I first name the binding constraint and the partition axis. I use DP to split
examples when a replica fits, TP to split within-layer matrices when one
replica or layer needs multiple devices, and PP to split the layer stack when
depth/state placement requires stages. Then I quantify the repeated tensor
traffic: DDP all-reduces gradients, FSDP all-gathers parameters and
reduce-scatters gradients, TP communicates inside layers, and PP transfers
boundary activations. I map high-frequency TP groups to the fastest links and
check pipeline bubbles and stage balance. To diagnose scaling, I inspect
per-rank input, compute, collective arrival/service/tail, and optimizer
timelines; the slowest rank and exposed communication, not average utilization
or total collective time, determine the step.

## Reusable Artifact

- Path: `week-08-training-inference-systems/code/distributed_training.py`
- Checks: four stages, **29 passed**
- How to reuse: replace the ring and padded-slot assumptions with measured
  topology and scheduler data while preserving named tensor, group, traffic,
  and critical-path components.
