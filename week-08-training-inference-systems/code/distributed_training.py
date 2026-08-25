"""Week 08 Day 03 - distributed training is critical-path byte accounting.

THE THESIS
    Data parallelism partitions examples.
    Tensor parallelism partitions work inside a layer.
    Pipeline parallelism partitions layers.
    Collectives and bubbles decide whether those partitions improve step time.

The ring and pipeline models here are explicit baselines, not promises about a
particular NCCL algorithm or production scheduler.

Run:
    python distributed_training.py
    python distributed_training.py collectives
    python distributed_training.py --list
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass
from typing import Literal, Sequence


MIB = 1024**2
GIB = 1024**3

Collective = Literal["all_gather", "reduce_scatter", "all_reduce"]
DataParallelStrategy = Literal["ddp", "fsdp"]


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def _require_real(name: str, value: float, *, minimum: float, strict: bool) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= minimum if strict else value < minimum)
    ):
        operator = ">" if strict else ">="
        raise ValueError(f"{name} must be a finite number {operator} {minimum}; got {value!r}")


@dataclass(frozen=True)
class RingCollectivePlan:
    """Traffic for one equal-chunk ring collective.

    `logical_tensor_bytes` is the full tensor size before sharding.
    `bytes_sent_per_rank` counts one direction; the symmetric receive volume is
    recorded separately. `ideal_seconds` assumes those directions overlap.
    """

    collective: Collective
    logical_tensor_bytes: int
    world_size: int
    chunk_bytes: int
    rounds: int
    bytes_sent_per_rank: int
    bytes_received_per_rank: int

    def ideal_seconds(
        self,
        *,
        per_round_latency_seconds: float,
        bandwidth_bytes_per_second: float,
    ) -> float:
        _require_real(
            "per_round_latency_seconds",
            per_round_latency_seconds,
            minimum=0.0,
            strict=False,
        )
        _require_real(
            "bandwidth_bytes_per_second",
            bandwidth_bytes_per_second,
            minimum=0.0,
            strict=True,
        )
        return (
            self.rounds * per_round_latency_seconds
            + self.bytes_sent_per_rank / bandwidth_bytes_per_second
        )


@dataclass(frozen=True)
class ParallelismGroups:
    """The DP/PP/TP coordinate and process groups containing one flat rank."""

    rank: int
    coordinate: tuple[int, int, int]
    data_parallel_group: tuple[int, ...]
    pipeline_parallel_group: tuple[int, ...]
    tensor_parallel_group: tuple[int, ...]


@dataclass(frozen=True)
class DataParallelTraffic:
    """Ideal per-rank sent bytes for one data-parallel training step."""

    strategy: DataParallelStrategy
    parameter_all_gathers_per_step: int
    parameter_bytes_sent_per_rank: int
    gradient_bytes_sent_per_rank: int

    @property
    def total_bytes_sent_per_rank(self) -> int:
        return (
            self.parameter_bytes_sent_per_rank
            + self.gradient_bytes_sent_per_rank
        )


@dataclass(frozen=True)
class PipelineSlotPlan:
    """A padded-slot pipeline model with explicit bubble and imbalance losses."""

    stage_seconds: tuple[float, ...]
    microbatches: int
    total_slots: int
    slot_seconds: float
    step_seconds: float
    bubble_fraction: float
    balance_efficiency: float
    hardware_utilization: float


# ---------------------------------------------------------------------------
# Exercises - implement one TODO, then run its stage
# ---------------------------------------------------------------------------

def ring_collective_plan(
    logical_tensor_bytes: int,
    *,
    world_size: int,
    collective: Collective,
) -> RingCollectivePlan:
    """Plan an equal-chunk ring collective.

    Support `all_gather`, `reduce_scatter`, and `all_reduce`. Reject non-positive
    tensor sizes, invalid world sizes, unknown collectives, and tensors that do
    not divide evenly across ranks. A one-rank group has zero rounds and traffic.
    """
    _require_int("logical_tensor_bytes", logical_tensor_bytes, minimum=1)
    _require_int("world_size", world_size, minimum=1)
    if collective not in ("all_gather", "reduce_scatter", "all_reduce"):
        raise ValueError(f"unknown collective: {collective!r}")
    if logical_tensor_bytes % world_size != 0:
        raise ValueError("logical_tensor_bytes must divide evenly across ranks")

    chunk_bytes = logical_tensor_bytes // world_size
    phases = 2 if collective == "all_reduce" else 1
    rounds = phases * (world_size - 1)
    traffic_per_rank = rounds * chunk_bytes

    return RingCollectivePlan(
        collective=collective,
        logical_tensor_bytes=logical_tensor_bytes,
        world_size=world_size,
        chunk_bytes=chunk_bytes,
        rounds=rounds,
        bytes_sent_per_rank=traffic_per_rank,
        bytes_received_per_rank=traffic_per_rank,
    )

def parallelism_groups_for_rank(
    rank: int,
    *,
    data_parallel_size: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
) -> ParallelismGroups:
    """Map one flat rank into a DP-major, PP-middle, TP-minor process mesh.

    The layout is:

        rank = ((dp * pipeline_parallel_size) + pp) * tensor_parallel_size + tp

    Return the `(dp, pp, tp)` coordinate and each group containing `rank`.
    Reject invalid dimensions and ranks outside the implied world size.
    """
    _require_int("rank", rank, minimum=0)
    _require_int("data_parallel_size", data_parallel_size, minimum=1)
    _require_int("pipeline_parallel_size", pipeline_parallel_size, minimum=1)
    _require_int("tensor_parallel_size", tensor_parallel_size, minimum=1)

    world_size = (
        data_parallel_size
        * pipeline_parallel_size
        * tensor_parallel_size
    )
    if rank >= world_size:
        raise ValueError(f"rank must be less than world size {world_size}; got {rank}")

    outer, tp = divmod(rank, tensor_parallel_size)
    dp, pp = divmod(outer, pipeline_parallel_size)

    def flat(data: int, pipeline: int, tensor: int) -> int:
        return (
            (data * pipeline_parallel_size + pipeline)
            * tensor_parallel_size
            + tensor
        )

    return ParallelismGroups(
        rank=rank,
        coordinate=(dp, pp, tp),
        data_parallel_group=tuple(
            flat(data, pp, tp)
            for data in range(data_parallel_size)
        ),
        pipeline_parallel_group=tuple(
            flat(dp, pipeline, tp)
            for pipeline in range(pipeline_parallel_size)
        ),
        tensor_parallel_group=tuple(
            flat(dp, pp, tensor)
            for tensor in range(tensor_parallel_size)
        ),
    )

def data_parallel_step_traffic(
    parameter_bytes: int,
    gradient_bytes: int,
    *,
    world_size: int,
    strategy: DataParallelStrategy,
    parameter_all_gathers_per_step: int,
) -> DataParallelTraffic:
    """Compare ideal DDP and FSDP sent bytes for one step.

    DDP performs no parameter all-gather and one gradient all-reduce, so callers
    must pass zero parameter gathers. FSDP performs an explicit positive number
    of parameter all-gathers and one gradient reduce-scatter. Use the ring
    planner rather than duplicating its formulas.
    """
    _require_int("parameter_bytes", parameter_bytes, minimum=1)
    _require_int("gradient_bytes", gradient_bytes, minimum=1)
    _require_int("world_size", world_size, minimum=1)
    _require_int(
        "parameter_all_gathers_per_step",
        parameter_all_gathers_per_step,
        minimum=0,
    )

    if strategy == "ddp":
        if parameter_all_gathers_per_step != 0:
            raise ValueError("DDP does not all-gather parameters")
        parameter_traffic = 0
        gradient_traffic = ring_collective_plan(
            gradient_bytes,
            world_size=world_size,
            collective="all_reduce",
        ).bytes_sent_per_rank
    elif strategy == "fsdp":
        if parameter_all_gathers_per_step == 0:
            raise ValueError("FSDP requires at least one parameter all-gather")
        parameter_traffic = (
            parameter_all_gathers_per_step
            * ring_collective_plan(
                parameter_bytes,
                world_size=world_size,
                collective="all_gather",
            ).bytes_sent_per_rank
        )
        gradient_traffic = ring_collective_plan(
            gradient_bytes,
            world_size=world_size,
            collective="reduce_scatter",
        ).bytes_sent_per_rank
    else:
        raise ValueError(f"unknown data-parallel strategy: {strategy!r}")

    return DataParallelTraffic(
        strategy=strategy,
        parameter_all_gathers_per_step=parameter_all_gathers_per_step,
        parameter_bytes_sent_per_rank=parameter_traffic,
        gradient_bytes_sent_per_rank=gradient_traffic,
    )


def pipeline_slot_plan(
    stage_seconds: Sequence[float],
    *,
    microbatches: int,
) -> PipelineSlotPlan:
    """Estimate bubbles and imbalance in an ideal padded-slot pipeline.

    Treat each stage time as useful forward-plus-backward work for one
    microbatch. Pad every slot to the slowest stage and use `m + S - 1` slots.
    Return bubble fraction, balance efficiency, and their composed hardware
    utilization. Reject an empty pipeline, non-positive/non-finite times, and
    invalid microbatch counts.
    """
    _require_int("microbatches", microbatches, minimum=1)
    raw_stage_seconds = tuple(stage_seconds)
    if not raw_stage_seconds:
        raise ValueError("stage_seconds must be non-empty")
    for seconds in raw_stage_seconds:
        _require_real("stage_seconds", seconds, minimum=0.0, strict=True)

    times = tuple(float(seconds) for seconds in raw_stage_seconds)
    stages = len(times)
    total_slots = microbatches + stages - 1
    slot_seconds = max(times)
    bubble_fraction = (stages - 1) / total_slots
    balance_efficiency = (sum(times) / stages) / slot_seconds
    hardware_utilization = (
        (1.0 - bubble_fraction) * balance_efficiency
    )

    return PipelineSlotPlan(
        stage_seconds=times,
        microbatches=microbatches,
        total_slots=total_slots,
        slot_seconds=slot_seconds,
        step_seconds=total_slots * slot_seconds,
        bubble_fraction=bubble_fraction,
        balance_efficiency=balance_efficiency,
        hardware_utilization=hardware_utilization,
    )


# ---------------------------------------------------------------------------
# Staged self-check harness
# ---------------------------------------------------------------------------

_TALLY = {"pass": 0, "fail": 0}


def _check(name: str, condition: bool, detail: str = "") -> None:
    condition = bool(condition)
    _TALLY["pass" if condition else "fail"] += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{suffix}")
    if not condition:
        raise AssertionError(name)


def _raises_value_error(function, *args, **kwargs) -> bool:
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    return False


class Ctx:
    def __init__(self, fast: bool = False):
        self.fast = fast
        self.payload = GIB
        self.world_size = 8


def stage_collectives(ctx: Ctx) -> None:
    """TODO 1 - exact ring chunks, rounds, traffic, and ideal time"""
    gather = ring_collective_plan(
        ctx.payload,
        world_size=ctx.world_size,
        collective="all_gather",
    )
    scatter = ring_collective_plan(
        ctx.payload,
        world_size=ctx.world_size,
        collective="reduce_scatter",
    )
    reduce = ring_collective_plan(
        ctx.payload,
        world_size=ctx.world_size,
        collective="all_reduce",
    )

    _check(
        "1 GiB / 8 ranks gives exact 128 MiB ring chunks",
        gather.chunk_bytes == 128 * MIB
        and scatter.chunk_bytes == 128 * MIB
        and reduce.chunk_bytes == 128 * MIB,
    )
    _check(
        "all-gather takes 7 rounds and sends/receives 896 MiB per rank",
        gather
        == RingCollectivePlan(
            collective="all_gather",
            logical_tensor_bytes=GIB,
            world_size=8,
            chunk_bytes=128 * MIB,
            rounds=7,
            bytes_sent_per_rank=896 * MIB,
            bytes_received_per_rank=896 * MIB,
        ),
    )
    _check(
        "reduce-scatter has the same ring traffic as all-gather",
        scatter.rounds == gather.rounds
        and scatter.bytes_sent_per_rank == gather.bytes_sent_per_rank
        and scatter.bytes_received_per_rank == gather.bytes_received_per_rank,
    )
    _check(
        "all-reduce is reduce-scatter plus all-gather: 1.75 GiB/rank",
        reduce.rounds == gather.rounds + scatter.rounds
        and reduce.bytes_sent_per_rank
        == gather.bytes_sent_per_rank + scatter.bytes_sent_per_rank
        and reduce.bytes_sent_per_rank == 1_792 * MIB,
        f"{reduce.bytes_sent_per_rank / GIB:.2f} GiB",
    )

    expected_seconds = 14 * 5e-6 + (1_792 * MIB) / (100 * 10**9)
    got_seconds = reduce.ideal_seconds(
        per_round_latency_seconds=5e-6,
        bandwidth_bytes_per_second=100 * 10**9,
    )
    _check(
        "alpha-bandwidth time adds 14 latencies to one-direction sent bytes/BW",
        math.isclose(got_seconds, expected_seconds, rel_tol=0.0, abs_tol=1e-15),
        f"{got_seconds * 1_000:.6f} ms",
    )

    singleton = ring_collective_plan(
        GIB,
        world_size=1,
        collective="all_reduce",
    )
    _check(
        "a one-rank process group communicates zero bytes in zero rounds",
        singleton.rounds == 0
        and singleton.bytes_sent_per_rank == 0
        and singleton.bytes_received_per_rank == 0,
    )

    invalid_calls = (
        lambda: ring_collective_plan(0, world_size=8, collective="all_reduce"),
        lambda: ring_collective_plan(GIB, world_size=0, collective="all_reduce"),
        lambda: ring_collective_plan(10, world_size=3, collective="all_gather"),
        lambda: ring_collective_plan(
            GIB,
            world_size=8,
            collective="broadcast",  # type: ignore[arg-type]
        ),
    )
    _check(
        "invalid sizes, uneven chunks, and unknown collectives are rejected",
        all(_raises_value_error(call) for call in invalid_calls),
    )


def stage_mesh(ctx: Ctx) -> None:
    """TODO 2 - flat-rank coordinates and orthogonal process groups"""
    del ctx
    got = parallelism_groups_for_rank(
        11,
        data_parallel_size=2,
        pipeline_parallel_size=2,
        tensor_parallel_size=4,
    )
    _check(
        "rank 11 maps to coordinate (dp=1, pp=0, tp=3)",
        got.coordinate == (1, 0, 3),
        str(got.coordinate),
    )
    _check(
        "TP is the contiguous fastest-varying group",
        got.tensor_parallel_group == (8, 9, 10, 11),
        str(got.tensor_parallel_group),
    )
    _check(
        "PP varies only the pipeline coordinate",
        got.pipeline_parallel_group == (11, 15),
        str(got.pipeline_parallel_group),
    )
    _check(
        "DP varies only the data coordinate",
        got.data_parallel_group == (3, 11),
        str(got.data_parallel_group),
    )

    plans = [
        parallelism_groups_for_rank(
            rank,
            data_parallel_size=2,
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
        )
        for rank in range(16)
    ]
    _check(
        "all 16 ranks have unique DP/PP/TP coordinates",
        len({plan.coordinate for plan in plans}) == 16,
    )
    _check(
        "every group has the size of the axis it varies",
        all(
            len(plan.data_parallel_group) == 2
            and len(plan.pipeline_parallel_group) == 2
            and len(plan.tensor_parallel_group) == 4
            for plan in plans
        ),
    )

    invalid_calls = (
        lambda: parallelism_groups_for_rank(
            0,
            data_parallel_size=0,
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
        ),
        lambda: parallelism_groups_for_rank(
            16,
            data_parallel_size=2,
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
        ),
    )
    _check(
        "invalid mesh dimensions and out-of-range ranks are rejected",
        all(_raises_value_error(call) for call in invalid_calls),
    )


def stage_data_parallel(ctx: Ctx) -> None:
    """TODO 3 - DDP all-reduce versus explicit FSDP materializations"""
    ddp = data_parallel_step_traffic(
        ctx.payload,
        ctx.payload,
        world_size=ctx.world_size,
        strategy="ddp",
        parameter_all_gathers_per_step=0,
    )
    fsdp_keep = data_parallel_step_traffic(
        ctx.payload,
        ctx.payload,
        world_size=ctx.world_size,
        strategy="fsdp",
        parameter_all_gathers_per_step=1,
    )
    fsdp_reshard = data_parallel_step_traffic(
        ctx.payload,
        ctx.payload,
        world_size=ctx.world_size,
        strategy="fsdp",
        parameter_all_gathers_per_step=2,
    )

    _check(
        "DDP moves no parameters and all-reduces 1.75 GiB of gradients/rank",
        ddp.parameter_bytes_sent_per_rank == 0
        and ddp.gradient_bytes_sent_per_rank == 1_792 * MIB
        and ddp.total_bytes_sent_per_rank == 1_792 * MIB,
    )
    _check(
        "one FSDP gather plus reduce-scatter equals DDP's leading bytes when W=G",
        fsdp_keep.parameter_bytes_sent_per_rank == 896 * MIB
        and fsdp_keep.gradient_bytes_sent_per_rank == 896 * MIB
        and fsdp_keep.total_bytes_sent_per_rank
        == ddp.total_bytes_sent_per_rank,
    )
    _check(
        "re-gathering for backward raises FSDP traffic to 2.625 GiB/rank",
        fsdp_reshard.parameter_bytes_sent_per_rank == 1_792 * MIB
        and fsdp_reshard.gradient_bytes_sent_per_rank == 896 * MIB
        and fsdp_reshard.total_bytes_sent_per_rank == 2_688 * MIB,
        f"{fsdp_reshard.total_bytes_sent_per_rank / GIB:.3f} GiB",
    )
    _check(
        "two-gather FSDP moves exactly 1.5x DDP bytes when W=G",
        2 * fsdp_reshard.total_bytes_sent_per_rank
        == 3 * ddp.total_bytes_sent_per_rank,
    )

    ddp_larger_parameters = data_parallel_step_traffic(
        2 * GIB,
        GIB,
        world_size=8,
        strategy="ddp",
        parameter_all_gathers_per_step=0,
    )
    _check(
        "DDP communication depends on gradient bytes, not parameter bytes",
        ddp_larger_parameters.total_bytes_sent_per_rank
        == ddp.total_bytes_sent_per_rank,
    )

    singleton = data_parallel_step_traffic(
        GIB,
        GIB,
        world_size=1,
        strategy="fsdp",
        parameter_all_gathers_per_step=2,
    )
    _check(
        "all data-parallel communication vanishes at world size one",
        singleton.total_bytes_sent_per_rank == 0,
    )

    invalid_calls = (
        lambda: data_parallel_step_traffic(
            GIB,
            GIB,
            world_size=8,
            strategy="ddp",
            parameter_all_gathers_per_step=1,
        ),
        lambda: data_parallel_step_traffic(
            GIB,
            GIB,
            world_size=8,
            strategy="fsdp",
            parameter_all_gathers_per_step=0,
        ),
        lambda: data_parallel_step_traffic(
            GIB,
            GIB,
            world_size=8,
            strategy="other",  # type: ignore[arg-type]
            parameter_all_gathers_per_step=0,
        ),
    )
    _check(
        "incompatible gather counts and unknown strategies are rejected",
        all(_raises_value_error(call) for call in invalid_calls),
    )


def stage_pipeline(ctx: Ctx) -> None:
    """TODO 4 - bubble loss, stage imbalance, and composed utilization"""
    del ctx
    equal = pipeline_slot_plan((1.0, 1.0, 1.0, 1.0), microbatches=8)
    _check(
        "four stages and eight microbatches require 11 padded slots",
        equal.total_slots == 11 and equal.step_seconds == 11.0,
    )
    _check(
        "equal-stage bubble fraction is 3/11",
        math.isclose(equal.bubble_fraction, 3 / 11)
        and math.isclose(equal.hardware_utilization, 8 / 11),
        f"{100 * equal.hardware_utilization:.2f}%",
    )
    _check(
        "balanced stages have 100% balance efficiency",
        equal.balance_efficiency == 1.0,
    )
    _check(
        "utilization decomposes into bubble efficiency times balance efficiency",
        math.isclose(
            equal.hardware_utilization,
            (1.0 - equal.bubble_fraction) * equal.balance_efficiency,
        ),
    )

    imbalanced = pipeline_slot_plan((1.0, 1.0, 1.0, 2.0), microbatches=8)
    _check(
        "[1,1,1,2] stage times have 62.5% balance efficiency",
        imbalanced.slot_seconds == 2.0
        and imbalanced.step_seconds == 22.0
        and math.isclose(imbalanced.balance_efficiency, 0.625),
    )
    _check(
        "bubble and imbalance compose to 5/11 = 45.45% utilization",
        math.isclose(imbalanced.hardware_utilization, 5 / 11),
        f"{100 * imbalanced.hardware_utilization:.2f}%",
    )

    deeper = pipeline_slot_plan((1.0, 1.0, 1.0, 1.0), microbatches=32)
    _check(
        "more microbatches reduce bubbles without changing stage balance",
        math.isclose(deeper.bubble_fraction, 3 / 35)
        and deeper.balance_efficiency == equal.balance_efficiency
        and deeper.hardware_utilization > equal.hardware_utilization,
    )

    invalid_calls = (
        lambda: pipeline_slot_plan((), microbatches=8),
        lambda: pipeline_slot_plan((1.0, 0.0), microbatches=8),
        lambda: pipeline_slot_plan((1.0, math.inf), microbatches=8),
        lambda: pipeline_slot_plan((1.0, 1.0), microbatches=0),
    )
    _check(
        "empty pipelines, invalid stage times, and zero microbatches are rejected",
        all(_raises_value_error(call) for call in invalid_calls),
    )


STAGES = [
    ("collectives", stage_collectives),
    ("mesh", stage_mesh),
    ("data_parallel", stage_data_parallel),
    ("pipeline", stage_pipeline),
]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, function in STAGES:
            print(f"  {name:<14} {function.__doc__}")
        return 0

    fast = "--fast" in argv
    names = [argument for argument in argv if not argument.startswith("-")]
    known = {name for name, _ in STAGES}
    unknown = [name for name in names if name not in known]
    if unknown:
        print(
            f"unknown stage(s): {', '.join(unknown)}\n"
            f"known: {', '.join(sorted(known))}"
        )
        return 2

    ctx = Ctx(fast=fast)
    selected = [
        (name, function)
        for name, function in STAGES
        if not names or name in names
    ]
    todo: list[str] = []
    broken: list[str] = []

    for name, function in selected:
        print(f"\n=== {name} ===")
        try:
            function(ctx)
        except NotImplementedError:
            print("  [TODO] not implemented yet - skipped")
            todo.append(name)
        except AssertionError:
            broken.append(name)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            broken.append(name)

    print(f"\n{'=' * 60}")
    print(
        f"checks: {_TALLY['pass']} passed, {_TALLY['fail']} failed"
        + (f" | stages TODO: {', '.join(todo)}" if todo else "")
        + (f" | stages failing: {', '.join(sorted(set(broken)))}" if broken else "")
    )
    if not todo and not broken:
        print("All invariants passed.")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
