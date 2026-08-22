"""Week 08 Day 02 - fine-tuning memory is four independent axes.

THE THESIS
    LoRA reduces TRAINABLE state.
    QLoRA also compresses FROZEN weights.
    ZeRO/FSDP shards REPLICATED state.
    Gradient checkpointing reduces SAVED ACTIVATIONS.

The numbers in this exercise are transparent lower bounds, not peak-memory
promises. They deliberately exclude temporary kernels, dequantization,
communication/materialization buffers, allocator state, and runtime headroom.

Assumptions for trainable parameters match Day 01:

    BF16 weight + BF16 gradient + FP32 master + two FP32 Adam moments
    = 2 + 2 + 4 + 8 = 16 bytes/trainable parameter

Run:
    python finetuning_memory.py
    python finetuning_memory.py lora
    python finetuning_memory.py --list
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Literal, Sequence
import math


KIB = 1024
MIB = 1024**2
GIB = 1024**3

Method = Literal["full", "lora", "qlora"]


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class LinearTarget:
    """A repeated family of same-shaped linear projections."""

    name: str
    d_in: int
    d_out: int
    count: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        for field_name in ("d_in", "d_out", "count"):
            _require_int(field_name, getattr(self, field_name), minimum=1)


@dataclass(frozen=True)
class MemoryBreakdown:
    """Persistent model-state bytes, kept separate so totals cannot hide a bad bucket."""

    frozen_weights: int
    trainable_weights: int
    gradients: int
    master_weights: int
    optimizer_states: int

    def __post_init__(self) -> None:
        for field_name in (
            "frozen_weights",
            "trainable_weights",
            "gradients",
            "master_weights",
            "optimizer_states",
        ):
            _require_int(field_name, getattr(self, field_name), minimum=0)

    @property
    def total(self) -> int:
        return (
            self.frozen_weights
            + self.trainable_weights
            + self.gradients
            + self.master_weights
            + self.optimizer_states
        )


@dataclass(frozen=True)
class CheckpointPlan:
    """Simplified chain-model activation slots for every interval k=1..layers."""

    layers: int
    peak_slots_by_interval: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_int("layers", self.layers, minimum=1)
        if len(self.peak_slots_by_interval) != self.layers:
            raise ValueError("peak_slots_by_interval must contain one value for every interval")
        for slots in self.peak_slots_by_interval:
            _require_int("peak slots", slots, minimum=1)

    def peak_slots(self, interval: int) -> int:
        _require_int("interval", interval, minimum=1)
        if interval > self.layers:
            raise ValueError("interval cannot exceed layers")
        return self.peak_slots_by_interval[interval - 1]

    @property
    def minimum_slots(self) -> int:
        return min(self.peak_slots_by_interval)

    @property
    def best_intervals(self) -> tuple[int, ...]:
        best = self.minimum_slots
        return tuple(
            interval
            for interval, slots in enumerate(self.peak_slots_by_interval, start=1)
            if slots == best
        )


LLAMA_7B_QV_TARGETS = (
    LinearTarget("q_proj", d_in=4096, d_out=4096, count=32),
    LinearTarget("v_proj", d_in=4096, d_out=4096, count=32),
)


# ---------------------------------------------------------------------------
# Exercises - implement one TODO, then run its stage
# ---------------------------------------------------------------------------

def lora_parameter_count(targets: Sequence[LinearTarget], rank: int) -> int:
    """Count LoRA A and B parameters over all targeted matrices.

    A target with shape `(d_out, d_in)` contributes
    `rank * (d_in + d_out)` parameters per occurrence. Reject rank < 1 and an
    empty target list instead of returning a success-shaped zero.
    """
    _require_int("rank", rank, minimum=1)
    if not targets:
        raise ValueError("targets must be non-empty")

    return sum(
        rank * (target.d_in + target.d_out) * target.count
        for target in targets
    )



def finetuning_state_memory(
    base_parameters: int,
    *,
    method: Method,
    adapter_parameters: int = 0,
) -> MemoryBreakdown:
    """Return unsharded persistent state under the exercise's fixed assumptions.

    `full`:
        The base is trainable at 2/2/4/8 bytes and no adapter is allowed.
    `lora`:
        The base is frozen BF16 (2 bytes/parameter); the adapter is trainable.
    `qlora`:
        The base is frozen at an ideal 4 bits/parameter, rounded up to a whole
        byte for the total tensor; the adapter is trainable.

    LoRA and QLoRA require a positive adapter count. Reject unknown methods and
    incompatible argument combinations.
    """
    _require_int("base_parameters", base_parameters, minimum=1)
    _require_int("adapter_parameters", adapter_parameters, minimum=0)

    if method == "full":
        if adapter_parameters != 0:
            raise ValueError("full fine-tuning does not use adapter parameters")
        frozen_weights = 0
        trainable_parameters = base_parameters
    elif method in ("lora", "qlora"):
        if adapter_parameters == 0:
            raise ValueError(f"{method} requires a positive adapter count")
        frozen_weight_bits = 16 if method == "lora" else 4
        frozen_weights = _ceil_div(base_parameters * frozen_weight_bits, 8)
        trainable_parameters = adapter_parameters
    else:
        raise ValueError(f"unknown fine-tuning method: {method!r}")

    return MemoryBreakdown(
        frozen_weights=frozen_weights,
        trainable_weights=2 * trainable_parameters,
        gradients=2 * trainable_parameters,
        master_weights=4 * trainable_parameters,
        optimizer_states=8 * trainable_parameters,
    )


def zero_per_rank_memory(
    state: MemoryBreakdown,
    *,
    world_size: int,
    stage: int,
    shard_frozen_weights: bool = False,
) -> MemoryBreakdown:
    """Partition a state breakdown using idealized ZeRO stage semantics.

    Stage 0 shards nothing. Stage 1 shards master weights and optimizer states.
    Stage 2 additionally shards gradients. Stage 3 additionally shards trainable
    weights. Frozen weights remain replicated unless `shard_frozen_weights=True`;
    ordinary ZeRO stage numbering alone does not say how an external frozen base
    is placed.

    Use ceiling division for every sharded component because real shard sizes
    need not divide evenly. Reject world_size < 1 and stages outside 0..3.
    """
    _require_int("world_size", world_size, minimum=1)
    _require_int("stage", stage, minimum=0)
    if stage > 3:
        raise ValueError(f"stage must be in 0..3; got {stage}")

    def partition(value: int, enabled: bool) -> int:
        return _ceil_div(value, world_size) if enabled else value

    return MemoryBreakdown(
        frozen_weights=partition(state.frozen_weights, shard_frozen_weights),
        trainable_weights=partition(state.trainable_weights, stage >= 3),
        gradients=partition(state.gradients, stage >= 2),
        master_weights=partition(state.master_weights, stage >= 1),
        optimizer_states=partition(state.optimizer_states, stage >= 1),
    )


def checkpoint_plan(layers: int) -> CheckpointPlan:
    """Build the simplified checkpoint curve for all integer intervals.

    For each interval `k` from 1 through `layers`, use:

        peak activation slots = ceil(layers / k) + k

    Return the full curve; `CheckpointPlan` derives the minimum and *all*
    minimizing intervals. This is a chain-model shape estimate, not a universal
    byte estimate.
    """
    _require_int("layers", layers, minimum=1)
    peak_activation_slots = []
    for k in range(1, layers + 1):
        peak_activation_slots.append(_ceil_div(layers, k) + k)

    return CheckpointPlan(
        layers=layers,
        peak_slots_by_interval=tuple(peak_activation_slots),
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
        self.adapter_parameters = 8_388_608


def stage_lora(ctx: Ctx) -> None:
    """TODO 1 - low-rank parameter count, multiplicity, and validation"""
    got = lora_parameter_count(LLAMA_7B_QV_TARGETS, rank=16)
    _check(
        "32 layers x q/v x rank 16 = 8,388,608 parameters",
        got == ctx.adapter_parameters,
        f"got {got:,}",
    )

    rectangular = (LinearTarget("rect", d_in=4, d_out=6, count=3),)
    _check(
        "rectangular matrices count both thin factors and multiplicity",
        lora_parameter_count(rectangular, rank=2) == 60,
    )
    _check(
        "doubling rank doubles adapter parameters exactly",
        lora_parameter_count(LLAMA_7B_QV_TARGETS, rank=32) == 2 * got,
    )

    one_full_matrix = 4096 * 4096
    one_adapter = lora_parameter_count(
        (LinearTarget("one", 4096, 4096),),
        rank=16,
    )
    _check(
        "one targeted 4096-square matrix uses exactly 0.78125% as many parameters",
        one_adapter * 100_000 == one_full_matrix * 781.25,
        f"{100 * one_adapter / one_full_matrix:.5f}%",
    )
    _check(
        "rank zero is rejected",
        _raises_value_error(lora_parameter_count, LLAMA_7B_QV_TARGETS, 0),
    )
    _check(
        "an empty target list is rejected instead of returning zero",
        _raises_value_error(lora_parameter_count, (), 16),
    )


def stage_methods(ctx: Ctx) -> None:
    """TODO 2 - full FT, LoRA, and idealized QLoRA component accounting"""
    full = finetuning_state_memory(7_000_000_000, method="full")
    _check(
        "full FT preserves the 2/2/4/8-byte component split",
        full
        == MemoryBreakdown(
            frozen_weights=0,
            trainable_weights=14_000_000_000,
            gradients=14_000_000_000,
            master_weights=28_000_000_000,
            optimizer_states=56_000_000_000,
        ),
    )
    _check(
        "full FT totals 112 GB = 104.31 GiB before activations",
        full.total == 112_000_000_000,
        f"{full.total / GIB:.5f} GiB",
    )

    lora = finetuning_state_memory(
        7_000_000_000,
        method="lora",
        adapter_parameters=ctx.adapter_parameters,
    )
    _check(
        "LoRA keeps the complete BF16 base in the frozen bucket",
        lora.frozen_weights == 14_000_000_000,
    )
    _check(
        "LoRA pays exactly 16 bytes per adapter parameter across named buckets",
        (
            lora.trainable_weights,
            lora.gradients,
            lora.master_weights,
            lora.optimizer_states,
        )
        == (
            2 * ctx.adapter_parameters,
            2 * ctx.adapter_parameters,
            4 * ctx.adapter_parameters,
            8 * ctx.adapter_parameters,
        ),
    )
    _check(
        "scoped LoRA total is 13.164 GiB",
        lora.total == 14_134_217_728,
        f"{lora.total / GIB:.6f} GiB",
    )

    qlora = finetuning_state_memory(
        7_000_000_000,
        method="qlora",
        adapter_parameters=ctx.adapter_parameters,
    )
    _check(
        "idealized QLoRA changes only the frozen base to 4 bits/parameter",
        qlora.frozen_weights == 3_500_000_000
        and (
            qlora.trainable_weights,
            qlora.gradients,
            qlora.master_weights,
            qlora.optimizer_states,
        )
        == (
            lora.trainable_weights,
            lora.gradients,
            lora.master_weights,
            lora.optimizer_states,
        ),
    )
    _check(
        "scoped idealized QLoRA total is 3.385 GiB",
        qlora.total == 3_634_217_728,
        f"{qlora.total / GIB:.6f} GiB",
    )
    _check(
        "4-bit packing rounds an odd parameter count up to a whole byte",
        finetuning_state_memory(3, method="qlora", adapter_parameters=1).frozen_weights
        == 2,
    )

    invalid_calls = (
        lambda: finetuning_state_memory(7, method="full", adapter_parameters=1),
        lambda: finetuning_state_memory(7, method="lora"),
        lambda: finetuning_state_memory(7, method="qlora"),
        lambda: finetuning_state_memory(7, method="other"),  # type: ignore[arg-type]
    )
    _check(
        "incompatible and unknown method configurations are rejected",
        all(_raises_value_error(call) for call in invalid_calls),
    )


def stage_sharding(ctx: Ctx) -> None:
    """TODO 3 - component-wise ZeRO stages and the replicated frozen-base floor"""
    full = MemoryBreakdown(
        frozen_weights=0,
        trainable_weights=14_000_000_000,
        gradients=14_000_000_000,
        master_weights=28_000_000_000,
        optimizer_states=56_000_000_000,
    )
    expected = {
        0: (14_000_000_000, 14_000_000_000, 28_000_000_000, 56_000_000_000),
        1: (14_000_000_000, 14_000_000_000, 3_500_000_000, 7_000_000_000),
        2: (14_000_000_000, 1_750_000_000, 3_500_000_000, 7_000_000_000),
        3: (1_750_000_000, 1_750_000_000, 3_500_000_000, 7_000_000_000),
    }
    totals = []
    for stage, components in expected.items():
        got = zero_per_rank_memory(full, world_size=8, stage=stage)
        totals.append(got.total)
        _check(
            f"ZeRO-{stage} shards exactly the named components",
            (
                got.trainable_weights,
                got.gradients,
                got.master_weights,
                got.optimizer_states,
            )
            == components,
            f"{got.total / GIB:.5f} GiB/rank",
        )

    _check(
        "7B totals reproduce 104.31 / 35.86 / 24.45 / 13.04 GiB",
        totals == [112_000_000_000, 38_500_000_000, 26_250_000_000, 14_000_000_000],
        str([round(total / GIB, 5) for total in totals]),
    )
    _check(
        "persistent state falls strictly with each higher stage at N=8",
        all(left > right for left, right in zip(totals, totals[1:])),
    )

    adapter = ctx.adapter_parameters
    lora = MemoryBreakdown(
        frozen_weights=14_000_000_000,
        trainable_weights=2 * adapter,
        gradients=2 * adapter,
        master_weights=4 * adapter,
        optimizer_states=8 * adapter,
    )
    adapter_only = zero_per_rank_memory(lora, world_size=8, stage=3)
    _check(
        "ZeRO-3 on ordinary LoRA leaves the 14 GB frozen base replicated",
        adapter_only.frozen_weights == 14_000_000_000
        and adapter_only.total == 14_016_777_216,
        f"{adapter_only.total / GIB:.6f} GiB/rank",
    )
    all_sharded = zero_per_rank_memory(
        lora,
        world_size=8,
        stage=3,
        shard_frozen_weights=True,
    )
    _check(
        "frozen-base sharding is an explicit independent choice",
        all_sharded.frozen_weights == 1_750_000_000
        and all_sharded.total == 1_766_777_216,
        f"{all_sharded.total / GIB:.6f} GiB/rank",
    )

    odd = MemoryBreakdown(5, 5, 5, 5, 5)
    rounded = zero_per_rank_memory(
        odd,
        world_size=2,
        stage=3,
        shard_frozen_weights=True,
    )
    _check(
        "uneven shards use a safe ceiling per component",
        rounded == MemoryBreakdown(3, 3, 3, 3, 3),
    )
    _check(
        "invalid world sizes and stages are rejected",
        _raises_value_error(zero_per_rank_memory, full, world_size=0, stage=3)
        and _raises_value_error(zero_per_rank_memory, full, world_size=8, stage=4),
    )


def stage_checkpoint(ctx: Ctx) -> None:
    """TODO 4 - the full integer checkpoint curve, including a flat optimum"""
    del ctx
    plan = checkpoint_plan(32)
    _check(
        "the curve contains one entry for every integer interval",
        len(plan.peak_slots_by_interval) == 32,
    )
    _check(
        "hand-computable intervals follow ceil(L/k)+k",
        (
            plan.peak_slots(1),
            plan.peak_slots(4),
            plan.peak_slots(6),
            plan.peak_slots(8),
            plan.peak_slots(32),
        )
        == (33, 12, 12, 12, 33),
        str(
            [
                plan.peak_slots(interval)
                for interval in (1, 4, 6, 8, 32)
            ]
        ),
    )
    _check(
        "L=32 has minimum 12 slots",
        plan.minimum_slots == 12,
    )
    _check(
        "report ALL minimizers: k=4 through k=8, not a guessed sqrt(L) point",
        plan.best_intervals == (4, 5, 6, 7, 8),
        str(plan.best_intervals),
    )

    square = checkpoint_plan(64)
    _check(
        "the perfect-square case has the unique k=sqrt(L) optimum",
        square.minimum_slots == 16 and square.best_intervals == (8,),
        f"{square.minimum_slots}, {square.best_intervals}",
    )
    _check(
        "invalid layer counts are rejected",
        _raises_value_error(checkpoint_plan, 0),
    )


STAGES = [
    ("lora", stage_lora),
    ("methods", stage_methods),
    ("sharding", stage_sharding),
    ("checkpoint", stage_checkpoint),
]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, function in STAGES:
            print(f"  {name:<12} {function.__doc__}")
        return 0

    fast = "--fast" in argv
    names = [argument for argument in argv if not argument.startswith("-")]
    known = {name for name, _ in STAGES}
    unknown = [name for name in names if name not in known]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}\nknown: {', '.join(sorted(known))}")
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
