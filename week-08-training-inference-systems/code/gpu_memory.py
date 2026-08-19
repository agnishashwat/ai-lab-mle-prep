"""Week 08 Day 01 - first-principles GPU memory calculator.

The central rule is to count peak live bytes, not just model parameters:

    training ~= model states + saved activations + temporary buffers + runtime
    inference ~= weights + KV cache + temporary buffers + runtime

Keep every assumption explicit. In particular, "Adam uses 16 bytes/parameter"
is true only for the mixed-precision layout represented by the defaults below.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence


KIB = 1024
MIB = 1024**2
GIB = 1024**3


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


@dataclass(frozen=True)
class TransformerSpec:
    parameter_count: int
    layers: int
    hidden_size: int
    query_heads: int
    kv_heads: int

    def __post_init__(self) -> None:
        for name in ("parameter_count", "layers", "hidden_size", "query_heads", "kv_heads"):
            _require_int(name, getattr(self, name), minimum=1)
        if self.hidden_size % self.query_heads != 0:
            raise ValueError("hidden_size must be divisible by query_heads")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.query_heads


@dataclass(frozen=True)
class ModelStateMemory:
    weights: int
    gradients: int
    master_weights: int
    optimizer_states: int

    @property
    def total(self) -> int:
        return self.weights + self.gradients + self.master_weights + self.optimizer_states


def model_state_memory(
    parameter_count: int,
    *,
    weight_bytes: int = 2,
    gradient_bytes: int = 2,
    master_weight_bytes: int = 4,
    optimizer_state_count: int = 2,
    optimizer_state_bytes: int = 4,
) -> ModelStateMemory:
    """Return the persistent model-state bytes.

    Defaults model classic mixed-precision Adam:
      bf16 weights + bf16 gradients + fp32 master weights + two fp32 moments.

    Set omitted components to zero for another regime. For example, inference
    uses gradient_bytes=master_weight_bytes=optimizer_state_count=0.
    """
    _require_int("parameter_count", parameter_count, minimum=1)
    for name, value in (
        ("weight_bytes", weight_bytes),
        ("gradient_bytes", gradient_bytes),
        ("master_weight_bytes", master_weight_bytes),
        ("optimizer_state_count", optimizer_state_count),
        ("optimizer_state_bytes", optimizer_state_bytes),
    ):
        _require_int(name, value, minimum=0)

    return ModelStateMemory(
        weights=parameter_count * weight_bytes,
        gradients=parameter_count * gradient_bytes,
        master_weights=parameter_count * master_weight_bytes,
        optimizer_states=parameter_count * optimizer_state_bytes * optimizer_state_count,
    )

def tensor_bytes(shape: Sequence[int], *, element_bytes: int, copies: int = 1) -> int:
    """Return storage for `copies` dense tensors with the supplied shape."""
    _require_int("element_bytes", element_bytes, minimum=1)
    _require_int("copies", copies, minimum=1)
    for index, dimension in enumerate(shape):
        _require_int(f"shape[{index}]", dimension, minimum=1)

    return prod(shape) * element_bytes * copies


def kv_cache_bytes(
    spec: TransformerSpec,
    *,
    batch_size: int,
    sequence_length: int,
    element_bytes: int = 2,
) -> int:
    """Return persistent decoder KV-cache bytes.

    Cache layout per layer is K and V, each shaped:
        [batch_size, kv_heads, sequence_length, head_dim]
    """
    _require_int("batch_size", batch_size, minimum=1)
    _require_int("sequence_length", sequence_length, minimum=1)
    _require_int("element_bytes", element_bytes, minimum=1)

    return (
        2
        * spec.layers
        * spec.kv_heads
        * spec.head_dim
        * batch_size
        * sequence_length
        * element_bytes
    )


def max_kv_token_slots(
    spec: TransformerSpec,
    *,
    capacity_bytes: int,
    weights_bytes: int,
    reserve_bytes: int,
    kv_element_bytes: int = 2,
) -> int:
    """Return the maximum aggregate cached tokens under a simplified budget.

    Aggregate token slots are sum(sequence_length_i) over active requests.
    Equal-length batch capacity is therefore token_slots // batch_size.
    """
    _require_int("capacity_bytes", capacity_bytes, minimum=1)
    _require_int("weights_bytes", weights_bytes, minimum=0)
    _require_int("reserve_bytes", reserve_bytes, minimum=0)
    _require_int("kv_element_bytes", kv_element_bytes, minimum=1)

    per_token = kv_cache_bytes(
        spec,
        batch_size=1,
        sequence_length=1,
        element_bytes=kv_element_bytes,
    )
    remaining = capacity_bytes - weights_bytes - reserve_bytes
    if remaining <= 0:
        return 0
    return remaining // per_token


def saved_hidden_state_bytes(
    spec: TransformerSpec,
    *,
    batch_size: int,
    sequence_length: int,
    copies_per_layer: int,
    element_bytes: int = 2,
) -> int:
    """A transparent activation scenario, not a universal training estimate."""
    return tensor_bytes(
        (spec.layers, batch_size, sequence_length, spec.hidden_size),
        element_bytes=element_bytes,
        copies=copies_per_layer,
    )


def naive_attention_matrix_bytes(
    spec: TransformerSpec,
    *,
    batch_size: int,
    sequence_length: int,
    copies_per_layer: int = 1,
    element_bytes: int = 2,
) -> int:
    """Bytes if full [B, query_heads, T, T] matrices are materialized."""
    return tensor_bytes(
        (spec.layers, batch_size, spec.query_heads, sequence_length, sequence_length),
        element_bytes=element_bytes,
        copies=copies_per_layer,
    )


def gib(byte_count: int) -> float:
    return byte_count / GIB


LLAMA2_7B_LIKE = TransformerSpec(
    parameter_count=7_000_000_000,
    layers=32,
    hidden_size=4096,
    query_heads=32,
    kv_heads=32,
)
