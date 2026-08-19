"""Checks for the Week 08 Day 01 GPU memory calculator."""

import pytest

from gpu_memory import (
    GIB,
    KIB,
    MIB,
    LLAMA2_7B_LIKE,
    TransformerSpec,
    kv_cache_bytes,
    max_kv_token_slots,
    model_state_memory,
    naive_attention_matrix_bytes,
    saved_hidden_state_bytes,
    tensor_bytes,
)


def test_model_state_inference_and_mixed_precision_adam() -> None:
    inference = model_state_memory(
        7_000_000_000,
        gradient_bytes=0,
        master_weight_bytes=0,
        optimizer_state_count=0,
    )
    assert inference.weights == 14_000_000_000
    assert inference.total == 14_000_000_000

    training = model_state_memory(7_000_000_000)
    assert training.weights == 14_000_000_000
    assert training.gradients == 14_000_000_000
    assert training.master_weights == 28_000_000_000
    assert training.optimizer_states == 56_000_000_000
    assert training.total == 112_000_000_000


def test_dense_tensor_storage() -> None:
    assert tensor_bytes((8, 4096, 4096), element_bytes=2) == 256 * MIB
    assert tensor_bytes((2, 3), element_bytes=4, copies=5) == 120


def test_kv_cache_known_7b_values() -> None:
    assert kv_cache_bytes(
        LLAMA2_7B_LIKE,
        batch_size=1,
        sequence_length=1,
    ) == 512 * KIB
    assert kv_cache_bytes(
        LLAMA2_7B_LIKE,
        batch_size=1,
        sequence_length=4096,
    ) == 2 * GIB
    assert kv_cache_bytes(
        LLAMA2_7B_LIKE,
        batch_size=16,
        sequence_length=8192,
    ) == 64 * GIB


def test_gqa_reduces_only_the_kv_head_factor() -> None:
    gqa = TransformerSpec(
        parameter_count=7_000_000_000,
        layers=32,
        hidden_size=4096,
        query_heads=32,
        kv_heads=8,
    )
    mha_bytes = kv_cache_bytes(
        LLAMA2_7B_LIKE,
        batch_size=2,
        sequence_length=4096,
    )
    gqa_bytes = kv_cache_bytes(gqa, batch_size=2, sequence_length=4096)
    assert mha_bytes == 4 * gqa_bytes


def test_activation_scenarios_expose_linear_and_quadratic_scaling() -> None:
    one_hidden_copy = saved_hidden_state_bytes(
        LLAMA2_7B_LIKE,
        batch_size=8,
        sequence_length=4096,
        copies_per_layer=1,
    )
    assert one_hidden_copy == 8 * GIB

    attention_4k = naive_attention_matrix_bytes(
        LLAMA2_7B_LIKE,
        batch_size=8,
        sequence_length=4096,
    )
    attention_8k = naive_attention_matrix_bytes(
        LLAMA2_7B_LIKE,
        batch_size=8,
        sequence_length=8192,
    )
    assert attention_4k == 256 * GIB
    assert attention_8k == 4 * attention_4k


def test_kv_capacity_planner_uses_aggregate_token_slots() -> None:
    slots = max_kv_token_slots(
        LLAMA2_7B_LIKE,
        capacity_bytes=24 * GIB,
        weights_bytes=13 * GIB,
        reserve_bytes=2 * GIB,
    )
    assert slots == 18_432
    assert slots // 4 == 4_608


def test_capacity_planner_returns_zero_when_fixed_memory_exhausts_device() -> None:
    assert max_kv_token_slots(
        LLAMA2_7B_LIKE,
        capacity_bytes=16 * GIB,
        weights_bytes=15 * GIB,
        reserve_bytes=2 * GIB,
    ) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0, "sequence_length": 1}, "batch_size"),
        ({"batch_size": 1, "sequence_length": 0}, "sequence_length"),
    ],
)
def test_invalid_kv_dimensions_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        kv_cache_bytes(LLAMA2_7B_LIKE, **kwargs)
