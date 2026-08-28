"""Medium PyTorch puzzle: advance a beam-search cache.

Run after implementing:
    .venv/bin/python tensor_practice/advance_beam_cache.py

Task
----
During beam-search decoding, each new beam chooses a parent from the previous
step. Reorder the cached history according to those parent choices, then append
one new vector to every resulting beam.

For new beam ``k`` in batch ``b``, ``parent_beams[b, k]`` identifies the old
beam whose entire history should be copied. ``next_vectors[b, k]`` is then
appended as that new beam's final time step.

Example
-------
cache = tensor([[[[10.], [11.]],
                 [[20.], [21.]],
                 [[30.], [31.]]]])

parent_beams = tensor([[2, 0, 2]])
next_vectors = tensor([[[32.], [12.], [33.]]])

advance_beam_cache(cache, parent_beams, next_vectors)

# tensor([[[[30.], [31.], [32.]],
#          [[10.], [11.], [12.]],
#          [[30.], [31.], [33.]]]])

Constraints
-----------
- ``cache`` has shape (B, K, T, D) and floating-point dtype.
- ``parent_beams`` has shape (B, K), dtype torch.long, and values in ``[0, K)``.
- ``next_vectors`` has shape (B, K, D) and the same dtype as ``cache``.
- All tensors are on the same CPU or CUDA device.
- B, K, T, and D are at least 1.
- Parent indices may repeat; some old beams may not be selected.
- The output has shape (B, K, T + 1, D).
- The output preserves cache's dtype, device, and autograd connectivity.
- Gradients to a parent selected multiple times must accumulate correctly.

Rules
-----
- Python loops are forbidden.
- Do not modify any input tensor in place.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def advance_beam_cache(
    cache: torch.Tensor,
    parent_beams: torch.Tensor,
    next_vectors: torch.Tensor,
) -> torch.Tensor:
    """Reorder beam histories by parent and append the next vectors."""
    B, K, T, D = cache.shape
    current_parents = torch.gather(cache, dim=1, index=parent_beams.unsqueeze(-1).unsqueeze(-1).expand(B, K, T, D))
    return torch.cat([current_parents, next_vectors.unsqueeze(dim=2)], dim=2)


def _run_checks() -> None:
    cache = torch.tensor(
        [[[[10.0], [11.0]], [[20.0], [21.0]], [[30.0], [31.0]]]],
        requires_grad=True,
    )
    parent_beams = torch.tensor([[2, 0, 2]])
    next_vectors = torch.tensor(
        [[[32.0], [12.0], [33.0]]],
        requires_grad=True,
    )
    original_cache = cache.detach().clone()
    original_next = next_vectors.detach().clone()

    result = advance_beam_cache(cache, parent_beams, next_vectors)

    expected = torch.tensor(
        [[[[30.0], [31.0], [32.0]], [[10.0], [11.0], [12.0]], [[30.0], [31.0], [33.0]]]]
    )

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(cache, original_cache)
    torch.testing.assert_close(next_vectors, original_next)
    assert result.shape == (1, 3, 3, 1)
    assert result.dtype == cache.dtype and result.device == cache.device

    result.sum().backward()
    expected_cache_grad = torch.tensor(
        [[[[1.0], [1.0]], [[0.0], [0.0]], [[2.0], [2.0]]]]
    )
    torch.testing.assert_close(cache.grad, expected_cache_grad)
    torch.testing.assert_close(next_vectors.grad, torch.ones_like(next_vectors))

    identity_cache = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]]]],
        dtype=torch.float64,
    )
    identity_parents = torch.tensor([[0, 1]])
    identity_next = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float64)
    identity_result = advance_beam_cache(
        identity_cache,
        identity_parents,
        identity_next,
    )
    expected_identity = torch.tensor(
        [[[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(identity_result, expected_identity)

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
