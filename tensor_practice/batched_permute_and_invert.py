"""Easy PyTorch tensor puzzle: batched permutation and inverse.

Run after implementing:
    .venv/bin/python tensor_practice/batched_permute_and_invert.py

Task
----
Apply a different permutation to each batch row of ``x`` and construct the
inverse permutation.

For every batch ``b`` and position ``j``:

    reordered[b, j] = x[b, order[b, j]]
    inverse_order[b, order[b, j]] = j

Example
-------
x = tensor([[[10., 11.], [20., 21.], [30., 31.]],
            [[ 1.,  2.], [ 3.,  4.], [ 5.,  6.]]])

order = tensor([[2, 0, 1],
                [1, 2, 0]])

reordered, inverse_order = permute_and_invert(x, order)

reordered == tensor([[[30., 31.], [10., 11.], [20., 21.]],
                     [[ 3.,  4.], [ 5.,  6.], [ 1.,  2.]]])

inverse_order == tensor([[1, 2, 0],
                         [2, 0, 1]])

Constraints
-----------
- ``x`` has shape (B, N, D) and floating-point dtype.
- ``order`` has shape (B, N), dtype torch.long, and the same device as ``x``.
- Every row of ``order`` is a permutation of the integers ``0`` through ``N - 1``.
- B, N, and D are at least 1.
- ``reordered`` has shape (B, N, D) and preserves x's dtype, device, and
  autograd connectivity.
- ``inverse_order`` has shape (B, N), dtype torch.long, and order's device.
- Gathering ``reordered`` with ``inverse_order`` must reconstruct ``x``.

Rules
-----
- Use ``torch.gather`` (or ``Tensor.gather``) to create ``reordered``.
- Use ``torch.scatter`` (or ``Tensor.scatter``) to create ``inverse_order``.
- Python loops over batches, positions, or features are forbidden.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def permute_and_invert(
    x: torch.Tensor,
    order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply each row's permutation and return it together with its inverse."""
    B, N, D = x.shape
    order_expanded = order.unsqueeze(-1).expand(B, N, D)
    reordered = torch.gather(x, dim=1, index=order_expanded)

    positions = torch.arange(N).unsqueeze(0).expand(B, N)
    inverse_order = torch.zeros_like(order)
    inverse_order.scatter_(dim=-1, index=order, src=positions)
    return reordered, inverse_order

def _run_checks() -> None:
    x = torch.tensor(
        [
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        ],
        requires_grad=True,
    )
    order = torch.tensor([[2, 0, 1], [1, 2, 0]])

    reordered, inverse_order = permute_and_invert(x, order)

    expected_reordered = torch.tensor(
        [
            [[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]],
            [[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]],
        ]
    )
    expected_inverse = torch.tensor([[1, 2, 0], [2, 0, 1]])

    torch.testing.assert_close(reordered, expected_reordered)
    torch.testing.assert_close(inverse_order, expected_inverse)
    assert reordered.shape == x.shape
    assert reordered.dtype == x.dtype and reordered.device == x.device
    assert inverse_order.dtype == torch.long and inverse_order.device == order.device

    restored = reordered.gather(
        dim=1,
        index=inverse_order.unsqueeze(-1).expand_as(reordered),
    )
    torch.testing.assert_close(restored, x)

    reordered.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))

    singleton_x = torch.tensor([[[4.0]]], dtype=torch.float64)
    singleton_order = torch.tensor([[0]])
    singleton_reordered, singleton_inverse = permute_and_invert(
        singleton_x,
        singleton_order,
    )
    torch.testing.assert_close(singleton_reordered, singleton_x)
    torch.testing.assert_close(singleton_inverse, singleton_order)

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
