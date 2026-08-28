"""Easy PyTorch tensor puzzle: batched vector updates.

Run after implementing:
    .venv/bin/python tensor_practice/batched_vector_updates.py

Task
----
Replace selected vectors in each batch row of ``x`` with vectors from
``updates``. Each batch row has its own replacement positions.

For every batch ``b`` and update ``m``:

    output[b, positions[b, m], :] = updates[b, m, :]

Vectors at all other positions must remain unchanged. Do not modify ``x``
in place.

Example
-------
x = tensor([[[10., 11.], [20., 21.], [30., 31.], [40., 41.]],
            [[ 1.,  2.], [ 3.,  4.], [ 5.,  6.], [ 7.,  8.]]])

positions = tensor([[1, 3],
                    [0, 2]])

updates = tensor([[[100., 101.], [300., 301.]],
                  [[ -1.,  -2.], [ -5.,  -6.]]])

batched_vector_updates(x, positions, updates)

# tensor([[[10., 11.], [100., 101.], [30., 31.], [300., 301.]],
#         [[-1., -2.], [  3.,   4.], [-5., -6.], [  7.,   8.]]])

Constraints
-----------
- ``x`` has shape (B, N, D) and floating-point dtype.
- ``positions`` has shape (B, M), dtype torch.long, and values in ``[0, N)``.
- ``updates`` has shape (B, M, D) and the same dtype as ``x``.
- All tensors are on the same CPU or CUDA device.
- B, N, M, and D are at least 1, with M <= N.
- Positions are unique within each batch row.
- The output must preserve x's shape, dtype, device, and autograd connectivity.
- Gradients must flow to unchanged values in ``x`` and inserted ``updates``.

Rules
-----
- Use ``torch.scatter`` or ``Tensor.scatter``.
- Python loops and advanced-index assignment are forbidden.
- Do not modify any input tensor in place.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def batched_vector_updates(
    x: torch.Tensor,
    positions: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """Return x with each batch row's selected vectors replaced."""
    B, N, D = x.shape
    M = positions.shape[-1]
    output = x.clone()
    output.scatter_(dim=1, index=positions.unsqueeze(-1).expand(B, M, D), src=updates)
    return output


def _run_checks() -> None:
    x = torch.tensor(
        [
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )
    positions = torch.tensor([[1, 3], [0, 2]])
    updates = torch.tensor(
        [
            [[100.0, 101.0], [300.0, 301.0]],
            [[-1.0, -2.0], [-5.0, -6.0]],
        ],
        requires_grad=True,
    )
    original_x = x.detach().clone()

    result = batched_vector_updates(x, positions, updates)
    expected = torch.tensor(
        [
            [[10.0, 11.0], [100.0, 101.0], [30.0, 31.0], [300.0, 301.0]],
            [[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0], [7.0, 8.0]],
        ]
    )

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(x, original_x)
    assert result.shape == x.shape
    assert result.dtype == x.dtype and result.device == x.device

    result.sum().backward()
    expected_x_grad = torch.tensor(
        [
            [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        ]
    )
    torch.testing.assert_close(x.grad, expected_x_grad)
    torch.testing.assert_close(updates.grad, torch.ones_like(updates))

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
