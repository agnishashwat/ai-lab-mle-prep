"""Easy PyTorch tensor puzzle: sliding-window range.

Run after implementing:
    .venv/bin/python tensor_practice/sliding_window_range.py

Task
----
For every contiguous window of width ``window_size`` in each row of ``x``,
compute the window's range:

    maximum value - minimum value

Return one result per valid window. Windows use zero-based, left-to-right
indexing and do not extend beyond either end of a row.

Example
-------
x = tensor([[ 1.,  4.,  2., 5., 3.],
            [ 0., -1., -1., 2., 2.]])

sliding_window_range(x, window_size=3)

# tensor([[3., 3., 3.],
#         [1., 3., 3.]])

Constraints
-----------
- ``x`` has shape (B, N), floating-point dtype, and may be on CPU or CUDA.
- B and N are at least 1.
- ``window_size`` is a Python integer satisfying 1 <= window_size <= N.
- The output has shape (B, N - window_size + 1).
- The output must preserve x's dtype, device, and autograd connectivity.
- If ``window_size == 1``, return zeros with shape (B, N).

Rules
-----
- Python loops over batches, positions, or windows are forbidden.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def sliding_window_range(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Return max-minus-min for every contiguous window in each row."""
    windows = x.unfold(dimension=-1, size=window_size, step=1)
    maxval = torch.max(windows, dim=-1)[0]
    minval = torch.min(windows, dim=-1)[0]
    return maxval - minval


def _run_checks() -> None:
    x = torch.tensor(
        [
            [1.0, 4.0, 2.0, 5.0, 3.0],
            [0.0, -1.0, -1.0, 2.0, 2.0],
        ],
        requires_grad=True,
    )

    result = sliding_window_range(x, window_size=3)
    expected = torch.tensor([[3.0, 3.0, 3.0], [1.0, 3.0, 3.0]])

    torch.testing.assert_close(result, expected)
    assert result.shape == (2, 3)
    assert result.dtype == x.dtype and result.device == x.device

    result.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape

    singleton_x = torch.tensor([[2.0, -3.0, 5.0]], dtype=torch.float64)
    singleton_result = sliding_window_range(singleton_x, window_size=1)
    torch.testing.assert_close(
        singleton_result,
        torch.zeros(1, 3, dtype=torch.float64),
    )

    full_width_x = torch.tensor([[3.0, -2.0, 7.0, 1.0]])
    full_width_result = sliding_window_range(full_width_x, window_size=4)
    torch.testing.assert_close(full_width_result, torch.tensor([[9.0]]))

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
