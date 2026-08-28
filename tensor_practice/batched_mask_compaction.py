"""Medium PyTorch tensor puzzle: batched mask compaction.

Run after implementing:
    python tensor_practice/batched_mask_compaction.py

Task
----
For each row of ``x``, keep entries whose corresponding value in ``mask`` is
True, preserving their left-to-right order. Pad shorter rows to the largest
number of kept entries in any row.

Return:
    values:  Shape (B, M, D), padded with ``pad_value``.
    indices: Shape (B, M), containing source column indices and -1 for padding.
    lengths: Shape (B,), containing the number of kept entries in each row.

Here, ``M = mask.sum(dim=1).max()``. If the entire mask is False, ``M`` is zero.

Constraints
-----------
- ``x`` has shape (B, N, D), floating-point dtype, and B, N, D >= 1.
- ``mask`` has shape (B, N), dtype torch.bool, and is on the same device as x.
- ``values`` must preserve x's dtype, device, and autograd connectivity.
- ``indices`` and ``lengths`` must have dtype torch.long and x's device.
- Python loops over the batch or sequence dimensions are forbidden.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def compact_masked(
    x: torch.Tensor,
    mask: torch.Tensor,
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact masked entries in each batch row and return values, indices, lengths."""
    batch_size, _, feature_size = x.shape
    lengths = mask.sum(dim=1, dtype=torch.long)
    max_length = int(lengths.max().item())

    values = x.new_full((batch_size, max_length, feature_size), pad_value)
    indices = torch.full(
        (batch_size, max_length), -1, dtype=torch.long, device=x.device
    )

    batch_indices, source_indices = mask.nonzero(as_tuple=True)
    compact_indices = mask.long().cumsum(dim=1)[
        batch_indices, source_indices
    ] - 1

    values[batch_indices, compact_indices] = x[batch_indices, source_indices]
    indices[batch_indices, compact_indices] = source_indices
    return values, indices, lengths


def _run_checks() -> None:
    x = torch.tensor(
        [
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[50.0, 51.0], [60.0, 61.0], [70.0, 71.0], [80.0, 81.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [False, True, False, True],
            [True, False, False, False],
            [False, False, False, False],
        ]
    )

    values, indices, lengths = compact_masked(x, mask, pad_value=-9.0)

    expected_values = torch.tensor(
        [
            [[20.0, 21.0], [40.0, 41.0]],
            [[1.0, 2.0], [-9.0, -9.0]],
            [[-9.0, -9.0], [-9.0, -9.0]],
        ]
    )
    expected_indices = torch.tensor([[1, 3], [0, -1], [-1, -1]])
    expected_lengths = torch.tensor([2, 1, 0])

    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(lengths, expected_lengths)
    assert values.dtype == x.dtype and values.device == x.device
    assert indices.dtype == torch.long and indices.device == x.device
    assert lengths.dtype == torch.long and lengths.device == x.device

    values.sum().backward()
    expected_grad = mask.unsqueeze(-1).expand_as(x).to(x.dtype)
    torch.testing.assert_close(x.grad, expected_grad)

    all_masked_x = torch.randn(2, 3, 4)
    all_masked = torch.zeros(2, 3, dtype=torch.bool)
    empty_values, empty_indices, empty_lengths = compact_masked(
        all_masked_x, all_masked, pad_value=0.0
    )
    assert empty_values.shape == (2, 0, 4)
    assert empty_indices.shape == (2, 0)
    torch.testing.assert_close(empty_lengths, torch.zeros(2, dtype=torch.long))

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
