"""Easy PyTorch tensor puzzle: weighted class totals.

Run after implementing:
    .venv/bin/python tensor_practice/weighted_class_totals.py

Task
----
For each batch row, group elements by their integer class ID and sum their
weights. Return one total for every class, including classes that do not appear.

Example
-------
class_ids = tensor([[0, 2, 2, 1],
                    [1, 1, 3, 0]])

weights = tensor([[ 1.0, 0.5, 2.0, -1.0],
                  [ 3.0, 4.0, 2.0,  0.5]])

weighted_class_totals(class_ids, weights, num_classes=4)

# tensor([[ 1.0, -1.0, 2.5, 0.0],
#         [ 0.5,  7.0, 0.0, 2.0]])

Constraints
-----------
- ``class_ids`` has shape (B, N), dtype torch.long, and values in
  ``[0, num_classes)``.
- ``weights`` has shape (B, N), floating-point dtype, and the same device as
  ``class_ids``.
- B, N, and ``num_classes`` are at least 1.
- The output has shape (B, num_classes).
- Missing classes must have total zero.
- Repeated class IDs must have all their weights accumulated.
- The output must preserve weights' dtype, device, and autograd connectivity.

Rules
-----
- Python loops over batches, elements, or classes are forbidden.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def weighted_class_totals(
    class_ids: torch.Tensor,
    weights: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Sum weights by class independently for each batch row."""
    B, N = class_ids.shape 
    class_total = torch.zeros(size=(B, num_classes), dtype=weights.dtype)
    class_total.scatter_add_(dim=1, index=class_ids, src=weights)
    return class_total



def _run_checks() -> None:
    class_ids = torch.tensor([[0, 2, 2, 1], [1, 1, 3, 0]])
    weights = torch.tensor(
        [[1.0, 0.5, 2.0, -1.0], [3.0, 4.0, 2.0, 0.5]],
        requires_grad=True,
    )

    totals = weighted_class_totals(class_ids, weights, num_classes=4)
    expected = torch.tensor([[1.0, -1.0, 2.5, 0.0], [0.5, 7.0, 0.0, 2.0]])

    torch.testing.assert_close(totals, expected)
    assert totals.shape == (2, 4)
    assert totals.dtype == weights.dtype and totals.device == weights.device

    totals.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))

    repeated_ids = torch.tensor([[2, 2, 2]])
    repeated_weights = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    repeated_totals = weighted_class_totals(
        repeated_ids,
        repeated_weights,
        num_classes=4,
    )
    torch.testing.assert_close(
        repeated_totals,
        torch.tensor([[0.0, 0.0, -0.5, 0.0]], dtype=torch.float64),
    )

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
