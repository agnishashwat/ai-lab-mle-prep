"""Easy PyTorch tensor puzzle: nearest prototype.

Run after implementing:
    .venv/bin/python tensor_practice/nearest_prototype.py

Task
----
Given a batch of query vectors and a set of prototype vectors, find the
prototype with the smallest squared Euclidean distance to each query.
Do not take the square root.

Return:
    distances: Shape (B,), containing each minimum squared distance.
    indices:   Shape (B,), containing the corresponding prototype index.

Example
-------
queries = tensor([[1., 2.],
                  [0., 0.],
                  [3., 3.]])

prototypes = tensor([[0., 0.],
                     [2., 2.],
                     [4., 1.]])

distances, indices = nearest_prototype(queries, prototypes)

distances == tensor([1., 0., 2.])
indices   == tensor([1, 0, 1])

Constraints
-----------
- ``queries`` has shape (B, D) and ``prototypes`` has shape (K, D).
- B, K, and D are at least 1.
- Both inputs have the same floating-point dtype and device.
- ``distances`` must preserve the input dtype, device, and autograd connectivity.
- ``indices`` must have dtype torch.long and the same device as the inputs.
- If multiple prototypes tie, return the smallest prototype index.

Rules
-----
- Python loops over queries or prototypes are forbidden.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def nearest_prototype(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the minimum squared distance and prototype index for each query."""
    distances =  ((queries.unsqueeze(1) - prototypes.unsqueeze(0))**2).sum(dim=-1)
    smallest_dist, smallest_idx = torch.min(distances, dim=-1)
    return smallest_dist, smallest_idx


def _run_checks() -> None:
    queries = torch.tensor(
        [[1.0, 2.0], [0.0, 0.0], [3.0, 3.0]],
        requires_grad=True,
    )
    prototypes = torch.tensor(
        [[0.0, 0.0], [2.0, 2.0], [4.0, 1.0]],
        requires_grad=True,
    )

    distances, indices = nearest_prototype(queries, prototypes)

    torch.testing.assert_close(distances, torch.tensor([1.0, 0.0, 2.0]))
    torch.testing.assert_close(indices, torch.tensor([1, 0, 1]))
    assert distances.shape == (3,)
    assert indices.shape == (3,)
    assert distances.dtype == queries.dtype and distances.device == queries.device
    assert indices.dtype == torch.long and indices.device == queries.device

    distances.sum().backward()
    assert queries.grad is not None and queries.grad.shape == queries.shape
    assert prototypes.grad is not None and prototypes.grad.shape == prototypes.shape

    tie_queries = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    tie_prototypes = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0]],
        dtype=torch.float64,
    )
    tie_distances, tie_indices = nearest_prototype(tie_queries, tie_prototypes)
    torch.testing.assert_close(tie_distances, torch.tensor([1.0], dtype=torch.float64))
    torch.testing.assert_close(tie_indices, torch.tensor([0]))

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
