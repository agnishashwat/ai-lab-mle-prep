"""Easy PyTorch tensor puzzle: target scores and one-hot labels.

Run after implementing:
    .venv/bin/python tensor_practice/target_scores_and_one_hot.py

Task
----
Given class scores and one target class per position:

1. Read the score of the target class at every position.
2. Build a one-hot tensor marking each target class.

Example
-------
scores = tensor([[[ 0.1, 0.7, 0.2],
                  [ 2.0, -1.0, 0.0]],
                 [[ 0.5, 0.4, 0.3],
                  [-2.0, 3.0, 1.0]]])

targets = tensor([[1, 0],
                  [2, 1]])

target_scores, one_hot = target_scores_and_one_hot(scores, targets)

target_scores == tensor([[0.7, 2.0],
                         [0.3, 3.0]])

one_hot == tensor([[[0., 1., 0.],
                    [1., 0., 0.]],
                   [[0., 0., 1.],
                    [0., 1., 0.]]])

Constraints
-----------
- ``scores`` has shape (B, N, C) and floating-point dtype.
- ``targets`` has shape (B, N), dtype torch.long, and values in ``[0, C)``.
- Both tensors are on the same CPU or CUDA device.
- B, N, and C are at least 1.
- ``target_scores`` has shape (B, N) and preserves scores' dtype, device, and
  autograd connectivity.
- ``one_hot`` has shape (B, N, C) and preserves scores' dtype and device.
- Every vector ``one_hot[b, n]`` contains exactly one value equal to 1.

Rules
-----
- Use ``torch.gather`` or ``Tensor.gather`` to create ``target_scores``.
- Use ``torch.scatter`` or ``Tensor.scatter`` to create ``one_hot``.
- Do not use ``torch.nn.functional.one_hot``.
- Python loops and advanced indexing are forbidden.
- Do not modify either input in place.
"""

from __future__ import annotations

import torch


def target_scores_and_one_hot(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-class scores and one-hot target vectors."""
    target_scores = torch.gather(input=scores,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    one_hot = torch.zeros_like(scores)
    one_hot.scatter_(dim=-1, index=targets.unsqueeze(-1), src=torch.ones_like(scores))
    return target_scores, one_hot

def _run_checks() -> None:
    scores = torch.tensor(
        [
            [[0.1, 0.7, 0.2], [2.0, -1.0, 0.0]],
            [[0.5, 0.4, 0.3], [-2.0, 3.0, 1.0]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[1, 0], [2, 1]])
    original_scores = scores.detach().clone()
    original_targets = targets.clone()

    target_scores, one_hot = target_scores_and_one_hot(scores, targets)

    expected_scores = torch.tensor([[0.7, 2.0], [0.3, 3.0]])
    expected_one_hot = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ]
    )

    torch.testing.assert_close(target_scores, expected_scores)
    torch.testing.assert_close(one_hot, expected_one_hot)
    torch.testing.assert_close(scores, original_scores)
    torch.testing.assert_close(targets, original_targets)
    assert target_scores.shape == targets.shape
    assert one_hot.shape == scores.shape
    assert target_scores.dtype == scores.dtype
    assert one_hot.dtype == scores.dtype
    assert target_scores.device == scores.device
    assert one_hot.device == scores.device

    target_scores.sum().backward()
    torch.testing.assert_close(scores.grad, expected_one_hot)

    singleton_scores = torch.tensor([[[4.0]]], dtype=torch.float64)
    singleton_targets = torch.tensor([[0]])
    singleton_selected, singleton_one_hot = target_scores_and_one_hot(
        singleton_scores,
        singleton_targets,
    )
    torch.testing.assert_close(singleton_selected, torch.tensor([[4.0]], dtype=torch.float64))
    torch.testing.assert_close(singleton_one_hot, torch.ones_like(singleton_scores))

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
