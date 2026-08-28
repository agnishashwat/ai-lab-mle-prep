"""Medium PyTorch puzzle: masked token cross-entropy.

Run after implementing:
    .venv/bin/python tensor_practice/masked_token_cross_entropy.py

Task
----
Language-model batches often contain padded tokens that must not affect the
training loss. Compute the negative log-probability of the target vocabulary
class at every valid position.

Return both:

- ``loss``: the mean over valid positions only.
- ``token_losses``: one value per position, with exactly zero at masked positions.

Example
-------
logits = tensor([[[0.000000, 0.000000],
                  [1.098612, 0.000000],
                  [0.000000, 1.098612]]])

targets = tensor([[0, 0, 0]])
loss_mask = tensor([[True, True, False]])

loss, token_losses = masked_token_cross_entropy(logits, targets, loss_mask)

loss == tensor(0.490415)
token_losses == tensor([[0.693147, 0.287682, 0.000000]])

Constraints
-----------
- ``logits`` has shape (B, T, V) and floating-point dtype.
- ``targets`` has shape (B, T), dtype torch.long, and values in ``[0, V)``.
- ``loss_mask`` has shape (B, T) and dtype torch.bool.
- All tensors are on the same CPU or CUDA device.
- B, T, and V are at least 1, and ``loss_mask`` contains at least one True.
- ``loss`` is a scalar tensor.
- ``token_losses`` has shape (B, T).
- Both outputs preserve logits' dtype, device, and autograd connectivity.
- The mean is taken across all valid tokens in the batch, not per sequence.

Rules
-----
- Python loops are forbidden.
- Do not call built-in cross-entropy or negative-log-likelihood loss functions.
- The result must remain finite for large finite logits.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import math

import torch


def masked_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the masked mean loss and per-token losses."""

    logprob_all = torch.log_softmax(logits, dim=-1)
    logprob_targets = torch.gather(logprob_all, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    masked_logprob_targets = logprob_targets * loss_mask
    mean_loss = -masked_logprob_targets.sum()/loss_mask.sum()
    return mean_loss, -masked_logprob_targets




def _run_checks() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0],
                [math.log(3.0), 0.0],
                [0.0, math.log(3.0)],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([[0, 0, 0]])
    loss_mask = torch.tensor([[True, True, False]])

    loss, token_losses = masked_token_cross_entropy(logits, targets, loss_mask)

    expected_token_losses = torch.tensor(
        [[math.log(2.0), math.log(4.0 / 3.0), 0.0]],
        dtype=torch.float64,
    )
    expected_loss = expected_token_losses.sum() / 2

    torch.testing.assert_close(token_losses, expected_token_losses)
    torch.testing.assert_close(loss, expected_loss)
    assert loss.shape == ()
    assert token_losses.shape == targets.shape
    assert loss.dtype == logits.dtype and loss.device == logits.device
    assert token_losses.dtype == logits.dtype
    assert token_losses.device == logits.device

    loss.backward()
    assert logits.grad is not None and logits.grad.shape == logits.shape
    torch.testing.assert_close(
        logits.grad[~loss_mask],
        torch.zeros_like(logits.grad[~loss_mask]),
    )

    large_logits = torch.tensor(
        [[[10_000.0, -10_000.0, 0.0], [-10_000.0, 10_000.0, 0.0]]]
    )
    large_targets = torch.tensor([[0, 1]])
    large_mask = torch.tensor([[True, True]])
    large_loss, large_token_losses = masked_token_cross_entropy(
        large_logits,
        large_targets,
        large_mask,
    )
    assert torch.isfinite(large_loss)
    assert torch.isfinite(large_token_losses).all()
    torch.testing.assert_close(large_loss, torch.tensor(0.0))

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
