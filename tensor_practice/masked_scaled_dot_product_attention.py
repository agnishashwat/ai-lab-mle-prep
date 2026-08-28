"""Medium PyTorch puzzle: masked scaled dot-product attention.

Run after implementing:
    .venv/bin/python tensor_practice/masked_scaled_dot_product_attention.py

Task
----
Implement single-head scaled dot-product attention with a key-padding mask.
Use the standard scale factor based on the query/key feature dimension, and
normalize attention independently for every query over the key positions.

``key_mask[b, j] == True`` means key/value position ``j`` is valid.
Invalid positions must receive exactly zero attention probability.

Function signature
------------------
def masked_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...

Example
-------
queries = tensor([[[1.],
                   [0.]]])

keys = tensor([[[0.],
                [0.693147],
                [100.]]])

values = tensor([[[ 3.,  0.],
                  [ 0.,  6.],
                  [99., 99.]]])

key_mask = tensor([[True, True, False]])

output, weights = masked_attention(queries, keys, values, key_mask)

output == tensor([[[1.0, 4.0],
                   [1.5, 3.0]]])

weights == tensor([[[1/3, 2/3, 0.0],
                    [1/2, 1/2, 0.0]]])

Constraints
-----------
- ``queries`` has shape (B, Tq, D).
- ``keys`` has shape (B, Tk, D).
- ``values`` has shape (B, Tk, Dv).
- All three have the same floating-point dtype and device.
- ``key_mask`` has shape (B, Tk), dtype torch.bool, and the same device.
- B, Tq, Tk, D, and Dv are at least 1.
- Every batch row contains at least one valid key.
- ``output`` has shape (B, Tq, Dv).
- ``weights`` has shape (B, Tq, Tk).
- Both outputs preserve the input dtype, device, and autograd connectivity.
- Attention weights sum to one across valid keys for every query.

Rules
-----
- Python loops are forbidden.
- The computation must remain finite for large-magnitude attention scores.
- Do not call a built-in scaled-dot-product-attention function.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import math

import torch


def masked_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum('bqd,btd->bqt', queries, keys)/queries.shape[-1]**0.5
    scores = torch.masked_fill(scores, mask=~key_mask.unsqueeze(1), value=-float('inf'))
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum('bqk,bkv->bqv', weights, values)
    return output, weights




def _run_checks() -> None:
    queries = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64, requires_grad=True)
    keys = torch.tensor(
        [[[0.0], [math.log(2.0)], [100.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = torch.tensor(
        [[[3.0, 0.0], [0.0, 6.0], [99.0, 99.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    key_mask = torch.tensor([[True, True, False]])

    output, weights = masked_attention(queries, keys, values, key_mask)

    expected_output = torch.tensor(
        [[[1.0, 4.0], [1.5, 3.0]]],
        dtype=torch.float64,
    )
    expected_weights = torch.tensor(
        [[[1.0 / 3.0, 2.0 / 3.0, 0.0], [0.5, 0.5, 0.0]]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(weights, expected_weights)
    assert output.shape == (1, 2, 2)
    assert weights.shape == (1, 2, 3)
    assert output.dtype == queries.dtype and output.device == queries.device
    assert weights.dtype == queries.dtype and weights.device == queries.device
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2, dtype=torch.float64))
    assert torch.count_nonzero(weights[..., 2]) == 0

    output.sum().backward()
    assert queries.grad is not None and queries.grad.shape == queries.shape
    assert keys.grad is not None and keys.grad.shape == keys.shape
    assert values.grad is not None and values.grad.shape == values.shape

    large_queries = torch.tensor([[[1000.0]]])
    large_keys = torch.tensor([[[1.0], [0.999]]])
    large_values = torch.tensor([[[1.0], [3.0]]])
    large_mask = torch.tensor([[True, True]])
    large_output, large_weights = masked_attention(
        large_queries,
        large_keys,
        large_values,
        large_mask,
    )
    assert torch.isfinite(large_output).all()
    assert torch.isfinite(large_weights).all()
    torch.testing.assert_close(
        large_weights,
        torch.tensor([[[0.7310586, 0.2689414]]]),
        rtol=1e-5,
        atol=1e-5,
    )

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
