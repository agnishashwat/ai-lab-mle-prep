"""Medium PyTorch puzzle: top-k mixture-of-experts routing.

Run after implementing:
    .venv/bin/python tensor_practice/topk_expert_routing.py

Task
----
A mixture-of-experts router selects ``K`` experts for every token. Combine the
selected expert outputs using their routing weights, and count how many tokens
were assigned to each expert.

``expert_outputs[t, e]`` contains expert ``e``'s output for token ``t``.
``expert_indices[t]`` identifies the selected experts for that token, in routing
rank order. Expert load counts assignments regardless of routing weight.

Return:

- ``combined``: the weighted expert output for every token.
- ``expert_load``: the number of assignments received by every expert.

Example
-------
expert_outputs = tensor([
    [[ 1.,  0.], [ 0.,  2.], [ 3.,  3.]],
    [[10.,  0.], [ 0., 20.], [30., 30.]],
])

expert_indices = tensor([[2, 0],
                         [1, 2]])

routing_weights = tensor([[0.25, 0.75],
                          [0.60, 0.40]])

combined, expert_load = combine_topk_experts(
    expert_outputs,
    expert_indices,
    routing_weights,
)

combined == tensor([[ 1.5,  0.75],
                    [12.0, 24.00]])

expert_load == tensor([1, 1, 2])

Constraints
-----------
- ``expert_outputs`` has shape (T, E, D) and floating-point dtype.
- ``expert_indices`` has shape (T, K), dtype torch.long, and values in ``[0, E)``.
- ``routing_weights`` has shape (T, K) and the same dtype as ``expert_outputs``.
- All tensors are on the same CPU or CUDA device.
- T, E, D, and K are at least 1, with K <= E.
- Selected expert indices are unique within each token.
- Routing weights are nonnegative and sum to one for each token.
- ``combined`` has shape (T, D) and preserves floating-point dtype, device, and
  autograd connectivity.
- ``expert_load`` has shape (E,), dtype torch.long, and the input device.
- Experts receiving no assignments must have load zero.

Rules
-----
- Python loops over tokens, experts, or routing choices are forbidden.
- Do not modify any input tensor in place.
- Use only PyTorch; do not use NumPy or third-party packages.
"""

from __future__ import annotations

import torch


def combine_topk_experts(
    expert_outputs: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine selected expert outputs and return per-expert assignment counts."""
    T, E, D = expert_outputs.shape
    K = expert_indices.shape[-1]
    outputs_selected = torch.gather(input=expert_outputs, dim=1, index=expert_indices.unsqueeze(-1).expand(T, K, D))
    combined = torch.einsum('tkd,tk->td', outputs_selected, routing_weights)
    expert_load_per_token = torch.zeros(T, E, dtype=torch.long, device=expert_outputs.device)
    expert_load_per_token.scatter_(1,expert_indices,1.0)
    expert_load = expert_load_per_token.sum(dim=0)
    return combined, expert_load

def _run_checks() -> None:
    expert_outputs = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]],
            [[10.0, 0.0], [0.0, 20.0], [30.0, 30.0]],
        ],
        requires_grad=True,
    )
    expert_indices = torch.tensor([[2, 0], [1, 2]])
    routing_weights = torch.tensor(
        [[0.25, 0.75], [0.60, 0.40]],
        requires_grad=True,
    )

    combined, expert_load = combine_topk_experts(
        expert_outputs,
        expert_indices,
        routing_weights,
    )

    torch.testing.assert_close(
        combined,
        torch.tensor([[1.5, 0.75], [12.0, 24.0]]),
    )
    torch.testing.assert_close(expert_load, torch.tensor([1, 1, 2]))
    assert combined.shape == (2, 2)
    assert expert_load.shape == (3,)
    assert combined.dtype == expert_outputs.dtype
    assert combined.device == expert_outputs.device
    assert expert_load.dtype == torch.long
    assert expert_load.device == expert_outputs.device

    combined.sum().backward()
    expected_expert_grad = torch.tensor(
        [
            [[0.75, 0.75], [0.0, 0.0], [0.25, 0.25]],
            [[0.0, 0.0], [0.60, 0.60], [0.40, 0.40]],
        ]
    )
    expected_weight_grad = torch.tensor([[6.0, 1.0], [20.0, 60.0]])
    torch.testing.assert_close(expert_outputs.grad, expected_expert_grad)
    torch.testing.assert_close(routing_weights.grad, expected_weight_grad)

    single_outputs = torch.tensor([[[2.0], [5.0], [8.0]]], dtype=torch.float64)
    single_indices = torch.tensor([[1]])
    single_weights = torch.tensor([[1.0]], dtype=torch.float64)
    single_combined, single_load = combine_topk_experts(
        single_outputs,
        single_indices,
        single_weights,
    )
    torch.testing.assert_close(single_combined, torch.tensor([[5.0]], dtype=torch.float64))
    torch.testing.assert_close(single_load, torch.tensor([0, 1, 0]))

    print("All checks passed.")


if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError as error:
        print(f"TODO: {error}")
