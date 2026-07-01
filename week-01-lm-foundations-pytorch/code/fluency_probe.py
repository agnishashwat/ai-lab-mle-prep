"""Day 1 fluency probe -- YOUR baseline.

Implement each function FROM MEMORY, no references. Time yourself per function and
record the wall-clock in `deliverable.md`. These are the primitives you should be able
to write in your sleep by interview time.

Self-check:
    python week-01-lm-foundations-pytorch/code/fluency_probe.py

Every check compares your implementation against a torch reference. Aim: all PASS,
under ~10 minutes total, no lookups.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# 1. Numerically stable softmax over the last dim.
#    Hint: subtract the row-max before exp.
# ----------------------------------------------------------------------------------
def softmax(logits: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


# ----------------------------------------------------------------------------------
# 2. Mean next-token cross-entropy.
#    logits: (N, V) float, targets: (N,) long. Match F.cross_entropy.
#    Do it via log-softmax + gather (do NOT call F.cross_entropy).
# ----------------------------------------------------------------------------------
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


# ----------------------------------------------------------------------------------
# 3. Masked mean of `values` over positions where mask == 1.
#    values: (B, T) float, mask: (B, T) in {0,1}. Return a scalar: the mean over
#    UNMASKED entries only (correct denominator = number of 1s).
# ----------------------------------------------------------------------------------
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


# ----------------------------------------------------------------------------------
# 4. One manual SGD step (no torch.optim). Update w in-place: w <- w - lr * w.grad.
#    Assume w.grad is already populated. Return the updated tensor.
# ----------------------------------------------------------------------------------
def sgd_step(w: torch.Tensor, lr: float) -> torch.Tensor:
    raise NotImplementedError


# ==================================================================================
# Self-check harness -- do not edit.
# ==================================================================================
def _check(name: str, ok: bool) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main() -> None:
    torch.manual_seed(0)
    results = []

    try:
        z = torch.randn(4, 7)
        results.append(_check("softmax", torch.allclose(softmax(z), F.softmax(z, dim=-1), atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"softmax (raised: {e})", False))

    try:
        z = torch.randn(8, 10)
        y = torch.randint(0, 10, (8,))
        results.append(_check("cross_entropy", torch.allclose(cross_entropy(z, y), F.cross_entropy(z, y), atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"cross_entropy (raised: {e})", False))

    try:
        v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        expected = torch.tensor((1.0 + 2.0 + 4.0) / 3.0)
        results.append(_check("masked_mean", torch.allclose(masked_mean(v, m), expected, atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"masked_mean (raised: {e})", False))

    try:
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        (w.sum() * 2).backward()  # grad = [2, 2]
        out = sgd_step(w, lr=0.5)  # expect [0.0, 1.0]
        results.append(_check("sgd_step", torch.allclose(out.detach(), torch.tensor([0.0, 1.0]), atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"sgd_step (raised: {e})", False))

    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
