"""Day 1 smoke test: confirm the env + autograd + one optimizer step work.

Run:
    python week-01-lm-foundations-pytorch/code/smoke_test.py
"""
import torch


def one_gd_step_toward_min(start: float = 0.0, lr: float = 0.1, steps: int = 50) -> float:
    """Minimize f(x) = (x - 3)^2 by gradient descent; the min is at x = 3."""
    x = torch.tensor(start, requires_grad=True)
    for _ in range(steps):
        loss = (x - 3.0) ** 2
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()
    return x.item()


def main() -> None:
    print("torch", torch.__version__)
    final = one_gd_step_toward_min()
    print(f"x after GD: {final:.4f} (target 3.0)")
    assert abs(final - 3.0) < 1e-2, "GD did not converge toward the minimum"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
