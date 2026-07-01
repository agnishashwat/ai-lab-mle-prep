"""Day 2 -- token-level language modeling primitives, from scratch.

Implement each function WITHOUT calling F.cross_entropy / F.log_softmax / F.nll_loss.
You may use torch.exp, torch.log, .sum, .amax, .gather, indexing, etc.

Self-check (quick, no pytest needed):
    ./.venv/bin/python ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/token_lm.py

Full check (pytest):
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_token_lm.py -q

Shapes (2-D / "flattened LM" convention):
    logits:  (N, V) float  -- raw scores, N positions over a vocab of size V
    targets: (N,)   long   -- gold next-token id at each position
"""
from __future__ import annotations

import torch


# ----------------------------------------------------------------------------------
# 1. Numerically stable log-softmax over the last dim.
#    Derive it: log softmax(z)_i = z_i - logsumexp(z).
#    Use the max-subtraction trick so you never exp() a large positive number.
#    Return shape == logits.shape.
# ----------------------------------------------------------------------------------
def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    maxval = torch.amax(logits, dim=-1,keepdim=True)
    logits_small = logits - maxval
    return logits_small - torch.log(torch.exp(logits_small).sum(dim=-1, keepdim=True))


# ----------------------------------------------------------------------------------
# 2. Mean next-token cross-entropy. Must match F.cross_entropy(logits, targets) to ~1e-6.
#    Use YOUR log_softmax + gather. Average over the N positions.
# ----------------------------------------------------------------------------------
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_softmax_all = log_softmax(logits)
    log_softmax_target = log_softmax_all.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -log_softmax_target.mean()


# ----------------------------------------------------------------------------------
# 3. Per-token perplexity = exp(mean cross-entropy). Scalar.
#    Sanity: uniform logits over V => perplexity == V.
# ----------------------------------------------------------------------------------
def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, targets))


# ----------------------------------------------------------------------------------
# 4. The analytic gradient of the MEAN cross-entropy w.r.t. the logits.
#    Per position the gradient is (softmax(z) - onehot(y)); but cross_entropy
#    averages over N positions, so think carefully about the 1/N factor.
#    Return shape == logits.shape. This will be gradient-checked vs autograd.
# ----------------------------------------------------------------------------------
def ce_grad_analytic(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    onehot = torch.zeros_like(prob)
    onehot = onehot.scatter(dim=-1, index=targets.unsqueeze(-1), value=1)

    return (prob - onehot)/prob.shape[-2]



# ==================================================================================
# Quick self-check harness -- do not edit. (pytest version lives in test_token_lm.py)
# ==================================================================================
def _check(name: str, ok: bool) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main() -> None:
    import torch.nn.functional as F

    torch.manual_seed(0)
    results = []

    # 1. log_softmax
    try:
        z = torch.randn(4, 7)
        results.append(_check("log_softmax", torch.allclose(log_softmax(z), F.log_softmax(z, dim=-1), atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"log_softmax (raised: {e})", False))

    # 1b. log_softmax stays finite on huge logits
    try:
        z = torch.tensor([[1000.0, 1001.0, 999.0]])
        results.append(_check("log_softmax (stable on big logits)", torch.isfinite(log_softmax(z)).all().item()))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"log_softmax stable (raised: {e})", False))

    # 2. cross_entropy
    try:
        z = torch.randn(8, 10)
        y = torch.randint(0, 10, (8,))
        results.append(_check("cross_entropy", torch.allclose(cross_entropy(z, y), F.cross_entropy(z, y), atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"cross_entropy (raised: {e})", False))

    # 3. perplexity of uniform logits over V == V
    try:
        V = 17
        z = torch.zeros(5, V)
        y = torch.randint(0, V, (5,))
        results.append(_check("perplexity (uniform == V)", torch.allclose(perplexity(z, y), torch.tensor(float(V)), atol=1e-4)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"perplexity (raised: {e})", False))

    # 4. analytic gradient == autograd gradient
    try:
        z = torch.randn(6, 9, requires_grad=True)
        y = torch.randint(0, 9, (6,))
        cross_entropy(z, y).backward()
        analytic = ce_grad_analytic(z.detach(), y)
        results.append(_check("ce_grad_analytic == autograd", torch.allclose(z.grad, analytic, atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"ce_grad_analytic (raised: {e})", False))

    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
