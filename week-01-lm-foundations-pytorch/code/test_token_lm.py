"""pytest checks for Day 2 token-LM primitives.

Run:
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_token_lm.py -q
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from token_lm import cross_entropy, ce_grad_analytic, log_softmax, perplexity


def test_log_softmax_matches_torch():
    torch.manual_seed(0)
    z = torch.randn(4, 7)
    assert torch.allclose(log_softmax(z), F.log_softmax(z, dim=-1), atol=1e-6)


def test_log_softmax_stable_on_large_logits():
    z = torch.tensor([[1000.0, 1001.0, 999.0]])
    out = log_softmax(z)
    assert torch.isfinite(out).all()
    # rows of exp(log_softmax) must sum to 1
    assert torch.allclose(out.exp().sum(-1), torch.ones(1), atol=1e-6)


def test_cross_entropy_matches_torch():
    torch.manual_seed(1)
    z = torch.randn(8, 10)
    y = torch.randint(0, 10, (8,))
    assert torch.allclose(cross_entropy(z, y), F.cross_entropy(z, y), atol=1e-6)


def test_perplexity_uniform_equals_vocab():
    V = 17
    z = torch.zeros(5, V)
    y = torch.randint(0, V, (5,))
    assert torch.allclose(perplexity(z, y), torch.tensor(float(V)), atol=1e-4)


def test_ce_grad_matches_autograd():
    torch.manual_seed(2)
    z = torch.randn(6, 9, requires_grad=True)
    y = torch.randint(0, 9, (6,))
    cross_entropy(z, y).backward()
    analytic = ce_grad_analytic(z.detach(), y)
    assert torch.allclose(z.grad, analytic, atol=1e-6)


def test_ce_grad_matches_finite_difference():
    """Independent check: no autograd, just perturb one logit."""
    torch.manual_seed(3)
    N, V = 3, 5
    z = torch.randn(N, V, dtype=torch.float64)
    y = torch.randint(0, V, (N,))
    analytic = ce_grad_analytic(z, y)
    eps = 1e-6
    num = torch.zeros_like(z)
    for i in range(N):
        for j in range(V):
            zp = z.clone(); zp[i, j] += eps
            zm = z.clone(); zm[i, j] -= eps
            num[i, j] = (cross_entropy(zp, y) - cross_entropy(zm, y)) / (2 * eps)
    assert torch.allclose(analytic, num, atol=1e-5)
