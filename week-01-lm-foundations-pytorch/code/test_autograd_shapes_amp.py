"""pytest checks for Day 3 autograd / shapes / mixed-precision drills.

Run:
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_autograd_shapes_amp.py -q
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from autograd_shapes_amp import (
    bf16_stable_softmax,
    fp16_overflow,
    linear_ce_grad_wrt_W,
    masked_max_logit_mean,
    masked_mean,
)


def test_masked_mean_basic():
    vals = torch.tensor([[1.0, 2.0, 3.0, 100.0], [4.0, 5.0, 100.0, 100.0]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).mean()
    assert torch.allclose(masked_mean(vals, mask), ref, atol=1e-6)


def test_masked_mean_all_false_no_nan():
    vals = torch.randn(3, 4)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    out = masked_mean(vals, mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)


def test_masked_max_logit_mean():
    torch.manual_seed(0)
    B, T, V = 2, 4, 5
    scores = torch.randn(B, T, V)
    bias = torch.randn(V)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    ref = (scores + bias).max(dim=-1).values[mask].mean()
    assert torch.allclose(masked_max_logit_mean(scores, bias, mask), ref, atol=1e-6)


def test_linear_ce_grad_matches_autograd():
    torch.manual_seed(2)
    N, D_in, H, V = 4, 6, 8, 5
    x = torch.randn(N, D_in)
    W = torch.randn(H, D_in, requires_grad=True)
    b = torch.randn(H, requires_grad=True)
    W2 = torch.randn(V, H, requires_grad=True)
    b2 = torch.randn(V, requires_grad=True)
    y = torch.randint(0, V, (N,))

    h = torch.relu(x @ W.T + b)
    z = h @ W2.T + b2
    F.cross_entropy(z, y).backward()

    got = linear_ce_grad_wrt_W(x, W.detach(), b.detach(), W2.detach(), b2.detach(), y)
    assert torch.allclose(got, W.grad, atol=1e-5)


def test_fp16_overflows():
    z = torch.tensor([100.0, 200.0, 300.0])
    assert torch.isinf(fp16_overflow(z)).any()


def test_bf16_stable_softmax_finite_and_normalized():
    z = torch.tensor([[100.0, 200.0, 300.0], [-50.0, 0.0, 50.0]])
    out = bf16_stable_softmax(z)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-2)
