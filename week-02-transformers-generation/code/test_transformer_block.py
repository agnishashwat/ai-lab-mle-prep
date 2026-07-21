"""pytest checks for Day 2 transformer block / TinyGPT.

Run:
    ./.venv/bin/pytest ai-lab-mle-prep/week-02-transformers-generation/code/test_transformer_block.py -q
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from transformer_block import (
    MLP,
    CausalSelfAttention,
    GPTConfig,
    RMSNorm,
    TinyGPT,
    TransformerBlock,
)


def _cfg():
    return GPTConfig(V=16, d=64, n_heads=4, n_layers=2, T=16)


def test_rmsnorm_unit_rms_and_shape():
    d = 64
    rms = RMSNorm(d)
    z = torch.randn(3, 5, d) * 7.0 + 2.0
    zn = rms(z)
    assert zn.shape == z.shape
    rms_val = zn.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-3)


def test_mlp_shape():
    mlp = MLP(64, mult=4)
    x = torch.randn(2, 16, 64)
    assert mlp(x).shape == x.shape


def test_block_preserves_shape():
    blk = TransformerBlock(64, 4, 4)
    x = torch.randn(2, 16, 64)
    assert blk(x).shape == x.shape


def test_gpt_logits_shape():
    cfg = _cfg()
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.V, (2, cfg.T))
    assert model(idx).shape == (2, cfg.T, cfg.V)


def test_causal_property():
    """Editing tokens strictly after position t must not change logits at <= t."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = TinyGPT(cfg).eval()
    idx = torch.randint(0, cfg.V, (2, cfg.T))
    t = cfg.T // 2
    with torch.no_grad():
        base = model(idx)
        idx2 = idx.clone()
        idx2[:, t + 1:] = (idx2[:, t + 1:] + 1) % cfg.V
        pert = model(idx2)
    assert torch.allclose(base[:, : t + 1], pert[:, : t + 1], atol=1e-5)
    assert not torch.allclose(base[:, t + 1:], pert[:, t + 1:], atol=1e-5)


def test_overfit_one_batch():
    cfg = _cfg()
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    gen = torch.Generator().manual_seed(0)
    start = torch.randint(0, cfg.V, (8, 1), generator=gen)
    x = (start + torch.arange(cfg.T).unsqueeze(0)) % cfg.V
    y = (x + 1) % cfg.V
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    last = None
    for _ in range(300):
        loss = F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = loss.item()
    assert last < 0.1
