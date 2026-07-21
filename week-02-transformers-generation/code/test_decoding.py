"""pytest checks for Day 3 decoding strategies.

Run:
    ./.venv/bin/pytest ai-lab-mle-prep/week-02-transformers-generation/code/test_decoding.py -q
"""
from __future__ import annotations

import torch

from decoding import (
    apply_temperature,
    beam_search,
    generate,
    sample,
    top_k_filter,
    top_p_filter,
    _train_successor,
)


def _entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-12)).sum(-1).mean())


def _finite_count(logits: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(logits).sum(-1)


def _peaked():
    return torch.log(torch.tensor([[0.90, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002]]))


def _flat():
    return torch.log(torch.tensor([[0.25, 0.22, 0.20, 0.18, 0.07, 0.04, 0.02, 0.02]]))


def test_temperature_limits():
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    assert torch.equal(apply_temperature(z, 0.01).argmax(-1), z.argmax(-1))
    assert _entropy(apply_temperature(z, 100.0)) > _entropy(z) > _entropy(apply_temperature(z, 0.01))


def test_top_k_keeps_k_and_normalizes():
    fk = top_k_filter(_peaked(), 3)
    assert torch.equal(_finite_count(fk), torch.tensor([3]))
    assert torch.allclose(torch.softmax(fk, -1).sum(-1), torch.ones(1), atol=1e-6)


def test_top_k_none_is_noop():
    z = _flat()
    assert torch.equal(top_k_filter(z, None), z)


def test_top_p_adaptive():
    n_peaked = int(_finite_count(top_p_filter(_peaked(), 0.9)))
    n_flat = int(_finite_count(top_p_filter(_flat(), 0.9)))
    assert n_peaked < n_flat


def test_top_p_nucleus_covers_p_and_normalizes():
    p = 0.9
    filt = top_p_filter(_peaked(), p)
    n = int(_finite_count(filt))
    assert float(torch.softmax(_peaked(), -1)[0, :n].sum()) >= p - 1e-6
    assert torch.allclose(torch.softmax(filt, -1).sum(-1), torch.ones(1), atol=1e-6)


def test_sample_reproducible():
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    x = _flat().repeat(50, 1)
    assert torch.equal(sample(x, g1), sample(x, g2))


def test_greedy_deterministic_and_successor():
    model, cfg = _train_successor(0)
    prompt = torch.tensor([[3, 4, 5]])
    a = generate(model, prompt, 5, greedy=True)
    b = generate(model, prompt, 5, greedy=True)
    assert torch.equal(a, b)
    assert a[0, prompt.shape[1]:].tolist() == [(5 + i + 1) % cfg.V for i in range(5)]


def test_seeded_sampling_generate_reproducible():
    model, _ = _train_successor(0)
    prompt = torch.tensor([[3, 4, 5]])
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = generate(model, prompt, 5, temperature=1.0, top_k=5, generator=g1)
    b = generate(model, prompt, 5, temperature=1.0, top_k=5, generator=g2)
    assert torch.equal(a, b)


def test_beam_shape_and_successor():
    model, cfg = _train_successor(0)
    prompt = torch.tensor([[3, 4, 5]])
    beam = beam_search(model, prompt, beam_width=3, max_new_tokens=5, length_penalty=1.0)
    assert beam.shape == (1, prompt.shape[1] + 5)
    assert beam[0, prompt.shape[1]:].tolist() == [(5 + i + 1) % cfg.V for i in range(5)]
