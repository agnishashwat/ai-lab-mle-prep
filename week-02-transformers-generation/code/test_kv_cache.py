"""pytest for Week 02 Day 04 -- KV cache + inference memory math.

    ./.venv/bin/pytest ai-lab-mle-prep/week-02-transformers-generation/code/test_kv_cache.py -q
"""
from __future__ import annotations

import torch

from transformer_block import GPTConfig, TinyGPT
from decoding import generate
from kv_cache import gpt_forward_cached, generate_cached, kv_cache_bytes


def _model(seed: int = 0):
    cfg = GPTConfig(V=32, d=128, n_heads=4, n_layers=3, T=64)
    torch.manual_seed(seed)
    return TinyGPT(cfg).eval(), cfg


def test_kv_cache_bytes_formula():
    cfg = GPTConfig(V=32000, d=4096, n_heads=32, n_layers=32, T=4096)
    assert kv_cache_bytes(cfg, seq_len=4096, batch=1, dtype_bytes=2) == 2 * 1024**3
    assert kv_cache_bytes(cfg, seq_len=1, batch=1, dtype_bytes=2) == 512 * 1024


def test_kv_cache_bytes_linear_and_gqa():
    cfg = GPTConfig(V=32000, d=4096, n_heads=32, n_layers=32, T=4096)
    base = kv_cache_bytes(cfg, seq_len=1024, batch=1)
    assert kv_cache_bytes(cfg, seq_len=2048, batch=1) == 2 * base          # linear in seq
    assert kv_cache_bytes(cfg, seq_len=1024, batch=4) == 4 * base          # linear in batch
    assert kv_cache_bytes(cfg, seq_len=1024, batch=1, n_kv_heads=8) * 4 == base  # GQA 4x


def test_prefill_matches_full_forward():
    model, cfg = _model()
    prompt = torch.randint(0, cfg.V, (2, 8))
    with torch.no_grad():
        full = model(prompt)
        pre, caches = gpt_forward_cached(model, prompt, None, pos_offset=0)
    assert torch.allclose(full, pre, atol=1e-4)
    assert len(caches) == cfg.n_layers
    assert tuple(caches[0][0].shape) == (2, cfg.n_heads, 8, cfg.d // cfg.n_heads)


def test_incremental_matches_full_and_grows():
    model, cfg = _model()
    idx = torch.randint(0, cfg.V, (2, 8))
    with torch.no_grad():
        logits, caches = gpt_forward_cached(model, idx, None, pos_offset=0)
        for _ in range(6):
            nt = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nt], dim=-1)
            logits, caches = gpt_forward_cached(model, nt, caches, pos_offset=idx.shape[1] - 1)
            assert caches[0][0].shape[2] == idx.shape[1]              # grew by 1
            assert torch.allclose(model(idx)[:, -1], logits[:, -1], atol=1e-4)


def test_cached_greedy_equals_uncached():
    model, cfg = _model()
    prompt = torch.randint(0, cfg.V, (2, 8))
    a = generate(model, prompt, max_new_tokens=20, greedy=True)
    b = generate_cached(model, prompt, max_new_tokens=20, greedy=True)
    assert torch.equal(a, b)


def test_cached_sampling_reproducible():
    model, cfg = _model()
    prompt = torch.randint(0, cfg.V, (2, 8))
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    s1 = generate_cached(model, prompt, 10, temperature=1.0, top_k=5, generator=g1)
    s2 = generate_cached(model, prompt, 10, temperature=1.0, top_k=5, generator=g2)
    assert torch.equal(s1, s2)
