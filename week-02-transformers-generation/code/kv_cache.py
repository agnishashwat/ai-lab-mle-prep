"""Week 02 Day 04 -- KV cache, incremental decode & inference memory math.

The Day-3 `generate()` reruns the WHOLE prefix through the model every step:
at context length T that's O(T^2) attention per token -> O(N^3) to emit N tokens,
and it throws away every logit row except the last. But K and V for past tokens
NEVER change (causal), so we cache them. Each new step then computes q,k,v for the
ONE new token, attends to the cached K/V, and appends its own k,v. Attention per
step drops O(T^2) -> O(T); total O(N^3) -> O(N^2).

Two phases:
    prefill : run the whole prompt in one parallel pass, fill the cache (compute-bound)
    decode  : feed one token at a time, reuse the cache      (memory-bandwidth-bound)

You reuse the EXACT weights of your Day-2 TinyGPT (so we can prove cached output is
token-identical to the uncached Day-3 generate). Your job is the five TODOs: a
cache-aware attention, block, and model forward; the prefill+decode `generate_cached`;
and the `kv_cache_bytes` memory calculator.

    ./.venv/bin/python ai-lab-mle-prep/week-02-transformers-generation/code/kv_cache.py
    ./.venv/bin/pytest ai-lab-mle-prep/week-02-transformers-generation/code/test_kv_cache.py -q

Shapes:  B batch   S new-tokens-this-step   P past/cached len   V vocab
         d model dim   H heads   d_k = d // H
Cache layout: a python list over layers; entry ell is a (k, v) tuple, each
              (B, H, P, d_k), or None before anything is cached.
"""
from __future__ import annotations

import math
import time

import torch

from transformer_block import GPTConfig, TinyGPT
from decoding import apply_temperature, top_k_filter, top_p_filter, sample, generate


# ==================================================================================
# TODO 1 -- cache-aware attention for ONE sublayer.
#
#   attn : a CausalSelfAttention module (has .Wq/.Wk/.Wv/.Wo, .H, .d_k, ._split, ._merge)
#   x        : (B, S, d)   the NEW tokens' hidden states (S == prompt len at prefill, 1 at decode)
#   past_kv  : (past_k, past_v) each (B, H, P, d_k), or None on the first call
#   pos_offset: absolute position of x[:, 0]  (0 at prefill; current length at decode)
#   Return (out, new_kv) where out is (B, S, d) and new_kv = (k_all, v_all), each
#          (B, H, P+S, d_k) -- i.e. the cache to store for next step.
#
#   Steps:
#     1. q,k,v = attn._split(attn.Wq/Wk/Wv(x))          # each (B,H,S,d_k)
#     2. if past_kv is not None: cat the PAST k,v with the new k,v along the TIME dim (=2)
#        -> k_all, v_all are (B,H,P+S,d_k). (q is NOT concatenated -- Q is never cached.)
#     3. scores = q @ k_all.transpose(-1,-2) / sqrt(d_k)  # (B,H,S,P+S)
#     4. CAUSAL MASK with offset: query row i is absolute position pos_offset+i; key col j
#        is absolute position j. Keep iff  j <= pos_offset + i. Build a (S, P+S) bool mask
#        from arange's and masked_fill(~mask, -inf). (Decode: S=1 -> all True, no masking.
#        Prefill: P=0, pos_offset=0 -> standard lower-triangular. ONE formula covers both.)
#     5. out = attn.Wo(attn._merge(softmax(scores,-1) @ v_all))
#   Return (out, (k_all, v_all)).
# ==================================================================================
def create_causal_mask(pos_offset, T, S, device):
    mask = torch.ones(T, S, dtype=torch.bool)
    return torch.tril(mask, diagonal=pos_offset)
     



def sdpa_cached(attn, x: torch.Tensor, past_kv, pos_offset: int):
    q, k, v = attn._split(attn.Wq(x)), attn._split(attn.Wk(x)), attn._split(attn.Wv(x))
    if past_kv is not None:
        past_k, past_v = past_kv[0], past_kv[1]
        k_all, v_all = torch.cat([past_k, k], dim=-2), torch.cat([past_v, v], dim=-2)
    else:
        k_all, v_all = k, v
    
    scores = torch.einsum('...id,...jd->...ij', q, k_all)/math.sqrt(q.shape[-1])
    causal_mask = create_causal_mask(pos_offset, scores.shape[-2], scores.shape[-1], x.device)
    scores = scores.masked_fill(~causal_mask, -math.inf)
    out = attn.Wo(attn._merge(torch.softmax(scores, dim=-1) @ v_all))
    return (out, (k_all, v_all)) 


# ==================================================================================
# TODO 2 -- cache-aware PRE-LN block forward.
#   block : a TransformerBlock (has .norm1, .attn, .norm2, .mlp)
#   Mirror the Day-2 wiring but route attention through sdpa_cached and thread the cache:
#       a, new_kv = sdpa_cached(block.attn, block.norm1(x), past_kv, pos_offset)
#       x = x + a
#       x = x + block.mlp(block.norm2(x))
#   Return (x, new_kv).  (RMSNorm and MLP are position-wise -> no cache needed there.)
# ==================================================================================
def block_forward_cached(block, x: torch.Tensor, past_kv, pos_offset: int):
    f, new_kv = sdpa_cached(block.attn, block.norm1(x), past_kv, pos_offset)
    x = x + f
    x = x + block.mlp(block.norm2(x))
    return (x, new_kv)


# ==================================================================================
# TODO 3 -- cache-aware TinyGPT forward.
#   model    : a TinyGPT (has .tok_emb, .pos_emb, .blocks, .norm_f, .head, .cfg)
#   idx      : (B, S) long -- the NEW tokens (whole prompt at prefill, 1 token at decode)
#   past_kvs : list of length n_layers of (k,v) tuples, or None on the first call
#   pos_offset: absolute position of idx[:, 0]
#   Return (logits, new_kvs) where logits is (B, S, V) and new_kvs is the updated list.
#
#   CRITICAL BUG TO AVOID: TinyGPT uses a LEARNED absolute positional table. The new
#   tokens are at absolute positions pos_offset .. pos_offset+S-1, so:
#       positions = pos_offset + torch.arange(S)
#   NOT arange(S). If you re-encode position 0 for every decoded token, generation is wrong.
#
#   Steps: h = tok_emb(idx) + pos_emb(positions); loop blocks threading each layer's cache
#   into a new list; h = head(norm_f(h)); return (logits, new_kvs).
# ==================================================================================
def gpt_forward_cached(model: TinyGPT, idx: torch.Tensor, past_kvs, pos_offset: int):
    B, S = idx.shape[0], idx.shape[1]
    positions = pos_offset + torch.arange(S)
    h = model.tok_emb(idx) + model.pos_emb(positions)
    new_kvs = []

    for i, block in enumerate(model.blocks):
        if past_kvs is None: 
            past_kv = None
        else:
            past_kv = past_kvs[i]
        h, new_kv = block_forward_cached(block, h, past_kv, pos_offset)
        new_kvs.append(new_kv)
    h = model.head(model.norm_f(h))
    return (h, new_kvs)

# ==================================================================================
# TODO 4 -- generate with a KV cache: PREFILL then incremental DECODE.
#   prompt : (B, L) long.  Return (B, L + max_new_tokens).
#   Sampling knobs mirror Day-3 generate(): greedy | temperature/top_k/top_p + generator.
#
#   1. model.eval(); assert L + max_new_tokens <= model.cfg.T  (learned-pos table cap;
#      a real impl would evict the oldest cache entries -- out of scope here).
#   2. Under torch.no_grad():
#      a. PREFILL: logits, caches = gpt_forward_cached(model, prompt, None, pos_offset=0)
#      b. loop max_new_tokens times:
#           next_logits = logits[:, -1]                       # (B, V)
#           pick next_tok (B,1): greedy=argmax, else temp->top_k->top_p->sample
#           idx = cat([idx, next_tok], dim=-1)
#           DECODE step: feed ONLY next_tok with pos_offset = idx.shape[1]-1 (its abs pos),
#             logits, caches = gpt_forward_cached(model, next_tok, caches, pos_offset)
#   Return idx.
# ==================================================================================
def generate_cached(
    model: TinyGPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    model.eval()
    B, L = prompt.shape[0], prompt.shape[-1]
    assert L + max_new_tokens <= model.cfg.T
    with torch.no_grad():
        logits, kv_cache = gpt_forward_cached(model, prompt, None, 0)
        context_full = prompt
        for t in range(max_new_tokens):
            next_logits = logits[:, -1]
            if greedy:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = apply_temperature(next_logits, temperature)
                next_logits = top_k_filter(next_logits, top_k)
                next_logits = top_p_filter(next_logits, top_p)
                next_tok = sample(next_logits, generator)
            context_full = torch.cat([context_full, next_tok], dim=-1)
            logits, kv_cache = gpt_forward_cached(model, next_tok, kv_cache, context_full.shape[-1]-1)
    return context_full

            






# ==================================================================================
# TODO 5 -- KV-cache size in BYTES.
#   Formula:  bytes = 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes
#     2         = K and V
#     n_kv_heads= KV heads (== n_heads for MHA; SMALLER for GQA/MQA -- this is what shrinks)
#     d_head    = cfg.d // cfg.n_heads
#   Args: cfg, seq_len, batch=1, dtype_bytes=2 (fp16), n_kv_heads=None -> default cfg.n_heads.
#   Return an int number of bytes.
# ==================================================================================
def kv_cache_bytes(
    cfg: GPTConfig,
    seq_len: int,
    batch: int = 1,
    dtype_bytes: int = 2,
    n_kv_heads: int | None = None,
) -> int:
    if n_kv_heads is None: 
        n_kv_heads = cfg.n_heads
    return 2 * cfg.n_layers * n_kv_heads * (cfg.d // cfg.n_heads) * seq_len * batch * dtype_bytes
    raise NotImplementedError("TODO 5: implement the KV-cache memory formula")


# ==================================================================================
# Self-checks -- all should print PASS once your code is correct.  (Provided.)
# ==================================================================================
def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _train_successor(seed: int = 0):
    """Overfit the successor task so greedy decoding should 'count up' mod V (Day-3 task)."""
    import torch.nn.functional as F
    cfg = GPTConfig(V=16, d=64, n_heads=4, n_layers=2, T=16)
    torch.manual_seed(seed)
    model = TinyGPT(cfg)
    gen = torch.Generator().manual_seed(seed)
    start = torch.randint(0, cfg.V, (8, 1), generator=gen)
    x = (start + torch.arange(cfg.T).unsqueeze(0)) % cfg.V
    y = (x + 1) % cfg.V
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        loss = F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model, cfg


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---------- kv_cache_bytes: the 7B example must be 2 GiB ----------
    cfg7b = GPTConfig(V=32000, d=4096, n_heads=32, n_layers=32, T=4096)
    b_mha = kv_cache_bytes(cfg7b, seq_len=4096, batch=1, dtype_bytes=2)
    _check("kv_cache_bytes 7B MHA @4k,B1,fp16 == 2 GiB", b_mha == 2 * 1024**3,
           f"{b_mha/1024**3:.3f} GiB")
    per_tok = kv_cache_bytes(cfg7b, seq_len=1, batch=1, dtype_bytes=2)
    _check("kv_cache_bytes per-token == 512 KiB", per_tok == 512 * 1024,
           f"{per_tok/1024:.0f} KiB/token")
    b_gqa = kv_cache_bytes(cfg7b, seq_len=4096, batch=1, dtype_bytes=2, n_kv_heads=8)
    _check("GQA n_kv=8 is 4x smaller than MHA", b_mha == 4 * b_gqa,
           f"MHA={b_mha/1024**3:.2f}GiB GQA={b_gqa/1024**3:.2f}GiB")
    b_batched = kv_cache_bytes(cfg7b, seq_len=8192, batch=16, dtype_bytes=2)
    _check("scales linearly in seq*batch (8k,B16 == 64 GiB)", b_batched == 64 * 1024**3,
           f"{b_batched/1024**3:.1f} GiB")

    # ---------- correctness: cache vs full recompute (random weights) ----------
    cfg = GPTConfig(V=32, d=128, n_heads=4, n_layers=3, T=64)
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.V, (2, 8))

    # prefill logits must match a plain full forward
    with torch.no_grad():
        full = model(prompt)                                   # (B, L, V)
        pre_logits, caches = gpt_forward_cached(model, prompt, None, pos_offset=0)
    _check("prefill logits match full forward",
           torch.allclose(full, pre_logits, atol=1e-4),
           f"max|Δ|={ (full-pre_logits).abs().max():.2e}")
    _check("cache has n_layers entries", len(caches) == cfg.n_layers)
    k0, v0 = caches[0]
    _check("cached K/V shape (B,H,P,d_k)",
           tuple(k0.shape) == (2, cfg.n_heads, 8, cfg.d // cfg.n_heads),
           f"{tuple(k0.shape)}")

    # incremental decode: each step's logits match a full recompute; cache grows by 1
    with torch.no_grad():
        idx = prompt.clone()
        logits, caches = gpt_forward_cached(model, prompt, None, pos_offset=0)
        grow_ok, match_ok = True, True
        for step in range(6):
            nt = logits[:, -1].argmax(-1, keepdim=True)        # greedy next token
            idx = torch.cat([idx, nt], dim=-1)
            logits, caches = gpt_forward_cached(model, nt, caches, pos_offset=idx.shape[1] - 1)
            if caches[0][0].shape[2] != idx.shape[1]:
                grow_ok = False
            full_step = model(idx)[:, -1]                       # what a full recompute says
            if not torch.allclose(full_step, logits[:, -1], atol=1e-4):
                match_ok = False
    _check("cache grows by 1 per decode step", grow_ok, f"len={caches[0][0].shape[2]}")
    _check("incremental logits match full recompute each step", match_ok)

    # ---------- MARQUEE: cached greedy == uncached Day-3 generate (token-identical) ----------
    g_uncached = generate(model, prompt, max_new_tokens=20, greedy=True)
    g_cached = generate_cached(model, prompt, max_new_tokens=20, greedy=True)
    _check("cached greedy == uncached greedy (token-identical)",
           torch.equal(g_uncached, g_cached),
           f"cached={g_cached[0].tolist()}")

    # cached sampling is reproducible under a fixed seed
    gg1 = torch.Generator().manual_seed(7)
    gg2 = torch.Generator().manual_seed(7)
    s1 = generate_cached(model, prompt, 10, temperature=1.0, top_k=5, generator=gg1)
    s2 = generate_cached(model, prompt, 10, temperature=1.0, top_k=5, generator=gg2)
    _check("cached seeded sampling reproducible", torch.equal(s1, s2))

    # ---------- trained successor: cached greedy still counts up mod V ----------
    tmodel, tcfg = _train_successor(0)
    tprompt = torch.tensor([[3, 4, 5]])
    tc = generate_cached(tmodel, tprompt, max_new_tokens=5, greedy=True)
    cont = tc[0, tprompt.shape[1]:].tolist()
    expected = [(5 + i + 1) % tcfg.V for i in range(5)]         # 6,7,8,9,10
    _check("cached greedy reproduces successor sequence", cont == expected,
           f"got={cont} expected={expected}")

    # ---------- timing: cached beats uncached for a long generation ----------
    cfg_big = GPTConfig(V=64, d=256, n_heads=8, n_layers=6, T=512)
    torch.manual_seed(0)
    big = TinyGPT(cfg_big).eval()
    bprompt = torch.randint(0, cfg_big.V, (1, 16))
    N = 200
    t0 = time.perf_counter(); _ = generate(big, bprompt, N, greedy=True); t_un = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = generate_cached(big, bprompt, N, greedy=True); t_ca = time.perf_counter() - t0
    _check(f"cached faster than uncached for N={N} (speedup {t_un/t_ca:.2f}x)", t_ca < t_un,
           f"uncached={t_un*1e3:.0f}ms cached={t_ca*1e3:.0f}ms")
