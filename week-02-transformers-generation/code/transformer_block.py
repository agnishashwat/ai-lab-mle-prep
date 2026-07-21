"""Week 02 Day 02 -- the decoder-only transformer block, from scratch.

You assemble the full block on a residual stream and a tiny GPT on top of it:

    x <- x + Attn(Norm(x))     # causal MHA: the ONLY cross-position mixing
    x <- x + MLP(Norm(x))      # position-wise d->4d->d: per-token "thinking"

Design of the day = PRE-LN (norm inside the sublayer, residual path is a clean
identity => gradient highway). Attention is PROVIDED (you built it on Day 1); your
job is the four NEW pieces marked TODO: RMSNorm, MLP, TransformerBlock, TinyGPT.

Fill in every TODO, then run the self-check harness at the bottom:
    ./.venv/bin/python ai-lab-mle-prep/week-02-transformers-generation/code/transformer_block.py

Full check (pytest):
    ./.venv/bin/pytest ai-lab-mle-prep/week-02-transformers-generation/code/test_transformer_block.py -q

Shapes:
    B batch   T time/seq   V vocab   d model dim   H heads   d_k = d // H
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================================
# PROVIDED -- causal multi-head self-attention (your Day-1 core, cleaned up).
# Read it, don't edit it. This is the "mix across positions" sublayer.
# ==================================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.H = n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,d) -> (B,H,T,d_k)
        B, T, _ = x.shape
        return x.reshape(B, T, self.H, self.d_k).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:  # (B,H,T,d_k) -> (B,T,d)
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).reshape(B, T, H * d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,d) -> (B,T,d)
        B, T, _ = x.shape
        q, k, v = self._split(self.Wq(x)), self._split(self.Wk(x)), self._split(self.Wv(x))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_k)  # (B,H,T,T)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v                                            # (B,H,T,d_k)
        return self.Wo(self._merge(out))


# ==================================================================================
# TODO 1 -- RMSNorm.  RMSNorm(x) = gamma * x / sqrt(mean(x^2, over features) + eps)
#   - normalize over the LAST dim (feature dim), keepdim so it broadcasts.
#   - learnable per-feature gain `self.weight` (gamma), shape (d,), init to ones.
#   - NO mean-centering, NO bias (that is the whole point vs LayerNorm).
#   - compute the mean-square in fp32 for stability, then cast back to x.dtype.
# ==================================================================================
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., d) -> (..., d)
        return (self.weight * x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True, dtype=torch.float32)+ self.eps))


# ==================================================================================
# TODO 2 -- position-wise MLP:  d -> hidden -> d  with GELU, hidden = mult * d.
#   Two nn.Linear layers (bias ok), GELU in between. Applied independently per
#   position (nn.Linear already broadcasts over B and T).
# ==================================================================================
class MLP(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, d * mult)
        self.fc2 = torch.nn.Linear(d * mult, d)
        self.gate = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,d) -> (B,T,d)
        return self.fc2(self.gate(self.fc1(x)))


# ==================================================================================
# TODO 3 -- one PRE-LN decoder block.
#   norm1, attn, norm2, mlp as submodules. forward:
#       x = x + attn(norm1(x))
#       x = x + mlp(norm2(x))
#   Note: norm is INSIDE each sublayer's input; the residual add is a clean identity.
# ==================================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.mlp = MLP(d, mult)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,d) -> (B,T,d)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

@dataclass
class GPTConfig:
    V: int = 16
    d: int = 64
    n_heads: int = 4
    n_layers: int = 2
    T: int = 16          # max sequence length (for the positional embedding table)
    mult: int = 4


# ==================================================================================
# TODO 4 -- TinyGPT: token embed + learned positional embed + N blocks + final norm
#           + LM head -> logits (B,T,V).
#   In __init__:
#     - self.tok_emb = nn.Embedding(V, d)
#     - self.pos_emb = nn.Embedding(T, d)              # learned absolute positions
#     - self.blocks  = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
#     - self.norm_f  = RMSNorm(d)                      # final norm before the head
#     - self.head    = nn.Linear(d, V, bias=False)
#     - (already provided) self.apply(self._init_weights) then the 1/sqrt(2L) rescale
#   In forward(idx) where idx is (B,T) long:
#     - positions = arange(T); h = tok_emb(idx) + pos_emb(positions)   # (B,T,d)
#     - run through each block, then norm_f, then head -> logits (B,T,V)
# ==================================================================================
class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(self.cfg.V, self.cfg.d)
        self.pos_emb = nn.Embedding(self.cfg.T, self.cfg.d)
        self.blocks = nn.ModuleList([TransformerBlock(self.cfg.d, self.cfg.n_heads, self.cfg.mult) for _ in range(self.cfg.n_layers)])
        self.norm_f  = RMSNorm(self.cfg.d)
        self.head    = nn.Linear(self.cfg.d, self.cfg.V, bias=False)
        # TODO 4: build the submodules listed above.
        # ---- provided init (uncomment once submodules exist) ----
        self.apply(self._init_weights)
        # GPT-2 residual-projection scaling: shrink each sublayer's OUTPUT proj by
        # 1/sqrt(2 * n_layers) so the residual-stream variance stays O(1) with depth.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for blk in self.blocks:
            with torch.no_grad():
                blk.attn.Wo.weight.mul_(scale)
                blk.mlp.fc2.weight.mul_(scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # (B,T) -> (B,T,V)
        B, T = idx.shape
        positions = torch.arange(T)
        h = self.tok_emb(idx) + self.pos_emb(positions)
        for blk in self.blocks:
            h = blk(h)
        h = self.head(self.norm_f(h))
        return h


# ==================================================================================
# Self-checks -- all should print PASS once your code is correct.  (Provided.)
# ==================================================================================
def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _make_successor_batch(gen, B, T, V):
    start = torch.randint(0, V, (B, 1), generator=gen)
    x = (start + torch.arange(T).unsqueeze(0)) % V
    y = (x + 1) % V
    return x, y


def _train_overfit(model, x, y, steps=300, lr=3e-3):
    """Thin adapter: overfit ONE fixed batch. Reuses the Week-1 loop invariants."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    V = model.cfg.V
    last = None
    for _ in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = loss.item()
    return last


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig(V=16, d=64, n_heads=4, n_layers=2, T=16)

    # --- RMSNorm: unit RMS + right shape ---
    rms = RMSNorm(cfg.d)
    z = torch.randn(3, 5, cfg.d) * 7.0 + 2.0
    zn = rms(z)
    rms_val = zn.pow(2).mean(-1).sqrt()   # gamma=1 => should be ~1 per row
    _check("RMSNorm shape", tuple(zn.shape) == (3, 5, cfg.d), f"{tuple(zn.shape)}")
    _check("RMSNorm unit RMS", torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-3),
           f"mean rms={rms_val.mean():.4f}")

    # --- block preserves stream shape ---
    blk = TransformerBlock(cfg.d, cfg.n_heads, cfg.mult)
    h = torch.randn(2, cfg.T, cfg.d)
    _check("block preserves shape", tuple(blk(h).shape) == (2, cfg.T, cfg.d))

    # --- TinyGPT logits shape ---
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.V, (2, cfg.T))
    logits = model(idx)
    _check("TinyGPT logits shape", tuple(logits.shape) == (2, cfg.T, cfg.V), f"{tuple(logits.shape)}")

    # --- MARQUEE causal test: editing tokens > t must NOT change logits at <= t ---
    model.eval()
    with torch.no_grad():
        base = model(idx)
        idx2 = idx.clone()
        t = cfg.T // 2
        idx2[:, t + 1:] = (idx2[:, t + 1:] + 1) % cfg.V   # perturb the FUTURE only
        pert = model(idx2)
    causal_ok = torch.allclose(base[:, : t + 1], pert[:, : t + 1], atol=1e-5)
    future_changed = not torch.allclose(base[:, t + 1:], pert[:, t + 1:], atol=1e-5)
    _check("causal: past logits unchanged when future edited", causal_ok,
           f"max|Δ|_past={ (base[:, :t+1]-pert[:, :t+1]).abs().max():.2e}")
    _check("sanity: future logits DID change", future_changed)

    # --- overfit one batch: a correct block should drive CE well below ln V ---
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    gen = torch.Generator().manual_seed(0)
    x, y = _make_successor_batch(gen, B=8, T=cfg.T, V=cfg.V)
    final = _train_overfit(model, x, y, steps=300)
    _check(f"overfit-one-batch (final CE {final:.4f} << ln V {math.log(cfg.V):.3f})",
           final < 0.1)
