"""Week 02 Day 01 -- positional encodings from scratch.

Attention is permutation-equivariant: swap two input positions and the outputs
swap identically. So order must be *injected*. Implement the two families that
matter most in a frontier-lab interview:

    1) sinusoidal absolute encoding  (the original Transformer)
    2) RoPE -- rotary relative encoding (Llama / most modern decoders)

Fill in every TODO, then run the self-check harness at the bottom:
    ./.venv/bin/python ai-lab-mle-prep/week-02-transformers-generation/code/positional_encoding.py

Shapes:
    T  sequence length      D  (head) dim / model dim
    q, k for RoPE:  (..., T, D)   with D even
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# 1) Sinusoidal absolute positional encoding  ->  (T, D)
#    PE[pos, 2i]   = sin(pos / base^(2i/D))
#    PE[pos, 2i+1] = cos(pos / base^(2i/D))
# ---------------------------------------------------------------------------
def sinusoidal_positional_encoding(T: int, D: int, base: float = 10000.0) -> torch.Tensor:
    assert D % 2 == 0, "D must be even"
    # TODO 1: positions -> (T, 1) column of [0..T-1] (float)
    # TODO 2: inv_freq -> (D/2,) = base^(-2i/D) for i in 0..D/2-1
    #         (equivalently: exp(-(2i/D) * log(base)))
    # TODO 3: angles -> (T, D/2) = positions * inv_freq (outer product)
    # TODO 4: interleave sin on even indices, cos on odd indices -> (T, D)
    #         (build a (T, D) zero tensor, fill [..., 0::2]=sin, [..., 1::2]=cos)
    positions = torch.arange(0, T, dtype=torch.float)
    idx = torch.arange(0, D//2, dtype=torch.float)
    inv_freq = torch.exp(-2*idx/D * math.log(base))
    angles = torch.outer(positions, inv_freq)
    m = torch.zeros(T, D)
    m[:, 0::2] = torch.sin(angles)
    m[:, 1::2] = torch.cos(angles)
    return m
    

# ---------------------------------------------------------------------------
# 2) RoPE -- rotary positional embedding.
#    View each consecutive pair (x_2i, x_2i+1) as a 2-D vector and rotate it by
#    angle  theta_i * pos,  where theta_i = base^(-2i/D).
#    Key property: <RoPE(q, m), RoPE(k, n)> depends only on (m - n).
# ---------------------------------------------------------------------------
def rope_cos_sin(T: int, D: int, base: float = 10000.0):
    """Return (cos, sin) caches, each (T, D), where each frequency is duplicated
    across the pair so it can multiply x elementwise."""
    assert D % 2 == 0, "D must be even"
    # TODO 5: inv_freq -> (D/2,) = base^(-2i/D)
    # TODO 6: angles -> (T, D/2) = outer(positions, inv_freq)
    # TODO 7: duplicate each column so cos/sin are (T, D):
    #         freqs_full = torch.repeat_interleave(angles, 2, dim=-1)  # (T, D)
    #         return freqs_full.cos(), freqs_full.sin()
    idx = torch.arange(D//2, dtype=torch.float)
    inv_freq = torch.exp(-2*idx/D * math.log(base))
    positions = torch.arange(T, dtype=torch.float)
    angles = torch.outer(positions, inv_freq)
    freqs_full = torch.repeat_interleave(angles, 2, dim=-1)
    return torch.cos(freqs_full), torch.sin(freqs_full)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Given x = [x0, x1, x2, x3, ...] (pairs), return [-x1, x0, -x3, x2, ...].
    This is the '90-degree rotation' partner used by the RoPE formula."""
    # TODO 8: split x into even/odd along last dim, then interleave (-odd, even).
    #   x_even = x[..., 0::2]  ; x_odd = x[..., 1::2]
    #   stack (-x_odd, x_even) on a new last axis and flatten back to (..., D)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2, -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, D). cos/sin: (T, D). Returns rotated x, same shape.
    Formula (elementwise):  x * cos + rotate_half(x) * sin."""
    # TODO 9: broadcast cos/sin over the leading dims of x and apply the formula.
    return x * torch.broadcast_to(cos, x.shape) + _rotate_half(x) * torch.broadcast_to(sin, x.shape)
    


# ---------------------------------------------------------------------------
# Self-checks -- all should print PASS once your code is correct.
# ---------------------------------------------------------------------------
def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


if __name__ == "__main__":
    torch.manual_seed(0)
    T, D = 16, 8

    # --- sinusoidal ---
    pe = sinusoidal_positional_encoding(T, D)
    _check("sinusoidal shape", tuple(pe.shape) == (T, D), f"{tuple(pe.shape)}")
    _check("sinusoidal bounded in [-1,1]", bool(pe.abs().max() <= 1.0 + 1e-6))
    _check("sinusoidal rows distinct",
           bool((pe[1:] - pe[:-1]).abs().sum(-1).min() > 1e-6),
           "adjacent positions differ")

    # --- RoPE norm preservation (rotation is orthogonal) ---
    cos, sin = rope_cos_sin(T, D)
    q = torch.randn(2, 4, T, D)  # (B, H, T, D)
    q_rot = apply_rope(q, cos, sin)
    _check("RoPE preserves norm",
           torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5),
           f"max|Δ|={ (q_rot.norm(-1)-q.norm(-1)).abs().max():.2e}")

    # --- RoPE at position 0 is identity ---
    _check("RoPE(pos=0) == identity", torch.allclose(q_rot[..., 0, :], q[..., 0, :], atol=1e-6))

    # --- RoPE relative property: <rope(q,m), rope(k,n)> depends only on (m-n) ---
    qv = torch.randn(D)
    kv = torch.randn(D)
    Q = qv.expand(1, 1, T, D).clone()
    K = kv.expand(1, 1, T, D).clone()
    Qr = apply_rope(Q, cos, sin)[0, 0]  # (T, D), row t = qv rotated by pos t
    Kr = apply_rope(K, cos, sin)[0, 0]
    scores = Qr @ Kr.T  # (T, T): scores[m, n] = <rope(qv,m), rope(kv,n)>
    # same relative offset (m-n) => same score, regardless of absolute m,n
    diag_diffs = []
    for offset in range(-(T - 1), T):
        d = torch.diagonal(scores, offset=offset)
        diag_diffs.append((d.max() - d.min()).item())
    _check("RoPE relative-position invariance",
           max(diag_diffs) < 1e-4,
           f"max spread along a constant-offset diagonal = {max(diag_diffs):.2e}")
