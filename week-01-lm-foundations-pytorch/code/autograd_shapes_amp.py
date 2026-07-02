"""Day 3 -- autograd, tensor shapes/broadcasting/masking, and mixed precision.

Implement each function. The point is reflexive fluency with the ops that break
live-coding rounds: shape traps, manual backward, fp16 NaNs, masked means.

Self-check (quick, no pytest needed):
    ./.venv/bin/python ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/autograd_shapes_amp.py

Full check (pytest):
    ./.venv/bin/pytest ai-lab-mle-prep/week-01-lm-foundations-pytorch/code/test_autograd_shapes_amp.py -q
"""
from __future__ import annotations

import torch


# ----------------------------------------------------------------------------------
# 1. SHAPE / BROADCASTING drill.
#    You are given:
#      scores : (B, T, V)  float  -- per-position logits
#      bias   : (V,)       float  -- a per-vocab bias to ADD to every position
#      mask   : (B, T)     bool   -- True = real token, False = padding
#    Return the *masked mean* over real tokens of the max-logit-after-bias, as a
#    scalar tensor. Concretely:
#      1) add bias to scores (broadcast over B, T),
#      2) take the max over the vocab dim -> (B, T),
#      3) average that (B, T) quantity over ONLY the True positions in mask.
#    Do NOT let padding positions contribute to the denominator.
#    (This bundles a broadcast trap + a masked-mean trap in one.)
# ----------------------------------------------------------------------------------
def masked_max_logit_mean(
    scores: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    scores = scores + bias.unsqueeze(0).unsqueeze(0)
    max_logits = torch.amax(scores, dim=-1)

    return max_logits[mask].sum()/mask.sum()




# ----------------------------------------------------------------------------------
# 2. MASKED MEAN done right (isolated, so you can see the wrong-vs-right denominator).
#    values : (B, T)  float
#    mask   : (B, T)  bool  -- True = keep
#    Return the mean of `values` over ONLY the True positions. If mask is all False,
#    return a 0.0 scalar (don't divide by zero -> NaN).
# ----------------------------------------------------------------------------------
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_sum = mask.sum()
    if mask_sum==0: return values.sum() * 0
    return torch.sum(values[mask])/mask_sum


# ----------------------------------------------------------------------------------
# 3. MANUAL BACKWARD for  x -> Linear(W,b) -> ReLU -> Linear(W2,b2) -> CE.
#    Compute the gradient of the mean cross-entropy w.r.t. W (the FIRST layer weight)
#    BY HAND (no autograd, no .backward()). You may use your own forward math.
#
#    Shapes:
#      x  : (N, D_in)     W  : (H, D_in)   b  : (H,)
#      h  = relu(x @ W.T + b)         -> (N, H)
#      W2 : (V, H)        b2 : (V,)
#      z  = h @ W2.T + b2             -> (N, V)   (logits)
#      y  : (N,)  long                (targets)
#      L  = mean cross-entropy(z, y)
#
#    Return dL/dW with shape (H, D_in). It will be checked against autograd.
#    Hint: dL/dz = (softmax(z) - onehot(y)) / N   [your Day-2 result]
#          then push it back through Linear2, ReLU (gate by h>0), Linear1.
# ----------------------------------------------------------------------------------
def linear_ce_grad_wrt_W(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    N = x.shape[0]
    h1 = x @ W.transpose(-1, -2) + b.unsqueeze(0)
    h = torch.relu(h1)    # [N, H]
    z = h @ W2.transpose(-1, -2) + b2.unsqueeze(0)
    log_softmax_all = torch.log_softmax(z, dim=-1)
    log_softmax_targets = log_softmax_all.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    L = -log_softmax_targets.mean()
    onehot = torch.zeros_like(z)
    onehot = onehot.scatter(dim=-1, index=y.unsqueeze(-1), value=1)
    dldz = (torch.softmax(z, dim=-1) - onehot)/N    # [N, V]
    dldh = dldz @ W2                                # [N, H]
    mask = torch.masked_fill(torch.zeros_like(h1), h1>0, value=1)
    
    dldh1 = dldh * mask                             # [N, H]
    dldW = dldh1.transpose(-1, -2) @ x                                 # [H, D_in]
    return dldW



# ----------------------------------------------------------------------------------
# 4a. fp16 OVERFLOW repro. Given a float32 tensor `z`, cast to float16 and return
#     torch.exp(z_fp16). For the harness input this should contain +inf (overflow).
#     One line. The point is to SEE the failure, not avoid it.
# ----------------------------------------------------------------------------------
def fp16_overflow(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(z.to(torch.float16))


# ----------------------------------------------------------------------------------
# 4b. THE FIX. Same conceptual computation, but numerically safe.
#     Compute a stable softmax of `z` under bf16 autocast and return it in float32.
#     Requirements:
#       - wrap the compute in `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`
#       - the result must be all-finite and each row must sum to 1
#     (bf16 has fp32's exponent range, so exp(large) doesn't overflow the way fp16 does;
#      combined with the max-subtraction it stays finite.)
# ----------------------------------------------------------------------------------
def bf16_stable_softmax(z: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        e = torch.exp(z - z.amax(dim=-1, keepdim=True))
        return e/e.sum(dim=-1, keepdim=True)


# ==================================================================================
# Quick self-check harness -- do not edit.
# ==================================================================================
def _check(name: str, ok: bool) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main() -> None:
    torch.manual_seed(0)
    results = []

    # 1. masked_max_logit_mean
    try:
        B, T, V = 2, 4, 5
        scores = torch.randn(B, T, V)
        bias = torch.randn(V)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        biased = scores + bias  # reference
        per_pos = biased.max(dim=-1).values  # (B,T)
        ref = per_pos[mask].mean()
        got = masked_max_logit_mean(scores, bias, mask)
        results.append(_check("masked_max_logit_mean", torch.allclose(got, ref, atol=1e-6)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"masked_max_logit_mean (raised: {e})", False))

    # 2. masked_mean (normal + all-False)
    try:
        vals = torch.tensor([[1.0, 2.0, 3.0, 100.0], [4.0, 5.0, 100.0, 100.0]])
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        ref = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).mean()
        ok1 = torch.allclose(masked_mean(vals, mask), ref, atol=1e-6)
        ok2 = torch.allclose(
            masked_mean(vals, torch.zeros_like(mask, dtype=torch.bool)),
            torch.tensor(0.0),
            atol=1e-6,
        )
        results.append(_check("masked_mean (incl. all-False)", ok1 and ok2))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"masked_mean (raised: {e})", False))

    # 3. manual backward vs autograd
    try:
        N, D_in, H, V = 4, 6, 8, 5
        x = torch.randn(N, D_in)
        W = torch.randn(H, D_in, requires_grad=True)
        b = torch.randn(H, requires_grad=True)
        W2 = torch.randn(V, H, requires_grad=True)
        b2 = torch.randn(V, requires_grad=True)
        y = torch.randint(0, V, (N,))

        h = torch.relu(x @ W.T + b)
        z = h @ W2.T + b2
        loss = torch.nn.functional.cross_entropy(z, y)
        loss.backward()
        ref_dW = W.grad

        got_dW = linear_ce_grad_wrt_W(x, W.detach(), b.detach(), W2.detach(), b2.detach(), y)
        results.append(_check("linear_ce_grad_wrt_W == autograd", torch.allclose(got_dW, ref_dW, atol=1e-5)))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"linear_ce_grad_wrt_W (raised: {e})", False))

    # 4a. fp16 overflow actually overflows
    try:
        z = torch.tensor([100.0, 200.0, 300.0])  # exp(300) overflows fp16 (max ~65504)
        out = fp16_overflow(z)
        results.append(_check("fp16_overflow (produces +inf)", bool(torch.isinf(out).any())))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"fp16_overflow (raised: {e})", False))

    # 4b. bf16 stable softmax is finite and normalized
    try:
        z = torch.tensor([[100.0, 200.0, 300.0], [-50.0, 0.0, 50.0]])
        out = bf16_stable_softmax(z)
        finite = bool(torch.isfinite(out).all())
        sums_to_1 = torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-2)
        results.append(_check("bf16_stable_softmax (finite + normalized)", finite and sums_to_1))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"bf16_stable_softmax (raised: {e})", False))

    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
