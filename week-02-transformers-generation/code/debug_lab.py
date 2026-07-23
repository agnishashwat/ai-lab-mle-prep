"""Week 02 Day 05 -- Generation Debug Lab: the regression suite that would have
caught every bug class from this week's TinyGPT + decoding + KV-cache stack.

Each test asserts an INVARIANT -- a property that must hold regardless of the
learned weights -- so the tests are cheap, fast, and catch bugs before training
ever starts. Map from test -> bug it catches (the four "contract" buckets):

    test_causal_no_leakage        -> CAUSAL   : future leaks into the past
    test_cached_equals_uncached   -> TRAIN/INF: KV cache / pos-offset / q@k_all bugs
    test_masked_row_not_all_inf   -> NUMERIC  : all-(-inf) softmax row -> NaN
    test_no_nan_forward_backward  -> NUMERIC  : init/eps/fp overflow at step 0
    test_shape_contract           -> SHAPE    : (B,T,V) vs (B,T) CE trap (silent when T==V)
    test_overfit_one_batch        -> END-TO-END: the model can actually learn (marquee)

Run the human-readable harness:
    ./.venv/bin/python ai-lab-mle-prep/week-02-transformers-generation/code/debug_lab.py

Or as a real pytest suite:
    ./.venv/bin/pytest ai-lab-mle-prep/week-02-transformers-generation/code/debug_lab.py -q

YOUR JOB: fill in the six `test_*` bodies (the assertions). Everything else --
helpers, the harness, and the planted-bug demo that proves the suite has teeth --
is provided. Read each docstring: it tells you exactly what to assert and why.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from transformer_block import GPTConfig, TinyGPT
from decoding import generate
from kv_cache import generate_cached, gpt_forward_cached, create_causal_mask


# ==================================================================================
# Provided helpers -- so each test focuses on the ASSERTION, not the setup.
# ==================================================================================
def _small_cfg(**overrides) -> GPTConfig:
    """A tiny config. Pass overrides to break square-shape coincidences (e.g. V!=T)."""
    base = dict(V=16, d=64, n_heads=4, n_layers=2, T=16, mult=4)
    base.update(overrides)
    return GPTConfig(**base)


def _random_model(cfg: GPTConfig | None = None, seed: int = 0) -> TinyGPT:
    cfg = cfg or _small_cfg()
    torch.manual_seed(seed)
    return TinyGPT(cfg)


def _successor_batch(cfg: GPTConfig, B: int = 8, seed: int = 0):
    """The Week-1/2 marquee task: y = (x + 1) mod V, x = a run counting up mod V."""
    gen = torch.Generator().manual_seed(seed)
    start = torch.randint(0, cfg.V, (B, 1), generator=gen)
    x = (start + torch.arange(cfg.T).unsqueeze(0)) % cfg.V
    y = (x + 1) % cfg.V
    return x, y


def _trained_successor(seed: int = 0):
    """Overfit the successor task; returns (model, cfg). Greedy should count up mod V."""
    cfg = _small_cfg()
    model = _random_model(cfg, seed)
    x, y = _successor_batch(cfg, B=8, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        loss = F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model, cfg


# ==================================================================================
# TEST 1 -- test_causal_no_leakage  (CAUSAL contract)
#
#   Invariant: editing tokens STRICTLY AFTER position t must not change the logits
#   at positions <= t. If it does, the future is leaking into the past (mask missing,
#   wrong shape, applied after softmax, or an off-by-one).
#
#   Recipe:
#     - model = _random_model(); model.eval()   (no training needed -- invariant test)
#     - idx = random (B, T) longs in [0, V)
#     - under torch.no_grad(): base = model(idx)
#     - clone idx, perturb ONLY columns > t (e.g. t = T//2): idx2[:, t+1:] = (.. + 1) % V
#     - pert = model(idx2)
#     - ASSERT: base[:, :t+1] allclose pert[:, :t+1]   (atol ~1e-5)  -> past unchanged
#     - ASSERT: base[:, t+1:] NOT allclose pert[:, t+1:]             -> future DID change
#       (the second assert is a sanity guard: proves the edit was actually applied)
# ==================================================================================
def test_causal_no_leakage():
    model = _random_model()
    model.eval()
    B = 12
    idx = torch.randint(low=0, high=model.cfg.V, size=(B, model.cfg.T))
    with torch.no_grad():
        base = model(idx)
        new_idx = idx.clone()
        new_idx[:, model.cfg.T//2:] = (new_idx[:, model.cfg.T//2:] + 1) % model.cfg.V
        pert = model(new_idx)
        assert torch.norm(base[:, :model.cfg.T//2] - pert[:, :model.cfg.T//2]) < 1e-5
        assert torch.norm(base[:, model.cfg.T//2:] - pert[:, model.cfg.T//2:]) > 1e-5


# ==================================================================================
# TEST 2 -- test_cached_equals_uncached  (TRAIN/INFERENCE contract)
#
#   Invariant: greedy generation through the KV-cache path must be TOKEN-IDENTICAL
#   to the naive uncached path. Catches the pos-offset bug (D), the q@k not q@k_all
#   bug (B), the offset-mask bug (A), and cache-ordering bugs in one shot.
#
#   Recipe:
#     - model, cfg = _trained_successor()   (or a random model -- equality holds either way,
#       but a trained model makes the greedy path exercise realistic tokens)
#     - prompt = torch.tensor([[3, 4, 5]])
#     - naive  = generate(model, prompt, max_new_tokens=5, greedy=True)
#     - cached = generate_cached(model, prompt, max_new_tokens=5, greedy=True)
#     - ASSERT torch.equal(naive, cached)
# ==================================================================================
def test_cached_equals_uncached():
    # A RANDOM (untrained) model is deliberate: equality is a weights-independent
    # invariant, and a random model's logits depend on ALL positions via attention,
    # so a broken causal/offset mask actually changes the greedy tokens. A model
    # trained on the position-wise successor task is INSENSITIVE to attention bugs
    # (next = current + 1 needs no attention) -> the test would be toothless.
    model = _random_model()
    B = 1; prompt_len = 3
    prompt = torch.randint(low=0, high=model.cfg.V, size=(B, prompt_len))
    non_cached = generate(model, prompt, max_new_tokens=5, greedy=True)
    cached = generate_cached(model, prompt, max_new_tokens=5, greedy=True)
    assert torch.equal(non_cached, cached)


# ==================================================================================
# TEST 3 -- test_masked_row_not_all_inf  (NUMERIC contract)
#
#   Invariant: no attention query row may be fully masked. A row that is all -inf
#   before softmax -> softmax over all -inf -> 0/0 -> NaN. For a valid causal mask,
#   query i attends to >= 1 key (itself), so EVERY row must have at least one True.
#
#   Recipe (test the offset causal mask across prefill AND decode-ish offsets):
#     - for (pos_offset, T_rows, S_cols) in a few cases that mimic prefill and decode:
#         prefill: create_causal_mask(pos_offset=0, T=S, S=S, device='cpu')  (S rows, S cols)
#         decode : create_causal_mask(pos_offset=P, T=1, S=P+1, device='cpu') (1 row, P+1 cols)
#       (match the (T, S) = (query_len, key_len) convention used in kv_cache.sdpa_cached)
#     - ASSERT mask.any(dim=-1).all()   -> every query row keeps >= 1 key
#     - BONUS: build scores = zeros, masked_fill(~mask, -inf), softmax(-1),
#       ASSERT torch.isfinite(result).all()  -> proves no NaN is produced
# ==================================================================================
def test_masked_row_not_all_inf():
    pos_offset, S, P = 5, 7, 10
    mask_prefill = create_causal_mask(pos_offset=0, T=S, S=S, device='cpu')
    mask_decode = create_causal_mask(pos_offset=pos_offset, T=1, S=P+1, device='cpu')
    assert mask_prefill.any(dim=-1).all()
    assert mask_decode.any(dim=-1).all()

    # NOTE: zeros must be FLOAT (zeros_like(bool_mask) is bool -> -inf casts to True,
    # and isfinite(bool) is trivially True). And check finiteness AFTER softmax:
    # -inf scores are legitimate (that's the mask); softmax turns each -inf into a
    # finite 0 UNLESS the whole row is -inf (0/0 -> NaN), which is what we guard against.
    scores_prefill = torch.zeros(mask_prefill.shape, dtype=torch.float32)
    scores_prefill = scores_prefill.masked_fill(~mask_prefill, float("-inf"))
    assert torch.isfinite(torch.softmax(scores_prefill, dim=-1)).all()

    scores_decode = torch.zeros(mask_decode.shape, dtype=torch.float32)
    scores_decode = scores_decode.masked_fill(~mask_decode, float("-inf"))
    assert torch.isfinite(torch.softmax(scores_decode, dim=-1)).all()



# ==================================================================================
# TEST 4 -- test_no_nan_forward_backward  (NUMERIC contract, step 0)
#
#   Invariant: a single train step on a fresh model yields a FINITE loss and FINITE
#   grads for every parameter. Catches init-scale / eps / fp-overflow config bugs
#   that NaN at step 0 (before the LR can ever matter).
#
#   Recipe:
#     - model = _random_model(); cfg = model.cfg
#     - x, y = _successor_batch(cfg)
#     - logits = model(x); loss = F.cross_entropy(logits.reshape(-1, cfg.V), y.reshape(-1))
#     - ASSERT torch.isfinite(loss)
#     - loss.backward()
#     - ASSERT every p.grad is not None and torch.isfinite(p.grad).all() for p in params
# ==================================================================================
def test_no_nan_forward_backward():
    model = _random_model(); cfg = model.cfg
    x, y = _successor_batch(cfg)
    logits = model(x); loss = F.cross_entropy(logits.reshape(-1, cfg.V), y.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    

# ==================================================================================
# TEST 5 -- test_shape_contract  (SHAPE contract)
#
#   Invariant: logits are (B, T, V), and the loss MUST be computed via reshape to
#   (-1, V) / (-1). The naive F.cross_entropy(logits, targets) with (B,T,V)+(B,T) is
#   the silent K-dim trap: it does NOT crash when T==V (treats time as classes ->
#   garbage), and DOES crash when T!=V. So: use a cfg with T != V to expose it.
#
#   Recipe:
#     - cfg = _small_cfg(T=12, V=16)    # T != V so the bug can't hide
#     - model = _random_model(cfg); x, y = _successor_batch(cfg)
#     - logits = model(x)
#     - ASSERT tuple(logits.shape) == (x.shape[0], cfg.T, cfg.V)
#     - correct = F.cross_entropy(logits.reshape(-1, cfg.V), y.reshape(-1))
#     - ASSERT torch.isfinite(correct)  and correct.ndim == 0   (a scalar)
#     - ASSERT the WRONG form raises: with pytest.raises / try-except, confirm
#       F.cross_entropy(logits, y) raises RuntimeError (shape mismatch, T != V)
# ==================================================================================
def test_shape_contract():
    cfg = _small_cfg(T=12, V=16)
    model = _random_model(cfg); x, y = _successor_batch(cfg)
    logits = model(x)
    assert tuple(logits.shape) == (x.shape[0], cfg.T, cfg.V)
    loss = F.cross_entropy(logits.reshape(-1, cfg.V), y.reshape(-1))
    assert torch.isfinite(loss) and loss.ndim == 0
    # The naive (no-reshape) form MUST raise when T != V: it hits the K-dim path
    # (input (N,C,d1)=(B,T,V), target (B,T)) and the trailing dims V vs T mismatch.
    # This is what proves the reshape is NECESSARY, not merely tidy.
    try:
        F.cross_entropy(logits, y)
    except RuntimeError:
        pass
    else:
        raise AssertionError("naive CE did NOT raise on T!=V -- the silent K-dim trap is live")

# ==================================================================================
# TEST 6 -- test_overfit_one_batch  (END-TO-END marquee)
#
#   Invariant: a correct model can DRIVE THE LOSS to ~0 on one fixed batch. This is
#   the single most valuable sanity check -- if it fails, something upstream is broken
#   (mask, shapes, init, optimizer wiring) before you waste a real training run.
#
#   Recipe:
#     - model, cfg = _trained_successor()   (already overfits 300 steps)
#     - x, y = _successor_batch(cfg)
#     - with torch.no_grad(): final = F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1))
#     - ASSERT final < 0.1        (<< ln V = ln 16 ~ 2.77)
#     - BONUS: greedy-generate from prompt [3,4,5] and assert it counts up: [6,7,8,9,10]
# ==================================================================================
def test_overfit_one_batch():
    model, cfg = _trained_successor()
    x, y = _successor_batch(cfg)
    with torch.no_grad(): final = F.cross_entropy(model(x).reshape(-1, cfg.V), y.reshape(-1))
    assert final < 0.1
    prompt = torch.tensor([[3,4,5]])
    out = generate(model, prompt, max_new_tokens=5, greedy=True)
    assert torch.equal(out[:, 3:], torch.tensor([[6,7,8,9,10]]))


# ==================================================================================
# Harness -- runs the six tests, prints PASS/FAIL, then a PLANTED-BUG DEMO that
# proves the suite has teeth. (Provided -- do not edit.)
# ==================================================================================
def _run(test_fn) -> tuple[bool, str]:
    try:
        test_fn()
        return True, ""
    except NotImplementedError as e:
        return False, f"TODO: {e}"
    except AssertionError as e:
        return False, f"assert failed: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _buggy_mask_diag0(pos_offset, T, S, device):
    """Bug A: drops the pos_offset on the causal mask -> decode attends only to pos 0."""
    return torch.tril(torch.ones(T, S, dtype=torch.bool, device=device), diagonal=0)


def _planted_bug_demo():
    """Monkeypatch Bug A into the cache path and prove test_cached_equals_uncached CATCHES it."""
    import kv_cache
    print("\n=== planted-bug demo: inject Bug A (mask diagonal=0) into the KV cache ===")
    original = kv_cache.create_causal_mask
    kv_cache.create_causal_mask = _buggy_mask_diag0
    try:
        ok, detail = _run(test_cached_equals_uncached)
        # With the bug present, the test SHOULD fail. If it "passes", the test is toothless.
        caught = not ok
        _check("regression suite catches Bug A (cached != uncached under the bug)", caught,
               "test correctly failed on the buggy mask" if caught
               else "TEST TOOTHLESS: bug slipped through!")
    finally:
        kv_cache.create_causal_mask = original


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [
        ("test_causal_no_leakage", test_causal_no_leakage),
        ("test_cached_equals_uncached", test_cached_equals_uncached),
        ("test_masked_row_not_all_inf", test_masked_row_not_all_inf),
        ("test_no_nan_forward_backward", test_no_nan_forward_backward),
        ("test_shape_contract", test_shape_contract),
        ("test_overfit_one_batch", test_overfit_one_batch),
    ]
    print("=== Week-2 regression suite ===")
    n_pass = 0
    for name, fn in tests:
        ok, detail = _run(fn)
        n_pass += ok
        _check(name, ok, detail)
    print(f"\n{n_pass}/{len(tests)} invariants pass.")

    # Only run the teeth-check once the cached==uncached test is implemented & passing.
    if _run(test_cached_equals_uncached)[0]:
        _planted_bug_demo()
