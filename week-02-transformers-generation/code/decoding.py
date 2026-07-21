"""Week 02 Day 03 -- autoregressive decoding strategies, from scratch.

You build the full decode toolkit on top of the Day-2 `TinyGPT`:

    logits = model(idx)            # (B, T, V)
    next_logits = logits[:, -1]    # (B, V)  -- the ONLY row generation cares about
    token = <some rule>(next_logits)
    idx = cat([idx, token], dim=1) # append, repeat

Every strategy is a different "rule". They split on ONE axis:
    search (greedy / beam)  = maximize likelihood -> bland, repetitive on open text
    sampling (temp + top-k / top-p) = inject controlled entropy -> human-like

Your job: fill in the six TODOs. The first three are PURE functions on a logits
tensor whose LAST dim is the vocab (shape (..., V)); they never see the model.

    ./.venv/bin/python ai-lab-mle-prep/week-02-transformers-generation/code/decoding.py
    ./.venv/bin/pytest ai-lab-mle-prep/week-02-transformers-generation/code/test_decoding.py -q

Shapes:  B batch   T ctx len   V vocab.   `prompt`/`idx` are (B, L) long tensors.

NOTE on positions: TinyGPT has a LEARNED positional table capped at cfg.T. Your
`generate` loop MUST crop the context to the last cfg.T tokens before each forward
call, or the pos-embedding lookup will index out of range.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from transformer_block import GPTConfig, TinyGPT


# ==================================================================================
# TODO 1 -- temperature.  p = softmax(z / T).
#   - Return the RESCALED LOGITS z/T (NOT probabilities) so it composes with the
#     filters below (which also operate on logits).
#   - T == 1.0 is a no-op. Guard tiny T (e.g. clamp to >= 1e-6) so you don't div by 0.
#   - Recall the log-odds view: log(p_i/p_j) = (z_i - z_j)/T  -> T scales separation.
# ==================================================================================
def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:  # (...,V)->(...,V)
    T_safe = max(T, 1e-6)
    return logits/T_safe


# ==================================================================================
# TODO 2 -- top-k filter.  Keep the k highest-logit entries per row, set the rest to
#   -inf (so softmax gives them exactly 0). Operate on the LAST dim.
#   - If k is None or k >= V, return logits unchanged.
#   - Idiom: torch.topk(logits, k, dim=-1) gives the kth-largest as values[..., -1:];
#     mask entries strictly below that threshold with float("-inf").
#   - Returns MASKED LOGITS (not probs). Do NOT renormalize here -- softmax does that.
# ==================================================================================
def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:  # (...,V)->(...,V)
    if k is None or k >= logits.shape[-1]: return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits<kth, -float('inf'))
    

# ==================================================================================
# TODO 3 -- top-p (nucleus) filter.  Keep the SMALLEST set of tokens whose cumulative
#   probability >= p; set the rest to -inf. Operate on the LAST dim.
#   - If p is None or p >= 1.0, return logits unchanged.
#   - Steps: sort logits desc -> softmax -> cumsum -> the nucleus is every token up to
#     AND INCLUDING the first one that crosses p (always keep >= 1 token).
#   - Build a boolean "remove" mask in SORTED order, then scatter it back to the
#     ORIGINAL vocab order (torch.gather/scatter with the sort indices). Classic bug:
#     forgetting to un-sort, so you mask the wrong token ids.
# ==================================================================================
def top_p_filter(logits: torch.Tensor, p: float | None) -> torch.Tensor:  # (...,V)->(...,V)
    if p is None or p>=1: return logits
    logits_sorted_vals, idx = torch.sort(logits, dim=-1, descending=True)
    prob_sorted = torch.softmax(logits_sorted_vals, dim=-1)
    cumsum = torch.cumsum(prob_sorted, dim=-1)
    remove_sorted = cumsum >= p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted, dtype=torch.bool)
    remove.scatter_(dim=-1, index=idx, src=remove_sorted)
    return logits.masked_fill(remove, -float('inf'))


# ==================================================================================
# TODO 4 -- sample one token id per row from a (..., V) logits tensor.
#   - Convert logits -> probs (softmax over -1), draw with torch.multinomial.
#   - Thread `generator` through for reproducibility (multinomial takes generator=).
#   - Return shape is the batch shape with the V dim removed, e.g. (B,) from (B,V).
#   - multinomial wants a 2-D (N, V) input; reshape in/out if you want to be general,
#     but (B, V) -> (B,) is all the harness needs.
# ==================================================================================
def sample(logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    prob_all = torch.softmax(logits, dim=-1)
    return torch.multinomial(prob_all, 1,  generator=generator)


# ==================================================================================
# TODO 5 -- the generation loop.
#   prompt: (B, L) long.  Return (B, L + max_new_tokens).
#   Each step:
#     1. crop context to the last cfg.T tokens
#     2. logits = model(ctx); take next_logits = logits[:, -1]      # (B, V)
#     3. if greedy: next_tok = argmax(next_logits, -1)
#        else:      next_logits = apply_temperature -> top_k_filter -> top_p_filter
#                   next_tok = sample(next_logits, generator)
#     4. append next_tok (as (B,1)) to idx
#   Wrap the loop in torch.no_grad() and call model.eval() (inference hygiene!).
#   Greedy is temperature/filter-invariant -> ignore those knobs when greedy=True.
# ==================================================================================
def generate(
    model: TinyGPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:  # (B,L) -> (B, L+max_new_tokens)
    model.eval()
    context_full = prompt
    with torch.no_grad():
        for t in range(max_new_tokens):
            context = context_full[..., -model.cfg.T:]
            logits = model(context)
            next_logit = logits[..., -1, :]
            if greedy:
                next_tok = torch.argmax(next_logit, dim=-1, keepdim=True)
            else:
                logits_temp = apply_temperature(next_logit, temperature)
                logits_topk_filtered = top_k_filter(logits_temp, top_k)
                logits_topp_filtered = top_p_filter(logits_topk_filtered, top_p)
                next_tok = sample(logits_topp_filtered, generator)
            context_full = torch.cat([context_full, next_tok], dim=-1)
    return context_full


# ==================================================================================
# TODO 6 -- minimal beam search.  prompt: (1, L). Return (1, L + max_new_tokens).
#   Maintain `beam_width` hypotheses, each a (sequence, cumulative logprob).
#   Each step, for every live beam:
#     - logits = model(ctx crop)[:, -1]; logp = log_softmax(logits, -1)
#     - every candidate next token adds logp[tok] to the beam's score
#   Pool all beam*V candidates, keep the top `beam_width` by LENGTH-NORMALIZED score
#       score / (len ** length_penalty)
#   (length_penalty=1.0 => per-token avg logprob; 0.0 => raw sum, length-biased).
#   Return the single best beam's ids. A simple python loop over beams is fine (no
#   need to vectorize) -- clarity over speed here.
# ==================================================================================
def beam_search(
    model: TinyGPT,
    prompt: torch.Tensor,
    beam_width: int,
    max_new_tokens: int,
    length_penalty: float = 1.0,
) -> torch.Tensor:  # (1,L) -> (1, L+max_new_tokens)
    model.eval()
    L0 = prompt.shape[-1]

    def norm_len_score(ids, score):
        seq_len = ids.shape[-1] - L0
        return score / seq_len ** length_penalty
    
    beams = [(prompt, 0)]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            pool = []
            for ids, score in beams:
                context = ids[..., -model.cfg.T:]
                logits_next = model(context)[..., -1, :]
                logprob_all = torch.log_softmax(logits_next, dim=-1)[0]
                for v in range(logprob_all.shape[-1]):
                    new_ids = torch.cat([ids, ids.new_tensor([[v]])], dim=-1)
                    new_score = score + logprob_all[v].item()
                    pool.append((new_ids, new_score))
            pool.sort(key=lambda c: norm_len_score(*c), reverse=True)
            beams = pool[:beam_width]
    best_seq = max(beams, key=lambda c:norm_len_score(*c))[0]
    return best_seq

    model.eval()
    L0 = prompt.shape[-1]

    def norm_len_score(ids, score):
        seq_len = ids.shape[-1] - L0
        return score / seq_len ** length_penalty

    beams = [(prompt, 0.0)]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            pool = []
            for ids, score in beams:
                context = ids[..., -model.cfg.T:]
                logits_final = model(context)[..., -1, :][0]
                logprob_all = torch.log_softmax(logits_final, dim=-1)
                for v in range(logprob_all.shape[-1]):
                    new_ids = torch.cat([ids, ids.new_tensor([[v]])], dim=-1)
                    new_score = score + logprob_all[v].item()
                    pool.append((new_ids, new_score))
            pool.sort(key=lambda c: norm_len_score(*c), reverse=True)
            beams = pool[:beam_width]
    beams.sort(key=lambda c: norm_len_score(*c), reverse=True)
    return beams[0][0]
            



    





# ==================================================================================
# Self-checks -- all should print PASS once your code is correct.  (Provided.)
# ==================================================================================
def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-12)).sum(-1).mean())


def _finite_count(logits: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(logits).sum(-1)


def _train_successor(seed: int = 0):
    """Overfit the successor task so greedy decoding should 'count up' mod V."""
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
    V = 8
    # a peaked and a flat logits row for the filter tests
    peaked = torch.log(torch.tensor([[0.90, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002]]))
    flat = torch.log(torch.tensor([[0.25, 0.22, 0.20, 0.18, 0.07, 0.04, 0.02, 0.02]]))

    # --- temperature limits ---
    z = torch.randn(4, V)
    lo_T = apply_temperature(z, 0.01)
    hi_T = apply_temperature(z, 100.0)
    _check("temp T->0 ~ argmax", torch.equal(lo_T.argmax(-1), z.argmax(-1)))
    _check("temp raises entropy (hi_T > native > lo_T)",
           _entropy(hi_T) > _entropy(z) > _entropy(lo_T),
           f"H(lo)={_entropy(lo_T):.3f} H(1)={_entropy(z):.3f} H(hi)={_entropy(hi_T):.3f}")

    # --- top-k keeps exactly k, renormalizes to 1 ---
    fk = top_k_filter(peaked, 3)
    _check("top-k keeps exactly k", torch.equal(_finite_count(fk), torch.tensor([3])),
           f"kept={_finite_count(fk).tolist()}")
    _check("top-k probs sum to 1", torch.allclose(torch.softmax(fk, -1).sum(-1), torch.ones(1), atol=1e-6))

    # --- top-p is adaptive: fewer on peaked, more on flat ---
    pk = top_p_filter(peaked, 0.9)
    pf = top_p_filter(flat, 0.9)
    n_peaked, n_flat = int(_finite_count(pk)), int(_finite_count(pf))
    _check("top-p adaptive (peaked keeps fewer than flat)", n_peaked < n_flat,
           f"peaked={n_peaked} flat={n_flat}")
    _check("top-p nucleus >= p", float(torch.softmax(peaked, -1)[0, :n_peaked].sum()) >= 0.9 - 1e-6,
           f"cum={float(torch.softmax(peaked,-1)[0,:n_peaked].sum()):.3f}")
    _check("top-p probs sum to 1", torch.allclose(torch.softmax(pf, -1).sum(-1), torch.ones(1), atol=1e-6))

    # --- sample is seeded-reproducible ---
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    s1 = sample(flat.repeat(50, 1), g1)
    s2 = sample(flat.repeat(50, 1), g2)
    _check("sample reproducible under fixed seed", torch.equal(s1, s2))

    # --- greedy is deterministic ---
    model, cfg = _train_successor(0)
    prompt = torch.tensor([[3, 4, 5]])
    g_a = generate(model, prompt, max_new_tokens=5, greedy=True)
    g_b = generate(model, prompt, max_new_tokens=5, greedy=True)
    _check("greedy deterministic", torch.equal(g_a, g_b))

    # --- greedy on successor model counts up mod V ---
    cont = g_a[0, prompt.shape[1]:].tolist()
    expected = [(5 + i + 1) % cfg.V for i in range(5)]  # 6,7,8,9,10
    _check("greedy reproduces successor sequence", cont == expected,
           f"got={cont} expected={expected}")

    # --- seeded sampling generate is reproducible ---
    gg1 = torch.Generator().manual_seed(7)
    gg2 = torch.Generator().manual_seed(7)
    s_gen1 = generate(model, prompt, 5, temperature=1.0, top_k=5, generator=gg1)
    s_gen2 = generate(model, prompt, 5, temperature=1.0, top_k=5, generator=gg2)
    _check("seeded sampling generate reproducible", torch.equal(s_gen1, s_gen2))

    # --- beam search runs and returns the right shape; on this task it should also
    #     count up (low-entropy => search is fine here) ---
    beam = beam_search(model, prompt, beam_width=3, max_new_tokens=5, length_penalty=1.0)
    _check("beam output shape", tuple(beam.shape) == (1, prompt.shape[1] + 5),
           f"{tuple(beam.shape)}")
    _check("beam reproduces successor sequence",
           beam[0, prompt.shape[1]:].tolist() == expected,
           f"got={beam[0, prompt.shape[1]:].tolist()}")
