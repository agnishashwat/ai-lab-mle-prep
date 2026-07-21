# Deliverable - Week 02 Day 03: Decoding Strategies

## What I Built / Derived

`code/decoding.py` (+ `test_decoding.py`, 9/9 pytest, 13/13 self-check) — the full autoregressive
decode toolkit built on the Day-2 `TinyGPT` (`forward(idx) -> (B,T,V)`):

- **`apply_temperature(logits, T)`** — returns rescaled **logits** `z/T` (not probs, so it composes
  with the filters); guards tiny `T`.
- **`top_k_filter(logits, k)`** — threshold at the k-th largest via `topk(...).values[..., -1:]`,
  mask below with `-inf`; batch-shape-agnostic, no explicit mask tensor.
- **`top_p_filter(logits, p)`** — sort desc → softmax → cumsum → `remove = cumsum >= p`, **shift
  right by one** (keep the crossing token), always keep top-1, then **scatter the mask back to
  original vocab order**.
- **`sample`** — `multinomial` on `softmax(logits)`, `generator=` threaded for reproducibility.
- **`generate(...)`** — `eval()` + `no_grad()` loop; crops context to the last `cfg.T` tokens each
  step; greedy branch bypasses temp/filters; keeps a **full** running sequence separate from the
  cropped model input.
- **`beam_search(...)`** — expand every beam by all `V` tokens → pool `beam_width×V` candidates →
  prune to top `beam_width` by **length-normalized** score; `B=1` by design.

Self-check evidence: temperature sweeps mean entropy `0.000 → 1.744 → 2.079` (= `ln V`, uniform
ceiling) as `T: 0 → 1 → ∞`; top-p keeps **1** token on a peaked dist vs **5** on a flat one (adaptive);
greedy and beam both recover the trained successor sequence `[6,7,8,9,10]` (low-entropy ⇒ they agree).

## Core Takeaway

Decoding turns next-token logits into a chosen token, and every strategy trades off on one axis:
**search (greedy/beam) maximizes likelihood; sampling (temperature + nucleus) matches the entropy of
human text.** Temperature rescales confidence (`softmax(z/T)`, log-odds scale by `1/T`); top-k caps
the tail with a fixed budget; top-p (nucleus) caps it **adaptively** — few tokens when confident,
many when uncertain — which is why it beats top-k on open generation. Beam search fails on open text
because the max-likelihood continuation of a calibrated LM is a **repetition attractor** (the model
attends to its own output, so each repeat is lower-entropy and higher-probability), and human text
lives in the typical set, not at the mode. Beam stays correct for low-entropy tasks (translation,
code, ASR).

## Key Equations

- Temperature: `p = softmax(z/T)`; log-odds `log(p_i/p_j) = (z_i - z_j)/T` (monotone → greedy is
  T-invariant). `T→0` ⇒ argmax; `T→∞` ⇒ uniform; entropy monotone in `T`, `0 → ln V`.
- Nucleus set: smallest `V^(p)` with `Σ_{x∈V^(p)} P(x) >= p` (boundary test is **`>=`**; `>` keeps
  one extra token at exact ties — the HF quirk).
- Beam score (length-normalized): `S_α = (1/L^α) Σ_t log p_t`; `α=0` raw sum (length-biased toward
  short), `α=1` per-token average, `α≈0.6–0.7` typical. Raw `Σ log p_t ≤ 0` ⇒ every extra token
  lowers the score ⇒ length bias.

## Implementation Notes

- **Filters return masked LOGITS, not probabilities** — so `temp → top_k → top_p` all speak "logits"
  and a single final `softmax` normalizes once. Renormalizing mid-pipeline would be wrong after a
  later filter.
- **top-p un-sort is the #1 silent bug:** the remove-mask is built in *sorted* order and must be
  `scatter_(-1, sort_idx, mask)` back to vocab order; masking in sorted order zeroes the wrong ids.
- **Shift-right in top-p:** `remove[i]` must mean "the prefix *before* `i` already reached `p`", so
  `remove[..., 1:] = remove[..., :-1].clone()` (+ `.clone()` to avoid read/write aliasing) and force
  `remove[..., 0] = False` to always keep ≥1 token.
- **`generate` must crop to `cfg.T` INSIDE the loop** (learned absolute pos-table is hard-capped at
  `T=16`); keep the full sequence in a separate accumulator so the return isn't truncated.
- `x.new_tensor([[v]])` to append a token — inherits `ids` dtype+device (GPU-safe); `torch.cat(...,
  dim=-1)` (default `dim=0` is the trap). `.item()` on beam scores to keep Python floats.
- Threshold idiom `values[..., -1:]` (or `..., -1, None`) keeps a trailing size-1 axis so it
  broadcasts against `(B,V)`; a bare `[..., -1]` gives `(B,)` and misaligns unless `V==B` (square-
  shape silent bug).

## Failure Modes / Debugging Notes

- **Top-k failure is symmetric:** fixed `k` on a peaked dist drags in a junk tail; on a flat dist it
  amputates real mass. Top-p adapts to the distribution's entropy and avoids both.
- **Neural text degeneration:** greedy/beam fall into a repetition loop — a positive-feedback
  attractor where each repeat raises the probability of repeating. Fix = temperature + nucleus
  sampling (adds variance to escape the basin, truncates the unreliable tail).
- **Batched beam is not free:** sampling batches trivially (rows independent), but beam couples the
  `k` hypotheses of each prompt via the top-k prune → needs a `(B, k)` layout, a per-row top-k over
  `k×V`, and a beam-reindex (`//V`, `%V`) every step. Hence the `B=1` stub scope.
- **Inference hygiene** (Week-1 D5): `eval()` + `no_grad()` around the generate loop — else dropout
  stays on and the activation graph leaks memory.

## Interview Answer

Prompt:

> "Why does beam search produce repetitive, bland text on open-ended generation, and what would you
> use instead?"

Answer:

Beam search maximizes sequence log-likelihood, but for a well-calibrated LM the maximum-likelihood
continuation of open-ended text is a **repetition loop**. Transformers attend to their own context,
so once a phrase repeats, repeating it again is the lowest-entropy, highest-probability continuation
— a positive-feedback attractor that beam, which explicitly searches for the highest-scoring
sequence, walks straight into. The deeper issue is that likelihood is the wrong objective here: human
text sits in the **typical set**, not at the mode (good writing periodically picks informative,
lower-probability tokens), so maximizing probability optimizes for the wrong target. The fix is
stochastic decoding — **temperature + nucleus (top-p) sampling** — which injects enough variance to
escape the attractor while truncating the unreliable tail. Beam is still the right tool for
**low-entropy, closed-ended** tasks (machine translation, code, ASR), where a single high-likelihood
answer exists and the mode ≈ the target; there you also add **length normalization** (`1/L^α`) to
undo beam's bias toward shorter sequences.

## Reusable Artifact

- Path: `code/decoding.py`, `code/test_decoding.py`
- How to reuse: `generate(model, prompt, max_new_tokens, temperature=, top_k=, top_p=, greedy=,
  generator=)` is a drop-in sampler for any `TinyGPT`-shaped model (`(B,L) → (B,V)` last-step
  logits); the four pure filters (`apply_temperature`, `top_k_filter`, `top_p_filter`, `sample`)
  compose in that order on any `(..., V)` logits. `_train_successor()` gives a deterministic
  low-entropy model for decode sanity checks. Day 4 (KV cache + batching) makes this `generate()`
  fast by not recomputing all `T` positions each step.
