# Week 02 Day 03 — Decoding Strategies

## Objective

Understand and implement the full autoregressive decoding toolkit — greedy, temperature, top-k,
top-p (nucleus), beam search — plus repetition and length controls. Know the bias each introduces
and when each is the right tool. Build a `generate()` on top of the Day-2 `TinyGPT`.

## Prerequisites

- Day 2 trained `TinyGPT` producing logits `(B, T, V)`.
- Week 1: stable softmax/log-softmax, `gather`/`scatter`, perplexity = `exp(CE)`.

## Required Reading

- "The Curious Case of Neural Text Degeneration" (nucleus sampling) — intro + the top-p section.
- Optional: a short note on beam search + length normalization.

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Days 1-2)

1. Pre-LN block equations + the gradient-highway argument.
2. Why \(1/\sqrt{d_k}\); RMSNorm vs LayerNorm.
3. The causal-property test — what exactly does it check on the logits?
4. `log_softmax` vs `softmax().log()` — why the former.

### 0:10-0:35 - Concept Explanation (First Principles)

Decoding = repeatedly turning next-token logits into a chosen token. **Temperature** \(T\): divide
logits by \(T\) before softmax — \(T\to 0\) → argmax (greedy/deterministic), \(T\to\infty\) → uniform.
It rescales *confidence*: low \(T\) sharpens, high \(T\) flattens. **Top-k**: keep the k highest-prob
tokens, renormalize, sample — bounds the tail but k is a fixed budget regardless of how peaked the
distribution is. **Top-p (nucleus)**: keep the smallest set whose cumulative prob ≥ p — *adaptive*
truncation (few tokens when confident, many when uncertain), which is why it beats top-k on open
generation. **Greedy/beam**: beam keeps the B highest-prob *sequences*; great for low-entropy tasks
(translation, code) but produces bland, repetitive text on open-ended generation (the "degeneration"
result) and needs **length normalization** because \(\sum \log p\) penalizes longer sequences.
**Repetition penalty / no-repeat-n-gram** counter loops. Sampling vs search = diversity vs likelihood.

### 0:35-1:00 - Derivation / Paper Deconstruction

Show: (a) temperature is a monotone reparam of the softmax — \(\text{softmax}(z/T)\) — and its limit
behavior; (b) why beam search's raw score \(\sum_t \log p_t\) is length-biased and how length
normalization \(\frac{1}{L^\alpha}\sum \log p_t\) corrects it; (c) why top-p is scale-adaptive where
top-k is not (walk a peaked vs flat distribution). Deconstruct the degeneration argument: max-prob
decoding lands in a repetition attractor because human text is not the most-likely continuation.

### 1:00-1:45 - Coding / Experiment Task

`code/decoding.py` with stubs + self-check/pytest. Implement pure functions on a logits vector:

- `apply_temperature(logits, T)`, `top_k_filter(logits, k)`, `top_p_filter(logits, p)` (return masked
  logits with removed entries at `-inf`), `sample(logits, ...)`, and a `generate(model, prompt, max_new_tokens, ...)` loop.
- `greedy` and a minimal `beam_search(model, prompt, beam_width, length_penalty)`.

Self-check assertions: temperature limits (T→0 ≈ argmax; large T → near-uniform entropy up);
top-k keeps exactly k (or fewer) tokens and renormalizes to 1; top-p keeps the minimal nucleus and
is ≤ k-equivalent when peaked; filtered probs still sum to 1; greedy is deterministic; seeded
sampling is reproducible. Sanity-generate from the successor-trained `TinyGPT` and confirm it emits
the successor sequence under greedy.

### 1:45-1:55 - Interview Drill

"Top-k vs top-p — when and why?" and "Why does beam search produce repetitive/bland text on
open-ended generation, and what would you use instead?"

### 1:55-2:00 - Deliverable & Next Session

`deliverable.md`: the decoding toolkit + a small table of entropy/diversity vs temperature. Next:
KV cache + batching + inference memory math to make `generate()` fast.

## Checkpoint Questions

1. Temperature limits at \(T\to 0\) and \(T\to\infty\), and what "confidence" rescaling means.
2. Top-k vs top-p: which is adaptive and why does that matter?
3. Why is raw beam score length-biased, and what's the fix?
4. Explain neural text degeneration in one or two sentences.
5. Greedy/beam vs sampling: the likelihood-vs-diversity trade-off, mapped to task types.

## Frontier-Lab Interview Angle

Decoding shows up in eval design (temperature affects pass@k), RLVR sampling (Week 6 group
sampling), and production latency/quality trade-offs. Interviewers probe whether you know *why*
nucleus beat top-k and how sampling params interact with reasoning benchmarks.
