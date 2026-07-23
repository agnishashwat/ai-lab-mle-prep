# Deliverable - Week 02 Day 05: Generation Debug Lab

## What I Built / Derived

- `code/debug_lab.py` — a **regression suite of six invariant tests** that would have caught
  every bug class from Week 2's TinyGPT + decoding + KV-cache stack, plus a **planted-bug demo**
  that monkeypatches a broken causal mask into the cache and proves the suite actually fires.
  6/6 pass under both the readable harness and `pytest`.
- `day-05-.../work.ipynb` — the Week-2 **generation notebook**: attention heatmaps (causal
  lower-triangular, 0 mass above the diagonal), all five decoding strategies on one prompt,
  cached==uncached equality + a **2.85x decode speedup** over 200 tokens, and a live NaN
  repro/fix (fully-masked row).
- Derived the **four-contract debugging map** and the **two crown-jewel invariant tests**.

## Core Takeaway

Almost every transformer/generation bug is a violation of one of **four contracts**: **causal**
(no future leakage), **shape/axis** (each axis means one thing; worst when two dims are equal),
**numeric** (reductions stay finite and normalized), and **train/inference** (the model must see a
token the same way at generation as during training). Debugging is: symptom -> which contract ->
**cheapest exclusion test** -> **decisive localization test**. The two highest-value tests are the
**causal-leakage test** (perturb tokens > t, assert logits <= t are bit-identical) and the
**cached==uncached test** (KV-cache greedy is token-identical to the naive path). Both are
*invariants* — true regardless of weights — so they are cheap CI guards that need no trained model.

## Key Equations / Facts

- Fully-masked softmax row: `softmax([-inf, ..., -inf])` -> `exp(-inf)=0` everywhere ->
  `0 / sum(0) = 0/0 = NaN`. Prevent by guaranteeing every query row keeps >= 1 key (the diagonal).
- CE K-dim trap: `F.cross_entropy((B,T,V), (B,T))` does NOT crash when `T==V` (treats time as
  classes -> garbage); crashes when `T!=V`. ALWAYS `reshape(-1, V)` / `reshape(-1)`.
- Temporal clue: **NaN at step 0 = config/forward bug (rules out LR); NaN after N steps = something
  grew** (grad norms, fp overflow).

## Implementation Notes / Bugs Surfaced

- **Toothless-test discovery:** `test_cached_equals_uncached` on a model trained on the
  *successor* task did NOT catch a broken decode mask — because `next = current + 1` is a
  **position-wise** function that needs no attention, so a broken attention mask leaves the output
  unchanged. Fix: use a **random model**, whose logits depend on all positions via attention, so a
  broken mask actually changes the greedy tokens. Lesson: **an invariant test's power depends on the
  input's sensitivity to the thing under test.**
- **Bool-dtype mask check:** `torch.zeros_like(bool_mask)` inherits `dtype=bool`, so
  `masked_fill(-inf)` casts `-inf -> True` and `isfinite` is trivially all-True. Use a float scores
  tensor AND check finiteness **after softmax** (the `-inf` scores are legitimate; softmax turns
  each into a finite 0 unless the whole row is `-inf`).
- Planted bugs A-E (from the bug hunt): (A) mask `diagonal=0` drops `pos_offset` -> decode attends
  only to pos 0 (silent in prefill); (B) `q@k` not `q@k_all` (silent prefill, crashes decode);
  (C) CE shape trap; (D) `arange(S)` not `pos_offset+arange(S)` -> "repeats the prefix forever";
  (E) uncropped context overruns the learned-pos table (crop the INPUT, keep a full accumulator).

## Interview Answers

**Prompt:** *"Model trains fine but generates gibberish — walk me through debugging."*

> "Trains fine" tells me the weights are good, so the bug is in the generation path. First split:
> feed a known training example and compare **teacher-forced** logits vs my **free-running**
> generate loop — if teacher-forced is correct and free-running is garbage, it's 100% the
> generation machinery, not the model. Then a cheap exclusion: swap greedy/beam for nucleus
> sampling to rule out expected degeneration. Then localize with **cached==uncached** and walk the
> mismatch checklist: `eval()`/`no_grad`, positional offset in decode, context cropped to `T`, KV
> append/order/`q@k_all`, and preprocessing/tokenization skew between train and inference.

**Prompt:** *"Attention output is NaN — enumerate causes fastest-first."*

> First, step 0 or step N? Step-0 is config/forward (rules out LR); step-N means something grew.
> Fastest-first: (1) fully-masked softmax row (0/0) — the #1 cause; (2) fp16 overflow -> Inf ->
> Inf-Inf=NaN (fix bf16 / fp32 softmax); (3) LN/RMS eps=0 division; (4) exploding grads at step N
> (clip, warmup); (5) bad inputs / NaN in data / target id >= V.

## Reusable Artifact

- Path: `code/debug_lab.py` (regression suite + planted-bug teeth check),
  `day-05-generation-debug-lab/work.ipynb` (generation demo notebook).
- How to reuse: drop the six invariant tests into any decoder-only codebase's CI. The
  causal-leakage and cached==uncached tests are weights-independent, so they run pre-training and
  catch the highest-severity bug classes (leakage, cache/position mismatch) instantly.
- Run: `./.venv/bin/python ai-lab-mle-prep/week-02-transformers-generation/code/debug_lab.py`
  or `./.venv/bin/pytest .../code/debug_lab.py -q`.
