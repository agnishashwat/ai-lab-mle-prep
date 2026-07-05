# Week 02 Day 05 — Generation Debug Lab

## Objective

Convert the week's transformer knowledge into *debugging reflexes*. Diagnose and fix the canonical
transformer/generation bugs — masking leaks, shape mismatches, NaNs, positional/off-by-one errors,
and KV-cache divergence — from symptoms, fast. This is the week's capstone and completes the
annotated decoder-only mini-implementation + generation notebook.

## Prerequisites

- Days 1-4: attention, `TinyGPT`, decoding, KV cache — all in `week-02-.../code/`.
- Week 1 debug habits: "NaN at step 0 = config bug; NaN after stable = something grew"; verify the
  eval/measurement pipeline before diagnosing the model.

## Required Reading

- None new. Re-skim your own Day 1-4 `deliverable.md` notes.

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Days 1-4)

1. KV-cache memory formula + what makes it explode.
2. Why decode is memory-bandwidth bound.
3. Top-k vs top-p; temperature limits.
4. Pre-LN vs post-LN; the \(1/\sqrt{d_k}\) argument.

### 0:10-0:35 - Concept Explanation (Symptom → Cause Map)

Build the transformer debugging lookup table:

- **Loss won't drop / model "cheats" to ~0 too fast** → causal mask missing or wrong (future leakage);
  the model sees the answer. Test: the causal-property check.
- **NaN at step 0** → init scale / LR too high / no warmup / bad LN placement (post-LN deep) / fp16
  without loss scaling. **NaN after stable steps** → grew: check grad norms, a `-inf` row fully masked
  (all-`-inf` softmax → NaN), or attention overflow (accumulate softmax/LN in fp32).
- **Good train loss, garbage generations** → train/inference mismatch: wrong positional offset during
  incremental decode, KV cache not appended/ordered correctly, or eval sampling params.
- **Cached ≠ uncached output** → off-by-one in position ids, or Q attending to a mis-indexed cache.
- **Repetitive/degenerate text** → greedy/low-temp on open-ended gen (decoding, not a bug).
- **Shape bugs** → the Week-1 `cross_entropy` `(B,T,V)` vs `(B,T)` silent trap when `T==V`; head
  reshape `(B,T,H,d)` transpose errors.

### 0:35-1:00 - Deconstruction (Guided Bug Hunt)

Given a set of **planted-bug variants** of the Day-1-4 code (provided as `buggy_*.py` snippets or
described symptoms), for each: state the symptom, form a hypothesis, name the *one* matched fix, and
the test that would catch it. Emphasize minimal-diff fixes and a test-per-bug.

### 1:00-1:45 - Coding / Experiment Task

`code/debug_lab.py` — a regression test suite that would have caught this week's bug classes on
`TinyGPT`:

- `test_causal_no_leakage` — editing token `>t` must not change logits at `t`.
- `test_cached_equals_uncached` — KV-cache path is token-identical to the naive path.
- `test_no_nan_forward_backward` — one train step on a batch produces finite loss and grads.
- `test_masked_row_not_all_inf` — no fully-masked softmax row (would NaN).
- `test_shape_contract` — logits `(B,T,V)`; `cross_entropy` uses `reshape(-1,V)`/`reshape(-1)`.
- `test_overfit_one_batch` — reuse Week-1's marquee pipeline sanity check on the real model.

Then take one planted bug, watch the relevant test fail, fix it, watch it pass. Wire everything into
the **generation notebook** (`work.ipynb`): load `TinyGPT`, show attention maps, sample with each
decoding strategy, and demonstrate the cached==uncached equality + a NaN repro/fix.

### 1:45-1:55 - Interview Drill

"Your model trains fine but generates gibberish at inference — walk me through your debugging."
(train/inference mismatch checklist) and "Attention output is NaN — enumerate causes fastest-first."

### 1:55-2:00 - Deliverable, Week Wrap & Next Week

`deliverable.md`: the symptom→cause table + the regression suite + notebook. Then write
`week-summary.md` + `retrospective.md` for Week 2. Next week: **RL foundations for post-training**
(MDPs, policy gradient, baselines/advantage, GAE, KL-constrained RL).

## Checkpoint Questions

1. What single test catches a causal-mask leak, and why is leakage so dangerous?
2. NaN at step 0 vs after N stable steps — different hypothesis sets; list them.
3. Name three distinct causes of "trains fine, generates garbage."
4. How would a fully-masked attention row produce NaN, and how do you prevent it?
5. Which of this week's failure modes are bugs vs expected behavior (e.g. repetition)?

## Frontier-Lab Interview Angle

Debugging a large ML codebase from symptoms is *the* on-the-job signal frontier labs screen for.
A crisp symptom→cause map plus "I'd write the regression test that catches it" demonstrates exactly
the trustworthy-iteration mindset they want. This is the strongest interview-story block of Week 2.
