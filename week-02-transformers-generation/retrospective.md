# Week 02 Retrospective - Transformer Internals & Efficient Generation

## What Clicked

- The **residual-stream + two-sublayer** mental model, and **pre-LN as a gradient highway** (the
  additive `+1` that a product of small Jacobians can't kill). Derivations were consistently strong.
- **Attention as soft differentiable retrieval**; the `1/√d_k` STD-normalization argument (not `d_k`);
  softmax saturation = zero Jacobian = vanishing gradient.
- **RoPE relative-position invariance** (`<rope(q,m),rope(k,n)>` depends only on `m−n`) — proven numerically.
- **KV-cache memory math** — went from a wobbly derivation (double-divide, dropped `n_heads`, unit slip
  on D4) to landing `2·L·n_kv·d_head·T·B·bytes` cold, with the "KV dwarfs weights" insight.
- **The four-contract debug map** — unified the whole week's bug taxonomy into a derivable table.

## What Was Confusing

- **Decoding idioms** (D3) needed heavy line-by-line Q&A: advanced-indexing vs scatter, the `-1:`/`None`
  broadcast axis, `>=` vs `>` at the nucleus boundary, batched-beam coupling. Concepts were fine;
  the tensor plumbing was the friction.
- **Arithmetic-intensity / roofline** phrasing — had the symptom (GPU idles on weight loads) but
  under-stated the *mechanism* (≈1 op/byte; batching amortizes weight streaming) on first pass.
- **Attention-NaN enumeration** (D5 Drill 2) — the one block that came out as guesses instead of a
  structured fastest-first checklist. Clear soft spot.

## What Needs Review

- [ ] Attention-NaN checklist COLD: temporal-clue → masked-row → fp16-overflow → eps → grads → data.
- [ ] Roofline/arithmetic-intensity as a *mechanism*; BW levers (batch, GQA, quant, speculative decode).
- [ ] RoPE base-rescaling (NTK/YaRN) for long context.
- [ ] Beam length-norm edge case (no-op when all beams equal length; only bites with EOS).
- [ ] Masked-mean static form `(x*mask).sum(-1)/mask.sum(-1)` (carried over from W1, still worth a rep).

## Interview Takeaways

- Lead debugging answers with the **discriminating observation** that halves the search space
  (teacher-forced vs free-running; NaN step-0 vs step-N), THEN enumerate fastest-first.
- The two invariant tests (**causal-leakage**, **cached==uncached**) are the strongest "I'd write the
  regression test" signal — and they're weights-independent, so they run pre-training.
- **A passing test can still be wrong**: surfaced two test-design bugs this week (toothless cache test
  on a position-wise task; bool-dtype mask check). Naming *why* a test is (in)sensitive is senior signal.

## Discipline Notes

Kept the 6-block 2-hour structure across all 5 days. D5 ran long due to genuinely valuable tangents
(teacher-forcing vs free-running, overfit-one-batch-then-greedy, the toothless-test investigation) —
those were worth it. The stubs-plus-self-check-harness pattern continues to work well: the learner
implements, bugs surface live, and fixing them (especially the test-design bugs) is where the deepest
learning happens. Next week (RL foundations) is the other declared top gap — plan to lean hard on
first-principles derivations (policy gradient, baselines/advantage, GAE) with the same quiz-first cadence.
