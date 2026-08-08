# Week 05 Retrospective — Preference Optimization: DPO and Variants

## What Clicked

- **Re-scoping Day 1 from "fit a Bradley-Terry model" to "what can preference data identify?"**
  was the highest-leverage decision of the week, and everything downstream inherited it. W4 D1
  had already fitted a BT reward model; re-running it would have produced a second copy of the
  same file. Asking the *identification* question instead — infinite data, a perfectly specified
  tabular model, what is still unknown? — produced Ford's condition on Day 1 and then explained
  Days 2, 4 and 5 without any new machinery. **DPO's divergence is not a DPO fact; it is a
  statement about the wins digraph, and it was already provable on Monday.**
- **Three degrees of freedom, not one — and the flat/ascent distinction.** Location invariance
  and separability *look* like the same kind of non-identifiability and they are not. Flat
  means the likelihood is constant along the direction: infinitely many finite maximisers,
  fixed by gauge-fixing. Ascent means the likelihood strictly increases forever: **zero**
  maximisers, and centring cannot help. That distinction is worth more than the theorem.
- **The 5×5 grid.** Collapsing four papers into `(reward f) × (link ℓ)` and then asserting
  the grid reproduces five hand-written implementations **elementwise at max deviation
  0.000e+00** turned "the variants differ in principled ways" from a slogan into a check. And
  **occupying an empty cell** (`ref_mean × squared_target`) is what proves the grid is a
  *mechanism* rather than a taxonomy — it inherited τ's cap from IPO and the length
  amplification from SimPO, exactly as predicted, with no paper to copy from.
- **Designing a prediction on Day 1 and testing it on Day 5.** `tilt(c·r, c·β) = tilt(r, β)`
  was measured to ~1e-16 on Monday as an algebraic identity. On Friday it became an
  *experimental* claim: a β-sweep and an ε-sweep, sharing no code path, matched only through
  `Δ̂* = logit(1−ε)/β`, land on the same KL to **2.8e-16**. That is the best-designed
  experiment of the five weeks, because the prediction existed before the apparatus did.
- **Modes as FIXTURES, not anecdotes.** One dataset function with four knobs, and each knob
  produces exactly one failure mode. The design rule that made it work: *if a mode needs a
  bespoke dataset builder, you have not understood which knob causes it.* That constraint
  forced real understanding twice — M4 in particular refused to be a delta and had to be
  re-derived as a **level**.
- **Building the diagnostic's negative half.** `diagnose_dpo` is asserted **silent on a
  healthy run**, and M3/M4 are asserted not to fire on fixtures they do not own. This is
  the direct fix for W4 D4's −20% entropy threshold, which sat *inside* the range a healthy
  run occupies. **A rule that fires on everything is not a rule**, and the only way to know
  is to build the control.
- **The healthy CONTROL in the inversion stage.** Without it, "gold peaks early" is a
  statement about training length. With it (healthy peaks at step 1270 and gives back
  **exactly 0.0000**) it is a statement about the *data*. One extra run, and it converted the
  week's headline from an observation into evidence.
- **Two unplanted results on Day 5**, both better than anything designed: the **1/β² law**
  (halving β costs 4× the compute, not 2×) and the discovery that **W4 D4's seed-spread claim
  was a 3-seed artifact**. The second is the more valuable, because it is a correction rather
  than an addition.

## What Was Confusing

- **Which ε.** Two independent ε live in the same file — `DataCfg.eps` (annotator flip rate,
  which sets the cap) and `RunCfg.eps` (cDPO label smoothing). Reading the wrong one makes
  the cap move with the *training* config, so a β-sweep silently sweeps its own target. This
  is the whole WRONG-LEVEL family in one variable name, and it cost real time.
- **Link units vs policy units.** `M = β·Δ̂` shows up everywhere and the two are constantly
  confusable. D4 already hit it (`M* = 2.197225` vs `Δ̂* = 7.324082`, requiring
  `logratio_optimum()` to exist at all); D5 hit it again, and then *measured* the cost:
  across the same β sweep the margin goes **up** while `Δ̂` goes **down**. **Report the wrong
  one and you report the opposite conclusion.** The reflex that would have prevented every
  instance: *any number with a β in it — divide by β and check whether the story flips.*
- **Whether the panel should log levels or ratios.** `logp_w`/`logp_l` as log-ratios feels
  natural (everything else in DPO is a ratio) and is exactly wrong: the difference of the
  ratios equals `margin/β` to **1.39e-16**, so the pair becomes perfectly redundant with the
  instrument it exists to check. The general principle only became clear afterwards: **a
  panel needs at least one level next to every difference**, because the difference is
  precisely what hides the level.
- **That "bounded loss" and "finite minimiser" are different properties.** KTO's `prospect`
  is bounded in `[0, 2.5]` and its margin still diverges *faster* than DPO's. `inf σ(−u) = 0`
  is attained only at `u = ∞`. What boundedness buys is bounded **influence** — a
  noise-robustness property — and reading it as a P1 fix would have been a confident wrong
  answer in an interview.
- **`ndarray` vs list as an accumulator.** Reaching for `history.step.append(...)` when
  `History` holds arrays. The record/builder distinction is obvious once stated and was not
  obvious while writing.

## What Needs Review

- [ ] **Ford's condition, stated cold.** BT MLE finite ⟺ wins digraph strongly connected, and
      *why* that is exactly linear separability of `e_w − e_l`. This is the load-bearing
      theorem of the week and it needs to be reproducible without notes.
- [ ] **The four modes with their instruments and their fixes**, as the 4-row table. Specifically
      that M1 and M3 look identical on a loss curve and have **opposite** fixes.
- [ ] **The numbers.** The D1–D4 recall quiz scored **45%** on Day 5 — the lowest of the week —
      while the code went 92/92. The gap is retrieval of measured results, not mechanism.
      Drill the numbers card, not the derivations.
- [ ] **Why cDPO's cap is `logit(1−ε)` and IPO's is `(2f−1)τ`**, and which one is a statement
      about the *annotators* versus about the *loss*. Only the first needs no held-out set.
- [ ] **KTO's prospect link** — implemented, never stress-tested as its own failure surface.
      The one cell of the grid with an asymmetric value function and no D5 fixture.
- [ ] **The second clause of every question.** Sixth week open.

## Interview Takeaways

- **The one-liner:** *DPO's failure modes are consequences of what the data format can
  identify. The loss is exactly constant along every direction the data do not constrain, so
  the loss going down is not evidence of anything.*
- **When asked "what's wrong with DPO", do not say "it diverges".** Say: the BT MLE is finite
  iff the wins digraph is strongly connected; a real DPO dataset is a **matching**, where
  every completion appears once, so noise has nowhere to act and the graph is a DAG at any
  noise level. The divergence is structural. *Then* give the fix.
- **Never quote pairwise accuracy without a level next to it.** Measured: 1.000 at step 1
  of 6000, while the margin ran 0.02 → 5.84.
- **Distinguish over-optimisation from optimisation pathology, out loud.** The objective being
  wrong and the optimiser going too far look identical on a loss curve and have opposite
  fixes. Having a run where gold goes **+0.75 → −0.45** at 100% train accuracy makes this
  concrete rather than theoretical.
- **Early stopping is an ESTIMATOR.** Checkpoint on the implicit-reward margin against
  `logit(1−ε)` — computable from the training batch, needs **no held-out set**, because the
  cap is a statement about the annotators. Not epochs (accuracy saturates at step 1); not KL
  (bounded, so it saturates while the reward diverges, and one mode is invisible in it).
- **Know one thing everyone gets wrong.** Good candidates from this week: IPO *compresses*
  rather than caps; SimPO's `1/|y|` *amplifies* length bias; a bounded loss is not a finite
  minimiser; and more preference data gives you a *smaller* margin, not a bigger one.

## Discipline Notes

**The reference-implementation rule paid for the fifth and sixth consecutive time**, and on
Day 5 it caught something new: **two of my own fixtures were wrong**. M4's did not exhibit M4
under a delta rule (off-data mass *falls* during training), and M3's had gold *rising* on the
original seed. Both would have surfaced as "his code is broken". **Always write and run a
private reference of every TODO before handing the file over** — it is now the single most
reliable process rule in this repo.

**What did not work: skeleton-first did not transfer.** D4's headline intervention — *write
the numbered skeleton with blank slots first, then fill each slot* — produced a 3.7× swing on
an identical question. On D5 it was not used unprompted, and the recall quiz fell to 45%.
Either it is a **prompted** technique that must be re-issued every session, or it only works
under retrieval-reset conditions. **W6 D1 should test this directly:** issue it explicitly in
Block 0, withhold it in Block 1, compare. That is cheap and decisive, and guessing is worse
than measuring.

**Coding ownership drifted, third consecutive session.** D3 and D4 the request was for loop
plumbing; D5 it extended to three whole functions. The split that emerged is real and mostly
defensible — **he writes the mathematical core and delegates the scaffolding** — and the core
he wrote was excellent (`make_dataset` went 20/20 and reproduced every calibrated number
bit-for-bit). But *design* skills are the ones being delegated: thresholds with air on both
sides, negative controls, dispatch tables. Week 6 should hand him one **diagnostic** TODO
specifically, since that is where the skipped practice lives.

**The structure held.** All five days ran the full block sequence, and the "give FULL blocks,
never compress for the clock" rule was obeyed throughout. The one change worth making for
Week 6: **open with a numbers-card drill instead of a cold-recall quiz.** Five weeks of
evidence say recognition works where retrieval does not, and Day 5 produced the cleanest
version of that evidence yet — 45% on recalling the week's own results while building a
92-check lab correctly.
