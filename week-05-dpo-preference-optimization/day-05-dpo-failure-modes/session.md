# Week 05 Day 05 — DPO Failure Modes, and the Week's Trainer (Week-5 capstone)

## Objective

The capstone. Build the week's deliverable — a from-scratch DPO trainer with **\(\beta\) and
data-quality ablations** — and use it to demonstrate, on ≥3 seeds, the four failure modes
this week derived. Then write the DPO half of the "when would you not use PPO" answer with
numbers in it.

Two structural commitments carried from Week 4, both of which paid:

**(a) A chained regression proving the week is one code path.** W4 D5 recovered REINFORCE →
+baseline → +critic → GAE → PPO → KL-shaping from a single implementation by flipping
switches, elementwise to \(7.749\times10^{-15}\). Week 5's analogue: **BT-fit-then-tilt →
DPO → cDPO → IPO → SimPO recovered from one code path**, with the D2 equivalence
(DPO \(\equiv\) tilt of the BT MLE) as the anchor at one end and D4's grid at the other.

**(b) A `diagnose` built from mechanisms, not thresholds** — and this time built with W4 D5's
two calibration findings fixed from the start: **no `rel()` divided by a first value that can
be exactly 0**, and **no threshold sitting inside the range a healthy run occupies.** Both
bugs are in last week's file; port the function and fix them as the first task, so the fix is
a diff and not a memory.

The four failure modes, each with a mechanism already derived this week:

1. **Divergence / rejected-logprob collapse** — Ford (D1), measured (D2). Silent: accuracy
   1.0, loss → 0.
2. **Both logprobs falling** — the unconstrained marginal (D2) plus \(\mu\ne\pi_{\text{ref}}\)
   (D3).
3. **Length exploitation** — the extensive implicit reward with **no resistance term** (D3);
   worse than PPO's case, where the resistance existed but was bounded (W4 D4).
4. **Off-distribution drift the loss cannot see** — the constraint is evaluated at dataset
   points only (D2). This is DPO's version of W4 D4's *over-optimization* box, and it is the
   one with **no instrument in the training logs at all**, which is the point.

Reuse W4 D4's split explicitly: **objective-wrong (1, 3, 4) vs optimizer-broken (2)** —
identical on a loss curve, opposite fixes.

## Prerequisites

- The whole week. W4 D4 (failure taxonomy, seeds, pessimism) and W4 D5 (trainer structure,
  checkpoint selection, `diagnose`, the ablation grid).

## Required Reading

None new. Re-read your own W4 D5 `deliverable.md` §"Interview answers", specifically answer 3.
You are going to rewrite its DPO paragraph with this week's numbers.

## 2-Hour Plan

### 0:00-0:10 — Cold recall of the week

Numbers on the page before prose, every time.

1. Ford's condition; the noiseless corollary; what it predicts about a DPO dataset.
2. The \(\varepsilon\)-cap and cDPO's optimal margin — the same number from two sides. Values
   at \(\varepsilon=0.1\).
3. `tilt(cr, cβ)` and the annotator-agreement consequence.
4. Where the KL went, and on which distribution DPO's implied constraint is evaluated.
5. The 2-axis variant grid, from memory.
6. Perennial, final ask of the week: the REINFORCE variance sources.

### 0:10-0:22 — Recall quiz (graded, whole week)

1. The three conditions for DPO ≡ RLHF; your measured agreement; which condition a real setup
   violates first and what its signature is in the logs.
2. Both logprobs fall: two causes, distinguishing experiment, your measured numbers.
3. Why DPO is more exposed to length than PPO — and why PPO's protection was insufficient
   anyway (cite W4 D4's bounded-resistance result and \(\kappa\)).
4. IPO's motivation in the language of Ford's theorem; its cost; the fixture where the cost
   appears.
5. **Re-ask, W4 D5 drill 3, final form: "when would you not use PPO?"** Now with a week of
   DPO numbers behind it. This is the third ask; it should be complete.

### 0:22-0:40 — Concept: what a DPO instrument panel must contain

W4 D5's rule was **instruments in pairs**, because a single number cannot distinguish
"working" from "hacking." Build DPO's panel the same way, and name for each pair the failure
it discriminates:

- **chosen logp** vs **rejected logp** — *never the margin alone*. The margin is the one
  quantity that looks identical under "learning the preference" and "destroying the rejected
  completion."
- **implicit-reward magnitude** vs **its closed-form cap** (\(\frac1\beta\log\frac{1-\varepsilon}\varepsilon\)
  from your measured \(\varepsilon\)). Exceeding the cap means you are fitting past what the
  data can support — a *statistical* statement, available with no held-out set.
- **exact KL (or a held-out proxy)** vs **training accuracy**. Accuracy 1.0 with rising KL is
  mode 1 and is otherwise invisible.
- **mean generated length** vs **length-controlled win rate**. W4 D4's result stands: raw win
  rate cannot answer, because length correlates with quality even when explicitly penalized
  (measured marginal slope **+0.1751** against a formula penalty of **−0.4/token**).
- **on-dataset margin** vs **on-policy margin** (score the model's *own* samples with the
  implicit reward). This pair is the only instrument for mode 4, and it is exactly what DPO's
  training loop does not compute — you have to add it deliberately.

Then the honest inventory, said unprompted (W4 D5's habit): what this toy has and what it does
not.

### 0:40-1:00 — Derivation / design block

1. **Checkpoint selection for DPO.** W4 D5: gold peaked at iter 8 of 40 and shipping the last
   checkpoint cost **1.2522**; early stopping is the *estimator*, not regularization. DPO has
   no KL ladder to walk, so derive the analogue: what is the monotone quantity you checkpoint
   *along*? (Candidates: implicit-reward magnitude; exact/proxy KL; epoch count. Argue for
   one and say why the others fail.)
2. **What \(\beta\) means here, precisely.** In PPO, \(d\mathbb E[r]/d\mathrm{KL}=\beta\).
   In DPO there is no such envelope statement because there is no constrained problem being
   solved online — \(\beta\) instead sets the **scale at which the implicit reward saturates**
   the link. Derive the relationship between \(\beta\), the converged margin, and the KL
   actually reached, and connect it to D1's scale identity: **\(\beta\) and annotator
   agreement are the same knob, so a \(\beta\) sweep and a data-quality sweep should trace the
   same curve.** That is a *prediction*, and the ablation grid tests it.
3. **Why pessimism/ensembling does not transfer.** W4 D4: use a **minimum**, not a mean, over
   an RM ensemble — an LCB, because \(\mathbb E[\max]>\max\mathbb E\). DPO has no explicit RM
   to ensemble. State what the analogue would have to be, and whether it exists.

### 1:00-1:10 — Interview Drill

1. **"When would you not use PPO?"** Final form, graded hard, with this week's numbers.
2. **"Your DPO model wins on your preference eval and loses in production. Diagnose."**
   Wants mode 4 named as *structurally invisible to the training loop*, the on-policy-margin
   instrument, length control, and the fact that the preference eval is drawn from the same
   \(\mu\) as the training data.

### 1:10-1:50 — Coding: `code/dpo_lab.py` (the week's deliverable)

**Port first (5 min):** copy `week-04-rlhf-ppo/code/failure_modes.py::diagnose` and fix its
two known calibration bugs before writing anything new — the zero-denominator `rel()` and the
−20% entropy threshold that fires on healthy runs.

**TODOs:**
1. `make_dataset(rng, gold_r, sampler, n_pairs, eps, length_confound)` — one function, four
   knobs, so every failure mode is a fixture rather than an anecdote.
2. `train_dpo(cfg)` — the trainer, logging the full panel above, with a `variant` switch
   wired to D4's grid.
3. `diagnose_dpo(history)` — the four modes, each firing on its own mechanism, each with a
   **reason string carrying the number that fired it** (D4's rule: the reason string is the
   deliverable).
4. `select_checkpoint(history, gold_eval)` — the estimator.
5. `ablate(grid, seeds)` — \(\beta\) sweep × \(\varepsilon\) sweep × variant, ≥3 seeds,
   mean ± spread.

**Checks with teeth:**
- **The chained regression.** BT-fit-then-tilt → DPO → cDPO(\(\varepsilon\)) → IPO(\(\tau\))
  → SimPO recovered from one code path; the DPO ≡ tilt anchor to \(10^{-10}\), and each rung
  reproducing D2/D3/D4's independently-written implementations elementwise. Assert
  **non-vacuity** the way W4 D5 did — show the rung that would look identical if a switch
  were ignored, and show it is not identical.
- **All four modes fire, on their own fixtures, and are SILENT on the healthy one.** The
  silence is the harder half and is the check most likely to be missing.
- **The \(\beta\)/agreement collapse prediction.** Assert that a \(\beta\) sweep at fixed
  \(\varepsilon\) and an \(\varepsilon\) sweep at fixed \(\beta\) trace the **same** curve in
  (implicit-margin, KL) space, to a stated tolerance. If it holds, it is the week's cleanest
  single result; if it fails, the *deviation* is the finding and must be explained.
- **Seeds.** ≥3 everywhere, report mean ± spread, and check W4 D4's amplification claim in
  this setting: does seed spread grow with optimization pressure here too? Report the factor.
- **Held-out gold.** A gold evaluation independent of the preference data, as in W4 —
  otherwise every number this week is measuring the proxy.
- **The inversion, if it exists.** W4's headline was proxy up / gold down. Look for DPO's
  analogue (training accuracy up, gold down) and report it *or* report that it did not appear
  and why — a negative result stated honestly is worth more than a manufactured one.

### 1:50-2:00 — Deliverable, week-summary, retrospective

`deliverable.md`, `week-summary.md`, `retrospective.md`. Rewrite the DPO paragraph of W4 D5's
answer 3 with this week's measurements. Next week: **Week 6 — GRPO, RLVR and reasoning**,
where the group baseline removes exactly the per-prompt constant \(c(x)\) that Bradley-Terry
could never identify — the thread that started on W4 D1 and closes there.

## Checkpoint Questions

1. Give DPO's instrument panel in pairs, and say which failure each pair discriminates. Which
   mode has no instrument in a standard training loop?
2. What do you checkpoint *along* in DPO, given there is no KL ladder? Defend the choice.
3. State the \(\beta\)/annotator-agreement equivalence as a testable prediction, and report
   whether your grid confirmed it.
4. Which of the four failure modes are objective-wrong and which are optimizer-broken? Give
   the opposite fixes.
5. "When would you not use PPO?" — full answer, with numbers.

## Frontier-Lab Interview Angle

The capstone artifact is the deep-dive: a from-scratch DPO trainer whose failure modes are
*derived* rather than catalogued, whose \(\beta\) ablation is predicted by an identifiability
argument, and whose diagnostic panel was designed from mechanisms. Two things to lead with in
an interview: the theorem that DPO's optimum is at infinity for typical preference data, and
the measurement that a \(\beta\) sweep and a data-quality sweep trace the same curve. Both are
statements about preference *data*, which is where the field's attention is moving.
