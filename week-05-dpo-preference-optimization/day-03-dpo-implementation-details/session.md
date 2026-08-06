# Week 05 Day 03 — DPO Implementation Details: Length, the Reference, and Everything That Is Not in the Loss

## Objective

Day 2 gave the loss. Today is the gap between the loss and a training run that works, and
every item in the gap is something you have already met from another angle:

**(a) Sum vs mean is a length prior — fourth sighting.** W3 D3's `reduce`, W4 D3's
`reduce="mean"` broadcast bug, W4 D4's "PPO uses sum over tokens, mean over sequences ⇒ no
built-in brake." DPO's \(\log\pi_\theta(y|x)=\sum_t\log\pi_\theta(y_t|\cdot)\) is a **sum**,
so the implicit reward is extensive in length. Derive the bias, measure it, and connect it to
the fact that SimPO's entire contribution (Day 4) is dividing by \(|y|\).

**(b) The reference model, and the three ways to not have one.** Cache logprobs at
preprocessing time (\(\pi_{\text{ref}}\) is frozen and the completions are fixed — unlike
PPO, where they are *not* fixed, which is why PPO must cache at rollout time). LoRA-with-the-
adapter-off. Or delete it entirely and change the algorithm (ORPO/SimPO). Quantify the memory
each buys against W4's 16-vs-2 bytes/param bill.

**(c) Both logprobs fall — the mechanism, in code.** Day 2 derived that nothing constrains
the marginal. Today measure it on a real tokenizer-shaped setup, decompose the margin into
its two terms, and show the standard fix (SFT on chosen first, or an added NLL term) moving
each term separately.

**(d) Label smoothing is D1's \(\varepsilon\)-cap put back by hand.** Conservative DPO
replaces the target label 1 with \(1-\varepsilon\), which turns the loss into a mixture and
**restores a finite optimum**. Derive the smoothed loss, find its optimal implicit margin in
closed form — it is exactly \(\operatorname{logit}(1-\varepsilon)/\beta\), i.e. D1's cap
divided by \(\beta\) — and verify numerically. This is the day's headline: *the fix for
DPO's divergence is to re-inject the annotator noise you removed.*

**(e) Numerics.** `logsigmoid` rather than `log(sigmoid(...))`; why the difference of two
sums of \(T\) logprobs loses precision; why the reference logprobs must be computed in the
same precision and with the same attention implementation as the policy's, or the margin is
dominated by kernel noise.

## Prerequisites

- W5 D1 (the \(\varepsilon\)-cap), W5 D2 (the loss, the gradient, the assumption audit).
- W4 D4's lengthening analysis: \(\Delta = \kappa + \Delta r_{\text{content}} -
  \beta\mathbb E[\mathrm{KL}_t]\) and the finding that a *bounded* per-token resistance
  cannot stop a length confound.
- W1 D3 (bf16/memory), W2 D2 (numerical stability, fp32 reductions).

## Required Reading

- **DPO §5 and the experimental setup appendix** — specifically the \(\beta\) values used and
  the "we initialize from an SFT model trained on the chosen completions" sentence. You will
  be able to say why that sentence is load-bearing.
- Short: the **cDPO / conservative-DPO** note (label smoothing for DPO), and the
  **SimPO** length-normalization section — skim only; Day 4 does variants properly.

## 2-Hour Plan

### 0:00-0:08 — Cold recall

1. The DPO loss, cold. Then: where did the KL go, and on which distribution is the implied
   constraint evaluated?
2. The DPO gradient's weight, and the two consequences.
3. D1's \(\varepsilon\)-cap, value at \(\varepsilon=0.1\). **Number first.**
4. Your D2 numbers: the DPO-vs-tilt agreement, and the KL at which training accuracy hit 1.0
   on the matching fixture. Both written before speaking.
5. Perennial: the REINFORCE variance sources you still have not named.

### 0:08-0:18 — Recall quiz (graded, D1-D2)

1. Ford's condition + the noiseless corollary + what it predicts for DPO.
2. The three conditions for DPO ≡ RLHF, and which one a real setup violates first.
3. Why do both chosen and rejected logprobs fall? Give the term that is missing from the
   gradient, and the two independent causes (marginal unconstrained; data off
   \(\pi_{\text{ref}}\)).
4. `tilt(cr, cβ)` and the annotator-agreement consequence.
5. Sum-vs-mean as a length prior: state the rule for when lengthening pays under each, in
   symbols. (Third re-ask of this family.)

### 0:18-0:40 — Concept

- **Length, properly.** \(\hat r(y)=\beta\sum_t\log\frac{\pi_\theta(y_t|\cdot)}{\pi_{\text{ref}}(y_t|\cdot)}\).
  Every token contributes; a token on which policy and reference agree contributes exactly 0,
  so the *drift* is what accumulates — but drift has a sign that correlates with length
  through the data. Derive the condition under which a longer chosen completion is rewarded
  purely for being longer, and connect to W4 D4's measured \(\kappa\): the mechanism is the
  same confound, the difference is that DPO has **no \(\beta\cdot\mathrm{KL}_t\) resistance
  term at all**, because there is no per-token penalty anywhere. ⇒ **DPO is *more* exposed to
  length exploitation than PPO, not less.**
- **The reference model.** Frozen, inference-only, and — crucially — evaluated on a **fixed**
  set of completions. So its logprobs are a *preprocessing artifact*: compute once, store as
  two floats per example, never load the model during training. Contrast PPO, where
  completions are generated fresh each iteration so reference logprobs must be computed in
  the rollout loop (and *then* can be discarded, per W4 D5). Memory arithmetic: DPO's
  training-time footprint is one trainable model, i.e. **~16 B/param**, versus PPO's four
  models. This is the concrete content of "DPO deletes three of the four."
- **The initialization coupling.** \(\pi_{\text{ref}}\) appears in the loss and
  \(\pi_\theta\) is initialized *at* \(\pi_{\text{ref}}\), so the margin starts at exactly 0
  and the gradient weight starts at exactly \(\sigma(0)=1/2\) for every pair. If you
  initialize elsewhere, the first phase of training is spent on the initialization gap rather
  than on preferences — the Day 2 result, now as an engineering rule.
- **Batching and effective sample size.** All pairs are independent; there is no rollout, no
  epoch staleness, no \(\theta_{\text{old}}\), no clipping, and therefore **none of PPO's
  freeze-line hazards.** Say what that buys and what it costs — this is the clean half of
  DPO and it deserves to be stated as clearly as the failure modes.
- **Numerics.** `log σ(z)` = `-softplus(-z)`. In fp32, a sufficiently large *positive*
  \(z\) makes `sigmoid(z)` round to exactly 1, so the naive log becomes exactly 0 and
  erases a small but nonzero loss; a sufficiently negative \(z\) can underflow sigmoid
  and turn a finite NLL into infinity. The threshold depends on dtype. The margin is a
  difference of two \(O(T)\) sums; in bf16 with \(T=512\) the summation error can be
  comparable to \(\beta\times\) the margin you are trying to measure ⇒ accumulate
  logprobs in fp32, and compute reference logprobs with the identical attention kernel.

### 0:40-1:00 — Derivation: conservative DPO, and the optimum that exists

1. **The smoothed loss.** With label noise \(\varepsilon\), the correct likelihood is a
   mixture: \(-(1-\varepsilon)\log\sigma(\beta\hat\Delta) - \varepsilon\log\sigma(-\beta\hat\Delta)\).
2. **Its minimizer.** Differentiate in \(\hat\Delta\), set to zero, and solve:
   \(\sigma(\beta\hat\Delta) = 1-\varepsilon\), i.e.
   \(\hat\Delta^* = \frac{1}{\beta}\log\frac{1-\varepsilon}{\varepsilon}\). **Finite for
   every \(\varepsilon>0\), infinite at \(\varepsilon=0\).** This is D1's cap, divided by
   \(\beta\), reached from the policy side. Say the unification out loud: *D1 measured the
   cap that noisy data imposes on a reward model; cDPO imposes the same cap by declaring
   noise that may not exist.*
3. **Second derivative / behaviour past the optimum.** Show the smoothed loss is *increasing*
   in \(\hat\Delta\) beyond \(\hat\Delta^*\) — so it does not merely slow divergence, it
   **reverses** it. Contrast with adding weight decay on \(\hat r\), which slows but does not
   reverse, and with IPO (Day 4), which achieves the same by bounding the loss instead of
   the label.
4. **Choosing \(\varepsilon\).** It should be your measured annotator disagreement rate. If
   you have adjudicated your data to 91% agreement, \(\varepsilon=0.09\) and the cap is
   \(2.31/\beta\). Full circle to D1's drill 2.

### 1:00-1:10 — Interview Drill

1. **"Your DPO run has rising accuracy, falling loss, and the outputs are getting worse.
   Walk me through your diagnosis."** Wants: plot the two logprobs *separately*, not the
   margin; check the implicit-reward magnitude against the cap; check length; check whether
   the pairs came from \(\pi_{\text{ref}}\); and the three fixes ordered by what they cost.
2. **"Why does DPO need an SFT stage on the chosen completions first?"** Wants the
   \(\mu\ne\pi_{\text{ref}}\) argument from Day 2 with today's measurement attached.

### 1:10-1:50 — Coding: `code/dpo_trainer.py`

A real DPO trainer on the Week-2 `TinyGPT` (your own model, from `week-02/code/`), on a
synthetic preference task with a **controllable length confound and a controllable annotator
noise rate** — the two knobs the day is about.

**TODOs:**
1. `sequence_logprob_sum` and `sequence_logprob_mean` — one flag, two length priors, and a
   check that they differ only by \(1/|y|\).
2. `precompute_reference_logprobs(model_ref, dataset)` — the caching path; assert the trainer
   never touches `model_ref` afterwards.
3. `dpo_loss(policy_logps, ref_logps, beta, label_smoothing, length_normalize)` — one
   function covering vanilla DPO, cDPO, and length-normalized DPO, implemented with
   `logsigmoid`.
4. `train(...)` with per-step logging of: chosen logp, rejected logp, margin, implicit-reward
   magnitude, mean generated length, and (on a held-out enumerable subset) exact KL.
5. `sft_warmup(...)` — the fix, so it can be ablated rather than assumed.

**Checks with teeth:**
- **`logsigmoid` vs naive, with dtype stated**: in fp32, assert the naive form returns
  exactly 0.0 at \(z=+40\) while stable `logsigmoid` preserves the small negative value;
  at \(z=-100\), assert the naive NLL is infinite while stable `-logsigmoid` is 100.
- **The length prior without a brittle training-curve assertion.** On a controlled
  per-token-drift fixture, assert `sum` mode has the predicted positive length slope and
  `mean` mode has zero slope. Log generated length during TinyGPT training, but do not
  require it to be monotone under every seed.
- **cDPO's finite optimum.** Sweep \(\varepsilon\); assert the converged implicit margin
  matches \(\frac1\beta\log\frac{1-\varepsilon}{\varepsilon}\) within tolerance at each, and
  that \(\varepsilon=0\) does not converge. **Same sweep shape as D1's, from the other side —
  print them in one table.**
- **Both logprobs are instrumented; mismatch uses the stable diagnostic.** Log chosen and
  rejected logprobs separately, but do not assert either sign: Day 2 showed both can fall
  even when \(\mu=\pi_{\text{ref}}\), and the sign can be transient. Reuse the measured
  \(5.11\times\) KL mismatch tax; assert instead that `sft_warmup` raises chosen-data
  likelihood and that cloning the warmed checkpoint into both policy and reference
  restores exactly zero initial log-ratios.
- **Reference caching is exact**: cached logprobs equal freshly-computed ones to \(0.0\), and
  fp32 accumulation vs bf16 differs by a reported amount at \(T=256\) — with the second
  number compared against \(\beta\times\) the typical margin, so the reader can see whether
  it matters.
- **Boundary regressions, corrected.** At \(\beta\to0\), DPO's gradient with respect to
  the log-ratio margin goes to 0. At `label_smoothing=0.5`, the target is a tie: the loss
  is minimized at zero margin and increases away from it; it is not constant.

### 1:50-2:00 — Deliverable & Next Session

`deliverable.md` with the two sweeps in one table, the length slopes, and the caching
arithmetic. Next: **Day 4 — variants**, where IPO, KTO, ORPO and SimPO are each read as *a
specific answer to one of this week's three problems*: unbounded loss, missing reference, or
length prior.

## Checkpoint Questions

1. Why is DPO's implicit reward extensive in length, and why does DPO have *less* resistance
   to length exploitation than PPO? Give the missing term.
2. Derive cDPO's optimal implicit margin. What is it at \(\varepsilon=0.1,\beta=0.1\)? How
   does it relate to D1's cap?
3. Give three ways to avoid holding a reference model in memory and what each costs.
4. Both logprobs are falling. Name the two independent causes and the experiment that
   distinguishes them.
5. Why must reference logprobs be computed with the same precision and kernel as the
   policy's? Give the magnitude of the error at \(T=512\) versus \(\beta\times\) margin.

## Frontier-Lab Interview Angle

This is the block of questions that separates "read the DPO paper" from "shipped a DPO run."
The length prior, the both-logprobs-fall diagnosis, the reference-caching trick, and the fact
that label smoothing has a closed-form optimum that equals the annotator-noise cap are all
things an interviewer can probe for thirty minutes. The one-table comparison of D1's
data-side cap and D3's loss-side cap is the artifact to have.
