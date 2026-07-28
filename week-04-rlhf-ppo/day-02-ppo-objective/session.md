# Week 04 Day 02 — The PPO Objective: Ratio, Clipping, Epochs

## Objective

Derive PPO as the answer to one question: **how many gradient steps can I take on data I
already collected?** Everything else — the ratio, the clip, the epochs, the minibatches —
falls out of that. Three results to own by the end:

1. \(\nabla_\theta L^{\text{IS}}\big|_{\theta=\theta_{\text{old}}} = \nabla_\theta J\).
   The importance-sampled surrogate is **not a new algorithm**; at the first step it *is*
   Week 3's policy gradient. PPO is REINFORCE-with-advantages plus permission to keep going.
2. \(L^{\text{CLIP}}\) is a **pessimistic lower bound** on the surrogate, and its gradient
   is exactly zero in the half of each clipped case where the update would push further
   out. Get the 2×2 case table cold.
3. Clipping is an **implicit, local, per-token trust region**; D5's KL penalty is an
   **explicit, global, sequence-level** one. They are not substitutes, and today you
   *measure* a regime where clipping is satisfied and the sequence KL still runs away.
   That measurement is why InstructGPT ships both.

## Prerequisites

- Day 1's `toy_rlhf.py` (`exact_J`, `exact_sequence_kl`, enumeration ground truth).
- W3 D2 (score-function estimator, `episode_gradient`, finite-difference ground truth),
  W3 D3 (advantages), W3 D5 (KL-constrained objective, the \(\beta\) identities).
- Importance sampling; a second-order Taylor expansion; Fisher information (lightly).

## Required Reading

- **PPO (Schulman et al. 2017) §3 only** — Eq. (6)-(7) and Figure 1. Ten minutes. Read
  Figure 1 as the 2×2 case table it is, not as two pictures.
- Optional: TRPO §3 for the constrained problem PPO is approximating — read only far
  enough to see that the constraint is \(\bar{\mathrm{KL}}(\pi_{\text{old}}\|\pi_\theta)\le\delta\),
  a *forward* KL to the old policy, which is a different object from D5's reverse KL to
  the reference. That contrast is a real interview question.

## 2-Hour Plan

### 0:00-0:06 — Cold recall of CONCLUSIONS (Day 1)

One line each, no whiteboard: the BT identifiability statement and its GRPO consequence;
which stage of the pipeline can *teach*; the four models and which are frozen; what your
measured RM-error-vs-KL curve did; the two reasons a prompt's comparisons batch together.

### 0:06-0:16 — Recall Quiz (Day 1, graded)

1. Derive the Bradley-Terry gradient and say what \(\sigma(-\Delta)\) implies about which
   pairs actually train the RM. Where does annotator noise do the most damage, and why?
2. Your `exact_sequence_kl` check compared two routes to the same number. Name both
   routes, say why they agree, and name the Week-3 bug that check was written to catch.
3. **Perennial — off-policy-by-accident.** Same question as yesterday, cold. Today it
   stops being trivia: PPO's ratio is the fix for the *deliberate* version, and by Day 4
   you will have measured the accidental version's gradient error.
4. **Perennial — variance sources.** Seven, with the technique that kills each.

### 0:16-0:40 — Concept: why off-policy at all

- **The economics.** In LLM RLHF a rollout costs a full generation pass over \(T\) tokens
  for a batch of prompts; a gradient step costs one forward+backward. Sampling is
  \(10\)-\(100\times\) the cost of the update. Taking one step per rollout batch (vanilla
  PG) wastes the expensive thing. So: reuse each batch for \(K\) epochs. **The instant you
  take step 2, your data is off-policy** — and that is *deliberate* off-policy, in contrast
  to Day 4's accidental kind.
- **Importance sampling, and where it breaks.** \(\mathbb{E}_{\pi_\theta}[f] =
  \mathbb{E}_{\pi_{\text{old}}}[\rho f]\) with \(\rho = \pi_\theta/\pi_{\text{old}}\).
  Unbiased for any \(\theta\), but \(\mathrm{Var}\) scales with the \(\chi^2\) divergence
  between the two, so it is worthless once they separate. **You already measured this
  shape**: D5's k3 variance win was 423× at KL=0.004 and 4.4× *worse* at KL=0.65, for
  exactly the same reason — \(\rho\) is a likelihood ratio, and ratios blow up when the
  distributions come apart. Say that number out loud here; it is the same phenomenon.
- **The sequence-level catastrophe.** A whole-sequence ratio is
  \(\prod_{t=1}^{T}\rho_t\). With \(T=500\) and each \(\rho_t\) within 1% of 1, the product
  ranges over \(e^{\pm 5}\). It is log-normal, so its *median* collapses while its *mean*
  stays 1 — a handful of samples carry all the weight. **This is the argument for a
  per-token ratio**, and it is the concrete version of the point you have been half-making
  in the off-policy drill.
- **Two ways to keep \(\rho\) near 1.** (a) TRPO: solve
  \(\max L^{\text{IS}}\) s.t. \(\bar{\mathrm{KL}}(\pi_{\text{old}}\|\pi_\theta)\le\delta\)
  — a hard constraint, needs a Fisher-vector product and a line search, second-order and
  awkward. (b) PPO: **give up on solving the constrained problem and instead destroy the
  incentive to leave the region**, by flattening the objective outside \([1-\epsilon,1+\epsilon]\).
  First-order, one line of code, no line search. Say clearly: **PPO is a heuristic that
  approximates a constraint by removing the gradient, not by enforcing a bound.** Nothing
  stops \(\rho\) from being 5; clipping only stops it from being *rewarded* for it.
- **Minibatches.** Within an epoch, the batch is split; each minibatch step moves
  \(\theta\), so \(\rho\ne1\) even *within* epoch 1 after the first minibatch. Two knobs
  multiply into "how off-policy am I": epochs × minibatches per epoch.

### 0:40-1:05 — Derivation

1. **The surrogate and its gradient at \(\theta_{\text{old}}\).**
   \(L^{\text{IS}}(\theta)=\mathbb{E}_{\pi_{\text{old}}}[\rho_t(\theta)\hat A_t]\).
   Show \(\nabla_\theta\rho_t = \rho_t\nabla_\theta\log\pi_\theta\), hence at
   \(\theta=\theta_{\text{old}}\) (where \(\rho_t=1\)):
   \(\nabla L^{\text{IS}} = \mathbb{E}[\hat A_t\nabla\log\pi_\theta]\) — **the Week-3
   policy gradient exactly.** State where the hypothesis \(\rho=1\) is consumed. This is
   also the reason the surrogate's *value* is meaningless (W3: it is a gradient generator,
   not an objective) — at \(\theta_{\text{old}}\) it equals \(\mathbb{E}[\hat A]\approx0\).
2. **The clipped objective, by cases.**
   \(L^{\text{CLIP}}=\mathbb{E}\big[\min\big(\rho\hat A,\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\hat A\big)\big]\).
   Fill the 2×2 table by hand — \(\hat A>0\) vs \(\hat A<0\), crossed with \(\rho\) above
   vs below the band — and for each cell state the value, the gradient, and the intent:
   - \(\hat A>0,\ \rho>1+\epsilon\): clipped, **gradient 0** (stop increasing a good action
     we've already increased enough).
   - \(\hat A>0,\ \rho<1-\epsilon\): **not** clipped by the `min` — gradient flows, pulling
     it back up. Clipping is one-sided per cell.
   - \(\hat A<0,\ \rho<1-\epsilon\): clipped, gradient 0.
   - \(\hat A<0,\ \rho>1+\epsilon\): gradient flows, pushing it down — and note this one is
     *unbounded*, which is why the "\(\hat A<0\) with a large ratio" cell is where PPO
     produces its biggest steps. (This is the cell dual-clip / PPO variants target.)
   Then prove **\(L^{\text{CLIP}}\le L^{\text{IS}}\) pointwise** — the `min` makes it a
   pessimistic bound, which is the honest statement of what "trust region" means here.
3. **Clipping as an implicit KL trust region.** Two steps.
   - \(\log\rho_t = \log\pi_\theta-\log\pi_{\text{old}}\) is exactly the **k1 estimator**
     of \(\mathrm{KL}(\pi_\theta\|\pi_{\text{old}})\) at that token (yours, from D5). So
     \(|\rho-1|\le\epsilon\) bounds \(|k_1|\lesssim\epsilon\).
   - Second order: \(\mathrm{KL}(\pi_{\text{old}}\|\pi_\theta)\approx\tfrac12\Delta\theta^\top
     F\Delta\theta\), and for a single token the per-token KL implied by \(|\rho-1|\le\epsilon\)
     is \(\mathcal O(\epsilon^2)\) — recover this from **k2** \(=\tfrac12(\log\rho)^2\), which
     is where the square comes from. \(\epsilon=0.2\Rightarrow\) per-token KL \(\approx0.02\).
   Now the **three gaps** that make it *not* a real trust region, and each gap is a Day-4
   failure mode: (i) it is **per token**, so \(T\) tokens each at the boundary give a
   sequence KL of \(T\epsilon^2/2\), which grows without bound in \(T\); (ii) it is
   **per sample**, and the *expectation* of the KL is not bounded by a bound that holds
   sample-wise only where the clip binds; (iii) it is relative to \(\pi_{\text{old}}\)
   (this batch), not \(\pi_{\text{ref}}\) (the start), so **satisfying the clip at every
   step is compatible with arbitrary drift** — clipping bounds the step, never the walk.
   \(\Rightarrow\) InstructGPT keeps an explicit \(\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\)
   term *in addition*. **Write down the prediction now, before you measure it: over \(N\)
   PPO iterations, clip-only drift should accumulate roughly like a random walk / linearly
   in \(N\), while the \(\beta\)-penalized run should plateau at the level set by
   \(d\mathbb{E}[r]/d\mathrm{KL}=\beta\).**
4. **What clipping does to the bias.** \(L^{\text{CLIP}}\) is a biased estimator of the
   improvement. Note which W3 category it lands in: the clip zeroes a gradient contribution
   based on \(\rho\), which depends on \(a_t\) — so this is an **action-dependent**
   modification, i.e. genuinely bias-inducing (contrast a baseline, which must be
   \(h_t\)-measurable). PPO is *deliberately* biased and the bias is the product.

### 1:05-1:47 — Coding: `week-04-rlhf-ppo/code/ppo_loss.py`

Staged harness; imports Day 1's `toy_rlhf` for `exact_J`, `exact_sequence_kl`, and the
enumeration. Ground truth is again exact, not sampled.

**TODOs, one stage each:**

1. `token_logratios(theta, theta_old, batch)` → per-token \(\log\rho_t\) (keep it in log
   space; never form \(\pi/\pi_{\text{old}}\) directly).
2. `is_surrogate(logratios, advantages)` → \(\mathbb{E}[\rho\hat A]\).
3. `clipped_surrogate(logratios, advantages, eps)` → \(L^{\text{CLIP}}\), plus the
   **clip fraction** and the fraction clipped *from above* vs *below* (the instrument you
   will read on Day 5).
4. `ppo_update(theta, batch, n_epochs, n_minibatches, eps, lr, ...)` — the reuse loop:
   freeze `theta_old` and the advantages once, then epoch × minibatch.
5. `drift_experiment(...)` — run \(N\) PPO iterations under (a) clip only, (b) explicit KL
   penalty only, (c) both; log per-iteration clip fraction, \(\mathrm{KL}(\pi_\theta\|\pi_{\text{old}})\),
   and **exact** \(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\).

**Checks with teeth:**

- **Regression to Week 3 (the check that proves the derivation).** At
  \(\theta=\theta_{\text{old}}\), assert `grad(is_surrogate)` equals W3 D3's
  `weights_with_baseline` policy gradient **elementwise to ~1e-12**, and that both match
  the finite-difference `exact_J` gradient. Result 1 becomes an assertion.
- Also assert \(L^{\text{IS}}(\theta_{\text{old}})=\mathbb{E}[\hat A]\) and that this is
  ≈0 for a centered advantage — i.e. the loss value is uninformative, deliberately.
- **The 2×2 table, hand-computed.** Four scalar cases with pencil-checkable values *and*
  gradients; assert the gradient is **exactly `0.0`** in the two clipped cells (not
  "small") and **nonzero** in the other two. A test that only checks the value would pass
  on a wrong implementation that clips both sides.
- Pointwise \(L^{\text{CLIP}}\le L^{\text{IS}}\) over a randomized sweep of
  \((\rho,\hat A)\), including \(\hat A=0\) and \(\rho\) exactly at the boundary.
- \(\epsilon\to\infty\) reproduces `is_surrogate` **exactly**; \(\epsilon\to0\) gives zero
  gradient. Limits as assertions.
- **Log-normal blow-up, measured.** Compute the sequence ratio \(\prod_t\rho_t\) for
  sequences of increasing length at a fixed per-token perturbation; report the median and
  the 99th percentile and assert the spread grows with \(T\). This turns the
  product-of-\(T\)-ratios argument into a number you can quote.
- **The headline experiment.** In `drift_experiment`, assert that the clip-only run
  reaches a strictly larger exact \(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\) than the
  \(\beta\)-penalized run *while its per-iteration clip fraction stays low* — i.e. the
  trust region is satisfied and the policy has still walked away. Then the **negative
  control**: with `n_epochs=1` and one minibatch, clip-only and no-clip must agree to
  machine precision (nothing is off-policy yet), so the effect is attributable to reuse.
- **Epoch sweep.** \(K\in\{1,2,4,8,16\}\): measure clip fraction, exact
  \(\mathrm{KL}(\pi_{\text{old}}\|\pi_\theta)\), and — since you have ground truth — the
  **angle between the PPO update direction and the true \(\nabla J\)**. Expect it to be 0°
  at \(K=1\) and to degrade with \(K\); that plot *is* "why not just do 100 epochs."
- **Negative control for clipping itself:** run \(K=16\) with clipping disabled and assert
  the update direction degrades substantially more than with clipping on. Otherwise the
  clip is decoration and the test has no teeth.

**Traps wired in as assertions** (his recurring classes): float dtype on the ratio
accumulator; `torch.softmax` with no `dim`; appending tensors to history lists instead of
floats; and a **square-shape guard** — deliberately configure `V != T_max` in the fixtures
so a transposed `(batch, T)` vs `(batch, V)` cannot pass silently.

### 1:47-1:57 — Interview Drill

1. **"Derive PPO's clipped objective and tell me exactly what it guarantees."** The second
   half is the scored half: it guarantees *no reward for going further out*, not a bound on
   \(\rho\), not a bound on the KL to the reference, and nothing at all about the walk over
   many iterations. Cite your measured drift numbers.
2. **"Why not one sequence-level ratio instead of per-token ratios?"** Wants the
   product-of-\(T\), log-normal median-vs-mean argument, with your measured percentile
   spread; then the follow-up — *what does the per-token choice cost you?* (it is no longer
   an unbiased IS estimator of the sequence-level objective; you have swapped a correct
   high-variance estimator for a biased usable one, which is the same trade as clipping).

### 1:57-2:00 — Deliverable & Next Session

`deliverable.md`: the 2×2 table, the \(\rho=1\) gradient identity with its Week-3
regression, and the drift plot with the clip-only vs \(\beta\)-penalized numbers. Next:
**Day 3 — token-level PPO for LLMs**: the value head trained by regression (so Day 4's
critic error is *earned* rather than injected by hand), the full InstructGPT loss, and
per-token KL wired into the reward.

## Checkpoint Questions

1. Show \(\nabla L^{\text{IS}}|_{\theta_{\text{old}}} = \nabla J\). What does that make PPO,
   in one sentence, relative to REINFORCE?
2. Fill the 2×2 clip table from memory: value, gradient, and intent per cell. Which cell is
   unbounded and what do variants do about it?
3. In what precise sense is clipping a KL trust region, and name the three ways it isn't.
   Which of the three did you measure, and what number did you get?
4. Why per-token ratios rather than a sequence ratio? What is given up?
5. You raise epochs from 4 to 32 and reward improves on the training batch but the eval
   drops. Give the mechanism in terms of the surrogate's validity region, and the two
   instruments you'd check first.

## Frontier-Lab Interview Angle

"Derive the PPO objective" is the most common RL-depth question after "walk me through
RLHF," and the discriminating follow-up is always some form of *what does clipping
actually guarantee*. Most candidates say "it keeps the policy close to the old one," which
is false as stated — and the interviewer knows it, because the gap between the clip and a
real trust region is precisely why their production stack also carries a KL penalty, an
adaptive \(\beta\), and a KL-based early stop. Being able to say "clipping bounds the step,
not the walk; here is the run where the clip fraction stayed at 3% and the KL to reference
still hit 8 nats" is a senior answer.
