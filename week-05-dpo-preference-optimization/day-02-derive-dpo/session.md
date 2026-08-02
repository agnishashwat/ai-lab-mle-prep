# Week 05 Day 02 — Deriving DPO, and Then Past Where the Paper Stops

## Objective

**The derivation itself is yours already and you should be told so before you start.** On
W3 D5 you wrote \(r = \beta\log\frac{\pi^*}{\pi_{\text{ref}}} + \beta\log Z\) *unprompted*
and observed that \(Z\) depends on the prompt only. On W4 D1 you derived Bradley-Terry. DPO
is those two lines composed. **You will not be walked through it; you will be asked to
produce it cold in ten minutes, and the block is graded.** If it takes longer than ten
minutes, that is the finding.

The day's real content is the four things that come *after* the cancellation, which is where
almost every candidate stops and where every DPO failure mode actually lives:

**(a) The gradient, and what it reveals.** \(\nabla\mathcal L_{\text{DPO}}\) carries the
weight \(\sigma(-\beta\hat\Delta)\) — **structurally identical** to W4 D1's BT gradient
weight \(\sigma(-\Delta)\). That is not a coincidence and it is the sentence to have ready:
**DPO is a reward model whose parameterization happens to be a policy.** Everything true of
reward-model fitting is therefore true of DPO training, including yesterday's existence
theorem.

**(b) The equivalence, verified rather than asserted.** On a fully enumerable toy you will
fit a BT reward model, tilt it exactly (W3 D5's closed form), separately run DPO on the same
pairs, and check the two resulting distributions agree **elementwise to \(\sim10^{-10}\)**.
Then you will find the fixtures where they *stop* agreeing, which is the interesting half.

**(c) What the derivation assumes.** Three assumptions, each load-bearing and each violated
in practice: the policy class must be able to represent the tilt exactly; the BT MLE must
exist (yesterday: usually it does not); and \(\pi_{\text{ref}}\) in the tilt must be the same
distribution the pairs were drawn from (usually it is not — pairs come from a *stronger*
model or from a mixture).

**(d) The asymmetry that is the whole DPO-vs-PPO story.** PPO estimates
\(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\) **under \(\pi_\theta\)** — on samples the
current policy actually generates. DPO's constraint appears only through the ratio evaluated
**at the dataset's completions**. Off the dataset, *nothing whatsoever constrains
\(\pi_\theta\)*. The KL in the derivation is not the KL that gets enforced. Say this
precisely and you have the best available answer to "what is DPO's real weakness."

## Prerequisites

- **W5 D1** — Ford's existence condition, the \(\varepsilon\)-cap, the scale-\(\beta\)
  identity.
- **W3 D5** — the tilt, \(J^*=\beta\log Z\), reverse-KL support confinement.
- **W4 D1/D2** — BT NLL and gradient; the ratio \(\rho\) and why \(\nabla\rho\) at
  \(\theta=\theta_{\text{old}}\) *is* the policy gradient.

## Required Reading

- **DPO §4** in full now, including the gradient, plus **Appendix A.1-A.2** (the derivation
  of the optimum and the reparameterization lemma). Read A.2 carefully: it is the statement
  that every reward is *equivalent* to some policy under the induced equivalence class, and
  it is the part people quote without reading.
- Optional: **Appendix A.3**, the Plackett-Luce version, now that you have PL from D1.

## 2-Hour Plan

### 0:00-0:08 — Cold recall of CONCLUSIONS

1. Ford's condition, in one sentence. Then: noiseless annotators, strict ordering, infinite
   data — does the MLE exist?
2. The \(\varepsilon\)-cap formula and its value at \(\varepsilon=0.1\). **Number on the page
   before you speak.**
3. \(\text{tilt}(cr,c\beta)\) — state the identity and the one-clause consequence for
   \(\beta\).
4. Your own measured numbers from D1: \(\|\hat r\|\) behaviour on the transitive tournament
   vs the \(\varepsilon=0.1\) fixture. Both numbers, written first.
5. Perennial, keep asking: three of the seven REINFORCE variance sources you have never
   named.

### 0:08-0:20 — The derivation, cold and graded

No hints, no scaffolding, ten minutes on the clock, then two minutes of grading. Produce:

\[
\mathcal L_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}
\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
- \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
\]

Graded on four things, separately: (i) inverting the tilt correctly including the
\(\beta\log Z\) term; (ii) naming *where* the cancellation happens and *why* — the
\(Z(x)\)-depends-on-the-prompt-only fact, which must be stated as the reason and not merely
used; (iii) the substitution into BT with the difference taken inside \(\sigma\); (iv) saying
out loud what has been *assumed* to write \(r = \beta\log(\pi^*/\pi_{\text{ref}})+\beta\log Z\)
— namely that \(\pi^*\) is the exact optimizer for that \(r\), which is a statement about
the policy class.

### 0:20-0:45 — Concept: what the reparameterization actually claims

- **The bijection, stated carefully.** The map \(r \mapsto \pi_r \propto
  \pi_{\text{ref}}e^{r/\beta}\) is onto the set of policies with
  \(\mathrm{supp}(\pi)\subseteq\mathrm{supp}(\pi_{\text{ref}})\), and its inverse is
  \(\pi\mapsto \beta\log(\pi/\pi_{\text{ref}})\) **up to a per-prompt constant**. Two things
  fall straight out:
  - **The equivalence classes match.** BT cannot identify \(c(x)\); the tilt cannot see
    \(c(x)\). The quotient spaces are the same space. That is the actual content of the
    paper's Lemma — not "rewards and policies are the same," but *"the thing rewards fail to
    identify is exactly the thing policies fail to encode."*
  - **Support confinement is inherited.** \(\pi_\theta\) can only put mass where
    \(\pi_{\text{ref}}\) does, in the derivation. In a real DPO run it is a softmax over the
    full vocabulary and can put mass anywhere. **The constraint that made the derivation true
    is not present in the optimizer.**
- **The gradient.** Derive
  \(\nabla_\theta\mathcal L = -\beta\,\mathbb E\big[\sigma(-\beta\hat\Delta)\,
  (\nabla\log\pi_\theta(y_w|x)-\nabla\log\pi_\theta(y_l|x))\big]\)
  and read it three ways: (i) it is a **weighted contrastive log-likelihood** — push up
  chosen, push down rejected; (ii) the weight is *how wrong the implicit RM currently is*,
  the same \(\sigma(-\Delta)\) as BT, so **correctly-ranked pairs contribute almost nothing**
  and the fit is driven by near-ties and mislabels; (iii) there is **no term that constrains
  the marginal likelihood** — only the difference is penalized, which is why chosen and
  rejected logprobs can both fall.
- **Where the KL went.** In PPO the KL is a *term in the reward*, estimated on-policy every
  iteration. In DPO the KL is not a term at all: it was consumed in the algebra and survives
  only as the \(\pi_{\text{ref}}\) in the ratio, evaluated at **dataset** completions. Two
  consequences worth stating as separate line items:
  - DPO enforces nothing at all on \(\pi_\theta\)'s behaviour at completions it actually
    generates. The failure mode "DPO output goes off-distribution in a way the loss cannot
    see" is not a surprise, it is the derivation.
  - **PPO's KL is reverse KL under \(\pi_\theta\); DPO's implied constraint is evaluated
    under the data distribution.** Different measure, different guarantee. This is the same
    forward-vs-reverse distinction you nailed on W3 D5 for a different pair of objects — say
    it with that vocabulary.
- **Off-policy, and what that costs.** DPO is fully offline: no rollouts, no sampling loop,
  and therefore no way to observe your current policy's mistakes. W4 D5's answer already had
  this — "paid for with static off-policy data" — today makes it precise: the dataset was
  drawn from some \(\mu\), the derivation wanted \(\pi_{\text{ref}}\), and any mismatch is an
  unrepaired importance-weighting problem with **no ratio anywhere in the loss to repair it.**
- **The existence problem, restated in DPO's variables.** D1: the BT MLE is at infinity for a
  matching. Under the reparameterization \(\hat r = \beta\log(\pi/\pi_{\text{ref}})\),
  "\(r_l\to-\infty\)" is literally "\(\pi_\theta(y_l|x)\to0\)". So **DPO's objective is
  minimized by driving the rejected completion's probability to zero**, and since the
  vocabulary is a simplex, that mass has to go *somewhere the loss cannot see*. Yesterday's
  theorem, today's mechanism, and Day 5's plot.

### 0:45-1:05 — Derivation block: three things the paper does not do

1. **DPO's optimum, exactly, on a tabular policy.** For a fixed prompt with enumerable
   completions and unconstrained tabular \(\pi_\theta\), characterize the minimizer of
   \(\mathcal L_{\text{DPO}}\) and show it equals \(\text{tilt}(\hat r_{\text{BT}},\beta)\)
   where \(\hat r_{\text{BT}}\) is the BT MLE on the same pairs. Note precisely which step
   requires the policy class to be unconstrained.
2. **What breaks with a restricted policy class.** Show by construction that if
   \(\pi_\theta\) cannot represent \(\text{tilt}(\hat r,\beta)\), DPO's optimum is *not* the
   projection of the tilt in any obvious divergence — the loss weights errors by
   \(\sigma(-\beta\hat\Delta)\) at *dataset pairs only*, so the approximation error is
   allocated by the data distribution, not by the policy's own occupancy. One sentence, and
   it is the honest version of "DPO and RLHF-with-PPO are equivalent."
3. **The reference-mismatch term.** Suppose pairs are drawn from \(\mu\ne\pi_{\text{ref}}\).
   Write down what the DPO loss is now the MLE of, and identify the missing importance
   weight \(\mu/\pi_{\text{ref}}\). Then state the practical corollary that everybody hits:
   **if your preference data came from a stronger model, \(\pi_{\text{ref}}\) assigns it low
   probability, so \(\log(\pi_\theta/\pi_{\text{ref}})\) starts large and positive on
   *both* completions** — and the first thing DPO does is fix that, not your preferences.
   This is the mechanism behind the standard "run SFT on the chosen completions first"
   advice; you can now say *why* rather than *that*.

### 1:05-1:15 — Interview Drill

1. **"Derive DPO."** Full, out loud, including the assumption audit. Then the follow-up
   nobody prepares for: *"where did the KL constraint go, and what enforces it during
   training?"*
2. **"Is DPO equivalent to RLHF?"** The answer is "yes under three conditions, and here they
   are, and here is which one your setup violates." Give the numbers from the coding block.

### 1:15-1:50 — Coding: `code/dpo.py`

Built on W4 D1's `toy_rlhf.py` environment so every sequence is enumerable and every claim
is exact. A private reference implementation is run first.

**TODOs:**
1. `dpo_loss_and_grad(theta, pairs, theta_ref, beta)` — analytic gradient, FD-checked.
2. `implicit_reward(theta, theta_ref, beta)` — \(\beta\log(\pi_\theta/\pi_{\text{ref}})\)
   over all enumerable sequences.
3. `train_dpo(...)` — full-batch gradient descent with per-step logging of
   \(\log\pi(y_w)\), \(\log\pi(y_l)\), the implicit margin, and
   \(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\) computed **exactly** by enumeration.
4. `bt_then_tilt(pairs, beta)` — the PPO-side reference path: fit BT (D1's `fit_bt`), tilt
   exactly (W3 D5).

**Checks with teeth:**
- **The equivalence.** On a fixture whose comparison graph *is* strongly connected (so the
  BT MLE exists), assert `train_dpo` and `bt_then_tilt` agree elementwise to \(10^{-10}\) —
  and assert the check is **non-vacuous** by showing a wrong-\(\beta\) tilt differs by orders
  of magnitude. This is the week's headline check and it is the W4 D5 chained-regression
  trick applied across two *algorithms* rather than two Ψ rungs.
- **The implicit reward recovers the BT MLE up to a per-prompt constant**, to \(10^{-10}\)
  after centering — and *not* before centering. Both halves asserted.
- **Divergence on a matching.** On the perfect-matching fixture: assert
  \(\log\pi_\theta(y_l)\) decreases monotonically without bound, the loss \(\to0\), and the
  exact KL grows without bound while the *training* metrics (accuracy, margin) look perfect.
  Report KL at the step where accuracy first hits 1.0 and at 10× that many steps.
- **Both logprobs fall.** On a fixture where pairs are drawn from \(\mu\ne\pi_{\text{ref}}\),
  assert \(\log\pi_\theta(y_w)\) also decreases even though \(y_w\) is being "preferred", and
  report the decomposition: margin up, both terms down. Then assert that with
  \(\mu=\pi_{\text{ref}}\) the chosen logprob rises. **One line of the fixture changes.**
- **Gradient weight is \(\sigma(-\beta\hat\Delta)\)**: assert the per-pair gradient norm is
  proportional to it, to \(10^{-12}\), and report the ratio of contributions between the
  best-ranked and worst-ranked decile.
- Regression: DPO's loss is invariant to \(r\to r+c(x)\) applied to the *gold* reward used to
  generate labels — exactly \(0.0\) change.

### 1:50-2:00 — Deliverable & Next Session

`deliverable.md`: the derivation verbatim, the assumption audit, the equivalence numbers, the
divergence numbers. Commit `code/dpo.py`.

Next: **Day 3 — implementation details**, where the length prior, the reference-model
logistics, \(\log\sigma\) stability, and label smoothing (which is D1's \(\varepsilon\)-cap
put back by hand) all become code.

## Checkpoint Questions

1. Derive DPO. Then name the three assumptions and say which one your setup violates.
2. Where does the KL constraint go? What enforces it at training time, and on which
   distribution is it evaluated?
3. Write the DPO gradient. What is the weight, what does it imply about which pairs drive
   training, and what is *missing* from the gradient that lets both logprobs fall?
4. Under what exact conditions is DPO equivalent to fitting a BT reward model and running
   KL-regularized RL to convergence? Cite your measured agreement.
5. Your preference pairs came from a model stronger than your SFT checkpoint. What does DPO
   spend its first phase of training doing, and what is the standard fix?

## Frontier-Lab Interview Angle

"Derive DPO" is now a standard question and a large fraction of candidates can do the
algebra. Almost none can answer the follow-up — *where did the KL go, and what enforces it* —
or state the three conditions for equivalence, or explain why both logprobs fall. The
verified equivalence experiment is a portfolio asset in its own right: it turns "DPO is
equivalent to RLHF" from a citation into a number, and the fixtures where it *fails* to hold
are the interesting part of the story.
