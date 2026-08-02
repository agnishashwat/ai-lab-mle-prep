# Week 05 Day 01 — Bradley-Terry Revisited: What Preference Data Can and Cannot Identify

## Objective

**Read this first, out loud: Week 5 is already half-derived.** W3 D5 gave you
\(\pi^*\propto\pi_{\text{ref}}e^{r/\beta}\), the identity \(J^*=\beta\log Z\), and the
observation — which you made *unprompted* — that \(Z\) depends on the **prompt only**.
W4 D1 gave you Bradley-Terry, its NLL, its gradient \(\sigma(-\Delta)\), and the fact that
\(r\) is identified only up to a per-prompt constant. **Day 2 is those two facts composed
and nothing else.** Today is not a warm-up for that; today is the part nobody does, which
is asking what the first fact is actually *worth* statistically.

The day has one thesis and everything follows from it:

> **The scale of a Bradley-Terry reward is not a property of quality. It is a measurement
> of annotator disagreement.**

Four consequences, each provable and each measurable today:

**(a) Zermelo/Ford existence.** The BT maximum-likelihood estimate is finite **iff the
"beats" digraph is strongly connected.** Corollary that should feel wrong the first time
you read it: *with a noiseless annotator and a strictly ordered item set, the MLE does not
exist no matter how much data you collect.* Disagreement is not noise polluting the signal;
disagreement is the **only** thing pinning the scale down.

**(b) The DPO corollary, one week early.** A preference dataset is
\((x, y_w, y_l)\) triples in which each completion typically appears **exactly once**. That
comparison graph is a **perfect matching** — the most disconnected non-trivial graph there
is. So the BT optimum for a DPO dataset is at infinity *by construction*, and since Day 2
will show DPO's implicit reward is \(\hat r=\beta\log(\pi_\theta/\pi_{\text{ref}})\),
"the rejected logprob goes to \(-\infty\)" is not a bug report. It is this theorem.

**(c) Noise is the regularizer.** Label flips at rate \(\varepsilon\) cap the achievable
margin at \(\operatorname{logit}(1-\varepsilon)\) — a *finite* number. Every fix in Day 4
(IPO's bounded loss, cDPO's label smoothing) is a way of putting back a cap that clean data
removed.

**(d) The RM's scale and \(\beta\) are the same knob.** \(\text{tilt}(r,\beta)\) and
\(\text{tilt}(cr, c\beta)\) are the *identical distribution*, exactly. Only \(r/\beta\) is
identified by the policy. Combine with (a)–(c): the annotator agreement rate sets the scale
of \(r\), which sets what \(\beta\) means. **\(\beta\) is a property of your annotation
process**, which is why it does not transfer across datasets and why "just use 0.1" is a
confession rather than a default.

Build `code/preference_models.py` to make all four assertions rather than claims.

## Prerequisites

- **W4 D1** — Bradley-Terry NLL, its gradient, per-prompt shift invariance, and the
  measured RM-error-vs-KL curve. Today extends; it does not repeat.
- **W3 D5** — the exponential tilt, \(J^*=\beta\log Z\), \(d\mathbb{E}[r]/d\mathrm{KL}=\beta\),
  reverse-KL support confinement.
- Logistic regression, and the notion of **linear separability**.
- W4's staged-harness convention.

## Required Reading

- **DPO (Rafailov et al. 2023) §3 and §4** — §3 sets up BT + the KL-regularized objective in
  the paper's notation (you know both), §4 derives the loss. **Read §4 but stop before the
  gradient**; you derive the gradient yourself tomorrow.
- **DPO §4 / Appendix A.3, the Plackett-Luce paragraph.** Short, and it is the honest
  generalization of BT to \(K\)-wise rankings.
- Optional but high-value: the two-page classical result — **Ford (1957)**, "Solution of a
  ranking problem from binary comparisons," for the strong-connectivity condition. If you
  only read one thing today that is not the DPO paper, read this.

## 2-Hour Plan

> **Clock note.** Blocks are delivered in full. W4 D5's two quiz blanks and all three
> skipped drills landed at the *end* of a 14.5-hour session, so this week the graded blocks
> are front-loaded and **the interview drills come immediately after the derivation, before
> the coding block**, not after it.

### 0:00-0:10 — Cold recall of CONCLUSIONS

One line per result, spoken, no whiteboard. **New rule, and it is not optional: before you
speak on any question that has a number in it, write the number on the page first.** Six
consecutive sessions have now failed on "state your own measured value under a diagnostic
framing" while succeeding under a result framing. Verbal reminders have failed six times, so
the mechanism changes: **numbers on the page before prose exists.**

1. Closed form of the KL-regularized optimum. Then \(J^*\) in one symbol.
2. Why is \(Z\) intractable? One clause — and it is not "we lack logits." Then the second
   clause that makes Week 5 possible: **what does \(Z\) depend on?**
3. Bradley-Terry: the model, and what \(r\) is identified up to. Then the three consequences.
4. \(d\mathbb{E}[r]/d\mathrm{KL}=?\) and the shape of the frontier.
5. **Clipping is not a trust region — at three levels.** Say the three words first
   (*token / sample / iteration*), then one number for each. The iteration level is the one
   you have missed three times; the fact is that \(\pi_{\text{old}}\) **resets**.
6. **Perennial, now three sessions overdue: the sources of variance in REINFORCE.** Best so
   far is 6 of 7 — but you have never once named the policy's own **entropy**, the
   **environment's** stochasticity, or **off-policy-ness itself**. Keep SOURCES separate
   from MECHANISMS.
7. **k1/k2/k3, asked the way it now matters.** Define all three; which is unbiased, which is
   \(\ge 0\), which is both and by what mechanism. Then the engineering question:
   your \(\beta\) controller's *input* is a k1 estimate that was **negative on 20-44% of
   samples**. Which estimator should drive a controller, and why is the answer not simply
   "the unbiased one"?

### 0:10-0:25 — Graded re-ask: the three W4 D5 drills, cold

These were written for you as model answers at 00:10 after a 14-hour session and never
attempted. They are graded now, cold, and **drill 3 is the important one because Week 5 is
its answer.**

1. **"Design an RLHF system from scratch."** Four stages, four models, the memory bill, the
   four load-bearing orderings, the instrument panel in pairs, and the three
   non-negotiables. Numbers on the page first.
2. **"Why does PPO look the way it does?"** Both chains. Chain 1 is how you get a gradient
   and then how you afford it; chain 2 is why there is a *second* mechanism at all.
3. **"When would you *not* use PPO?"** — the one that matters. PPO's bill; what DPO deletes
   and what it pays; what GRPO deletes and what it pays; **the one thing PPO gives that
   neither does**; and how you would choose. Cite your own ablation: what did the critic buy
   on a short-horizon terminal-reward task, and what does that generalize to?

> **Grading rubric, unchanged and enforced.** Every multi-clause question is scored as two
> line items — *first clause* and *everything after*. First clause only caps the item at 0.5.
> Fourth week running that clause two is where the points go. Count the clauses in the
> question and the paragraphs in your answer before you speak.

### 0:25-0:50 — Concept: preference data as a statistical object

**Framing.** W4 D1 treated the RM as a *function approximation* problem: fit \(r_\psi\),
worry about generalization off-distribution. Today treats it as an **identification**
problem: forget approximation error entirely, give yourself infinite data and a tabular
\(r\) with one free parameter per completion, and ask *what is still unknown*. The answer is
much larger than "an additive constant per prompt," and the extra part is the whole of
Week 5's failure surface.

- **The three degrees of freedom.** Given preference data alone:
  - **Location** — \(r(x,\cdot)+c(x)\) is exactly equivalent. Known from W4 D1. Cancels in
    every pairwise loss, cancels in the tilt, and is exactly what GRPO's group mean removes.
  - **Scale** — \(r \to cr\) is **not** equivalent (it changes \(\sigma(\Delta)\)), so the
    data *does* identify scale. **But it identifies it through the disagreement rate, not
    through quality.** This is the day's new content.
  - **Everything outside the support of the comparison graph** — items never compared,
    directly or transitively, to the rest are not identified at all. This is a graph
    property, and it is the next bullet.
- **Ford/Zermelo: when does the MLE exist?** The BT log-likelihood is concave (it is logistic
  regression on difference features), so the MLE is unique *if it exists*. It exists — i.e.
  the maximizer is finite — **iff for every partition of the items into non-empty \(S\) and
  \(S^c\), someone in \(S\) beat someone in \(S^c\) and vice versa**; equivalently, the wins
  digraph is **strongly connected**. Otherwise the likelihood is strictly increasing along
  the direction "raise all of \(S\), lower all of \(S^c\)" and the supremum sits at infinity.
  - **This is exactly linear separability.** BT is logistic regression with feature vector
    \(e_w - e_l\); a separating direction is precisely a partition never contradicted.
  - **Under gradient descent the symptom is specific and recognizable**: the loss decays to 0
    like \(1/t\)-ish, \(\|r\|\) grows *logarithmically and without bound*, and the
    **direction converges** to the max-margin separator. Nothing errors. Nothing NaNs. Your
    accuracy metric reads 100%. (Soudry et al., implicit bias of gradient descent — the same
    result that explains why unregularized separable logistic regression "works.")
- **The corollary that should stop you.** Suppose annotators are *perfect* and completions
  have a strict true ordering. Then every comparison agrees with the order, the wins digraph
  is a **transitive tournament** — a DAG — and no partition is ever contradicted. **The MLE
  is at infinity, with infinite data.** More data makes it worse, not better: it adds edges
  but never a cycle.
  - Say the consequence out loud: *the finiteness of a reward model is bought entirely with
    annotator noise.* Improve your rubric, train your raters, raise agreement — and you have
    **increased** the scale of the reward your data implies.
- **What that does to \(\beta\).** \(\pi\propto\pi_{\text{ref}}e^{r/\beta}\) depends on
  \(r\) and \(\beta\) **only through \(r/\beta\)**. So the policy cannot distinguish "a
  reward twice as large" from "a \(\beta\) half as large." Chain it:
  agreement rate \(\Rightarrow\) scale of \(r\) \(\Rightarrow\) meaning of \(\beta\)
  \(\Rightarrow\) the KL you land at. **\(\beta\) is a property of the annotation process.**
  This is a strictly stronger statement than W4 D5's "\(\beta\) is a property of the RM's
  data distribution" — that one was about *where* the data is, this one is about *how noisy
  its labels are*, and they are independent failure axes.
- **Noise as a cap, quantitatively.** Let true preferences follow BT with reward \(r^*\), and
  let the annotator flip the label independently with probability \(\varepsilon\). The
  observed probability is
  \[
    p = (1-2\varepsilon)\,\sigma(\Delta^*) + \varepsilon,
  \]
  which is **not** \(\sigma\) of anything for \(\varepsilon>0\): it is squashed into
  \([\varepsilon, 1-\varepsilon]\). In the **saturated** case (one free margin per pair) the
  MLE matches the empirical frequency, so \(\hat\Delta=\operatorname{logit}(p)\) and as
  \(\Delta^*\to\infty\), \(\hat\Delta\to\log\frac{1-\varepsilon}{\varepsilon}\) — a **finite
  ceiling set by \(\varepsilon\) alone.** At 10% flips the ceiling is 2.197; at 25% it is
  1.099; at 0% it is \(\infty\). Be precise about the scope: with \(n\) items and \(n-1\)
  free parameters against \(\binom n2\) pair frequencies, BT under flip noise is
  **misspecified**, so a dense-graph fit lands between the tightest and loosest per-pair
  logits rather than on the cap exactly — finite and monotone in \(\varepsilon\), which is
  the part that matters. *That* is the regularizer, and Day 4's IPO and cDPO are two ways of
  putting it back by hand.
- **The accuracy ceiling, and why "78% pairwise accuracy" was the right thing to be
  suspicious of** (W4 D1 drill 2). Under BT the Bayes-optimal classifier is "predict the
  higher \(r\)", with accuracy \(\mathbb{E}[\max(p,1-p)]\). If humans agree with each other
  70% of the time, an RM reporting 78% is not 8 points better than a human — it is either
  measuring an easier pair distribution or fitting annotator identity. **Report RM accuracy
  against the agreement ceiling or do not report it.**
- **Plackett-Luce, briefly.** \(K\) ranked completions:
  \(P(\tau)=\prod_{k}\frac{e^{r(y_{\tau(k)})}}{\sum_{j\ge k}e^{r(y_{\tau(j)})}}\) — i.e. a
  sequence of softmax choices over the *remaining* items. \(K=2\) is exactly BT. Two things
  to notice: it is the same softmax-of-rewards object as the tilt, and it inherits the same
  location invariance (the constant cancels in each factor).
- **What BT assumes, and the failure that is not about noise.** BT forces **stochastic
  transitivity** by representing every item with one scalar. Real populations are
  intransitive when preferences are multi-attribute and annotators weight attributes
  differently — a genuine rock-paper-scissors, not sampling error. A single scalar RM cannot
  represent a *disagreement*; it will represent the population's mode and call it a
  preference. This is the honest argument for per-attribute reward heads and for treating
  "helpful + harmless + honest" as three numbers.

### 0:50-1:12 — Derivation

Four parts. Parts 1-2 are the day's theorem; Parts 3-4 are the ones that pay next week.

1. **Existence.** Write the BT NLL as logistic regression on \(e_w-e_l\). Show the
   log-likelihood is concave. Then show: if there is a partition \(S,S^c\) with no
   \(S^c\)-beats-\(S\) comparison, the direction \(d = \mathbb 1_S - \mathbb 1_{S^c}\) has
   \(\nabla\mathcal{L}\cdot d < 0\) *everywhere* and \(\mathcal L(r+td)\) is strictly
   decreasing in \(t\) with no finite minimizer. Conclude Ford's condition. Then do the
   converse direction informally: with a strongly connected graph, every direction has some
   comparison pushing back, so the NLL is coercive.
2. **The noise cap.** Derive \(p=(1-2\varepsilon)\sigma(\Delta^*)+\varepsilon\) and show that
   in the saturated case the MLE matches the empirical frequency, hence
   \(\hat\Delta=\operatorname{logit}(p)\to\log\frac{1-\varepsilon}{\varepsilon}\). State the
   two limits: \(\varepsilon\to 0\) recovers Part 1's divergence, \(\varepsilon\to 1/2\)
   gives \(\hat\Delta\to0\) (no information). **Part 1 is the \(\varepsilon\to0\) limit of
   Part 2** — say that out loud; it is what makes the day one result and not two. Then state
   the misspecification caveat for the parametric case *before* the harness shows it to you.
3. **Scale-\(\beta\) equivalence.** Show \(\text{tilt}(cr,c\beta)=\text{tilt}(r,\beta)\)
   pointwise, for all \(c>0\). Then state precisely what is *not* invariant: the KL you land
   at is a function of \(r/\beta\), so **holding \(\beta\) fixed and improving your
   annotators moves your KL** even with the identical policy, prompts, and rollouts.
4. **The setup for tomorrow — do not answer it today.** You have
   \(\pi^*\propto\pi_{\text{ref}}e^{r/\beta}\): every reward induces a policy. Write the
   question on the board and leave it: *is the map invertible — does every policy induce a
   reward, and if so, what is the reward of \(\pi_\theta\)?* If you find yourself answering
   it, stop and write it down for tomorrow instead; that is Day 2's derivation and it is
   worth doing at full attention.

### 1:12-1:22 — Interview Drill (moved BEFORE coding, deliberately)

Two, ~5 min each. Numbers on the page before prose.

1. **"Your reward model gets 96% accuracy on held-out preference pairs. Is that good?"**
   Wants: the agreement ceiling; the possibility that 96% means the pairs are separable and
   the *fit is diverging*; what \(\|r\|\) is doing; that accuracy is invariant to scale while
   everything downstream depends on scale; and what you would actually plot instead.
2. **"We cleaned up our preference data — better rubric, adjudicated disagreements, inter-
   annotator agreement up from 68% to 91%. Our DPO runs now collapse. What happened?"**
   Wants the full causal chain: agreement ↑ ⇒ separability ↑ ⇒ MLE scale ↑ ⇒ effective
   \(r/\beta\) ↑ at fixed \(\beta\) ⇒ the same \(\beta\) now buys far more KL ⇒ collapse.
   Then the three fixes with their trade-offs (raise \(\beta\); label-smooth / cDPO; a
   bounded loss, i.e. IPO) and — the part that separates a good answer — **why "collect
   cleaner data" made things worse is not a paradox but the theorem.**

### 1:22-1:52 — Coding: `week-05-dpo-preference-optimization/code/preference_models.py`

Staged harness, repo convention (`shared/templates/exercise-harness-template.py`; reference
`week-04-rlhf-ppo/code/toy_rlhf.py`, which you import from). Run under
`-W error::RuntimeWarning`. A private reference implementation is written and run *before*
this is handed to you, per the W4 D5 process win, so every threshold below is known reachable.

**Your TODOs, one stage each:**

1. `bt_nll_and_grad(r, pairs)` — carried over from W4 D1 but now on a **tabular \(r\) over
   completions** rather than per-prompt. Analytic gradient, checked against central FD.
2. `wins_digraph(pairs, n_items)` and `is_strongly_connected(adj)` — Tarjan or repeated
   BFS; small \(n\), so an \(O(n^2)\) reachability closure is fine and easier to verify.
3. `fit_bt(pairs, n_items, n_steps, lr)` returning \(\hat r\) **and the trajectory** of
   \(\|\hat r\|\), the loss, and the unit direction \(\hat r/\|\hat r\|\).
4. `noisy_labels(rng, gold_r, pairs, eps)` — BT sampling, then an independent flip at rate
   \(\varepsilon\). Keep the two sources of randomness separate and separately seeded; you
   need to vary \(\varepsilon\) with the BT draws held fixed.
5. `plackett_luce_nll(r, rankings)` — the \(K\)-wise generalization.
6. `tilt(r, pi_ref, beta)` — import from W3 D5 / W4 D1 rather than rewriting; add the
   scale-equivalence probe.
7. `bayes_accuracy(gold_r, pairs)` — \(\mathbb{E}[\max(p,1-p)]\), the ceiling.

**Checks that must have teeth:**

- **The existence theorem, both directions, same items and same gold reward — only the
  pairing changes.** (i) A **transitive tournament** (all pairs, noiseless labels): assert
  `is_strongly_connected` is **False**, and that over a long fit \(\|\hat r\|\) is
  **monotone increasing and unbounded** while the loss \(\to 0\) and pairwise accuracy is
  **exactly 1.0**. (ii) The **same pairs with \(\varepsilon=0.1\) labels**: assert strongly
  connected **True** and \(\|\hat r\|\) **converges** (relative change over the last decade
  of steps below a tight threshold). The point of the pairing is that *nothing about the
  reward or the items changed* — the difference is entirely in the labels.
- **Direction converges even when the norm does not.** On the separable fixture, assert
  \(\cos(\hat r_t, \hat r_{2t}) \to 1\) while \(\|\hat r_{2t}\|/\|\hat r_t\| > 1\) at every
  decade. This is what makes "it looks like it's training fine" a *measured* statement.
- **The perfect-matching fixture — the DPO shape.** Build pairs where every completion
  appears exactly once. Assert not strongly connected, and assert the fitted margin on
  *every* pair exceeds any threshold you like given enough steps. Then the headline number:
  report \(\min_i \hat r_i\) and note that under Day 2's identification this is
  \(\beta\log(\pi/\pi_{\text{ref}})\), i.e. the rejected logprob.
- **The noise cap.** Sweep \(\varepsilon\in\{0, 0.02, 0.05, 0.1, 0.2, 0.35\}\) on a dense
  graph and assert the fitted max margin tracks \(\log\frac{1-\varepsilon}{\varepsilon}\)
  within tolerance, and is **monotone decreasing** in \(\varepsilon\). Assert the
  \(\varepsilon=0\) arm does not converge at all — the sweep's own left endpoint is Part 1.
- **Scale-\(\beta\) equivalence, exact.** Assert `tilt(c*r, c*beta) == tilt(r, beta)`
  elementwise to \(\sim10^{-15}\) over a grid of \(c\). Then the *consequence* check: hold
  \(\beta\) fixed, fit the RM at two agreement levels, and assert the resulting
  \(\mathrm{KL}(\pi\|\pi_{\text{ref}})\) **differs by a large factor** — the same \(\beta\),
  the same items, different raters.
- **Location invariance is still exactly free** (regression test on W4 D1): adding \(c(x)\)
  to a prompt's rewards changes neither the NLL nor the tilt, to \(0.0\).
- **Plackett-Luce reduces to BT at \(K=2\)**, elementwise to \(\sim10^{-15}\). Cheap, and it
  is the check that catches a wrong denominator (summing over *all* items instead of the
  remaining ones).
- **Bayes ceiling.** Assert a fitted RM's held-out accuracy does **not** exceed
  `bayes_accuracy` by more than sampling error, on data generated from the gold reward.
  Report both numbers side by side; this is the drill-1 answer as a measurement.

### 1:52-2:00 — Deliverable & Next Session

Write `deliverable.md`: the existence theorem with your two fixtures' numbers, the
\(\varepsilon\)-cap table, the scale-\(\beta\) identity, and the answer to drill 2 written
out in full. Commit `code/preference_models.py`.

Next: **Day 2 — deriving DPO.** You will invert the tilt, substitute into Bradley-Terry,
watch \(Z(x)\) cancel because it depends on the prompt only, and then keep going past where
the paper stops: what the derivation *assumes*, why DPO's constraint is enforced only at the
datapoints in the dataset, and why — given today — DPO is a maximum-likelihood problem whose
optimum is at infinity.

## Checkpoint Questions

1. State Ford's condition for the BT MLE to exist, and prove the "only if" direction in two
   sentences. Then answer: with noiseless annotators and a strictly ordered item set, does
   more data help?
2. A preference dataset has each completion appearing exactly once. What is the comparison
   graph, does the MLE exist, and what does that predict about a DPO run on this data?
3. Derive the ceiling that flip-noise \(\varepsilon\) puts on the fitted margin. Give the
   value at \(\varepsilon=0.1\), and state both limits.
4. Show that \(\text{tilt}(cr,c\beta)=\text{tilt}(r,\beta)\). Then: your annotator agreement
   improves from 68% to 91% and you change nothing else. What happens to your KL, and why is
   this a *prediction* rather than an anecdote?
5. Your RM reports 96% pairwise accuracy. Give three distinct things that could be true, and
   the single plot that distinguishes them.

## Frontier-Lab Interview Angle

Everyone can say "we train a reward model on preference pairs." The differentiators are
today's: knowing that the BT likelihood has a **finiteness condition** and that clean data
violates it; being able to say *why* a rejected logprob goes to \(-\infty\) without hand-
waving about "the loss only cares about the difference"; knowing that RM scale and \(\beta\)
are a single degree of freedom, so \(\beta\) does not transfer across annotation pipelines;
and being the person in the room who, when told "we improved data quality and the model got
worse," reaches for the theorem instead of the changelog. It also sets up the strongest
possible version of the DPO-vs-PPO answer, because it locates DPO's central weakness in
*statistics* rather than in *engineering*.
