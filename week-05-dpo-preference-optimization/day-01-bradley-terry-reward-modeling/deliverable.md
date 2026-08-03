# Deliverable — Week 05 Day 01: What Preference Data Can and Cannot Identify

**Artifact:** `week-05-dpo-preference-optimization/code/preference_models.py` — nine stages,
**74 checks**, 38 s full / 8 s `--fast`, clean under `-W error::RuntimeWarning`.

W4 D1 treated the reward model as a **function-approximation** problem. Today treated it as an
**identification** problem: infinite data, a tabular \(r\) with one free parameter per
completion, a perfectly specified model — and asked what is *still* unknown. The answer is much
larger than "an additive constant per prompt," and the extra part is Week 5's whole failure
surface.

## The thesis

> **The scale of a Bradley-Terry reward is not a property of quality. It is a measurement of
> annotator disagreement.**

Three degrees of freedom in preference data, not one:

| | what it is | status |
|---|---|---|
| **Location** | \(r(x,\cdot) + c(x)\) is exactly equivalent | known since W4 D1; a **flat** direction, harmless, gauge-fixable |
| **Scale** | \(r \to cr\) changes \(\sigma(\Delta)\), so it *is* identified — **but by the disagreement rate** | today |
| **Connectivity** | items outside the comparison graph's strong components are unidentified | today, and it is much larger than it looks |

The distinction that made the day: location invariance is a **flat** direction — the likelihood
is *constant* along it, so there are infinitely many maximisers, all finite, and you fix it by
gauge-fixing. Separability is an **ascent** direction — the likelihood is *strictly increasing*
along it forever, so there are **zero** maximisers, and centring cannot help.

---

## Headline 1 — Ford's condition, and the corollary that should stop you

> The BT maximum-likelihood estimate is **finite iff the wins digraph is strongly
> connected** (Ford 1957 / Zermelo 1929) — i.e. iff for every partition into non-empty
> \(S, S^c\), someone in \(S\) beat someone in \(S^c\) *and* vice versa.

BT is logistic regression on the difference features \(e_w - e_l\), so this is exactly **linear
separability**: a separating direction *is* a partition no comparison contradicts.

**Proof of the "only if" half, in two lines.** Suppose no comparison goes from \(S^c\) to \(S\).
Take \(d = \mathbb 1_S - \mathbb 1_{S^c}\). Under \(r \to r + td\), every within-block margin is
unchanged and every cross-block margin increases by \(2t\) **in the winning direction**. So
every term of \(\log\sigma(\Delta)\) is strictly increasing in \(t\), the likelihood has no
finite maximiser, and the supremum sits at infinity. \(d\) is not the constant direction, so
centring does not remove it.

**The corollary.** A *perfect* annotator on a strictly ordered item set produces a **transitive
tournament** — a DAG. A DAG has no cycles, so no partition is ever contradicted at any \(k\).

> **The MLE is at infinity with infinite data. More data adds edges but never a cycle, so
> collecting more of it makes the problem worse, not better.**

Equivalently: **the finiteness of a reward model is bought entirely with annotator
disagreement.** Improve your rubric, train your raters, adjudicate your ties — and you have
*increased* the scale of the reward your data implies.

### Measured

Eight completions, strict ordering, all 28 comparisons, **and the fit re-centred at every
step** so the location gauge is fixed and the divergence provably is *not* the additive
constant:

| t | 100 | 1e3 | 2e3 | 1e4 | 2e4 | 1e5 | 2e5 |
|---|---|---|---|---|---|---|---|
| ‖r̂‖ | 6.2179 | 14.4230 | 17.7700 | 26.7964 | 31.0299 | 41.2068 | **45.6597** |

Not strongly connected. Monotone increasing across every decade. nll **2.086e-04**, pairwise
accuracy **exactly 1.0**, max margin **49.9676**.

**The control that makes it a result rather than an anecdote:** the *same* eight items, the
*same* 28 comparisons, relabelled at \(\varepsilon = 0.10\) → **strongly connected**, ‖r̂‖
**converges to 4.6967**. Relative change over the last decade: **1.081e-01** (still moving) vs
**0.000e+00** (converged); norms **45.66 vs 4.70**. Nothing about the reward or the items
changed. The difference is entirely whether the raters ever disagreed.

### And it is invisible

| decade | cos(r̂_t, r̂_2t) | ‖r̂_2t‖/‖r̂_t‖ |
|---|---|---|
| 1e3 → 2e3 | 0.9999873 | 1.2321 |
| 1e4 → 2e4 | 0.9999938 | 1.1580 |
| 1e5 → 2e5 | **0.9999985** | **1.1081** |

The **direction converges to the max-margin separator** while the **norm grows logarithmically
without bound** (Soudry et al., implicit bias of gradient descent — the same result that
explains why unregularised separable logistic regression "works"). Nothing errors, nothing
NaNs, the loss curve is beautiful, accuracy reads 100%, and the parameter you care about is
running to infinity.

---

## Headline 2 — noise is the regularizer, and it is the same result

Let true preferences follow BT with reward \(r^*\) and let the annotator flip independently at
rate \(\varepsilon\):

\[
  p = (1-2\varepsilon)\,\sigma(\Delta^*) + \varepsilon
\]

which is **not \(\sigma\) of anything** — it is squashed into \([\varepsilon, 1-\varepsilon]\).
In the **saturated** case (one free margin per pair) the MLE matches the empirical frequency,
so \(\hat\Delta = \operatorname{logit}(p)\), and as \(\Delta^* \to \infty\):

\[
  \boxed{\ \hat\Delta \longrightarrow \log\tfrac{1-\varepsilon}{\varepsilon}\ }
\]

Measured, exact win counts, agreement to **1e-14**:

| ε | 0.02 | 0.05 | 0.10 | 0.20 | 0.35 |
|---|---|---|---|---|---|
| fitted margin | 3.8918 | 2.9444 | 2.1972 | 1.3863 | 0.6190 |

**Two limits, and they unify the day.** \(\varepsilon \to 1/2\) gives \(\hat\Delta \to 0\) — no
information. \(\varepsilon \to 0\) gives \(\hat\Delta \to \infty\) — **which is Headline 1.**
The existence theorem is not a separate result; it is the \(\varepsilon \to 0\) limit of the
cap.

**Scope, stated honestly.** With \(n-1\) free parameters against \(\binom n2\) pair frequencies,
BT under flip noise is **misspecified**: every pair wants margin
\(\operatorname{logit}(1-\varepsilon)\) and no single ordering can deliver that, so the
parametric fit compromises. Spread on 8 items: **12.29 / 7.49 / 5.00 / 2.76 / 1.22** at
ε = .02/.05/.10/.20/.35 — finite and monotone, which is the part that matters, but *not* equal
to the cap.

⇒ **Day 4's IPO (bounded loss) and Day 3's cDPO (label smoothing) are two ways of putting back
by hand a cap that clean data removed.**

---

## Headline 3 — RM scale and β are one knob

\(\pi \propto \pi_{\text{ref}} e^{r/\beta}\) depends on \(r\) and \(\beta\) **only through
\(r/\beta\)**, so

\[
  \text{tilt}(c\,r,\ c\,\beta) = \text{tilt}(r, \beta) \quad \forall c > 0
\]

verified elementwise to **3.886e-16**. The policy is physically unable to distinguish "a reward
twice as large" from "a β half as large."

Chain it with Headline 2:

> agreement rate ⟹ scale of \(\hat r\) ⟹ meaning of β ⟹ the KL you land at

Measured — same β = 0.5, same \(\pi_{\text{ref}}\), same ranking, only the scale changing:

| reward scale | ×0.5 | ×1 | ×2 | ×4 |
|---|---|---|---|---|
| KL(π‖π_ref) | 1.0156 | 1.8047 | 2.6198 | 2.8802 |

> **β is a property of your annotation process.**

Strictly stronger than W4 D5's "β is a property of the RM's *data*, retune it when you improve
the RM." That one was about **where** the data is; this one is about **how noisy its labels
are**. Independent axes, both retune β. And it retires "just use 0.1" — a β tuned on one
annotation pipeline is meaningless on another, and nothing in the logs will say so.

Regression checks that still hold exactly: location is free in **both** the tilt and the NLL
(< 1e-15), and support confinement is exact — \(\pi^*[a] = 0.0\) where \(\pi_{\text{ref}}[a]=0\).

---

## Headline 4 — the DPO corollary, one week early

A preference dataset is \((x, y_w, y_l)\) triples in which each completion appears **exactly
once**: a **perfect matching**, the most disconnected non-trivial graph there is. Measured on
the same eight items:

| t | 1e3 | 1e4 | 1e5 |
|---|---|---|---|
| ‖r̂‖ | 8.7780 | 12.0436 | 15.3013 |
| min r̂ᵢ | −3.1035 | −4.2580 | **−5.4098** |
| min margin | 6.2070 | 8.5161 | 10.8196 |

Not strongly connected; nll **2.0e-05**; accuracy **1.0**. And the sharp bit:

> **In a perfect matching, annotator noise cannot help you at all.** Flipping a label just
> swaps which item won — a matching has no cycles under *any* labelling. Cycles require an item
> to appear in ≥ 2 comparisons. **Noise has nowhere to act.**

Which is the real argument for \(K\) completions per prompt and all \(\binom K2\) comparisons:
density is what gives disagreement somewhere to register. InstructGPT's choice is not only a
compute trick.

Tomorrow: DPO's implicit reward is \(\hat r = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}\), so
"\(r_l \to -\infty\)" reads literally as \(\pi_\theta(y_l|x) \to 0\), and the displaced mass has
to go somewhere the loss cannot see.

> **"DPO drives the rejected logprob to −∞" is not a bug report. It is Ford's theorem.**

---

## Headline 5 — the accuracy ceiling

Under BT the Bayes-optimal classifier is "predict the higher \(r\)", with accuracy
\(\mathbb E[\max(p, 1-p)]\). Measured with a genuinely stochastic annotator:

- Bayes ceiling **0.7675** (n = 11200, SE **0.0040**)
- the **gold** reward itself scores **0.7747** — it *is* the Bayes rule
- the fitted RM scores **0.7747** — reaches Bayes and **not a hair more**

Two consequences worth having ready:

1. **The ceiling is set by the annotator, not the model.** 96% here is impossible at any model
   quality. Report RM accuracy against the agreement ceiling or do not report it.
2. **Accuracy depends only on the SIGN of the margin**, so it is invariant to scale: \(\hat r\)
   and \(100\,\hat r\) have identical accuracy. It is structurally incapable of distinguishing
   a converged fit from one whose norm is running to infinity. Recall Headline 1: loss
   **2.1e-04**, accuracy **exactly 1.0**, ‖r̂‖ at **45.66 and climbing**.

---

## Interview answers (final)

### 1. "Your reward model gets 96% pairwise accuracy on held-out preference pairs. Is that good?"

**It is not reassuring; it is the alarm.**

Three things could be true. **(a)** The pair distribution is easy — the Bayes ceiling is near 1
because the true margins are large. **(b)** The annotators are near-deterministic: held-out
accuracy is capped by the label noise, so 96% is direct evidence that \(\varepsilon \lesssim
4\%\), which by Ford's condition is the near-separable regime. **(c)** The fit is diverging and
accuracy cannot tell you, because **accuracy depends only on the sign of the margin and is
invariant to scale**. My own fixture reaches loss 2.1e-04 and accuracy exactly 1.0 with ‖r̂‖ at
45.66 and still climbing.

**First thing I check: is ‖r̂‖ still growing with training steps?** Plot the norm, not the loss.
Then the margin distribution — bimodal and pushed to the extremes is the max-margin direction
with a runaway norm. Then the comparison graph: how many times does each completion appear, and
is the wins digraph strongly connected? Then get the Bayes ceiling from an agreement study and
report accuracy *against* it. Mine was **0.7675**, with the gold reward itself scoring 0.7747.

### 2. "We cleaned up our preference data — better rubric, adjudicated disagreements, inter-annotator agreement 68% → 91%. Same model, same β. Our DPO runs now collapse. What happened?"

**The chain.**
1. Agreement 68% → 91% is a flip rate going from ~0.32 to ~0.09.
2. Fewer upsets ⇒ the wins digraph sheds cycles ⇒ it approaches a transitive tournament ⇒
   Ford's condition fails on more and more partitions.
3. The fitted scale runs up. The per-pair cap moves \(\operatorname{logit}(0.68) = 0.7538 \to
   \operatorname{logit}(0.91) = 2.3137\) — a **3.07×** larger achievable margin, heading to
   unbounded as the last cycles disappear.
4. Only \(r/\beta\) is identified, so a 3.07× larger \(\hat r\) at fixed β **is the same object
   as the old \(\hat r\) at β/3.07**. You cut your KL budget 3× without touching a config file.
5. In DPO there is no separate RM whose scale you can inspect — the scale lives *inside the
   policy* as \(\beta\log(\pi_\theta/\pi_{\text{ref}})\). The only way to express a larger
   margin is to make \(\pi_\theta(y_l)\) smaller, and with the graph a DAG there is **no finite
   target**. The rejected logprob falls without limit and the mass goes somewhere the loss
   cannot see. My toy's KL *saturates* — 1.0156 → 2.8802 — which looks reassuring until you ask
   what it saturates *at*: a **point mass on the argmax**. In an LLM that is the degeneracy.

**Three fixes and their costs.**
- **Raise β ~3×.** Zero code. But it is a *global* rescale, so it damps genuinely large
  preferences as hard as marginal ones — and it does not restore existence. You are
  rate-limiting a divergence, not stopping it.
- **Label smoothing / cDPO at the measured ε = 0.09.** Restores a finite optimum at
  \(\hat\Delta^* = \frac1\beta\log\frac{1-\varepsilon}{\varepsilon} = 2.3137/\beta\) — exactly
  the cap, reached from the policy side. Cost: you paid a team to remove noise and are
  re-injecting it by decree, uniformly, when real annotator error is concentrated near ties.
- **A bounded loss (IPO).** Finite optimum for *any* dataset because it **pushes back** on
  over-large margins rather than merely slowing them. Cost: symmetric, so it caps genuine huge
  preferences as hard as marginal ones, and τ becomes another per-dataset knob.
- **And the one that is actually right: fix the graph, not the loss.** Collect \(K\) completions
  per prompt and all \(\binom K2\) comparisons. In a matching, noise has nowhere to act.

**Why it is not a paradox.** Quality and scale are different quantities and the fit only ever
measured the second. Cleaning the data improved the *reliability of the ranking* and in the same
stroke **destroyed the only signal that pinned the reward's magnitude**.

> **You cleaned your data and tripled your learning rate in reward-space without editing a
> config file.**

---

## Bugs hit, and what each generalises to

| bug | lesson |
|---|---|
| `closure ** k` for matrix power | **`**` is ELEMENTWISE.** On a bool array it upcasts to float64 and returns the array *unchanged* — silent, no warning, no shape change. The fix is boolean matmul (`@`), which numpy does logically so there is no overflow. Third sighting of *"the wrong operation is type-compatible"* (W3 D2's `np.zeros(dtype=np.long)`, W2 D2's `TransformerBlock(d, n_heads, n_heads)`) |
| `reward += lr * grad` on an NLL | **Gradient ascent on a loss.** Undetectable on separable data: an ascent bug and a descent bug both send ‖r‖ → ∞ and differ only in *sign*. Second sighting of W4 D4's **"a symmetric probe cannot detect a flip."** The asymmetric probe is data whose MLE is finite and known in closed form — 80/20 wins ⇒ logit(0.8) = **1.3863**, and the buggy fit returned **−317.66** |
| `history[step]` while testing `step+1 in record` | Test and key must use one convention. Loop from 1 rather than fixing it up at the use site |
| `np.random.random(n_items)` init | **Unseeded**, in a function whose entire purpose is a reproducible *trajectory comparison* — two identical calls gave −238.75 and −239.53. Should be `zeros`: every initial margin is 0 ⇒ every initial gradient weight is exactly σ(0)=½ ⇒ the run-away direction is purely a property of the data. (Same property as DPO's init, where π_θ = π_ref.) |
| `nll` returned from before the last update | One-step staleness. Same family as W3 D1's argmax-on-last-iteration-Q lag |
| `noisy_labels` computed the winner, then **overwrote** it from the raw columns | **A dead store that discarded `gold_r` entirely.** It survived six stages because `all_pairs` was emitted as (i, j) with i > j, so column 0 was *already* the gold winner and the dead code and the live code agreed. **Fourth sighting of "silent because two things that should differ happen to coincide"** (W1 D5's `T==V`, W2 D2's `n_heads`, W3 D4's `Batch(..., n_iters, rng)`). New rule: **when a function's output has the same type and shape as one of its inputs, test it on an input where "return the input" is wrong.** |

**Harness weakness, and it was mine.** A fixture on which the naive answer is correct is not a
fixture. `all_pairs` now emits a **scrambled column order** (raw column 0 has gold accuracy
**0.571**) and `graph` asserts both non-vacuity and that a deterministic annotator labels every
comparison correctly. Verified with the dead store reinstated: the suite fails at **stage 2
instead of stage 9**. Fifth time a scaffold design became the day's lesson (W1 D5
noiseless-overfit, W2 D5 toothless cached==uncached, W3 D3 oracle-baseline, W3 D4
λ=0-is-biased).

**Process win, repeated.** A private reference implementation was written and run *before*
handover. It caught three dishonest thresholds — a `rel()` divide-by-zero when the converged
fit's change is exactly 0 (**the same family as W4 D5's `diagnose` calibration bug**), a ceiling
tolerance the *gold* reward violated by 1.7 SE, and a 2m49s runtime — plus a fourth during
handover verification (a "loss → 0" threshold unreachable in `--fast`, now relative rather than
a constant).

---

## Checkpoint questions — answers

**1. Ford's condition, the "only if" proof, and does more data help?**
See Headline 1. **No.** More comparisons from a deterministic annotator add edges but never a
cycle, so no partition ever becomes contradicted and the supremum stays at infinity.

**2. Each completion appears exactly once — what is the graph, does the MLE exist, what does it
predict for DPO?**
A **perfect matching**. Not strongly connected, so the MLE is at infinity. Predicts an unbounded
implicit margin, i.e. \(\log\pi_\theta(y_l) \to -\infty\), with training accuracy 1.0 and loss
→ 0 throughout. Measured: min r̂ᵢ **−3.10 → −4.26 → −5.41**, min margin **10.82**.

**3. Derive the ε-cap; value at ε = 0.1; both limits.**
See Headline 2. At ε = 0.1 the cap is \(\log 9 = 2.1972\). ε → ½ gives 0 (no information);
ε → 0 gives ∞, which is question 1.

**4. Show tilt(cr, cβ) = tilt(r, β). Agreement 68% → 91%, nothing else changed — what happens
to KL, and why is it a prediction?**
See Headline 3 and interview answer 2. It is a *prediction* because the mechanism is an
identifiability argument with a closed-form magnitude (3.07×), not an observation — it says in
advance which direction KL moves and roughly by how much, for a change that touched no
hyperparameter.

**5. 96% pairwise accuracy — three things that could be true, and the one plot.**
See interview answer 1. The plot is **‖r̂‖ against training steps**.

---

## Session record

| block | score |
|---|---|
| Block 0 — cold recall (7 items) | **4.55 / 7** (65%) — up from W4 D5's 62.5%, D4's 56% |
| Block 1 — the three W4 D5 drills, cold | **1.45 / 3** (48%) — previously unattempted |
| "does the MLE exist?" instinct | 0.3 — gave the *location* invariance, which is the wrong pathology |
| "first thing you check at 96%?" | 0.4 — right object (the graph), inverted direction |
| "what is the scale measuring?" | 0.7 — right, but gave the correlate instead of the contrast |
| the cleaned-data drill | 0.25 — first link only, of three requested parts |
| coding | **74 / 74** |

**Retired today:** clipping ≠ trust region at **3/3 mechanisms** (token / sample / iteration —
the mnemonic worked); **RM noise** finally named as a REINFORCE variance source after five asks.

**Still open:** the same three variance sources (policy entropy, environment stochasticity,
off-policy-ness itself); the second-clause habit, fourth week; and the numbers gap, now on its
seventh session — mechanisms and live derivations are strong, *stored scalars* are not, so the
mechanism has switched from retrieval to **recognition** via `shared/numbers-card.md`, read
aloud at the start of every recall block.

## Reusable artifact

- **Path:** `week-05-dpo-preference-optimization/code/preference_models.py`
- **Reuse:** Day 2 imports `fit_bt` and `tilt` for the equivalence check
  (DPO ≡ tilt of the BT MLE); Day 3 reuses the ε sweep as the data-side half of the cDPO
  table; Day 4 reuses the **separable fixture** as the arm that breaks DPO and is fixed by
  IPO/cDPO; Day 5 reuses `wins_digraph` / `is_strongly_connected` as a **dataset diagnostic**.
