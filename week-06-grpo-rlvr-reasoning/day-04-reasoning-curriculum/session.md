# Week 06 Day 04 — Reasoning Data Curriculum: the set point is not 0.5

## Objective

**Read this first, out loud.** Three days ago you priced the group size `G`. Two days ago you
priced the verifier in units of `G`. Yesterday you found out that `G` and `k` are the same
variable and that pass@k is a capability composed with a selector. Today all three become one
object, and the object is a **controller**.

Start from the collision, because there is one again. Every day this week has produced a function
of `p`, and today they get multiplied together:

```
D1   signal      E[Σ|A|] = 2(G−1)·p(1−p)                     dies at BOTH walls
D2   precision   1 − P(hack|rewarded) = p(1−α)/p_obs          dies at the LOW wall
D3   metric      ∂pass@k/∂p = k(1−p)^(k−1)                    puts its weight at the LOW wall
```

**A curriculum is the policy that chooses `p`.** That is the whole of today. Not "order the data
easy to hard" — that is a *sort*, and a sort is an open-loop controller with no feedback, which is
why every hand-built curriculum decays after a few hundred steps. The question today answers is
the quantitative one: **what `p` should you train at, what stops you going there, and what makes
the answer move while you are training?**

Now the day's thesis, in three parts.

> **(1) The set point is `p* = 1/(k+1)`, where `k` is the budget you will be EVALUATED at.**
>
> Not 0.5. It is 0.5 **only at k = 1**, and 0.5 is the answer people give because they are
> implicitly maximising the *gradient*, which is a means, not an end. Maximise the thing you
> actually want — improvement in the reported metric per rollout — and you get
>
> ```
> maximise   progress(p) × weight_k(p)  =  [2(G−1)p(1−p)] × [k(1−p)^(k−1)]  ∝  p(1−p)^k
> ⇒  p* = 1/(k+1)          k = 1  → 0.5000    k = 8  → 0.1111    k = 64 → 0.0154
> ```
>
> and more generally, if the signal-to-progress map goes like `p^a(1−p)^b`, then
> **`p* = a/(a+b+k−1)`**. `(a, b)` is an **empirical** quantity of your stack, not a constant of
> nature — today you measure it rather than assume it. The invariant that survives any `(a,b)`:
> **`p*` is strictly decreasing in `k`, and it equals 1/2 only at `k = 1`.**

> **(2) The verifier does NOT move the set point. It scales the signal, and it can flip its sign.**
>
> This is the day's cleanest derivation and it kills a very natural wrong answer ("my verifier is
> unreliable on hard problems, so I should train on easier ones"). Let `s = ±1` be the *gold*
> label and `r ∈ {0,1}` the *verifier's* label. The advantage mass that lands on the truth is
>
> ```
> U(p) = E[ Σ_i A_i · s̃_i ]  =  2(G−1)·Cov(r, s)  =  2(G−1)·p(1−p)·(1 − α − γ)
>
> with the ±1 convention s̃ = 2s−1. Under s ∈ {0,1} you get exactly half:
> U = (G−1)·Cov(r, s) = (G−1)p(1−p)J, no factor 2. Say which one you are quoting.
> With the ±1 convention, `U = J·M_gold`, where
> `M_gold = 2(G−1)p(1−p)` is the counterfactual gold-reward advantage mass. The
> observed verifier-reward mass is instead `M_obs = 2(G−1)p_obs(1−p_obs)`, so the
> useful fraction is `U/M_obs`, not generally `J`.
> ```
>
> The verifier enters as **`J = 1 − α − γ`, Youden's J — a multiplicative constant with no `p` in
> it.** So `argmax_p U` is exactly `argmax_p p(1−p)`, whatever the verifier is. `J = 0`
> (`α + γ = 1`) gives **identically zero** useful signal at every difficulty, and `J < 0` gives
> **negative** signal — the run trains away from gold at every `p`. Verified by Monte Carlo today
> at `α=0.6, γ=0.5`: **U = −0.2903 against a closed form of −0.2940.**
>
> What *does* move with `p` is the **variance**, not the mean: the useful fraction of the total
> signal is `p(1−p)J / [p_obs(1−p_obs)]`, and as `p → 0` that goes to zero **linearly in p** while
> the total stays at `γ(1−γ)`. **The verifier does not tilt your curriculum, it puts a FLOOR under
> it** — and the floor is D2's selection law solved for `p`.

> **(3) You already have a curriculum you did not choose, and it is the `/std` in GRPO.**
>
> Whiten the advantages and every **alive** group carries exactly `Σ A² = G`, for any `c`. So the
> whitened objective weights a prompt by `P(alive) = 1 − p^G − (1−p)^G` **alone** — a function
> that is *flat in the middle and falls off at both walls*, i.e. a difficulty weighting. Relative
> to centering it tilts toward the extremes by up to **`G/4`** — measured **2.0157× at G = 8** and
> **16.0000× at G = 64** — and it drags the optimal *explicit* set point down with it: at
> `G = 64, k = 4` the whitened optimum is **0.0474 against 0.2000**, a factor of **0.237**.
>
> This is Dr. GRPO's critique of GRPO with a different name on it. Third sighting of the same
> object: W4 D3 saw it as **variance reweighting across groups**, W5 saw it as a **per-batch LR
> rescale**, and today it is a **curriculum**. Same division, three readings.

And the constraint that makes today an engineering day rather than an optimisation exercise:

> **The curriculum you want and the curriculum you can LABEL are different, and they part company
> at a `k` you can compute.** D2's selection law, solved for the `p` at which you still tolerate a
> hack rate `τ`:
>
> ```
> p_floor  =  R/(1+R),   R = γ(1−τ) / [τ(1−α)]        γ=0.01, α=0.10, τ=0.10  ⇒  p_floor = 1/11
> ```
>
> and `1/(k+1) = 1/11` at **exactly k = 10**. Below k = 10 the metric's optimum is inside the
> verifier's safe zone; above it, the floor binds and you are training somewhere you did not
> choose. The bill: at k = 64 the floor keeps you at `p = 0.0909` where the optimum is `0.0154`,
> and you retain **3.58% of the metric-weighted signal**.

Four things get proved and measured today, and each is a sentence you can say in an interview:

**(a) `p* = 1/(k+1)`, and the general family `a/(a+b+k−1)`.** Verified numerically at 16 points.
The corollary is a group size: keeping 90% of groups alive at `p*` needs
`G = ln(0.1)/ln(1−p*) ≈ 2.3026·k` — **20 at k = 8, 149 at k = 64**. **The metric you report sets
your group size.** And the flip side, which is the number to be alarmed by: at `p*` with a
*standard* `G = 8`, the dead-group fraction is **38.97% / 61.57% / 88.34%** at k = 8 / 16 / 64.
*Training at the difficulty your metric wants, with the group size everybody uses, throws away
most of your compute.*

**(b) The band is MULTIPLICATIVE, not additive.** Keep the prompts within 50% of the peak value
`p(1−p)^k` and the band's *width* shrinks with k (0.707 → 0.037 from k = 1 to k = 64) while its
**ratio** `hi/lo` grows and settles at about **11×**. A decade of `p`, wherever the set point is.
Consequence for the controller: **act on the log-odds of `p`, not on `p`.** A ±0.05 tolerance is
meaningless at `p* = 0.0154` and meaningless at `p* = 0.5`, for opposite reasons.

**(c) The cliff and the ceiling are the SAME theorem, with the sign flipped.** Yesterday's null
result — `coverage_collapse` never fired, because on a prompt whose gold is never sampled every
rollout scores the same and the group-relative advantage is identically zero — read *forwards*
says: **the algorithm cannot ACQUIRE coverage it is not sampling either.** The dead group is
symmetric. It protects you from unlearning and it prevents you from learning. So a curriculum's
real job is to keep `p` off **both** walls, and *the two walls need opposite fixes*: scaffolding
at the bottom, harder problems at the top. A hint that moves `p` from 0.01 to 0.15 multiplies the
signal by **12.88×** and cuts the dead fraction at G = 8 from **92.27% to 27.25%** — the same
effect as raising `G` from 8 to 230, for free.

**(d) Contamination inflates pass@1 and is INVISIBLE at large k — the exact mirror of yesterday's
training gain.** With 5% contaminated items on a set whose honest pass rate is 0.30, the inflation
is **3.500 / 1.201 / 0.288 / 0.000 points** at k = 1 / 4 / 8 / 64. Both facts are the same
statement — *a saturating metric cannot see a change in something already saturated* — and they
point in opposite directions: RL's gain shows up at small k, and so does contamination's lie.
**Reporting only pass@1 maximises your exposure to both.** And the *training* cost of
contamination is D1's: a memorised item is a **saturated dead group**, so a contaminated fraction
`c` is exactly `c` of your rollout budget returning zero gradient.

Build `code/curriculum.py` to make all of this assertions rather than claims.

## Prerequisites

- **W6 D1** — the dead-group law `p^G + (1−p)^G`, `G = ln(0.1)/ln(1−p)`, the exact `(G−1)/G`
  shrinkage, the `1/std` amplification, and `grpo.py` as an importable module (`TokenEnv`,
  `SeqTable`, `build_seq_table`, `make_task`, `group_advantage`, `train_grpo`, `RunCfg`,
  `exact_expectation`, `exact_entropy`, `exact_kl`, `exact_pass_rate`).
- **W6 D2** — `V ⊆ C` / `C ⊆ V`, the selection law `P(hack|rewarded) = γ(1−p)/[p(1−α)+γ(1−p)]`,
  `p_observed(p,α,γ) = p(1−α) + (1−p)γ`, the Rényi generalisation, and `rlvr.py`
  (`make_checker`, `verifier_rates`, `dead_fraction_levels`, `g_for_alive`, `p_observed`).
- **W6 D3** — `pass@k = 1 − (1−p)^k`, the unbiased estimator, **`∂pass@k/∂p = k(1−p)^(k−1)`**
  (today's weight function, and it is the entire reason the answer is not 0.5), the three
  selectors and their three limits, and `sampling.py` (`pass_at_k`, `pass_at_k_exact`, `dpass_dp`,
  `selector_curves`, `temper`, `top_p_truncate`).
- **W4 D2** — open-loop vs closed-loop control, which is the vocabulary today reuses wholesale.
  You already said "a KL in the reward is an open-loop controller." Today: *a sorted dataset is an
  open-loop curriculum, and it fails for the same reason.*
- **W4 D4** — the four-way failure taxonomy and "≥3 seeds is mandatory"; W5 D5's upgrade of that
  to **≥5 seeds for tail-sensitive expectations, paired**.

Today's file **imports all three of grpo / rlvr / sampling** — fourth day in a row, and by now
that is the point rather than a convenience. If any import needs a change to the imported file,
that is a signal the earlier day got an interface wrong; note it, do not paper over it.

## Required Reading

- **DeepSeek-R1 (2025), §2.3.1 "Cold Start" and §2.3.3 "Rejection Sampling and SFT."** Read it as
  a *curriculum specification*, not as a system description: every stage listed there is an
  intervention on `p`. Ask, for each stage, which wall it is pushing away from.
- **STaR (Zelikman et al. 2022), §2 — specifically "rationalization."** This is scaffolding in
  its purest form: when the model fails, hand it the answer and ask for the reasoning. It is
  precisely the "move `p` off zero" move, and the paper's own failure discussion is the
  anneal problem. One page.
- **Prioritized Level Replay (Jiang et al. 2021), §3.** The closed-loop curriculum, from RL rather
  than from LLMs, and its scoring function is *literally the magnitude of the advantage*. Read the
  scoring section and notice it is today's `2(G−1)p(1−p)` with a different derivation.
- **GSM1k — "A Careful Examination of LLM Performance on Grade School Arithmetic" (Zhang et al.
  2024).** The perturbed-twin methodology and the size of the drop. Read the *method* first and
  predict the control they had to run before you read what they found.
- Optional: **Teacher–Student Curriculum Learning (Matiisen et al. 2017)** for "learning progress"
  as an explicit objective, and why the naive version chases noise.

## 2-Hour Plan

> **Clock note, unchanged and non-negotiable.** Blocks are delivered **in full**, however long the
> session runs. Nothing is compressed for the clock.

> **Rendering.** All maths in this file is plain Unicode. No LaTeX anywhere.

> **Four standing rules, every one of them earned by direct test on D1–D3:**
> 1. **Skeletons are PRE-PRINTED and every slot is TYPED** (D1: 0/7 when told, 4/4 when emitted;
>    D2: an untyped `____` scores like an open question).
> 2. **A numeric slot prints the FORMULA, which argument VARIES, which are HELD FIXED, and the
>    units/precision** (D3, his catch: a type constrains the codomain, it does not identify the
>    function; re-issued that way, 6/6).
> 3. **The NUMBER slot comes FIRST in every skeleton** (D2's clean signature: 5/5 definitional
>    slots filled, 0/5 number-and-consequence slots).
> 4. **"MY OWN EVIDENCE" is ASSIGNED, never chosen** (D3: given a four-item menu he returned the
>    menu; the slot bundles an authorship judgement with a technical skill and the authorship half
>    blocks the technical half). One named result, then `number → provenance → scope → caveat`.

### Block 0 · 0:00–0:05 — Numbers-card drill (RECOGNITION, not retrieval)

Open `shared/numbers-card.md`, read the **Week 6 Day 3** section out loud start to finish, then
close it. Three questions immediately after, closed-book.

1. **The metric's reweighting.** Formula: `∂pass@k/∂p = k·(1−p)^(k−1)`. Varying **k**; the two
   problems are held fixed at `p_hard` and `p_easy` with `(1−p_hard)/(1−p_easy) = 4.75`.

   ```
   the hard/easy marginal-value ratio, as a formula in k: ______________________
   at k=1: [a number, 1 dp] ______   at k=8: [a number, 3 sig figs] ______
   at k=16: [a number, 3 sig figs] ______
   at k=1, the share of the metric carried by EACH of 6 equally-spaced problems:
     [a percentage, 1 dp] ______
   at k=256, the share carried by the two HARDEST: [a percentage, 4 dp] ______
   the one-sentence consequence: [complete the sentence] "changing k changes ____________________"
   ```

2. **Filter vs ranker.** Formulas: filter composition
   `P(hack|accepted emission) = γ(1−p)/[p(1−α)+γ(1−p)]`; filter output risk
   `[1−(1−p_obs)^n]P(hack|accepted emission)`; ranker
   `P(any top-scored hack) = 1 − [1−(1−p)γ]^n`. Varying **n**; held fixed at
   `γ = 0.01, α = 0.10, p = 0.10`.

   ```
   FILTER output at n = 1 / 8 / 64 / 256: [four percentages, 4 dp] ______ ______ ______ ______
   FILTER accepted-output composition: [percentage, 4 dp] ______; property: ______________
   RANKER at n = 1 / 8 / 64 / 256: [four percentages, 4 dp] ______ ______ ______ ______
   the ratio at n = 256: [a number, 2 dp] ______×
   the ranker's formula IS pass@n for [a noun phrase] ____________________
   ```

3. **The rule that never fired.** Statistic: `min over all runs of [pass@k_max(final) −
   pass@k_max(0)]`, over 24 configurations then 8 seeds × 3 fixtures.

   ```
   the value: [a number, in scientific notation with its sign] ______________
   the MECHANISM, in one sentence about the group advantage: ____________________
   the SCOPE condition: [a sentence beginning "holds only for..."] ____________________
   what it licenses about the claim "RL reduces pass@k": ____________________
   ```

### Block 1 · 0:05–0:25 — Cold recall (skeletons pre-printed, no notes, no scrollback)

**Standing rule from D2: cold recall tests PRIOR material only.** Nothing below is taught today.

1. **The dead-group law and its inverse.** Formula: `P(dead) = p^G + (1−p)^G`. Varying `p` at
   fixed `G = 8`; then the inverse, varying nothing.

   ```
   at p = 0.05: [a percentage, 4 dp] ______   at p = 0.50: [a percentage, 4 dp] ______
   the closed form for the G that keeps 90% of groups alive: G = ______________________
   at p = 0.05 that is G = [an integer] ______
   the symmetry of P(dead), stated as an equation: ______________________
   and the thing that BREAKS the symmetry (one noun, from D3): ______________
   ```

2. **The shrinkage.** Formula: the exact factor relating the group-mean baseline's advantage to
   the true-mean baseline's. Varying `G`.

   ```
   the factor: ______________________
   at G = 8 that is [a number, 4 dp] ______ ; at G = 2, [a number] ______
   what it does to the GRADIENT, in one clause: ____________________
   why it is NOT a bias (one sentence, and the word "measurable" should appear): ______________
   ```

3. **The selection law, D2.** Formula: `P(hack|rewarded) = γ(1−p)/[p(1−α)+γ(1−p)]`. Varying `p`;
   held fixed at `γ = 0.01, α = 0.10`.

   ```
   p = 0.500: [a percentage, 4 dp] ______   p = 0.100: [a percentage, 4 dp] ______
   p = 0.020: [a percentage, 4 dp] ______   p = 0.005: [a percentage, 4 dp] ______
   the limit as p → 0: [a number] ______
   the one-sentence consequence FOR A CURRICULUM: [complete] ____________________
   ```

4. **W3, the estimator question, because today is another instance.**

   ```
   E[Σ_i (X_i − X̄)²] over G iid draws equals [a formula in G and σ²] ______________
   therefore the SAMPLE variance with denominator G is [biased / unbiased] ______
   and the (G−1) in today's formulas is the same object as ____________________
   ```

5. **Sources of variance in REINFORCE — SEVENTH ask.** Seven slots; the record is four. One is
   identically zero for an LLM.

   ```
   1. ______  2. ______  3. ______  4. ______  5. ______  6. ______  7. ______
   the one that is identically zero for an LLM, and the one-clause reason: ____________________
   ```

6. **MY OWN EVIDENCE — SEVENTH ask, and per D3's rule it is ASSIGNED, not chosen.** The result is
   **yesterday's `coverage_collapse` null**. Number first.

   ```
   the number, with units:  [a number] ____________________
   where it came from:      [file, stage, #configurations, #seeds] ____________________
   the SCOPE condition:     [a sentence beginning "holds only when..."] ____________________
   the CAVEAT:              [a sentence beginning "it does NOT establish..."] ____________________
   ```

### Block 2 · 0:25–0:50 — Concept: a curriculum is a controller, and its set point is a number

**Framing.** The standard telling is "order your data easy to hard." Three things are wrong with
that sentence and each one is a separate result today.

**Wrong thing 1: "difficulty" is not a property of a problem.** `p` is a property of the *pair*
(problem, current policy). It is a *state variable*, and it moves in exactly the direction that
takes a problem out of the useful band. A static sort is therefore an **open-loop controller**:
it commits to a schedule computed once, at step 0, against a plant it will never observe again.
You have already said the analogous sentence about a KL in the reward (W4 D2). Same failure, same
fix: measure `p` and close the loop.

**Wrong thing 2: "easy to hard" names a direction, not a target.** There is a target and it is a
number. Write down what you actually want — improvement in the reported metric per rollout — and
maximise it:

```
value(p)  =  progress(p) · weight_k(p)
progress(p) ∝ 2(G−1)·p(1−p)          the advantage mass the group produces      [D1]
weight_k(p)  = k·(1−p)^(k−1)          what a unit of progress is worth to pass@k [D3]
⇒ value(p) ∝ p(1−p)^k                 ⇒  p* = 1/(k+1)
```

Say the assumptions out loud before you believe the formula, because one of them is doing all the
work:

```
H1  binary verifiable reward, group-CENTERED advantage (no /std)
H2  a LOCAL statement: one gradient step, Δp small
H3  problems are INDEPENDENT — no transfer between them
H4  the eval is mean pass@k with the SAME k on every problem
H5  Δp ∝ signal — the map from advantage mass to actual movement in p is the same at all p
```

**H5 is the suspicious one and it is today's measurement.** In a two-outcome tabular softmax,
exact gradient ascent gives `dp/dt ∝ p²(1−p)²`, not `p(1−p)` — the exponents are `(a,b) = (2,2)`
and `p*` becomes `2/(k+3)`, which at k = 8 is **0.1818, not 0.1111**. The general family is

```
progress(p) ∝ p^a (1−p)^b     ⇒     p* = a / (a + b + k − 1)
```

verified numerically at 16 `(a,b,k)` points today. **So the deliverable is not the number
`1/(k+1)`; it is the procedure: measure `(a,b)` on your stack, then read off `p*`.** What
survives every choice of `(a, b)`: **`p*` is strictly decreasing in `k`, and it is 1/2 only at
`k = 1`.** H3 is the one that makes this a toy — real curricula work because of *transfer*, which
this model does not contain, and you should say so before anyone else does.

**Wrong thing 3: the band is not a ±.** Take the prompts within 50% of the peak value:

```
k       p*         band [lo, hi]                width      hi/lo
1     0.500000   [0.146447, 0.853553]         0.707106     5.83
4     0.200000   [0.050366, 0.451048]         0.400682     8.96
8     0.111111   [0.026940, 0.270773]         0.243834    10.05
16    0.058824   [0.013963, 0.149912]         0.135950    10.74
64    0.015385   [0.003590, 0.040676]         0.037086    11.33
```

The absolute width collapses by **19.07×** from k = 1 to k = 64; the ratio grows and settles near
**11×**. **The band is one decade of `p`, wherever it sits.** So a controller with a fixed
absolute tolerance is wrong at both ends, and the right control variable is `logit(p)`, in which
the band is a constant-width interval. This is the same lesson as W5 D5's "ask whether the
signature is a LEVEL or a CHANGE", one level up: *ask what the natural scale of your control
variable is before you write a threshold in it.*

**And the wall you cannot see from inside the formula.** `value(p) ∝ p(1−p)^k` is maximised at
`p* → 0` as `k → ∞`. The unconstrained optimum is degenerate: it says train on problems you
cannot solve. It is the *constraints* that make a curriculum, and every one of them is a previous
day of this week:

```
D1  the dead group     G must grow like 2.3/p*, or most of your rollouts return zero gradient
D2  the precision floor p ≥ p_floor(γ, α, τ), or most of what you reward is a hack
D3  the metric          p* = 1/(k+1) pulls DOWN, and it is the only force that does
```

### Block 3 · 0:50–1:00 — Derivation (1): the signal function, and the covariance law

Two derivations, both short, and the second is the one to remember.

**(i) The signal function.** Group of `G` rollouts, binary reward, `c` successes,
`c ~ Binomial(G, p)`, centered advantage `A_i = r_i − r̄`.

```
given c:   Σ_i |A_i| = c·(G−c)/G + (G−c)·c/G = 2c(G−c)/G
           Σ_i A_i²  = c((G−c)/G)² + (G−c)(c/G)² = c(G−c)/G
E[c(G−c)] = G·E[c] − E[c²] = G²p − (Gp(1−p) + G²p²) = G(G−1)·p(1−p)
⇒  E[Σ|A|] = 2(G−1)·p(1−p)          E[ΣA²] = (G−1)·p(1−p)
```

Both exact (verified to machine precision today at three `p` values). Note what appeared without
being invited: **the `(G−1)` is D1's shrinkage**, and it is the same `(G−1)` as the unbiased
sample variance. Do not memorise three facts; memorise that they are one fact.

**(ii) The covariance law — the derivation of the day.** Let `s_i = +1` if rollout `i` is
gold-correct and `−1` otherwise, and let `r_i` be the *verifier's* label. Define the useful
signal as the advantage mass that lands on the truth, `U = E[Σ_i A_i s_i]`:

```
Σ_i (r_i − r̄) s_i  =  Σ_i r_i s_i − (1/G)(Σ_i r_i)(Σ_j s_j)
E[(Σ r)(Σ s)] = G·E[rs] + G(G−1)·E[r]E[s]            (G diagonal terms, G(G−1) independent pairs)
⇒  U = G·E[rs] − E[rs] − (G−1)E[r]E[s]  =  (G−1)·Cov(r, s)      [s ∈ {0,1}]
    and with s̃ = 2s−1 every term doubles:  U = 2(G−1)·Cov(r, s)
```

then evaluate the covariance, which is four lines and no cleverness:

```
E[r] = p(1−α) + (1−p)γ = p_obs        E[s] = 2p − 1        E[rs] = 2p(1−α) − p_obs
Cov  = 2p(1−α) − p_obs − p_obs(2p−1) = 2p[(1−α) − p_obs] = 2p(1−p)(1 − α − γ)
⇒  U(p) = 2(G−1)·p(1−p)·(1 − α − γ)
```

Now answer these before moving on:

1. `1 − α − γ` has a name in the diagnostics literature. Give it, and say what it equals for a
   coin-flip verifier and for a perfect one.
2. The factorisation is `[difficulty] × [verifier]` with **no interaction term**. State the
   consequence for the curriculum in one sentence, and then state the consequence you would
   *wrongly* have expected.
3. `J < 0` is a verifier that is *worse than a coin*. What does the run do? Give the sign of the
   gradient and say what it converges to.
4. The mean factorises; the **signal-to-noise ratio** does not. Write `U/E[Σ|A|]` under the
   verifier's own labels and take `p → 0`. Which one of "mean" and "variance" is the reason low-`p`
   training fails?

### Block 4 · 1:00–1:20 — INTERVIEW DRILL (MANDATORY — this block does not get cut)

Rules: asked cold, answered **out loud**, no notes, no scrollback, 60–120 seconds each. I grade
after. Model answers come **after** yours, as a diff — never instead.

**You write the entry into `interview-bank/rl-post-training-qa.md`.** On D2 you asked me to write
five entries and I did; the reference value is intact but the retrieval value of those five is
spent. The line holds: **I write at most one as a worked model, and you write the rest.** An empty
entry means the block did not happen.

Three questions, and I will stub them in the bank before the block:

1. **"How would you design the data curriculum for an RLVR run?"** The median answer says "easy to
   hard, and filter out the ones the model always gets right." A good answer names a **set point**
   with a formula, says what the set point depends on (**the k you report**), names the **two
   walls and their opposite fixes**, and says the word **closed-loop**. A great answer adds that
   the `/std` already imposes a curriculum you did not choose.
2. **"Your GRPO run's reward has been flat since step 0. Walk me through the diagnosis."** The
   correct answer distinguishes the **zero-reward cliff** from a **saturated set** from a **broken
   verifier** from a **broken loss** — and gets there with the *cheapest* measurement first.
   Required: `dead_frac` alone cannot separate the first two (D3's identity: it is symmetric under
   `p ↔ 1−p`), so name the one extra number that can, and say what each one implies you do next.
3. **"How do you know your reasoning benchmark isn't contaminated, and what does contamination do
   to TRAINING as opposed to eval?"** Required: the perturbed-twin method, **the control that
   converts it into evidence**, the `pass@1` vs `pass@k` signature with its direction, and the
   training answer — a memorised item is a *saturated dead group*, so it costs exactly its share
   of the rollout budget and returns zero gradient.

### Block 5 · 1:20–1:45 — Derivations (2)–(5)

**(2) The set point, and the family.** Maximise `value(p) ∝ p^a (1−p)^{b+k−1}`:

```
d/dp [ p^a (1−p)^{b+k−1} ] = 0  ⇒  a(1−p) = (b+k−1)p  ⇒  p* = a/(a + b + k − 1)
(a,b) = (1,1):  p* = 1/(k+1)      = 0.5000 / 0.2000 / 0.1111 / 0.0588 / 0.0154  at k = 1/4/8/16/64
(a,b) = (2,2):  p* = 2/(k+3)      = 0.5000 / 0.2857 / 0.1818 / 0.1053           at k = 1/4/8/16
```

Then the corollary that turns it into a config file. Keeping 90% of groups alive at `p*` needs

```
G = ln(0.1)/ln(1 − p*)  =  ln(10)/ln(1 + 1/k)  →  2.3026·k        (the limit, from above)
k  =  1        2        4        8       16       64      256
G  =  3.3219   5.6789  10.3189  19.5494  37.9810  148.5138  590.6123    ⇒ ceil: 4 / 6 / 11 / 20 / 38 / 149 / 591

**CORRECTION, measured in Block 6.** That row is single-sided: ln(0.1)/ln(1−p*)
counts only the all-FAIL tail. At k=1 the set point is 1/2, where the all-PASS
tail is exactly as large, so it undercounts deaths by 2× and gives 4 where the
honest criterion p^G + (1−p)^G ≤ 0.1 needs **5**. By k=2 (p* = 1/3) the easy tail
is 1.5% of the total and the two agree forever after. So the true row is

    5 / 6 / 11 / 20 / 38 / 149 / 591

This is your own W6 D1 numbers-card correction — the two-branch
`G = ln(0.1)/ln(max(p, 1−p))` — recurring, and it recurs at *precisely* the one
k where the set point sits at 1/2. The approximation G* ≈ 2.3026k is unaffected.
2.3026k = 2.30  4.61   9.21    18.42    36.84   147.37   589.46
```

and the number to be alarmed by — the *same* set points with the `G = 8` everybody actually uses:

```
P(dead | G=8, p=p*)  =  0.78% / 3.92% / 16.78% / 38.97% / 61.57% / 88.34%   at k = 1/2/4/8/16/64
```

**If you report pass@64 and train at the difficulty that maximises it with G = 8, 88% of your
groups return exactly zero gradient.** State the two ways out (raise `G`, or raise `p` and accept
a worse-weighted signal) and say which one you would pick and why.

**(3) The precision floor, and the crossover.** Take D2's selection law, demand
`P(hack|rewarded) ≤ τ`, and solve for `p`:

```
γ(1−p) ≤ τ[p(1−α) + γ(1−p)]  ⇒  p/(1−p) ≥ γ(1−τ)/[τ(1−α)] = R  ⇒  p_floor = R/(1+R)

γ=0.01, α=0.10, τ=0.10 → p_floor = 0.090909      crossover with 1/(k+1) at k = 10.0000 exactly
γ=0.01, α=0.10, τ=0.05 → p_floor = 0.174312      crossover at k = 4.74
γ=0.03, α=0.10, τ=0.10 → p_floor = 0.230769      crossover at k = 3.33
γ=0.10, α=0.10, τ=0.10 → p_floor = 0.500000      crossover at k = 1.00
```

and the bill for being pinned at the floor, as a fraction of the available metric-weighted signal
`v(p) = p(1−p)^k`:

```
k          8        10        16        32        64
v(floor)/v(p*)   0.9793   1.0000    0.8872    0.3804    0.0358
```

**At k = 64 the verifier's precision costs you 96.4% of the signal your metric was asking for.**
The last line of the last row is the interview sentence: *a 1% false-positive rate is a
constraint on your CURRICULUM, not just on your reward.* Note also the fourth row: a verifier with
`γ = 0.10` has `p_floor = 0.5`, which means **it cannot safely be used below the median difficulty
at all** — and yet its Youden `J = 0.80` is perfectly respectable. *Accuracy is not the property
that matters; the floor is.*

**(4) The implicit curriculum in `/std`.** Whiten within the group,
`A_i = (r_i − r̄)/σ`, `σ² = (1/G)Σ(r_i − r̄)²`:

```
Σ_i A_i²  =  Σ(r_i − r̄)²/σ²  =  G          for EVERY alive group, any c
Σ_i |A_i| =  2√(c(G−c))                     ∈ [2√(G−1), G]   — a factor of only 1.51 at G = 8
⇒ E[ΣA²]_whiten = G·P(alive) = G·(1 − p^G − (1−p)^G)      vs     E[ΣA²]_center = (G−1)p(1−p)
```

Verified today to machine precision at eight `(G, p)` points. Now take the ratio of the two as a
function of `p` — that is the reweighting whitening applies relative to centering:

```
ratio(p) = G(1 − p^G − (1−p)^G) / [(G−1)p(1−p)]
G= 8:  ratio(0.05) = 8.0982   ratio(0.5) = 4.5357   tilt = 1.785×,  limit p→0 = 2.0157×
G=64:  ratio(0.05) = 20.5843  ratio(0.5) = 4.0635   tilt = 5.066×,  limit p→0 = 16.0000×
limit as p→0, in closed form:  G / [4(1 − 2^(1−G))]  ≈  G/4
```

and the consequence for the *explicit* curriculum — the whitened set point solves
`argmax_p P(alive)·(1−p)^{k−1}`, which is **not** `1/(k+1)`:

```
              k = 1      k = 4      k = 8      k = 16     k = 64
G= 8  p*_wh   0.5000     0.1499     0.0909     0.0520     0.0148      (× 1.00 / 0.75 / 0.82 / 0.88 / 0.96)
G=64  p*_wh   0.4428     0.0474     0.0356     0.0256     0.0109      (× 0.89 / 0.24 / 0.32 / 0.44 / 0.71)
```

**Your advantage normalisation moves your optimal curriculum by up to 4×.** Say that sentence with
the number attached, and then say the general one: *any per-group rescaling is a reweighting of
the problem distribution, i.e. a curriculum, whether or not you call it one.* Third sighting —
W4 D3 (variance reweighting across groups), W5 (per-batch LR rescale), W6 D4 (curriculum).

**(5) Contamination, and why it is the mirror of yesterday.** A contaminated item is one the model
has memorised: `p ≈ 1`, and — this is the part that matters — `p ≈ 1` *for a reason that does not
generalise*. Two consequences, with opposite visibilities:

```
EVAL:      inflation of mean pass@k  =  c · (1 − pass@k(p_true))        ← DECAYS in k
           c = 0.05, p_true = 0.30:  3.500 / 2.450 / 1.201 / 0.288 / 0.017 / 0.000 points
                                     at k = 1 / 2 / 4 / 8 / 16 / 64
TRAINING:  a memorised item is a SATURATED dead group (c = G, unanimous success)
           ⇒ exactly c of the rollout budget returns zero gradient, and it is INVISIBLE
             on the reward curve because those groups score 1.0
```

Put yesterday's number next to today's and say what they have in common:

```
D3  RL's pass@k gain:      +0.4691 at k=1  →  +0.0000 at k=1024      (decays in k)
D4  contamination's lie:   +0.0350 at k=1  →  +0.0000 at k=64        (decays in k)
```

**A saturating metric cannot see a change in something already saturated** — that single sentence
generates both rows. And the practical consequence is uncomfortable: *the metric that shows your
improvement most clearly is also the one that shows contamination's inflation most clearly, and
you cannot have one without the other.*

Detection, and the control. The perturbed twin: take the item, change the surface (numbers, names,
order) without changing the reasoning, and measure `Δ = p(original) − p(twin)`.

```
honest item:     Δ ≈ 0
memorised item:  Δ ≈ 1 − p_true
```

**The control is what converts this from an observation into evidence** (W5 D5's standing lesson,
and GSM1k's whole methodology): apply the *same* perturbation to items you know are clean. Without
it you are measuring the difficulty of your perturbation, not memorisation, and a perturbation that
is itself hard will convict every item on the list.

### Block 6 · 1:45–2:45 — Coding: `week-06-grpo-rlvr-reasoning/code/curriculum.py`

Staged harness, W4/W5/W6 convention. **Imports `grpo.py`, `rlvr.py` and `sampling.py`** — same
`TokenEnv(vocab=3, t_max=3)`, 15 enumerable sequences — so every expectation is checked against
exact enumeration and nothing this week is re-derived.

TODOs (yours — four, and the diagnostic is again one of them):

1. **`signal_curve(p, G, mode)`** → exact `E[Σ|A|]` and `E[ΣA²]` over `c ~ Binomial(G, p)` for
   `mode ∈ {"center", "whiten"}`, by enumeration over `c`, vectorised over `p`.
   **Contract:** `center` matches `2(G−1)p(1−p)` and `(G−1)p(1−p)` to `< 1e-15`; `whiten` matches
   `G·P(alive)` to `< 1e-15`; and **dead groups contribute exactly zero, not NaN** — that is the
   trap, because `σ = 0` on a dead group and you will divide by it. Skip the term, do not epsilon
   it: `c ∈ {0, G}` has advantage identically 0 whatever you divide by, so the *right* denominator
   is "no term at all". (HANDOFF: *the fix for a zero denominator is the right denominator, not an
   epsilon on the wrong one* — this is the cleanest instance of that rule you will get.)

2. **`useful_signal(p, alpha, gamma, G)`** → `U = E[Σ A_i s_i]` by exact enumeration over the joint
   of (gold, verifier) within a group, **not** by Monte Carlo. Returns `U`, the total mass
   `E[Σ|A|]` under the verifier's own labels, and their ratio.
   **Contract:** `U = 2(G−1)p(1−p)(1−α−γ)` to `< 1e-12`; `U = 0` **identically in p** on the
   `α+γ = 1` line (that is the negative control and it must be exact, not "small"); `U < 0`
   everywhere when `α+γ > 1`; and the ratio `→ 0` linearly as `p → 0` while `U/[p(1−p)]` stays
   constant. **The trap is the level, eleventh sighting:** `U` is a per-*group* expectation and
   the benchmark quantity is the mean over prompts — do not average `p` first.

3. **`set_point(k, a, b, G, alpha, gamma, tau, mode)`** → the constrained set point. Returns the
   unconstrained `a/(a+b+k−1)`, the precision floor `R/(1+R)`, the whitened optimum when
   `mode == "whiten"` (numerically, there is no closed form), the final `p*`, and — the part that
   is actually the deliverable — **which constraint binds**, as a string.
   **Contract:** the analytic and grid optima agree to the grid resolution at 16 `(a,b,k)` points;
   the binding-constraint string flips from `"metric"` to `"precision"` at **k = 10** for
   `(γ, α, τ) = (0.01, 0.10, 0.10)`, and that crossover is asserted **exactly**, by solving, not
   by scanning a coarse grid and reporting the first hit. Also return the **band** at 50% of peak
   as `(lo, hi)` and assert `hi/lo → ~11` while `hi − lo` shrinks — the multiplicative-band result.

4. **`diagnose_curriculum(history)` — the diagnostic TODO, deliberately yours for the fourth day.**
   Contract unchanged: return `list[(name, detail)]`, **silent on a healthy run**, every threshold
   keyed on a **level**, calibrated from the gap between *distributions* over **≥5 paired seeds
   per fixture** (W5 D5's upgrade — and `pass@k` at large k is exactly the tail-sensitive shape
   that forced it). Rules to implement:
   - `cliff` — a share of prompts pinned at `p ≈ 0` **and** their `p` not moving over the run.
     Needs the conjunction: hard prompts that are *moving* are the point of the exercise.
   - `saturated_band` — the share of prompts above `p_hi` rising **and** the alive fraction
     falling. This is the open-loop decay and it is the rule the controller experiment exists to
     make fire.
   - `floor_violated` — the band's realised `p` below `p_floor(γ, α, τ)`, i.e. you are training
     where your verifier cannot be trusted. This one is a **level**, not a change (W5 D5).
   - `budget_dead` — the fraction of rollouts in dead groups above a level, with the **failure
     side vs success side reported separately**, because D3's identity says `dead_frac` alone
     cannot tell you which, and the two need opposite fixes.
   - **And one candidate rule must be REJECTED during calibration.** The candidate this time is
     `mean_p_rising`, and the *suspected* reason is a new one: **redundancy** — it may fire on
     exactly the same fixture set as `saturated_band`, in which case the panel has two names for
     one measurement. **Measure the fire-sets and compare them; do not assume.** If it is instead
     rejected for firing on healthy runs (rising `p` *is* learning), record that reason instead —
     the point is the measurement, not the prediction. Four days, four cut rules, and the reasons
     so far are: **instrument** (D1) / **semantics** (D2) / **the system does not exhibit it**
     (D3) / **?** (D4).

Provided (not yours): the env/table import, the fixtures, `make_curriculum_task` (a multi-prompt
task whose per-prompt gold sets are chosen to span a target `p`-range), `train_with_curriculum`
(a thin wrapper over D1's `train_grpo` that resamples prompts each step according to a supplied
`sampler`), the two samplers (`static_sort` and `band_tracker`), the perturbation map used for the
contamination twin, and the exact-enumeration helpers.

Stages (ten), asserting at minimum:

1. `stage_signal` — the two closed forms to `< 1e-15` on a `p`-grid × `G ∈ {2,4,8,16,64}`; the
   `(G−1)` factor identified by regressing on `G`; the `p ↔ 1−p` symmetry **exact**; dead groups
   contributing exactly 0 under both modes.
2. `stage_covariance` — `U = 2(G−1)p(1−p)(1−α−γ)` to `< 1e-12` over a `(p, α, γ)` grid; the
   `α+γ=1` null **identically zero in p**; the sign flip; the SNR → 0 linearly at `p → 0` with the
   mean unchanged. **This stage must show that the mean and the variance disagree** — that is the
   whole point of the day's second thesis.
3. `stage_setpoint` — analytic vs grid `p*` at 16 `(a,b,k)` points; monotone decreasing in `k`;
   equal to 1/2 **iff `a = b + k − 1`** (so for the `(1,1)` model, iff `k = 1` — but state the
   general condition, because `(a,b) = (2,1)` reaches 1/2 at `k = 2` and that is the check that
   proves you solved it rather than pattern-matched it); the band's `hi/lo → ~11` and `hi − lo`
   shrinking by **19.07×** from k = 1 to k = 64.
4. `stage_exponents` — **measure `(a, b)` in the actual toy** rather than assume them: run exact
   gradient ascent from many initialisations, regress `log(dp/dstep)` on `log p` and `log(1−p)`,
   report `(a, b)` with a confidence interval, and then report the `p*` that implies. **If the
   fit is poor, say so and report that `p*` is not identified** — a bad fit is a result, and the
   honest output here is more valuable than a number. (D3's bracket lesson.)
5. `stage_group_size` — `G* = ln(0.1)/ln(1−p*)` against `2.3026·k` (the asymptotic, and where it
   breaks at small k); `P(dead | G=8, p*)` = **0.78 / 3.92 / 16.78 / 38.97 / 61.57 / 88.34%** at
   k = 1/2/4/8/16/64.
6. `stage_floor` — the four `(γ, α, τ)` rows; the crossover at **k = 10.0000** solved exactly; the
   value-retention row **0.9793 / 1.0000 / 0.8872 / 0.3804 / 0.0358**; and the `γ = 0.10` case
   whose floor is exactly 0.5 despite `J = 0.80` (accuracy is not the property that matters).
7. `stage_whiten` — `E[ΣA²]_whiten = G·P(alive)` exactly; the tilt limit `G/[4(1−2^(1−G))]`
   against a `p → 0` numerical limit; the whitened set points and the **0.237× at (G=64, k=4)**;
   and the negative control — at `G = 2` whitening and centering must agree on the *argmax*.
8. `stage_controller` — **the day's experiment.** Same task, same budget, two samplers: a
   `static_sort` (rank prompts by step-0 `p`, sample the middle band, never re-rank) and a
   `band_tracker` (re-estimate `p` every `m` steps from the rollouts you already paid for, sample
   the log-odds band). Track: alive fraction, share of prompts in-band, mean useful signal per
   rollout, and the eval metric at k = 1 and k = k_max. **≥5 paired seeds; report the spread; an
   effect inside the spread is a NULL, not a result.** Predict the direction and the mechanism
   *before* running it, in writing, in the file.
9. `stage_contamination` — inject a contaminated fraction `c`, verify the inflation
   `c(1 − pass@k(p_true))` = **3.500 / 2.450 / 1.201 / 0.288 / 0.017 / 0.000** points at
   k = 1/2/4/8/16/64 for `c = 0.05, p_true = 0.30`; verify the training cost is exactly `c` of the
   budget in saturated dead groups; then the twin detector **with its control** — the same
   perturbation on clean items must give `Δ ≈ 0`, and the stage must **fail** if the control moves.
10. `stage_diagnose` — his `diagnose_curriculum`, ≥4 fixtures × ≥5 paired seeds, silent on healthy,
    thresholds with air on **both** sides (W5 D5), the cut rule and its measured reason documented
    in the file, and the fire-set collinearity check between every pair of surviving rules.

Run everything under `-W error::RuntimeWarning`. This file divides by `σ` on groups that can be
dead, by `p` on prompts that can be at zero, by `p_obs` inside the selection law, and takes means
over "the prompts in band" which can be **empty** — four separate members of D2's
six-in-one-session family, and the empty-band mean is the one that will actually bite, because a
controller's whole job is to keep that set non-empty and the failure mode under test is exactly
that it does not.

### Block 7 · 2:45–2:55 — Deliverable & next session

Fill in `deliverable.md`, add every measured number to `shared/numbers-card.md`, update
`progress.md` and `HANDOFF.md`, and **write the interview-bank entries in your own words** (I write
at most one as a worked model).

Day 5 is the **mini RLVR/GRPO loop** — the week's deliverable, and it is now fully specified by
the four days behind it: D1 gives the estimator and the group size, D2 gives the verifier and its
soundness test, D3 gives the evaluation and the selector, D4 gives the data policy and the
controller. **Day 5 writes no new theory.** It composes, and the only question it asks is whether
the four pieces agree with each other when they are in the same process — which, on the evidence
of four consecutive reference runs, they will not, on at least one point.

## Checkpoint Questions

1. A colleague says "we sort the data easy-to-hard and it's working." Give the two questions that
   determine whether that is true, and the measurement that settles it in one run.
2. Derive `p* = 1/(k+1)`. Then name every assumption you used, and say which one you would test
   first on a real stack and why.
3. Your verifier has `α = 0.10, γ = 0.01` and you will report pass@64. State the difficulty you
   should train at, the difficulty the metric wants, which one wins, and what fraction of the
   available signal you lose.
4. You want to report pass@64. What group size does that imply, and what fraction of your groups
   are dead if you use `G = 8` instead?
5. Someone shows that whitening advantages "just rescales the learning rate." Give the
   counter-example with a number, and say what it is doing to the problem distribution.
6. `dead_frac = 0.90` on your run. Give the two possible causes, the one extra number that
   separates them, and the *opposite* fixes each one requires.
7. Your benchmark score jumped 4 points at pass@1 and 0.1 points at pass@64. Give two explanations
   that both predict that pattern, and the experiment that separates them.
8. Useful signal is `2(G−1)p(1−p)(1−α−γ)`. Your verifier improves from `α=0.30` to `α=0.10`. By
   what factor does the useful signal change, and does your optimal curriculum move? Justify the
   second answer in one sentence.

## Frontier-Lab Interview Angle

"Curriculum" is where an interview finds out whether you have run RL or read about it, because the
literature is almost entirely qualitative and the practice is entirely quantitative. The median
answer is "start easy, get harder, filter out the saturated ones." Five moves that mark a real one:

- **Give the set point as a formula, and make it depend on the metric.** *"`p* = 1/(k+1)` under
  the simplest signal model, so it is 0.5 only if you report pass@1 — if you report pass@64 the
  metric wants `p = 0.015`, and the general family is `a/(a+b+k−1)` where `(a,b)` is something I
  would measure, not assume."* Nobody says this, and it is the sentence that reframes the
  curriculum as an *optimisation* rather than a *heuristic*.
- **Turn the set point into a group size.** *"If you report pass@k, keeping 90% of groups alive
  needs `G ≈ 2.3k`. At pass@64 with a standard `G = 8`, 88% of your groups return exactly zero
  gradient."* That is a config-file consequence of a metric choice, which is the register these
  interviews reward.
- **Say that the verifier does not tilt the curriculum, it floors it.** The useful signal
  factorises as `p(1−p) × (1−α−γ)` with **no interaction**, so verifier quality is a multiplier,
  not a preference — but the *precision* constraint puts a hard floor under `p`, and for
  `γ = 1%, τ = 10%` that floor crosses the metric's optimum at **exactly k = 10**. Being able to
  say "the curriculum I want and the curriculum I can label part company at k = 10" is a very
  short path to a real conversation.
- **Name the curriculum you did not choose.** *"Before we design one — the `/std` in GRPO is
  already a curriculum. It makes every alive group carry the same advantage mass, so it weights
  prompts by `P(alive)` and tilts toward the extremes by up to `G/4`, which at `G = 64` is 16×,
  and it moves the optimal explicit set point by up to 4×."* This is the Dr. GRPO point arriving
  from a direction the interviewer is not expecting.
- **Answer the contamination question with a control, not with a method.** Everyone knows
  "perturbed twins." Almost nobody volunteers *"and the control is the same perturbation on
  known-clean items, otherwise you are measuring your perturbation's difficulty"* — and almost
  nobody knows the training-side answer, that a memorised item is a **saturated dead group** that
  silently eats its share of the rollout budget while scoring 1.0 on the reward curve.

Have a number for each. Four days now compose into one quantitative story with no gaps in it —
*the signal dies at both walls, the precision dies at the low wall, the metric puts all its weight
at the low wall, and the curriculum is the controller that has to satisfy all three at once* — and
that story is a far better deep-dive than any description of a training run.

---

## Block 6 addendum — what the code changed about the day

Eight harness defects, and four of them changed a claim in this file.

**1. The `U/mass` ratio is not constant in `p`, and the corrected version is better.**
`mass = E[Σ|A|]` is evaluated under the *verifier's* labels, at `p_obs`, not at `p`.
So

    ratio = U / mass = p(1−p)·J / [p_obs(1−p_obs)]        p_obs = p(1−α) + (1−p)γ

which is the **SNR row already in this file**, not a new object:

    0.8973 / 0.8980 / 0.6454 / 0.3109 / 0.0825   at p = 0.5 / 0.1 / 0.02 / 0.005 / 0.001

The "verifier is a multiplier" thesis survives, stated properly: **`argmax_p U = 1/2`
for every `(α, γ)` with `J > 0`** — verified at `(0,0)`, `(.1,.01)`, `(.3,.05)`,
`(.45,.50)`. A verifier cannot *tilt* a curriculum. And separately, the useful
*fraction* collapses linearly at low `p` with slope `J/[γ(1−γ)] = 89.90`, unless
`γ = 0`, in which case **`ratio → 1` as `p → 0` for any `α`**. Soundness is exactly
the property that stops the collapse. That is the mechanism under the precision floor.

**2. At `(G=64, k=1)` the whitened set point does not exist.**
The whitened objective is `G·P(alive)` times the metric weight; at `k=1` that weight
is 1 and `P(alive)` at `G=64` is 1.0 to machine precision across the middle of the
range. Flat to `1e-15` over `p ∈ [0.4166, 0.5834]` — a plateau 0.167 wide. The
argmax is whatever your grid returns. Report "no preference", not a third
significant figure. This is the whitening thesis in extremis: with a large `G` and a
`k` that asks for nothing, there is nothing left to prefer.

**3. The whitening tilt is a ratio of ratios.**

    whitened/centered at p → 0    =  G²/(G−1)                  9.143 at G=8
    whitened/centered at p = 1/2  =  4G(1−2^(1−G))/(G−1)       4.536 at G=8
    TILT = the quotient           =  G/[4(1−2^(1−G))]  ≈ G/4   2.0157 at G=8, 16.0000 at G=64

The `(G−1)` cancels, which is why the answer is clean.

**4. `(a, b)` measured: `(1.900, 1.911)`, `R² = 0.9979`.**
The prediction `(2,2)` was written into the stage docstring *before* the run — exact
ascent moves `p` by `~lr·‖∇p‖² ∝ p²(1−p)²`, so the advantage-mass model `(1,1)` is
the wrong law for this optimiser. It moves the `k=8` set point from `0.1111` to
`0.1758`, a factor of 1.58. **Measure `(a,b)`; the `+k−1` is arithmetic, the
exponents are not.**

### The cut rule, and why the prediction was wrong

`mean_p_rising` is cut, but **not** for the redundancy I pre-registered. Measured:
the rise in share-above-`p_hi` is **+0.796…+0.938 on the healthy fixture and only
+0.373…+0.410 on the sick one.** It does not merely fire on healthy runs — it fires
*twice as hard* on them, because rising `p` **is** learning. Reason: fires-on-healthy.

Two consequences, and the second is the useful one:

- the `rise` conjunct inside `saturated_band` was dead weight and was removed;
  the alive-fraction **fall** carries all the discrimination
  (healthy `−0.319…+0.244`, sick `+0.594…+0.667`)
- the redundancy the prediction expected *was* real, but in the wrong place: the
  collinearity check found `budget_dead` and `saturated_band` with identical
  fire-sets — which turned out to be a **fixture-set** defect, not a rule defect.
  Three fixtures cannot separate four rules. The bench went to six, and
  `cliff` and `floor_violated` fired for the first time.

Four days, four cut rules, four different reasons: instrument (D1) / semantics (D2) /
the system does not exhibit it (D3) / **it fires on healthy, harder than on sick (D4)**.

### Two calibration laws the panel taught

- **A threshold is calibrated for a regime.** At 150 steps the alive-fall gap is 0.35
  wide; at 60 steps it is 0.014 wide and *no* threshold separates. The panel runs at
  its calibrated regime even under `--fast`. You may re-derive thresholds for a
  faster one; you may not borrow them.
- **A change statistic must not overlap the estimator's burn-in.** `p_hat` is an EMA
  from a prior, so a head window at step 0 measures the *sensor* converging: prompts
  whose true `p` never left `1e-4` read as having moved `0.047…0.276`, and `cliff`
  could not fire on the fixture built for it. Start the head window after burn-in.

And when two fixtures landed inside a threshold's uncertainty band (`coin_flip`
3/5, `bad_verifier` 2/5), they were **reported as borderline rather than tuned
away**. Moving a threshold until a fixture behaves is fitting the panel to the bench.

### The defect that would have invalidated everything

`_tilt_to_target` bisected a **non-monotone** map. The scalar logit bonus is applied
to every `(state, action)` pair on a correct sequence's path, and those pairs are
*shared* with incorrect sequences through the same prefix — so for ~30% of
correct-sets a bonus of `−40` drives `p` to **1.0**. Bisection assumes a monotone
bracket, silently returned an endpoint when it did not have one, and the fixture's
advertised "exact known difficulty" was wrong on **40% of prompts, with errors up to
0.97** — while every stage downstream quoted those difficulties to four decimals.
The twin-gap control read `+0.097` instead of `0`; it now reads `+0.000000`.

**Verify the bracket. If you cannot, do not use a root-finder that requires one.**
