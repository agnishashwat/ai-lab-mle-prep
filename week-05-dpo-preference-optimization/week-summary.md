# Week 05 Summary — Preference Optimization: DPO and Variants

**Days 1–5 complete. 385 property-based checks across the week**, every one asserted against
exact enumeration, a closed form, or an independent second algorithm:
`preference_models.py` **74**, `dpo.py` **42**, `dpo_trainer.py` **51**,
`variants.py` **126**, `dpo_lab.py` **92**.

## The 5-Minute Version

Week 4 built RLHF as an **optimisation** problem: PPO, a critic, a KL controller, and four
failure modes of the optimiser. Week 5 asks the prior question — **what can preference data
identify at all?** — and the answer reframes everything downstream. Every DPO failure mode
this week turns out to be a consequence of the *data format*, not of the algorithm.

**Day 1** treats the reward model as an **identification** problem: infinite data, a tabular
`r` with one free parameter per completion, a perfectly specified model — and asks what is
*still* unknown. Three degrees of freedom, not one. **Location** (`r + c(x)`) is a *flat*
direction: the likelihood is constant along it, infinitely many finite maximisers, fixed by
gauge-fixing. **Scale** is identified — but *by the annotator disagreement rate*, not by
quality. And **connectivity**: by Ford's theorem the BT MLE is finite **iff the wins digraph
is strongly connected**, and separability is an *ascent* direction — the likelihood strictly
increases forever, so there are **zero** maximisers and centring cannot help. The corollary
that should stop you: **the MLE is at infinity with infinite data.** More data adds edges but
never a cycle.

**Day 2** derives DPO and then goes past where the paper stops. The thesis:
**DPO is a reward model whose parameterization happens to be a policy.** The gradient weight
is `σ(−M)` — the *same function of the same object* as the BT gradient weight — so Day 1's
existence theorem applies verbatim. Two results the paper does not state: **both logprobs
falling is not a divergence artifact, it is what the converged optimum looks like**; and
**the KL in the derivation is not the KL that gets enforced** — PPO's KL is re-estimated on
samples the policy actually generates, while DPO's survives only as `π_ref` inside a ratio
evaluated *at the dataset's completions*. Off the dataset, nothing constrains `π_θ`.

**Day 3** makes the implementation's five contracts explicit — sequence-level reduction, a
bitwise-exact frozen reference cache, a stable cDPO loss with the correct finite optimum,
one SFT checkpoint cloned into *both* the policy and reference roles, and diagnostics that
record levels rather than only pair accuracy. That last contract is what makes step 0 of
every run in this week's harness read `loss = log 2 = 0.693147` and `KL = 0` **exactly**.

**Day 4** collapses the variant zoo into **one 5×5 grid**: `(implicit reward f) × (link ℓ)`.
Four named papers are four cells; the grid reproduces five hand-written implementations
**elementwise, max deviation 0.000e+00**. The axes act on different things — **the link
decides whether the margin is finite, the reward decides where the length goes** — and an
unoccupied cell (`ref_mean × squared_target`, "length-normalised IPO") behaves exactly as the
grid predicts. Four received claims are measured and falsified, the sharpest being that
**IPO does not cap the margin — it compresses it**, squeezing a 99/1 preference and a 60/40
preference from a ratio of 11.33 into 4.90.

**Day 5** assembles all of it into a failure-mode lab where every mode is a **fixture**: one
dataset function, four knobs, and each knob produces exactly one mode. The week's designed
prediction is confirmed — **a β-sweep and an ε-sweep matched on `Δ̂* = logit(1−ε)/β` land on
the same KL to 2.8e-16**, because the policy is identified only by `r/β`. An unplanted result
falls out: **steps-to-target scales as 1/β², measured 4.04× / 4.03× / 4.02×** — halving β
costs *four* times the compute, not two. And the sharpest single measurement of the week:
a DPO run whose gold reward goes **+0.7514 → −0.4532**, worse than the reference it started
from, at **100% train accuracy with the loss decreasing monotonically the whole way**.

## The one-sentence version

> **DPO's failure modes are consequences of what the data format can identify. The loss is
> exactly constant along every direction the data do not constrain — so the loss going down
> is not evidence of anything.**

## Key Concepts (crisp)

- **Ford's condition.** BT MLE finite ⟺ wins digraph strongly connected. BT is logistic
  regression on difference features `e_w − e_l`, so this is exactly linear separability.
- **Flat vs ascent directions.** Location invariance is *flat* (infinitely many finite
  maximisers, gauge-fixable). Separability is *ascent* (zero maximisers). Different problems
  with different fixes; conflating them is the classic error.
- **Scale is a measurement of annotator disagreement, not of quality.** `r → c·r` changes
  `σ(Δ)`, so scale *is* identified — by the flip rate. Cleaner annotators ⇒ larger β.
- **β and reward scale are one knob.** `tilt(c·r, c·β) = tilt(r, β)` to ~1e-16. The policy is
  identified only by `r/β`. **Cleaning your data and tripling your learning rate in
  reward-space are the same edit.**
- **The implicit reward.** `r̂(y) = β·[log π_θ(y) − log π_ref(y)]`, exactly 0 everywhere when
  `π_θ = π_ref`. The reward does not start large; the *ratio the labels demand* is what is large.
- **The cap.** Under ε-noisy annotators the largest justifiable margin is `logit(1−ε)`, and
  `+inf` when ε = 0. A statement about the **annotators**, so it needs **no held-out set**.
- **The 5×5 variant grid.** `L = mean_k ℓ(f(y_w), f(y_l))`. Link ⇒ finiteness. Reward ⇒ length.
- **Over-optimisation vs optimisation pathology** (carried from W4 D4, now measured in DPO):
  the objective is wrong vs the optimiser is broken. Identical on a loss curve, opposite fixes.

## Key Equations

```text
BT           P(y_w > y_l) = sigma( r(y_w) - r(y_l) )
KL-opt       pi*(y) = pi_ref(y) e^{r(y)/beta} / Z          (W3 D5)
invert       r(y)   = beta log( pi*(y)/pi_ref(y) ) + beta log Z
DPO          L = -mean_k log sigma( beta[ log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l) ] )
gradient wt  sigma(-M)                                     the SAME weight as BT
implicit r   r_hat(y) = beta [ log pi(y) - log pi_ref(y) ]
margin       M = mean_k [ r_hat(y_w) - r_hat(y_l) ]        LINK units
Dhat         M / beta                                      POLICY units -- the invariant one
cap          logit(1 - eps) = log((1-eps)/eps)             +inf when eps = 0
cDPO opt     M* = logit(1 - eps)
IPO opt      M* = (2f - 1) tau                             finite where DPO is not
identity     tilt(c*r, c*beta) = tilt(r, beta)
KL bound     KL(pi || pi_ref) <= max_y -log pi_ref(y)       BOUNDED in a finite space
```

## What I Implemented

| Day | Artifact | Checks | What it establishes |
|---|---|---:|---|
| 1 | `preference_models.py` | 74 | Ford's condition, the three DOF, β as annotator quality |
| 2 | `dpo.py` | 42 | DPO ≡ tilt(BT MLE); margin `~ log t`; both logprobs fall at the optimum |
| 3 | `dpo_trainer.py` | 51 | the five contracts, on a real `TinyGPT` |
| 4 | `variants.py` | 126 | the 5×5 grid; four papers are four cells; an empty cell occupied |
| 5 | `dpo_lab.py` | 92 | four modes as fixtures; the β/ε collapse; the panel; early stopping |
| | **total** | **385** | |

## Failure Modes I Now Understand

| # | Mode | Mechanism | Instrument | Fix |
|---|---|---|---|---|
| M1 | divergence / rejected collapse | BT MLE has no finite solution; `margin ~ log t` | `margin` vs `cap`, `logp_l` | early stop on margin/cap; or a bounded **link** (IPO τ, cDPO ε) |
| M2 | both logprobs fall | the gradient constrains only the **difference**; the level is free | `logp_w` and `logp_l` as **levels** | a term outside the pairwise difference (SFT/NLL, i.e. ORPO/RPO) |
| M3 | length exploitation | annotator ranks by `gold + κ·\|y\|`; length is the cheapest satisfying direction | `elen` **and** `gold` together | change the **reward** axis (mean reduction, γ margin) — stopping earlier does not help |
| M4 | off-distribution drift | the loss is **exactly constant** off the comparison set | `offdata_mass` (a **level**) | on-policy sampling scored by something; no training-panel field sees it |

**Two of these are invisible to KL**, and KL is *bounded* (6.3216 here) so it saturates while
the reward diverges. KL is **necessary, not sufficient**.

## The numbers worth remembering

```text
P(strongly connected)    matching 0.00 at EVERY eps  |  all_pairs 0.00 -> 0.58 -> 0.83 -> 1.00
more comparisons         2 pairs 6.05  >  7 pairs 4.19  >  105 pairs 1.67   (6/6 seeds)
DPO == tilt(BT MLE)      max |dev| 1.164e-13 over all 15 sequences
margin ~ log t           +0.7163 then +0.7078 per doubling of the budget
cDPO stops at            2.1972 = ln 9 = logit(1 - 0.10)
IPO stops at             2.0000 = (2f - 1) tau        IPO compression 11.33 -> 4.90 (2.31x)
beta/eps collapse        |KL diff| 9.6e-05 / 6.6e-05 / 2.8e-16 / 2.5e-04
steps ~ 1/beta^2         76 / 307 / 1237 / 4978        ratios 4.04, 4.03, 4.02
acc saturates            1.000 at step 1 of 6000, while the margin runs 0.0245 -> 5.8439
both logprobs fall       y_w -0.1852 AND y_l -0.4826 while the margin rose +0.5947
SFT term fixes it        logp_w -0.1852 (DPO) -> +0.0213 (DPO+SFT)
M4 blind spot            19.4% of the model's mass where the loss is exactly constant
gold inversion           +0.7514 -> -0.4532 at 100% train accuracy
early stopping           selected 0.4963 vs last 0.3300 vs oracle 0.6737
KL bound                 6.3216 (point mass, TabEnv(3,3), ref_scale 0.7)
step 0 anchors           loss = log 2 = 0.693147 exactly;  KL = 0 exactly;  r_hat = 0 exactly
```

## Corrections this week made to earlier weeks

- **W4 D4's "seed spread grows with optimisation pressure" is a 3-SEED ARTIFACT.** Same runs:
  gold spread amplifies **6.09× at 3 seeds** and **1.04× at 5 seeds**. Two extra seeds flip
  the conclusion entirely. On those same runs `margin` spread *contracts* to exactly 0.000000
  and `kl` spread grows **71×**. **≥3 seeds is a FLOOR, and it is not enough for a
  tail-sensitive metric.**
- **W4 D4's `rel()` had two calibration bugs**, ported and fixed on D5. It divided by
  `max(|x[0]|, 1e-9)` — and at θ = θ_ref the implicit reward is *exactly* 0.0, so `margin[0]`
  is exactly 0.0 in **every** DPO run and it returned `1e9 · x[-1]`. The lesson generalises:
  **the fix for a zero denominator is the RIGHT denominator, not an epsilon on the wrong one.**
- **W4 D4's −20% entropy threshold sat inside the range a healthy run occupies.** D5's
  diagnostic is therefore asserted **silent on a healthy run**, and its M3 gate has 2.8× air
  below and 4.4× above.
- **W4 D5 answer 3's DPO paragraph** should now cite this week's numbers rather than
  a qualitative claim about DPO's KL.

## Open Questions / To Revisit

- **Everything here is tabular and enumerable.** `TabEnv(vocab=3, t_max=3)` has 15 sequences,
  so `exact_kl`, `offdata_mass` and gold are computable. At scale, `offdata_mass` has **no
  estimator** — which is precisely why M4 is the mode with no instrument. What is the
  practical proxy? (On-policy samples scored by a held-out RM is the obvious candidate; it
  inherits that RM's own identification problems.)
- **Does the 1/β² law survive a non-tabular parameterization?** The derivation (gradient
  scale ∝ β, distance ∝ 1/β) does not obviously depend on tabularity, but it does assume plain
  GD — an adaptive optimiser should change the exponent. Worth one measurement.
- **`select_checkpoint` needs ε.** In production the annotator flip rate is not handed to you.
  Estimating it from inter-annotator agreement is standard, but the cap is *sensitive* to it
  (ε = 0.10 ⇒ 2.197, ε = 0.25 ⇒ 1.099). How much does a mis-estimated ε cost?
- **KTO's prospect link was implemented but never stress-tested** as its own failure surface —
  it is the one cell of the grid with an asymmetric value function and no D5 fixture.
- **Online / iterative DPO** is the obvious answer to M4 (regenerate the comparison set from
  the current policy) and was out of scope this week. It is the bridge back to W4's PPO.
