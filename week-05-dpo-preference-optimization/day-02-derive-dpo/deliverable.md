# Deliverable — Week 05 Day 02: Deriving DPO, and Then Past Where the Paper Stops

**Artifact:** `week-05-dpo-preference-optimization/code/dpo.py` — eight stages, **42 checks**,
24 s full / 12 s `--fast`, clean under `-W error::RuntimeWarning`. Imports D1's `fit_bt`,
`tilt`, `sigmoid`, `logsigmoid` rather than re-deriving them.

The derivation was not the day's content — it was produced cold in the first block and graded
**3.55/4**. The day's content is the four things *after* the cancellation, and the file turns
each of them from an assertion into a number.

## The thesis

> **DPO is a reward model whose parameterization happens to be a policy.**

The gradient weight is `σ(−M)`, the *same function of the same object* as the Bradley-Terry
gradient weight fitted on W4 D1 and again yesterday. So everything true of reward-model fitting
is true of DPO training — including Day 1's existence theorem. The reparameterization does not
change the statistics; it only changes what the parameters are called.

And the corollary that is the best available answer to "what is DPO's real weakness":

> **The KL in the derivation is not the KL that gets enforced.** PPO's KL is reverse KL under
> π_θ, re-estimated every iteration on samples the policy actually generates. DPO's constraint
> survives only as π_ref inside a ratio evaluated **at the dataset's completions**. Off the
> dataset, nothing whatsoever constrains π_θ.

---

## The derivation, verbatim

Start from the KL-regularized optimum (W3 D5) and invert it:

```
pi*(y|x) = pi_ref(y|x) e^{r(x,y)/beta} / Z(x)
  =>  r(x,y) = beta log( pi*(y|x) / pi_ref(y|x) ) + beta log Z(x)
```

Substitute into Bradley-Terry, which reads only reward *differences* at a fixed prompt:

```
P(y_w > y_l | x) = sigma( r(x,y_w) - r(x,y_l) )
                 = sigma( beta log(pi_th(y_w|x)/pi_ref(y_w|x)) + beta log Z(x)
                        - beta log(pi_th(y_l|x)/pi_ref(y_l|x)) - beta log Z(x) )
```

**Z(x) cancels because it is a function of the prompt alone** — the completion has been summed
out of it — and both completions in a pair share the prompt. Stated as the *reason*, not merely
performed. Hence

```
L_DPO(theta) = - E_(x,y_w,y_l) [ log sigma( beta log(pi_th(y_w|x)/pi_ref(y_w|x))
                                          - beta log(pi_th(y_l|x)/pi_ref(y_l|x)) ) ]
```

**The assumption audit, said while writing line 1, not afterwards:** this requires π_θ to *be*
the exact optimizer for that r — which is a statement about the **policy class**, not about the
reward.

### The gradient

```
grad_theta L_DPO = - beta E[ sigma(-M) ( grad log pi_th(y_w|x) - grad log pi_th(y_l|x) ) ]
        with     M = beta[ log(pi_th(y_w)/pi_ref(y_w)) - log(pi_th(y_l)/pi_ref(y_l)) ]
```

Read three ways:

1. **A weighted contrastive log-likelihood.** Strip the weight and it is MLE on the chosen plus
   *unlikelihood* on the rejected. No rollouts, no critic — a supervised objective with a sign
   flip on half the data. That is the entire practical appeal.
2. **The weight is how wrong the implicit RM currently is** — `σ(−M)`, the BT weight. Measured:
   the best-ranked decile carries weight **0.049334**, the worst **0.951741**, a **19.3×** ratio.
   So correctly-ranked pairs contribute almost nothing and the fit is driven by near-ties and
   mislabels. **A mislabelled pair has large negative M ⇒ weight → 1: label noise receives the
   maximum gradient weight in the batch.** (That is what cDPO/label smoothing on D3 undoes, and
   it is D1's ε-cap put back by hand.)
3. **Nothing constrains the marginal likelihood.** Only the *difference* is penalized, which is
   why both logprobs can fall.

**And π_ref is absent from the direction.** It is not differentiable in θ, so it survives only
inside the scalar weight. Measured: two different reference models giving genuinely different
weights (σ(−M) = **0.448705** vs **0.673414**) produce gradients that agree exactly once divided
by their weights — **2.776e-17**.

> At the gradient level, **π_ref is a per-pair learning-rate multiplier and nothing else.**

---

## Headline 1 — The equivalence, verified rather than asserted

Two algorithms sharing no code path:

- `train_dpo` — full-batch GD on `L_DPO`, tabular softmax logits over a 7-state tree.
- `bt_then_tilt` — fit a tabular BT reward over **sequences** with D1's `fit_bt`, then tilt
  π_ref by it exactly. That is "fit an RM, then run KL-regularized RL to convergence" with
  every source of optimization noise deleted.

On a fixture whose comparison graph **is** strongly connected (105 comparisons over 15
sequences, labelled at ε=0.15, so the BT MLE exists and is finite):

| quantity | value |
|---|---|
| `max_y \|pi_DPO(y) - pi_BT-tilt(y)\|` | **7.772e-16** |
| wrong-β tilt at β = 0.25 / 1.0 / 2.0 | **6.15e-02 / 8.25e-02 / 1.63e-01** |
| non-vacuity factor | **2.10e+14** |

Machine precision, over the whole 15-sequence distribution, elementwise. The wrong-β control is
what makes it evidence: the check *can* fail, by fourteen orders of magnitude.

> **"DPO is equivalent to RLHF" is now a number rather than a citation.**

### And the three conditions it rests on — which are literally the three steps of the proof

**Proof (tabular, one prompt, enumerable completions).** Write `r̂ = β log(π/π_ref)`. Then
`L_DPO(π) = L_BT(r̂)` term by term — DPO's loss, as a function of the implicit reward, **is** the
BT NLL. The map `π ↦ r̂` is injective with image

```
S = { r : sum_y pi_ref(y) exp(r(y)/beta) = 1 }        (codimension 1)
```

`L_BT` is constant along the missing direction (`L_BT(r+c) = L_BT(r)`), and every `r ∈ R^|Y|`
has exactly one representative in S — namely `r − β log Z(r)`, unique because `c ↦ e^{c/β}Z` is
strictly monotone. So **S is a complete set of coset representatives and `L_BT` is constant on
cosets**, hence `min_π L_DPO = min_{r∈S} L_BT = min_r L_BT`, attained at the tilt of the BT MLE. ∎

| # | condition | where it enters | reality |
|---|---|---|---|
| 1 | **π_θ can represent the tilt exactly** | the image of `π ↦ r̂` is *all* of S | a transformer reaches a sliver of S |
| 2 | **the BT MLE exists** (Ford, D1) | `min L_BT` is attained at a finite point | a DPO dataset is a **matching** ⇒ it is not |
| 3 | **π_ref is the distribution the pairs came from** | the tilt is *around* π_ref; μ picks the pairs | pairs come from a stronger model |

Plus the silent fourth on the other side: the PPO path must be **run to convergence**. Nobody
does that either.

**Which bites hardest: condition 1** — and not merely as "approximation error". Once the class
cannot hit the tilt, DPO's optimum is **not even the projection** of the tilt in any divergence,
because `L_BT` is a finite sum over *dataset pairs* weighted by `σ(−M)`. So:

> DPO's approximation error is allocated by the **data distribution**; PPO's by the **policy's
> own occupancy**. "DPO and RLHF-with-PPO are equivalent" is exactly as true as "your policy
> class contains the tilt", and no truer.

---

## Headline 2 — The gauge is fixed by the simplex, and that is the divergence mechanism

Asked mid-session whether `Σ_y π_ref e^{r̂/β} = 1` is not *identically* true. It is — and that
is the point.

```
r_hat = beta log( pi_th / pi_ref ) = beta log( e^{r/beta}/Z(x) ) = r - beta log Z(x)
```

Measured: recovering `r − β log Z` from a policy built out of a known `r` agrees to **4.441e-16**,
with offset `β log Z = +1.425614`, and it is emphatically **not** r. And
`Σ_y π_ref(y) e^{r̂(y)/β} = 1.000000000000000`.

> The BT equivalence class `{r + c(x)}` has infinitely many members and a policy is the same for
> all of them. But **inverting a policy returns one specific member — the one whose normalizer
> is 1.** The parameterization chooses the gauge for you.

**Consequence, and it is the whole failure surface.** In reward space `r → r + c` is free. In
policy space it is not, because probability is conserved: `r̂` is confined to a codimension-1
surface, so **you cannot lower `r̂(y_l)` without raising `r̂` somewhere else.**

Independent confirmation from the same run: the implicit reward equals the BT MLE **after
centering** to **1.332e-15**, and **not** before — the raw gap is **0.7454**, exactly the
`−β log Z` offset. Both halves asserted.

---

## Headline 3 — The divergence, and one place the plan was wrong

D1: the BT MLE is at infinity on a **matching**, and a DPO dataset *is* a matching. Under the
reparameterization, `r̂(y_l) → −∞` reads literally as `π_θ(y_l|x) → 0`.

On the 7-pair perfect matching, plain GD, no early stop:

| step | 2 (accuracy first hits 1.0) | 20 (10×) | 20 000 |
|---|---|---|---|
| exact KL(π_θ‖π_ref) | **0.1129** | **0.8486** | **1.4660** |
| mean log π(y_l) | — | — | **−25.9537** |
| loss | — | — | **4.834e-05** |
| accuracy | **1.0** | 1.0 | 1.0 |

`log π(y_l)` decreases **monotonically at every one of 20 000 steps**; the losers' total
probability goes **0.343396 → 1.649e-09**; the KL rises monotonically throughout. The training
metrics a default script prints — loss, margin, accuracy — are perfect from step 2 onward.

**The plan said "the KL grows without bound". That is false in a finite space and the harness
asserts the true version.** `KL(π_θ‖π_ref)` is capped by the **point-mass bound**
`max_y −log π_ref(y) = 4.6894`: collapsing to a point mass is the worst π_θ can do. Measured
1.4660 and still climbing. In a real LM that bound is `−log π_ref` of the single likeliest
string over `|V|^T` completions, so "unbounded" is the *practical* statement, not the
mathematical one — and saying it that way is strictly better in an interview.

---

## Headline 4 — Both logprobs fall, at the converged optimum

Not on the divergent fixture. On the **strongly connected** one, where the BT MLE exists and DPO
lands on `tilt(r̂_BT, β)` to 7.772e-16:

| | start | end | Δ |
|---|---|---|---|
| mean implicit margin | +0.0000 | **+0.6155** | **up** |
| mean log π(y_w) | −3.1249 | **−4.0003** | **−0.8754** |
| mean log π(y_l) | −3.3174 | **−5.4238** | −2.1064 |

> **Both logprobs falling is not a divergence artifact. It is what the converged optimum looks
> like**, because no term in the gradient mentions the marginal likelihood. Every sequence is a
> winner in some pairs and a loser in others, mass concentrates on the top few, and the mean
> chosen logprob falls while every margin improves.

---

## Headline 5 — The reference-mismatch tax is paid in KL, and the loss never reports it

Two fixtures, **the same eight sequences, columns swapped — one line of the fixture changes**:

| | mean log π_ref(y_w) | final margin | final loss | **final KL** |
|---|---|---|---|---|
| winners in π_ref's **bulk** (μ ≈ π_ref) | −2.2076 | +9.1364 | 1.250e-04 | **0.6640** |
| winners in π_ref's **tail** (μ ≠ π_ref) | −4.0865 | +8.3982 | 2.288e-04 | **3.3932** |

**5.11× the KL for the same margin at comparable loss.** Nothing in the DPO objective reports
it. The pairs are identical; only which side is labelled the winner changed.

Formally, with candidates drawn from μ the population objective is
`L_μ = −E_{(y_a,y_b)~μ⊗μ}[log σ(M)]` while the derivation wanted `π_ref ⊗ π_ref`; the missing
weight is `π_ref(y_a)π_ref(y_b) / (μ(y_a)μ(y_b))` and **there is no ratio anywhere in the DPO
loss that could carry it** — contrast W4 D2, where ρ is right there in the objective. DPO is
off-policy with the correction mechanism structurally deleted.

**A second place the plan was wrong.** It claimed the chosen logprob *rises* when μ = π_ref. It
does not: `Δlog π(y_w)` is **−1.8978** (bulk) and **−0.0954** (tail), and on `tail` the sign is a
*transient* — positive until roughly step 5000. **Do not build a diagnostic on that sign.** What
is stable in both is that the margin is bought almost entirely by pushing y_l down
(|Δw|/|Δl| = **0.0941** and **0.0056**), and the KL gap, which is present from the first step.

**Practical corollary, now with a mechanism instead of folklore.** At θ = π_ref the log-ratio is
**exactly 0 everywhere** (asserted) and every pair starts at weight σ(0) = 0.5 — so it is *not*
that the ratio starts large. It is that **the ratio the labels require is large**, and DPO's
first phase is spent moving mass onto the data manifold rather than learning preferences. Hence:
**SFT on the chosen completions first**, so π_ref ≈ μ, the required implicit reward is small, and
the KL budget you bought with β goes to preferences.

---

## Headline 6 — A result that is not in the paper: shared prefixes cancel exactly

`∇log p(y) = Σ_t [e_{y_t} − π(·|s_t)]`. For a pair sharing a prefix *with the same actions on
it*, those per-state terms are identical and cancel in the difference. Measured at the root
state of two prefix-sharing sequences: **0.000e+00**.

> In a tabular policy, the shared prefix of a preference pair contributes **exactly nothing** to
> the DPO gradient. So "pushing y_l down drags y_w down with it" cannot happen through shared
> **tokens** — that failure mode requires shared **parameters**.

Which is the honest scope statement for the whole file: this environment reproduces the
*statistical* pathologies exactly (existence, gauge, uncontrolled marginals, the μ tax) and
deletes the *function-approximation* ones by construction. Naming which is which is the point.

---

## Interview answers (final)

### 1. "Derive DPO." → and the follow-up: "where did the KL go, and what enforces it?"

Derivation as above, in five beats: name the two ingredients and the destination first; invert
the tilt including `β log Z`; **state** why Z cancels; the assumption audit *while* writing line
1; then offer the follow-up yourself.

The follow-up, in three clauses:

1. **Where it went** — consumed in the algebra. It survives only as π_ref inside the ratio, and
   in the gradient only inside the scalar weight `σ(−M)`.
2. **What enforces it during training** — off the dataset, **nothing**.
3. **Under what measure** — PPO's KL is reverse KL **under π_θ**, on samples the policy
   generates, re-estimated every iteration. DPO's implied constraint is evaluated **under the
   data distribution**, at fixed completions, forever. *The KL in the derivation is not the KL
   that gets enforced.*

### 2. "Is DPO equivalent to RLHF?"

"Yes, under three conditions, and here is which one your setup violates." The three from
Headline 1, plus the silent fourth. Then the number: **7.772e-16 elementwise on a fixture that
satisfies all three, non-vacuous by 2.10e+14 against a wrong-β control** — and the honest
qualifier that condition 1 is what a transformer breaks, after which DPO's optimum is not even
the projection of the tilt.

### 3. "Your preference pairs came from a model stronger than your SFT checkpoint. What does DPO spend its first phase doing?"

Moving mass onto the data manifold, not learning your preferences — because the log-ratio starts
at exactly 0 and the ratio the labels *require* is large. It is paid for in KL, invisibly:
**5.11× the KL for the same margin**, same pairs, comparable loss. Fix: SFT on the chosen
completions first, so π_ref ≈ μ.

---

## Bugs hit, and what each generalises to

1. **Missing `−` on the NLL — second sighting in two days.** `L = logsigmoid(M).mean()` is a
   log-*likelihood*. The signature was `analytic == −FD` to **1.161e-10**: a random disagreement
   means the gradient is wrong, a sign-exact disagreement means the loss and gradient are
   inconsistent by a constant and **exactly one of them** is wrong. Resolved by the
   **asymmetric probe**: at θ = θ_ref every margin is 0, so the NLL is `+log 2 = +0.693147`; the
   buggy version read **−0.693147**. (W4 D4's rule again: `σ(0) = 0.5` under *both* signs, so the
   symmetric point cannot detect a flip. Yesterday's version of this bug was undetectable
   because separable data sends ‖r‖ → ∞ under either sign.)
2. **`pi_final = softmax_np(theta)` — states ≠ sequences, second sighting.** Shape (7,3) summing
   to **7.0** instead of (15,) summing to 1.0. A distribution over *actions at states* is not a
   distribution over *sequences*; only `Σ_t` crosses that boundary. The trap that makes it
   survive: **the gradient of a sequence quantity has token shape** — `grad_sequence_logprob`
   correctly returns (7,3) — so shape alone tells you the level of the *parameters*, not of the
   *value*. Same family as W3 D5's per-state KL 1.053 vs sequence KL 5.056.
   **The one-line test: does it sum to 1, or to n_states?**
3. **`history = []` then `history[t].append(...)`** → `IndexError`. It is a dict keyed by step and
   you assign, not append. **Latent**: no stage passes `record`, so the suite would have gone
   42/42 with the bug in it. Untested branch, same reason W3 D5 passed 94/94 on a wrong number.
4. **Returning `pi` where the contract said `(pi_star, r_bt)`** — the only thing standing between
   35/35 and 42/42. Read the return annotation as a checklist, the same way the parameter list
   is one (W4 D4).
5. **Re-implementing a provided helper.** `implicit_reward` was written as a hand-rolled loop
   that `all_sequence_logprobs` already is. Standing rule: **if you finish a file without calling
   a provided helper, that is a bug report about your own code.**
6. *(harness-side, mine)* `‖g_pair‖ ∝ σ(−M)` is **false** — it is
   `β·σ(−M)·‖score_w − score_l‖`, and the score norm varies **1.81×** across pairs. Divide out
   *both* factors and every pair returns exactly β (**3.331e-16**). A proportionality check must
   divide by everything that varies.

---

## Session record

| block | score |
|---|---|
| Block 0 — cold recall (6 items) | **3.95 / 6** (66%) — up from D1's 65% |
| Block 1 — the derivation, cold, graded on 4 axes | **3.55 / 4** (89%) |
| — the gradient, derived cold | **1.0** |
| — (a) the `σ(−M)` weight | 0.6 — mechanism right, the BT identity not named |
| — (b) what the gradient does not control | 0.35 — **known-result reflex**, gave shift invariance |
| Block 4 — "where did the KL go" | 0.6 — location right, **measure** missing |
| Block 4 — "is DPO equivalent to RLHF" | 0.25 — could not enumerate the three conditions |
| coding | **42 / 42** |

**Retired today:** the Z(x) cancellation stated *as a reason* — the Block 1 gap (0.7) was closed
within the session on the re-derivation, written on the page unprompted.

**New finding — a structure gap, distinct from the second-clause gap.** The three conditions
came out **unprompted as a margin annotation while deriving** in Block 1, and could not be
produced **on demand as a list** forty minutes later. The material is there; the retrieval
structure is not. Fix: the three conditions *are* the three steps of the Part 1 proof, so
recover them by replaying the proof and asking "what did I use here?"

**Still open:** the second-clause habit (fifth week — the derivation + (a) + (b) request came
back as the derivation only); the known-result reflex (third sighting — *"is that the
obstruction, or just an obstruction you already know?"*); and the same three REINFORCE variance
sources on the fifth ask.

## Reusable artifact

- **Path:** `week-05-dpo-preference-optimization/code/dpo.py`
- **Reuse:** Day 3 adds `log σ` stability (the divergence run drives M past the region where
  `np.log(sigmoid(·))` returns exactly 0), the length prior, and label smoothing — which is D1's
  ε-cap and today's `σ(−M) → 1` on mislabels, put back by hand. Day 4 runs IPO/cDPO/KTO against
  the **matching** fixture, which is the arm that breaks DPO. Day 5 reuses the divergence panel
  (`logp_w`, `logp_l`, exact KL) as the failure-mode plot, and the `bulk`/`tail` pair as the
  reference-mismatch demo.
