# W4 D5 — The Toy PPO-RLHF Loop, and Adaptive β as Dual Gradient Ascent

**Code:** `week-04-rlhf-ppo/code/ppo_rlhf.py` — **90/90 checks**, ten stages, all under
`-W error::RuntimeWarning`, full suite in **26 s**. Fixture env `vocab=4, t_max=5`
(S=121, 364 enumerable sequences). Imports all four earlier files; almost every stage is
integration and ablation rather than new math.

**Session score:** Block 0 cold recall **5.0/8** (up from 4.45), Block 1 quiz **1.4/5**
(two blanks), concept checkpoint **1.6/3**. Drills delivered as model answers.

**Design decision preserved from Days 1–4:** every *measurement* is an exact expectation by
enumeration (`exact_panel` reads the policy, never the batch), even when the *updates* are
sampled. Bias claims are exact; variance claims are sampled. The two are never mixed.

---

## Headline 1 — adaptive β IS dual gradient ascent, and the fixed point is the envelope theorem

The derivation, once:

$$\max_\theta \mathbb{E}[r]\ \text{s.t.}\ \mathrm{KL}\le\delta
\quad\Longrightarrow\quad
\mathcal L(\theta,\beta)=\mathbb{E}[r]-\beta(\mathrm{KL}-\delta)$$

For **fixed β** this is exactly W3 D5's objective up to the constant `βδ` ⇒ **penalty form and
constraint form are primal and dual views of one problem, and β was always a dual variable.**

`g(β) = max_θ L(θ,β)` is a **max of affine functions of β** ⇒ convex, *with no assumptions on the
inner problem*. You are running a 1-D **convex** optimization on top of a hopeless non-convex
one. By the envelope theorem

$$\frac{\partial g}{\partial\beta} = \delta - \mathrm{KL}(\pi^*_\beta\|\pi_{\text{ref}})
\quad\Longrightarrow\quad \beta \leftarrow \beta + \eta(\mathrm{KL}-\delta)$$

and setting `η = K_β·β/δ` gives InstructGPT's shipped form
`β ← β(1 + K_β(KL/δ − 1))`. **Multiplicative because it is a step in log β** (scale-free; β
spans orders of magnitude) **and because it keeps β > 0**, which is not numerical hygiene —
β is a multiplier on an *inequality* constraint, so β < 0 is **infeasible**.

### The check the day exists for

The controller's fixed point vs `dE[r]/dKL` measured as a **secant slope of two enumerated
tilted distributions** — a quantity whose computation contains no β anywhere.

| δ | β_∞ (iterative, sampled) | KL(β_∞) | dE[r]/dKL (closed-form, exact) | rel err |
|---|---|---|---|---|
| 0.25 | 1.23079 | 0.25000 | 1.17216 | **5.00%** |
| 0.50 | 0.83637 | 0.50000 | 0.82765 | **1.05%** |
| 1.00 | 0.56914 | 1.00000 | 0.55652 | **2.27%** |
| 2.00 | 0.37505 | 2.00000 | 0.35735 | **4.95%** |

> **Dual ascent's fixed point and the envelope identity are the same number, reached by two
> routes that share no code.** One is a feedback loop that knows nothing about exponential
> tilts; the other is a finite difference over enumerated distributions.

And the frontier is **concave** — **100%** of consecutive secant slopes non-increasing.
**β is a price, and the price falls as you buy more nats.**

### The caveat, measured rather than asserted

Dual ascent assumes the inner problem is solved to **optimality**. PPO takes K clipped epochs.
Running the same controller with a **PPO** primal instead of the exact tilt:

| primal | β_∞ | at KL |
|---|---|---|
| **exact** (`p* ∝ p_ref e^{r/β}`) | **0.6173** | 0.8646 |
| **PPO** (4 epochs × 4 minibatches, clipped) | **0.1563** | 0.8646 |

**Gap 74.7%.** This is a *justification* for the controller's form, not a proof of convergence —
and now the overreach has a number attached.

### K_β is a gain, and it has two opposite failure modes

| K_β | settling | max KL/δ | std(log β), last 30 |
|---|---|---|---|
| **0.01** | **never** | 1.9716 | 4.34e-02 |
| 0.10 | 46 | 1.9716 | 1.01e-06 |
| **0.50** | **8** | 1.9716 | 1.11e-16 |
| **2.00** | **never** | **6.5981** | **6.00e+00** |

Small gain ⇒ sustained overshoot; large gain ⇒ **ringing over ~400× in β**. And the
controller's *input* is a sampled KL: **k1 is negative on 20–44% of samples** (W3 D5), so at
high gain β is chasing a quantity whose **sign flips**. ⇒ **Fix the sensor (k3) before detuning
the gain** — lowering K_β low-passes the noise and permanently costs response speed, while the
estimator stays broken.

---

## Headline 2 — the chained regression: seven days, one code path

Flipping switches on `rlhf_iteration` must recover each earlier day **elementwise**, and does:

| switch | recovers | max abs diff |
|---|---|---|
| β=0, ε=∞, 1 epoch, λ=1, no critic | **W3 D2 REINFORCE** (Ψ = G_t) | **7.749e-15** |
| + centering | **W3 D3 baseline** — gradient *unchanged* | **7.749e-15** |
| + learned critic | W3 D3 with Ψ = G_t − V(s_t) | 7.749e-15 |
| λ ∈ {0, 0.5, 0.95} | **W3 D4 GAE** | exact |
| ε = 0.2 at ρ=1 | **W4 D2 PPO** — clipping cannot bite | 7.749e-15 |
| β = 0.7 | **W4 D3 shaping** — β enters only through `r` | 7.749e-15 |

The baseline check is non-vacuous by a **1.3e11×** margin: advantages moved **9.897e-04** while
the gradient moved **7.749e-15**. That is EGLP as an equality on an enumerated batch, not an
approximation.

**Interview line:** *"REINFORCE, +baseline, +critic, GAE, PPO and KL shaping are one code path
in my repo, and each rung is asserted against the previous one elementwise to 1e-15."*

---

## Headline 3 — the ablation grid: the week in one inversion

7 arms × **3 seeds**, identical rollout budget (`n_rollout × n_iters`). π_ref's gold = **4.4191**.

| arm | gold | proxy | KL | H | len | selected gold |
|---|---|---|---|---|---|---|
| **full** (clip + KL + critic + adaptive β) | **5.1501** ±0.109 | 0.6153 | 0.361 | 0.777 | 4.928 | 5.1689 |
| fixed β | 5.0148 ±0.024 | 0.4778 | 0.181 | 0.833 | 4.922 | 5.0200 |
| **no KL** | **4.9098** ±0.407 | **0.8728** | 1.722 | 0.510 | 4.936 | 5.2055 |
| no clip | 5.1508 ±0.088 | 0.6134 | 0.360 | 0.778 | 4.928 | 5.1618 |
| no critic | **5.1786** ±0.036 | 0.6341 | 0.382 | 0.769 | 4.930 | 5.1848 |
| REINFORCE (matched budget) | 4.7112 ±0.045 | 0.1297 | 0.034 | 0.901 | 4.914 | 4.7112 |
| top-p 0.9 | 4.8559 ±0.202 | 0.3868 | 0.144 | 0.859 | 4.918 | 4.9505 |

> **The no-KL arm beats the full loop on the PROXY (0.8728 vs 0.6153) and loses on GOLD
> (4.9098 vs 5.1501).** Optimising the objective harder made the model worse, and the dashboard
> reported success the whole way. **That single inversion is the week.**

Two arms behaved differently from the plan, and both are honest findings:

- **`no_clip` ≈ `full`** (5.1508 vs 5.1501). At a tight KL budget the clip is **not the binding
  constraint — the penalty is.** This is D2's "InstructGPT ships clip AND an explicit penalty"
  read operationally: with the penalty doing the work, removing the clip costs nothing here.
- **`no_critic` slightly BEAT `full`** (5.1786 vs 5.1501) with **lower** seed spread
  (±0.036 vs ±0.109). **The critic's value is a function of horizon**, and this horizon is 5.

  > **CORRECTION (W6 D1).** "BEAT" is not supported and neither is "lower seed spread." The
  > effect is **0.0285** against `full`'s own spread of **±0.109** — **3.8× the effect** — and
  > `no_critic`'s interval `[5.1426, 5.2146]` is *contained* in `full`'s `[5.0411, 5.2591]`.
  > That fails this repo's own rule (*claim a result only if the intervals do not overlap*).
  > The spread comparison is worse: it is a **3-seed** spread comparison, and W5 D5 showed the
  > same runs give a **6.09×** spread ratio at 3 seeds and **1.04×** at 5. **The supported
  > claim is a NULL: deleting the critic cost nothing measurable at T=5 on a terminal-only
  > reward.** That is the stronger form anyway — a null is exactly what justifies a deletion.
  With a terminal-only reward and T=5 there is almost nothing for bootstrapping to buy, and a
  batch-mean baseline captures most of it. *Do not generalize "PPO needs a critic" from a
  short-horizon task.*

---

## Headline 4 — checkpoint selection: early stopping is the ESTIMATOR

Gold along the KL ladder on the no-KL run (41 checkpoints):

```
4.419 4.616 4.781 4.923 5.019 5.089 5.139 5.168 5.189 5.188 5.177 5.170 5.157 5.130
5.098 5.060 5.033 4.993 4.975 4.949 4.912 4.869 4.838 4.796 4.729 4.678 4.641 4.600
4.550 4.497 4.457 4.385 4.347 4.290 4.238 4.183 4.145 4.118 4.069 4.003 3.937
```

| | iter | gold | KL |
|---|---|---|---|
| **selected** | **8** | **5.1894** | 0.503 |
| final | 40 | **3.9372** | 3.605 |
| π_ref | 0 | 4.4191 | 0.000 |

**Shipping the final policy costs 1.2522 gold and lands BELOW the SFT model you started from**
— and it does so while the proxy climbs monotonically. This is the case D4 said the learned-RM
fixture *didn't* produce; **starving the RM to 30 comparisons produces it.**

> **In RLHF, early stopping is not regularization — it is the estimator for the optimum.**
> In supervised learning the training objective's optimum is a place you decline to reach out of
> caution. Here the proxy's optimum is *in the wrong place*, gold's optimum is at a finite
> interior KL, and running to convergence **overshoots it**. The stopping point is a parameter
> estimate and the KL ladder is its grid.

Corollaries: **you do not grid-search β on final reward** — you sweep, checkpoint, and select;
the gold eval must be **independent of the RM** or you have re-created the problem inside your
evaluator; and **≥3 seeds is mandatory** (the no-KL arm's spread is **±0.407**, four times the
full loop's effect over fixed-β).

---

## Headline 5 — `diagnose` inside a real loop, and what it caught unprompted

D4's artifact, unmodified, run on three real runs:

| run | KL | entropy | gold | verdicts |
|---|---|---|---|---|
| **healthy** (δ=0.25 adaptive) | 0.362 | 0.944 → 0.777 | 4.419 → **5.157** | **(silent)** |
| no KL | 3.142 | 0.944 → 0.279 | 4.419 → 4.173 | reward_hacking, entropy_collapse |
| top-p 0.8 | 1.248 | 0.944 → 0.687 | 4.419 → 4.154 | reward_hacking, entropy_collapse, **value_divergence** |

**The false-positive check survived the transition from synthetic histories to real runs** —
the first thing D4's handoff said to verify.

**Unplanted result: `value_divergence` fired on the top-p arm, EV 0.000 → −5.19e9.**
Nobody injected a bad critic. The **deadly triad** assembled itself: function approximation +
bootstrapping + **off-policy-ness supplied by the sampler**. This is D4's *"a distorted sampler
does not only bias the ACTOR; it is also the leg that lets the CRITIC diverge"* reproduced
inside a full loop, without being asked for.

**Unplanted result 2 — the top-p ratchet, as a crash.** Reachable sequences under an
enumerated top-p rollout, iteration by iteration:

```
40 → 40 → 39 → 18 → 18 → 17 → 18 → 12 → 30 → 9 → 12 → 9 → ...
```

(and at the reference policy: top-p 1.0 → **364** sequences, 0.9 → **176**, 0.8 → **40**.)

The excluded tail takes the `−p_k` share of every update and never the `+1[k=a]` share, so
support shrinks and **cannot recover**. Eventually `B < n_minibatches` and `np.array_split`
returns an empty split whose weights sum to 0 → `0/0`. **The one-way ratchet stopped being a
sentence and became a stack trace.**

### Two calibration findings about `diagnose` (both real, both worth fixing)

1. **`rel()` divides by the first value, and the first value is exactly 0 by construction.**
   The RM is standardised under π_ref and the policy *starts* at π_ref, so `proxy[0] = 0` and
   the reason string reads `proxy 0.000 -> 1.114 (+111362994095%)`. The *verdict* is right; the
   *reason* is garbage — which violates D4's own design constraint that **the reason string is
   the deliverable**. Fix: report an absolute delta when the baseline is near zero.
2. **The −20% entropy threshold sits inside the range a healthy run occupies.** At `kl_target
   0.4` a run whose gold rose **monotonically** to 5.334 dropped entropy **−26%** and fired
   `entropy_collapse`. Thresholds calibrated on synthetic histories do not automatically
   transfer to real ones.

---

## The loop, and the split that is silent

**Once per batch (frozen at θ_old):** old logprobs (the ratio *denominator*), reference
logprobs, shaped rewards, `V_old`, advantages, **value targets**, whitening statistics.
**Per minibatch step (live θ):** new logprobs → ρ, `L^CLIP`, new values → `L^VF`, entropy,
gradient, step.

Two orderings that are not stylistic:
- **`value_targets` BEFORE whitening.** The critic regresses onto *returns*, in reward units.
- **Shaping BEFORE advantages.** The KL term is **reward shaping, not a loss term** — no
  estimator downstream contains a β. Consequence: `π_ref` can be **dropped from memory after
  the shaping step**; `θ_old`'s logprobs cannot, because they are live for the whole inner loop.
  *Anything upstream of the reward is droppable; anything inside the estimator is not.*

**The bug, and why it is silent.** Recompute `logp_old` inside the epoch loop ⇒ `ρ ≡ 1` ⇒
clipping never fires ⇒ `clip_frac` reads a *healthy* **0.000** ⇒ you have replaced PPO with K
steps of vanilla PG on stale data with **no trust region**. And the loss cannot tell you:
`L_IS(θ_old) = E[A] ≈ 0` **by construction**.

> **The probe is `std(log ρ)`.** Exactly 0 under the bug; **0.098999** measured on a real
> 4-epoch update. `clip_frac` is **symmetric** across the two hypotheses ("bug" and "policy
> barely moved") and therefore cannot discriminate; `std(log ρ)` is **asymmetric**.
> Same principle as D4's `sigmoid(0) = 0.5` and W2 D5's toothless cache test.

---

## The instrument panel, in PAIRS

| pair | detects | evidence from this week |
|---|---|---|
| proxy **+ gold** | reward hacking | proxy 0.8728 up / gold 4.9098 down |
| KL **+ β** | drift vs. controller health | KL parked at δ with β settled = working |
| entropy **+ unique-completion count** | entropy collapse | fired at **KL 0.30** — invisible to KL |
| length **+ length-controlled win rate** | length exploitation | fired at **KL 0.55** |
| clip fraction **+ KL-to-old** | step size | `clip 0.0466 ≤ oob 0.0639` |
| **EV + value loss** | critic health | EV → **−5.19e9** on the top-p arm |

> **A single metric is a hypothesis. A pair is a test.**
> A reward-only dashboard detects **none** of the four failures. Reward + KL detects **two**.

---

## Interview answers (final, verbatim)

### 1. "Design an RLHF system from scratch."

Four stages: pretrain → **SFT** → **RM** → **PPO with a KL anchor** → evals. Each fixes a
different thing and cannot fix the others: SFT teaches format and task, the RM encodes ranking,
RL optimises for the ranking, evals catch what the RM got wrong. If the model lacks a fact, none
of the later three will teach it.

**Four models, and the memory bill drives everything downstream:** policy and critic trainable
at ~**16 B/param** (params + grads + 2 Adam moments), reference and RM frozen at ~2 B/param,
plus the rollout **KV cache**. ~**112 GB** for a 7B trainable side before activations. That bill
is why DPO and GRPO exist — DPO deletes three of the four, GRPO deletes the critic.

**Loop:** prompts → generate at **T=1.0 untruncated** → RM scores one scalar at EOS → cache
reference logprobs → shape the per-token reward → values → GAE → **targets, then whiten** →
**freeze θ_old and the advantages** → K epochs × M minibatches of `L^CLIP + c₁L^VF − c₂H` →
update β → log → periodically gold-eval and checkpoint.

Four orderings are load-bearing: **T=1.0 untruncated** is the only actual fix for
off-policy-by-accident (measured bias vs exact ∇J: on-policy **2.95e-10**, temp 0.7 **0.2284**,
top-p 0.9 **0.1169**; IS repairs temperature to **2.953e-10** and *fails* for top-p —
**0.1169 → 0.1267, it got worse** — because IS needs `supp(π) ⊆ supp(q)` and **7.2%** of the
mass is unobservable); **π_ref is droppable after shaping, θ_old is not**; **targets before
whitening**; and **the freeze line is the algorithm** — recomputing `logp_old` inside the loop
is silent, and the probe is `std(log ρ)`, not `clip_frac`.

**Panel in pairs** (as above). **A reward-only dashboard detects none of the four failures.**
Entropy collapse fired at KL **0.30**, length at **0.55** — both invisible to KL, because a
sharpened policy lives inside π_ref's support and length changes one decision many times.

**Checkpointing is part of the algorithm.** Gold peaked at **5.1894 (iter 8, KL 0.503)** and
fell to **3.9372 at KL 3.605** — below the SFT model's **4.4191**. Shipping the last checkpoint
costs **1.2522**. Checkpoint along the KL ladder, select on gold: **early stopping is the
estimator for the optimum, not regularization.**

**β:** fixed states a *price* (`dE[r]/dKL = β`), adaptive states a *budget*. Ship adaptive when
runs must be comparable across tasks or RMs; fixed when deliberately mapping the frontier. And
say what adaptive β **cannot** do: with a length confound κ = **0.6276** against a resistance
`β·KL̄_t` **bounded at 0.1411/token**, no β stops it — **fix length in the data.**

**Three non-negotiables:** ≥3 seeds (spread amplifies **89× / 259× / 667×**; at 30 comparisons
the spread **±1.60** exceeds the entire mean improvement **+0.95**); a gold eval **independent
of the RM**; RM ensembling with a **minimum, not a mean** — `E[max] > max E`, so averaging
shrinks the error without changing its sign and the optimizer harvests it anyway. The minimum
turns disagreement into a penalty. That is an LCB.

### 2. "Why does PPO look the way it does?"

**Chain 1 — how you get a gradient, then how you afford it.**
You can't backprop for **two** independent reasons: the reward isn't differentiable **and** the
sampling distribution depends on θ. ⇒ **score function** ⇒ high variance ⇒ **reward-to-go**
(delete provably zero-mean terms), **baselines** (state-only ⇒ unbiased by EGLP), **GAE**
(bias/variance dial). Every algorithm from REINFORCE to PPO differs **only in Ψ_t** — verified
as one code path, each rung elementwise to **7.7e-15**. ⇒ rollouts dominate cost ⇒ **reuse** ⇒
**off-policy** ⇒ importance ratios ⇒ but `∏_t ρ_t` is log-normal, mean exactly 1 with a
collapsing median, `σ(T) = σ₁√T`, and at **T=500 a 1024-sequence batch has ESS 0.38** ⇒
**per-token ratios**, paid for by no longer being an unbiased estimator of the sequence
objective ⇒ ratios must stay near 1 ⇒ **TRPO** ⇒ too expensive ⇒ **clip**.

**Be precise about clipping: it deletes the gradient, it does not enforce a bound.** Three
levels where it isn't a trust region — **token** (`T·ε²/2` ≈ 10 nats at T=500), **sample**
(`clip_frac 0.237` vs `oob_frac 0.570`; only where the advantage points outward), **iteration**
(**π_old resets** ⇒ it bounds the step, not the walk: **4.68 nats at a 5.3% clip fraction** vs
**0.75** with a penalty, **6.2×**).

**Chain 2 — why there is a second mechanism.** The reward is a proxy and the policy is an
optimizer pointed at it, so it climbs the **argmax of the error** (RM error **0.177 → 4.89**,
27.6×, along a KL ladder with the RM unchanged) ⇒ a trust region **relative to the RM's data**,
and since `π_ref = π_SFT` *is* the RM's sampling distribution, KL literally measures distance
from the RM's training set ⇒ **β is a property of the RM's data; retune it when you improve the
RM.** The penalty must be **dense** because the RM reward is one scalar at EOS ⇒ put it in the
**reward**, per token — and that is exact, because `ρ₀` and the dynamics cancel in the ratio ⇒
its price should be a **budget** ⇒ **adaptive β = dual ascent**, sign forced by the envelope
theorem, multiplicative form forced by scale-freedom and feasibility.

**Verified:** the controller's fixed point matches `dE[r]/dKL` to **5.00% / 1.05% / 2.27% /
4.95%** at four budgets. **Honest caveat, unprompted:** dual ascent assumes the inner problem is
solved to optimality and PPO takes K clipped epochs — measured cost **β_∞ 0.1563 vs 0.6173
exact at the same KL, a 74.7% gap.** It is a justification, not a proof.

### 3. "When would you *not* use PPO?"

**PPO's bill:** four models; a critic that must be trained and **can diverge** (deadly triad —
I saw EV → **−5.19e9** with the *only* change being a top-p sampler); a large hyperparameter
surface (ε, λ, c₁, c₂, K, M, β or δ and K_β, whitening, value clip); and an **online rollout
loop** — generation dominates wall clock and forces the reuse that creates the off-policy
problem PPO then spends its machinery correcting.

**DPO:** no rollouts, no RM, no critic — one model plus a frozen reference. It is the attempt to
get back to a **pathwise** estimator: closed-form over fixed pairs, low variance, no sampling
loop. **Paid for with static off-policy data** — you never see your current policy's mistakes.
Use when data is fixed and plentiful and compute is tight.

> **UPDATED W5 D5, with measurements.** "Static off-policy data" understates it, and the
> precise version is a better answer. The DPO loss is a function *only* of the sequences that
> appear in a comparison and is **exactly constant** everywhere else — so the price is not
> staleness, it is that **an arbitrary fraction of the model is unconstrained by the
> objective** (measured: **19.4%** of the trained policy's mass, on data covering 6 of 15
> sequences), and **no field of the training panel reports it**. Three further corrections:
> (a) DPO's KL does **not** diverge — it is *bounded* by `max_y −log π_ref(y)` (6.3216 in that
> environment); what diverges is the **margin**, at rate `log t`, because the Bradley–Terry
> MLE is finite only if the wins digraph is strongly connected and a DPO dataset is a
> **matching**, where it never is at any noise level. (b) DPO's KL is not enforced the way
> PPO's is: it survives only as `π_ref` inside a ratio evaluated *at the dataset's
> completions*. (c) β is not a free hyperparameter — the policy is identified only by `r/β`,
> and **cleaning your annotations and scaling your reward are the same edit**. Halving β also
> costs **4× the compute, not 2×** (measured 4.04× / 4.03× / 4.02×).

**GRPO:** no critic — the baseline is the group mean over K samples of the same prompt, which is
unbiased *and* removes exactly the per-prompt constant `c(x)` that Bradley-Terry cannot identify
anyway. **Paid for with K× rollouts and sequence-level credit.** Use with a verifier (RLVR),
where a per-token critic has almost nothing to learn. Watch the sum-vs-mean length prior — the
thing Dr. GRPO fixes.

**The one thing PPO gives that neither does: per-token advantages from an online, on-policy
signal.** So PPO wins when the reward is **dense** or the horizon is **long** — multi-turn, tool
use, agentic trajectories, where *which step* went wrong is the question.

**How I'd choose:** DPO if I have preference data and no budget for rollouts; GRPO once I have a
verifier; PPO when I need token-level credit on long trajectories and can afford four models.
And I'd note that in my own grid the critic bought **nothing** on a short-horizon terminal-reward
task — **5.1786 without it vs 5.1501 with it, and lower seed spread.** The critic's value is a
function of horizon; if the horizon is short you are paying for a model that isn't earning.

---

## Bugs hit, and what each generalizes to

| bug | lesson |
|---|---|
| `np.add.at(...) / N` → `TypeError: NoneType / int` | **Operations that mutate return `None`.** If a call's whole job is to modify its first argument, it has nothing to hand back. Third family sighting after `list.sort()` and `+=` rebinding |
| clipped `KL/δ` instead of `KL/δ − 1`, **twice** (once with the wrong bounds, once with the right bounds on the wrong quantity) | The clipped quantity must be **0 at the setpoint**. Trace the fixed point through the expression before running anything |
| `clip(...)` — the *parameter* `clip` invoked as a function | **Naming a parameter after an operation makes the operation unsayable in that scope.** Three name collisions in one session: `clip`, `lr` (next to `cfg.lr`), `iter` (a builtin) |
| `batch.logp_ref[batch.mask]` | **`x[bool_mask]` flattens to 1-D.** Cost a **2.16×** KL error on a 3-row batch and a clean factor of **B** on any sampled batch — β would pin at the clip ceiling forever and look like a working budget. **A defensive guard that alters shape is worse than no guard**; use `np.where` when you want the mask |
| `adv / batch.lengths` — `(B,T) / (B,)` | Broadcasting aligns **trailing** axes. Explicit `[:, None]` when a per-sequence vector meets a per-token array. **Latent**: no stage sets `reduce="mean"`, so 90/90 passes with it broken — same class as the `T == V` cross-entropy bug |
| `k_beta * clip = 2.0 × 0.5 = 1.0` ⇒ β = 0 | **Zero is a multiplicative absorbing state.** The floor is not about numerical positivity — it keeps the *controller alive*, because a dead multiplier means the constraint has silently ceased to exist |
| the positivity check **passed** while the bug was live | **A test is only as sensitive as its input.** The check ran at `k_beta·clip = 0.06`, nowhere near the boundary. When you write a guard, ask *"what input drives this to its boundary?"* and test **that**. Third sighting (D4's `sigmoid(0)`, W2 D5's cache test) |
| empty minibatch under the top-p ratchet | A distribution's **support is not static** under optimization. `array_split` on `B < M` returns a zero-row array whose weights sum to 0 |

---

## What this toy has, and what it does not — say this unprompted

**Has:** real token-level logprobs, real EOS and variable length, a **learned** BT reward model
from 30 noisy comparisons over 23 features, a **learned** critic, and **exact ground truth by
enumeration**.

**Lacks, and each costs something specific:** no **shared trunk** (so `c₁` coupling is simulated,
not a real contest between two heads over shared parameters); no **sampling cost** (so
epoch-reuse economics are *stated*, not *felt* — in a real run generation dominates wall clock,
which is the entire reason to reuse a batch); no **distributed anything** (where most real RLHF
engineering time actually goes); and **no real gold eval** — ours is a known function, which is
a **luxury no production run has**.

> **"We had ground truth, so unbiasedness was an assertion rather than a plot. That is the
> strength of the setup and also its limit — every result is exact, and every result is at
> V=4, T=5."**

Volunteering the limitation is what makes the strength credible.

---

## Checkpoint questions — answers

1. **Loop order + the split.** Above. The silent mistake is recomputing `logp_old` (or the
   advantages) inside the epoch loop: `ρ ≡ 1`, `clip_frac = 0.000`, no trust region, and
   `L_IS(θ_old) = E[A] ≈ 0` means the loss curve cannot report it. Probe with `std(log ρ)`.
2. **Adaptive β as dual ascent.** Lagrangian → `g(β) = max_θ L` is a max of affines ⇒ convex →
   envelope theorem gives `∂g/∂β = δ − KL` ⇒ `β ← β + η(KL − δ)` ⇒ multiplicative with
   `η = K_β β/δ`. Multiplicative because it is scale-free (a step in log β) and because β > 0 is
   a feasibility requirement. **Dual ascent assumes the inner max is solved exactly; PPO takes K
   clipped epochs — measured gap 74.7%.**
3. **Predicting β at budget δ.** `β_∞ ≈ dE[r]/dKL` at `KL = δ` on the closed-form frontier —
   agreement is meaningful because the two are structurally independent (iterative + sampled vs
   closed-form + exact). Disagreement is diagnostic: β_∞ too low at the right KL ⇒ the primal is
   not reaching its argmax; β_∞ right but KL ≠ δ ⇒ controller steady-state error or a saturated
   clip; sign flipped ⇒ `δ − KL` instead of `KL − δ`, which looks healthy and diverges.
4. **Panel in pairs.** Above. Reward-only is blind to all four because each single number is
   ambiguous — reward up is good *or* Goodhart, KL up is exploration *or* drift, entropy down is
   confidence *or* collapse, length up is helpfulness *or* exploitation. The second reading
   disambiguates.
5. **Early stopping as the estimator.** In supervised learning the training optimum is a place
   you decline to reach out of caution about noise. In RLHF the *proxy's* optimum is in the wrong
   place while gold's optimum sits at a finite interior KL, so running to convergence
   **overshoots** it. Stopping is not a safety margin — it is a parameter estimate, and the KL
   ladder is its grid. Measured: peak **5.1894 at KL 0.503**, final **3.9372 at KL 3.605**,
   below π_ref's **4.4191**.
