# Deliverable - Week 04 Day 02: The PPO Objective (Ratio, Clipping, Epochs)

## What I Built / Derived

`week-04-rlhf-ppo/code/ppo_loss.py` — **112 staged checks, 5.3 s** (`--fast` 109 in 1.5 s).
Seven TODOs, nine independent stages, importing Day 1's `enumerate_sequences` / `exact_J` /
`exact_sequence_kl` as ground truth. Nothing in here is established by a plot.

Built: `token_logratios` (per-token `log rho`, log space), `is_surrogate`,
`clipped_surrogate` (with `clip_frac` / `oob_frac` / `frac_high` / `frac_low`),
`surrogate_gradient` (analytic, with the clip GATE), `ppo_update` (the epoch x minibatch
reuse loop), `sequence_ratio_stats` (the exact distribution of `prod_t rho_t` by
enumeration), and `drift_experiment` (N iterations under clip / penalty / both / none).

**The design decision that carries the day: `exact_batch`.** It returns every terminating
sequence with `weights = p_theta_old(y)` instead of `1/B`, so an estimator evaluated on it is
not an estimate but the exact expectation. That is what turns "the surrogate gradient at
`theta_old` is the policy gradient" into an assertion at **1e-11** rather than a claim about
standard errors. Same move as Week 3's finite-MDP ground truth, one level up.

### Results

1. **The rho=1 identity, as a regression.** `grad(is_surrogate)` at `theta == theta_old`
   equals Week 3's score-function gradient **elementwise at exactly 0.0**, and both match a
   central finite difference of the enumerated `J` at **1.9e-11**. Gradient rows sum to zero
   (2.1e-16, softmax over-parameterisation) and centring the advantages moves the gradient by
   **9.7e-17** — a constant baseline is free, exactly, by EGLP.

2. **The 2x2 clip table, hand-computed, with gradients.** Gradient is **exactly `0.0`** in the
   two clipped cells and nonzero in the other two; `eps -> inf` reproduces `is_surrogate`
   exactly; `L_CLIP` is monotone nondecreasing in `eps`; `L_CLIP <= L_IS` pointwise over a
   randomised sweep. **Only one cell is unbounded** (`A<0, rho>1+eps`): its value is exactly
   linear in rho (`-4.0 -> -12.0` at `rho` 2 -> 6) while the other three are flat in rho.

3. **THE HEADLINE — clipping bounds the step, never the walk.** 25 iterations, beta=1.0:

   | mode | KL(pi_theta \|\| pi_ref) | J | clip_frac mean | clip_frac max |
   |---|---|---|---|---|
   | clip only | 0.015 -> **4.6836** | 0.64 -> 4.36 | **0.0532** | 0.1653 |
   | penalty only | 0.076 -> **0.7497** | 0.83 -> 1.95 | 0 | 0 |
   | both | 0.015 -> **0.7488** | 0.64 -> 4.35 -> 1.95 | 0.0060 | 0.0553 |
   | neither | 0.076 -> **5.6133** | 0.83 -> 4.67 | 0 | 0 |

   **6.2x more drift under clip-only, at a 5.3% clip fraction.** The penalty run plateaus (last
   8 iterations span **0.034 nats**) while clip-only rises another **1.33 nats** over the same
   window. Mean per-iteration step KL is 0.017 against a final walk of 4.68 — **270x** — which
   is the arithmetic of "small steps, unbounded walk" in one ratio.

4. **The beta frontier, and `dE[r]/dKL = +beta` measured.** Four penalty-only runs to
   convergence: beta 0.7/1.0/1.5/2.5 -> KL 1.890/0.849/0.342/0.115, J 2.910/2.085/1.475/1.046.
   Every secant slope **brackets its beta pair** (0.793 in [0.7,1.0]; 1.203 in [1.0,1.5];
   1.892 in [1.5,2.5]) and the slopes **fall monotonically as KL grows** (1.892 -> 1.203 ->
   0.793) — the concavity of the frontier, measured.

5. **Epoch reuse degrades the update direction, and clipping makes that WORSE.** Angle to the
   true `grad J`, exact batch, lr=0.2:

   | K | angle (clip) | angle (no clip) | KL travelled (clip) | KL (no clip) | J (clip) | J (no clip) |
   |---|---|---|---|---|---|---|
   | 1 | **0.000** | **0.000** | 0.00055 | 0.00055 | 0.224 | 0.224 |
   | 8 | 15.8 | 3.2 | 0.0122 | 0.0521 | 0.349 | 0.503 |
   | 64 | **43.8** | **11.8** | **0.0676** | **1.9784** | 0.576 | 2.457 |

   Clipping costs **32 degrees** of direction and buys **29.3x** less KL travelled. Those
   degrees ARE the deliberate bias, measured. And the no-clip run scores higher J on the same
   batch (2.46 vs 0.58) — the run that left the trust region looks better on every instrument
   visible from inside it.

6. **The sequence-ratio blow-up, exactly, plus the scaling law.** `E_old[rho] = 1` to
   **1e-15** at every T (it is a likelihood ratio) while the median sinks and p99/median grows
   1.06 -> 1.22 -> 1.38 -> 1.89 -> **2.50** over T = 1..12, and ESS/n falls 0.9991 -> **0.8387**.
   `sigma(T) = sigma_1 sqrt(T)` holds to **1.1%** and `ESS/n = exp(-sigma^2)` to **1.6%**.
   Extrapolating the verified law with `sigma_1 = 0.1257`, measured while **88.5% of individual
   tokens sit inside the |rho-1| <= 0.2 band**: at T=500, ESS/n = **3.7e-4**, i.e. a batch of
   1024 sequences has an effective size of **0.38**.

## Core Takeaway

PPO answers exactly one question: **how many gradient steps can I take on data I already
collected?** Rollouts cost 10-100x a gradient step, so vanilla policy gradient — one step per
batch — spends almost all its compute on data it uses once. Reuse the batch, and the instant
you take step 2 you are off-policy, which forces an importance ratio. The ratio is unbiased
for any theta but its variance scales with the chi-square divergence, so it is worthless once
the policies separate; and a *sequence*-level ratio is a product of T per-token ratios, hence
log-normal, hence has mean exactly 1 while its median collapses. So: ratio per token, and clip
it. Clipping is not a constraint — it **deletes the gradient** outside the band instead of
enforcing a bound, which is why nothing stops rho from being 5 and why the guarantee is only
"no reward for going further out." At `theta = theta_old` the surrogate's gradient IS Week 3's
policy gradient, so PPO is REINFORCE-with-advantages plus permission to keep going; everything
past the first step is a biased, deliberately conservative approximation.

## Key Equations

    L_IS(theta)    = E_old[ rho_t(theta) A_t ],        rho_t = pi_theta(a_t|s_t)/pi_old(a_t|s_t)
    grad rho_t     = rho_t grad log pi_theta(a_t|s_t)
    grad L_IS |_{theta_old} = E[ A_t grad log pi_theta ]        <- Week 3's PG, exactly
    L_CLIP         = E[ min( rho A , clip(rho, 1-eps, 1+eps) A ) ]   <= L_IS pointwise
    Var(rho f)     ~ 1 + chi^2(pi_theta || pi_old)
    ESS/n          = 1/E[rho^2] = 1/(1 + chi^2),   E_old[rho] = 1 always
    sigma_seq(T)   = sigma_1 sqrt(T)   =>   ESS/n = exp(-sigma_1^2 T)
    per-token KL implied by |rho-1|<=eps:  k1 ~ eps,  k2 = (log rho)^2/2 ~ eps^2/2
      eps = 0.2  =>  0.2 nats by k1, 0.02 nats by k2

**The 2x2 table (get this cold).**

| | rho > 1+eps | rho < 1-eps |
|---|---|---|
| **A > 0** | value `(1+eps)A`; **grad 0**; stop raising a good token we've raised enough | value `rho A`; grad flows; pull a good token back up |
| **A < 0** | value `rho A`; grad flows; push it down — **UNBOUNDED**, dual-clip's target | value `(1-eps)A`; **grad 0**; stop suppressing what's suppressed enough |

Clipping is one-sided *per cell*: the two cells where the gradient flows are cells where rho is
*already* outside the band.

**Why clipping is not a trust region — three gaps, one per level of aggregation.**
(i) **Per token**: T tokens at the boundary give sequence KL `T eps^2/2`, unbounded in T — 10
nats at T=500, eps=0.2. (ii) **Per sample, only where the clip binds**: a bound holding on the
subset where the gate is closed says nothing about `E[KL]`. (iii) **Relative to `pi_old`, which
resets every iteration** — N bounded steps in a consistent direction is an unbounded walk.
Measured: 4.68 nats at 5.3% clip fraction.

**TRPO vs InstructGPT KL direction.** TRPO constrains `KL(pi_old || pi_theta)` — *forward*,
mass-covering, "don't abandon what the old policy did." InstructGPT penalises
`KL(pi_theta || pi_ref)` — *reverse*, mode-seeking, support-confining, "don't go where the
reference wouldn't." Different objects, different failure modes.

## Implementation Notes

- **The frozen `Batch` is the design.** `logp_old` and `advantages` are fields computed once
  under `theta_old`; the reuse loop has access to nothing else, so "freeze theta_old and the
  advantages" cannot be got wrong. In a real stack `logp_old` is the only thing kept from the
  rollout worker, which is also why `pi_ref` can be dropped after rollout (query it once on the
  sampled tokens — 14 GB of a frozen 7B you never keep resident).
- **The KL penalty is reward shaping, not a loss term.** `r(y) - beta[log p_old(y) - log
  p_ref(y)]`, applied before the advantages are formed, so no estimator in the file knows the
  penalty exists. Consistent with the Psi ladder: the beta term modifies `r`, so it changes
  every rung at once. It is stale after the first gradient step, exactly like the ratio.
- **Two reductions, and they differ**: weighted sum over sequences (`w_b` is a probability),
  plain sum over t (the PG theorem has `sum_t`). Swapping them only rescales the gradient, so
  it masquerades as a learning-rate problem forever.
- **`L_IS(theta_old)` is not zero even with centred advantages** — it equals
  `sum_b w_b len(y_b) A_b`, so it vanishes only if LENGTH is uncorrelated with advantage.
  Measured leak +0.010993; exactly 0.0 in a one-step bandit. That residual is RLHF length bias
  showing up two days before Day 4 names it, and it is a second reason never to debug a PPO run
  from its loss curve.
- **`eps=0.0` is not `eps=None`.** At eps=0 the band collapses to `{1}` and the gate still
  closes on two of four cells; `None` closes nothing.
- **`clip_frac` != `oob_frac`.** Every out-of-band token is a candidate; only the ones whose
  sign of A points further out lose their gradient, so `clip_frac <= oob_frac` always (measured
  0.237 vs 0.570). Reading them as the same number means you have clipped both sides.
- Shape trap that earned its keep: V=5, T=3, B=85 all distinct. Two broadcasting bugs
  (`weights` needing `(B,1)`, the coefficient needing `(B,T,1)`) **raised** instead of silently
  broadcasting the time axis against the vocab axis. With Day 1's `vocab=4, t_max=4` they would
  have computed garbage.

### Live bugs caught

- `log_pi_theta_all[actions]` indexes **axis 0**, not the vocab axis; and `take_along_axis`
  needs the new length-1 axis on the axis being gathered (`actions[..., None]`, not
  `actions[:, None]`). Fourth appearance of "name the axis you operate on."
- Two missing broadcast axes in `surrogate_gradient` (see shape trap above).
- `frac_high` / `frac_low` **swapped** — and my harness did not catch it, because the only
  check touching them was `frac_high + frac_low == clip_frac`, which is symmetric in the two.
  Added semantic checks pinning each label to its cell. General rule: a check on an aggregate
  cannot validate a decomposition.
- Logratios computed once per epoch instead of per minibatch — theta moves every minibatch
  step, so the ratios were stale from the second minibatch onward. This is the concept block's
  "rho != 1 even within epoch 1" as a bug.
- `list - list` in `sequence_ratio_stats` (D1's `list + ndarray` rule from the other side).
- `drift_experiment` returned `(theta, hist)` instead of a dict with `theta` as a key.
- Weighted vs unweighted quantiles: `median`/`p99` are quantiles under `p_old`, not of a list.
  Third appearance of the occupancy-weighting bug family (W3 D5's actual bug, D1's assertion).

### Harness bugs I shipped before any learner code ran (six sessions running)

1. "The A<0 cell has an unbounded **gradient**" — false in a tabular softmax: driving rho up
   saturates pi, `grad_z log pi = e_a - p -> 0`, so the gradient was exactly 0.0. The
   unboundedness lives in the **objective** and in rho, not in this parameterisation.
2. Two exact-equality checks that ignored the 4e-7 of mass which still terminates early when
   eos is suppressed to -14 rather than -inf.
3. The concavity assertion had the inequality **backwards** (iterating by increasing beta is
   *decreasing* KL).
4. A `vocab=2` spread fixture that was a silent null experiment: with one real token its
   probability is already ~1, so perturbing the logit changes nothing and sigma read 0.0000 at
   every T.
5. `--fast` truncating the T ladder made a true extrapolation assertion false; restated as an
   effective sample *count* so the threshold does not move with the fixture.

## Failure Modes / Debugging Notes

- **"Clipping keeps the policy close to the old policy" is false as stated.** It removes the
  incentive to go further out. Nothing bounds rho; a single large step can put it at 5 and the
  clip will not pull it back, it will only decline to pay for the trip.
- **Never debug PPO from the loss curve.** `L_IS(theta_old) = E[A] ~ 0` by construction. Read
  clip fraction, `KL(pi_theta || pi_ref)`, entropy, and (Day 3) the value head's explained
  variance.
- **Log the walk, not just the step.** `KL(pi_theta || pi_old)` looked fine in every run above
  (mean 0.017); the run that had gone 4.68 nats from `pi_ref` was invisible in that instrument.
- **`frac_high` and `frac_low` call for opposite responses.** `frac_high` climbing = the policy
  is trying to escape upward; `frac_low` climbing = entropy collapse. A dashboard with them
  transposed is worse than one without them.
- **Epochs x minibatches, not epochs**, is how far off-policy you are. "We only do 4 epochs" is
  not a safety argument; the clip fraction is the measurement.
- **Don't switch an instrument off with the mechanism it measures.** Reporting `oob_frac = 0`
  when `eps is None` asserts "nothing drifted," which is exactly false in the run where you
  most need the reading. Compute it at a fixed reference eps regardless.
- The `A<0, rho>1+eps` cell is where PPO takes its biggest steps and is unbounded; dual-clip
  patches it with an outer `max(., c A)`, c ~ 3.

## Interview Answer

Prompt:

> Explain PPO clipping and why old logprobs are necessary.

Answer:

PPO exists because rollouts are 10-100x more expensive than gradient steps, so you want to
take many steps per batch. The moment you take the second step your data is off-policy, and
correcting for that requires the ratio `rho_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)` — which is
why you cache the old log-probs at rollout time. They are not bookkeeping; they are the
denominator of the estimator, and they are the only thing you need to keep from the rollout
worker.

The ratio is per **token**, not per sequence, and that is a real decision. A sequence ratio is
`prod_t rho_t`, so its log is a sum of T terms and it is log-normal: the mean is exactly 1
because it is a likelihood ratio, while the median collapses. I measured the scaling law
exactly by enumeration — `sigma(T) = sigma_1 sqrt(T)` to 1.1%, and `ESS/n = exp(-sigma^2)` to
1.6% — and extrapolating it with 88.5% of individual tokens *inside* the eps=0.2 band gives an
effective sample size of 0.38 out of a 1024-sequence batch at T=500. A correct
high-variance estimator that concentrates all its weight on one sequence is not usable, so PPO
swaps it for a biased usable one.

Clipping then flattens the objective outside `[1-eps, 1+eps]`: `L_CLIP = E[min(rho A,
clip(rho) A)]`, which is a pessimistic lower bound on the surrogate, and whose gradient is
exactly zero in the two cells where the update would push further out — `A>0` with
`rho>1+eps`, and `A<0` with `rho<1-eps`. In the other two the gradient flows at full strength,
including the `A<0, rho>1+eps` cell, which is unbounded and is what dual-clip variants target.

What it guarantees is only this: **no reward for going further out.** It does not bound rho —
PPO approximates a constraint by deleting the gradient, not by enforcing anything, and a large
step can leave the band with the clip declining to pull it back. It does not bound the KL to
the reference: the clip is per token (T tokens at the boundary give `T eps^2/2`, unbounded in
T), it binds per sample rather than in expectation, and it is measured against `pi_old`, which
**resets every iteration**. So clipping bounds the step, never the walk. I ran that: 25
iterations at a mean clip fraction of **5.3%** — every step comfortably inside the trust
region — and the policy still walked **4.68 nats** from `pi_ref`, against **0.75 nats** for a
beta-penalised run that plateaued. That is 6.2x, and it is why InstructGPT ships the clip *and*
an explicit KL penalty: they bound different objects and neither implies the other.

One more thing worth knowing about what clipping does and doesn't buy. I measured the angle
between the PPO update and the true gradient (I had exact ground truth by enumeration), and
clipping makes that angle **worse** — 43.8 degrees versus 11.8 at 16 epochs — while cutting the
KL travelled by **29.3x**. Clipping protects step size, not direction; the extra degrees are
the bias, and PPO is deliberately biased because a conservative update is the product. And note
the unclipped run scored higher reward on the same batch, which is what a Goodhart failure
looks like from the inside: the run that has left the trust region looks better on every
instrument you can see from within it.

## Reusable Artifact

- Path: `week-04-rlhf-ppo/code/ppo_loss.py`
- How to reuse:
  - `from ppo_loss import Batch, collect_batch, exact_batch, token_logratios, is_surrogate,
    clipped_surrogate, surrogate_gradient, ppo_update` — Day 3 adds a value head and a
    per-token KL reward to exactly this loop; Day 5's ablation grid flips its switches.
  - `exact_batch(env, theta, reward_fn)` makes any new estimator's expectation exact, so
    unbiasedness stays an assertion.
  - `score_function_gradient` + `finite_difference_gradient` are the two independent references
    to check any new gradient against.
  - `drift_experiment(..., mode=...)` is the 2x2 of (clip on/off) x (penalty on/off); Day 5
    replaces its fixed beta with a dual-gradient-ascent controller and checks the fixed point
    against the `dE[r]/dKL = beta` identity measured here.
  - `./.venv/bin/python ai-lab-mle-prep/week-04-rlhf-ppo/code/ppo_loss.py --list` for stages;
    `--fast` for a 1.5 s iteration loop.
