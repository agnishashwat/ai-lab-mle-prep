# Week 03 Summary - RL Foundations for Post-Training

## The 5-Minute Version

The week builds the RL substrate under PPO, GRPO and DPO from the ground up, on a 2-state
finite MDP chosen for one reason: it is the only setting where **every quantity has an exact
closed-form ground truth**, so "is my estimator unbiased?" and "how much does bootstrapping
bias it?" become printed numbers rather than arguments.

Day 1 established the MDP machinery: Bellman expectation versus optimality, both as
gamma-contractions, and `V^pi`, `Q^pi`, `A^pi` with the identity `E_{a~pi}[A^pi] = 0` per
state -- the reason `V` is the natural baseline. Day 2 derived the policy gradient theorem and
built REINFORCE, with `exact_J` (T Bellman backups) plus central finite differences giving a
**ground-truth gradient**, so unbiasedness became an assertion. Day 3 added baselines: the
EGLP lemma, the measurability condition that makes a baseline valid, the optimal baseline
`b*`, and the entropy bonus. Day 4 added bootstrapping and GAE, sweeping lambda against three
critics chosen along interpretable axes (exact / wrong-level / wrong-ranking). Day 5 changed
the *objective*: KL-constrained RL, its closed-form optimum, and why post-training pins to a
reference.

The through-line is a single equation. Every algorithm from REINFORCE to GRPO is
`grad J = E[sum_t Psi_t grad log pi(a_t|s_t)]` and differs **only in `Psi_t`**. The week walks
the whole ladder and measures each rung on identical rollouts:

```
Psi_t = R(tau)                   variance 20.033   1.00x
      = G_t                              6.317     3.17x   (reward-to-go: delete zero-mean terms)
      = G_t - b(s_t)                     1.765    11.35x   (pooled baseline)
      = G_t - V_t(s_t)                   1.142    17.5x    (time-indexed critic)
      = gamma^t A^GAE_t(lambda)          tunable            (bootstrap; bias = critic error)
```

**17.5x variance reduction at fixed mean**, every step verified unbiased against the
finite-difference gradient. Then Day 5: none of that matters if you optimize the wrong
objective, because nobody maximizes reward -- everyone maximizes reward minus
`beta * KL(pi||pi_ref)`, whose optimum is an exponential tilt of the reference and whose beta
is an exchange rate in reward-per-nat.

## Key Concepts (crisp)

- **Why RL and not supervised learning.** `J(theta) = E_{tau ~ p_theta}[R(tau)]`: the
  *sampling distribution* depends on theta. That, not the non-differentiability of the reward,
  is the fundamental obstruction, and it forces the score-function trick.
- **Model-free-ness is a consequence, not an assumption.** In `p_theta(tau) = rho_0 prod pi P`
  only the `pi` terms carry theta, so `rho_0` and `P` vanish from `grad log p_theta`.
- **Measurability is the master concept.** "`X` is `h_t`-measurable" = if I tell you `h_t` you
  can tell me `X`. It settles reward-to-go (valid), episode-internal baselines (biased, 38%
  measured), critic accuracy (irrelevant to bias), and the GAE `l >= 1` terms (zero-mean). The
  discriminator is measurability, **not** theta-dependence.
- **The baseline/signal division of labour.** `Psi_t` MUST depend on `a_t` (a state-only
  `Psi` gives an identically zero gradient by EGLP); the subtracted baseline must NOT. The
  measurability requirement applies to the baseline alone, never to `Psi_t`.
- **Unbiased and variance-reducing are independent properties.** A horizon-mismatched oracle
  `V^pi` was unbiased and *increased* variance 27%. Low variance alone is worthless too: the
  biased within-episode baseline had the lowest variance of anything measured (25.6x) with a
  mean 38% wrong. Only "variance reduction at fixed mean" is a claim.
- **gamma is part of the problem, lambda is part of the estimator.** gamma changes `J`, `V`,
  and the optimal policy; lambda changes only bias/variance of an estimator of a fixed target.
- **Bootstrap bias is the state-VARYING part of the critic's error.** A uniform offset is a
  constant, a constant is a baseline, and a baseline is free. Wrong level costs variance;
  wrong ranking costs bias.
- **Post-training is constrained optimization, not reward maximization.** The KL term is
  simultaneously epistemic (where the RM is valid), structural (support confinement), and
  numerical (the only dense per-token signal).
- **Reverse KL confines you to the reference's support.** Hence RLHF elicits, it does not
  teach; SFT must precede RL; and nucleus-sampled rollouts are unrepairable.
- **An entropy bonus is a KL penalty to a uniform reference** (`KL(pi||Unif) = log A - H`).
  RLHF's one change is replacing "uniform" with "known-good".

## Key Equations

```
objective      J(theta) = E_{tau~p_theta}[R(tau)],   p_theta(tau) = rho_0 prod_t pi_theta P

policy grad    grad J = E[ sum_t Psi_t grad log pi_theta(a_t|s_t) ]
EGLP           E_{a~pi}[grad log pi] = grad sum_a pi = grad 1 = 0
               => any h_t-measurable baseline is unbiased (mean unchanged, variance lower)

Bellman        V^pi(s) = sum_a pi [ r + gamma sum_s' P V^pi(s') ]        (expectation)
               V^*(s)  = max_a [ r + gamma sum_s' P V^*(s') ]            (optimality)
advantage      A^pi = Q^pi - V^pi,     E_{a~pi}[A^pi(s,.)] = 0

optimal b      b*(s) = E[ ||grad log pi||^2 G ] / E[ ||grad log pi||^2 ]   (V^pi = unweighted)

TD / GAE       delta_t = r_t + gamma V_{t+1}(s_{t+1}) - V_t(s_t),  V_T = 0
               A^(k)_t = sum_{l<k} gamma^l delta_{t+l}
                       = ( sum_{l<k} gamma^l r_{t+l} + gamma^k V(s_{t+k}) ) - V(s_t)
               A^GAE_t = (1-lam) sum_{k>=1} lam^{k-1} A^(k)_t = sum_l (gamma lam)^l delta_{t+l}
               scan:  A_t = delta_t + gamma lam (1 - done_t) A_{t+1},  A_T = 0

KL-constrained J(pi) = E_{a~pi}[r] - beta KL(pi || pi_ref)
               pi*(a|s) = pi_ref(a|s) e^{r(s,a)/beta} / Z(s),  J* = beta log Z(s)
               dJ*/dbeta = -KL(pi*||pi_ref),   dE[r]/dKL = beta
               supp(pi*) subset supp(pi_ref)  for every beta > 0

sequence KL    KL(p_theta(tau)||p_ref(tau)) = sum_t E_{d_t}[ KL(pi_theta(.|s)||pi_ref(.|s)) ]
estimators     k1 = lr,  k2 = 0.5 lr^2,  k3 = expm1(-lr) + lr,   lr = log(pi/pi_ref)
shaping        r_t <- r_t - beta k_i(lr_t);  E[sum_t penalty] = beta * sequence KL

implicit r     r = beta log(pi*/pi_ref) + beta log Z;  pairwise => Z cancels => DPO
```

## What I Implemented

| Day | File | Checks | Headline number |
|---|---|---|---|
| 1 | `code/mdp.py` | 7/7 | `V^* = [10,10] = 1/(1-gamma)`; iterative eval == linear solve |
| 2 | `code/reinforce.py` | 9/9 | both estimators match the FD gradient (`max err 0.0069`); reward-to-go 3.15x |
| 3 | `code/actor_critic.py` | 39/39 | variance ladder to 13.57x; oracle `V^pi` unbiased but +27% variance |
| 4 | `code/gae.py` | 64/64 | bootstrap bias 0.0010 (wrong level) vs 1.5557 (wrong ranking) |
| 5 | `code/kl_rl.py` | 94/94 | `dE[r]/dKL = beta` to `rtol 1e-4`; Goodhart hump at `beta = 0.16` |

**213 property-based checks**, all against exact or finite-difference ground truth. The
harness convention (staged, one stage per TODO, `NotImplementedError -> skip`,
`--fast`/`--list`/named-subset CLI, lazy seeded `Ctx` fixtures, `_fail_fast` probes) was
promoted repo-wide on Day 3 and is now the template for every exercise.

## Failure Modes I Now Understand

- **Accumulator dtype is part of correctness.** `np.zeros(dtype=np.long)` for a gradient
  buffer silently truncates every float to 0 and all downstream checks "pass" on zeros.
- **Action-dependent baselines DELETE the gradient**, not merely bias it: with `b = Q^pi` the
  mean is exactly the zero matrix. Per-sample the weights look perfectly centred.
- **`theta += numpy_array` is not in-place.** numpy takes over, rebinds the name, and theta
  becomes a fresh tensor with `requires_grad=False` -- only the property you depend on dies.
- **Silent positional-argument bugs need two adjacent same-typed parameters.** Seen three
  times (`T==V`, `TransformerBlock(d, n_heads, n_heads)`, `Batch(..., n_iters, ...)`). The
  last one *passed* because more episodes meant better convergence -- a bug that improves your
  metric is the hardest kind to find. The tell was the wall clock.
- **Untested branches are broken until proven otherwise** (`critic="exact"` was a `TypeError`
  under a 64/64 green run). Coverage, not pass count.
- **`np.where` is not lazy**, and guarding the *result* is not guarding the *argument*.
  Guarding the log then breaks the `+inf` convention, so the convention has to be asserted.
- **A metric can be wrong in a direction no test can see.** `kl_history` measured an
  unweighted per-state mean instead of the state-distribution-weighted sequence KL -- 1.053
  where the truth was 5.056 -- and all four assertions passed because they tested
  monotonicity in beta. The tell: a provided helper the code never called.
- **Over-optimization is not an RL artefact, it is Goodhart**, and it eventually takes the true
  reward *below* the reference's own.
- **The `beta * f(x)` trap**: `beta * f(x)` still evaluates `f(x)` when `beta == 0`.
- **`no_grad` scopes by code region**, so a graph you need cannot be built inside it; and
  logging tensors instead of floats retains that iteration's whole graph.

## The methodological lesson of the week

**Four of five days, a harness assertion I wrote was wrong, and each time checking the CLAIM
before the code produced the day's best result:**

- W3 D3: "the oracle `V^pi` baseline also stacks" -- it doesn't (unbiased, +27% variance)
  => unbiasedness and variance reduction are independent.
- W3 D4: "GAE(lambda=0) with a mismatched critic IS biased" -- it isn't => bootstrap bias is
  the state-*varying* part of the error, found only by inventing a second wrong critic.
- W3 D4: "GAE(lambda) lies between the lambda=0 and lambda=1 estimators" -- simply false.
- W3 D4: an unvisitable `(t=0, s != s0)` bucket read 0.0 legitimately.

Corollary now standing as a rule: **when a well-motivated check fails, check the claim before
the code**, and **assert that the bug is detectably a bug** (negative controls, relative
thresholds).

## Open Questions / To Revisit

- PPO's clipped ratio as an *implicit* local KL trust region versus Day 5's explicit penalty --
  when do they disagree?
- The same lambda sweep against a **learned** `V_phi` fit by regression rather than a table, so
  critic error is realistic (correlated across states, shrinking over training). Prediction:
  early in training the error is dominated by a uniform level offset (harmless) and only later
  becomes ranking error (harmful), which would explain why PPO tolerates a badly-fit critic
  for the first few hundred steps.
- KL as reward shaping versus KL as an explicit loss term with an analytic gradient. The local
  term has a closed-form gradient and needs no sampling; only the *visitation* part of the KL
  gradient requires a score-function estimator, and every implementation silently drops it.
- Adaptive beta (InstructGPT's KL-target controller) and where the reward/KL frontier that a
  real run traces sits relative to the closed-form Pareto curve.
- PG regularity conditions cold: the `grad`/`sum` swap, `supp(p_theta)` independent of theta
  (top-p violates it), `R(tau)` not a function of theta.
