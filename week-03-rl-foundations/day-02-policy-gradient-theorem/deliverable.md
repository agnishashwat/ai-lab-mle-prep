# Deliverable - Week 03 Day 02: Policy Gradient Theorem

## What I Built / Derived

Derived the policy gradient theorem from first principles (log-derivative trick ->
dynamics vanish -> reward-to-go causality -> the `Psi_t` ladder), then built
`code/reinforce.py`: a tabular REINFORCE implementation on the Day-1 two-state chain
whose self-check harness turns the derivation's two claims into assertions.

The design trick that makes this more than a toy: the environment is a **finite MDP**,
so `J_T(theta) = E[sum_{t<T} gamma^t r_t]` is computable exactly by `T` Bellman backups
from `V_0 = 0`. A central finite difference of that closed form is a **ground-truth
gradient**, so "REINFORCE is unbiased" becomes a numerical test rather than a claim.
This is the RL analogue of the W1-D2 finite-difference gradient check.

Results (9/9 self-checks pass, ~27 s):

```
MC discounted return == exact J_T      MC 1.9490 vs exact 1.9469
total_return estimator is unbiased     max|err| 0.0069
reward_to_go estimator is unbiased     max|err| 0.0038
reward_to_go variance                  6.220 vs 19.567  -> 3.15x reduction
REINFORCE improves over random         7.916 vs 4.018 (max 8.0)
learned pi = [[0.020 0.980],           == pi* from Day-1 value iteration
              [0.994 0.006]]              ("reach and stay")
```

Two structural facts fell out of the printed gradient that I had not set up:
every gradient **row sums to zero** (softmax is over-parameterized -- `J` is flat along
the all-ones direction, the same null space as the `diag(s) - ss^T` Jacobian), and the
state-0 gradient is 2.7x larger than state-1's (discounted **state-visitation** weighting:
early states are visited with weight `gamma^0`). Corollary: a policy-gradient row that
does not sum to zero is a bug.

## Core Takeaway

In RL the sampling distribution itself depends on `theta`, so you cannot backprop
through the objective; you use the score-function estimator
`grad E_{p_theta}[f] = E[f * grad log p_theta]`, which needs only that you can EVALUATE
`f` (black-box reward, discrete tokens) and differentiate your own log-policy. The
trajectory density factorizes as `rho_0 * prod(pi * P)`, and since neither `rho_0` nor
`P` carries `theta`, the score collapses to `sum_t grad log pi_theta(a_t|s_t)` -- that is
what "model-free" means. Everything after this in RL is variance reduction on top of an
estimator that only ever sees a scalar score.

## Key Equations

```
J(theta)      = E_{tau ~ p_theta}[R(tau)]
grad J        = E[ R(tau) * grad log p_theta(tau) ]                (log-derivative)
p_theta(tau)  = rho_0(s_0) * prod_t pi_theta(a_t|s_t) P(s_{t+1}|s_t,a_t)
grad log p    = sum_t grad log pi_theta(a_t|s_t)                   (rho_0, P drop out)
grad J        = E[ sum_t grad log pi_theta(a_t|s_t) * Psi_t ]      (the general form)
EGLP          = E_{a~pi}[grad log pi(a|s)] = grad sum_a pi = grad 1 = 0
```

Causality / reward-to-go, for `t' < t`:

```
E[ grad log pi(a_t|s_t) * r_t' ]
  = E_{h_t}[ r_t' * E_{a_t~pi}[ grad log pi(a_t|s_t) | h_t ] ]     (tower rule; r_t' is
  = E_{h_t}[ r_t' * 0 ] = 0                                         h_t-measurable)
```

Given the past, `a_t` is still a fresh draw whose score has mean zero, and the past
reward is a constant multiplier. For `t' >= t` the argument fails (`r_t'` depends on
`a_t`) -- that asymmetry IS the causality statement.

The `Psi_t` ladder, high variance to low:

| `Psi_t` | Bias | Note |
|---|---|---|
| `R(tau)` | unbiased | worst; credits actions for the past |
| `G_t` reward-to-go | unbiased | today |
| `G_t - b(s_t)` | unbiased (EGLP) | Day 3 |
| `Q^pi(s_t,a_t)` | unbiased | expectation replaces a sample |
| `A^pi = Q - V` | unbiased | lowest unbiased |
| `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)` | biased if `V != V^pi` | lowest |
| `GAE(gamma, lambda)` | tunable | interpolates; Day 4 -> PPO |

## Implementation Notes

- **A policy gradient is cross-entropy on your own samples, reweighted.** Identical
  `log_softmax -> gather` idiom as LM training; only the gathered token (sampled, not
  gold) and the per-position weight (`Psi_t`, not `1/BT`) differ.
- `grad_z log pi(a|s) = e_a - softmax(z)` -- the exact **negative** of W1's
  `dCE/dz = p - e_y`. When `Psi_t < 0` the update pushes mass AWAY from what was
  sampled; SFT structurally cannot express that.
- Indexing: `theta[states]` for whole-row lookup (an embedding lookup; backward is a
  **scatter-add**, so repeated state visits accumulate correctly), `gather` for
  one-element-per-position. `gather` requires `index.shape == output.shape`;
  `take_along_dim` broadcasts it for you. Batched-table case needs
  `theta[arange(B)[:,None], states]` -- `theta[:, states]` silently gives `(B,B,T,A)`.
- The surrogate `sum_t Psi_t log pi_t` is a **gradient generator, not an objective**;
  its value is meaningless (watch mean reward, not `policy_loss`). Must be `.sum()`,
  not `.mean()`, or the gradient is off by a factor of `T`.
- `torch.autograd.grad` returns a **tuple** (one entry per input), leaves `.grad`
  untouched -- the right tool inside a variance-measurement loop, where `.backward()`
  would silently accumulate. `.detach()` on the result is redundant at
  `create_graph=False`.
- The `gamma^t` factor on the reward-to-go weight is needed for exact unbiasedness
  w.r.t. the discounted objective. PPO/GRPO **drop it** -- formally biased, empirically
  better conditioned.
- Optimizer form: optimizers DESCEND, so ascent needs `theta.grad = -g` (or
  `SGD(..., maximize=True)`). Production code never materializes a gradient array --
  it minimizes `loss = -(w * logp).sum() / batch` and calls `.backward()`.

## Failure Modes / Debugging Notes

- **Accumulator dtype is part of correctness.** `np.zeros(..., dtype=np.long)` for a
  gradient buffer truncates every float to 0, and nothing raises -- the unbiasedness
  check would have "passed" a perfectly consistent gradient of zero. Same family as the
  W2-D5 `zeros_like(bool_mask)` test-design bug.
- **An episode-internal baseline is biased.** `G_t - G.mean()` over the SAME episode
  makes `b` depend on `a_t` and all future actions, so EGLP no longer applies. A valid
  baseline must be `h_t`-measurable -- known before `a_t` is drawn. It is a batch-level
  quantity; a single-episode function structurally cannot compute one.
- **Off-policy-by-accident is the #1 RLHF sampling bug.** Rolling out at temperature
  0.7 / top-p 0.9 means the behavior policy is not `pi_theta`: both distortions
  oversample already-likely tokens, so you reinforce the mode and get entropy collapse
  ("reward up, then degeneration"). Fix: sample RL rollouts at T=1.0 untruncated;
  importance-sample if you must (variance explodes over a product of T ratios -> the
  reason PPO uses a clipped per-token ratio); note that clipping cannot repair top-p,
  which is a **support** mismatch. Instrument entropy and `KL(pi_theta || pi_ref)`.
- `.detach()` is not in-place -- a bare `g.detach()` statement is a no-op (use
  `g = g.detach()` or `g.detach_()`).
- In-place update of a leaf that requires grad raises; scope the step in `no_grad`.
- `np.var` already divides by `n`. Per-episode variance (estimator property) vs variance
  of the batch mean (`/n`, the `1/n` law) are different quantities -- batch size is the
  expensive variance lever; baselines and GAE attack the numerator instead.

## Interview Answer

Prompt:

> Derive the policy gradient theorem, and explain why the transition model disappears.

Answer:

Start from `J(theta) = E_{tau ~ p_theta}[R(tau)] = sum_tau p_theta(tau) R(tau)`.
Differentiating, `grad J = sum_tau R(tau) grad p_theta(tau)`; multiply and divide by
`p_theta` to get `sum_tau p_theta R grad log p_theta = E[R(tau) grad log p_theta(tau)]`.
That is the log-derivative or score-function trick, and it's necessary because the
distribution we're averaging over is itself a function of theta -- I can't just
differentiate the integrand. Regularity: we assume we can exchange differentiation and
integration, and that the support of `p_theta` doesn't depend on theta, which holds for
a softmax policy.

Now expand the trajectory density:
`p_theta(tau) = rho_0(s_0) prod_t pi_theta(a_t|s_t) P(s_{t+1}|s_t,a_t)`. Taking logs
turns the product into a sum, and neither the initial-state distribution nor the
transition kernel contains theta, so both differentiate to zero:
`grad log p_theta(tau) = sum_t grad log pi_theta(a_t|s_t)`. That's what "model-free"
means -- the dynamics still shape which trajectories I see, but they never appear in the
gradient, so I need to be able to *sample* the world, not model it. Model-based methods
need `dP/ds` to backprop through a simulator; this doesn't.

One more refinement: the naive estimator weights every score term by the whole-trajectory
return, including rewards that occurred before the action. Conditioning on the history
`h_t`, those past rewards are constants and `E_{a_t~pi}[grad log pi(a_t|s_t)] = 0` by the
EGLP lemma, so those cross terms are exactly zero in expectation -- pure noise, no signal.
Dropping them gives the reward-to-go form at identical bias and strictly lower variance;
I measured 3.15x on a toy chain, and the ratio grows slowly and monotonically with the
horizon (1.84x at T=2, 3.09x at T=32, 4.09x at T=128 -- both variances grow ~T^2.3, so the
*ratio* does NOT grow like T^2; it is a free, consistent, horizon-growing win). The general
form is `grad J = E[sum_t grad log pi(a_t|s_t) Psi_t]`, and every algorithm from REINFORCE
to PPO differs only in the choice of `Psi_t`.

## Reusable Artifact

- Path: `week-03-rl-foundations/code/reinforce.py`
- How to reuse:
  - `./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/reinforce.py` runs
    all 9 invariants (~27 s).
  - **`exact_J` + `exact_gradient` are the reusable pattern**: on any finite MDP you get
    a closed-form objective and a finite-difference ground-truth gradient, so any new
    estimator (baseline, actor-critic, GAE, an importance-sampled/PPO surrogate) can be
    checked for unbiasedness numerically instead of by argument. Days 3-4 plug straight
    into `episode_weights` -- add a `Psi_t` mode and rerun the same harness.
  - `collect_batch` returns per-episode gradients, which is what makes variance
    measurable; production code would use a single batched `.backward()` instead.
