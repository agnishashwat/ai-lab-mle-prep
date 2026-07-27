# Deliverable - Week 03 Day 04: GAE, Bootstrapping & the Bias/Variance Tradeoff

## What I Built / Derived

Derived GAE as an exponentially-weighted average of the `k`-step advantage estimators, then
built `code/gae.py`, which plugs `delta_t` and `GAE(gamma, lambda)` into Day 3's
`weight_fn` machinery. Because the environment is still the Day-1 finite MDP, Day 2's
finite-difference `exact_gradient` remains available as ground truth -- so "how much does
bootstrapping bias the gradient?" is a **measured number per lambda**, not an argument.
**64/64 staged checks pass (~60 s).**

Also built the time-indexed finite-horizon critic `V_t(s)` that Day 3 left as an open item,
which turns Day 3's cautionary "unbiased but worse" result into a constructive fix.

## The Headline: the lambda dial, measured against three critics

Identical rollouts (20k episodes, one fixed `Batch`); only `Psi_t` differs. `MSE` is the
single-sample `||bias||^2 + variance`.

```
EXACT finite-horizon V_t  -- the right critic
  lambda  max|bias|  ||bias||   variance      MSE
    0.00     0.0003    0.0005      0.217    0.217   <- best
    0.50     0.0011    0.0017      0.274    0.274
    0.90     0.0023    0.0033      0.752    0.752
    0.95     0.0025    0.0035      0.919    0.919
    1.00     0.0027    0.0038      1.147    1.147

WRONG RANKING (+1/-1 per state), right level
  lambda  max|bias|  ||bias||   variance      MSE
    0.00     1.5557    2.3369      1.494    6.954
    0.50     0.7649    1.1490      1.773    3.094
    0.90     0.1531    0.2289      3.006    3.059   <- best (INTERIOR)
    0.95     0.0775    0.1153      3.397    3.410
    1.00     0.0023    0.0044      3.930    3.930

WRONG LEVEL (infinite-horizon V^pi), right ranking
  lambda  max|bias|  ||bias||   variance      MSE
    0.00     0.0010    0.0017      0.975    0.975   <- best
    0.50     0.0011    0.0018      1.223    1.223
    0.90     0.0037    0.0055      4.065    4.065
    0.95     0.0042    0.0062      5.511    5.511
    1.00     0.0046    0.0071      7.769    7.769
```

### The three results that matter

1. **`lambda = 1` is unbiased for ANY critic, however wrong.** At `lambda = 1` the critic
   appears only as `-V(s_t)` -- a state-only function, therefore `h_t`-measurable,
   therefore a Day-3 baseline. Measured at `max|err| = 0.0023-0.0046` for all three
   critics including the deliberately-broken one. Correspondingly, **`lambda < 1`
   bootstraps, and is unbiased only if the critic is right.**

2. **With an exact critic, EVERY lambda is unbiased -- so `lambda = 0` strictly wins.**
   Bias is flat at ~0.003 across the whole sweep while variance rises 5.3x from
   `lambda = 0` to `lambda = 1`. The `l >= 1` terms of GAE are *pure zero-mean noise* when
   the critic is correct (proof below). Corollary worth saying out loud in an interview:
   **`lambda > 0` exists only as a hedge against a critic you do not trust.** An interior
   optimum appears exactly when the critic is wrong -- `lambda* = 0.9` for the
   wrong-ranking critic, with MSE 3.059 versus 6.954 at `lambda=0` and 3.930 at
   `lambda=1`.

3. **Bootstrap bias comes from the critic's error being STATE-DEPENDENT, not from its
   magnitude.** The infinite-horizon `V^pi` is off by **+2.174 in every row** and yet
   bootstraps essentially bias-free (`max|err| 0.0010` at `lambda=0`), because the error
   enters `delta_t` only through `gamma * e(s') - e(s)`, and a state-independent error is
   a constant, and a constant is a baseline. Meanwhile a critic with the *right* average
   level but a `+1/-1` per-state skew is biased by **1.5557** at `lambda = 0`, a factor of
   ~1500 worse. So: **wrong LEVEL costs variance; wrong RANKING costs bias.** An
   unconverged PPO critic with a uniform offset is far less dangerous than one that ranks
   states incorrectly, even at identical explained variance.

### The baseline ladder: time-indexing is the constructive fix for Day 3

All at `lambda = 1`, identical rollouts, only the value table differs:

```
  no baseline (V = 0)          6.317   1.00x    (== Day 2 reward_to_go, exactly)
  pooled b(s)      [Day 3]     1.765   3.58x
  fitted V_t(s)    [today]     1.142   5.53x    (1.55x on top of Day 3)
  exact V_t(s)     [oracle]    1.147   5.51x
  infinite-horizon V^pi        7.769   0.81x    <- Day 3's result, now explained
```

The MC-fitted `V_t(s)` is **statistically indistinguishable from the exact oracle table**
(1.142 vs 1.147 -- the fitted one is nominally lower, which is noise). Day 3's mismatched
`V^pi` was not a mysterious anomaly: it was the *infinite-horizon* value function applied
to a *finite-horizon* objective. Conditioning the baseline on `t` as well as `s` fixes it
and buys another 1.55x for free.

Cumulative from Day 2's naive `total_return` (20.033): **17.5x variance reduction**, all of
it at fixed mean.

## Core Takeaway

**GAE's bias is exactly the critic's error, and nothing else.** The `lambda` knob does not
trade "accuracy for speed" or "signal for noise" -- it trades *how much you delegate to the
critic*. With a perfect critic there is nothing to trade and `lambda = 0` dominates; the
entire practical case for `lambda ~ 0.95` is that real value heads are wrong. And what
"wrong" means is specific: it is the state-varying part of the error that costs you, since
a uniform offset cancels in `gamma*e(s') - e(s)` and is absorbed as a baseline.

## Key Equations

```
TD error       delta_t = r_t + gamma V_{t+1}(s_{t+1}) - V_t(s_t),      V_T = 0

k-step         A^(k)_t = sum_{l=0}^{k-1} gamma^l delta_{t+l}
                       = ( sum_{l<k} gamma^l r_{t+l} + gamma^k V(s_{t+k}) ) - V(s_t)
               "k real rewards, then bootstrap, minus the baseline"
               ASSUMES NOTHING ABOUT V -- pure algebra on one sampled trajectory

telescoping    k = T-t with V_T = 0   =>   A^(T-t)_t = G_t - V_t(s_t)

GAE            A^GAE_t = (1-lam) sum_{k>=1} lam^(k-1) A^(k)_t
                       = sum_{l>=0} (gamma lam)^l delta_{t+l}
               weight on delta_{t+l} is (1-lam) sum_{k>=l+1} lam^(k-1) = lam^l

reverse scan   A_t = delta_t + gamma lam A_{t+1},   A_T = 0        (O(T), stable)
with dones     A_t = delta_t + gamma lam (1 - done_t) A_{t+1}

limits         lam=0 -> delta_t        (1-step TD; min variance, critic's error as bias)
               lam=1 -> G_t - V_t(s_t) (Monte-Carlo; unbiased for ANY V, max variance)

weight         Psi_t = gamma^t * A^GAE_t     (gamma^t = the discounted-objective factor)
```

**Two different powers of gamma.** `gamma^t` out front is "how much the objective cares
about decision `t`" -- it is the leftover from re-basing `sum_{t'>=t} gamma^{t'} r_{t'}`
into `gamma^t G_t`, and it is what keeps the estimator exactly unbiased for `grad J_T`.
`gamma^l` inside `A^GAE_t` is discounting *within* the advantage estimate, indexed by
offset from `t`. Conflating them is the likeliest silent bug in `weights_gae`.

### gamma vs lambda: not the same knob

- **`gamma` is part of the PROBLEM.** It defines `J = E[sum gamma^t r_t]`, so changing it
  changes what "optimal" means, what `V`/`Q`/`A` denote, and the effective horizon
  `~1/(1-gamma)`. Lowering gamma "for variance" quietly optimizes a different objective.
- **`lambda` is part of the ESTIMATOR.** It appears in no true quantity; the target is
  identical for every lambda. There is no lambda-optimal policy.
- Schulman's one-liner: **gamma introduces bias regardless of the value function's
  accuracy; lambda introduces bias only when the value function is inaccurate.**

### Why GAE is unbiased at every lambda when the critic is exact

The estimator is `g_hat = sum_t gamma^t (sum_l (gamma lam)^l delta_{t+l}) grad log pi_t`.
By linearity this is a double sum over pairs `(t, l)`; the two cases die by *different*
arguments.

- **`l = 0`:** condition on `(s_t, a_t)`. `grad log pi(a_t|s_t)` is a deterministic
  function of what was conditioned on, so it pulls out. The Bellman peel gives
  `E[r_t + gamma V_{t+1}(s_{t+1}) | s_t,a_t] = Q_t(s_t,a_t)`, hence
  `E[delta_t | s_t,a_t] = A_t`. Summing over `t` gives exactly `grad J_T` -- the advantage
  form of the policy gradient theorem.
- **`l >= 1`:** condition on `h_{t+l} = (s_0,a_0,...,s_{t+l})` -- up to and including
  `s_{t+l}`, but NOT `a_{t+l}`. Then `grad log pi(a_t|s_t)` is `h_{t+l}`-measurable
  (since `t < t+l`) and pulls out, while
  `E[delta_{t+l} | h_{t+l}] = E_{a~pi}[A_{t+l}(s_{t+l},a)] = 0` by the Day-1 identity.
  Every such term is **exactly zero**.

Same tower-rule-plus-EGLP argument as Day 3, just conditioned at `t+l` instead of `t`.

### Where it breaks when V is wrong

With `e_t(s) = V(s) - V_t(s)`:

```
E[ delta_t | s_t, a_t ] = A_t(s_t,a_t) + gamma E_{s'~P(.|s_t,a_t)}[ e_{t+1}(s') ] - e_t(s_t)
```

- `-e_t(s_t)` is a **state-only function = a baseline**. Contributes exactly 0. Free.
- `gamma E_{s'}[e_{t+1}(s')]` is a deterministic function of `(s_t, a_t)` -- it **depends
  on the action**. That makes it an **action-dependent baseline**, which is Day 3's
  catastrophic case in a milder dose (there, `b = Q^pi` deleted the gradient entirely).

**Condition for the bias to vanish anyway:** `E_{s'~P(.|s,a)}[e_{t+1}(s')]` must not depend
on `a`. The clean sufficient case is `e_{t+1}` constant across states -- a uniform offset,
which is then just a constant, i.e. a baseline. This is what the "wrong LEVEL" row of the
sweep measures.

## Implementation Notes

- **Value tables are `(T+1, S)` with `values[T, :] = 0`.** Materializing the terminal row
  as *data* rather than as a remembered convention is what makes the last TD error correct
  without a special case: `s_T` is never recorded by `Batch` and never needs to be. Same
  trick used in `gae_reverse` (`A = np.zeros(T+1)`, return `A[:T]`), which makes the
  classic GAE off-by-one at the horizon structurally impossible.
- **`lambda=1` reproduces Day 3 and Day 2 EXACTLY, elementwise** (`max|diff| ~ 1e-16`),
  because the telescoping identity is algebraic, not statistical. The strongest kind of
  regression test: a more general estimator that provably contains the old one. And
  `V = 0` at `lambda = 1` reproduces the 6.317 variance to the digit.
- **Keep the slow reference structurally independent of the fast one.** `gae_naive` (double
  loop) and `k_step_advantage` must not be implemented via `gae_reverse`, or they can only
  ever agree with it. This caught a real bug: `A[T-1]` never being assigned was invisible
  to any scan-vs-scan comparison but failed immediately against `k_step_advantage`.
- **Don't use the cumsum trick for `k`-step.** `sum_{l<k} gamma^l delta_{t+l} =
  gamma^{-t}(C_{t+k} - C_t)` is O(T) and correct, but `gamma^{-t}` is a numerical landmine
  (`0.9^-999 ~ 1e45`) multiplying a catastrophically-cancelled difference. The reverse scan
  is O(T) *and* stable -- which is exactly why `discount_cumsum` is a backward recursion.
- Fully vectorized `k`-step:
  `sliding_window_view(np.concatenate([d, np.zeros(k-1)]), k) @ gamma**np.arange(k)`
  (a stride view, no copy).

## Failure Modes / Debugging Notes

- **`torch.softmax(theta, ...)` into a numpy function raises `Can't call numpy() on Tensor
  that requires grad`.** The traceback names `.numpy()`, which appears nowhere in the code
  -- the Day-3 heuristic again: *an error naming an operation you never called is a
  type-coercion fallback; look at operand types.* Fix: `policy_probs(theta).detach()
  .numpy()`. Detaching is not a workaround, it is the correct assertion -- the critic is
  DATA, and PPO keeps actor and critic graphs separate for the same reason. Detaching the
  *input* (`torch.softmax(theta.detach(), ...)`) is marginally better: no graph is built at
  all rather than built and discarded.
- **`return A[t]` after a `for` loop returns a scalar**, and `np.allclose` broadcasts it
  against the reference without complaint, producing a numeric diff rather than a shape
  error. Never read a loop variable after its loop -- if `T == 1` the range is empty and
  it is unbound, giving `UnboundLocalError` from an innocuous-looking index.
- **Silent positional-argument bug, third occurrence.** `Batch(mdp, theta, T, s0, n_iters,
  rng)` passed `n_iters=200` where `batch_size=32` belongs, so `batch_size` did nothing and
  every iteration rolled 6.25x too many episodes. It **passed the test**, because more
  episodes means less gradient noise means better convergence -- *a bug that improves your
  metric is the hardest kind to find*. The tell was the wall clock: 102 s versus the 60 s
  the same file takes when correct. Same family as `TransformerBlock(d, n_heads, n_heads)`
  (W2-D2) and the `T==V` cross-entropy shape (W1-D5). **Use keyword arguments when two
  adjacent parameters share a type.**
- **Untested branches are broken until proven otherwise.** `critic="exact"` called
  `exact_time_indexed_values(mdp, theta)` without `T` -- an immediate `TypeError` that
  survived a 64/64 green run because `stage_train` only exercises `critic="fit"`.
- **`history.append(batch.total_returns)`** appended the whole `(n,)` array instead of its
  mean, and passed only because `np.mean` over equal-sized batches flattens to the right
  number by coincidence. Day-3's "log floats, not tensors", one container over.
- `np.bincount`'s third positional parameter is `minlength`, with the optional `weights` in
  between -- always pass it by keyword.
- Returning `V[:T]` from the value-table builder silently drops the terminal zero row that
  the telescoping identity depends on.

## Test-Design Notes (harness)

The harness's own claims were wrong **three times** before the file was handed over, which
is now the fourth session where "check the CLAIM before the code" produced the day's best
lesson (W1-D5 noiseless-overfit, W2-D5 toothless cached==uncached, W3-D3 oracle-stacks, and
now this):

- `fitted critic approximates the exact table` failed because the `(t=0, s != s0)` bucket
  can never be visited -- every episode starts at `s0`. It reads 0.0 legitimately. Fix:
  restrict the comparison to visited buckets and assert the unvisited one is exactly 0.0.
- `GAE(lam=0) with the mismatched critic IS biased` failed **because it isn't** -- which is
  how result #3 above was discovered. Fixing the assertion required inventing a *second*
  wrong critic (wrong ranking rather than wrong level), and the contrast between the two is
  now the most valuable thing in the file.
- `GAE(lam) lies between the lam=0 and lam=1 estimators` was simply false (the `delta`
  signs vary), and was replaced with a hand-computable closed form: a constant `delta`
  sequence must give the geometric partial sums `sum_l (gamma lam)^l`.

Also kept from Day 3, and it paid off immediately: **assert that the bug is detectably a
bug.** `without dones the boundary LEAKS` is the negative control for
`dones reset the scan` -- a green check whose negative control also passes is the only kind
worth trusting.

## Interview Answer

Prompt:

> In RLHF the reward arrives only at EOS. How do you assign credit to individual tokens,
> and how would you tune it?

Answer:

The textbook answer is a value head plus GAE. Train a critic `V_phi(s_t)` to predict the
return from each partial sequence, form the TD errors
`delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`, and take the advantage as
`A_t = sum_l (gamma lam)^l delta_{t+l}`, computed by a reverse scan over the sequence. The
point is that even though the environment reward is a single scalar at EOS, the *critic's*
value estimate changes at every token, so `delta_t` is nonzero everywhere -- the critic
converts a terminal reward into a dense per-token signal. Tokens after which the expected
score jumps get positive credit; tokens after which it drops get negative credit.

`lambda` controls how much you trust that critic. At `lambda = 1` the critic appears only
as `-V(s_t)`, a state-only baseline, so the estimator is unbiased no matter how bad the
value head is -- but every token in a 2000-token sequence gets essentially the same
sequence-level advantage, which is the credit-assignment problem you were trying to solve.
At `lambda = 0` you get the sharpest per-token signal and inherit the critic's error as
bias. So the choice is: how good is my value head on this distribution?

The non-obvious part -- and I measured this on a toy MDP where I have a finite-difference
ground-truth gradient -- is that *the bias is exactly the critic's error and nothing else*.
With an exact value function every lambda is unbiased, because the `l >= 1` terms have
conditional mean zero: `grad log pi(a_t|s_t)` is measurable with respect to the history up
to `s_{t+l}`, and `E[A_{t+l} | s_{t+l}] = 0`. So with a perfect critic `lambda = 0` wins
outright, and I measured variance rising 5.3x monotonically from `lambda = 0` to
`lambda = 1` with no bias to show for it. `lambda ~ 0.95` is purely a hedge.

The result I'd actually lead with is *which kind* of critic error hurts. In `delta_t` the
error enters only as `gamma e(s') - e(s)`, so a critic that is uniformly off by a constant
contributes a constant -- and a constant is a baseline, which is free. I measured a critic
biased by +2.17 in every state that bootstrapped essentially bias-free at `lambda = 0`
(`max|err| 0.001`), while a critic with the correct average level but the two states'
values swapped by +/-1 was off by 1.56 -- three orders of magnitude worse. So the diagnostic
for a PPO value head isn't "how close is `V` to the return", it's "does it *rank* states
correctly". A systematically pessimistic critic is nearly harmless; a critic that
mis-orders states poisons every bootstrap.

For debugging unstable PPO, the practical ladder is: check the critic's explained variance
first; if it's near zero, raise lambda toward 1 to stop propagating its error, and in the
limit you've fallen back to a REINFORCE-with-baseline scheme that is unbiased by
construction. If raising lambda fixes the instability, the value head was the problem. And
if zeroing the baseline entirely *reduces* your gradient variance, the critic is worse than
nothing. That last one is worth knowing because the failure is silent -- no NaNs, no crash,
advantages still look zero-mean after whitening.

## Reusable Artifact

- Path: `week-03-rl-foundations/code/gae.py`
- How to reuse:
  - `./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/gae.py` runs all 64
    checks (~60 s); `--fast` for ~17 s; `--list` for the stages; `<stage> [<stage> ...]`
    for a subset. `td kstep scan` need no rollouts and run in ~1 s.
- `gae_reverse` is the production-shaped function: it is exactly what PPO runs, `dones`
  handling included.
- `exact_time_indexed_values` is a reusable oracle for any finite-horizon tabular
  experiment -- it makes "is my critic wrong, and in which way?" a measurable question,
  which is what made result #3 findable.
- The three-critic pattern (exact / wrong-level / wrong-ranking) is the reusable
  experimental design: to characterize an estimator's bias, do not perturb the critic
  randomly -- perturb it along *interpretable* axes and see which axis the estimator is
  sensitive to.

## Open / Next

- **Day 5 (KL-constrained RL)** closes the week: why post-training pins to a reference
  policy. Connects directly to the Day-3 finding that a uniform entropy target is wrong for
  text and `KL(pi_theta || pi_ref)` is the regularizer that actually ships.
- Worth building in Week 4: the same sweep with a **learned** `V_phi` fit by regression
  rather than by table lookup, so critic error is realistic (correlated across states,
  shrinking over training) rather than injected by hand. The prediction to test is that
  early in training the critic's error is dominated by a uniform level offset (harmless)
  and only later becomes ranking error (harmful) -- which would explain why PPO tolerates a
  badly-fit critic in the first few hundred steps.
