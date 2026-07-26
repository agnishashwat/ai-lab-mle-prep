# Deliverable - Week 03 Day 03: Baselines, Advantage & Variance Reduction

## What I Built / Derived

Proved the baseline theorem four ways (EGLP -> any `h_t`-measurable `b` -> why `b` may not
see `a_t` -> the variance-optimal `b*`), then built `code/actor_critic.py`, which extends
Day 2's REINFORCE with baselines, an entropy bonus, and a **staged** self-check harness.
Day 2's `exact_J` / `exact_gradient` finite-difference ground truth carries over, so every
claim below is an assertion, not an argument. **39/39 checks pass (~31 s).**

The headline table (identical rollouts; only `Psi_t` differs, so any difference is
attributable to the estimator and not to sampling luck):

```
variance ladder                     variance    vs total_return   max|err| vs truth
  total_return                        20.033        1.00x             --
  reward_to_go                         6.317        3.17x           0.0040
  rtg + fitted baseline b(s)           1.765       11.35x           0.0011
  rtg + optimal baseline b*            1.476       13.57x           0.0013
  rtg + oracle V^pi (mismatched)       7.769        2.58x           0.0046   <-- WORSE
  rtg + within-episode mean (bug)      0.784       25.56x           0.3411   <-- BIASED

b_fit = [1.2687 1.1006]   b* = [1.5507 1.4696]   V^pi = [3.4075 3.3649]
training (baseline='fit'): return 7.966 / 8, pi = [[0.023 0.977],[0.994 0.006]]
                                                = Day-1 pi* ("reach & stay")
```

`reward_to_go` reproducing Day 2's 3.15x as **3.17x** is a free regression test that the
rewrite did not perturb the estimator.

### The three results that matter

1. **Baselines stack with reward-to-go: 3.58x on top, 11.35x cumulative.** They attack
   different noise -- reward-to-go deletes provably zero-mean cross terms, the baseline
   centres what remains.
2. **Unbiasedness and variance reduction are INDEPENDENT properties.** The oracle `V^pi`
   baseline is perfectly unbiased (`max|err| 0.0046`, same as everything else) and yet
   **increases** variance 27%. Cause: `V^pi ~ 1/(1-gamma) = 3.4` is the infinite-horizon
   value, while `Psi_t` uses the T-step reward-to-go pooled over `t`, which averages 1.27
   because late `t` has almost no reward left. Subtracting 3.4 *decentres* `Psi_t`. A
   baseline can be arbitrarily wrong and still unbiased -- wrongness is paid for in
   variance, and the payment can be negative.
3. **Low variance alone is worthless.** The buggy within-episode baseline has by far the
   *lowest* variance (25.56x, more than double the best unbiased estimator) while its mean
   is 38% too small (0.561 vs 0.902). `Psi_t = 0` has zero variance. The only meaningful
   claim is **variance reduction at fixed mean**, which is why every variance number in
   this file is paired with a ground-truth unbiasedness assertion.

### Entropy-bonus ablation

```
  beta   return  entropy   pi(.|s0)        (uniform H = 0.6931)
  0.00    7.966   0.0738   [0.023 0.977]
  0.50    7.925   0.1182   [0.046 0.954]
  2.00    7.634   0.2941   [0.192 0.808]
  5.00    6.375   0.5216   [0.298 0.702]
```

Monotone in both directions -- the exploration price, quantified. `beta = 0.05` moved
entropy by 0.0024 (nothing): the entropy gradient `dH/dz_k = -p_k(log p_k + H)` **shrinks
as the policy sharpens**, so the bonus is weakest exactly where it is most needed. It only
bites at `beta ~ O(1)`, the scale of the policy gradient itself. Corollary: an entropy
bonus is a collapse **preventative, not a cure** -- set it before collapse, because after
`pi` has sharpened the restoring force has decayed along with the probabilities.

## Core Takeaway

`grad log pi(a_t|s_t)` is the **direction** ("make this action more likely here") and
`Psi_t` is the signed scalar **credit** for that decision. The division of labour is
forced: `Psi_t` MUST depend on `a_t` (that is where all the signal is -- a state-only
`Psi` gives an identically zero gradient by EGLP), and the baseline `b` must NOT (that is
what keeps the mean intact). `A^pi = Q^pi - V^pi` states both halves at once: `Q^pi`
carries the action dependence, `V^pi` is the largest `a`-independent thing you can strip
off it. Everything from REINFORCE to PPO to GRPO is a choice of how to estimate that split.

## Key Equations

```
EGLP        E_{a~pi}[grad log pi(a|s)] = sum_a pi * (1/pi) grad pi
                                       = grad sum_a pi = grad 1 = 0

baseline    E[grad log pi(a_t|s_t) * b] = E_{h_t}[ b * E_{a_t~pi}[grad log pi | h_t] ]
                                        = E_{h_t}[ b * 0 ] = 0      for ANY h_t-measurable b

optimal b   b*(s) = E[ ||grad log pi(a|s)||^2 * G ] / E[ ||grad log pi(a|s)||^2 ]
              = the SCORE-MAGNITUDE-WEIGHTED mean return
              = V^pi(s) exactly when ||grad log pi||^2 is uncorrelated with G

advantage   grad J = E[ sum_t grad log pi(a_t|s_t) * A^pi(s_t,a_t) ],  E_{a~pi}[A^pi] = 0

entropy     H(pi(.|s)) = -sum_a pi log pi;   dH/dz_k = -p_k (log p_k + H)
            (rows sum to zero: sum_k -p_k(log p_k + H) = H - H = 0)
```

**Measurability, in one line:** "`X` is `h_t`-measurable" means *if I tell you `h_t`, you
can tell me `X`" -- `X` is a deterministic function of what is already known. That single
notion settles every question today:

| quantity | `h_t`-measurable? | consequence |
|---|---|---|
| `r_t'` for `t' < t` | yes (in `h_t`) | pulls out of `E_{a_t}` -> reward-to-go is valid |
| `r_t'` for `t' >= t` | **no** (depends on `a_t`) | cannot pull out -> the `t' >= t` terms survive; this asymmetry IS causality |
| `b(s_t)`, `V^pi(s_t)`, `V_phi(s_t)` | yes | unbiased baseline, regardless of accuracy |
| `G_t - mean_t(G_t)` (same episode) | **no** | BIASED (measured: mean 38% too small) |
| group mean over K completions | **no** (contains `R_k`) | biased only by a scale factor -- see below |

The discriminator is measurability, **not** theta-dependence: past rewards depend on theta
too, so theta-dependence cannot distinguish the two cases. (Slipped on this twice before.)

### Why an action-dependent baseline is catastrophic, not merely biased

Take `Psi_t = G_t` and the "tight" baseline `b = Q^pi(s_t,a_t)`. Per-sample the weight
`G_t - Q^pi` is nonzero, two-sided, and looks beautifully centred. But condition on
`(s_t,a_t)`: now `grad log pi(a_t|s_t)` is the CONSTANT (a function of `s_t,a_t`) and `G_t`
is the random thing, so it pulls out and

```
E[G_t - Q^pi(s_t,a_t) | s_t,a_t] = Q^pi - Q^pi = 0        (definition of Q^pi)
```

The estimator's mean is the **zero matrix**; the bias is `-grad J` exactly. Nothing about
the samples looks wrong -- only a ground-truth gradient comparison finds it. Generally, for
any theta-independent `b(s,a)`, `E[grad log pi * b(s,a)] = grad E_{a~pi}[b(s,a)]`, i.e. you
subtract the gradient of "make `b` large", which for `b = Q^pi` is the whole objective.
This is why action-dependent baselines (Q-Prop etc.) must add a correction term back.

### GRPO: the group mean is unbiased in direction

`b = (1/K) sum_j R_j` contains `R_k`, so it is not `h_t`-measurable for completion `k`.
But

```
R_k - (1/K) sum_j R_j = (1 - 1/K) * ( R_k - (1/(K-1)) sum_{j != k} R_j )
```

so the group-mean advantage is exactly `(K-1)/K` times the leave-one-out (RLOO) advantage,
which IS unbiased. The "bias" is a pure scale factor absorbed into the learning rate. The
genuinely biasing part of GRPO is the **std division**: nonlinear in the group, does not
factor out, and detonates on degenerate groups (all-correct / all-wrong -> `sigma ~ 0`).
This is what Dr. GRPO argues for dropping.

Tradeoff in one sentence: **PPO buys per-token low-variance advantages from a learned
critic and pays with the critic's approximation bias plus a second network's memory and
compute; GRPO drops the critic -- no critic bias, no critic memory -- and pays Kx rollout
cost plus the loss of per-token credit assignment, since one sequence-level advantage is
broadcast to every token.**

## Implementation Notes

- **Hold the rollouts fixed across estimators.** `batch_gradient_statistics` takes a
  `weight_fn(batch, i, gamma)` closure, so the estimator under test is a value while the
  DATA is constant. Any variance difference is then attributable to `Psi_t`. Paired
  comparison, same discipline as the W1-D5 ablation day.
- **Fit the baseline on a SEPARATE batch.** That is the concrete mechanism for
  "independent of the episode it weights". Costs 2x rollouts here; a real critic amortizes
  it by carrying `V_phi` across iterations (and PPO's reuse of the same batch is a real,
  small, well-known bias of the same family as GRPO's self-inclusion).
- **Closures over the baseline:** `partial(weights_with_baseline, baseline=b)` or
  `lambda B,i,gm: weights_with_baseline(B,i,gm,b)`. Python closures capture the
  *variable*, not its value -- fine when called in the same iteration, a bug if the
  lambdas are stored and called later (fix: `lambda ..., b=b:` or `partial`).
- **Vectorizing the score magnitude:** one index array -> whole rows (`pi[states]` ->
  `(n,T,A)`); two aligned index arrays -> one element each (`pi[states, actions]` ->
  `(n,T)`, the numpy `gather`). Better still, expand the norm:
  `||e_a - p||^2 = 1 - 2 p_a + ||p||^2`, which avoids materializing an `(n,T,V)` one-hot --
  at V=128k that is the whole reason production code never builds one-hots.
- `grad` of a SCALAR w.r.t. a matrix has the matrix's shape: `policy_entropy(theta).mean()`
  is `()`, so `autograd.grad(..., theta)[0]` is `(S,A)`. Jacobian layout questions only
  arise for non-scalar outputs.
- Entropy: `log_pi = log_softmax(theta, dim=-1)`, `pi = log_pi.exp()` -- one stable pass,
  and `pi`/`log_pi` are exactly consistent. `torch.softmax` has **no default `dim`**.

## Failure Modes / Debugging Notes

- **`theta += numpy_array` is NOT an in-place update.** numpy takes over, returns a new
  object, and Python rebinds the name -- `theta` silently becomes a fresh tensor with
  `requires_grad=False`, `is_leaf=True`. Measured:

  ```
  theta += numpy                  requires_grad=False  same_object=False
  theta += as_tensor(numpy)       requires_grad=True   same_object=True
  ```

  The values are right and the dtype survives; only the property you depend on is
  destroyed. Same family as `np.long` accumulators and `zeros_like(bool_mask)`. Rule for
  this repo: numpy is the storage/statistics layer, torch is the autograd layer, and every
  crossing gets an explicit `torch.as_tensor(..., dtype=theta.dtype)`.
- **`numpy_array + torch_tensor` raises; `torch_tensor + numpy_array` works.**
  `Tensor.__array_priority__ = 1000` only wins when the tensor is on the left. The error
  reads `Concatenation operation is not implemented for NumPy arrays` -- an operation
  nowhere in the code. Debug heuristic: **an error naming an operation you never called
  means a type-coercion fallback; look at operand types, not the named operation.**
- **`no_grad` scopes by code region.** Calling `entropy_gradient(theta)` inside the
  `no_grad` block raises `element 0 of tensors does not require grad` -- there is no graph
  to differentiate. Build graphs outside, keep only the parameter update inside. (The loud
  version of the W1 "no_grad on the branch you optimize -> silent no-learning" bug.)
  Also: `beta * f(x)` still evaluates `f(x)` when `beta == 0.0`; guard with `if beta:`.
- **Log floats, not tensors.** `history.append(policy_entropy(theta).mean())` retains a
  live graph node per iteration. Use `.item()` / `float()`. On GPU `.item()` forces a
  device sync -- fine once per logging interval, a pipeline stall in a hot loop.
- **`np.bincount` needs 1-D input** (`.ravel()` both the indices and the weights) and
  `weights` must be an array, not a scalar. Guard the empty bucket (`count[count==0] = 1`)
  or unvisited states become `nan` and silently poison every weight in that state.
- **A ddof=1 variance over a single sample warns and returns nan** -- the `_fail_fast`
  probe batch needs `n >= 2`.

## Test-Design Notes (harness)

Rebuilt the harness as **staged** so each function is testable the moment it is written:
one stage per TODO, `NotImplementedError` -> `[TODO]` and skip, other exceptions printed
but the run continues, `--fast` / `--list` / named-subset CLI, lazy `Ctx` fixtures each
with their own seeded RNG, and `_fail_fast` probes so a missing TODO surfaces before a
20k-episode rollout. Promoted to a repo convention
(`shared/templates/exercise-harness-template.py`).

Two design principles earned today:

- **Assert that the BUG is detectably a bug.** `within-episode baseline is BIASED (must
  miss the ground truth)` is worth more than any check on the fix, because it proves the
  test has teeth. Threshold written relative to the unbiased error (`> 5x`) rather than
  absolute, so it survives changes in batch size.
- **A harness assertion can be wrong.** `oracle V^pi also stacks` FAILED, and the harness
  was wrong, not the code -- the horizon mismatch is real and became result #2 above. When
  a well-motivated check fails, check the claim before checking the code. (Third time this
  pattern has produced the day's best lesson: W1-D5 noiseless-overfit, W2-D5 toothless
  cached==uncached, now this.)

## Interview Answer

Prompt:

> Prove that subtracting a baseline doesn't bias the policy gradient, and explain why
> you'd use V rather than Q.

Answer:

Take one term of the policy gradient and subtract a baseline: I want to show
`E[grad log pi(a_t|s_t) * b] = 0`. Condition on the history `h_t = (s_0,a_0,...,s_t)` and
use the tower rule. If `b` is `h_t`-measurable -- meaning it's determined by information
available *before* `a_t` is drawn -- then it's a constant inside the inner expectation and
pulls out, leaving `E_{a_t~pi}[grad log pi(a_t|s_t)]`. That's the EGLP lemma:
`sum_a pi * grad log pi = sum_a grad pi = grad 1 = 0`. So the whole term vanishes: the mean
of the gradient is untouched. What *does* change is the variance, and that's the entire
point -- a baseline is a control variate.

The condition is measurability, not "it depends only on state" -- history-dependent
baselines are fine. What is *not* fine is dependence on `a_t`. Concretely, if you use
`b = Q^pi(s_t,a_t)`, then conditioning on `(s_t,a_t)` makes the score the constant and
`E[G_t - Q^pi | s_t,a_t] = 0` by definition of `Q^pi` -- the estimator's mean becomes
exactly zero. You haven't biased the gradient, you've deleted it, and per-sample it still
looks perfectly healthy. That's the sharp reason `b` may not see the action.

As for V versus Q: they answer different questions. `Q^pi` is a fine choice of `Psi_t` --
the credit term *must* depend on the action, since a state-only `Psi` gives an identically
zero gradient by the same lemma. `V^pi` is the right choice of *baseline* -- the largest
action-independent piece you can strip off `Q^pi`. Doing both gives the advantage form,
`A^pi = Q^pi - V^pi`, which is the lowest-variance unbiased choice and has the right
semantics: "was this action better than what I'd normally do here," rather than "was the
outcome good." In an all-positive-reward environment the uncentred version increases the
probability of everything and the only thing that survives is noise.

Strictly, `V^pi` isn't optimal. Minimizing the variance gives
`b* = E[||grad log pi||^2 G] / E[||grad log pi||^2]`, the score-magnitude-weighted mean
return; `V^pi` is the unweighted one, and they coincide when score magnitude is
uncorrelated with return. Nobody uses `b*` because it's a ratio of noisy estimates and
technically wants a separate baseline per parameter, whereas `V^pi` is a reusable object
that also gives you bootstrapping and GAE. On a toy chain I measured `b*` at 1.476 versus
1.765 for a fitted `V` -- real, but small.

The last thing I'd add is a caution. Unbiasedness and variance reduction are independent
properties. I measured an oracle `V^pi` that was perfectly unbiased and yet *increased*
gradient variance 27%, because it was the infinite-horizon value while the advantage used
a finite-horizon return -- so it decentred `Psi_t` instead of centring it. That's exactly
the failure mode of an unconverged or mis-targeted PPO critic: it doesn't bias your
gradient, it makes it noisier than having no critic at all, and nothing crashes. The
diagnostic is the critic's explained variance against realized returns, and the decisive
experiment is to zero the baseline and re-measure gradient variance on the same rollouts:
if variance *falls*, the critic is worse than nothing.

## Reusable Artifact

- Path: `week-03-rl-foundations/code/actor_critic.py`
- How to reuse:
  - `./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/actor_critic.py`
    runs all 39 checks (~31 s); `--fast` for a few seconds; `--list` for the stages;
    `<stage> [<stage> ...]` for a subset.
  - **The `weight_fn` closure pattern is the reusable piece**: any new `Psi_t` (GAE
    tomorrow, a clipped PPO surrogate in Week 4, a GRPO group advantage in Week 6) plugs
    into `batch_gradient_statistics` and gets an unbiasedness check against
    `exact_gradient` plus a variance number on identical rollouts, for free.
  - `fit_state_baseline` is a tabular critic; swapping it for a learned `V_phi` is the
    only change needed to make this a real actor-critic.
  - The staged-harness runner at the bottom is copy-paste reusable and is now the repo
    convention for every coding exercise.

## Open / Next

- **Day 4 (GAE)** slots directly onto this: `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`
  and `GAE(gamma,lambda)` are two more `weight_fn`s, and the ground-truth gradient will
  quantify the bias that the mismatched-`V^pi` result already previewed.
- Worth building when convenient: a **time-indexed** finite-horizon baseline `V_t(s)`
  (`T-t` Bellman backups, a `(T,S)` table). It should beat both the pooled `b(s)` and
  `b*`, and it makes the "the critic must match the return definition you're actually
  using" lesson constructive rather than cautionary. In LLM RL the critic gets this for
  free, since the transformer state encodes position.
