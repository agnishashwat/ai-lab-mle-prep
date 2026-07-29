# Deliverable — Week 04 Day 01: The InstructGPT Pipeline & the Reward Model

## What I Built / Derived

**`week-04-rlhf-ppo/code/toy_rlhf.py` — 73 staged checks, 0 failed, 13.3 s (`--fast` 3.4 s).**
This file is the week's infrastructure: Days 2-5 import `TokenEnv`, `enumerate_sequences`,
`sequence_logprob`, `exact_J` and `exact_sequence_kl` from it.

**The design decision that makes Week 4 work.** Week 3's results were assertions rather than
claims because a finite MDP gave exact ground truth (T Bellman backups + central finite
differences). Week 4 keeps the discipline and changes the setting to one where *tokens, EOS,
variable length and sequence-level reward are literal*: a token-level LM environment small
enough to **enumerate every terminating sequence**. Vocab 4 (including a real EOS), `t_max=4`
⇒ 40 states, **121 terminating sequences** = 40 eos-terminated + 81 truncated. Enumeration
*is* this week's `exact_J`.

Built today:

| Function | What it gives the week |
|---|---|
| `enumerate_sequences` | the partition of the sample space; sum-to-one is the invariant |
| `sequence_logprob` | token-level `log_softmax`→gather over a tree |
| `exact_J` | **ground truth** for every estimator Days 2-5 introduce |
| `exact_sequence_kl` | the quantity β multiplies in InstructGPT Eq. (2) |
| `tilted_sequence_dist` | W3 D5's closed form lifted to whole completions |
| `bt_loss_and_grad` | Bradley-Terry NLL + analytic gradient |
| `fit_reward_model` | plain GD; deliberately does no centering |
| `sample_preferences` | stage-2 collection, with annotator noise as a parameter |

Also closed the carried-forward W3 chore: `kl_rl.py`'s `tilted_policy` guarded the *result*
of `np.log(pi_ref)` instead of its *argument*. Fixed to guard inside the log; re-ran with
`-W error::RuntimeWarning` so the warning's absence is enforced, not assumed. **94/94 green.**

## Core Takeaway

A Bradley-Terry reward model is identified **only up to a per-prompt additive constant**, and
that single fact explains three things that are usually taught separately: why raw RM scores
cannot be compared across prompts, why GRPO's per-prompt group mean is free, and why the
KL-regularised optimum is invariant to `r → r + c`. It is one degree of freedom seen from
three sides.

The day's second result reframes the KL penalty. The reward model is fit on completions drawn
from `pi_SFT`, and the penalty measures distance from `pi_ref = pi_SFT`. Those are the same
distribution, so **the KL penalty is a direct measurement, in nats, of how far the policy has
strayed from the reward model's own training data**. It is not a regulariser for its own sake;
it is a *confidence region for the reward model*. Measured: RM squared error grows monotonically
along a KL ladder, from **0.177 on-distribution to 4.89 at sequence KL 4.44 — 27.6×**.

## Key Equations

**Bradley-Terry and its gradient.**
```
P(y_w > y_l | x) = sigmoid( r(x,y_w) - r(x,y_l) )
L(psi)           = -E[ log sigmoid(delta) ] = E[ logaddexp(0, -delta) ],   delta = r_w - r_l
dL/dr_w          = -sigmoid(-delta),        dL/dr_l = +sigmoid(-delta)
```
Gradient magnitude `sigmoid(-delta)` is **monotone decreasing in how right the model already
is**: →0 when confidently correct, exactly 1/2 at a tie, →1 when confidently wrong.

**The invariance.** `L` sees `r` only through *within-prompt differences*, so `r(x,.)` and
`r(x,.) + c(x)` are exactly equivalent. *Hypothesis consumed:* both completions in a pair share
the same prompt `x`. Pair across prompts and the free direction collapses from `c(x)` to a
single global constant — which is exactly why anchor completions and cross-prompt comparisons
are a real fix.

**Validity vs distance.** `E[(r_psi - r_gold)^2]` under `pi_theta` increases with
`KL(pi_theta || pi_ref)`, because KL measures mass placed where `pi_ref` (= the RM's sampling
distribution) had none.

**InstructGPT Eq. (2), term by term.**
```
E[ r_psi(x,y) - beta * log( pi_theta(y|x) / pi_ref(y|x) ) ] + gamma * E_pretrain[ log pi_theta(x) ]
      ^ today            ^ W3 D5, wired on D3                      ^ ptx: the alignment tax
```
(1) is what you want, (2) is where you're allowed to look for it, (3) is what you promise not
to break. Note (3) is an **MLE** term — the only part of the objective that can move support.

## Measured Results

| Quantity | Measured |
|---|---|
| Terminating sequences enumerated | **121** = 40 eos + 81 truncated |
| Sum-to-one over all sequences | `1.000000000000000` (two unrelated thetas) |
| `exact_J` vs 12k-rollout Monte Carlo | z = +1.07; **mean z = −0.01 over 5 seeds** |
| Enumerated sequence KL vs occupancy route | **0.96993705 both ways** |
| — the unweighted per-state mean would read | 0.50890693 *(D5's actual bug)* |
| BT analytic gradient vs central FD | **1.785e-10** |
| RM recovery, raw | max err **8.6088** |
| RM recovery, per-prompt centered | max err **0.2743** — **31.4× better** |
| Per-prompt mean drift from its init | **6.9e-15** — data cannot move it |
| Refit from a different init: raw / centered diff | **1.7355 / 8.22e-15** |
| Pairwise accuracy within-prompt | **0.974** |
| Pairwise accuracy cross-prompt, raw scores | **0.445 — worse than chance** |
| Pairwise accuracy cross-prompt, centered | 0.976 |
| RM error on-distribution → at seq KL 4.44 | **0.177 → 4.89 (27.6×)** |
| Over-estimation, 300 → 20 000 comparisons | **3.231 → 0.038 (86×)** |
| Annotator noise 0 → 0.40, RM accuracy | 0.969 → 0.844 (±0.034, 3 seeds) |

## Implementation Notes

- **`np.add.at` is mandatory for the BT gradient.** Items repeat across pairs; `grad[p,w] -= g`
  gathers the RHS then scatters it back, so duplicate indices keep only the last write and the
  rest vanish. The index must be a **tuple** `(p, w)` — a stacked `(n,2)` array indexes axis 0
  only, which raises on a rectangular array and *silently* returns `(n,2,n_items)` on a square
  one. Square shapes hiding an indexing bug: fourth occurrence.
- **The free direction is a cheap structural test.** A correct BT gradient's rows sum to
  **exactly zero**, because the loss is flat along `c(x)`. The `-=` bug leaves a row sum of
  0.2437. Same argument as W3's "every policy-gradient row must sum to zero" under softmax
  over-parameterisation — different free direction, identical logic. Worth keeping in
  production code, where finite differences are not available.
- **Log space, and never exponentiate an unnormalised log-weight.** `exp(log p_ref + r/beta)`
  still overflows at beta=1e-3 (the exponent is ~6400). Build `z` and hand it to a softmax that
  subtracts the max. Doing this makes the beta→0 and beta→inf special cases *unnecessary*:
  softmax gives 1/K on K-way ties for free, `softmax(log p_ref) = p_ref` exactly, and
  `log 0 = -inf` propagates to an exact `0.0`, which is reverse-KL support confinement falling
  out of the arithmetic.
- **`lr=0.5` looks safe and does not converge.** The BT loss is a *mean* over pairs, so each
  item's gradient is `O(1/n_items)` regardless of data volume. Top Hessian eigenvalue ≈ 0.02
  ⇒ stable up to `lr ≈ 100`. Using 8.0.
- **`np.long` is version- and platform-dependent** (removed in numpy 1.24, reintroduced in 2.0
  as C `long`, which is 32-bit on Windows). Write `np.int64`. It is also the exact spelling of
  the W3 D2 gradient-buffer bug.
- `list + ndarray` adds; `list + list` **concatenates**. Never let `+` straddle the two when one
  side is caller-supplied.

## Failure Modes / Debugging Notes

- **Four harness assertions of mine were wrong before any learner code ran** (fifth session in a
  row that checking the CLAIM beat checking the code):
  1. `|corr| < 0.6` between per-prompt mean RM score and mean gold, over **6 prompts** — a
     correlation at n=6 has enormous sampling variance; it read −0.666. Replaced with a
     deterministic experiment: refit from a different init and assert centered agreement
     (8.2e-15) with raw disagreement (1.74). The theorem as an experiment, not a statistic.
  2. "Over-estimation grows along the KL ladder" — it *falls*, and the statistic degenerates:
     as beta→0 the tilted distribution collapses onto one sequence, so `E[max(err,0)]` becomes
     the error at a single point and reads exactly 0 whenever that point is under-rated. Moved
     to a data-size axis, where it is real and 86×.
  3. "Off-distribution error is >2× worse at every data size" — fitted to one dataset; across
     four seeds the middle sizes go down to 1.6×. Now: direction at every size, magnitude only
     where the fit has converged.
  4. Regret (`gold.max() - gold[argmax(psi)]`) asserted monotone in data — it hangs on a
     discrete argmax over 121 items, so one noisy estimate reaching the top moves it by a whole
     reward gap. Non-monotone across seeds, sometimes 0 at the smallest size. **Reported, not
     asserted.** A statistic that depends on an argmax needs seed-averaging before it can carry
     an assertion.
- **`--fast` must not break convergence.** It was reducing gradient *steps*, which made a true
  assertion false for reasons unrelated to the assertion. `--fast` now shrinks data and loosens
  tolerances only.
- **Correct-by-accident:** `enumerate_sequences` rebinds `frontier` *inside* the loop iterating
  over it. It works only because Python's `for` holds an iterator on the original object and
  rebinding a name doesn't disturb it. Right answer, fragile mechanism — it breaks under any
  refactor to in-place mutation.
- **Untested branches are broken until proven otherwise.** A first pass at `tilted_sequence_dist`
  added `beta == 0` and `beta == inf` special cases: ~7 lines, 4 bugs (`len()` of a scalar
  argmax; zeroing forbidden rewards instead of `-inf`, so a negative-reward world picks a
  forbidden sequence; `list > int`; returning log-probs). The harness calls neither branch. Same
  class as W3 D4's `critic="exact"` TypeError under a 64/64 green run.

## Interview Answer

Prompt:

> "Someone hands you a reward model with 78% pairwise accuracy on a held-out set. What do you
> still not know?"

Answer:

Almost everything that determines whether it is safe to optimise against.

First, **that number was measured on the SFT distribution, and the policy's job is to leave it.**
The RM was fit on completions sampled from `pi_ref`; off that distribution there is no data and
no constraint, so the model is not merely less accurate, it is *underdetermined*. I measured
this on a toy where I have gold: squared error 0.177 on-distribution rising to 4.89 at sequence
KL 4.44 — **27.6× worse**, monotone in KL the whole way. So held-out accuracy is an
on-distribution statistic being used to license off-distribution extrapolation.

Second, **the errors are not symmetric in consequence.** An under-estimate is harmless — the
policy just avoids that completion. An over-estimate is actively *sought*, because to an argmax
an over-estimate is indistinguishable from a genuinely good completion. So the statistic I want
is the upper tail of the error over the reachable region, not the mean. And that quantity is
governed by how much preference data the RM saw: over-estimation fell **86×** going from 300 to
20 000 comparisons in my measurement.

Third, **raw RM scores are not comparable across prompts.** Bradley-Terry sees `r` only through
within-prompt differences, so `r(x,·)` and `r(x,·)+c(x)` fit identically. I verified this by
refitting from a different initialisation on identical data: the within-prompt differences
agreed to 8e-15 while the raw scores differed by 1.7. Consequently cross-prompt pairwise
accuracy using raw scores came out at **0.445 — worse than chance** — while within-prompt was
0.974. If that 78% was computed across prompts, or if anyone is averaging RM score over a mixed
prompt set, the number is partly measuring the prompt mix.

Fourth, **where the errors sit.** The BT gradient magnitude is `sigmoid(-delta)`, so pairs the
model already ranks confidently contribute nothing and the fit is dominated by near-ties and
misrankings. Near-ties are also exactly where annotators disagree, so the decision boundary is
fit disproportionately from coin flips. That means 78% may be *at the noise ceiling* — with
inter-annotator agreement in the 70s, a much higher number would suggest a leak rather than a
better model. So I'd want accuracy broken out by `|delta|` bucket, and I'd want to know whether
the near-ties were filtered or margin-weighted.

Fifth, **calibration is not accuracy.** Ranking correctly says nothing about whether score
*differences* are on a consistent scale, and PPO consumes the magnitude, not just the order.

What I'd measure instead: accuracy as a function of KL from `pi_ref`; agreement with a gold or
human eval at checkpoints along a KL ladder; the RM's score distribution on the *current*
policy's samples versus its training samples; and entropy plus `KL(pi_theta||pi_ref)` logged
from step 0. And I'd never gate a release on RM score — the whole failure mode is that the proxy
reports success the entire way down.

## Reusable Artifact

- Path: `week-04-rlhf-ppo/code/toy_rlhf.py`
- How to reuse:
  - `from toy_rlhf import TokenEnv, enumerate_sequences, sequence_logprob, exact_J,
    exact_sequence_kl, tilted_sequence_dist` — Days 2-5 build on these directly.
  - `exact_J(env, theta, reward_fn)` is the ground truth for asserting any new estimator's
    unbiasedness by enumeration rather than by plotting.
  - `state_occupancy(env, theta)` gives the structurally independent route to the sequence KL;
    use it to cross-check anything that claims to be a KL.
  - `./.venv/bin/python ai-lab-mle-prep/week-04-rlhf-ppo/code/toy_rlhf.py --list` for stages;
    `--fast` for a 3.4 s iteration loop.
