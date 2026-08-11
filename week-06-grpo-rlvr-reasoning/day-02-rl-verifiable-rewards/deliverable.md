# Deliverable — Week 06 Day 02: RL with Verifiable Rewards

**Artifact:** `week-06-grpo-rlvr-reasoning/code/rlvr.py` — ten stages, **112 checks**,
0 failures. Full run 5.9 s, `--fast` 4.9 s (`--fast` shrinks Monte-Carlo counts only,
never step counts, so all four training fixtures are identical in both modes).

## Core Takeaway

> A verifier is a **SET**, not a rate. Everything that matters about it is a property of
> `V` — the set of responses it accepts — and its two containments with `C`, the set of
> genuinely correct ones. The scalar people quote ("our verifier is 97% accurate") is a
> property of `V` **and the policy it was measured under**, and RL exists to change that
> policy. Measured here: the same checker, on the same two arrays, has precision
> **0.672** at initialisation and **0.093** once the policy has tilted toward the
> responses it wrongly accepts.

The two containments are not two sides of one dial. They fail differently, cost
differently, and are fixed by opposite interventions.

```text
V ⊆ C    SOUND      V \ C = ∅    no false positives
C ⊆ V    COMPLETE   C \ V = ∅    no false negatives
```

| # | Consequence | Mechanism | Stage |
|---|---|---|---|
| C1 | a verifier's confusion is **policy-dependent** | rates are mass-weighted under π | `rates` |
| C2 | precision collapses as the task gets **harder**, at fixed FP rate | selection: rewards are a filtered sample | `selection` |
| C3 | dead groups are a **Rényi entropy** of the reward's output distribution | P(dead) = Σ q^G | `renyi` |
| C4 | false negatives are a **compute** bug, priced in rollouts | p_obs falls ⇒ G must rise | `fntax` |
| C5 | composite rewards **self-anneal**: influence ∝ c(1−c), not λ | centering is linear | `components` |
| C6 | soundness ⇒ zero gap, **every policy, every KL**; unsoundness ⇒ sup gap = 1 | a support property | `soundness` |

## The result the day was built to reach

**C6, and its consequence for instrumentation.** If `V ⊆ C` then `v_pass ≤ gold_pass`
for every policy at every KL — no scope condition, no tuning. That is the strongest
guarantee anything in post-training offers. And if `V \ C ≠ ∅`, the supremum of the gap
is **1.0** — attained by the point mass on a single false positive:

```text
  ONE false positive     sup(v_pass - gold_pass) = 1.000000
  SEVEN false positives  sup(v_pass - gold_pass) = 1.000000
  their eval accuracies under uniform:  93.3%  vs  53.3%
```

Reward hacking is a property of the **support** of `V \ C`, not its measure. The
statistic that ranks the two checkers as *evals* is silent on the thing that decides
what happens under *optimization*.

And the reason this cannot be caught from the reward panel — four real training runs,
differing only in which sequences the checker accepts:

```text
                    dead_frac    v_pass    gold_pass
  honest              0.0361     0.4400      0.4400     sound + complete: ONE column
  leaky (hacked)      0.7694     0.9681      0.0251
  starved             0.7833     0.0319      0.1383     gold ABOVE v: incompleteness
  saturated (solved)  0.8361     0.9803      0.9803
```

**The hacked run and the solved run agree on every column the reward can produce** —
`dead_frac` 0.769 vs 0.836, `v_pass` 0.968 vs 0.980 — and disagree by **0.025 vs 0.980**
on the held-out one. A group of size G with one accepted response gives that response
`Â = (G−1)/G = 0.875000` whether the acceptance was a hack or a genuine success: the
advantage vectors are **bit-identical**. No rule reading `v_pass`, `dead_frac`,
`adv_absmean`, `entropy` or `loss` can separate them. The gold column is not a
nice-to-have; it is the only column carrying the distinction.

## C1 — "97% accurate" is not a well-formed claim about a reward

Every rate is over **probability mass** under π, not over sequences. Counting sequences
gives a policy-independent number that looks like a property of the verifier and is not:

```text
  gamma, mass-weighted   0.1515        gamma, count-weighted   0.2222
```

Then hold the checker fixed — same `gold` array, same `accept` array — and tilt the
policy toward the responses it wrongly accepts, which is precisely what optimization
does:

```text
  gamma       0.1515  ->  0.7820
  precision   0.6720  ->  0.0926
```

## C2 — the selection law: precision is a function of difficulty

Training never sees a uniform sample of the checker's behaviour. It sees the sample the
checker **rewarded**, which is a biased draw:

    p_obs = p(1−α) + (1−p)γ
    P(hack | rewarded) = γ(1−p) / [ p(1−α) + γ(1−p) ]

At a **fixed 1% false-positive rate** with α = 0.10:

```text
  p = 0.500  ->   1.0989%
  p = 0.100  ->   9.0909%
  p = 0.020  ->  35.2518%
  p = 0.005  ->  68.8581%          -> 1 as p -> 0
```

Two subtleties, both of which are traps:

1. **Conditioning on the group being ALIVE does not move the per-sample rate**
   (`0.34902` vs `0.34902`). What moves is **concentration**.
2. `P(the whole alive group's positive signal is a hack)` = **0.33020** at p = 0.02
   against **5.9568e-04** at p = 0.5 — a **554×** swing with a 1% verifier in both runs.
   *(Recorded because it was mis-stated as 2.4e7× during the session: that figure is the
   conditional-on-k comparison, h¹ vs h⁴, not the marginal.)*

The pathology is **difficulty-induced, not verifier-induced.** Your verifier gets worse
as your model reaches harder problems, with no change to the verifier.

## C3 — the dead-group law generalises to a Rényi entropy

D1's `p^G + (1−p)^G` is the binary case of:

    P(dead) = Σ_j q_j^G = exp( −(G−1) · H_G(q) )

where `q` is the distribution of the reward's **output levels** and `H_G` is the Rényi
entropy of order G. So dead groups are not a property of the task — they are a property
of *how many distinguishable values your reward emits*, and reward design and group size
are the same knob priced differently:

```text
  binary, p = 0.05, G = 8                        66.3420% dead
  split the FAILURE mass -> (0.05, 0.40, 0.55)    0.9029% dead    73.5x reduction
  buying the same thing with G alone             G = 91.77 -> 92  11.5x rollout bill
```

**The rule is SPLIT THE MODE**, and it is strictly asymmetric:

```text
  at p = 0.05   splitting FAILURE mass:  66.3420% -> 0.9029%    (73.5x)
                splitting SUCCESS mass:  66.3420% -> 66.3420%   (nothing)
  at p = 0.95   exactly reversed
```

## C4 — false negatives are a compute bug, and the price is exact

```text
  30% answer-extraction failure:  p_obs  0.10 -> 0.0700
  G for 90% of groups alive:             22   -> 32          = +45.5% rollout compute
  a 5% FALSE-POSITIVE rate instead:      32   -> 19
```

A regex bug, priced in GPU-hours. And note the direction of the third line: **unsoundness
buys compute**, which is exactly why the cheap fix is the dangerous one. The two failures
move the same number in opposite directions, so no single scalar can express verifier
quality.

## C5 — a composite reward is a self-annealing curriculum

Centering is linear, so `A_total = Σ_j A_j` **exactly at finite G** (max dev 2.22e-16),
and each component's share of the gradient is well defined. For a binary component with
compliance c:

    E[ Σ_i |Â_i| ] = 2(G−1) · c(1−c)      per group

```text
  c = 0.10   emp 1.26493   closed 1.26000
  c = 0.50   emp 3.50591   closed 3.50000
  c = 0.90   emp 1.27215   closed 1.26000
  decay c = 0.50 -> 0.90 -> 0.99:  3.49869 / 1.24724 / 0.13178   = 26.55x
```

**Influence tracks inconsistency, not the coefficient.** At c ∈ {0, 1} the share is
*exactly* zero — at λ = 1, at λ = 1000, at any λ. A component the model has mastered has
left the gradient and **cannot be re-weighted back into relevance.** The same component at
c = 0.5 carries 0.5722 of the gradient, so the zero is a statement about the **model**,
not about the reward.

Corollary (`lexicographic`): a weighted sum encodes a priority only while
λ·max(partial) < min(correctness gap). At λ = 1.01 the argmax of `correct + λ·formatted`
is a **wrong but well-formatted** sequence. And `|A_norm|/|A_cent| = 2.850`, so a λ tuned
under `centered` is not tuned under `normalized`.

## C6 — the RM contrast: underdetermination vs extrapolation

A preference dataset is a **matching** on responses (degree ≤ 1), so the BT loss is flat
along every uncompared direction — the gradient on an uncompared response is **exactly
0.0**, and moving its reward by 1, −37.5, or 1e6 changes the loss by exactly 0. Two inits,
same data, converge to different argmaxes with different gold values:

```text
  init -50  ->  argmax seq 0,  gold 1.0
  init +50  ->  argmax seq 14, gold 0.0
```

**An RM's accept set is decided by its initialisation. A verifier's accept set is the
same object under both inits** — it is data, not a fit, so its worst case is computable
*before* training starts. That is the entire structural difference, and it is why the two
failure taxonomies (sound/complete vs underdetermined/extrapolating) must not be
cross-wired.

## Implementation Notes

- **Six unguarded denominators in one session**, every one producing a **NaN rather than a
  crash**. `component_shares` returned `[nan]` on the `_fail_fast` probe and the stage was
  still green; `increase_percent` hit `mean of empty slice` on an empty history. NaN never
  compares True, so a threshold built on one **silently never fires** — the worst possible
  behaviour for a diagnostic. Run with `-W error::RuntimeWarning`; treat "passes with a
  warning" as a failure.
- **`_tail` reads a LEVEL. A CHANGE needs two points far apart.** Measuring the change
  *inside* the tail cut the leaky run's signal from **+0.2981 to +0.00562** — a 53× loss of
  signal-to-noise against a threshold of exactly zero.
- **Enforce exclusivity with the rules, not with control flow.** An early `return` in the
  first branch makes disjointness an artifact of statement order. Verified by deleting it:
  the rules stayed correct, so the `gold` conjunct was genuinely load-bearing.
- Absolute value is **not** linear: summing components before centering destroys the
  attribution (`Σ|A_total| = 60097` vs `Σ_j Σ|A_j| = 94436`; the 34,339 gap is
  inter-component cancellation).
- `hack_share`'s closed form loses ~half its significant digits as a → 0 or 1
  (`4.999999965e-01` vs exact `4.999999913e-01` at a = 1e-8) — direct powers of near-1
  terms. Fine here, wrong tool at p = 1e-6.

## Failure Modes / Debugging Notes

- **The four-fixture calibration was done across 8 seeds before handover, and one fixture
  had to be re-tuned:** `starved` at lr 0.4 escaped starvation (dead 0.26–0.69 across
  seeds, mid-transition into saturation). At lr 0.1 it is stable (dead 0.650–0.844,
  v_pass 0.022–0.050). Thresholds must come from the gap between *distributions*, never
  between one healthy run and one broken value.
- **A "saturated" rule keyed on `dead_frac` and `v_pass` alone fires on the hacked run.**
  This was a real defect in the reference implementation, caught only by the 8-seed sweep.
  It is the day's thesis as an executable constraint.
- The rule to **refuse**: *"a component's gradient share is decaying to zero."* By C5 that
  is `2(G−1)c(1−c) → 0`, i.e. the model **mastered** the component — the health signature.
  Firing on it means alerting on success, and the fix it invites (raise λ) provably does
  nothing.

## Interview Answer

Prompt:

> Your math verifier is 97% accurate. Is that good enough to train against?

Answer:

97% of what, measured under which policy? Accuracy is a property of the verifier *and*
the distribution it was measured on, and RL exists to change that distribution — same
checker, precision 0.672 at init and 0.093 once the policy tilts toward its false
positives. It is also difficulty-dependent: at a fixed 1% FP rate, P(hack | rewarded) runs
1.10% → 68.86% as the true pass rate goes 0.50 → 0.005.

So I don't ask for one number, I ask two questions with different shapes. About false
positives: **is the set empty?** — a yes/no question, because any non-empty `V \ C` has
supremum gap 1.0, one false positive or seven. About false negatives: **what is the
rate?** — because that is exactly the price, in rollouts: 22 → 32 groups, +45% compute.

Then I train against a sound-but-incomplete verifier, pay for it in rollouts, and hold out
a gold set the reward never sees — because a hacked run and a solved run are identical on
every column the reward can produce (0.769/0.968 vs 0.836/0.980) and differ only on the
held-out one (0.025 vs 0.980).

## Reusable Artifact

- Path: `week-06-grpo-rlvr-reasoning/code/rlvr.py`
- How to reuse:
  - `python rlvr.py --list` — the ten stages
  - `python rlvr.py <stage> [<stage>...]` — run a subset
  - `python -W error::RuntimeWarning rlvr.py` — the way it should always be run
  - `Checker`, `make_checker(flavour=honest|leaky|starved|saturated)` and `train_rlvr`
    are a self-contained RLVR loop with a **held-out `gold_pass` column** that the reward
    never touches — the minimum instrumentation for detecting reward hacking.
  - `dead_fraction_levels` / `renyi_entropy` / `g_for_alive` / `p_observed` are the
    reward-design calculators: given a reward's output distribution, what G do you need.
