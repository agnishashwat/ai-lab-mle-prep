# Deliverable - Week 06 Day 03: Sampling, pass@k & Rejection Sampling

## What I Built / Derived

`week-06-grpo-rlvr-reasoning/code/sampling.py` — **73 staged checks**, ten stages, clean under
`-W error::RuntimeWarning`. Ten stages: `estimator`, `identity`, `curvature`, `temperature`,
`correlation`, `selectors`, `filterranker`, `bonkl`, `train`, `diagnose`.

Four TODOs: `pass_at_k` and two-thirds of `selector_curves` mine; `maj@k`, `bon_distribution`
and `diagnose_sampling` written by the tutor at my request at the end of a long session.

## Core Takeaway

**pass@k is not a capability. It is a capability *composed with a selector you do not have*.**

Every number on this page follows from that sentence. pass@k assumes an oracle picks the right
sample out of k. Ship anything and you replace the oracle with something real — a verifier, a
reward model, a vote — and the curve you measured is not the curve you get. The three selectors
have three different limits, and only one of them is pass@k:

| selector | what it is | limit as k → ∞ |
|---|---|---|
| oracle | pass@k | **1** (if the answer is in the support) |
| BoN / filter | accept-set membership | **the verifier's precision** |
| majority vote | mass | **1 if the modal answer is right, else 0** |

The vote can go to **zero** while pass@k goes to **one**, non-monotonically, on the same policy.
That is the sharpest available statement of **mass versus support**, and it is why "pass@64 =
85%" and "we can ship 85%" are different claims.

## Key Equations

```
pass@k              = 1 − (1−p)^k
unbiased estimator  = 1 − C(n−c, k)/C(n, k) = 1 − Π_{i=0}^{k−1} (n−c−i)/(n−i)
the dead-group law  = P(dead) = [1 − pass@G] + p^G
the reweighting     = ∂pass@k/∂p = k(1−p)^(k−1)
best-of-n as KL     = KL(BoN_n ‖ π) = log n − (n−1)/n          (continuous score)
BoN with ties       = P(max at level s) = (F_s + m_s)^n − F_s^n
filter composition  = γ(1−p) / [γ(1−p) + (1−α)p]               — conditional on acceptance
filter hack prob.   = [1−(1−p_obs)^n] · filter composition      — first-sample fallback
ranker hack prob.    = 1 − [1−(1−p)γ]^n                        — pass@n for hack mass (1−p)γ
```

## Implementation Notes

- **`c` is an array, `n` and `k` are scalars.** `c` is per-**problem**; the budget is yours and
  fixed. The benchmark is `pass_at_k(n, c, k).mean()` with the mean **outside**. On the same
  data, mean-then-estimate gives **0.9960** and estimate-then-mean gives **0.6250**.
- **Write the product, never the ratio.** `C(200,100) = 9.05e58`. The product never forms a
  number above 1, and both edge cases fall out with no branch: `c = 0` → every factor is 1;
  `k > n−c` → a factor is exactly 0. No clip is needed, because `n−c ∈ range(k)` whenever
  `k > n−c`, so the zero is hit *before* any negative factor.
- **`c.any() < 0` is always False.** It reduces the array to one bool, then compares that bool
  to 0. Comparison first, reduction second: `np.any(c < 0)`.
- **The plug-in `1 − (1−c/n)^k` is `f(p̂)` with `f` concave**, so Jensen makes it UNDER-report.
- **`maj@k` is a pushforward** onto `answer_id`. Everything else in the file lives on sequences.
  Enumerate compositions in **log space** — the multinomial underflows long before the sum does.
- **Ties are the whole of `bon_distribution`.** Group by distinct score level; everything
  sharing a score is one level.

## The Numbers

**The estimator.** Unbiased to **7.772e-16** against an exact Binomial sum over 9 points. At
n=10, p=0.1, k=10 the plug-in returns **0.493747** against a truth of **0.651322** — a bias of
**−0.157575 = 24.19% of the value, thrown away**. At k = n the unbiased estimator collapses to
exactly `1[c ≥ 1]`.

**The identity.** `P(dead) = [1 − pass@G] + p^G` holds to **5.551e-17** over a 501-point grid.
At G=8 the dead fraction is **66.3420% at both p=0.05 and p=0.95** — failure-side share
**100.00%** and **0.00%** respectively. `alive(0.95) = pass@8(0.05) = 0.3365796`: the same term,
read twice. *dead_frac cannot tell you WHICH unanimity you have, and the two need opposite fixes.*

**The reweighting.** `∂pass@k/∂p = k(1−p)^(k−1)`. Hard/easy marginal-value ratio is exactly
`4.75^(k−1)`: **1× at k=1, 5.456e+04× at k=8, 1.414e+10× at k=16**. At k=1 the metric weights
every problem **exactly equally** (1/6 each); at k=256 the two hardest carry **100.0000%** and
the easiest carries **0.000e+00**. *Changing k changes which benchmark you are running.*

**Temperature.** Heterogeneous set (8 easy + 8 hard): T* = **0.20/0.20/0.20/3.00/3.00/3.00** at
k=1/2/4/8/16/64 — the transition sits **between k=4 and k=8**. Homogeneous set (16 easy): T* =
**0.20 at every k**. *That is the SCOPE CONDITION — T\* moves with k only if p is heterogeneous
across problems.* For a single problem the whole **ordering** of temperatures is k-invariant,
because `1 − (1−p)^k` is strictly increasing in p and monotone maps preserve argmax.

**Temperature vs truncation.** Temperature preserves the support at **every finite T > 0**;
top-p 0.90 leaves **11 of 15** sequences alive. *Only one of the two knobs can lower
pass@infinity, and it is truncation.*

**Correlation.** The i.i.d. formula over-states pass@k at every k > 1 — and the gap is
**UNIMODAL, not monotone**: `+0.0706 / +0.2313 / +0.4176 / +0.4749 / +0.1921 / +0.0040 / +0.0000`
at k=2/4/8/16/64/256/1024. Peak **0.4749 at k=16**, back to zero by k=1024, because both curves
saturate. *The gap is largest exactly in the range people report.*

**Selectors.** An exact verifier (`V=C`) makes BoN@k **equal** pass@k at every k; soundness alone
does not suffice when the verifier is incomplete. An unsound verifier makes BoN converge to
**precision 0.321655** while pass@k → **1.000000**. maj@k on a fixture whose modal
answer is wrong: **0.3500 / 0.3517 / 0.3563 / 0.3508 / 0.3231 / 0.2764** at k=1/3/9/21/51/101 —
rises, peaks near k≈9, then falls, while pass@101 = **1.000000**. Control: when the mode is
correct, maj@k rises **0.4500 → 0.8701**.

**Filter vs ranker — the day's cleanest result.** Same verifier (γ=0.01, α=0.10), same p=0.10,
same budget:

```text
  n        FILTER P(hack output)        RANKER P(hack output)
  1              0.9000%                       0.9000%
  8              5.1426%                       6.9772%
  64             9.0794%                      43.9322%
  256            9.0909%                      90.1178%
```

The filter's **accepted-output composition** is n-invariant (closed form 9.0909%). Its
unconditional hack probability includes the chance of finding any accepted sample, so it equals
the ranker at n=1 and approaches 9.0909%. Ranker is `1 − [1−(1−p)γ]^n` — **pass@n for policy
hack mass `(1−p)γ`**. At n=256: **9.09% vs 90.12%, a 9.91× difference from the selection rule
alone.**

**Best-of-n as a KL knob.** `KL(BoN_n ‖ π) = log n − (n−1)/n` = **0.1931 / 1.2044 / 3.1745 /
4.5491** nats at n=2/8/64/256. `√KL` at n=64 is **1.7817**. Each doubling adds ~log 2 —
**measured 0.6853 nats** from n=64→128. With a **discrete** score the KL is strictly **less**
(**2.0507 < 3.1745** at n=64), because ties make best-of-n less selective; convergence to the
continuous law verified as levels → ∞ (error **4.27e-07** at L=20000).

**Training.** pass@1 **0.4447 → 0.9138**, and the gain **decays in k and is exactly zero at
k_max**: `+0.4691 / +0.3557 / +0.1945 / +0.0779 / +0.0216 / +0.0002 / +0.0000 / +0.0000` at
k=1/2/4/8/16/64/256/1024. *1024× the samples buys 0e+00 of the improvement.*

## Failure Modes / Debugging Notes

**THE RESULT OF THE DAY — a rule that was measured and refused to fire.**

`coverage_collapse` — pass@1 rising while pass@k_max falls below its step-0 value — is the
standard claim about RL on LLMs. It is well-posed, discriminative in principle, and would have
passed code review. Measured over **24 configurations** (2 task flavours × 4 init scales × 3
(lr, β) settings) and then 8 seeds on each of 3 fixtures:

```text
  min over all runs of [ pass@k_max(final) − pass@k_max(0) ]  =  +0.000e+00
```

**Never negative, anywhere.** The mechanism is GRPO's own advantage: on a prompt whose gold is
never sampled, every rollout in the group scores the same, so the group-relative advantage is
identically zero and so is the gradient. *The algorithm cannot push mass off coverage it is not
sampling* — D2's dead group, read as a **safety property** instead of as waste.

**SCOPE:** 15-sequence tabular softmax, sound verifier, group baseline. This licenses nothing at
10⁹ parameters. It says the collapse needs a **mechanism**, and that "RL reduces pass@k" is a
claim to be measured on the system in front of you.

Three days, three cut rules, three different reasons:

| | what was cut | why |
|---|---|---|
| **D1** | `sig_starved` | the **instrument** couldn't support it (t_max = 3 left no room) |
| **D2** | a share-based rule | the **semantics** didn't support it (share → 0 is health) |
| **D3** | `coverage_collapse` | the **system does not exhibit it** (0 of 24, with a mechanism) |

**A number that refused to be a number.** The crossover k — where the training gain drops under
the seed noise floor — came out **[16, 64, 16, 64, 64]** over five seeds. A 4× range. Seed 0
alone gives "the crossover is k = 16," exactly the kind of number that survives into a slide.
**Report the bracket, not the point estimate.**

**Rules that survived, and their 8-seed margins:**

| rule | statistic | broken | healthy | threshold |
|---|---|---|---|---|
| `no_coverage` | final pass@k_max | ≤ 0.3845 | ≥ 1.0000 | 0.70 |
| `mode_locked` | final entropy | ≤ 0.6597 | ≥ 0.9739 | 0.80 |
| `mass_without_support` | Δ pass@1 | ≥ 0.3697 | ≤ 0.1130 | 0.25 |

`mode_locked` **lost a conjunct to the data**: "and the spread is small" fails, because locked
spread ran 0.123–0.379 and healthy spread 0.323–0.556 — they overlap and cannot be thresholded.
Entropy alone separates. Fewer conditions, because that is what the measurement supports.

**Six harness defects caught by the private reference run** (most of any day): a vacuous
stability check (`1 − 1.1e-59` rounds to exactly 1.0, so `0 < v < 1` tested nothing); a
threshold with the wrong sign; an argmax destroyed by saturation at k=1024; an operator
precedence bug (`np.abs(a - b if False else c)` parses as `(a-b) if False else c`); two
byte-identical fixtures (`max(2, 15//2) == 15//2`); and the rejection above.

## Interview Answer

Prompt:

> Your model gets **85% pass@64** and **30% pass@1**. What do you actually know?

Answer:

You know the answer is **in the support** for 85% of problems. You know almost nothing about
what you can ship, because pass@64 is not a capability — it is a capability composed with an
**oracle selector** you do not have in production.

Replace the oracle with something real and the number moves to a different limit. A verifier
converges to its **precision**, not to 1 — I measured an unsound one saturating at **0.3216**
while pass@k went to **1.0000** on the same policy. A majority vote converges to **1 if the
modal answer is correct and 0 otherwise** — I measured maj@k *falling* from 0.3563 at k=9 to
0.2764 at k=101 while pass@101 was 1.000000. The vote gets worse with more compute.

So the first question is which selector you intend to ship, and the second is whether it is
**sound**. The gap between 30 and 85 is only bankable to the extent you have something that can
find the right sample — and *how* you spend the budget matters more than its size. Same
verifier, same n=256: used as a **filter**, 9.09% of what you emit is a false accept, and that
accepted-output composition is *independent of n*; under fallback its unconditional risk approaches
that value as abstention vanishes. Used as a **ranker**, the probability of at least one top-scored
hack is 90.12%, because a ranker is running pass@n on policy hack mass `(1−p)γ`.
**9.91× worse from the selection rule alone.**

One caveat on the 85% itself: if it was computed as `1 − (1 − c/n)^k` it is biased **low** by
Jensen — at n=10, k=10 that estimator returns 0.4937 against a truth of 0.6513, a **24%**
understatement. Use the unbiased `1 − C(n−c,k)/C(n,k)`, and average over problems, not before.
