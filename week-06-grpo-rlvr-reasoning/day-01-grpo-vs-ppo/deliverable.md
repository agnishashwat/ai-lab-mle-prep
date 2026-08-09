# Deliverable — Week 06 Day 01: GRPO vs PPO

**Artifact:** `week-06-grpo-rlvr-reasoning/code/grpo.py` — ten stages, **96 checks**,
0 failures. Full run 5.9 s, `--fast` 3.2 s (96 checks; `--fast` shrinks Monte-Carlo
counts only, never step counts).

## Core Takeaway

> GRPO does not delete the value function. It replaces a **learned** estimate of E[r|x]
> with a **Monte-Carlo** one — and unlike a critic, that estimate vanishes exactly where
> the policy succeeds. Every property below follows from the baseline being computed
> **from the same G samples it is subtracting from**.

The critic was a separate function approximator with its own parameters and its own
generalisation. The group mean is a statistic of the batch. Trading one for the other
buys 44% of the memory and costs four things, all measured here.

| # | Consequence | Mechanism | Stage |
|---|---|---|---|
| C1 | gradient shrinks by (G−1)/G | baseline is correlated with the sample it centres | `shrinkage` |
| C2 | ÷std re-weights prompts by **std**, not **variance** | std is nonlinear in the same G draws | `stdbias` |
| C3 | any within-group-constant reward term dies **exactly** | the group mean is a high-pass filter | `annihilate` |
| C4 | groups go **dead** as the policy improves | P(dead) = p^G + (1−p)^G | `deadgroup`, `train` |

## The result the day was built to reach

**C4 is the one with no PPO analogue.** A critic gives a signal on every prompt. The
group mean gives a signal only on prompts where the G completions **disagree**. So the
gradient supply is a function of the pass rate, and it collapses precisely as training
succeeds:

```text
P(a group carries zero signal) = p^G + (1-p)^G          G = 8
  p = 0.50 ->  0.78%      p = 0.90 -> 43.05%
  p = 0.70 ->  5.77%      p = 0.95 -> 66.34%
                          p = 0.99 -> 92.27%
```

Verified against simulation at 5 seeds per point, and visible in a real run — the two
fixtures differ **only** in which sequences the verifier accepts:

```text
                pass_rate      dead_frac
  healthy         0.4963         0.0354
  saturated       0.9742         0.8042      <- 23x more of the batch is silent
```

Three things fall out, and the third is the one worth carrying:

1. **Raising the pass rate 0.90 → 0.95 kills a further 23.3% of the batch.** Improving
   the model degrades its own training signal. This is a *success*-driven failure.
2. **Only 33.658% of prompts survive at p = 0.95, G = 8.** Two thirds of the rollout
   compute produces exactly zero gradient.
3. **The fix is G, and it is expensive.** Keeping 90% of prompts alive at p = 0.95 needs
   **G = 45** (G = 44 gives 89.533%; closed form ln(0.1)/ln(0.95) = 44.891). GRPO's saving
   is *training memory*; its bill is *inference compute*, and the bill grows as the model
   gets better.

## C1 — the shrinkage is exact, and it is not a bias in direction

For Â_i = r_i − mean(r), splitting E[r_j ∇log π(o_i)] on j = i versus j ≠ i and killing
the second with independence + EGLP gives, **exactly and at finite G**:

    E[ ĝ ] = ((G−1)/G) · ∇J

```text
  G = 2   measured 0.4994   (want 0.5000)    cos 0.999992   LOO 0.9987
  G = 4   measured 0.7496   (want 0.7500)    cos 0.999993   LOO 0.9994
  G = 8   measured 0.8755   (want 0.8750)    cos 0.999999   LOO 1.0005
```

Direction is untouched (cos > 0.99999 in every case), so the factor is absorbable into
the learning rate. **At G = 1 the gradient is identically zero, not small** — that is not
an edge case to patch, it is the defining property. GRPO is a *contrast* estimator.

LOO (mean over j ≠ i) removes the shrinkage exactly, and is the same estimator rescaled:

    Â_loo = Â_centered · G/(G−1)     ← reciprocal of the shrinkage; the line IS the proof

Because the factor is a deterministic scalar, the two are equal **pathwise**. Therefore
Var(ĝ_loo) = (G/(G−1))² · Var(ĝ_centered) — strictly *larger* in absolute terms, identical
in relative terms. **LOO cannot reduce variance, and it still dies at G = 1.**

## C2 — ÷std swaps the information weighting for its square root

With k successes of G under a binary verifier, every G cancels:

    Â_success = +sqrt((G−k)/k)          Â_failure = −sqrt(k/(G−k))

Summing |Â| over a group gives that group's total gradient weight:

```text
  k                1        2        4        6        7
  centered     1.7500   3.0000   4.0000   3.0000   1.7500    = 2G·p(1-p)        VARIANCE
  normalized   5.2915   6.9282   8.0000   6.9282   5.2915    = 2G·sqrt(p(1-p))  STD
```

The centered form weights each prompt by its **variance** — which for a Bernoulli *is* the
Fisher information, so that weighting is correct. Dividing by std replaces it with the
**standard deviation**. Amplification is exactly 1/std:

```text
  k = 4  ->  2.0000        k = 2  ->  2.3094        k = 1  ->  3.0237
  the least informative prompt is boosted 1.5119x relative to the balanced one
  and the balanced:extreme weight ratio COLLAPSES  2.2857  ->  1.5119
```

**Control, and it matters:** balanced still outweighs extreme under both (1.75 < 4.00 and
5.29 < 8.00). The bias **compresses** the weighting; it does not invert it. Dr. GRPO's fix
is simply to delete the ÷std.

## C3 — the group mean is a high-pass filter, and this is why KL goes in the loss

Any term constant within a group is removed **exactly at finite G**, not in expectation:

```text
  c(x) =  0      max dev 0.000e+00
  c(x) =  1      max dev 2.220e-16
  c(x) = -37.5   max dev 1.210e-14
  c(x) =  1e+06  max dev 2.039e-10
```

Fold a per-sequence KL penalty into the reward, r_i = r(x,o_i) − β·KL_i, and only its
**within-group deviation** survives:

```text
  survives 0.035726   vs deviation 0.035726     (identical to 6 dp)
  survives 0.035726   vs full penalty 0.6842    = 19.15x DELETED
  a UNIFORM drift penalty is invisible to the gradient:  max dev 2.50e-16
```

KL drift is overwhelmingly common-mode — the whole policy leaves π_ref together. So a β
folded into the reward controls a quantity it **cannot see**: the controller integrates an
error it cannot affect, β saturates, KL runs away, and nothing errors. Hence DeepSeekMath
puts KL **in the loss, outside the advantage**.

### and it uses k3, for feasibility rather than variance

With ℓ = log(π_ref/π_θ): k1 = −ℓ, k3 = e^ℓ − 1 − ℓ. Both unbiased against exact
enumeration to ten decimals (**0.1517183046** vs **0.1517183046**). The discriminator is
the sign:

```text
  k1 is negative on 50.0% of the probability mass
  k3 is negative on  0.0%      (min value +6.130e-02, pointwise)
```

A **negative** KL reading added to a **minimised** loss pays the policy to leave π_ref.
The preference for k3 is not variance — var(k1)/var(k3) *reverses* with KL magnitude,
423× at KL 0.004 and 0.23× at ~0.65 — it is that k1 is not usable as a penalty at all.

## The clip is inert at μ = 1

With one inner epoch and on-policy sampling, ρ ≡ 1 exactly, so min(ρÂ, clip(ρ)Â) = Â.
Measured as the fraction of tokens where the clip **actually bound** (the sign condition,
not merely ρ out of band):

```text
  mu = 1, lr 4.0   ->  max 0.000e+00        EXACTLY zero, every step
  mu = 8, lr 4.0   ->  max 0.2222           same lr, same seed, same task
```

**GRPO at μ = 1 is REINFORCE + a group baseline + a KL term, wearing PPO's notation.**
The negative control is single-variable — only μ changes — so the result is a property of
μ = 1 and not of this task.

## The memory bill (7B, bf16 + Adam; 16 B/param trainable, 2 B/param frozen)

| Configuration | policy | critic | RM | reference | total |
|---|---|---|---|---|---|
| PPO | 112 | 112 | 14 | 14 | **252 GB** |
| GRPO | 112 | — | 14 | 14 | **140 GB** (−44.4%) |
| GRPO + cached ref logprobs | 112 | — | 14 | — | **126 GB** |
| GRPO + RLVR (no RM) | 112 | — | — | — | **112 GB** (−55.6%) |

The saving is real, and C4 says the rollout cost rises to pay for it.

## The diagnostic

`diagnose_grpo` reads only the History panel — never the task, the config, or a gold
reward. Three rules, thresholds chosen from the **gap between distributions** over 8 seeds
per fixture, not from a single observation:

```text
                 healthy max | threshold | broken min       air
  dead_frac            0.185 |    0.50   |   0.783      0.315 / 0.283
  kl                   0.300 |    1.00   |   1.288      0.700 / 0.288
  entropy              0.121 |    0.50   |   0.701      0.379 / 0.201
```

**0 of 32 fixture-seeds disagree with the seed-0 verdict.**

A fourth rule, `length_drift`, was designed and then **cut during calibration**: with
t_max = 3, E|y| can only travel 1.985 → 3.0, and every run that learns lengthens, so
healthy (1.14) and broken (1.35) have no gap worth a threshold. Cutting a rule the
instrument cannot support is the correct move.

In its place the harness asserts an **asymmetry**: a run can leave π_ref while staying
stochastic (the KL fixture fires `kl_runaway` alone), but it cannot collapse onto a point
without leaving π_ref (the entropy fixture necessarily fires both). The three symptoms are
not independent, and a diagnostic that pretends they are will misattribute.

## Two harness lessons, both earned the hard way

**1. When a value cannot be tested reliably, assert the contract violation the library
reports.** `np.where` discarding a 0/0, and `np.divide(..., where=)` without `out=`, both
*usually* look like zeros — fresh pages arrive zeroed. A 20,000-row all-dead batch still
returned exactly 0 while reading uninitialised memory. The check now runs under
`warnings.catch_warnings` and fails on any warning, which catches both failure modes in
one line.

**2. Fix the fixture with the mechanism, not the knob.** The KL-runaway fixture was
contaminated — it fired `dead_groups` at 2 of 5 seeds, because `sharp` drives the pass
rate away from 0.5 and C4 then bites. Lowering `lr` destroys the runaway itself (KL falls
to 0.83). Raising **G from 8 to 32** fixes it, because P(dead) collapses in G:

```text
  G =  8   dead_frac 0.152 .. 0.544      contaminated
  G = 32   dead_frac 0.004 .. 0.265      clean at 8/8 seeds
```

The day's own law is what repaired the day's own fixture.

## Implementation notes

- **The numpy/torch boundary is a single line, and it is load-bearing.** Rewards and
  advantages are numpy — they are constants w.r.t. θ, so the advantage is detached *by
  construction* rather than by remembering to call `.detach()`. Log-probs are torch; they
  carry the graph. Crossed exactly once, in `train_grpo`. `np.exp` on a torch tensor
  silently returns a numpy array when no grad is required and **raises** when it is, so it
  passes value checks and then severs the graph.
- **`group_advantage` operates on the last axis**, so `(G,)` and `(n_groups, G)` share one
  code path. That forces `axis=-1, keepdims=True`.
- **Padded positions have ρ = 1, not 0.** `logp` is zeroed at pad, so `exp(0−0) = 1` and
  `min(ρÂ, clip(ρ)Â) = Â`. Forgetting the mask does not add zero — it adds a full copy of
  the advantage per pad slot.
- **The `token` length norm is a per-token gradient statement, not a scalar one.** At ρ = 1
  with a constant advantage, `token` and `batch` are equal by construction, so the length
  bias must be asserted on `∂/∂logp`: exactly **3.0×** vs exactly **1.0×** for a 3-token
  versus 1-token sequence.
- History rows record the policy that **produced** the batch, so row 0 is the reference
  checkpoint and `h.kl[0]` is exactly 0 — which is what makes row 0 usable as a baseline.

## Failure modes to check first on a real GRPO run

1. **`dead_frac`, against the pass rate.** Rising reward with a rising dead fraction is the
   signature failure. It looks like success in every other column.
2. **Where the KL term lives.** If it is inside the advantage, the β controller is
   open-loop — verify by checking whether a uniform drift moves the loss at all.
3. **`clip_frac` and μ together.** If μ = 1 and clip_frac > 0, the ratio is not what the
   code thinks it is (stale `logp_old`, or a recompute under a different mask).
4. **Whether ÷std is present**, and whether the pass-rate spread is wide enough for the
   1/std amplification to matter.
