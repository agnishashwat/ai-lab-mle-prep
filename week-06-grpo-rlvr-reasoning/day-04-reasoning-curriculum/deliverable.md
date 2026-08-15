# Week 6 Day 4 — Reasoning Data Curriculum

## What I Built / Derived

- The **set point family** `p* = a/(a + b + k − 1)`, collapsing to `p* = 1/(k+1)` at
  `a = b = 1`, and the condition for the folk answer `p* = 1/2`: **not `k = 1`, but
  `a = b + k − 1`**.
- The **precision floor** `p_floor = γ(1−τ) / [τ(1−α) + γ(1−τ)]` — the difficulty below
  which a majority of your positive rewards are false — and the **exact `k` at which
  it overtakes the metric's set point**, solved rather than scanned.
- The **`/std` implicit curriculum**: advantage whitening is itself a curriculum,
  tilting toward hard prompts by a factor of `G/[4(1−2^(1−G))] ≈ G/4`.
- **Contamination as the mirror of D3's decaying gain**: the eval lie decays in `k`,
  the training cost does not.
- `code/curriculum.py` — a 10-stage, 106-check staged harness with 4 learner TODOs.

## Core Takeaway

**A curriculum is a controller, and its set point is `p* = 1/(k+1)`, not 0.5.**

"Train on what the model is at 50% on" is the `k = 1` special case of a family, and
almost nobody reports `pass@1`. The set point is a function of the metric you report:
`p* = a/(a+b+k−1)`. Reporting `pass@8` and training at `p = 0.5` costs **22.17× the
useful signal per rollout** — you buy a 2.53× better progress rate and pay 56.12× in
metric weight.

**Second thesis: the verifier is a multiplier, not a preference. It does not tilt the
curriculum, it floors it.** `argmax_p U = 1/2` for *every* `(α, γ)` with `J > 0` — a
verifier literally cannot move the optimum. What it does instead is make a whole
region of difficulty *unusable*: below `p_floor`, most of your positive rewards are
false. `J` is a multiplier on signal magnitude; `p_floor` is a constraint on the
domain. **Different kinds of object**, which is why `J = 0.80` and `p_floor = 0.50`
coexist happily.

## Key Equations

    p*        = a / (a + b + k − 1)                     → 1/(k+1) at a = b = 1
    p* = 1/2  ⟺  a = b + k − 1                          (NOT ⟺ k = 1)
    G*        = ln(0.1)/ln(1 − p*) ≈ 2.3026·k           (single-sided; see caveat)
    U(p)      = E[Σ A_i·s̃_i] = 2(G−1)·p(1−p)·J          (±1 convention: s̃ = 2s−1)
              = J · M_gold,  M_gold = 2(G−1)p(1−p)       counterfactual gold mass
    mass_obs  = 2(G−1)p_obs(1−p_obs)                     verifier-reward advantage mass
    ratio     = U/mass_obs = p(1−p)J/[p_obs(1−p_obs)]   the SNR; → 1 as p→0 iff γ = 0
    p_floor   = γ(1−τ) / [τ(1−α) + γ(1−τ)]
    J         = 1 − α − γ                                (Youden)
    ΣA² (cen) = c(G−c)/G                                 vs G under whitening
    TILT      = [G²/(G−1)] / [4G(1−2^(1−G))/(G−1)] = G/[4(1−2^(1−G))] ≈ G/4
    inflation = c·(1 − pass@k(p_true))                   contamination, decays in k

## Implementation Notes

- **The ±1 convention is not cosmetic.** With `s ∈ {0,1}` failures contribute nothing
  to `Σ A_i s_i` even though their negative advantages are doing real work, and you
  get exactly half. `s̃ = 2s−1` gives the factor 2 and makes `U=J·M_gold`, where
  `M_gold` is counterfactual gold-reward advantage mass. Say which convention you are quoting.
- **`mass` is at `p_obs`, `U` is at `p`.** They have different denominators. Conflating
  them is what made me write "ratio → J, constant in p", which is false.
- **Verify the bracket.** `_tilt_to_target` bisected a non-monotone map — gold
  sequences share `(state, action)` pairs with non-gold ones, so a *negative* bonus
  can drive `p` to 1.0. 40% of fixture difficulties were silently wrong, errors up to
  0.97, while every stage quoted them to four decimals. Replaced with a dense scan
  plus rejection sampling that returns the achieved `p` so the caller must check it.
- **Closed forms are not automatically the reference.** `1 − p^G − (1−p)^G` cancels
  catastrophically at small `p`; exact enumeration beat it by 13 significant figures
  until it was rewritten with `expm1`/`log1p`.
- **Absolute tolerances need O(1) quantities.** `< 1e-15` on `2(G−1)p(1−p) = 31.5` is
  a demand for `1e-16` *relative* — below machine epsilon.

## The Numbers

    p* = a/(a+b+k−1) → 1/(k+1)     0.5000 / 0.2000 / 0.1111 / 0.0588 / 0.0154   k = 1/4/8/16/64
    G* both tails (ceil)           5 / 6 / 11 / 20 / 38 / 149 / 591             k = 1/2/4/8/16/64/256
    G* ≈ 2.3026·k                  approached from above
    P(dead | G=8, p*)              0.78 / 3.92 / 16.78 / 38.97 / 61.57 / 88.34 / 96.93 %
    p_floor (γ=.01, α=.10, τ=.10)  0.090909      crossover with 1/(k+1) at k = 10.0000 EXACTLY
    value retention v(floor)/v(p*) 0.9793 / 1.0000 / 0.8872 / 0.3804 / 0.0358   k = 8/10/16/32/64
    useful fraction (SNR)          0.8973 / 0.8980 / 0.6454 / 0.3109 / 0.0825   p = .5/.1/.02/.005/.001
      collapse slope               J/[γ(1−γ)] = 89.90;  γ = 0 ⇒ ratio → 1 at any α
    50%-of-peak band, k=1 → k=64   [0.1464, 0.8536] → [0.0036, 0.0407]
      width in p                   0.70711 → 0.03709      19.07× narrower
      width in logit-p             3.5255 → 2.4654         1.43× narrower   ⇒ ratio 13.33×
    centered ΣA², G=64             c=1 → 0.9844,  c=G/2 → 16.0             16.25×
    whitened ΣA², G=64             both → 64                               tilt 16.0000
    measured (a, b)                (1.900, 1.911), R² = 0.9979   ⇒ k=8 set point 0.1111 → 0.1758
    contamination c=.05, p=.30     3.500 / 2.450 / 1.201 / 0.288 / 0.017 / 0.000 pts  k=1/2/4/8/16/64
    train at 0.5, report pass@8    progress 2.53× better, weight 56.12× worse, net 22.17× LOST
    hint p 0.01 → 0.15             signal ×12.879; dead 92.27% → 27.25%; ≡ G from 8 to 130

## Failure Modes / Debugging Notes

**Which wall binds depends on your alive-tolerance, not on the system.** At a 10%
dead-tolerance the dead wall is `p = 0.2501` and it binds; at 50% it is `0.0830` and
the precision floor binds instead. Both are internally correct. Two people can argue
past each other for a week while both are right, because they never stated the
tolerance. State it.

**A noisy verifier manufactures aliveness out of a mastered prompt.** At `p = 1` with
`α = 0.1`, `p_obs = 0.9`, the group is not unanimous, and the trainer sees a healthy
pile of advantage mass — every scrap of it noise, since `U = 0` identically at the
ends. `dead_frac` reads the *labels*, so it is blind to both ends.

**The `/std` tilt is symmetric; the metric picks which tail wins.** Whitening lifts a
`c=1` group and a `c=G−1` group by exactly the same factor. It is `pass@k` that values
the hard side, which is why the whitened set point moves *down*.

**A rule can fire harder on healthy runs than sick ones.** `mean_p_rising` was cut for
exactly this: `+0.796…+0.938` healthy vs `+0.373…+0.410` sick. Rising `p` *is*
learning. The same measurement showed the `rise` conjunct inside `saturated_band` was
dead weight.

**Identical fire-sets can be a bench defect, not a rule defect.** `budget_dead` and
`saturated_band` looked collinear until the fixture set grew from 3 to 6; `cliff` and
`floor_violated` had never fired at all. **Three fixtures cannot separate four rules.**

**A threshold is calibrated for a regime.** Alive-fall gap: 0.35 wide at 150 steps,
0.014 wide at 60. You may re-derive thresholds for a faster regime; you may not
borrow them.

**A change statistic must not overlap the estimator's burn-in.** `p_hat` is an EMA
from a prior, so a head window at step 0 measures the *sensor* converging: prompts
whose true `p` never left `1e-4` read as having moved `0.047…0.276`.

**A detector is an estimator.** The twin-gap test has `se = sqrt(p(1−p)/n_contaminated)`;
at `n = 4` that is 0.23 and it cannot distinguish a clean set from a quarter-rotten one.

## Interview Answer

> **"How would you build a curriculum for RL on reasoning data?"**

Closed-loop, and keyed to the metric I report.

A curriculum is a controller. Its set point is not 0.5 — that is the `pass@1` special
case. Rollout value is progress rate times metric weight: if the local learning law is
`Δp ∝ p^a (1−p)^b` and I report `pass@k`, whose derivative is `k(1−p)^(k−1)`, then
value `∝ p^a (1−p)^(b+k−1)` and the set point is `p* = a/(a+b+k−1)`. At `a=b=1` that
is `1/(k+1)`. Reporting `pass@8` and training at 0.5 loses 22× the useful signal per
rollout.

Three things make that operational. **First, `p` is a property of the policy, so I
estimate it from recent rollouts — `c/G` is already paid for — rather than assigning a
static difficulty once.** A statically-sorted curriculum decays: it was configured with
a property of the *initial* policy. **Second, I control in logit-`p`.** From `k=1` to
`k=64` the useful band narrows 19× in `p` but only 1.4× in log-odds; controlling in `p`
makes the problem 13× harder than it is. **Third, I measure `(a, b)` rather than
assuming them** — in my own harness they came out `(1.90, 1.91)`, not `(1,1)`, which
moves the `k=8` set point by 1.58×. The `+k−1` is arithmetic; the exponents are not.

Two constraints bound the band from below. **Group size**: at `p*` you need
`G* ≈ 2.3026k` to keep 90% of groups alive, and with `G = 8` at `k = 64` you are 88%
dead. **The verifier's precision floor**: below `p_floor = γ(1−τ)/[τ(1−α)+γ(1−τ)]` most
of your positive rewards are false. The verifier does not *tilt* the curriculum —
`argmax_p U = 1/2` for any `(α, γ)` — it *floors* it. Those are different objects,
which is why a verifier with `J = 0.80` can still be unusable below the median. And
which of the two walls binds depends on the alive-tolerance you choose, so I state it.

Finally I would watch what my advantage normalisation is doing, because `/std` is
itself a curriculum: it tilts toward near-dead groups by `≈ G/4`, which is 16× at
`G = 64`. That is not a nudge — it is a different curriculum than the one I designed.

## Reusable Artifact

`week-06-grpo-rlvr-reasoning/code/curriculum.py` — 10 stages, 106 checks, 4 TODOs
(`signal_curve`, `useful_signal`, `set_point`, `diagnose_curriculum`). The diagnostic
panel ships 4 calibrated rules (`cliff`, `saturated_band`, `floor_violated`,
`budget_dead`) with thresholds derived from 5 paired seeds across 6 fixtures, plus one
rule cut for a measured reason.
