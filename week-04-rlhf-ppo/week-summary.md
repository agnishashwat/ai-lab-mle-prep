# Week 04 Summary — RLHF and PPO for LLMs

**Days 1–5 complete. 515 property-based checks across the week**, every one asserted against
exact enumeration or a central finite difference, all under `-W error::RuntimeWarning`:
`toy_rlhf.py` **73**, `ppo_loss.py` **112**, `token_ppo.py` **144**, `failure_modes.py` **96**,
`ppo_rlhf.py` **90**.

## The 5-Minute Version

Week 3 built the RL substrate on a 2-state finite MDP. Week 4 rebuilds it in a setting where
tokens, EOS, sequence reward and length are **literal** rather than analogized — a token-level
LM environment (`vocab=4, t_max=5`) small enough to **enumerate all 364 terminating
sequences**. That gives `exact_J`, `exact_sequence_kl` and an exact finite-difference gradient
for the whole week, so **every unbiasedness or bias claim is an assertion against ground truth,
not a plot**.

**Day 1** builds the pipeline and a Bradley-Terry reward model, and finds the identifiability
result: **the RM is determined only up to a per-prompt constant `c(x)`** — which is exactly what
GRPO's group-mean baseline removes, and exactly why raw RM scores are never comparable across
prompts.

**Day 2** derives `L^CLIP` and measures the three ways it **is not** a trust region, organized as
**three levels of aggregation**: token (`T·ε²/2` ≈ 10 nats at T=500), sample (`clip_frac 0.237`
vs `oob_frac 0.570` — it only bites where the advantage points outward), and iteration
(**π_old resets, so clipping bounds the step, not the walk**: 4.68 nats of drift at a 5.3% clip
fraction versus 0.75 with an explicit penalty, **6.2×**). That gap is *why InstructGPT ships
both a clip and a KL penalty*. Day 2 also shows why the ratios are **per-token**: `∏_t ρ_t` is
log-normal with mean exactly 1 and a collapsing median, so at **T=500 a 1024-sequence batch has
an effective sample size of 0.38**.

**Day 3** adds a learned value head and per-token KL shaping, and proves the shaping is **exact,
not an approximation**: `ρ₀` and the dynamics cancel in the ratio, so `Σ_t [log π_θ − log π_ref]`
*is* the sequence log-ratio (`0.405113796560` both ways). The critic result: **a critic cannot
bias you as a BASELINE, only as a BOOTSTRAP** — at λ=1 every critic reproduces the exact
gradient to **2.8e-10**; at λ=0.5 the same zero critic is **0.796** off. Bias tracks the
*state-varying* part of the error (Spearman **+0.995**), which is why **EV**, not MSE, is the
critic instrument.

**Day 4** plants and measures four failure modes, and the frame is the payload:
**over-optimization means the OBJECTIVE is wrong; optimization pathology means the OPTIMIZER is
broken.** They present identically on a reward curve and their fixes are opposite. Two of the
four are **invisible to KL** — entropy collapse happens inside π_ref's support, length
exploitation changes one decision (EOS) many times. Day 4 also closes the week's biggest
perennial: **off-policy-by-accident, measured against exact ∇J** (on-policy 2.95e-10,
temp 0.7 **0.2284**, top-p 0.9 **0.1169**), with an importance-sampling fix that repairs
temperature to **2.953e-10** and **provably cannot** repair top-p (**0.1169 → 0.1267, it got
worse**), because IS needs `supp(π) ⊆ supp(q)` and **7.2%** of the mass is unobservable.

**Day 5** assembles all of it and adds the week's one new piece of theory: **adaptive β is dual
gradient ascent.** β was never a hyperparameter — it is the Lagrange multiplier of
`max E[r] s.t. KL ≤ δ`, the envelope theorem gives `∂g/∂β = δ − KL`, and InstructGPT's
controller is proportional dual ascent on it. **Verified: the controller's fixed point equals
`dE[r]/dKL` on the closed-form frontier to 1–5% at four budgets**, by two structurally
independent routes. And the ablation grid states the week in one inversion: **removing the KL
term raises the proxy (0.8728 vs 0.6153) and lowers the gold (4.9098 vs 5.1501)**.

## Key Concepts (crisp)

- **BT identifiability.** The RM is identified only up to `R(x,y) → R(x,y) + c(x)`. A per-prompt
  baseline is therefore **free**; a cross-prompt one is not. This is why GRPO's group mean works.
- **Clipping ≠ trust region — three levels of aggregation.** Token / sample / iteration. PPO
  approximates a constraint by **deleting the gradient**, not by enforcing a bound.
- **Per-token ratios are a deliberate bias.** The sequence-level IS estimator is unbiased and
  unusable (ESS 0.38 at T=500); the per-token one is usable and no longer unbiased for the
  sequence objective. Same trade as clipping.
- **The KL penalty is REWARD SHAPING, not a loss term.** It modifies `r` upstream of every
  estimator, so nothing downstream contains a β — and π_ref can be dropped after the shaping step.
- **Critic bias is a bootstrapping phenomenon.** As a baseline a critic is always unbiased
  (EGLP); only `λ < 1` lets its *state-varying* error in. EV over MSE, for the same reason.
- **KL is a confidence region for the RM, not a regularizer.** `π_ref = π_SFT` *is* the RM's
  sampling distribution ⇒ **β is a property of the RM's data.** Retune it after improving the RM.
- **The four-way failure taxonomy, with the split first.** Objective wrong vs optimizer broken.
  Identical reward curves, opposite fixes, and **KL sees only half of them**.
- **Goodhart is a variance result, not just a hump.** Seed spread amplifies **89× / 259× / 667×**
  with optimization pressure; at 30 comparisons the spread (**±1.60**) exceeds the entire mean
  improvement over π_ref (**+0.95**). ⇒ **≥3 seeds is mandatory.**
- **Adaptive β = dual ascent.** Penalty form and constraint form are primal and dual views of one
  problem. `g(β)` is a max of affines ⇒ convex regardless of the primal.
- **Early stopping is the ESTIMATOR for the optimum**, not regularization: gold is concave in
  √KL with an interior maximum, so the final policy is systematically the wrong one to ship.
- **A single metric is a hypothesis; a pair is a test.** The panel is read in pairs, and a
  reward-only dashboard detects none of the four failures.

## Key Equations

- `∇J = E[Σ_t Ψ_t ∇log π(a_t|s_t)]` — every algorithm differs **only in Ψ_t**.
- `L^CLIP = E[min(ρ A, clip(ρ, 1±ε) A)]`; gate = 0 exactly where the update pushes further out.
- `r_t = r_ψ(y)·1[t = L−1] − β·k̂_t` — sparse from the RM, dense from the KL.
- `Σ_t [log π_θ(a_t|s_t) − log π_ref(a_t|s_t)] = log π_θ(y|x) − log π_ref(y|x)` — exact.
- `R̂_t = Â_t + V_φ_old(s_t)` — the critic's target. **No γ.**
- `π*(y) ∝ π_ref(y) exp(r(y)/β)`; `J* = β log Z`; `dJ*/dβ = −KL`; `dE[r]/dKL = β`.
- `L(θ,β) = E[r] − β(KL − δ)`; `∂g/∂β = δ − KL`; `β ← β(1 + K_β(KL/δ − 1))`.
- Lengthening pays iff `Δ = κ + Δr_content − β·E[KL_t] > 0`; **β\* is a fixed point, not a
  formula**, and the resistance is **bounded** (max 0.1411/token measured).
- Deadly-triad divergence condition: `A = Σ visitation·φ(γφ′ − φ)` — **−3.70 converges,
  +1.07 diverges to 4.4e13**.

## What I Implemented

| file | checks | contents |
|---|---|---|
| `toy_rlhf.py` | **73** | enumerable token env, `exact_J`, `exact_sequence_kl`, tilted distribution, BT loss + gradient, preference sampling |
| `ppo_loss.py` | **112** | IS and clipped surrogates, analytic `surrogate_gradient`, ratio statistics, epoch sweep, drift experiment, β-frontier |
| `token_ppo.py` | **144** | GAE, value table + regression, per-token KL shaping, whitening, `instructgpt_loss`, error decomposition, zero-baseline probe |
| `failure_modes.py` | **96** | linear BT RM over 23 features, distorted samplers, off-policy gradient + IS repair, length confound + `β*` prediction, TD divergence, **`diagnose`** |
| `ppo_rlhf.py` | **90** | SFT reference, **KL controller**, full loop, checkpointing, selection, 7-arm × 3-seed ablation, **dual-ascent fixed-point check** |

## Failure Modes I Now Understand

| failure | mechanism | instrument (a **pair**) | visible in KL? |
|---|---|---|---|
| **Reward hacking** | RM fit on π_ref data; the policy climbs the **argmax of the error** | proxy **+ gold** vs √KL | yes |
| **Length exploitation** | annotator confound ⇒ RM acquires κ>0 ⇒ EOS suppressed; sum-over-tokens gives no brake | length **+ length-controlled win rate** | barely |
| **Entropy collapse** | mode reinforced → sampled more → reinforced more; a distorted sampler causes it directly | entropy **+ unique-completion count** | often no |
| **Value divergence** | **deadly triad**: function approximation + bootstrapping + off-policy-ness | **EV** + value loss | no |
| **Policy drift** | clipping bounds the step, not the walk | KL **+ β** | it *is* the KL |

## Open Questions / To Revisit

- **`diagnose` thresholds need recalibration on real runs.** Its `rel()` divides by the first
  value, which is exactly 0 when the RM is standardised under π_ref; and the −20% entropy
  threshold sits inside the range a *healthy* budgeted run occupies (a monotonically-improving
  run at δ=0.4 dropped 26%).
- **The critic bought nothing at T=5** (5.1786 without vs 5.1501 with, and lower spread). Re-test
  the claim on a longer horizon before generalizing either way.
- **`no_clip` ≈ `full`** under a tight KL budget. Worth constructing a regime where the clip is
  the binding constraint, to see it earn its place.
- **Length-controlled win rate is named but not implemented.** It is the instrument for the one
  failure mode whose detector I never built.
- **Shared trunk.** `c₁` coupling is simulated here; the real object is two heads contending over
  shared parameters.
