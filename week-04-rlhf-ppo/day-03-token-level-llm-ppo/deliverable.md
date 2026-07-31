# W4 D3 — Token-Level PPO: Value Heads, Sequence Rewards, Per-Token KL

**Code:** `week-04-rlhf-ppo/code/token_ppo.py` — **144/144 checks**, all against enumeration
or finite-difference ground truth. Fixture env `vocab=5, t_max=4` (V=5, T=4, S=85, B=341 —
four different numbers; value arrays are `(B, T+1)` while everything else is `(B, T)`).

**Session score:** Block 0 cold recall 5.1/8, Block 1 quiz 3.55/6 (running 10.75/18).

---

## The objective, with each term's job

\[
L(\theta,\phi) = -\underbrace{L^{\text{CLIP}}(\theta)}_{\text{D2: off-policy reuse}}
  + c_1\underbrace{L^{\text{VF}}(\phi)}_{\text{today: the critic}}
  - c_2\underbrace{\mathbb{E}[H(\pi_\theta)]}_{\text{= 0 in InstructGPT}}
\]
\[
r_t \;=\; \underbrace{r_\psi(y)\,\mathbb 1[t=L-1]}_{\text{SPARSE, one scalar at EOS}}
      \;-\; \underbrace{\beta\,\hat k_t}_{\text{DENSE, every token}}
\]

**One number, T decisions.** That mismatch is all of LLM RL. Two attacks, both owned:
estimator-side (GAE spreads the terminal signal backwards) and reward-side (per-token KL
densifies the reward → densifies the critic's regression target → densifies every δ_t).

`c_2 = 0` is not laziness. A uniform-entropy target is *actively wrong* for text — you want
high entropy where genuinely uncertain, not after "the capital of France is." KL-to-reference
expresses a position-dependent target that a scalar entropy bonus cannot.

---

## Headline 1 — a critic cannot bias you as a baseline, only as a bootstrap

The reconciliation asked for in checkpoint Q3, now measured. **Same critics, same rollouts,
only λ changes:**

| critic | bias at λ=1 | bias at λ=0.5 |
|---|---|---|
| exact V\* | 2.8e-10 | 2.8e-10 |
| zero (untrained) | 2.8e-10 | **0.7959** |
| V\* + 25 | 2.8e-10 | **7.4464** |
| pure noise (σ=6) | 2.8e-10 | **3.8025** |
| anti-critic (−V\*) | 2.8e-10 | **1.5918** |

At λ=1 the critic enters **only** as `V(s_t)` subtracted — h_t-measurable by construction,
hence a baseline, hence free **for any critic however bad**. 2.8e-10 is the
finite-difference floor. At λ<1 it bootstraps and the state-varying error biases you.

> *Interview sentence:* "It depends on λ. As a baseline it cannot bias me — it's a function
> of s_t. Inside GAE with λ<1 it bootstraps, and there the state-varying part of the error
> does bias me."

---

## Headline 2 — bias tracks ‖ẽ‖, not ‖e‖, and EOS is what makes level error expensive

Learned critic, error decomposed `e = ē + ẽ` (occupancy-weighted) along the fit trajectory:

| critic | \|ē\| | ‖e‖ | ‖ẽ‖ | EV | bias |
|---|---|---|---|---|---|
| fit@0 (untrained) | 0.2949 | 0.9633 | 0.9170 | 0.0000 | 0.7959 |
| fit@200 | 0.0155 | 0.1679 | 0.1672 | 0.4138 | 0.0315 |
| fit@1000 | 0.0006 | 0.0397 | 0.0396 | 0.4272 | 0.0037 |
| fit@4000 | 0.0001 | 0.0095 | 0.0095 | 0.4280 | 0.0005 |
| noise 0.40 | 0.4309 | 0.7909 | 0.6633 | 0.2041 | 0.2721 |
| noise 2.50 | 1.3719 | 3.1468 | 2.8320 | **−3.6538** | 1.9812 |

Spearman(bias, ‖ẽ‖) = **+0.995**; Spearman(bias, EV) = **−0.995**. Pythagoras
‖e‖² = ē² + ‖ẽ‖² asserted. Negative EV is informative: worse than predicting the mean.

**A real regression's error is almost entirely the expensive kind** — ‖ẽ‖ ≈ ‖e‖ at every
checkpoint, ē dies immediately. The free component is the one gradient descent fixes first.

### The discriminating pair — this sharpens W3 D4

Two critics, **identical** ‖e‖ = 1.5: one pure level (`V* + c`), one pure ranking (`V* ± c`).

| λ=0.5, equal ‖e‖ | wrong LEVEL | wrong RANKING |
|---|---|---|
| **fixed length** (EOS suppressed) | **1.5e-10** *(= the FD floor — literally nothing)* | 0.9724 |
| **variable length** (EOS live) | **0.4468** | 0.8172 |

**This corrects `session.md`.** A constant offset shifts advantages by exactly
`−c·λ^(L−1−t)` (asserted elementwise to 3.6e-15). When L is fixed, the remaining horizon is
a function of t alone ⇒ h_t-measurable ⇒ a baseline ⇒ **exactly free even under
bootstrapping**. When EOS is live, **L is an action**, the shift is action-dependent, and it
biases. The pinned terminal row (V=0, unshiftable) is what breaks the uniformity.

> *The transferable claim:* "wrong level is free" is a statement about the **baseline role**.
> Under bootstrapping in an env with variable length, it is no longer free — and an LLM
> always has variable length.

---

## Headline 3 — EV, not MSE, and why

Negative control, `c = 25`:

| | plain V\* | V\* + 25 | verdict |
|---|---|---|---|
| MSE | 1.1240 | 626.124 | grows by **exactly c² = 625**, unbounded |
| EV | 0.428000392162 | 0.428000392162 | **bit-identical** |
| gradient (λ=1) | — | max\|Δ\| = 1.7e-15 | bit-identical |

Var is shift-invariant, and the shift is precisely W3 D4's free component. **EV is the right
instrument because its invariance matches the theorem's invariance** — choose a diagnostic
whose invariances match those of the thing you're detecting. A critic can have catastrophic
MSE and zero bias.

**EV converges to 0.4280, not 1** — an analytic ceiling `1 − E[(G−V*)²]/Var(G)`, the
irreducible within-state return variance. A critic at EV 0.43 here is *perfect*.

---

## The decisive experiment (closes the standing D4 gap, 0.1 → owned)

`zero_baseline_probe` — recompute gradient variance on the **same rollouts** with the
baseline zeroed. All four variants are unbiased at λ=1 (2.8e-10), so the comparison is
purely variance:

| baseline | Var(∇) | vs none |
|---|---|---|
| none | 4.582 | — |
| depth-only | 4.306 | **−6.0%** |
| exact V\* | 2.632 | **−42.6%** |
| anti-correlated | 10.453 | **+128.1%** |

> *Say it as one sentence:* "Zero the baseline and recompute the gradient variance on the
> same rollouts. Healthy critic reduces it; mis-specified critic increases it." Decisive
> because it needs **no ground-truth gradient** — which a real stack does not have, which is
> why "compare against true gradients" was unusable.

---

## Per-token KL shaping IS the D5 objective, sampled

\[
\mathbb E\Big[\textstyle\sum_t \text{penalty}_t\Big] = 0.405113796560
\quad=\quad \beta \cdot \text{KL}_{\text{seq}} = 0.3 \times 1.350379 = 0.405113796560
\]

Agreement to **1e-11**, for **both** k1 and k3, exact on the enumerated batch (not "within
MC error"). Sampled rollouts converge to it: 0.40376 at n=20,000. Now over an honest
terminal state and variable length — the EOS boundary D5's tabular version couldn't cover.

**k1 vs k3:** same mean to 1e-12 (0.132314344134 both). k1 pays a **bonus on 28.2%** of
token weight; k3 is ≥0 pointwise, never. Variance ratio **k3/k1 = 0.865**.
`E[e^{lr}] = 1` makes `e^{lr} − 1` a zero-mean control variate; `e^x ≥ 1+x` gives k3 ≥ 0.

**Why KL in the reward, not the loss:** (i) **density** — a scalar loss term gives one
number per sequence, the reward form gives the critic something to predict at every t, which
raises EV, which makes GAE's bootstrap trustworthy at smaller λ; (ii) it flows through the
**advantage**, so it's automatically credit-assigned and clipped with everything else;
(iii) the RL machinery never needs to know the KL exists.

---

## Value targets, the value clip, whitening, c₁

**Target is `Â + V_old`, not G_t.** At λ=1 they coincide *exactly* (8.9e-16 — telescoping is
algebra); at λ<1 the target is bootstrapped with **lower variance** (0.9016 vs 1.9649). The
critic rides the same bias/variance dial as the actor.

**The value clip is a `max` — pessimism applied to a regression.** Hand case: a critic that
jumps straight to the right answer (`V_new = target = 5`, `V_old = 0`) has plain loss **0**
and clipped loss **16**. *It punishes correctness.* No theoretical justification; in every
implementation. Note `clip_frac` counts where the max **selected** the clipped branch, not
where V left the band — overshooting *away* from the target leaves the plain branch larger.

**Whitening.** Centering leaves the exact gradient **identical** (2.8e-17) — a batch-level
constant baseline. Std division within one batch is exactly a learning-rate rescale (2.2e-16
after undoing it). **Across groups it is not:** two groups with a 5× spread, pooled scaling
preserves the ratio (0.200000), per-group scaling **destroys** it (1.000000) — the
low-variance group amplified 5×. That is Dr. GRPO's critique, in two lines.

**c₁ (shared trunk), J(θ₀) = 0.2798:**

| c₁ | 0.0 | 0.1 | 0.5 | 5.0 |
|---|---|---|---|---|
| gain | +1.2997 | +1.2596 | +1.3812 | **+0.0328** |

**Not monotone.** Neutral across the usual range, then catastrophic — c₁=5 keeps **2.5%** of
the gain. So "we tuned c₁ and it made no difference" is *not* evidence the coupling is
harmless. The value loss competes for the trunk.

**Sum vs mean:** `reduce="mean"` moves weight on len≤2 completions from **0.3371 → 0.5522**.
Not a normalization choice — a length preference. (Day 4 names it.)

---

## Bugs hit, and what each generalizes to

| bug | lesson |
|---|---|
| `np.add.at` used for a **gather** | gather vs scatter are adjoints: `V = φ[ids]` forward, `dφ[ids] += dV` backward. Ask "can two inputs collide on one output slot?" |
| `zeros_like(state_ids)` → int64 | dtype inherits from the **index** array; W3 D2's `np.long` buffer again — silent truncation, 0.7 → 0 |
| `δ = r + γ(1−d)(V' − V)` | the baseline subtraction is **unconditional**; `(1−d)` attaches only to V'. The spec was three lines above the bug |
| `reward_fn(state_id)` | **states ≠ sequences.** The RM scores completed text; there is no r(s). Use `batch.seqs[b]` |
| `whiten(..., whiten_mode)` | `bool("scale") == True` — a bool param fed a mode string fails **silently** |
| `E[A²] − E[A]²` → `sqrt(−1.78e-15)` → nan | catastrophic cancellation; use `Σw(A−μ)²`. And **`nan < 1e-5` is False** — a guard positioned after nan is created *inverts* |
| `log.append(phi)` | appends a **reference**; with in-place `-=` all snapshots alias. W3 D3's `history.append(tensor)` |

---

## Carried into Day 4

1. **The rare-state tail**: after 4000 steps the critic is still off by up to **0.942** at
   states with occupancy < 1e-4 (gradient ∝ occupancy ⇒ ~3000× slower). A critic can post
   healthy EV and be badly wrong exactly where a policy is about to go. That is the
   over-optimization channel.
2. **Length**: `reduce="mean"` upweighting (0.3371 → 0.5522) and level-error-costs-only-with-EOS
   both say the same thing — length is an action here.
3. **c₁'s non-monotonicity** as a warning about tuning conclusions from small sweeps.
