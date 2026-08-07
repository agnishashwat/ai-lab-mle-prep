# Deliverable — Week 05 Day 04: DPO Variants (IPO / KTO / ORPO / SimPO)

**Artifact:** `week-05-dpo-preference-optimization/code/variants.py` — ten stages,
**126 checks**, clean under `-W error::RuntimeWarning`. Full run 48.8 s, `--fast` 12.1 s.

## Core Takeaway

> The four "variants" are not four papers. They are four cells of one 5×5 grid, and the
> grid is `(implicit reward f) × (link ℓ)`. Every named method is one line.

```text
L(θ) = mean_k  ℓ( f(y_w) , f(y_l) )        [ + λ_SFT · mean_k −log π(y_w)/|y_w| ]
```

`f` maps a completion to a scalar; `ℓ` maps a pair of scalars to a per-pair loss. Choosing
`f ∈ {ref_sum, ref_mean, noref_sum, noref_mean, odds}` and
`ℓ ∈ {logsigmoid, logsigmoid_smoothed, squared_target, logsigmoid_margin, prospect}`
gives 25 cells, of which the literature has named four. `stage_reduce` asserts the grid
reproduces five hand-written direct implementations **elementwise, max deviation 0.000e+00**.

The two axes act on **different things and do not interfere**:

- the **link** decides whether the margin is **finite**,
- the **reward** decides where the **length** goes.

`stage_empty_cell` proves this by occupying `ref_mean × squared_target` — "length-normalised
IPO", a cell no paper writes. It inherits τ's cap from IPO and the mean reduction's length
amplification from SimPO, exactly as the grid predicts.

## The three problems, and which axis fixes each

D1–D3 measured three failures. Each variant is an answer to one of them.

| # | Problem | Measured on D1–D3 | Axis that fixes it | Named method |
|---|---|---|---|---|
| P1 | reward runs away on separable data | margin `~ log t`, `+0.7163` then `+0.7078` per doubling | **link** — give it an interior minimum | IPO (τ), cDPO (ε) |
| P2 | rejected logprob falls; both logprobs can fall | `Δ log π(y_l) = −0.1161` at convergence | **reward** + a term outside the pairwise difference | ORPO (odds + SFT) |
| P3 | length / reference confounds | `E[|y|]` drifts with the labels | **reward** — reduction and reference | SimPO (mean, no ref, γ) |

**Why a link fixes P1 and a reward cannot.** `stage_link` measures common-shift invariance:
add `c = +3.7` to both `r_w` and `r_l`.

```text
logsigmoid           2.220e-16     <- function of d alone
logsigmoid_smoothed  2.220e-16
squared_target       3.553e-15
logsigmoid_margin    2.220e-16
prospect             5.800e-01     <- sees the MARGINALS, not just d
```

Four of the five links **cannot see** a move that lowers both logprobs. That invariance *is*
P2, and it is why those four all require pairs. Only `prospect` breaks it — which is exactly
what lets KTO drop pairs.

## Key Equations

### The five implicit rewards

```text
ref_sum      r = β ( log π(y) − log π_ref(y) )
ref_mean     r = β ( log π(y) − log π_ref(y) ) / |y|
noref_sum    r = β log π(y)
noref_mean   r = β log π(y) / |y|
odds         r = β log[ P / (1−P) ],   P = exp( log π(y)/|y| )
```

### The five links (all return a LOSS; standing probe: `ℓ(d=0) = +log 2 = 0.693147`)

```text
logsigmoid           −log σ(d)
logsigmoid_smoothed  −(1−ε) log σ(d) − ε log σ(−d)
squared_target       (d − τ)²
logsigmoid_margin    −log σ(d − γ)
prospect             λ_D σ(−(r_w − z)) + λ_U σ(−(z − r_l))          d = r_w − r_l
```

### The four closed-form single-pair optima (`stage_optima`, each verified against a numerical minimiser)

```text
logsigmoid           M* = logit(f)                      diverges as f → 1
logsigmoid_smoothed  M* = logit( (f−ε) / (1−2ε) )       FINITE
squared_target       M* = (2f − 1) τ                    FINITE
logsigmoid_margin    solves f σ(γ−M) = (1−f) σ(M+γ)     NO CLOSED FORM; still diverges
```

**Only two of five links have a finite optimum at `f = 1`.** Adding a margin γ to BT *moves*
the runaway; it does not stop it (`21.2233` at `f = 1−1e−9`).

The margin link is worth remembering because the natural guess is wrong:

```text
naive  "M* = logit(f) + γ" = 0.905465        TRUTH = 0.533227      (f=0.60, γ=0.5)
```

γ is subtracted **whichever way the label points**, which destroys BT's antisymmetry. At
`γ = 0` it collapses back to `logit(f)`, as it must.

### cDPO's cap, reproducing D3 exactly

```text
  ε      M* = logit(1−ε)     Δ̂* = M*/β   (β = 0.3)
0.05            2.944439          9.814797
0.10            2.197225          7.324082
0.20            1.386294          4.620981
0.35            0.619039          2.063464
```

## The headline experiment

`stage_separable`, on D1's **non-strongly-connected matching** (Ford ⇒ the BT MLE is at
infinity). 8000 steps, plain GD, `lr = 0.5` shared by all three arms.

```text
variant     m(T/4)    m(T/2)      m(T)       d1       d2       KL    acc
DPO         4.5517    5.2680    5.9758   0.7163   0.7078   0.9511  1.000
cDPO        2.1961    2.1972    2.1972   0.0011   0.0000   0.4704  1.000
IPO         2.0000    2.0000    2.0000   0.0000   0.0000   0.2332  1.000
```

- **`acc = 1.000` for all three.** The stage's own check is named *"the metric that reports
  nothing."* Accuracy saturates in ~20 steps and then reports "perfect" for 7,980 more while
  the reward runs to infinity underneath it.
- **cDPO stops at `2.1972 = ln 9 = logit(1−ε)`** and IPO at **exactly `2.0000 = (2f−1)τ`** —
  both hitting the closed forms above.
- **KL is the cost ledger: `0.9511 → 0.4704 → 0.2332`.** cDPO buys back half the drift,
  IPO three quarters. The point-mass bound is `6.3216`, so nothing here is "unbounded".

## FOUR places where the received story is wrong

Every one of these was *asserted first, then measured*, and the measurement won.

### 1. "IPO caps the margin at the same value regardless of preference strength" — FALSE

IPO's optimum is `(2f−1)τ`, which still depends on `f`. The true statement is **compression**:

```text
strong pair f=0.99, weak pair f=0.6, τ=2.0
DPO/BT  margins 4.5951 vs 0.4055   ratio 11.3330
IPO     margins 1.9600 vs 0.4000   ratio  4.9000     <- 2.31× squeeze
```

**That is IPO's price.** A near-certain preference and a 60/40 preference get forced into the
same bounded band. Fixing P1 costs the ability to express *how much* better `y_w` is.

### 2. "DPO's KL diverges" — FALSE in a finite space

D2 established `KL(π‖π_ref) ≤ max_y −log π_ref(y)`. What diverges is the **margin**, and its
signature is not "% growth" (which decays even for a divergent run) but a **constant increment
per doubling of the budget**: `+0.7163` then `+0.7078`. That is `margin ~ log t`, the
separable-logistic-regression rate — a theorem about *gradient descent*, which is why the
harness forbids Adam.

### 3. "SimPO's `1/|y|` removes length bias" — FALSE; it AMPLIFIES it

At a **matched margin**, on both a longer-wins and a shorter-wins ladder:

```text
labels        margin   sum E|y|   d(sum)  mean E|y|  d(mean)
LONG wins        1.0     4.0732  +1.8342     4.4490  +2.2101
LONG wins        2.0     4.7591  +2.5202     4.9109  +2.6720
SHORT wins       1.0     1.3631  −0.8759     1.1041  −1.1349
SHORT wins       2.0     1.1476  −1.0914     1.0435  −1.1955
```

MEAN moves `E[|y|]` **further than SUM in all four cells**. Mechanism: `d(r)/d(log π) = β/|y|`
is largest at `|y| = 1`, so short sequences are the **cheapest lever**; the optimiser buys its
margin at the short end and pushes length further in *whichever direction the labels point*.

**Matched margin is mandatory to see this.** Run to convergence instead and both arms saturate
at `|y| = t_max`, hiding the effect completely.

What actually shortens SimPO's outputs is γ and reference-freeness. Reference-freeness is a
separate trade with its own bill, at matched margin `3.0` on equal-length pairs:

```text
reward         step    margin        KL     H(π)    E[|y|]
ref_sum         486    3.0003    1.2091   1.4627    2.0939
noref_sum       478    3.0014    2.0042   1.3131    2.2139        H(π_ref) = 2.0657
```

Reference-free collapses entropy further **and** moves 1.66× further from `π_ref` at the same
margin — because the `noref` reward *is* `log π`, so it rewards likely text. SimPO ships two
trades in one method, and γ is fighting the second one.

### 4. "KTO's loss is bounded, so unlike DPO it cannot run away" — FALSE

The most seductive of the four, because the premise is true. `prospect` really is bounded:

```text
prospect stays in [0, λ_D + λ_U] even at |r| = 1e6   (range [0.0000, 2.5000])
logsigmoid on the same input                        (max 2e+06)
```

But **a bounded loss is not a loss with a finite minimiser.** `inf_u σ(−u) = 0`, attained only
at `u = +∞`. So KTO's optimum on separable data is at infinity exactly like DPO's:

```text
run       m(T/4)   m(T/2)     m(T)      d1      d2
DPO       3.7549   4.4882   5.2088  0.7333  0.7207
KTO       2.8559   3.6151   4.5445  0.7592  0.9294     <- FASTER than DPO
```

Both gradients decay like `e^−u`, so both give `margin ~ log t`. cDPO and IPO stop because
their losses have an **interior minimum**; KTO has neither an ε nor a τ.

**What boundedness actually buys is bounded INFLUENCE:**

```text
d      |grad| −log σ(d)   |grad| σ(−(r−z))
−2.0      0.8807970780          1.050e-01
−20.0     0.9999999979          2.061e-09
```

A catastrophically mislabelled example pulls at strength **1 forever** under DPO, and is
**abandoned** under KTO. With 4/40 labels flipped:

```text
run           clean margin  clean acc  pull toward noise
DPO clean           4.1856     1.0000            −2.2216
DPO noisy           2.9445     0.9000            +2.8343
KTO clean           3.2888     1.0000            −1.8328
KTO noisy           2.6204     0.9000            +1.5870
```

DPO's clean-pair margin falls **1.2411** against KTO's **0.6684**. **KTO answers D3's noise
problem, not D1's P1.**

## ORPO: both ingredients matter, and only one is robust

`stage_orpo`, 2×2 ablation on a **strongly-connected** fixture (so nothing is a divergence
artifact), 3000 steps:

```text
arm           chosen NLL/tok    d(NLL)  d logp_w  d logp_l   margin    acc
DPO                 1.209615 −0.092598    0.3035   −0.1161   0.2098  0.750
DPO+SFT             1.042314 −0.259899    0.8435   +0.4751   0.1842  0.550
ORPO                1.029723 −0.272490    0.8699   +0.5218   0.1710  0.700
ORPO−noSFT          1.078406 −0.223808    0.6507   +0.1845   0.2098  0.750
```

`d logp_l = −0.1161` under plain DPO **is P2**, directly measured. Both ORPO ingredients turn
it around. Decomposition of the NLL gain: **odds link −0.131210, then adding SFT −0.048683** —
so "ORPO is just the SFT term" is too glib; the link does more of the NLL work.

**But run it at more than one β and the story changes:**

```text
  β    d logp_l (odds, no SFT)   d logp_l (odds + SFT)
0.25                   −1.8994                  +0.5660
0.50                   −0.4573                  +0.5282
1.00                   +0.1845                  +0.5218
2.00                   +0.3854                  +0.5566
```

**The odds-alone answer to P2 flips sign with β. The SFT answer holds at every β.**
Moral: *run the ablation at more than one hyperparameter before you name a mechanism.*

**Why it works at all** — P2 comes from the loss being a function of an **affine, equal-slope
difference**. Add `c = +0.37` to every logprob:

```text
ref_sum      moves 5.5511e-17     <- a common fall is INVISIBLE
noref_sum    moves 0.0000e+00
ref_mean     moves 1.2333e-01     = β·c·(1/|y_w| − 1/|y_l|), exactly
odds         moves 2.1883e-01     <- nonlinear in log π
```

`L_SFT` is not a difference at all, and the odds map is not affine
(`d(log-odds)/d(mean lp) = 1/(1−P) ≥ 1`, so it weights the winner more as `P_w → 1`). Length
normalisation breaks the invariance too — **but only when the lengths differ**; on equal-length
pairs `ref_mean` is shift-invariant again (`0.000e+00`).

## KTO: the paired loss with the pairing deleted, and you can see the seam

The prospect link is a **sum of two independent per-example terms**, so evaluating it never
required the two halves to come from the same pair. Asserted to machine precision:

```text
KTO(unpaired, flattened pairs) == 0.5 × mean(prospect link on those pairs)
0.475211814705432  vs  0.475211814705432
```

The factor 2 is just 2n examples against n pairs. **Dropping pairs is exact, not an
approximation.** `λ_D/λ_U` is then the knob for both loss aversion and class imbalance:

```text
λ_u   r_desirable   r_undesirable    margin
0.50       0.7587         −1.8907    2.6494
1.00       0.6781         −2.6107    3.2888
2.00       0.2416         −3.5897    3.8313
```

**This required a new environment.** `z_ref` mismatches the completion by rolling it one
position — and on an *unconditional* `TabEnv` a roll is a **permutation**, and the mean is
permutation-invariant, so the mismatch is a **no-op** (`−0.61677074380460184` vs
`−0.61677074380460173`, pure summation-order noise). Real KTO scores `y_{i+1}` under prompt
`x_i`: a different **set** of numbers, not a permutation. Hence `PromptEnv`, where the gap
becomes `0.020983` and the multisets genuinely differ. **The roll is load-bearing only when
the reward depends on the pairing.**

## The empty cell

`ref_mean × squared_target`, on a longer-wins ladder, 3000 steps:

```text
cell                                  m(T/2)      m(T)     incr    E[|y|]       KL
DPO      (ref_sum,  logsigmoid)       5.4154    6.5897   1.1744    4.9978   2.4150
IPO      (ref_sum,  squared)          1.6190    1.6190   0.0000    3.8282   0.6034
len-DPO  (ref_mean, logsigmoid)       3.8893    4.8564   0.9671    5.0000   2.5020
IPO_mean (ref_mean, squared)  <-      1.6190    1.6190   0.0000    4.8287   2.0178
```

Both squared cells stop dead; both logsigmoid cells keep climbing. Both squared cells converge
to the **identical margin 1.619048**, which hands us a matched-margin comparison for free — and
at that matched margin the mean reward lengthens further (`4.8287` vs `3.8282`), reproducing
`simpo`'s rule inside a cell no paper occupies. **The grid is a mechanism, not a taxonomy.**

## Implementation Notes

Bugs hit while building this, in order, and what each generalises to:

| Bug | What it generalises to |
|---|---|
| `lengths` vs `lengths[pairs[:,0]]` — 20 pairs vs 15 seqs | index one tensor by `pairs` and **every** tensor multiplied against it must be too |
| `max(t, 0).detach()` → `AttributeError: 'int'` | Python builtins use **object/sequence** semantics. Worse: at `π = π_ref` the mean is exactly `0.0`, the tie returns the Tensor, so it **passes step 0 and dies at step 1** |
| `torch.optim.Adam(...)` instead of plain GD | the only bug here that produces **plausible numbers** instead of a traceback. `margin ~ log t` is a theorem about GD; Adam erases the result while still printing a table |
| `spec.lr` (a per-arm learning rate) | `lr` is a **confound** in every comparison stage; it is a parameter of the function for exactly that reason |
| `lr * grad` where `grad` is `autograd.grad`'s tuple | `float * tuple` throws, but `int * tuple` **replicates the list** — `lr=2` silently builds a 2-element tuple. Unpack with `g, = ...` |
| `(r_w > r_l).mean()` on a Bool tensor | `.double()` first |
| hand-rolled `elen`, `ent`, `kl` — all three already existed as helpers | **grep before you write**; re-implementing what the file provides is a bug report about your reading |
| `E[|y|] = −10.93`, `H = −0.107`, `nll_w < 0` | **all three are outside the range of the quantity.** A length ∈ `[1, t_max]`, an entropy ≥ 0, a negative-log-prob > 0. One range check per metric catches all three without running a single step |
| `.double()` applied one line **above** the line that needed it | fix the line the traceback names, not the one near it |
| appends not wrapped in `float()` | converting at the boundary turns the conversion into an **assertion** — `float()` on a 20-element tensor raises, which is precisely the `nll_w` shape bug |

Harness bugs found by running a private reference implementation *before* handover:

- `prospect` asserted bounded on the **open** interval — false in float, since `σ(−50)`
  underflows to exactly `0`. Replaced with a closed interval at `|r| = 1e6`.
- `closed_form_optimum` mixed **units** — link-input `M` vs D3's log-ratio `Δ̂ = M/β`.
  `M* = 2.197225` but `Δ̂* = 7.324082`.
- `logsigmoid_margin` has **no closed form** (solved by bisection).
- `ref_mean`/`noref_mean` are **not** shift-invariant — only `_sum` has slope 1 for every sequence.
- **IPO at `lr = 2.0` → NaN** while DPO is fine at the same lr; an optimiser must never be the
  confound in a comparison stage.

## Failure Modes / Debugging Notes

- **`acc = 1.000` is the metric that reports nothing.** It saturated in 20 steps while the
  reward ran for 8000 more. Always log margin, KL, `E[|y|]`, entropy, and both marginals.
- **Never compare two objectives at equal step count** — that measures which trained longer.
  Use `stop_margin` and compare KL at a **matched margin**.
- **Run every ablation at more than one hyperparameter.** ORPO's odds term looks like the
  answer to P2 at β = 1 and is its own opposite at β = 0.25.
- **Length experiments must not be run to convergence** — both arms saturate at `t_max` and
  the effect vanishes entirely.
- **A bounded loss is not a finite optimum.** Check `inf ℓ`, and *where* it is attained.
- **Uniform reference is not a trust region.** Under SUM it is reference-free **plus a length
  bonus of `β log V` per token** — `0.549306` at `V=3`, and **`5.4125` at `V=50257`**. Under
  MEAN the constant cancels and it is bitwise reference-free (`1.110e-16`).

## Interview Answer

Prompt:

> **"DPO has known failure modes. Walk me through the variants and what each one actually
> buys you."**

Answer:

I'd start by refusing the paper zoo. There's one objective — `ℓ(f(y_w), f(y_l))` — and the
variants are choices of the implicit reward `f` and the link `ℓ`. I built the 5×5 grid; every
named method is one cell, reproduced elementwise to `0.000e+00`.

Three measured problems, and each axis fixes a different one.

**P1, the reward runs away on separable data.** This is Ford's theorem — the BT MLE is finite
iff the win digraph is strongly connected. On a matching it isn't, so DPO's margin grows like
`log t`: I measured `+0.7163` then `+0.7078` per *doubling* of the budget, with accuracy pinned
at 1.000 throughout. That's a **link** problem: `−log σ(d)` is monotone with its infimum at
infinity. Give the link an interior minimum and it stops. IPO's `(d−τ)²` lands on exactly
`2.0000 = (2f−1)τ`; cDPO's smoothing lands on `2.1972 = ln 9 = logit(1−ε)`. KL drops
`0.9511 → 0.4704 → 0.2332`.

**IPO's price is compression.** `(2f−1)τ` still depends on `f`, so the cap isn't uniform — but
the strong/weak ratio squeezes from `11.3330` to `4.9000`, a 2.31× loss of dynamic range. You
stop the runaway by refusing to express *how much* better the winner is.

**P2, the rejected logprob falls.** I measured `−0.1161` at convergence on a strongly-connected
fixture, so it isn't a divergence artifact. The cause is structural: four of the five links are
functions of `d = r_w − r_l` alone — I verified they're invariant to a common shift to `2e-16`.
A loss that can't see both logprobs falling can't prevent it. ORPO breaks the invariance twice,
with a non-affine odds map and an SFT term that isn't a difference at all. Both help
(`−0.131` and `−0.049` on chosen NLL) — but sweep β and the odds term **flips sign**
(`−1.899` to `+0.385`) while SFT holds at every β. Only one of the two is a mechanism.

**P3, length and reference confounds.** Here's where I'd push back hardest on the received
story: SimPO's `1/|y|` does not brake length, it **amplifies** it. At matched margin, on both a
longer-wins and a shorter-wins ladder, the mean reduction moved `E[|y|]` further than the sum
in all four cells. `d(r)/d(log π) = β/|y|` is largest at `|y| = 1`, so short sequences are the
cheapest lever and the optimiser buys margin at the short end — pushing length further in
whichever direction the labels point. What shortens SimPO is γ and reference-freeness, and
reference-freeness has its own bill: at matched margin `3.0` it costs `2.0042` KL against
`1.2091` and collapses entropy further, because a reference-free reward *is* `log π` and so
rewards likely text.

**And KTO is not a P1 fix.** Its loss is bounded — I verified it stays in `[0, 2.5]` at
`|r| = 1e6` — but a bounded loss is not a loss with a finite minimiser: `inf σ(−u) = 0` only at
`u = ∞`. On separable data KTO's margin climbed `+0.9294` per doubling next to DPO's `+0.7207`,
*faster*. What boundedness buys is bounded **influence** — a badly-wrong example's gradient
decays to `2e-9` where `−log σ`'s saturates at 1 — so KTO is a robustness method. With 4/40
labels flipped, DPO's clean margin fell `1.2411` against KTO's `0.6684`. Its real selling point
is unpaired data, and that's exact rather than approximate: the prospect link is a sum of two
independent per-example terms, so KTO on flattened pairs equals `0.5 ×` the paired loss to
1e-15.

If pressed on what to actually ship: 200k thumbs-up/down → KTO, then a small high-quality
preference set → IPO or cDPO if the annotators are near-deterministic, plain DPO if they
aren't. And instrument margin, KL, length, and entropy, because accuracy will read 1.000
through all of it.

## Reusable Artifact

- Path: `week-05-dpo-preference-optimization/code/variants.py`
- Run: `./.venv/bin/python week-05-dpo-preference-optimization/code/variants.py`
  - one stage: `... variants.py separable`
  - a few: `... variants.py reward link reduce`
  - fast fixtures: `... variants.py --fast`
  - list stages: `... variants.py --list`
- How to reuse: `named_specs()` returns DPO/cDPO/IPO/SimPO/ORPO/KTO as `Spec` objects, and
  `variant_loss(spec, ...)` is the whole grid in one call. To try a new method you write one
  `Spec` — you do **not** write a new loss. `train_variant(..., stop_margin=X)` gives the
  matched-margin comparison that every honest cross-objective KL claim needs.
