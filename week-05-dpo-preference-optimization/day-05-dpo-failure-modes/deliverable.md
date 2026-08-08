# Deliverable — Week 05 Day 05: DPO Failure Modes

**Artifact:** `week-05-dpo-preference-optimization/code/dpo_lab.py` — nine stages,
**92 checks**, 0 failures. Full run 65.6 s, `--fast` 34 s (89 checks).

## Core Takeaway

> DPO's failure modes are not bugs in the optimiser. They are consequences of what the
> **data format** can and cannot identify, and the training loss is **exactly constant**
> along every direction the data do not constrain. So the loss going down is not evidence
> of anything, and the metric everyone watches — pairwise accuracy — saturates in **1 step
> of 6000** and then reports nothing for the remaining 5999.

The four modes, each a **fixture** rather than an anecdote: one dataset function, four
knobs (`shape`, `eps`, `length_confound`, `n_pairs`), and every mode is one knob.

| # | Mode | Cause | Knob | Instrument that sees it |
|---|---|---|---|---|
| M1 | divergence / rejected collapse | BT MLE has no finite solution | `shape`, `eps` | `margin` vs `cap`, `logp_l` |
| M2 | both logprobs fall | gradient is a **difference**; levels are free | — (default DPO) | `logp_w` **and** `logp_l` |
| M3 | length exploitation | annotator ranks by `gold + κ·\|y\|`, gold does not | `length_confound` | `elen` **and** `gold` |
| M4 | off-distribution drift | loss is constant off the comparison set | `n_pairs` | `offdata_mass` |

## The result the week was built to reach

**M1 is a property of the DPO data format, not of DPO.** The Bradley–Terry MLE is finite
iff the *wins digraph* is strongly connected. Measured over 12 seeds:

```text
P(wins-digraph strongly connected)
  matching   (degree  1)  eps 0.00/0.10/0.25/0.35 -> 0.00 / 0.00 / 0.00 / 0.00
  cycle      (degree  2)  eps 0.00/0.10/0.25/0.35 -> 0.00 / 0.00 / 0.00 / 0.00
  all_pairs  (degree 14)  eps 0.00/0.10/0.25/0.35 -> 0.00 / 0.58 / 0.83 / 1.00
```

Two things fall out, and the second is the one worth carrying:

1. **Clean data are never strongly connected at any degree.** Orienting *any* graph by a
   total order gives a DAG. Noise is the only escape hatch.
2. **Noise needs density to act on.** In a matching, every completion appears in exactly
   one comparison, so no relabelling can create a cycle — annotator noise has *nowhere to
   act*. Verified directly: **2 of 7 labels flipped and it was still a DAG.**

**A real DPO dataset is the top row.** One prompt, two completions, one label. So the
finite-margin regime is not something you tune into; it is structurally unavailable, and
the margin diverges as `log t` forever.

The counter-intuitive corollary, measured 6/6 seeds at ε = 0.10:

```text
seed 0:   2 pairs -> 6.0470     7 pairs -> 4.1916   105 pairs -> 1.6687
seed 5:   2 pairs -> 6.3915     7 pairs -> 4.0515   105 pairs -> 1.6952
```

**More comparisons ⇒ SMALLER final margin.** Density creates the cycles that pin the MLE
down. (Step-by-step monotonicity along the full ladder holds in only 3/6 seeds — assert
the endpoint ordering, which is robust, not the fragile version.)

## Key Equations

```text
r̂(y)  =  β · [ log π_θ(y) − log π_ref(y) ]          the implicit reward
M     =  mean_k [ r̂(y_w) − r̂(y_l) ]                 the margin, in LINK units
Δ̂     =  M / β                                       the same thing in POLICY units
cap   =  logit(1 − ε)  =  log((1−ε)/ε)               and +inf when ε = 0
```

`cap` is the **whole** early-stopping rule. It is a statistical statement about the
*annotators*, so it needs **no held-out set**: exceeding `logit(1−ε)` means the model is
more confident than the labels can justify.

The identity the week's central experiment tests:

```text
tilt(c·r, c·β)  =  tilt(r, β)        the policy is identified only by r/β
```

## The β/ε collapse — the week's designed prediction

If the policy sees only `r/β`, then a β-sweep and an ε-sweep that land on the **same**
`Δ̂* = logit(1−ε)/β` must produce the **same KL**. Two structurally independent sweeps,
matched only through the identity:

```text
Δ̂* = 1.0986    beta-sweep KL 0.071109  vs  eps-sweep KL 0.071013   |diff| 9.59e-05
Δ̂* = 2.1972    beta-sweep KL 0.189279  vs  eps-sweep KL 0.189214   |diff| 6.56e-05
Δ̂* = 4.3944    beta-sweep KL 0.388616  vs  eps-sweep KL 0.388616   |diff| 2.78e-16
Δ̂* = 8.7889    beta-sweep KL 0.680533  vs  eps-sweep KL 0.680780   |diff| 2.47e-04
```

And the collapse is **non-trivial**: KL varies `0.0711 → 0.6805` across the sweep, so this
is not the statement that everything is equal.

### An unplanted result: steps-to-target ∝ 1/β²

```text
steps to 0.995·M*:  beta=2.0    76   beta=1.0   307   beta=0.5  1237   beta=0.25  4978
ratios: 4.04, 4.03, 4.02
```

**Halving β costs 4× the compute, not 2×.** The derivation: the gradient scale is ∝ β and
the distance to travel is ∝ 1/β, so time ∝ 1/β². The target `Δ̂*` only went up 8× while the
step count went up 65×. This is a practical statement — small β is not "gentler training",
it is quadratically more expensive training.

## The instrument panel, in PAIRS

The design rule: **each pair discriminates a failure the other half cannot see.**

| Pair | Sees |
|---|---|
| `(logp_w, logp_l)` | M2. **Never the margin alone** — it looks identical under "learned the preference" and "destroyed y_l" |
| `(r_absmax, cap)` | M1 causally, with **no held-out set** |
| `(kl, acc)` | M1 silently: `acc` hits 1.000 at **step 1 of 6000** while the margin runs `0.0245 → 5.8439` |
| `(elen, gold)` | M3 — raw win rate cannot answer, because length correlates with quality |
| `(onpolicy_r, data_r_mean, offdata_mass)` | M4 — the **only** instrument, and exactly what the training loop does not compute |

Why the margin needs a partner, measured:

```text
log pi(y_w) -3.1862 -> -3.3715 (-0.1852)  AND  log pi(y_l) -3.8037 -> -4.2863 (-0.4826)
the margin rose +0.5947 and told you neither
```

Both went **down**. The margin is a difference; the DPO gradient constrains only the
difference, so the common level is a free direction. Adding the SFT term fixes it —
`logp_w −0.1852 (DPO) → +0.0213 (DPO+SFT)` — which is precisely why ORPO and RPO carry one.

## The proxy/gold inversion, with a CONTROL

```text
                       train acc      gold peak            gold end   cost of shipping last
uncovered (M4)         0.000 -> 1.000  +0.6737 @   0/6001   +0.2424      0.4313
length confound (M3)   0.000 -> 1.000  +0.7571 @   2/6001   -0.4532      1.2103
separable (M1)         0.000 -> 1.000  +0.5054 @  77/6001   +0.4018      0.1036
CONTROL: healthy       0.000 -> 0.600  +0.4663 @1270/6001   +0.4663      0.0000
```

The control is the load-bearing row. Without it, "gold peaks early" is a statement about
training length; with it, it is a statement about the **data**. The healthy run peaks at
step 1270 and gives back **exactly 0.0000**.

And the sharpest single line of the week:

> **M3 gold `+0.7514 → −0.4532`.** DPO training made the model *worse than the reference it
> started from* — at 100% train accuracy, with the loss going down the whole way.

## Early stopping is an ESTIMATOR, not regularisation

`select_checkpoint` **never sees gold**. It gets the training panel and nothing else.
Three candidates; only one survives:

- **epoch count** — no units, and accuracy hits 1.0 at step 1 of 6000. Rejected.
- **KL** — necessary but not sufficient. It is **bounded** by `max_y −log π_ref(y) = 6.3216`
  in this environment, so it saturates while the reward diverges; and M4 is invisible in it.
  Rejected as primary.
- **implicit-reward magnitude against its cap** — monotone under GD, computed from the
  training batch alone, and the cap needs no held-out set. **This one.**

```text
gold: selected(step 105) 0.4963 | last 0.3300 | oracle(step 0) 0.6737
margin[105] = 2.2050 vs cap 2.1972          shipping last costs 0.3437
```

It recovers **0.1663 of the 0.3437** that shipping the last checkpoint throws away, from
the training panel alone. And on clean data it returns the last index — because ε = 0 gives
`cap = +inf` and there is **nothing to stop against**. That is not a limitation of the code;
it is M1.

## Implementation Notes

- **`History` is a record, not an accumulator.** `ndarray` has no `.append`, and `np.append`
  reallocates the whole array every call. Accumulate in Python lists inside a
  `@torch.no_grad()` **closure**, then `np.asarray` once. Without `no_grad` the panel
  retains a live autograd subgraph per logged tensor.
- **Two different ε live in this file.** `DataCfg.eps` is annotator flip rate (sets the cap);
  `RunCfg.eps` is cDPO label smoothing. Reading the wrong one makes the cap move with the
  training config, so a β-sweep silently sweeps its own target.
- **`logp_w`/`logp_l` are PLAIN logprobs, not ratios.** If logged as ratios,
  `logp_w − logp_l` equals `margin/β` to **1.39e-16** — the pair would be perfectly
  redundant with the margin it exists to check. Plain logprobs carry the *levels* the
  difference hides. Anchor: uniform over 15 sequences is `log(1/15) = −2.7081`.
- **The margin must route through `implicit_reward(spec.f_kind, ...)`.** Hard-coding
  `β·(log-ratio_w − log-ratio_l)` is `ref_sum`, correct for DPO/cDPO/IPO/KTO and silently
  wrong for ORPO (`odds`), SimPO (`noref_mean`) and IPO_mean (`ref_mean`).
- **`acc` is a rate, not a count** — `(r_w > r_l).double().mean()`. torch refuses `.mean()`
  on a bool tensor where numpy allows it.
- **`select_checkpoint` uses `np.nonzero(...)[0]`, not `np.argmax`.** `argmax` on an
  all-False array returns 0, silently claiming "ship the initial checkpoint" when the cap
  was never reached.
- **`rel()` — the ported W4 D4 bug.** Its old form divided by `max(|x[0]|, 1e-9)`, and at
  θ = θ_ref the implicit reward is *exactly* 0.0, so `margin[0]` is exactly 0.0 in **every**
  DPO run and it returned `1e9·x[-1]`. The fix for a zero denominator is the **right**
  denominator, not an epsilon on the wrong one: divide by `max_t |x_t|`, which bounds the
  result in `[−2, 2]`.
- **Fixture orientation is randomised** in `raw_edges`. On D1 `all_pairs` emitted `(i,j)`
  with `i > j`, which made "return column 0" a *correct* implementation on two of three
  fixtures. Randomising makes the non-vacuity check meaningful: raw column 0 now scores
  **0.5143** against the labels.

## Failure Modes / Debugging Notes

- **A diagnostic that always fires is not a diagnostic.** `diagnose_dpo` is asserted
  **silent** on a healthy run, and M3/M4 are asserted not to fire on the fixtures they do
  not own. W4 D4's −20% entropy threshold sat *inside* the range a healthy run occupies.
- **Build rules from mechanisms, not thresholds.** M1 fires on a *conjunction* — accuracy
  saturated **and** margin still rising **and** `logp_l` still falling, all over the run's
  second half — not on "the margin is large". On clean data the cap is `+inf` and a large
  margin is the correct answer to a badly-posed question.
- **M4 is a LEVEL, not a change.** Off-data mass *falls* during training (`0.4983 → 0.1943`)
  because pushing losers down moves mass onto covered winners, so a delta rule never fires.
  Support is a level. With `n_pairs=6` of 15 sequences, **19.4% of the model is unconstrained
  by the objective** and no field of the training panel reports it.
- **A length move alone is not a failure.** A healthy DPO+SFT run here moves `E[|y|]` by
  **+8.9%**. M3's gate is `elen` rise > 0.25 **and** gold falling from its peak — a threshold
  with **2.8× air below (healthy 0.089) and 4.4× above (fixture 1.110)**.
- **Correction to W4 D4: "seed spread grows with pressure" was a 3-SEED ARTIFACT.**

  ```text
  3 seeds spread(gold) 0.041696 @10 -> 0.253824 @6000 =  6.09x
  5 seeds spread(gold) 0.244638 @10 -> 0.253824 @6000 =  1.04x
  ```

  Same runs. Two extra seeds flip "optimisation amplifies seed noise 6×" to "it does not
  move it at all". **≥3 seeds is a FLOOR, and it is not enough for a tail-sensitive metric.**
  On the same runs, `margin` spread *contracts* to exactly 0.000000 and `kl` spread grows
  **71×** — KL is measured against each seed's *own* π_ref, so the further you travel the
  more the paths disagree.
- **Pairing is a change to the DESIGN, not to the code.** Varying `DataCfg.seed` and
  `RunCfg.seed` *together* shrinks the worst KL spread **2.07×** (0.7511 → 0.3621).
- **Report the effect against the noise, or do not report it.** On the very same runs:
  KL effect 0.4038 vs worst spread 0.3621 — **intervals overlap, so "β lowers KL" is not
  supported.** Margin effect 0.1201 vs spread 0.0282 — **4.3×, resolved.** The *metric*
  decides whether an ablation has power.
- **And the units decide the conclusion.** Across the same β sweep, `margin` goes
  **0.4746 → 0.5947 (up)** while `Δ̂ = margin/β` goes **1.8984 → 0.2974 (down)**. Report the
  wrong one and you report the opposite result.

## Interview Answer

Prompt:

> Your DPO run shows training loss falling smoothly and pairwise accuracy at 99%. The
> product team says the model got worse. Where do you look?

Answer:

Neither of those numbers can distinguish a good run from a broken one, and I would say so
first. Accuracy saturates almost immediately — I measured it hitting 1.000 at **step 1 of
6000**, after which the margin ran from 0.02 to 5.84 and accuracy reported none of it. Loss
falling is what a diverging margin *looks* like, because on the data DPO is trained on the
BT MLE has no finite solution.

So I would ask four questions, in this order, because they have different fixes.

**One: is the margin past the cap?** The implicit reward is
`r̂ = β·log(π_θ/π_ref)` and, under ε-noisy annotators, the largest margin the labels can
justify is `logit(1−ε)`. That is a statement about the *annotators*, so I can compute it
with no held-out set. If the mean margin is 2× that, the model is fitting past what the
data support. The fix is early stopping — and I would checkpoint **on the margin against
its cap**, not on epochs and not on KL. Not epochs because accuracy saturated at step 1; not
KL because KL is *bounded* in this setting (I measured the bound at 6.32) so it saturates
while the reward diverges, and because one of the four modes is invisible in it entirely.

**Two: did both logprobs fall?** The DPO gradient constrains only the *difference*, so the
common level is free and can drift down. I measured `log π(y_w)` falling −0.185 while the
margin rose +0.595 — the model got better at the comparison and worse at the thing you ship.
The instrument is `logp_w` and `logp_l` as **levels**, logged next to the margin. The fix is
a term outside the pairwise difference: an SFT/NLL term on the chosen response, which is
what ORPO and RPO carry and what turned −0.185 into +0.021 in my run.

**Three: did length move?** If annotators prefer longer answers even slightly, DPO will
learn length, because length is the cheapest direction that satisfies every comparison. A
length move alone is not proof — a healthy run in my setup moved `E[|y|]` +8.9% for
unrelated reasons — so the signature is length rising *together with* gold falling. This is
the one place a held-out gold eval genuinely earns its cost, because this is the
**objective-wrong** mode and only something outside the objective can say the objective is
wrong.

**Four: where is the model putting mass?** This is the one people miss and it is specific to
DPO. The loss is a function only of the sequences that appear in a comparison; it is
*exactly constant* everywhere else. In my run 19.4% of the trained model's probability mass
sat on completions no comparison ever touched — the loss returns the same value wherever
that mass lands, and no field of the training panel reports it. You need on-policy samples
scored by something, or you are blind on that fraction of the model.

The framing I would land on: **the first is the optimiser going too far, and you stop
earlier. The third is the objective being wrong, and stopping earlier does not help — you
have to change what you are optimising.** They look identical on a loss curve. I have a run
where gold went `+0.75 → −0.45` — worse than the reference it started from — at 100% train
accuracy with the loss decreasing monotonically the whole way.

## Reusable Artifact

- **Path:** `week-05-dpo-preference-optimization/code/dpo_lab.py`
- **Run:** `python dpo_lab.py` (92 checks, 65 s) · `--fast` (89 checks, 34 s) ·
  `--list` · or a stage name: `port dataset panel chain modes checkpoint collapse seeds inversion`
- **How to reuse:**
  - `make_dataset(env, DataCfg(...))` — one function, four knobs, every failure mode as a fixture
  - `train_dpo(env, data, RunCfg(...)) -> History` — the 14-field instrument panel, with
    `stop_margin` for **matched-margin** comparisons across objectives
  - `diagnose_dpo(h) -> [(mode, reason)]` — mechanism-based, silent on healthy runs, and every
    reason string **quotes the number that fired it**
  - `select_checkpoint(h) -> int` — early stopping from the training panel alone, no gold
  - `ablate(env, dcfg, rcfg, grid, seeds)` — dispatches on `__dataclass_fields__` so one
    function sweeps both `DataCfg` and `RunCfg` knobs; returns `(mean, spread)` per metric
