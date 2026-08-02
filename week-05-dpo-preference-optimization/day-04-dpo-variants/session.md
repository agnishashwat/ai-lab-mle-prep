# Week 05 Day 04 — DPO Variants: Each One Is an Answer to a Question You Already Asked

## Objective

The variant zoo is usually memorized. It should not be — by Wednesday this week you will have
derived, independently, the three problems that the entire zoo exists to solve:

| Problem | Where you found it | The variants that answer it |
|---|---|---|
| **The optimum is at infinity** (Ford; separable pairs) | D1 theorem, D2 measurement | **IPO** (bounded loss), **cDPO** (label smoothing), margin variants |
| **The implicit reward is extensive in length** | D3 derivation + slopes | **SimPO** (divide by \(|y|\)), Dr-GRPO's analogue in W6 |
| **You need paired data and a reference model** | D3 memory arithmetic | **KTO** (unpaired), **ORPO** (reference-free, one stage) |

So the day is not "here are four papers." It is: **take the three problems, propose the fix
yourself, then read what each paper actually did and score your proposal against it.** You
should be able to invent IPO before being told it exists — the derivation is two lines from
D1.

The output is a comparison table with **axes chosen by mechanism, not by paper**, plus one
harness that runs all of them on the *same* fixtures, including the separable fixture that
breaks DPO. A variant table where every row is "better on AlpacaEval" is worthless; a table
where every row says *which failure mode it removes and what it pays* is the artifact.

## Prerequisites

- D1 (existence, \(\varepsilon\)-cap, scale-\(\beta\)), D2 (loss, gradient, assumptions),
  D3 (length prior, reference logistics, cDPO's closed-form optimum).
- W4 D4's four-way failure taxonomy — the same **objective-wrong vs optimizer-broken** split
  applies here and should be reused explicitly.

## Required Reading

Skim, in this order, and time-box each to ten minutes — you are looking for **one equation
and one claimed problem** in each:

- **IPO** (Azar et al., "A General Theoretical Paradigm to Understand Learning from Human
  Preferences") — §4, the \(\Psi\)PO framing and the identity-mapping choice.
- **KTO** (Ethayarajh et al.) — the loss, and the prospect-theory value function.
- **ORPO** (Hong et al.) — the odds-ratio term and the combined SFT objective.
- **SimPO** (Meng et al.) — the length-normalized reward and the target margin \(\gamma\).

## 2-Hour Plan

### 0:00-0:08 — Cold recall

1. The three problems above, named without looking, with the D1/D3 result attached to each.
2. cDPO's optimal margin, closed form and value at \(\varepsilon=0.1\). **Number first.**
3. Your D3 length slopes, sum vs mean. Numbers first.
4. Ford's condition and the noiseless corollary.
5. Perennial: the REINFORCE variance sources still unnamed.

### 0:08-0:18 — Recall quiz (graded, D1-D3)

1. Where did the KL go in DPO, and on which distribution is the implied constraint evaluated?
2. Both logprobs fall: two independent causes, and the distinguishing experiment.
3. Why is DPO *more* exposed to length exploitation than PPO? Name the term PPO has that DPO
   does not, and say why it would not be enough anyway (W4 D4's bounded-resistance result).
4. `tilt(cr, cβ)`: the identity and the annotator-agreement consequence.
5. The three conditions for DPO ≡ RLHF, and which yours violates.

### 0:18-0:35 — Invent the variants before reading them

Ten minutes with no papers. For each of the three problems, propose a fix and predict its
side effect:

- **Unbounded optimum.** You have already seen two fixes (cap the label, cap the loss). Write
  down the "cap the loss" version: replace \(\log\sigma(\cdot)\) with something bounded.
  The simplest bounded choice is a **squared loss to a target margin**:
  \((\hat\Delta - \tau)^2\). Predict the side effect *before* reading IPO: a squared loss
  weights *over*-confident pairs as heavily as under-confident ones, so it will refuse to
  make any margin larger than \(\tau\) — which fixes divergence and simultaneously **caps how
  strongly the model can prefer anything**, including pairs where the preference is genuine
  and huge.
- **Length.** Divide by \(|y|\). Predict the side effect: the reward is now an *average*
  per-token drift, so a short completion with one very good token can beat a long one — and
  the loss no longer corresponds to any \(\log\pi\), so the DPO derivation no longer applies.
  SimPO is explicitly not a reparameterized RLHF objective; that is a real cost, not a
  technicality.
- **No pairs / no reference.** Two sub-problems. Unpaired: you have "good" and "bad" examples
  but not matched pairs, so you need a *per-example* reference point rather than a partner —
  predict that this must be some running or batch-level baseline (it is: KTO uses a KL term
  estimated from mismatched pairs in the batch). Reference-free: you must replace
  \(\log(\pi_\theta/\pi_{\text{ref}})\) with something that is already bounded relative to
  \(\pi_\theta\) itself — predict a *ratio internal to the policy*, e.g. odds. (It is: ORPO.)

Then read, and score each prediction. **The grade is on the prediction, not the recall.**

### 0:35-0:58 — Concept: the four variants, by mechanism

- **IPO.** \(\Psi\)PO with the identity mapping ⇒
  \(\big(\hat\Delta - \frac{1}{2\tau}\big)^2\). Two things to be precise about, because they
  are the interview content: (i) IPO's stated motivation is that
  \(\Psi=\log\frac{p}{1-p}\) (which is what DPO implicitly uses) is **unbounded as
  \(p\to1\)**, so deterministic preferences produce infinite targets — **this is D1's theorem
  in the paper's own language**, and you can say so; (ii) IPO regularizes toward
  \(\pi_{\text{ref}}\) *properly*, i.e. its optimum stays at finite KL for any dataset,
  whereas DPO's does not. The cost: a squared loss is symmetric, so IPO **actively pushes
  down** margins that are too large, and it needs \(\tau\) tuned per dataset.
- **KTO.** Drops pairing. Value function from prospect theory —
  concave in gains, convex in losses, with a **reference point** — applied to
  \(\beta\log(\pi_\theta/\pi_{\text{ref}}) - z_{\text{ref}}\), where \(z_{\text{ref}}\) is a
  batch-estimated KL term. Two consequences: (i) it can use the far larger pool of
  thumbs-up/thumbs-down data; (ii) the loss is **asymmetric** — losses are weighted more than
  gains — so it is deliberately more aggressive about avoiding bad outputs than about
  producing good ones. Say why that is a *product* decision encoded in a loss.
- **ORPO.** No reference model *and no separate SFT stage*: minimize
  \(\mathcal L_{\text{SFT}} + \lambda\mathcal L_{\text{OR}}\) where the odds-ratio term is
  \(-\log\sigma\!\big(\log\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\big)\).
  Read it against D3: the SFT term is exactly the missing constraint on the **marginal** that
  Day 2 identified, which is why ORPO does not exhibit the both-logprobs-fall pathology. The
  cost: no \(\pi_{\text{ref}}\) means **no trust region at all** — the anchor is the SFT term
  on the chosen completions only.
- **SimPO.** \(\hat r = \frac{\beta}{|y|}\sum_t\log\pi_\theta(y_t|\cdot)\), reference-free,
  plus a target margin \(\gamma\). Two ideas at once — **length normalization** and a
  **margin** — and it is worth insisting on separating them, because the margin is a
  divergence fix (problem 1) and the normalization is a length fix (problem 2), and the paper
  bundles them. Cost: reference-free means the implicit reward is \(\log\pi_\theta\) itself,
  so the objective now rewards *likely* text, and likely text is short and generic — the
  \(\gamma\) margin is doing more work than it appears to.
- **The unifying view worth writing down.** Every variant is
  \(\ell\big(f(\pi_\theta,\pi_{\text{ref}}, y)\big)\) for a choice of (i) **implicit reward
  \(f\)** — with/without reference, sum/mean over tokens — and (ii) **link \(\ell\)** —
  unbounded log-sigmoid, bounded square, asymmetric prospect value, plus optional margin and
  label smoothing. That 2-axis grid is the table; the papers are points in it, and there are
  empty cells you can name.

### 0:58-1:10 — Interview Drill

1. **"When would you use IPO over DPO?"** Wants the *theorem*, not a benchmark: deterministic
   or near-deterministic preferences, i.e. high annotator agreement / separable pairs — and
   the observation that this is the case where DPO looks *best* on training metrics. Then
   what IPO costs.
2. **"You have 200k thumbs-up/thumbs-down events from production and 5k curated pairs. What
   do you train?"** Wants KTO on the 200k (and *why* the unpaired data is usable at all),
   the 5k as evaluation or as a DPO/IPO stage, the distribution difference between the two
   sources, and an explicit statement of what the thumbs data is *actually* a label of.

### 1:10-1:50 — Coding: `code/variants.py`

**One harness, one set of fixtures, all five losses** (DPO, cDPO, IPO, ORPO-style,
SimPO-style), on the enumerable environment so every claim is exact.

**TODOs:**
1. `implicit_reward(kind, ...)` — the \(f\) axis: `{ref_sum, ref_mean, noref_sum, noref_mean,
   odds}`.
2. `link(kind, delta, ...)` — the \(\ell\) axis: `{logsigmoid, logsigmoid_smoothed,
   squared_target, prospect, logsigmoid_margin}`.
3. `variant_loss(f_kind, l_kind, **params)` — the grid, so every named method is one line and
   the empty cells are reachable.
4. `train_variant(...)` and `compare(fixtures, variants)` returning a tidy table.

**Checks with teeth:**
- **Named methods reduce correctly.** `(ref_sum, logsigmoid)` equals D2's `dpo_loss`
  elementwise to \(10^{-15}\); `(ref_sum, logsigmoid_smoothed)` equals D3's cDPO; and each
  named variant matches an independently-written direct implementation. This is the
  chained-regression trick again (W4 D5) — *the grid must reproduce the specials*.
- **The separable fixture, the day's headline.** On D1's perfect matching: assert DPO's exact
  KL diverges, and that IPO's and cDPO's **converge to finite, predictable values** —
  \(\tau\)-determined and \(\varepsilon\)-determined respectively, both compared to closed
  form. Report the KL of each at a fixed step budget in one column.
- **IPO's symmetric cost, measured.** Construct a fixture with one genuinely huge preference
  and many small ones; assert IPO's converged margin on the huge pair is **capped** at the
  same value as on the small ones, while DPO's is not. This is "the fix has a cost" as a
  number rather than a caveat — the W4 D5 honest-caveat habit, applied.
- **ORPO does not exhibit both-logprobs-fall.** On D3's \(\mu\ne\pi_{\text{ref}}\) fixture,
  assert chosen logp *rises* under ORPO and *falls* under DPO, same data, same steps. Then
  assert DPO **with an added NLL term** also rises — isolating the SFT term as the mechanism
  rather than the odds ratio.
- **SimPO's length behaviour and its price.** Assert mean length is flat under length
  normalization where DPO's grows (reusing D3's slopes as the control), **and** assert the
  reference-free reward's optimum has lower entropy / shorter outputs on a fixture with no
  length confound at all — i.e. show the side effect, not only the benefit.
- **Empty-cell probe.** Pick one grid cell no paper occupies (e.g. `ref_mean` +
  `squared_target`) and report it in the table. It is the cheapest possible demonstration
  that the grid is real and not a post-hoc taxonomy.

### 1:50-2:00 — Deliverable & Next Session

`deliverable.md`: the 2-axis grid, the comparison table with **failure-mode-removed** and
**cost-paid** columns, and the separable-fixture KL column. Next: **Day 5 — failure modes and
the week's trainer**, with \(\beta\) and data-quality ablations, ≥3 seeds, and a `diagnose`
for DPO built from this week's mechanisms.

## Checkpoint Questions

1. Give the 2-axis grid and place all five methods in it. Name one empty cell and predict its
   behaviour.
2. IPO's motivation, in the language of D1's theorem. What does it cost, and on what kind of
   pair does the cost show up?
3. Why does ORPO not show the both-logprobs-fall pathology? Which term is responsible —
   prove it by ablation, not by argument.
4. SimPO bundles two independent fixes. Name them, say which problem each solves, and give
   the side effect of the reference-free choice.
5. You have unpaired feedback. What can KTO do that DPO cannot, and what is the thumbs-down
   signal actually a label of?

## Frontier-Lab Interview Angle

"Compare DPO to IPO/KTO/ORPO/SimPO" is asked constantly and is almost always answered as a
benchmark recitation. The answer that lands is the mechanism grid: two axes, five points,
each variant introduced as *the fix for a named failure* with an explicit price. Being able to
derive IPO's motivation from Ford's condition — and to note that DPO's training metrics look
*best* exactly when IPO is most needed — is the kind of answer that ends the topic.
