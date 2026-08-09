# Week 06 Day 01 — GRPO vs PPO: the group IS the critic

## Objective

**Read this first, out loud.** Week 4 built PPO and measured, in your own ablation, that
deleting the critic did **not** hurt: `no_critic 5.1786 ± 0.036` vs `full 5.1501 ± 0.109`
at horizon 5 — an effect of **0.0285** against `full`'s own spread of **±0.109**, i.e. a
**null**, which is the strongest form the argument can take. Week 5 measured that a preference dataset is a **matching**, so the loss is
exactly constant off the comparison set — and that on-policy regeneration is the only
structural fix. Week 6 is those two results built into one algorithm. GRPO is
**DPO's off-data fix plus Week 4's PPO machinery minus the critic**, and today you find out
exactly what that subtraction costs.

The day has one thesis:

> **GRPO does not delete the value function. It replaces a LEARNED estimate of `E[r|x]`
> with a MONTE-CARLO estimate of the same quantity — and unlike a critic, that estimate
> vanishes exactly where the policy succeeds.**

Five consequences, each provable and each measurable today:

**(a) The group mean is not "approximately unbiased". It is exactly unbiased in
direction, with a known scalar shrinkage.** For `Â_i = r_i − mean(r)` over an i.i.d.
group of size `G` drawn from the same prompt,

```
E[ (1/G) Σ_i Â_i ∇log π(o_i) ]  =  ((G−1)/G) · ∇J     EXACTLY
```

Not `O(1/G)` bias — an exact scale factor. So the leave-one-in baseline is a **learning-rate
rescaling**, not a bias. Corollary that should feel wrong the first time: at `G = 1` the
GRPO gradient is **identically zero**, for every prompt, at every θ. PPO with a critic
learns from one sample; GRPO learns from none.

**(b) Dividing by `std(r)` destroys (a).** `(r − mean)/std` is a ratio of *correlated*
random variables, so it is genuinely biased, and the bias is **question-dependent**: the
`1/std` factor upweights prompts whose pass rate is near 0 or near 1 — precisely the
prompts that carry the least information. Same for the `1/|o_i|` token-length
normalisation, which shrinks the *penalty* on long wrong answers. These are the two
Dr. GRPO removals, and you will measure both rather than cite them.

**(c) The group mean ANNIHILATES anything constant within a group.** Two lines of algebra.
Therefore a KL penalty folded into the per-token reward (InstructGPT / W4 style) is
**partially cancelled by its own baseline** under GRPO — only the deviation from the group
mean survives. That is *why* GRPO puts KL as a separate additive loss term with the **k3**
estimator instead of shaping the reward. You now have three weeks of k3 material pointed
at one engineering decision.

**(d) The dead-group law — the result that makes Days 3 and 4 necessary.** For a binary
verifiable reward with pass rate `p`, a group carries **exactly zero** gradient signal when
all `G` samples agree:

```
P(dead group) = p^G + (1−p)^G
```

At `G = 8`: **0.78%** at `p = 0.5`, **43.0%** at `p = 0.9`, **92.3%** at `p = 0.99`. The
effective batch size **collapses as the model gets good**. This is W5's *"`acc = 1.000` is
the metric that reports nothing"* wearing a different hat, and it is why RLVR needs a
curriculum rather than merely benefiting from one.

**(e) Check whether the clip does anything at all.** With `μ = 1` inner epoch and
on-policy sampling, `ρ_{i,t} = 1` **exactly** on the first minibatch, so
`min(ρÂ, clip(ρ)Â) = Â` and the clipping is inert. Many GRPO implementations run `μ = 1`.
So GRPO-with-`μ=1` is REINFORCE + a group baseline + a KL penalty, wearing PPO's notation.
**Measure it. Do not take my word for it.** This is the *"does this actually do anything?"*
move that you have now made unprompted on two consecutive days — today it is assigned.

Build `code/grpo.py` to make all five assertions rather than claims.

## Prerequisites

- **W3 D1–D3** — the EGLP lemma (`E[∇log π] = 0` ⇒ any state-only baseline is unbiased),
  the advantage form, and the variance ladder.
- **W4 D1** — Bradley-Terry, and the identifiability result: `r(x,·)` and `r(x,·) + c(x)`
  fit identically. Today you use the constructive half of it.
- **W4 D2/D3** — the PPO surrogate, the ratio `ρ`, clipping at three levels, the critic and
  GAE, and the memory bill of four networks.
- **W4 D5 / W5 D5** — `no_critic` costing nothing measurable at horizon 5 (a **null**, effect
  0.0285 vs spread ±0.109); ≥5 seeds for anything tail-sensitive.
- **W5 D1** — location invariance `r + c(x)`, which is the *licence* for a group baseline.
- W4/W5's staged-harness convention.

## Required Reading

- **DeepSeekMath (Shao et al. 2024) §4.1** — the GRPO objective, Eq. (3), and §4.1.2/§4.1.3
  on outcome vs process supervision. **Read §4.1 and stop at §4.1.4.** The iterative-RL
  paragraph is Day 5's material.
- **DeepSeekMath §5.2.2, "Towards to a Unified Paradigm"** — three pages that write SFT,
  RFT, DPO, PPO and GRPO as one gradient with a *data source*, a *reward function*, and an
  *algorithm*. This is the most interview-useful page in the paper.
- Optional, high-value: **Dr. GRPO (Liu et al. 2025, "Understanding R1-Zero-Like Training")**
  §3 — the two bias terms. Read it *after* you have derived (b) yourself, not before.
- Optional: **DeepSeek-R1** §2.2 (the RLVR reward design) — Day 2's reading, skim only.

## 2-Hour Plan

> **Clock note, unchanged.** Blocks are delivered **in full**, however long the session
> runs. Nothing is compressed for the clock. Ordering is W5's, which worked:
> numbers-card → cold recall → graded re-ask → concept → derivation → drills → coding →
> deliverable.

> **Rendering.** All maths in this file is plain Unicode. No LaTeX anywhere.

### Block 0 · 0:00-0:05 — Numbers-card drill (RECOGNITION, not retrieval)

New this week, on the W5 retrospective's recommendation. Five weeks of evidence say
recognition works where retrieval does not (D5: 45% recall while building a 92-check lab
correctly). So: **open `shared/numbers-card.md` and read the Week 4 and Week 5 sections out
loud, start to finish, before answering anything.** Then close it.

Three questions immediately after, closed-book:

1. What did `no_critic` score against `full` in the W4 D5 ablation, and what was the
   *spread* on each? What did you conclude, and what was the *scope condition* on that
   conclusion?
2. `var(k1)/var(k3)` at KL 0.004 and at KL 0.65. Which way does it go, and what does that
   do to "use k3, it has lower variance"?
3. W5 D5's β-sweep: margin went **up**, `Δ̂ = M/β` went **down**. Give both pairs of
   numbers, then say which one is the invariant.

### Block 1 · 0:05-0:20 — Cold recall, and THE EXPERIMENT OF THE DAY (skeleton-first ISSUED)

**Explicit instruction, issued once, on the record: for every multi-part answer below,
type the numbered skeleton FIRST with blank slots, THEN fill each slot.** That is D4's
intervention — it produced 3.7× on an identical question — and D5 showed it is not used
unprompted. So today it is *prompted here in Block 1* and then **deliberately withheld in
Block 2**. If Block 2 scores like Block 1, it has transferred and I stop issuing it. If
Block 2 collapses, it is a prompted technique and it gets issued every single session
forever. Either answer is worth having. **Do not use skeleton-first in Block 2 unless you
would have done so anyway** — that is the whole experiment.

1. **PPO's objective, written out**, then the four networks and which one can WRITE.
2. **The memory bill** for a 7B policy under bf16+Adam: trainable bytes/param, frozen
   bytes/param, the naive four-network total, and the three cuts. Numbers on the page
   before prose.
3. **Clipping is not a trust region, at three levels.** Say *token / sample / iteration*
   first, then one number for each. (Retired on W5 D1 as a concept — this is a numbers
   check now.)
4. **`dE[r]/dKL = ?`** and the SHAPE of the frontier, in the right variable. Then: name
   the *other* curve and say how it differs.
5. **Perennial, now four sessions overdue: the SOURCES of variance in REINFORCE.** Keep
   sources separate from mechanisms. You have never once named the policy's own
   **entropy**, the **environment's** stochasticity, or **off-policy-ness itself** — and
   for an LLM one of those three is *exactly zero*, which is the interesting part.
6. **The load-bearing coincidence** of W4 D1: `π_ref = π_SFT = the RM's training
   distribution`. State it, then the three consequences.
7. **W5's one-line thesis on what a DPO dataset is, graph-theoretically**, and what that
   implies about where the loss is constant.

> ## RESULT — the skeleton-first experiment resolved in Block 1, and the control is now void
>
> Block 1 issued skeleton-first in bold as a standing instruction. **It was not executed on a
> single one of seven answers**, and the block scored **10.0/18 (56%)**. Q6 — a 1+3 slot
> question, and the six-week-old second-clause gap — scored **1.0/2**.
>
> Re-asked ~30 minutes later with the format **enforced** (post the empty skeleton as its own
> artifact, then fill it): **4.0/4 (100%)**, all three consequences correct including Gao's law
> with `d = √KL`.
>
> ```
> not issued            (W5 D5)      45%
> ISSUED, not executed  (W6 D1 B1)   56%   <- behaviourally identical to withheld
> enforced re-do        (W4 D4)      93%
> ENFORCED              (W6 D1)     100%
> ```
>
> **The operative variable is not instruction. It is the physical act of emitting the empty
> slots before any content exists.** Being told to do it does not cause it to happen — that was
> tested today at full volume, in bold, directly above the questions.
>
> **STANDING RULE from here on: the skeleton is PRE-PRINTED in the question by me; he fills
> it.** No compliance step. Block 2's withheld-control arm is therefore void — the withheld
> condition was already measured in Block 1, and running it again would cost accuracy for zero
> information.

### Block 2 · 0:20-0:35 — Graded re-ask: the two drills that are load-bearing today

Skeletons are **pre-printed** (see the result box above; the control arm was retired mid-block).

1. **"When would you not use PPO?"** — re-asked from W5 D1, because today is its answer.
   PPO's bill; what DPO deletes and what it pays; what GRPO deletes and what it pays;
   **the one thing PPO gives that neither does**; and how you choose. Cite your own
   ablation and state its scope condition.
2. **"Why is a baseline allowed to be anything that does not depend on the action?"**
   State the lemma, prove it in one line, then the consequence that is specific to LLMs:
   which per-prompt constants are free, and *why they are free twice over* (once by EGLP,
   once by W4 D1's identifiability).

> **Grading rubric, unchanged.** Every multi-clause question scores as two line items —
> *first clause* and *everything after*. First clause only caps the item at 0.5. Sixth week
> running that clause two is where the points are.

### Block 3 · 0:35-1:00 — Concept: the group IS the critic

**Framing.** The standard telling is "GRPO removes the critic to save memory." That is
true and it is the least interesting thing about it. The useful framing is that PPO and
GRPO are **two estimators of the same object**:

```
the object:     b(x) = E_{o ~ π_θ(·|x)} [ r(x, o) ]        (a per-prompt baseline)
PPO's estimator: a learned V_φ(s_t), amortised across prompts AND across time
GRPO's:          mean(r_1..r_G), Monte-Carlo, this prompt only, this step only
```

Everything follows from the difference between *amortised-and-learned* and
*fresh-and-empirical*:

- **What GRPO gains.** No second trainable model (no params, no grads, no two Adam
  moments, no warm-up, no value-loss coefficient to tune, no `value_clip`, no deadly
  triad — W4 D4's `+1.07 diverges to 4.4e13` cannot happen to a mean). The estimate is
  never *stale* and never *wrong*: it is unbiased by construction at every step.
- **What GRPO loses.** The critic is `V(s_t)` — a function of the **state**, i.e. of the
  prefix. It differs token by token, so `A_t = r_t + γV(s_{t+1}) − V(s_t)` performs
  **intra-sequence credit assignment**. GRPO's `Â_i` is ONE number per sequence, broadcast
  to every token. GRPO says *"this whole rollout beat the group average"* and pushes every
  token of it equally, including the tokens that were irrelevant or actively bad.
- **What GRPO pays.** `G` rollouts per prompt instead of 1. So the trade is
  **training memory → inference compute**, and it is a good trade in 2024-25 hardware terms
  precisely because rollout is memory-bandwidth-bound and embarrassingly parallel (W1's
  ~1 op/byte decode result), while a co-resident critic is a hard memory tax on the
  training job. Say the trade in that form and you have said something an interviewer has
  probably not heard phrased that way.
- **Why the critic was droppable HERE specifically.** For a sparse terminal binary reward,
  `V(s_t)` is literally *"the probability this partial chain of thought ends in a correct
  answer"* — a quantity as hard to estimate as the task itself, over horizons of thousands
  of tokens, from a scalar at the end. The critic is not merely expensive on reasoning
  tasks, it is **badly posed**. Your W4 D5 measurement (`no_critic` = `full` at horizon 5, a null)
  is the *easy* end of the same statement; this is the hard end.

The bill, so the claim is arithmetic and not vibes (7B, bf16 + Adam, W1's 16 B/param
trainable and 2 B/param frozen):

```
PPO :  policy 112 + critic 112 + RM 14 + ref 14  = 252 GB
GRPO:  policy 112 +      —     + RM 14 + ref 14  = 140 GB     (−44.4%)
GRPO with the reference logprobs CACHED at rollout time    = 126 GB
GRPO on a verifiable reward (no RM at all, just a checker)  = 112 GB   (−55.6%)
```

Note the last line: **RLVR deletes the RM as well**, which is Day 2. The cumulative
deletion — critic, then reward model — is the actual story of 2024-25 reasoning
post-training, and both deletions are paid for in sampling.

Also worth stating precisely, because it is where people get sloppy: **GRPO's group is not
a "group of prompts". It is `G` completions of the SAME prompt.** The baseline is only
valid because everything in the group shares `x`. That is exactly W4 D1's `c(x)`.

### Block 4 · 1:00-1:25 — Derivation

Do these in order; each is short and each has a measurable consequence.

**(1) The exact shrinkage.** Let `o_1..o_G ~ π_θ(·|x)` i.i.d., `r_i = r(x, o_i)`,
`r̄ = (1/G) Σ_j r_j`, `Â_i = r_i − r̄`, and `ĝ = (1/G) Σ_i Â_i ∇log π_θ(o_i)`. Show

```
E[ĝ] = ((G−1)/G) · ∇J(θ)
```

Hint on the one step that matters: split `E[r_j ∇log π(o_i)]` into `j = i` and `j ≠ i`. For
`j ≠ i` independence factorises it and EGLP kills one factor **exactly**. Then read off the
`G = 1` case and say out loud what it means. Then: what does the **leave-one-out** baseline
`r̄_{−i} = (1/(G−1)) Σ_{j≠i} r_j` give, and why is that the "clean" version?

**(2) Annihilation.** Show that for any function `c(x)` that does not depend on `o`,
centring by the group mean removes it **exactly**, elementwise, at finite `G` — not in
expectation. Then apply it to `r_i − β·(per-token KL of o_i)` and say what survives.
Conclude why DeepSeekMath puts KL in the loss, not in the reward. Then the second half,
which is the part that is easy to miss: **the KL term in the loss is NOT annihilated,
because it is not inside the advantage.** Two placements of the same symbol, two entirely
different objects.

**(3) Where the `1/std` bias comes from.** For a binary reward, compute `mean` and `std`
over a group in closed form as a function of the number of successes `k`. Write
`Â_i = (r_i − mean)/std` explicitly for a success and for a failure. What happens as
`k → 1` or `k → G−1`? Now say which prompts get the largest gradient weight, and compare
that to which prompts carry the most *information* (Bernoulli variance `p(1−p)`, maximal at
`p = 0.5`). State the mismatch in one sentence — that sentence is Dr. GRPO's contribution.

**(4) The dead-group law and the effective batch.** Derive `P(dead) = p^G + (1−p)^G` and
then the **effective** number of contributing prompts `N_eff = N(1 − p^G − (1−p)^G)`. Solve
for the `G` needed to keep `N_eff/N ≥ 0.9` at `p = 0.95`. Then state the design consequence
for Day 4 in one line, and connect it to the W5 retrospective's rule about never quoting a
rate without a level.

**(5) Optional if time — the unified paradigm.** Write SFT, RFT, DPO, PPO, GRPO as
`∇J = E[ Σ_t GC(o,t) ∇log π_θ(o_t|·) ]` and fill in the *gradient coefficient* `GC` for
each. Five lines. This is DeepSeekMath §5.2.2, and it is the single best whiteboard answer
to "compare the post-training algorithms."

### Block 5 · 1:25-1:40 — Interview drills

**Skeleton-first is REQUIRED here** (Block 1 was the withheld condition; this is not).
Type the numbered skeleton with blank slots, then fill.

1. **"Walk me through GRPO. Why does it work without a critic?"** — 4 minutes. Must
   include: the object both algorithms estimate; the exact shrinkage and what it implies at
   `G = 1`; what is lost (name it precisely — it is not "variance"); the memory arithmetic;
   and the scope condition under which the critic is worth keeping.
2. **"You are seeing reward go up and eval accuracy flat on a GRPO math run. Debug it."** —
   this is the RLVR version of W4's Goodhart drill and it has a *different* first move,
   because the reward is verifiable. Give your ordered checklist, and put the check that
   distinguishes *"the verifier is being gamed"* from *"the model got better at the train
   distribution only"* first.
3. **"Design the group size."** — `G` against: gradient signal, dead-group fraction,
   rollout cost, and wall-clock. Give the formula, then a number for a realistic pass rate,
   then say what you would measure to tune it rather than guess.

### Block 6 · 1:40-2:30 — Coding: `week-06-grpo-rlvr-reasoning/code/grpo.py`

Staged harness, W4/W5 convention. Built on `TokenEnv(vocab=3, t_max=3)` — **15 enumerable
sequences, 7 states** — reused from `week-05-.../code/dpo.py`, so every claim about an
*expectation* is checked against **exact enumeration**, and every claim about a *gradient*
against a central finite difference. Sampling appears only where the point of the exercise
is that something is sampled.

TODOs (yours):

1. `group_advantage(rewards, mode)` — the three variants (`centered`, `normalized`,
   `loo`) with the edge cases that matter: `G = 1`, and a group where every reward is
   identical (`std = 0`). Decide what each mode must return there and defend it.
2. `grpo_surrogate(logp, logp_old, adv, eps)` — the clipped per-token objective, with the
   advantage broadcast over tokens. Watch the LEVEL: `adv` is per-*sequence*, `logp` is
   per-*token*. (Seven sightings of the wrong-level family. This is number eight waiting to
   happen.)
3. `kl_penalty(logp, logp_ref, kind)` — `k1` and `k3`, as a separate loss term.
4. **`diagnose_grpo(history)` — the DIAGNOSTIC TODO, and this one is deliberately yours.**
   The W5 retrospective's finding is that you write the mathematical core and delegate the
   scaffolding, so the *design* skills — thresholds with air on both sides, negative
   controls, a rule that is silent on a healthy run — are being taught rather than
   practised. Contract: return a list of `(name, detail)`; assert **silent on a healthy
   run**; fire on `dead_groups`, `kl_runaway`, and `entropy_collapse`. Every threshold must
   key on a **level**, not a change — W5 D5's M4 fixture was wrong for exactly that reason.

   > A fourth rule, `length_drift`, was in the design and was **cut during calibration**:
   > with `t_max = 3`, `E[|y|]` can only travel 1.985 → 3.0, and every run that learns
   > lengthens, so healthy (ratio 1.14) and broken (1.35) have no gap worth a threshold.
   > Cutting a rule the instrument cannot support is the correct move — and the harness
   > now asserts the *asymmetry* instead: a run can leave `π_ref` while staying stochastic,
   > but it cannot collapse to a point without leaving `π_ref`.

Provided (not yours): the env port, exact enumeration, the finite-difference checker, the
fixtures, and `train_grpo`'s optimiser plumbing.

Stages assert, at minimum: the `(G−1)/G` factor to ≥3 significant figures against exact
enumeration; annihilation to `< 1e-15`; the `μ=1 ⇒ clip inert` claim as an exact equality
and its **negative control** (μ=8 at the *same* lr ⇒ clip fraction strictly positive); the
dead-group law
against a simulation at ≥5 seeds; and the `1/std` bias with a **matched-margin** control so
the effect is not confounded with scale.

### Block 7 · 2:30-2:40 — Deliverable & next session

Fill in `deliverable.md`, add every measured number to `shared/numbers-card.md`, update
`progress.md` and `HANDOFF.md`. Day 2 is **RL with verifiable rewards**, which deletes the
RM the same way today deletes the critic.

## Checkpoint Questions

1. GRPO's group baseline is unbiased "up to a factor". State the factor exactly, prove it
   in two lines, and say what the factor is at `G = 1` and what that means operationally.
2. Why is the group mean specifically a *legitimate* baseline for an LLM, in two
   independent senses — one from Week 3 and one from Week 4?
3. A colleague folds a per-token KL penalty into the reward before computing group
   advantages. What exactly goes wrong, and how much of the penalty survives?
4. You have a 7B policy. Give the GB bill for PPO and for GRPO, then for GRPO+RLVR, and
   name the *one* line item that is free if you are willing to store something at rollout
   time.
5. Your GRPO run's mean reward is 0.97 and rising, and the gradient norm is falling toward
   zero. Give two competing explanations and the single measurement that separates them.
6. What does the critic buy that a group mean cannot, in one precise phrase — and name a
   task shape where that phrase is worth 112 GB.

## Frontier-Lab Interview Angle

GRPO is the single most likely 2025 post-training question, and the median answer is
"it removes the critic to save memory," which is a *recitation*. The three things that mark
a real answer:

- **Naming the estimand.** "Both estimate `E[r|x]`; one amortises it in a network, one
  samples it." That sentence reframes the whole comparison and it takes five seconds.
- **Knowing what is actually lost.** Not variance — *intra-sequence credit assignment*.
  Then immediately: and for a sparse terminal reward on a 4000-token chain, the critic that
  would provide it is badly posed, which is why nobody misses it.
- **Knowing the failure mode nobody mentions.** The dead-group collapse: your effective
  batch size falls as your model improves, so a static prompt set silently stops training
  you. That is the bridge to curriculum (Day 4) and to pass@k (Day 3), and it is the answer
  to "what would you monitor?" — the *fraction of groups with non-zero advantage*, which
  almost nobody logs.

Have a number ready for each. Your own `no_critic = full` **null**, with its horizon-5
scope condition and its 3-seed caveat stated unprompted, is the kind of thing that turns a recitation into a
project story.
