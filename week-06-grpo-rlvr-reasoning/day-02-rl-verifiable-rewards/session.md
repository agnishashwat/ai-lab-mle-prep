# Week 06 Day 02 — RL with Verifiable Rewards: a verifier is a SET, not a rate

## Objective

**Read this first, out loud.** Yesterday deleted the critic and priced it: 252 → 140 GB, paid
for in rollouts, and the bill grows as the model improves (`P(dead) = p^G + (1−p)^G`, **66.34%**
at p = 0.95, G = 8). The last line of yesterday's memory table was **140 → 126 → 112 GB**, and
that last cut — deleting the reward model — is today. It is also, in GB, the *smallest* cut on
the table: **14 GB out of 252**. So if the motivation for RLVR were memory, RLVR would not be
interesting.

The day has one thesis, and it is not about memory:

> **A verifier used as an EVAL is summarised by a RATE. A verifier used as a REWARD is
> summarised by a SET.** Optimization does not sample a checker's errors in proportion to their
> probability — it *searches* for them. So the average-case accuracy that makes an eval
> trustworthy tells you almost nothing about whether the same checker survives being optimized
> against. The only property that survives is **soundness** (accept ⊆ correct); everything else
> you can measure about a verifier is a statement about **compute**, not correctness.

That splits the day cleanly in two, and this split is the thing to carry out of today:

```
INCOMPLETENESS  (false negatives, correct⊄accept)  costs you COMPUTE
UNSOUNDNESS     (false positives, accept⊄correct)  costs you the RUN
```

Five consequences, each provable and each measurable today:

**(a) False negatives are priced in GPU-hours, via yesterday's law.** A verifier that misses a
correct answer α of the time does not "add noise": it multiplies the observed pass rate,
`p_obs = p(1−α) + (1−p)γ`, and `p_obs` is the *only* input to the dead-group law. At `p = 0.10`,
an answer-extraction failure rate of **α = 30%** moves `p_obs` to **0.07** and the group size
needed for 90% alive from **G = 22 to G = 32**. **A regex costs 45% more rollout compute.** It
does not show up as a bug; it shows up as a budget.

**(b) False positives are not diluted by their rate — selection concentrates them.** Among the
samples that get rewarded,

```
P(hack | rewarded) = γ(1−p) / [ p(1−α) + γ(1−p) ]
```

At a **1%** false-positive rate and 10% false negatives, that is **1.10%** at p = 0.5, **9.09%**
at p = 0.1, and **35.25%** at p = 0.02. GRPO learns *only from alive groups*, and at low p an
alive group is usually one success against seven failures — so at the frontier of a curriculum,
**a third of your gradient direction is the hack**, at a verifier whose reported accuracy is
99%. The curriculum of Day 4 drives p down *deliberately*. This is the mechanism by which
"make the problems harder" and "the model started cheating" are the same event.

**(c) Dead groups are a Rényi entropy of the REWARD's output distribution, and reward design
is the cheap knob.** Yesterday's binary law generalises in one line: for a reward taking values
with probabilities `q_1..q_L`,

```
P(dead group) = Σ_j q_j^G  =  exp( −(G−1)·H_G(q) )        H_G = Rényi entropy of order G
```

Binary is `L = 2`. At `p = 0.05` (a hard problem), binary gives **66.34%** dead at G = 8.
Splitting the *failure* mass into "got the setup right" (0.40) and "nothing" (0.55) gives
**0.90%** — a **73× reduction from adding ONE reward level**, and matching it by raising G
instead would need **G = 92**, an **11.5×** rollout bill. The asymmetry is the design rule:
**splitting the mode is what helps.** At p = 0.05 you must split the failures; at p = 0.95
splitting the failures does nothing and you must grade *among the successes*.

**(d) Under GRPO, a reward component's influence is proportional to the model's INCONSISTENCY
on it — not to its coefficient.** This is yesterday's annihilation result (`c(x)` removed
exactly, max dev 2.220e-16) pointed at reward *design*. For a format bonus with compliance
rate `c`, the within-group variance is `c(1−c)`, so the component's share of the centered
gradient peaks at `c = 0.5`, is **25×** down at `c = 0.99`, and is **exactly 0** at `c = 1` —
for **any** λ. Composite RLVR rewards are therefore **self-annealing curricula, for free**, and
"the format reward was essential" and "the format reward does nothing" are both true, thirty
minutes apart. The corollary is a trap: once compliance is uniform, **you cannot fix that
component by raising λ.**

**(e) The soundness theorem, which is the one thing no learned RM can have.** If `V ⊆ C`
(accept implies correct), then for **every** policy, at **every** KL, `gold_pass ≥ v_pass` —
the optimization gap is bounded by zero, forever, with no scope condition. If `V \ C` is
non-empty, the worst case is **1**, and it is 1 whether that set holds one sequence or
fourteen. **Reward hacking is a support property, not a rate property.** Compare a learned RM,
where W5 D1/D5 already proved the failure is not even noise — it is *underdetermination*: the
BT loss is **exactly constant** along every direction the comparison graph does not cover, so
the gold-vs-proxy gap at convergence is a property of the **initialisation**, not of the data.

Build `code/rlvr.py` to make all five assertions rather than claims.

## Prerequisites

- **W6 D1 (yesterday), all of it** — the dead-group law, exact annihilation, the shrinkage
  `(G−1)/G`, `1/std` amplification, the clip being inert at μ = 1, and `grpo.py` as an
  importable module (`TokenEnv`, `SeqTable`, `exact_expectation`, `group_advantage`,
  `train_grpo`, `make_task`). Today's file **imports** it — that is deliberate.
- **W4 D5** — the proxy/gold divergence and Gao's law `dE[r_gold]/d√KL`, i.e. the RM version of
  today's (e). Today is that result stated in reverse and given a bound.
- **W4 D4** — reward hacking as an *optimization* phenomenon, and the `+1.07 → 4.4e13`
  divergence.
- **W5 D1 and W5 D5** — a preference dataset is a **matching**, the loss is exactly flat off
  the comparison set, and BT scale measures *annotator disagreement*, not quality. That is the
  "unbounded" half of (e).
- **W3 D1** — EGLP, the licence for any baseline that does not depend on the action.

## Required Reading

- **DeepSeek-R1 (2025) §2.2** — the R1-Zero reward design: a rule-based **accuracy** reward
  plus a **format** reward, and the explicit statement that a *neural* reward model was not
  used, because of reward hacking and the cost of retraining it. Read the reward-modeling
  subsection twice; it is three paragraphs and it is the entire day in prose.
- **Tulu 3 (Lambert et al. 2024), the RLVR section** — where the name comes from, and the
  recipe: verifiable ground truth, binary reward, no RM.
- Optional, high-value: **Let's Verify Step by Step (Lightman et al. 2023) §3–4** — PRM vs ORM.
  Read it *after* deriving (c) yourself: process supervision is exactly "add reward levels",
  and (c) prices its benefit while (b) prices its cost.
- Optional: **Dr. GRPO §3** again, one page, for how `1/std` interacts with a binary verifier.

## 2-Hour Plan

> **Clock note, unchanged and non-negotiable.** Blocks are delivered **in full**, however long
> the session runs. Nothing is compressed for the clock.

> **Rendering.** All maths in this file is plain Unicode. No LaTeX anywhere.

> **Two standing rules, both carried from D1 and both now permanent:**
> 1. **Skeletons are PRE-PRINTED by the tutor with empty slots; he fills them.** D1 tested this
>    directly: *told* to use skeleton-first → **0/7 compliance, 56%**; slots *emitted* → **100%**
>    on the identical question. Instruction is not the operative variable; emitting the slots is.
> 2. **The interview drill is MANDATORY and it runs at 1:00–1:20, BEFORE coding.** It has been
>    cut two weeks running. If the clock has to eat something today, it eats coding depth.

### Block 0 · 0:00–0:05 — Numbers-card drill (RECOGNITION, not retrieval)

Open `shared/numbers-card.md`, read the **Week 6** section out loud start to finish, then close
it. Three questions immediately after, closed-book. Skeleton pre-printed.

1. The dead-group law at G = 8: five percentages at p = 0.50 / 0.70 / 0.90 / 0.95 / 0.99.

   ```
   p=0.50 ____   p=0.70 ____   p=0.90 ____   p=0.95 ____   p=0.99 ____
   the direction that should feel wrong: ____________________
   ```

2. The annihilation numbers: how much of a reward-side KL survived the group mean, and what the
   *uniform-drift* number was.

   ```
   KL in reward, surviving: ______   vs full: ______   =  ______× deleted
   uniform drift removed to: ______
   one-line consequence:  ______________________
   ```

3. The clip at μ = 1 versus μ = 8, same lr and seed.

   ```
   μ=1: ______      μ=8: ______
   therefore GRPO at μ=1 IS: ____________________________
   ```

### Block 1 · 0:05–0:25 — Cold recall (skeletons pre-printed, no notes, no scrollback)

1. **The two error directions of a checker.** Before any theory today: name them, and say what
   each one costs. One word each in the cost slot.

   ```
   name ______________  = accept ⊄ correct?  correct ⊄ accept?  ______
   name ______________  = accept ⊄ correct?  correct ⊄ accept?  ______
   cost of the first : ____________
   cost of the second: ____________
   which one is bounded, and by what: ____________________
   ```

2. **W4 D5's proxy/gold result**, in the right variable, and the *shape*.

   ```
   the law:        dE[r_gold]/d____  =  ____
   the shape:      ____________________
   what happens at large KL: ____________________
   the SCOPE condition on your own version of this result: ____________________
   ```

3. **W5's identifiability result**, and then the second clause — which is where the points are
   (six weeks running).

   ```
   a preference dataset is, graphically: ____________________
   therefore the loss is exactly ______ along ____________________
   therefore the value at convergence is a property of ____________________
   and therefore, for a REWARD MODEL specifically: ____________________
   ```

4. **The memory table, all four rows**, and which line item today deletes.

   ```
   PPO   = ____ + ____ + ____ + ____ = ____ GB
   GRPO  = ____ GB   (−____%)
   +cached ref logprobs = ____ GB
   RLVR  = ____ GB   (−____%)
   today deletes: ______   worth ______ GB   = ______% of the original bill
   ```

5. **Sources of variance in REINFORCE — fifth ask, still open.** Sources, not mechanisms. You
   have never named the policy's own entropy, the environment's stochasticity, or
   off-policy-ness; and for an LLM exactly one of those three is **identically zero**.

   ```
   1. ______  2. ______  3. ______  4. ______  5. ______  6. ______  7. ______
   the one that is identically zero for an LLM, and why: ____________________
   ```

6. **MY OWN EVIDENCE — fifth ask, and today it does not move on until it is filled.** Not a
   general claim. A number **you** measured, plus the condition under which it holds, plus the
   thing it does *not* say.

   ```
   the number:              ____________________
   where it came from:      ____________________ (file, stage, seeds)
   the SCOPE condition:     ____________________
   the CAVEAT (what it does NOT establish): ____________________
   ```

   Three candidates from yesterday alone, if you want them: the `no_critic` **null**, the 23×
   `dead_frac` separation, and the G = 8 → 32 fixture repair.

> ## RESULT — pre-printed slots are not sufficient; they must be TYPED
>
> D1's finding was *emit the slots*. Block 1 today shows that is only half the rule. Three of
> six questions came back blank with the note **"a lot of these questions are vague"**, and he
> was right — the defect was in the slots, not the recall:
>
> - **Q1 tested TODAY's material in a cold-recall block.** `V \ C` / `C \ V` is Block 2 content.
>   Cold recall may only test what he already holds. Category error; question voided.
> - **Q3 said "graphically"** and expected a specific noun, without saying that a noun was what
>   the slot wanted.
> - **Q6 gave no anchor** — "the number" over an unbounded domain. **Four consecutive blank
>   asks is evidence the SLOT is broken, not that he is avoiding it.**
> - **Q5 named three of the seven answers in the stem**, so the enumeration could not score.
>
> Re-issued with each slot **typed** — `[graph-theory noun]`, `[a number]`, `[choose one of
> four]`, `[complete this sentence]`, and Q6 reduced to a three-item menu — both came back
> filled and substantive: **Q3 3.0/4, Q6 2.0/4**, against four prior blank asks on Q6.
>
> **STANDING RULE, refining D1's:** every pre-printed slot names the KIND of thing that fills
> it — a noun, a number with units, a choice from an explicit list, or a sentence to complete.
> An untyped slot is an open question with a line drawn under it, and it scores like one.

### Block 2 · 0:25–0:50 — Concept: incompleteness is a compute bug, unsoundness is a correctness bug

**Framing.** The standard telling is "verifiable rewards are better because they are correct."
That is a recitation and it is also not quite true — every real verifier is wrong somewhere.
The useful framing is that a checker has **two error sets with two different currencies**, and
almost every RLVR engineering decision is a trade between them.

```
C = the set of correct outputs           V = the set the checker ACCEPTS
V \ C = false positives = UNSOUNDNESS  → the entire reward-hacking surface
C \ V = false negatives = INCOMPLETENESS → a pass-rate multiplier → dead groups → G
```

Four things follow, and each has a number attached today:

- **Why a verifier beats an RM, stated precisely.** Not memory (14 GB, the smallest cut on the
  table). Not accuracy (an RM can be more accurate on-distribution). It is that **a verifier
  does not extrapolate**: its value at a point is *computed*, not interpolated, so it has no
  "off-distribution". An RM is a learned function, and RL is precisely a machine for finding
  the off-distribution argmax — that is W4 D5's `dE[r_gold]/d√KL < 0`, and W5 proved the
  failure is *underdetermination*, not noise. Consequence: the RM's error is unbounded and
  unpatchable without new labels; the verifier's is confined to a **fixed, enumerable,
  patchable set**.
- **The dev loop is the real advantage, and nobody says it.** Train → sort by reward → read the
  top-scoring samples → find the false positive → **patch the checker** → resume. That loop
  takes an afternoon. The RM equivalent is: collect new preferences, retrain the RM, re-validate
  it, and hope. **Edit distance to a fix** is the property that makes RLVR practical.
- **The famous hacks are all `V \ C` members, and they are all boring.** Code: read the test
  file, `sys.exit(0)`, special-case the literal test inputs, monkeypatch `assert`, catch and
  swallow. Math: emit every plausible answer and let a lenient extractor find the right one;
  restate the question; exploit `==` on strings so `0.5` ≠ `1/2` becomes an argument for a
  looser matcher, which is how false negatives get *converted into* false positives. **Every
  fix for incompleteness is a candidate unsoundness bug.** That is the central tension of
  verifier design, and it is why you cannot tune your way to a perfect checker.
- **The systems bill, which is the part that is actually hard.** Execution-based rewards put a
  sandbox fleet in the training loop's critical path. `N = 1024` prompts × `G = 8` = **8192
  executions per step**. At 300 ms each, serial, that is **41 minutes per step**; at 256
  parallel workers, ~10 s. And you wait for **all** 8192, so the step time is set by the tail,
  not the mean: with a 5 s timeout, a single hung sample costs 5 s even when the mean is 50 ms.
  Then the sting: **the timeout is simultaneously a wall-clock knob and a false-negative knob**,
  and it fires preferentially on the problems that need the most compute — so tightening it to
  save time silently biases the curriculum against hard problems. A wall-clock optimization
  becomes a curriculum bias. Say that in an interview and you have said something true that is
  not in any paper's abstract.

Finish the block by stating the honest bill out loud: **RLVR moves 14 GB off the GPU and buys a
distributed systems problem.**

### Block 3 · 0:50–1:00 — Derivation (1): the selection law

Short, and it is the day's most quotable result.

Let `p` = P(the model is actually correct), `α` = P(rejected | correct) (incompleteness),
`γ` = P(accepted | incorrect) (unsoundness). Derive:

```
p_obs = p(1−α) + (1−p)γ
P(hack | rewarded) = γ(1−p) / [ p(1−α) + γ(1−p) ]
```

Then evaluate at `γ = 0.01`, `α = 0.10`, for `p = 0.5 / 0.1 / 0.02 / 0.005`. State the limit as
`p → 0` and say it in one sentence. Then the second half, which is where the points are:
GRPO learns **only from alive groups**, so restate the number *conditional on a group being
alive*, and say what that does to the gradient direction, not just to the reward.

### Block 4 · 1:00–1:20 — INTERVIEW DRILL (MANDATORY — this block does not get cut)

**This is the block that has been skipped for two weeks and the `interview-bank/` files are
still empty after six.** Rules: asked cold, answered **out loud**, no notes and no scrollback,
60–120 seconds each. I grade after. Model answers come **after** yours, as a diff — never
instead. **You** write the entry into `interview-bank/rl-post-training-qa.md`; an empty slot
means the block did not happen.

**First, back-fill the three W6 questions already stubbed in the bank** — yesterday measured
the answer to all three and the block that owned them was cut:

1. **"GRPO vs PPO — what changes and why?"** Must contain: the estimand both algorithms
   estimate; the exact shrinkage and what it says at G = 1; what is actually lost (one precise
   phrase — it is not "variance"); the memory arithmetic; and the scope condition on your own
   `no_critic` null.
2. **"Why KL to a reference policy? What breaks without it?"** Must contain: where the KL goes
   and why *there* (yesterday's 19.15×); which estimator and why (feasibility, not variance);
   and what a uniform drift does to a reward-side penalty.
3. **"KL spikes, reward improves, human evals degrade — diagnose it."** Ordered checklist, and
   name the single measurement that separates over-optimization from a broken reward.

**Then the two new ones for today** (I will add them as stubs to the bank before the block):

4. **"What makes a reward *verifiable*, and when does RLVR fail?"**
5. **"Your math verifier is 97% accurate. Is that good enough to train against?"** — the
   correct answer is a question back, and then a set, not a rate.

### Block 5 · 1:20–1:45 — Derivations (2)–(5)

**(2) The Rényi dead-group law.** For a reward with output distribution `q_1..q_L`, derive
`P(dead) = Σ_j q_j^G` and identify it as `exp(−(G−1)·H_G(q))` with `H_G` the Rényi entropy of
order G. Check it collapses to yesterday's binary law. Then compute the two fixtures — binary
`p = 0.05` vs three-level `(0.05, 0.40, 0.55)` at G = 8 — and solve for the `G` that would buy
the same improvement. State the asymmetry rule in one line, then say what it implies for a
**saturated** model (p = 0.95) where splitting the failures does nothing.

**(3) The false-negative compute tax.** Compose `p_obs` with yesterday's `G = ln(0.1)/ln(1−p)`
and price an α = 30% extraction failure at p = 0.10. Then state it as an engineering rule in
the form "a rate of X in the checker equals Y% of the rollout budget."

**(4) Component annealing.** Write `r = r_correct + λ·r_format` and push it through the group
mean. Show the format term contributes only its within-group deviation, that the deviation's
scale is `√(c(1−c))` for a Bernoulli component with compliance `c`, and therefore that the
component's gradient share peaks at `c = 0.5` and is **exactly zero** at `c ∈ {0, 1}` for any λ.
Then the design corollary, and then the trap: for what λ does "wrong but perfectly formatted"
outrank "right but badly formatted"? Give the lexicographic condition.

**(5) The soundness theorem.** Prove: if `V ⊆ C` then `gold_pass(π) ≥ v_pass(π)` for every π.
Then prove the converse worst case: if `V \ C ≠ ∅`, `sup_π (v_pass − gold_pass) = 1`, and it is
1 regardless of `|V \ C|`. Conclude with the sentence that is the day's thesis, and contrast it
with the RM case, where the sup is not even a property of the data (W5: the loss is exactly flat
along the uncovered directions, so the gap is decided by initialisation).

### Block 6 · 1:45–2:45 — Coding: `week-06-grpo-rlvr-reasoning/code/rlvr.py`

Staged harness, W4/W5/W6 convention. **Imports `grpo.py`** — same `TokenEnv(vocab=3, t_max=3)`,
15 enumerable sequences, 7 states — so every expectation is checked against exact enumeration,
every gradient against a central finite difference, and yesterday's `group_advantage` /
`train_grpo` are reused rather than re-derived. The gold correctness mask `C` and the verifier
mask `V` are **both** explicit boolean vectors over the 15 sequences, which is what makes
"soundness" a checkable set relation instead of an adjective.

TODOs (yours — four, and one is the diagnostic again):

1. `verifier_rates(gold, vmask, probs)` → `(alpha, gamma, precision, recall, p_true, p_obs)`.
   **The trap is the LEVEL, and it is the tenth sighting waiting to happen:** these are rates
   over **probability mass under the current policy**, not over the 15 sequences. A verifier's
   confusion is *policy-dependent* — that is the whole point, and counting sequences silently
   gives you a policy-independent number that looks right.
2. `hack_share(gold, vmask, probs, G)` → `P(hack | rewarded)` and its **group-conditional**
   version `P(the alive group's positive advantage sits on a false positive)`. Closed form
   where one exists; exact enumeration otherwise.
3. `component_shares(components, weights, groups)` → each reward component's share of the
   **group-centered** gradient. Must reproduce the `c(1−c)` law and return exactly 0 at
   `c ∈ {0, 1}`.
4. **`diagnose_rlvr(history)` — the diagnostic TODO, deliberately yours again** (W5's finding:
   you write the mathematical core and delegate the scaffolding, so threshold design is being
   taught rather than practised). Contract: return `list[(name, detail)]`, **silent on a healthy
   run**, every threshold keyed on a **level** and calibrated from the gap between
   *distributions* over ≥8 seeds — never between one healthy run and the one broken value you
   happened to see. Rules to implement: `verifier_hacked` (v_pass rises while gold_pass falls),
   `signal_starved` (dead high **and** pass low), `saturated` (dead high **and** pass high — the
   D1 mode; note it needs the conjunction, because `dead_frac` alone cannot tell these apart).
   **And one candidate rule must be REJECTED during calibration:** "a component's gradient share
   → 0" looks alarming and is in fact the health signature from (4). Yesterday cut a rule the
   *instrument* could not support; today you cut one the *semantics* do not support. Write the
   reason down — that sentence is the deliverable for this TODO as much as the code is.

Provided (not yours): the env/table import, the fixtures, the exact-enumeration helpers, the
underdetermined-RM construction, and `train_rlvr`'s optimiser plumbing.

Stages assert, at minimum: the confusion rates against exact enumeration; `P(hack|rewarded)`
against a ≥5-seed simulation and its `p → 0` limit; the Rényi law against a simulation and the
binary special case to `< 1e-12`; the FN compute tax as an exact `G`; annihilation of a
saturated component to `< 1e-15` **with a negative control at c = 0.5**; the lexicographic
λ boundary from both sides (a wrong-but-formatted optimum above it, correctness restored
below); the soundness theorem as `gold ≥ v` over a sweep of policies including adversarial ones;
and the RM contrast showing the gap changing with **initialisation** at a fixed dataset — which
is what "underdetermined" means operationally.

### Block 7 · 2:45–2:55 — Deliverable & next session

Fill in `deliverable.md`, add every measured number to `shared/numbers-card.md`, update
`progress.md` and `HANDOFF.md`, and **write the five interview-bank entries in your own words**.
Day 3 is **sampling, pass@k and rejection sampling** — which is the other half of today's
compute currency: today priced false negatives in `G`, Day 3 prices `G` itself against
temperature and `k`.

## Checkpoint Questions

1. Your verifier is 97% accurate on a held-out set. Give the two follow-up questions you must
   ask before agreeing to train against it, and say why the 97% is nearly irrelevant to both.
2. A teammate loosens the answer matcher to fix false negatives on `1/2` vs `0.5`. Name the two
   quantities that move, in which direction, and which of the two you would monitor.
3. Derive the fraction of *rewarded* samples that are false positives, and evaluate it at a 1%
   FP rate and a 2% pass rate. Then explain why the curriculum makes this worse by design.
4. Under GRPO, you add a format bonus with λ = 0.2. Describe its influence over the course of
   training, in one sentence and one formula, and say what happens if you double λ at step 5000.
5. State the dead-group law for a reward with L levels, name the information-theoretic quantity
   it is, and say which side of the distribution you must split at p = 0.05 and at p = 0.95.
6. Prove that a sound verifier has a zero optimization gap for every policy. Then say why a
   reward model cannot have that property even in principle, using the word "underdetermined".
7. You are running execution-based rewards at N = 1024, G = 8. Give the executions per step, the
   quantity that sets the step time, and the second-order effect of lowering the timeout.

## Frontier-Lab Interview Angle

RLVR is the 2025 follow-up to the GRPO question, and the median answer is "you use a checker
instead of a reward model so it can't be hacked" — which is *false*, and being able to say why
is the whole differentiator. Three moves that mark a real answer:

- **Refuse the rate.** When someone offers you verifier accuracy, ask for the two error sets
  separately, then say the sentence: *incompleteness costs compute, unsoundness costs the run.*
  Almost nobody separates them, and every practical decision depends on the separation.
- **Price the false negatives.** "A 30% extraction failure at p = 0.10 raises the G needed for
  90% alive from 22 to 32" is a sentence that ends the debate about whether parsing is
  infrastructure or research. Have the formula, not just the anecdote.
- **Name the dev loop.** The reason labs run RLVR is not 14 GB and not purity — it is that a
  false positive is a **patchable bug** with an afternoon's edit distance, while an RM failure
  needs a new dataset. That is an engineering answer to a research question, which is exactly
  the register these interviews reward.

Have a number ready for each. Yesterday's dead-group law plus today's selection law is a
complete, quantitative story about why reasoning RL gets harder exactly as it starts working —
and that story is a much better project-deep-dive than a description of an algorithm.
