# Week 06 Day 03 — Sampling, pass@k & Rejection Sampling: coverage vs mass

## Objective

**Read this first, out loud.** Two days ago you priced the group size `G`. Yesterday you priced
the verifier in units of `G`. Today you find out that **`G` and `k` are the same variable**, and
that the number everyone quotes for a reasoning model — pass@k — is not a property of the model.

Start from the collision. Day 1's law and the pass@k formula share a term:

```
pass@k     = 1 − (1−p)^k
P(dead)    = p^G + (1−p)^G   =   [1 − pass@G]  +  p^G
```

**A dead group is a UNANIMOUS group, and pass@G asks about only one of the two unanimities.**
That is the whole relationship, and it is worth more than it looks:

```
p = 0.05, G = 8 :  dead = 66.3420%   of which  100.00%  is the failure side (1 − pass@8 = 66.3420%)
p = 0.95, G = 8 :  dead = 66.3420%   of which    0.00%  is the failure side (1 − pass@8 = 0.0000%)
```

**Identical dead fraction, opposite cause, and `dead_frac` cannot tell them apart — pass@G can.**
That is why D1's `saturated` vs `signal_starved` rules needed the *conjunction* with the pass
rate: `dead_frac` is symmetric under `p ↔ 1−p` and pass@G is not. And one more collision, which
is not a coincidence: **the alive fraction at p = 0.95 (33.658%) is exactly pass@8 at p = 0.05
(33.6580%)**, to 4e-11.

Now the day's thesis:

> **pass@k is not a better pass@1. It is a DIFFERENT FUNCTIONAL of the same policy, and every
> knob you own trades one for the other.** The derivative is the whole story:
>
> ```
> ∂ pass@k / ∂p  =  k·(1−p)^(k−1)
> ```
>
> The *relative* weight the metric puts on a hard problem versus an easy one is
> `((1−p_hard)/(1−p_easy))^(k−1)`, which **grows exponentially in k**. So as k rises, pass@k
> stops being a measurement of your model and becomes a measurement of your model **on the
> problems it is worst at**. At k = 1 you are measuring MASS (is the mode right?). At large k
> you are measuring SUPPORT (is the answer reachable at all?). Temperature, RL, distillation
> and decoding all buy one with the other.

And the second half, which is the part that makes it an engineering day rather than a metric day:

> **pass@k is a number you cannot cash without naming a selector.** pass@k assumes an oracle
> picks the right one of your k samples. Replace the oracle with something real and you get
> three different limits:
>
> ```
> pass@k → 1        whenever p > 0                    a property of the SUPPORT
> BoN@k  → 1        iff the verifier is SOUND         a property of the ACCEPT SET
> maj@k  → 1        iff correct is the MODAL answer   a property of the MASS
> ```
>
> And when the selector is wrong, the **way** it is wrong depends on whether it is a **filter**
> or a **ranker** — a distinction nobody makes and which decides whether the sampling budget is
> safe to spend.

Four things get proved and measured today, and each one is a sentence you can say in an
interview:

**(a) The standard pass@k estimator is unbiased and the obvious one is not — biased in a known
direction.** The plug-in `1 − (1−c/n)^k` is *concave* in p, so Jensen says it **under**-reports.
At n = 10, p = 0.1, k = 10: true **0.651322**, plug-in **0.493747** — a **−0.157575** bias, 24%
of the value. The combinatorial estimator `1 − C(n−c,k)/C(n,k)` is exact to **2.2e-16**, and the
proof is one line of exchangeability, not algebra: pick a uniformly random k-subset of your n
draws; the probability that all k are wrong is `(1−p)^k` because the subset was chosen
independently of the labels.

**(b) The optimal temperature rises with k — and ONLY on a heterogeneous prompt set.** This is
the received rule ("T = 0.2 for pass@1, T = 0.8+ for pass@100") and it has a scope condition
nobody states. Measured today on a 16-prompt set:

```
HETEROGENEOUS (8 easy + 8 hard):   T* = 0.20 / 0.20 / 0.20 / 3.00 / 3.00 / 3.00   at k = 1/2/4/8/16/64
HOMOGENEOUS  (16 easy)         :   T* = 0.20 / 0.20 / 0.20 / 0.20 / 0.20 / 0.20   at k = 1/2/4/8/16/64
```

**The k-dependence of temperature is a property of the PROBLEM DISTRIBUTION, not of the
sampler.** On a single problem — or on any set where every p is equal — the pass@k-optimal
temperature is exactly the pass@1-optimal temperature, because `1 − (1−p)^k` is monotone in p
and monotone transformations do not move an argmax. The whole effect is heterogeneity, routed
through `(1−p)^(k−1)`. Same mechanism as the thesis. Note also that the transition is **sharp**,
not gradual: between k = 4 and k = 8 here.

**(c) A filter's composition is n-invariant; a ranker's is not, and the ranker's failure is
literally pass@n pointed at your verifier's error set.** Same verifier (γ = 1% false positives),
same budget, two selection rules:

```
FILTER  (accept-set, emit a uniformly random accepted sample):
        P(hack | emitted) = γ(1−p)/[p(1−α)+γ(1−p)] = 9.09%  at n = 1, 2, 8, 64, 256 — IDENTICAL
RANKER  (argmax of the score, and the score is +∞ on V \ C):
        P(hack | emitted) = 1 − (1−γ)^n = 1.00% / 7.73% / 47.44% / 92.37%  at n = 1 / 8 / 64 / 256
```

**At n = 256 the same verifier gives 9.1% or 92.4% depending only on whether you filter or
rank.** Increasing the best-of-n budget does not change *what a filter emits* — it only changes
how often it emits anything. A ranker takes the **extreme** of a noisy score, and the extreme of
a noisy score is wherever the noise is largest. That is why best-of-n against an RM
over-optimizes and best-of-n against a unit test does not, and it is the inference-time version
of yesterday's `V \ C` result. Yesterday: *unsoundness costs the run*. Today: *unsoundness costs
the sample, and the bill grows with n*.

**(d) Best-of-n is a KL knob with an exactly known price.** For a continuous score,

```
KL(BoN_n ‖ π)  =  log n − (n−1)/n
```

which is **0.1931 / 1.2044 / 3.1745 / 4.5491** nats at n = 2 / 8 / 64 / 256, and adds ~**0.69**
nats per doubling. Compose it with W4 D5's Gao law `E[r_gold] vs √KL` and best-of-n stops being
"free inference compute" and becomes a point on the same over-optimization curve as your
training run — at a KL you can compute in closed form before you spend a GPU-hour. **√KL goes
0.4395 / 1.0975 / 1.7817 / 2.1329**, so n = 256 sits at the same distance from the reference as
a serious RL run.

Then the day's experiment, which is the current research question stated as a measurement:
**does RL add capability, or does it move mass?** Train with `grpo.py`, track pass@1 and pass@k
across k = 1…1024 for both the base and the trained policy, and find the **crossover k** where
the base model wins. If the crossover exists, the honest sentence is *"RL bought pass@1 by
spending pass@k"*, and the mechanism is (b) and the thesis: sharpening abandons the problems it
got wrong, and at large k those are the only problems the metric can see.

Build `code/sampling.py` to make all of this assertions rather than claims.

## Prerequisites

- **W6 D1** — the dead-group law `p^G + (1−p)^G`, `G = ln(0.1)/ln(1−p)` = **44.89 → 45** at
  p = 0.95, the exact `(G−1)/G` shrinkage, the `1/std` amplification, and `grpo.py` as an
  importable module (`TokenEnv`, `SeqTable`, `build_seq_table`, `make_task`, `group_advantage`,
  `train_grpo`, `exact_expectation`, `exact_entropy`, `exact_kl`, `exact_pass_rate`).
- **W6 D2, all of it** — `V ⊆ C` / `C ⊆ V`, the selection law
  `P(hack|rewarded) = γ(1−p)/[p(1−α)+γ(1−p)]`, the Rényi generalisation
  `P(dead) = Σ_j q_j^G`, and `rlvr.py` (`make_checker`, `verifier_rates`, `dead_fraction_levels`,
  `g_for_alive`, `p_observed`). Today's file **imports both** — that is deliberate, and it is
  the third day in a row it has been.
- **W4 D5** — `dE[r_gold]/d√KL` and the proxy/gold divergence. Today gives you a *closed-form
  KL* to put on its x-axis for the first time.
- **W2 D3** — decoding: temperature, top-k, top-p, and why they are support-modifying vs
  mass-modifying operations. Today makes that distinction quantitative and shows that
  **temperature preserves the support and truncation destroys it** — which is why top-p is a
  pass@k bug and temperature is not.
- **W3 D2/D3** — variance of a Monte-Carlo estimator, and why an unbiased estimator of a
  nonlinear functional is not the nonlinear functional of an unbiased estimator. That is
  literally (a).

## Required Reading

- **Codex / HumanEval (Chen et al. 2021) §2.1** — where pass@k comes from, the unbiased
  estimator, and the sentence about numerical stability. Two paragraphs; derive the estimator
  yourself before reading it.
- **"Does RLVR really incentivize reasoning capacity beyond the base model?" (Yue et al. 2025)**
  — the pass@k crossover. Read the figures, not the prose: the trained model wins at small k and
  the base model wins at large k. Today reproduces the *mechanism* in an enumerable toy.
- **Self-Consistency (Wang et al. 2022) §2** — majority vote. One page. Then ask the question the
  paper does not: *what happens when the mode is wrong?*
- Optional, high-value: **Cobbe et al. 2021 (GSM8K) §4** on verifiers and reranking, for the
  filter/ranker distinction in its original habitat.
- Optional: **Stiennon et al. 2020, Appendix** for `KL(BoN) = log n − (n−1)/n`, after you derive
  it. It is four lines with the probability-integral transform.

## 2-Hour Plan

> **Clock note, unchanged and non-negotiable.** Blocks are delivered **in full**, however long
> the session runs. Nothing is compressed for the clock.

> **Rendering.** All maths in this file is plain Unicode. No LaTeX anywhere.

> **Three standing rules, all earned by direct test on D1 and D2:**
> 1. **Skeletons are PRE-PRINTED by the tutor, and every slot is TYPED.** D1 proved that
>    *emitting* the slots beats *asking* for structure (0/7 compliance when told, 4/4 when
>    emitted). D2 proved that is only half of it: an untyped `____` scores like an open question.
>    Every slot below names the kind of thing that fills it — `[a number]`, `[a noun]`,
>    `[choose one of three]`, `[complete the sentence]`. And **never name the candidates in the
>    stem** — on D2 a stem that listed three options got exactly those three back.
> 2. **The NUMBER slot comes FIRST in every skeleton.** D2's clean signature: 5 of 5 definitional
>    slots filled, 5 of 5 number-and-consequence slots blank. Putting the number first is the fix.
> 3. **The interview drill is MANDATORY and runs at 1:00–1:20, BEFORE coding.** On D2 it repaired
>    a same-day regression (`d = √KL` failed cold recall, passed under drill conditions 40 minutes
>    later with no teaching in between). If the clock has to eat something, it eats coding depth.

### Block 0 · 0:00–0:05 — Numbers-card drill (RECOGNITION, not retrieval)

Open `shared/numbers-card.md`, read the **Week 6 Day 2** section out loud start to finish, then
close it. Three questions immediately after, closed-book.

1. The selection law, evaluated. A **fixed 1%** false-positive rate, α = 0.10.

   ```
   p=0.500: [a percentage] ____   p=0.100: [a percentage] ____
   p=0.020: [a percentage] ____   p=0.005: [a percentage] ____
   the limit as p → 0: [a number] ____
   the one-sentence consequence for a CURRICULUM: [complete the sentence] ____________________
   ```

2. Split-the-mode. p = 0.05, G = 8.

   ```
   binary: [a percentage] ______  →  three-level (0.05, 0.40, 0.55): [a percentage] ______
   that ratio: [a number]× ______
   buying the same with G alone needs G = [an integer] ______  = [a number]× the rollout bill
   at p = 0.95 you must split the [choose one: successes / failures] ______, because ____________
   ```

3. The soundness theorem, as a number and as a scope condition.

   ```
   V ⊆ C  ⇒  gold_pass − v_pass ≥ [a number] ______ , verified over [an integer] ______ policies
   the SCOPE condition on that statement: [one of: "at small KL" / "at convergence" / "none"] ____
   if V \ C ≠ ∅ the sup gap is [a number] ______ , and it is that number for |V\C| = [an integer]
     ______ as well as for [an integer] ______
   therefore reward hacking is a ______-property, not a ______-property [two nouns]
   ```

> ## RESULT — a typed slot is not enough; the slot must name the FUNCTION it evaluates
>
> Block 0 was issued with typed slots (`[a percentage]`, `[an integer]`) per D2's standing rule,
> and he stopped and said: *"please be a bit clearer on what I need to fill — it is not clear
> what function you want for the 3 questions."* He was right, and this is the **third refinement
> of the same rule, each one found by him or by direct test**:
>
> ```
> D1: EMIT the slots            (told to use skeleton-first → 0/7; slots emitted → 4/4)
> D2: TYPE the slots            ("[a number]" not "____"; 3 blank questions → 3.0/4 and 2.0/4)
> D3: NAME THE FUNCTION         write the formula above the slot, and say which of its
>                               arguments is being varied and which are held fixed
> ```
>
> The defect was concrete: `p=0.500: [a percentage] ____` is ambiguous between `p_obs` and
> `P(hack | rewarded)` — **two functions he derived on the same day, from the same three
> parameters**. A type constrains the *codomain* of the answer; it does not identify the
> *function*. Re-issued with `P(hack|rewarded) = γ(1−p)/[p(1−α)+γ(1−p)]` printed above the slots
> and "holding γ = 0.01 and α = 0.10 fixed, varying only p" stated: **6/6, all four values
> correct to 4 dp.**
>
> **STANDING RULE, superseding D2's:** a numeric slot prints (i) the formula, (ii) which argument
> varies, (iii) which are held fixed, and (iv) the units/precision. Anything less is a type
> annotation on an open question.

### Block 1 · 0:05–0:25 — Cold recall (skeletons pre-printed, no notes, no scrollback)

**Rule from D2: cold recall tests PRIOR material only.** Nothing below is taught today.

1. **The dead-group law, and the thing about it you have not been asked yet.**

   ```
   the law: P(dead) = ______________________
   at G = 8, p = 0.95: [a percentage] ______   at G = 8, p = 0.05: [a percentage] ______
   what you notice about those two numbers: [complete the sentence] ____________________
   the closed form for the G that keeps 90% of groups alive: G = ______________________
   at p = 0.95 that is G = [an integer] ______
   ```

2. **Where the KL goes, and the number that decided it.**

   ```
   KL in the REWARD survives the group mean at: [a number] ______ vs full [a number] ______
   = [a number]× deleted
   therefore a KL in the reward is a ______-loop controller [one word]
   which estimator, k1 or k3, and the reason in ONE word (it is not "variance"): ______ , ______
   the evidence for that word: [a percentage] ______ of the probability mass
   ```

3. **W2 D3, decoding.** Two operations, and the property that separates them.

   ```
   temperature T: pi_T(y) ∝ ______________________
   top-p / nucleus: ______________________
   the property one of them destroys and the other does not: [a set-theoretic noun] ____________
   therefore, as T → 0 and as T → ∞, the SUPPORT of the policy: ____________________
   ```

4. **W3, variance.** The estimator question, stated abstractly, because today is an instance.

   ```
   if E[X̂] = X, is f(X̂) an unbiased estimator of f(X)? [yes / no / only if ___] ____________
   the name of the inequality that gives you the DIRECTION: ______________
   for f CONCAVE, E[f(X̂)] is [above / below] f(X): ______
   ```

5. **Sources of variance in REINFORCE — SIXTH ask.** Sources, not mechanisms. Seven slots; you
   have never filled more than four. One of them is identically zero for an LLM.

   ```
   1. ______  2. ______  3. ______  4. ______  5. ______  6. ______  7. ______
   the one that is identically zero for an LLM, and the one-clause reason: ____________________
   ```

6. **MY OWN EVIDENCE — SIXTH ask. Number FIRST this time (D2's fix).** Pick from yesterday if you
   like: the 415-policy soundness sweep, the 73.5× split-the-mode, the 26.55× component decay,
   the 0.6720 → 0.0926 precision collapse.

   ```
   the number, with units:  [a number] ____________________
   where it came from:      [file, stage, #seeds] ____________________
   the SCOPE condition:     [a sentence beginning "holds only when..."] ____________________
   the CAVEAT:              [a sentence beginning "it does NOT establish..."] ____________________
   ```

> ## RESULT — the "MY OWN EVIDENCE" slot is broken, and the fix is to stop asking him to CHOOSE
>
> Sixth ask. History: four consecutive blanks (W6 D1 and earlier), then D2 reduced it to a
> three-item menu and got **2.0/4** — the only non-blank in the series. Today it was issued
> **with a four-item menu of his own measured results** and he returned **the menu itself**
> pasted into the number slot, with the provenance and scope slots blank.
>
> **Diagnosis, and it is not avoidance.** The slot asks him to make an *ownership* claim about
> results produced inside a session someone else designed. Selecting "my evidence" is a
> judgement about authorship; stating a scope condition is a technical skill. **The slot bundles
> them, and the authorship judgement blocks the technical one.** Evidence: on W6 D2 he produced
> an unforced scope condition — *"holds only at short horizon"* — on his `no_critic` null,
> **unprompted**, in a block that never asked for one. He can do the technical half. He cannot
> do the selection half.
>
> **STANDING RULE, replacing the open-ended ask:** **assign** one specific result and ask only
> for `number → provenance → scope → caveat` about *that* result. No menu, no choice. Re-run the
> choice version only after the scope-condition habit is established.

### Block 2 · 0:25–0:50 — Concept: coverage and mass are different functionals

**Framing.** The standard telling is that pass@k is "pass@1 with more tries", i.e. the same
measurement with a bigger budget. That is wrong in a way that matters, and the derivative says so.

```
pass@k(p) = 1 − (1−p)^k          ∂pass@k/∂p = k·(1−p)^(k−1)
```

Fix two problems, one easy (p = 0.8) and one hard (p = 0.05). The metric's marginal valuation of
progress on the hard one, relative to the easy one, is `(0.95/0.20)^(k−1) = 4.75^(k−1)`. At k = 1
they are worth the same. At k = 8 the hard problem is worth **~1.1e5** times as much. **pass@k
at large k is not an aggregate over your benchmark; it is an aggregate over the tail of your
benchmark**, and the weight is exponential in k. Everything else today is a corollary.

Four consequences, each with a number:

- **Aggregation order matters and papers get it wrong.** `E_problems[1 − (1−p_i)^k]` is not
  `1 − (1−E[p_i])^k`. The gap is Jensen again and it grows with k. Report per-problem pass@k and
  average; never average p first. Same shape as (a)'s estimator bias, one level up.
- **Temperature is a support-preserving, mass-moving operation; truncation is not.** For any
  T > 0, `π_T(y) ∝ π(y)^(1/T)` has exactly the same support as π, so **pass@∞ is
  temperature-invariant**: temperature cannot make an unreachable answer reachable, it can only
  change the k at which you reach it. Top-p and top-k *delete* support, so they lower pass@∞
  discontinuously — a decoding default silently caps your benchmark. That is the practical
  content of the W2 D3 distinction, and it is why "we sampled at top-p 0.95" is a load-bearing
  line in a pass@k table.
- **The k-dependence of T* is heterogeneity, and here is the proof in one line.** For a single
  problem, `argmax_T [1 − (1−p(T))^k] = argmax_T p(T)` for every k, because `1 − (1−·)^k` is
  strictly increasing. **A monotone transformation cannot move an argmax.** So any observed
  k-dependence must come from *averaging across problems with different p*. Measured above:
  T* = 0.20 → 3.00 between k = 4 and k = 8 on a mixed set, and T* = 0.20 at every k on a
  homogeneous one.
- **The iid assumption is where the remaining lies live.** `1 − (1−p)^k` assumes your k samples
  are independent. They are not: same prompt, same weights, shared prefixes, and — worst — a
  latent "approach" the model commits to in the first few tokens. Under positive correlation the
  true pass@k is **below** the iid formula, and the gap grows with k. Operationally: pass@k
  measured at low temperature is more correlated, so it is optimistic in exactly the regime where
  people quote it.

Then the punchline of the block, and it is the sentence to carry: **pass@k without a named
selector is a number you cannot cash.** The metric assumes an oracle chose the right sample. In
production you have a verifier, a reward model, or a vote — and the three of them have three
different limits and three different failure modes. Which is Block 3.

Finish the block by saying the honest version out loud: **RL post-training and pass@k are
adversaries by construction. RL moves mass; pass@k at large k measures support.**

### Block 3 · 0:50–1:00 — Derivation (1): the unbiased estimator, in one line

Do not do algebra. Do the exchangeability argument.

You have n samples, c of them correct. Draw a uniformly random k-subset S ⊆ [n], independent of
the labels. Then:

```
P(every element of S is wrong)  =  C(n−c, k) / C(n, k)          (counting, given c)
                                =  (1−p)^k                       (iid + independence of S)
```

so `1 − C(n−c,k)/C(n,k)` is an **exactly unbiased** estimator of `1 − (1−p)^k`. Then:

1. State why `1 − (1−c/n)^k` is biased and in which direction, in one word (the word is a name).
2. Give the product form that does not overflow: `C(n−c,k)/C(n,k) = Π_{i=0}^{k−1} (n−c−i)/(n−i)`.
   Say what happens at `c = 0`, at `k > n−c`, and at `k > n` — three cases, one of which is an
   error and two of which are values.
3. Evaluate the bias at n = 10, p = 0.1, k = 10 and say what fraction of the true value it is.

### Block 4 · 1:00–1:20 — INTERVIEW DRILL (MANDATORY — this block does not get cut)

Rules: asked cold, answered **out loud**, no notes, no scrollback, 60–120 seconds each. I grade
after. Model answers come **after** yours, as a diff — never instead.

**You write the entry into `interview-bank/*.md`.** On D2 you asked me to write five entries and
I did; the reference value is intact but the retrieval value of those five is spent. Today the
line holds: **I will write at most one as a worked model, and you write the rest.**

**First, the three entries that have been blank for six weeks** — these are the oldest debt in
the project and every one of them is material you have already measured:

1. **"Explain the full RLHF pipeline (SFT → RM → PPO → evals)."** Must contain: what each stage
   optimises and what it *cannot* fix downstream; where the KL sits and why; and one number of
   yours from W4.
2. **"Derive the PPO clipped objective and explain why old logprobs are necessary."** Must
   contain: what `ρ` is a ratio *of*; what the clip does to the gradient (not to the loss); the
   condition under which the clip is inert; and your own **0.000e+00 vs 0.2222** measurement.
3. **"Derive DPO from KL-regularized RLHF. When is DPO the wrong tool?"** Must contain: the
   closed-form optimal policy, the partition-function cancellation, the one sentence naming what
   DPO *is* ("a reward model whose parameterisation happens to be a policy"), and the failure
   mode that is a property of the DATA FORMAT rather than of the loss.

**Then the two new ones for today** (I will stub them in the bank before the block):

4. **"Your model gets 85% pass@64 and 30% pass@1. What do you actually know?"** — the correct
   answer names a selector before it names a capability.
5. **"Best-of-64 against your reward model beat greedy decoding. Ship it?"** — the correct answer
   is a closed-form KL and the filter/ranker distinction.

### Block 5 · 1:20–1:45 — Derivations (2)–(5)

> ## RESULT — the reference run, and the claim it killed
>
> `_ref_sampling.py` was written, run under `-W error::RuntimeWarning`, and deleted.
> **73/73 checks pass.** It caught **six** defects in my own harness before he ever saw it —
> the most any day has produced. Five were mine to fix; the sixth changed the day's thesis.
>
> 1. **Vacuous stability check.** `pass@k` at n=200, c=100, k=100 is `1 − 1.1e-59`, which
>    rounds to *exactly* 1.0 in float64. Asserting `0 < v < 1` tested nothing. Replaced with a
>    comparison against the exact `math.comb` big-integer ratio.
> 2. **Wrong sign on a threshold.** At k=1, `∂pass@k/∂p = 1` for every p, so the metric weights
>    every problem *exactly* equally (1/6 = 33.3%). I had asserted `< 30%`. The fixed check is
>    also the better one: **uniform at k=1, >99% on the two hardest at k=256.**
> 3. **Saturation destroyed an argmax.** The single-problem T*-invariance check ran to k=1024,
>    where pass@k = 1.0 at *every* temperature, so `argmax` over a constant array returned index
>    0. The maths was right; the instrument had no resolution. Now checked over k ≤ 64, plus a
>    stronger claim — the whole **ordering** of temperatures is k-invariant.
> 4. **Operator precedence.** `np.abs(a - b if False else c)` parses as `(a-b) if False else c`.
>    The check was taking `abs()` of the raw pass@k array, not of a difference.
> 5. **Two fixtures were byte-identical.** `easy` and `mixed` took the same code path, because
>    `max(2, 15//2) == 15//2`. `h_wasted` and `h_healthy` were the same run.
> 6. **The claim I had planned to teach is false in this system.** See below.
>
> ### The rejected rule is no longer a technicality
>
> I had planned to reject `entropy_collapse` for having no discriminative power. During
> calibration the *implemented* rule `coverage_collapse` — pass@1 rises while pass@k_max falls —
> **never fired once.** Swept 2 task flavours × 4 init scales × 3 (lr, β) settings = 24
> configurations, then 8 seeds on each of 3 fixtures:
>
> ```
> min over all runs of [ pass@k_max(final) − pass@k_max(0) ]  =  +0.000e+00
> ```
>
> Not small. **Never negative, anywhere.** The mechanism is GRPO's own advantage: on a prompt
> whose gold is never sampled, every rollout in the group scores the same, the group-relative
> advantage is identically zero, and so is the gradient. *The algorithm cannot push mass off
> coverage it is not sampling.* Yesterday's dead group, read as a **safety property** rather
> than as waste.
>
> **Scope:** 15-sequence tabular softmax, sound verifier, group baseline. This licenses nothing
> at 10⁹ parameters. What it does license: the collapse is not automatic, it needs a mechanism,
> and "RL reduces pass@k" is a claim to be *measured* on the system in front of you.
>
> So the three cut rules now have three genuinely different causes — and D3's is the expensive
> kind, the one that would have passed code review:
>
> | | what was cut | why |
> |---|---|---|
> | **D1** | `sig_starved` | the **instrument** couldn't support it (t_max = 3 left no room) |
> | **D2** | a share-based rule | the **semantics** didn't support it (share → 0 is health) |
> | **D3** | `coverage_collapse` | the **system does not exhibit it** (0 of 24, with a mechanism) |
>
> ### And one number that refused to be a number
>
> `stage_train` asks for the crossover k — where the training gain drops under the seed noise
> floor. Across five seeds it came out **[16, 64, 16, 64, 64]**: a 4× range. Seed 0 alone would
> have given "the crossover is k = 16," exactly the sort of number that survives into a slide.
> The stage now asserts the **bracket** (16 ≤ k ≤ 64 on every seed) *and* asserts that the point
> estimate is unstable. Reporting a bracket is the result; reporting 16 is a bug.
>
> Rules that survived calibration, with their 8-seed margins:
>
> | rule | statistic | broken | healthy | threshold |
> |---|---|---|---|---|
> | `no_coverage` | final pass@k_max | ≤ 0.3845 | ≥ 1.0000 | 0.70 |
> | `mode_locked` | final entropy | ≤ 0.6597 | ≥ 0.9739 | 0.80 |
> | `mass_without_support` | Δ pass@1 | ≥ 0.3697 | ≤ 0.1130 | 0.25 |
>
> `mode_locked` **lost a conjunct to the data**: the obvious "and the spread is small" fails,
> because locked spread ran 0.123–0.379 and healthy spread 0.323–0.556 — they overlap and cannot
> be thresholded. Entropy alone separates. Fewer conditions, because that is what the
> measurement supports.

**(2) The identity, and what it fixes.** Show `P(dead) = [1 − pass@G] + p^G` and therefore that
`dead_frac` is **symmetric under p ↔ 1−p** while pass@G is not. Compute the failure-side share at
p = 0.05 and p = 0.95, both at G = 8. Then state, in one sentence, why D1's `signal_starved` and
`saturated` rules each needed a conjunction with the pass rate — and note that you now have a
*second*, cheaper instrument for the same distinction.

**(3) The temperature/k theorem, and its scope condition.**

> ## RESULT — Block 5, 20.25/32 (63.3%), and a fourth skeleton defect
>
> Section (3) scored **5/5 — the first perfect section of the week.** What earned it was
> STEP 2: he explained the k-dependence of T* by reaching back to `∂pass@k/∂p = k(1−p)^(k−1)`
> **unprompted**. Six weeks of the second clause going missing; here it arrived on its own,
> in a slot that only asked for one clause. Log this as the first *spontaneous* mechanism link.
>
> **My defect (fourth of the day).** My stem read *"the **FAILURE-side** term `p^G`"*. `p^G` is
> the success side; the failure side is `(1−p)^G`. The label contradicted the formula. He
> answered the formula, consistently and correctly, and got full marks — but did **not flag
> the contradiction**. New coaching point: when a stem is self-inconsistent, say so. That is a
> scoring move in an interview, not a rude one.
>
> Running tally of skeleton defects on D3: (1) slots didn't name the function, (2) the
> MY-OWN-EVIDENCE slot bundled authorship with skill, (3) `_ref_sampling` caught six harness
> bugs, (4) a stem whose label contradicted its own formula. The skeleton contract is now the
> most-revised artefact of the week.
>
> ### The 13th wrong-level sighting — the sharpest instance so far
>
> ```
> he wrote : P(hack tops) = 1 − [1 − γ(1−p)/(γ(1−p)+(1−α)p)]^n = 1 − (1 − 0.0909)^n
> correct  : P(hack tops) = 1 − (1 − γ)^n                       = 1 − (1 − 0.01)^n
> ```
>
> He substituted a **posterior over the emitted set** where a **per-sample rate** was required.
> His arithmetic was flawless — `1 − 0.909091^8 = 53.35%` is exactly right *for the input he
> chose*. The modelling was one level off, and the arithmetic then faithfully propagated it.
> This is the cleanest demonstration yet that the error family is not carelessness: it is a
> systematic failure to check **what kind of object** a slot wants before filling it.
>
> ### New refinement of the second-clause gap: MECHANISM vs NAME
>
> Block 4 split the gap into MECHANISM / NUMBER / THESIS. Block 5 splits it again. On (4) he
> derived `F(u) = u^n`, `q(u) = n·u^(n−1)`, and the integral `log n − (n−1)/n` — all correct —
> while leaving blank **the probability integral transform** and **Uniform(0,1)**, the two
> names that license the whole derivation. Same on (5): correct filter formula and 9.0909%,
> blank on *why* it is n-invariant.
>
> **He can execute the object and cannot name it.** Identical to Jensen on Block 3, which he
> then repaired 40 minutes later once the name was said aloud. Prescription: the names are
> retrievable, they are just not *first*. Drill naming before deriving, not after.
>
> Numerics: `4.549084` written as 4.5490 (should be 4.5491). Per-doubling quoted as
> `log 2 = 0.6931` — the asymptote — where the **measured** n=64→128 value is **0.6853**.
> Quote the measurement, name the limit. that for a **single** problem
`argmax_T pass@k(T)` is independent of k. Then show that for a *mixture* of problems it is not,
by exhibiting the reweighting `∂pass@k/∂p_i = k(1−p_i)^(k−1)` and normalising it to an effective
distribution over problems. State the rule and its scope condition in the form: *"T* rises with k
**because** ____, and it does **not** rise with k when ____."*

**(4) The BoN KL.** Let the score be continuous with CDF F, and let `u = F(r)` — so `u ~ U(0,1)`
under the base policy (probability integral transform). Best-of-n emits `max(u_1..u_n)`, whose
density is `n·u^(n−1)`. Then

```
KL(BoN ‖ π) = ∫_0^1 n u^(n−1) · log(n u^(n−1)) du = log n + (n−1)·n·∫_0^1 u^(n−1) log u du
            = log n + (n−1)·n·(−1/n²) = log n − (n−1)/n
```

Evaluate at n = 2 / 8 / 64 / 256, give the per-doubling increment, and then the part that makes
it useful: put `√KL` on W4 D5's x-axis and say what n corresponds to a KL you have already seen
degrade gold. Finally, name the assumption that makes the result *n-only* (no dependence on the
reward or the policy) and say what breaks it — the answer is one word, and it is about ties.

**(5) Filter vs ranker.** Two selection rules against the same checker with false-positive rate γ:

```
FILTER : emit a uniformly random accepted sample.
         P(hack | emitted) = γ(1−p)/[p(1−α) + γ(1−p)]  — INDEPENDENT of n.
RANKER : emit argmax of a score that is maximal on V \ C.
         P(hack | emitted) = 1 − (1−γ)^n  — which is pass@n FOR THE HACK SET.
```

Prove the first by exchangeability (the same argument as Block 3 — this is not a coincidence, and
say why). Evaluate the second at γ = 1%, n = 1/8/64/256. Then the synthesis, which is the day's
best sentence: **best-of-n applies pass@n to your verifier's error set as enthusiastically as it
applies it to the solution set.** Then state the design rule that follows, in the imperative.

### Block 6 · 1:45–2:45 — Coding: `week-06-grpo-rlvr-reasoning/code/sampling.py`

Staged harness, W4/W5/W6 convention. **Imports `grpo.py` and `rlvr.py`** — same
`TokenEnv(vocab=3, t_max=3)`, 15 enumerable sequences — so every pass@k is checked against exact
enumeration, every estimator against an exact Binomial sum over all `c ∈ {0..n}`, and D1's
`train_grpo` is reused rather than re-derived.

TODOs (yours — four, and the diagnostic is again one of them):

1. **`pass_at_k(n, c, k)`** → the unbiased combinatorial estimator, vectorised over problems,
   numerically stable (the product form; `C(200,100)` is 9.05e58 and does not fit an int64 loop
   you write by hand). Plus `pass_at_k_plugin(n, c, k)` for the contrast.
   **Two traps, both from the log:** (i) **the level, tenth sighting** — `pass_at_k` is a
   per-problem estimator and the benchmark number is `mean_i pass_at_k(n, c_i, k)`; averaging `c`
   first is a different quantity and it is *larger*. (ii) **the unguarded denominator, D2's
   six-in-one-session family** — `k > n` must raise, `k > n−c` must return exactly 1.0, and
   `c = 0` must return exactly 0.0. Get those three from the product form, not from special-cases.

2. **`selector_curves(probs, gold, accept, answer_id, ks)`** → for each budget k, the exact
   success probability of **three** selectors, by closed form where one exists and exact
   enumeration otherwise: `oracle` (= pass@k), `bon` (filter on `accept`, uniform among accepted,
   fall back to the first sample when nothing is accepted), and `maj` (plurality over
   `answer_id`, ties broken uniformly). Contract: all three are exact, not sampled; `maj` must
   handle ties; and each must be **monotone in k or documented as not**.

3. **`bon_distribution(probs, score, n)`** → the exact induced distribution over the 15 sequences
   under best-of-n with the given score and uniform tie-breaking, plus its exact KL to the base.
   Closed form: `P(argmax is y) = Σ over tie patterns`, easiest via
   `P(max ≤ s) = (Σ_{score ≤ s} π)^n` and differencing, with the tie mass split evenly.
   **The tie handling is the whole exercise** — a discrete score is exactly the case where
   `log n − (n−1)/n` stops being right, and the harness will make you find the boundary.

4. **`diagnose_sampling(history)` — the diagnostic TODO, deliberately yours for the third day.**
   Contract: return `list[(name, detail)]`, **silent on a healthy run**, every threshold keyed on
   a **level**, calibrated from the gap between *distributions* over **8 seeds per fixture** —
   never between one healthy run and the one broken value you happened to see. Rules to
   implement:
   - `coverage_collapse` — pass@1 rising **and** pass@k_large falling below its own step-0 value.
     Needs the conjunction; either alone is normal.
   - `mode_locked` — entropy below its level **and** pass@k_large flat in k (the spread
     `pass@k_max − pass@1` under a threshold). Distinguish from `coverage_collapse`: this one can
     fire on a run that never had coverage.
   - `budget_wasted` — marginal gain per doubling of k below ε at the largest k, i.e. you are
     paying for samples that cannot change the answer.
   - **And one candidate rule must be REJECTED during calibration: `entropy_collapse` (entropy
     falls monotonically).** It fires on *every* successful run — it is the signature of learning,
     not of a bug. This is the third day running that a rule gets cut, and the reasons have all
     been different: D1 cut one the **instrument** could not support (t_max = 3 gave no room),
     D2 cut one the **semantics** did not support (a component's share → 0 is health), and today
     cuts one with **no discriminative power** (it fires on healthy and broken alike). **Write
     the distinction down — that sentence is the deliverable for this TODO as much as the code.**

Provided (not yours): the env/table import, the fixtures, `train_and_track` (a thin wrapper over
D1's `train_grpo` that records pass@k at every k on the grid), the correlated-sampler mixture, and
the exact-enumeration helpers.

Stages (ten), asserting at minimum:

1. `stage_estimator` — unbiasedness against an exact Binomial sum over all c, to **< 1e-15**, at
   9 (p, k) points; the plug-in's bias **negative at every point** and its magnitude at
   (n=10, p=0.1, k=10) equal to **−0.157575**; the three guards (`k>n` raises, `k>n−c` → 1.0,
   `c=0` → 0.0); stability at n = 200, k = 100.
2. `stage_identity` — `P(dead) = (1 − pass@G) + p^G` to **< 1e-15** over a p-grid; `dead(p) =
   dead(1−p)` exactly; the failure-side share **100.00% at p = 0.05 and 0.00% at p = 0.95**; and
   `alive(0.95) = pass@8(0.05)` to **< 1e-10**.
3. `stage_curvature` — `∂pass@k/∂p` against a central finite difference; the hard/easy weight
   ratio `4.75^(k−1)` exact; and the induced effective problem distribution at k = 1 vs k = 256.
4. `stage_temperature` — exact pass@k(T) on a T-grid × k-grid; `argmax_T` **increasing** in k on
   the heterogeneous set; **constant** in k on the homogeneous one (the negative control, and it
   is the scope condition); plus the support-invariance of temperature and the support *loss*
   under top-p, asserted as set equality/inequality.
5. `stage_correlation` — iid formula vs a correlated mixture sampler; the iid version
   **over**-states pass@k; the gap monotone in k; and exactly zero correlation reproduces the iid
   formula to **< 1e-15** (negative control).
6. `stage_selectors` — the three limits: oracle → 1, bon → 1 iff sound, maj → 1[modal]; a fixture
   with q = (0.35 correct, 0.40 wrong, 0.25 wrong) where **maj@k is non-monotone and → 0**
   (measured: 0.3500 / 0.3563 / 0.3225 / 0.0974 at k = 1 / 9 / 51 / 501) while pass@k → 0.9999.
7. `stage_filter_ranker` — filter composition **n-invariant to < 1e-15** in closed form and
   within MC error over n ∈ {1,2,8,64,256}; ranker `1 − (1−γ)^n` = **1.00 / 7.73 / 47.44 /
   92.37%**; and the two evaluated side by side at n = 256 for the **same** verifier.
8. `stage_bon_kl` — the exact enumerated KL of `bon_distribution` against `log n − (n−1)/n`,
   agreeing in the tie-free limit and **documented where it does not**; monotone in n; the
   per-doubling increment; and `n = 1` giving exactly 0.
9. `stage_train` — a real GRPO run from `grpo.py`; pass@k tracked for both base and trained
   across the k-grid; the **crossover k** reported (or `None`, honestly, with the reason);
   entropy and KL alongside. **≥ 3 seeds, and report the spread — a crossover inside the seed
   spread is a NULL, not a result.** (D1's `no_critic` lesson, applied before the fact this time.)
10. `stage_diagnose` — his `diagnose_sampling`, four fixtures × 8 seeds, silent on healthy, and
    the rejected rule documented in the file.

Run everything under `-W error::RuntimeWarning`. D2 produced **six unguarded denominators in one
session, every one a silent NaN rather than a crash**, one of which left a stage green while
returning `[nan]`. Today's file is full of `C(n−c,k)/C(n,k)`, `0/0` at `k > n`, and empty-slice
means over "the accepted samples" — it is the highest-density opportunity for that family yet.

### Block 7 · 2:45–2:55 — Deliverable & next session

Fill in `deliverable.md`, add every measured number to `shared/numbers-card.md`, update
`progress.md` and `HANDOFF.md`, and **write the interview-bank entries in your own words** (I
write at most one as a worked model). Day 4 is the **reasoning data curriculum** — which is this
week's three days composed: D1 says the signal dies at extreme p, D2 says precision collapses as p
falls, and today says the metric's weight moves onto the tail as k grows. A curriculum is the
policy that chooses p. Day 4 asks what the optimal p is, and the answer is not 0.5 for the reason
you would guess.

## Checkpoint Questions

1. A colleague reports pass@100 = 92% on a benchmark. Name the three things you must ask before
   that number means anything, and say which one changes it the most.
2. Show that for a single problem the pass@k-optimal temperature does not depend on k. Then say
   exactly which assumption you have to break to recover the standard "higher k wants higher T"
   advice, and give the sign of the effect.
3. Derive `1 − C(n−c,k)/C(n,k)` and say in one word why `1 − (1−c/n)^k` is biased. Give the
   direction and the magnitude at n = 10, p = 0.1, k = 10.
4. Your verifier has a 1% false-positive rate. Give `P(the emitted answer is a hack)` at n = 256
   under a filter and under a ranker, and explain the difference in one sentence about extremes.
5. Best-of-64 against an RM. Give the KL to the base policy in nats, without running anything,
   and say what you would compare it to before shipping.
6. `dead_frac = 0.66` on your GRPO run. Give the two possible causes, the single cheapest
   measurement that distinguishes them, and why `dead_frac` alone cannot.
7. maj@k is 34% at k = 5 and 31% at k = 25 on a subset of problems. What is happening, is it a
   bug, and what does pass@k do on the same subset?

## Frontier-Lab Interview Angle

pass@k is the most-quoted and least-understood number in reasoning-model evaluation, which makes
it a free differentiator. The median answer treats it as "accuracy with retries." Four moves that
mark a real one:

- **Name the selector.** "pass@k is an upper bound conditional on an oracle chooser; what selects
  in production?" Then the three limits — support / accept set / mode — and their three failure
  modes. This reframes an eval question as a systems question, which is the register these
  interviews reward.
- **Say the trade out loud.** *"RL post-training buys pass@1 by spending pass@k, and the exchange
  rate is `(1−p)^(k−1)`."* Then offer the crossover-k measurement as the way to settle it, and be
  honest that a crossover inside the seed spread is a null.
- **Price best-of-n in nats.** `log n − (n−1)/n` = **3.17 nats at n = 64**, on the same axis as
  your training KL. Almost nobody knows inference-time selection has a closed-form KL, and it
  turns "just crank n" into a point on a curve you have already seen bend.
- **Separate the filter from the ranker.** A set-membership selector's composition is
  **n-invariant**; an argmax selector's degrades as `1 − (1−γ)^n`. Same verifier, **9.1% vs 92.4%**
  at n = 256. That single sentence explains why unit-test filtering scales and RM reranking does
  not, and it is yesterday's `V \ C` result moved from the weights to the outputs — which also
  means it is **recoverable**: a bad sample is discarded, a bad gradient is not.

Have a number for each. D1 + D2 + D3 now compose into one quantitative story — *the signal dies
where the model succeeds, the precision dies where the model fails, and the metric moves its
weight to wherever you are worst* — and that story is a far better deep-dive than a description
of an algorithm.
