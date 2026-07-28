# Week 03 Retrospective - RL Foundations for Post-Training

## What Clicked

- **The finite-MDP ground-truth pattern.** Choosing an environment where `exact_J` (T Bellman
  backups) plus central finite differences gives a true gradient turned every claim of the
  week into a measurement. "Reward-to-go is unbiased and lower variance" stopped being a
  sentence from a paper and became `max|err| 0.0069, variance 6.317 vs 20.033`. This was the
  single highest-leverage design decision of the week and it should be copied in Week 4.
- **Measurability, stated as "if I tell you `h_t` you can tell me `X`".** Once that landed
  (Day 3, on request) it settled reward-to-go, baseline validity, why critic *accuracy* is
  irrelevant to bias, and the GAE `l >= 1` terms -- four separate results, one concept. It also
  cured a slip that had recurred four times (theta-dependence vs measurability).
- **The `Psi_t` ladder as the organizing frame.** Producing all seven rungs with a bias column
  cold on Day 3, and then seeing GAE slot in as a tunable interpolation, made
  REINFORCE -> A2C -> PPO -> GRPO feel like one algorithm with a parameter rather than four
  papers.
- **Day 4's headline arriving from a failed assertion.** "Bootstrap bias is the state-varying
  part of the critic's error" was not in the plan; it was found because a harness check that
  should have passed didn't. Fixing it required inventing a *second* wrong critic, and the
  contrast between wrong-level and wrong-ranking became the most interview-usable result of
  the week.
- **Day 5's derivation.** The Lagrangian, the closed form, `J* = beta log Z`, and the DPO
  cancellation all came out first try -- and the cancellation was seen unprompted. That is the
  cleanest evidence in the week that the RL substrate is now load-bearing rather than
  memorized.
- **Two identities that make beta feel physical**: `dJ*/dbeta = -KL` and `dE[r]/dKL = beta`.
  Being able to say "beta is the exchange rate in reward-per-nat" instead of "beta is the KL
  coefficient" is the difference between having used RLHF and understanding it.

## What Was Confusing

- **Horizon mismatch.** The infinite-horizon `V^pi` applied to a finite-horizon `J_T` caused
  two days of confusion (Day 3's "unbiased but worse" anomaly) before Day 4's time-indexed
  `V_t(s)` resolved it. Related slip: calling `G_t` "infinite horizon" twice.
- **Which powers of gamma live where.** `gamma^t` out front (the discounted-objective factor)
  versus `gamma^l` inside `A^GAE` (offset from `t`). Conflating them is the likeliest silent
  bug in `weights_gae`.
- **The direction of the lambda/bias relationship inverted under recall pressure** on Day 5's
  quiz -- stated correctly in a table on one page and backwards two pages later. Knowing a
  fact and being able to retrieve it cold are different states, and only the second one shows
  up in an interview.
- **`beta -> infinity` in the KL-regularized optimum.** Answered "uniform" instead of
  "`pi_ref`" -- substituting the entropy-bonus special case for the general result an hour
  after establishing that the entropy bonus *is* KL-to-uniform. Special cases learned first
  overwrite the general case.
- **The support restriction had to be pointed out twice** (asked for explicitly in the
  derivation prompt, still answered as an unrestricted argmax) before it stuck. It stuck
  properly only after the code asserted `pi*[a] == 0.0` exactly at three betas.

## What Needs Review

- [ ] **Retrieval under pressure, not comprehension.** Every Day-5 quiz error was a fact
      already demonstrably known (lambda/bias direction, `beta -> inf` limit). Fix: end-of-week
      cold recall on the *conclusions*, not the derivations -- one line per result, no
      whiteboard.
- [ ] Six/seven sources of REINFORCE variance -- now 6/7 (best of five attempts). Remaining
      miss: **reward-model noise**. And keep *sources* separate from *mechanisms*
      (Rao-Blackwellization is a cure, not a source).
- [ ] Off-policy-by-accident: 0.45 on Day 5, **down from 0.70 on Day 3**. Needs the causal
      mechanism (oversampling likely tokens -> mode reinforcement -> entropy collapse), why IS
      is not free (product of T ratios -> per-token clipping), that top-p is a *support*
      mismatch, and the entropy + KL instrumentation.
- [ ] PG regularity conditions (grad/sum swap; `supp(p_theta)` theta-independent; `R` not a
      function of theta).
- [ ] GRPO: the `(K-1)/K` x RLOO identity, and that the **std division** is the real bias.
- [ ] `b*` derivation cold.
- [ ] The two `gamma` powers in `weights_gae`, cold.

## Interview Takeaways

Ranked by how much they would distinguish a candidate:

1. **"Bootstrap bias is the state-varying part of the critic's error."** Wrong level costs
    variance, wrong ranking costs bias -- 0.0010 versus 1.5557, ~1500x. Therefore diagnose a
    value head by whether it *ranks* states correctly (explained variance), not by `|V - G|`.
2. **"Derive the optimal KL-regularized policy."** `pi* = pi_ref e^{r/beta}/Z`, `J* = beta log Z`,
    and the rearrangement whose `Z` cancels pairwise into DPO. This proves RLHF is understood
    as constrained optimization rather than as a recipe.
3. **Reverse KL confines the policy to the reference's support**, therefore RLHF elicits and
    cannot teach; therefore SFT precedes RL; therefore nucleus-sampled rollouts are
    unrepairable. One structural fact, three consequences across the pipeline.
4. **`beta` is an exchange rate**, `dE[r]/dKL = beta`, and over-optimization takes true reward
    *below* the reference (1.1286 at beta=0.16 -> -2.3250 at beta=0.02 versus 0.0270 for the
    reference). "Reward went up" is evidence of nothing.
5. **Unbiased != variance-reducing** (unbiased oracle baseline, +27% variance) and **low
    variance alone is worthless** (biased baseline, best variance of all, mean 38% wrong). Pair
    every variance number with an unbiasedness assertion.
6. **With an exact critic every lambda is unbiased**, so lambda = 0 strictly wins and
    lambda ~ 0.95 exists only as a hedge against a critic you don't trust.
7. **Green tests != correct code.** Two instances this week where 64/64 and 94/94 passed over
    real bugs, both because the wrong quantity preserved the property under test.

## Discipline Notes

**Structure.** The six-block 2-hour shape held on Days 1-3. Day 4 ran out of clock and dropped
its interview drills entirely; Day 5 absorbed them and still ran long, because the quiz grew to
8 questions and the coding block needed two review rounds. The staged harness (Day 3 process
change) was the single best change of the week -- being able to run one stage the moment a
function is written removed the "finish the file before testing anything" failure mode
completely.

**The behavioural gap that did not close.** *Answering the first clause of a question and
stopping* was flagged on Days 2, 3, 4, and 5. On Day 5 it was priced into the rubric in advance
(explicit cap at 0.5 for one-clause answers) and it happened anyway on P2 and on drill D2. It is
not a knowledge gap and it is not carelessness -- it is a default response length. Week 4 fix:
before submitting, count the clauses in the question and the paragraphs in the answer; if the
first number is larger, the answer is not finished. This is the highest-value single change
available, because the underlying knowledge is usually there.

**Not quoting own measured numbers.** Second-highest-value fix, and it improved: Day 5's Q6 and
drill D2 still omitted them, but they now appear unprompted in the derivations. The credibility
of a quantitative answer rests on the number being exact and reproducible -- and never rounded
in one's favour (3.15x is not "~4x").

**What to change in Week 4.**
1. Cap the cold quiz at 5 questions and put the carried-forward drills *first*, since drills
   are what gets dropped when the clock runs out.
2. Score the drills on the second half of the answer explicitly, as a separate line item, so
   the deficit is visible per-clause rather than as one number.
3. Keep the finite-MDP ground-truth pattern and the staged harness. Both earned their keep
   four times over.
4. Add one **cold recall of conclusions** block per day (no derivations, one line per result),
   because the Week-3 error pattern was retrieval, not understanding.
