# Week 04 Retrospective — RLHF and PPO for LLMs

## What Clicked

- **The enumerable token environment.** Copying Week 3's finite-MDP ground-truth pattern into a
  setting with real tokens, real EOS and real variable length was the highest-leverage decision
  of the week, exactly as the Week-3 retro predicted. It is what makes *"every unbiasedness
  claim in the repo is an assertion against exact enumeration, not a plot"* a true sentence, and
  that sentence is a statement about research taste rather than about RLHF.
- **The chained regression.** Forcing every Week-4 file to recover a Week-3 file by flipping
  switches turned "PPO generalizes the policy-gradient ladder" from a slogan into an
  assertion at **7.7e-15**, and it caught real bugs on three separate days. By Day 5 seven days
  of work collapse into one code path where each rung is checked against the previous one.
- **β as a dual variable.** The single best moment of the week was asking *"I thought β was a
  hyperparameter?"* — because the answer is that it is, in one of two formulations, and the
  other formulation makes the whole InstructGPT controller derivable rather than memorized. The
  sign, the multiplicative form, the positivity requirement and the meaning of the fixed point
  all fall out of one Lagrangian.
- **Verifying the controller against the closed-form frontier.** Two structurally independent
  routes to the same number (**1–5% agreement at four budgets**) is the strongest single check
  in the repo, and it is the kind of thing that is only possible because the environment is
  enumerable.
- **Measuring the caveat instead of stating it.** Dual ascent assumes the inner problem is
  solved to optimality; PPO doesn't. Rather than saying so, running the *same controller* on an
  exact primal and a PPO primal put a number on it: **74.7%**. That habit — turning a hedge into
  a measurement — is worth carrying into Week 5.
- **Two unplanted results on Day 5.** `value_divergence` firing on the top-p arm (EV → −5.19e9)
  reproduced D4's deadly-triad claim without being asked, and the top-p support ratchet
  (**40 → 9 reachable sequences**) turned "the excluded tail never recovers" into a stack trace.
  Both are better evidence than anything I designed.

## What Was Confusing

- **Which quantity the KL controller clips.** Got it wrong twice in a row — first the bounds,
  then the quantity — because `KL/δ` and `KL/δ − 1` describe the *same clipping set* under
  different bounds, and only one of them is centered on 0 where `1 + K_β(·)` needs it. The fix
  that would have saved both attempts: **trace `kl == kl_target` through the expression before
  running anything.**
- **Whether `sequence_reward_fn` returns a value or a function.** A closure factory is not
  obvious the first time; `return r` vs `return r(...)` is the whole distinction. Worth
  internalizing because the same pattern is everywhere in this repo.
- **Where the padding needs masking and where it doesn't.** Adding an unnecessary `[batch.mask]`
  guard to the KL introduced a **2.16× error** (a clean factor of B on any sampled batch) by
  flattening the batch axis. The safe form is `np.where`, which preserves shape.

## What Needs Review

- [ ] **Cold recall under a DIAGNOSTIC framing.** Six consecutive sessions without citing my own
      measured numbers when the question is phrased as "debug this" rather than "what did you
      find." The prescription is mechanical: **write the numbers at the top of the page before
      the sentence exists.**
- [ ] **Unanchored facts.** Both quiz blanks (adversarial selection, Gao's functional form) were
      the two questions with **no equation in them**. Retrieval is anchored to derivations, and
      facts without a derivation to hang on fall out. Needs a different rehearsal method.
- [ ] **Answering the second clause.** Cold recall Q1/Q3/Q5 and the concept checkpoint Q3 all
      lost points on clause two, not clause one. Fourth week running.
- [ ] Clipping ≠ trust region: **2 of 3 on Day 5** (up from 1 of 3 on D2 and D4). The missing
      third is still the iteration level — `π_old` resets. Store as **token / sample /
      iteration**.
- [ ] Length-controlled win rate — named the concept twice, named the *metric* zero times.

## Interview Takeaways

1. **"Every unbiasedness claim in this repo is an assertion against exact enumeration, not a
   plot."** The strongest single sentence available, and it is about research taste.
2. **REINFORCE → +baseline → +critic → GAE → PPO → KL shaping is one code path**, each rung
   asserted against the previous elementwise to 1e-15.
3. **Adaptive β is dual gradient ascent**, and its fixed point equals `dE[r]/dKL` on the
   closed-form frontier to 1–5% — plus the honest 74.7% gap when the primal is PPO instead of an
   argmax.
4. **Over-optimization vs optimization pathology**, and that **KL sees only half the taxonomy**.
5. **Early stopping is the estimator for the optimum**, with a run where shipping the final
   checkpoint costs **1.2522 gold** and lands *below the SFT model*.
6. **Off-policy-by-accident is repairable for temperature and provably not for top-p**, with
   the numbers.

## Discipline Notes

**Structure held, but the session ran ~14 hours of wall clock** across work and home. Full
blocks were delivered as the format preference requires, and no block was compressed for the
clock — which was right, but the two quiz blanks came late enough that fatigue is a plausible
contributor and worth watching.

**What worked:** the staged harness paid for itself again — implementing `kl_controller` first
(13 checks, five minutes) gave momentum before the 31-check `rlhf_iteration`. Verifying the
harness was *solvable* with a private reference implementation before handing it over was new
this week and caught four unreachable thresholds; keep doing that.

**What to change in Week 5:** the coding block is where the day's time actually goes, and most
of the debugging turned on **three name collisions** (`clip`, `lr`, `iter`) and **two shape
bugs** — not on the mathematics. A five-minute pre-flight before writing any function body —
*read the parameter list as a checklist, and check every name against what I might want to call
in that scope* — would have saved close to an hour. Also: **`--fast` must never change the
fixture's physics**, only its cost. A fast mode that changes the environment is a different
experiment wearing the same name, and its green checks tell you nothing about the real one.
