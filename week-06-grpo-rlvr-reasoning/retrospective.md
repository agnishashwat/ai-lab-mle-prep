# Week 06 Retrospective - GRPO, RLVR & Reasoning Post-Training

## What Clicked

- **"Every `p` needs an event type" unified the week.** The same algebra can be correct and the
  system still wrong if `p_obs`, `p_gold`, or `p_answer[a]` is substituted at the wrong boundary.
  Day 5 made this executable rather than mnemonic.
- **The learner caught the benchmark aggregation trap unprompted.** The first response to
  "85% pass@64 and 30% pass@1" was that these cannot coexist for one homogeneous `p`. Correct.
  The repair was the next level: benchmark reports are `mean_x f(p_x)`, not `f(mean_x p_x)`.
- **Truth-boundary testing separated cleanly from contamination testing.** Mutating hidden gold
  with the data policy fixed tests a software dependency; perturbed twins plus a clean control
  test memorisation. The learner articulated both and the need for the control.
- **Calibration versus objective repair became concrete.** The leaky arm's sensor improved by
  **0.3660**, yet final-policy gold fell **0.0051** and the verifier–gold gap stayed **0.2405**. A corrected
  coordinate can steer toward the wrong objective more accurately.
- **Null reporting held.** The final-policy controller effect was positive on every seed, but
  `+0.0041` per 1,000 rollouts remained inside `2×0.0021`. The result stayed a null exactly as
  pre-registered.
- **Diagnostics are estimators with negative controls.** Every Day 5 rule is strict, reads a tail
  level, carries its measured value and threshold, and is asserted silent on healthy histories.

## What Was Confusing

- **Why inactive prompts receive random uniforms.** The initial interpretation was that those
  rollouts could be reused later. They are never reused; pre-generation indexes randomness by
  `(step,prompt,draw)` so different controller arms cannot shift each other's random stream.
- **What a production check asks for.** `dead_frac` was proposed after the pass@64 question. It is
  the right training-signal instrument and the wrong deployment instrument; emitted-answer
  accuracy under the actual selector is the missing quantity.
- **Level versus trend in diagnostics.** `kl_runaway` was first described as increasing KL.
  A policy can settle far from `π_ref`; requiring continued increase would turn a dangerous
  plateau into health. The tail KL level is the rule.
- **The second clause remains harder than the mechanism.** Several answers named the right
  object but missed the consequence or the measurement: verifier reward versus gold, ranker
  output versus pass@k, and common-random-number pairing.
- **Low-temperature rollouts are off-policy.** Importance weighting can correct full-support
  temperature sampling in expectation but compounds sequence-level variance; it cannot repair
  support truncated by top-p.

## What Needs Review

- [ ] State cold which event drives **dead groups**, **curriculum value**, **oracle pass@k**, and
      **voting**.
- [ ] Derive `KL(BoN_n‖π)=log n−(n−1)/n` from `U_max~Beta(n,1)` without notes.
- [ ] Explain why pre-generating inactive-prompt uniforms is experimental control, not cached data.
- [ ] Give the truth-boundary metamorphic test with its non-vacuity control and scope condition
      (freeze the data policy).
- [ ] Distinguish sensor repair from objective repair using the leaky-arm numbers.
- [ ] Finish the REINFORCE variance-source list cold: policy entropy, environment stochasticity,
      reward/verifier noise, prompt/task heterogeneity, trajectory horizon, and off-policy
      importance ratios; then map each source to a matching intervention.

## Interview Takeaways

- **One-liner:** every `p` needs an event type. An RLVR system is correct only when its reward,
  controller, selector, and evaluation all use the event they claim.
- **GRPO answer:** it replaces a learned critic with a same-prompt Monte Carlo baseline. The
  memory win buys dead-group and rollout-compute costs; it does not prove lower variance.
- **Verifier answer:** soundness is the guarantee. Incompleteness costs compute; one reachable
  false positive can dominate optimisation.
- **Pass@k answer:** it is capability composed with an oracle selector. Always report the actual
  emitted selector against held-out gold.
- **Curriculum answer:** control current per-prompt `p` in logit space around a metric-dependent
  set point, then respect the precision and dead-group walls.
- **Validation answer:** mutate hidden truth to test code boundaries; perturb examples with a clean
  control to test contamination. They are different experiments.
- **Know the null:** band tracking gained **+0.0041 per 1,000 rollouts** over uniform with paired
  spread **0.0021**; the honest report is unresolved, not better.

## Discipline Notes

**The full-block rule held.** Recall, integration, interview, pre-registration, coding, and
closeout all ran; coding did not displace the mandatory interview drill.

**The private-reference rule paid again.** The complete capstone was first validated at 75/75,
then converted into learner TODOs. That made the public harness reject a no-op trainer, gold
leakage, raw-sensor misuse, mean-then-pass@k evaluation, and oracle selector copying rather than
merely checking happy paths.

**Coding ownership drifted substantially on Day 5 and must be recorded.** The learner wrote the
mathematical core of calibration; the tutor repaired validation, broadcasting, `J≤0`, and
projection semantics. The tutor completed `evaluate_policy` on request, then implemented the
training loop and diagnostics during continuation after conceptual checks. The artifact is
valid, but retrieval value for TODOs 2–4 is limited. Week 7 should make the learner own one
complete diagnostic or ablation end-to-end before assistance.

**Interview ownership improved.** The two outstanding Day 3 entries and the Day 5 capstone entry
were written from the learner's answers, with repairs made as explicit diffs instead of replacing
the response with a model answer.

**Four days, four cut diagnostics, four reasons:**

| Day | Candidate cut | Measured reason |
|---|---|---|
| 1 | `length_drift` / `sig_starved` | the short-horizon instrument had no separating range |
| 2 | decaying component share | the semantics say it is mastery, therefore health |
| 3 | `coverage_collapse` | the measured system never exhibited it |
| 4 | `mean_p_rising` | it fired harder on healthy runs than sick ones |

Day 5 kept all five rules because the calibrated bench separated every mechanism and remained
silent on health. Do not invent a cut to preserve a pattern.
