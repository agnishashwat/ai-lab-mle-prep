# Week 01 Retrospective - Role Calibration, LM Foundations & PyTorch Fluency

## What Clicked

- Math from first principles: `∂CE/∂z = p − e_y` derived cleanly and cross-checked (autograd +
  finite diff). RL/optimization intuition is strong and transfers.
- Numerics reasoning: logsumexp stability, fp16-vs-bf16 exponent range, and the memory accounting.
- Eval taste came together on Day 5 — statistical significance, seed spread, "verify eval first."

## What Was Confusing

- The training loop was the hard part (Day 4): needed heavy scaffolding and hit every classic
  bug live (AdamW on model not params, flat warmup, inverted eval %, backward-inside-accum,
  missing /N, double-softmax). Loop fluency is not yet automatic.
- Backward placement relative to grad-accum regressed on the Day-5 quiz (said "after the loop") —
  the reason (memory, not math) needs to stick.

## What Needs Review

- [ ] Masked-mean static form `(x*mask).sum(-1)/mask.sum(-1)` (soft spot seen 3×)
- [ ] Training-loop invariants + order, cold, with backward INSIDE the accum loop
- [ ] Eval hygiene: both consequences of dropping `no_grad` and both of dropping `eval()`
- [ ] LR↔batch critical batch size
- [ ] Bootstrap CIs (Week 7)

## Interview Takeaways

- Lead debugging with "is my measurement real?" before diagnosing model behavior — senior tell.
- Report mean ± spread over ≥3 seeds at fixed compute; claim effects only when they beat noise.
- Know the silent failures cold (double-softmax, K-dim CE shape bug, clip hiding LR divergence).

## Discipline Notes

Kept the 6-block 2-hour structure across all 5 days; quiz-first + code-stubs-not-solved worked
well. Day 5 ran long because the toy task's flawed assumptions (fake overfit gap, clip-tamed LR)
surfaced during coding — but those became the best lessons. Next week (transformers + generation):
give the learner more loop reps early to convert loop mechanics from effortful to automatic.
