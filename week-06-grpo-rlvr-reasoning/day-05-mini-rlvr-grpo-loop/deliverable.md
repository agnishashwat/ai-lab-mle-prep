# Deliverable - Week 06 Day 05: Mini RLVR / GRPO Loop

## What I Built / Derived

- `code/mini_rlvr.py`: one exact, enumerable RLVR/GRPO system composing the Week 6 artifacts
  instead of reimplementing them.
- Three verifier fixtures: **honest** (`V=C`), **starved** (`V⊂C`), and **leaky**
  (`V\C≠∅`).
- Raw and calibrated curriculum sensors, closed-loop band and uniform prompt samplers, fixed
  reference KL, and deployable filter/ranker/vote evaluation.
- A ten-stage harness with **75 checks**, clean in both fast and full regimes under
  `-W error::RuntimeWarning`.
- A truth-boundary metamorphic test, paired-seed controller ablation, leaky-objective negative
  control, and five calibrated diagnostics.

## Core Takeaway

**Every `p` needs an event type.** `p_obs` is verifier acceptance and determines reward-group
deaths; `p_gold` is actual correctness and determines oracle pass@k and the curriculum target;
`p_answer[a]` is the answer distribution and determines voting. They coincide only under the
right verifier and selector assumptions. Calibrating `p_obs` can repair the controller's
coordinate, but it cannot repair an unsound reward objective: in the full leaky run, calibration
reduced sensor error by **0.3660 on average** while final-policy gold pass@8 fell **0.0051** and
the verifier–gold gap remained **0.2405**.

## Key Equations

```text
p_obs = γ + (1−α−γ)p_gold = γ + Jp_gold
p̂_gold,raw = (p̂_obs−γ)/J,  J=1−α−γ > 0

A_i = r_i − mean_j r_j,  r_i = accept(y_i), never gold(y_i)
L = −mean[A_i log π_θ(y_i|x)] + β·mean[k3(π_θ,π_ref)]
k3 = exp(log π_ref−log π_θ) − 1 − (log π_ref−log π_θ)

pass@k_x = 1−(1−p_x)^k, then average over prompts
p* = 1/(k+1) at a=b=1
P(dead | p_obs,G) = p_obs^G + (1−p_obs)^G
```

## Implementation Notes

- The panel is recorded **before** each update, so row 0 is exactly the reference checkpoint.
- Reported ablation endpoints freshly evaluate `theta_final` after the last update. The primary
  controller gains are normalized by the full `42×4×8 = 1,344`-rollout budget to a per-1,000
  rollout scale.
- Rollout uniforms are pre-generated for every `(step,prompt,draw)`, including inactive prompts.
  A separate sampler RNG preserves common random events across paired ablation arms.
- Prompt selection consumes only the previously available sensor. Held-out `α,γ` refresh only on
  the declared calibration interval; completed rollouts then update the free observed-rate EMA.
- Rewards and group advantages cross the NumPy/Torch boundary detached. Only sequence
  log-probabilities and the separate fixed-reference k3 term carry gradients.
- `evaluate_policy` computes pass@k per prompt before averaging, weights verifier precision by
  policy mass, and forbids gold from the ranker score.
- Calibration preserves the raw inverse, projects only the scheduling coordinate, records the
  projection mask, and rejects invalid probabilities, incompatible shapes, and `J≤0`.

## Failure Modes / Debugging Notes

- **Event collision:** using `p_gold` in the dead-group law or `p_obs` in oracle pass@k can make a
  mathematically correct formula answer the wrong question.
- **Objective/controller confusion:** calibration can place prompts in the right difficulty band
  while the verifier still positively rewards wrong answers.
- **Oracle leakage:** copying gold pass@k into a selector column or reading gold in a ranker score
  turns evaluation into an unavailable deployment policy.
- **RNG confounding:** drawing only after selection makes different controller arms consume
  different random streams, so their difference is not purely the intervention.
- **False significance:** the final-policy closed-loop advantage was **+0.0041 per 1,000
  rollouts** with paired spread **0.0021**. It remains inside two paired standard deviations, so
  the pre-registered report is a **null**, not a win.
- **Trend-gated diagnostics:** a policy can remain at high KL after KL stops rising. Diagnostics
  therefore read tail levels with thresholds calibrated between healthy and sick distributions.

## Measured Results

| Result, full regime | Measurement |
|---|---:|
| Harness | **75/75 checks**, 0 warnings/failures |
| Honest raw/calibrated θ difference, 5 seeds | **0.000000 exactly** |
| Starved sensor-error repair | mean **0.076343**, range **0.063569–0.090097** |
| Leaky sensor-error repair | mean **0.365958**, range **0.311036–0.401402** |
| Final-policy closed-loop gold pass@8 gain / 1,000 rollouts | **+0.014309** mean |
| Final-policy uniform gold pass@8 gain / 1,000 rollouts | **+0.010175** mean |
| Paired controller effect / spread per 1,000 rollouts | **+0.004134 / 0.002114 → null** |
| Leaky calibrated gold gain over 1,344 rollouts | **−0.005075** mean |
| Final leaky verifier–gold gap | **0.240452** mean |

The leaky step-0 panel shows why the event names matter: verifier pass@8 **0.999956**, oracle
gold pass@8 **0.764608**, filter accuracy **0.391510**, ranker accuracy **0.419953**, vote
accuracy **0.332450**, and verifier precision **0.383829**. One policy does not have one
"pass probability."

## Interview Answer

Prompt:

> How would you validate an end-to-end RLVR/GRPO system without letting held-out gold leak into
> training?

Answer:

First test the software boundary: mutate gold while holding θ, samples, RNG, verifier accepts,
and the data policy fixed; the policy update must remain bit-identical. Then mutate reachable
verifier accepts and require the update to change, which makes the first test non-vacuous.

That is distinct from contamination. For benchmark leakage, build perturbed twins and apply the
same perturbation to a clean post-cutoff or privately authored control. A drop only on the
suspect set is evidence of memorisation; a drop on both says the perturbation itself is harder.
Gold may populate the held-out panel and periodic sensor audit, but it must never enter rollout
reward or policy loss.

## Reusable Artifact

- Path: `week-06-grpo-rlvr-reasoning/code/mini_rlvr.py`
- How to reuse: run one stage by name, the reduced suite with `--fast`, or the full suite:

```bash
./.venv/bin/python -W error::RuntimeWarning \
  week-06-grpo-rlvr-reasoning/code/mini_rlvr.py --fast
./.venv/bin/python -W error::RuntimeWarning \
  week-06-grpo-rlvr-reasoning/code/mini_rlvr.py
```
