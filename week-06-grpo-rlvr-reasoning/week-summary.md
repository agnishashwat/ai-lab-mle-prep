# Week 06 Summary - GRPO, RLVR & Reasoning Post-Training

**Days 1–5 complete. 462 staged checks across the week**, all run under
`-W error::RuntimeWarning`: `grpo.py` **96**, `rlvr.py` **112**, `sampling.py` **73**,
`curriculum.py` **106**, and `mini_rlvr.py` **75**.

## The 5-Minute Version

Week 6 asks what changes when post-training uses groups of sampled responses and programmatic
verifiers. The answer is not "PPO without a critic." **GRPO replaces a learned estimate of
`E[r|x]` with the mean reward of `G` samples from the same prompt.** That baseline is cheap in
memory, but it is computed from the samples it centres: the estimator shrinks by `(G−1)/G`,
vanishes at `G=1`, and gives exactly zero advantage whenever a group is unanimous. For binary
reward, `P(dead)=p^G+(1−p)^G`; at `p=.95,G=8`, **66.34%** of groups are dead. Improving the model
can therefore degrade its own training signal.

RLVR removes the learned reward model only if the verifier's accept set is actually trustworthy.
A verifier is a **set**, not an accuracy scalar. `V⊆C` is soundness and guarantees
`p_obs≤p_gold` for every policy; `C⊆V` is completeness and its failure costs rollout compute.
One false positive is qualitatively different: if `V\C` is non-empty, optimisation can move all
mass there and make the verifier–gold gap approach 1. The hacked and solved runs can be
bit-identical on every reward-derived column, so held-out gold is the only discriminating
instrument.

Evaluation then adds another event boundary. **Pass@k is capability composed with an oracle
selector that deployment does not have.** An oracle tends to 1 when any correct mass exists; a
filter tends to verifier precision; majority vote tends to 1 only when the correct answer is
modal. In one run maj@k fell **0.3563→0.2764** while pass@101 was 1. Best-of-n is also a KL move:
`KL(BoN_n‖π)=log n−(n−1)/n`, or **3.1745 nats at n=64** without ties. The selection rule alone
made filter hacking **9.09%** versus ranker hacking **90.12%** at n=256.

The curriculum is a feedback controller. Its measured variable is current per-prompt success,
not a static difficulty label. If local progress is `p^a(1−p)^b` and the reported metric is
pass@k, rollout value is `p^a(1−p)^(b+k−1)`, giving
`p*=a/(a+b+k−1)` and `1/(k+1)` at `a=b=1`. At pass@64, using `G=8` at that set point makes
**88.34%** of groups dead. The verifier does not tilt this optimum; Youden's
`J=1−α−γ` multiplies useful signal. False positives instead impose a precision floor
`p_floor=γ(1−τ)/[τ(1−α)+γ(1−τ)]`.

Day 5 composes these facts into one exact loop. Its rule is: **every `p` needs an event type**.
`p_obs` drives reward-group deaths, `p_gold` drives held-out oracle evaluation and curriculum
value, and `p_answer[a]` drives voting. Gold mutation leaves the policy update bit-identical when
the data policy is fixed; accept mutation changes it. Calibration repairs the scheduling
coordinate but not an unsound objective: in the leaky arm it reduced sensor error by **0.3660**
while final-policy gold pass@8 fell **0.0051** and the verifier–gold gap stayed **0.2405**. The
final-policy paired closed-loop effect was **+0.0041 per 1,000 rollouts** with spread **0.0021**,
so the pre-registered conclusion is a **null**.

## Key Concepts (crisp)

- **The group is the critic.** GRPO changes the baseline, not the policy-gradient objective.
- **Dead groups are exact zeros.** They are missing optimisation budget, not weak gradients.
- **A verifier is an accept set.** Soundness is a support guarantee; completeness is a compute
  property.
- **Verifier rates are policy-dependent.** `α`, `γ`, and precision are probability-mass-weighted
  under the current policy, not sequence counts.
- **Pass@k names an oracle.** A deployable filter, ranker, or vote is a different estimand.
- **A curriculum is a controller.** Measure current `p`, control in logit-`p`, and state which
  precision/dead-group wall binds.
- **Whitening is itself a curriculum.** Dividing by group standard deviation changes the
  prompt weighting by up to roughly `G/4`.
- **Calibration is not objective repair.** It can estimate gold difficulty for scheduling while
  reward still reinforces false positives.
- **Truth-boundary and contamination tests are distinct.** One tests software dependencies; the
  other tests memorisation and requires a clean perturbation control.
- **Nulls are results.** Report paired effect against paired spread; do not manufacture a win.

## Key Equations

```text
GRPO advantage      A_i = r_i − mean_j r_j
estimator mean      E[g_hat] = (G−1)/G · grad J
dead-group law      P(dead) = p^G + (1−p)^G

verifier sensor     p_obs = γ + (1−α−γ)p_gold = γ + Jp_gold
calibration         p_gold = (p_obs−γ)/J,  J=1−α−γ > 0
precision           P(gold|accept) = (1−α)p / [(1−α)p + γ(1−p)]

pass@k              1 − (1−p)^k                         per prompt, then mean
unbiased estimate   1 − C(n−c,k)/C(n,k)
metric weight       d pass@k/dp = k(1−p)^(k−1)
BoN KL              log n − (n−1)/n                     continuous, no ties

curriculum value    p^a(1−p)^(b+k−1)
set point           p* = a/(a+b+k−1)                    → 1/(k+1)
precision floor     γ(1−τ) / [τ(1−α)+γ(1−τ)]
useful signal       2(G−1)p(1−p)J
```

## What I Implemented

| Day | Artifact | Checks | What it establishes |
|---|---|---:|---|
| 1 | `grpo.py` | 96 | group baseline, shrinkage, dead groups, KL placement, diagnostics |
| 2 | `rlvr.py` | 112 | verifier set semantics, soundness, hacking, multi-level rewards |
| 3 | `sampling.py` | 73 | pass@k estimator, filter/ranker/vote, BoN KL, metric weighting |
| 4 | `curriculum.py` | 106 | set point, precision floor, whitening tilt, closed-loop controller |
| 5 | `mini_rlvr.py` | 75 | composed loop, truth boundary, paired ablations, calibrated panel |
| | **total** | **462** | |

## Failure Modes I Now Understand

| Failure | Observable mechanism | Correct response |
|---|---|---|
| dead reward groups | high `dead_frac`, split by low/high `p_obs` | scaffold hard prompts or retire solved prompts; raise G only if worth the compute |
| incomplete verifier | `p_obs<p_gold`, γ=0 | expand verifier coverage or pay more rollouts |
| unsound verifier | verifier improves while held-out gold stalls/falls | fix/remove false-positive support; calibration and compute cannot repair it |
| broken loss/optimizer | alive groups and useful verifier covariance, but θ does not move | inspect detached advantage, gradient path, LR, and update plumbing |
| selector mismatch | oracle pass@k materially exceeds emitted-answer accuracy | evaluate the production filter/ranker/vote against held-out gold |
| stale curriculum | selected prompts drift outside the useful band | close the loop on current per-prompt estimates in logit space |
| KL runaway | high tail KL even if it no longer rises | restore fixed-reference pressure or stop; use a level, not a trend gate |
| contamination | suspect perturbed-twin drop with clean control unchanged | retire/rebuild items and report on the same surviving subset |

Four candidate diagnostics were deliberately cut for four different measured reasons:
`length_drift` lacked instrument resolution; decaying component share was a health signature;
`coverage_collapse` never occurred in the measured system; `mean_p_rising` fired harder on
healthy runs than sick ones.

## Open Questions / To Revisit

- The controller result is unresolved: **+0.0041 per 1,000 rollouts with 0.0021 paired spread** on
  five toy seeds. Scale prompt count, horizon, and parameter sharing before claiming a curriculum
  benefit.
- `α` and `γ` were available from exact held-out enumeration. At scale, how much audit data is
  needed to track their policy dependence without making calibration too stale?
- The tabular prompts own independent parameter blocks. Shared representations may transfer
  updates between active and inactive prompts and break the clean budget interpretation.
- Which deployable selector should be trained jointly with the policy, and how should its
  inference-time KL move be included in the original optimisation budget?
- Advantage whitening's implicit curriculum was measured exactly here. How does it interact with
  batch-level normalisation, reward clipping, and distributed rollout aggregation in real stacks?
