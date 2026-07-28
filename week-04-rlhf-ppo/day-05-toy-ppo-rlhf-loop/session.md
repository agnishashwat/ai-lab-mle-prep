# Week 04 Day 05 — The Toy PPO-RLHF Loop (Week 4 capstone)

## Objective

Assemble Days 1-4 into one end-to-end RLHF run on the enumerable token environment —
SFT reference → Bradley-Terry reward model → PPO with GAE, a learned value head, per-token
KL shaping, clipping, epochs and minibatches — with a full instrument panel and a **gold
eval** that decides which checkpoint you would actually ship.

Then the one genuinely new piece of theory this week has been building toward:
**adaptive \(\beta\) is dual gradient ascent.** In W3 D5 you derived the KL-constrained
optimum by writing a Lagrangian, and \(\beta\) appeared as the multiplier. InstructGPT's
KL controller,
\[
  \beta \leftarrow \beta\Big(1 + K_\beta\Big(\tfrac{\widehat{\mathrm{KL}}}{\mathrm{KL}_{\text{target}}}-1\Big)\Big),
\]
is a proportional controller on exactly that multiplier — i.e. the system is solving the
*constrained* problem \(\max \mathbb{E}[r]\) s.t. \(\mathrm{KL}\le\delta\) by alternating
primal ascent (PPO steps) with dual ascent (the \(\beta\) update). Once you see it that way,
every property follows: why it is stable, why \(K_\beta\) is a gain and not a learning rate,
why it can oscillate, and why "fixed \(\beta\)" and "target KL" are the primal and dual
statements of the same design.

Finish by writing `week-summary.md` and `retrospective.md`.

## Prerequisites

- All of Week 4 (`toy_rlhf.py`, `ppo_loss.py`, `token_ppo.py`, `failure_modes.py`).
- W3 D5's Lagrangian derivation and the two envelope identities
  (\(dJ^*/d\beta=-\mathrm{KL}\), \(d\mathbb{E}[r]/d\mathrm{KL}=\beta\)).
- Constrained optimization: Lagrangian duality, dual ascent, weak/strong duality (lightly).

## Required Reading

- **InstructGPT Appendix** — the adaptive-KL controller paragraph (it is short; the
  formula above and \(K_\beta\)). Also skim the hyperparameter table and notice how few of
  the values you would have guessed.
- Optional: PPO §5 / Spinning Up's "adaptive KL penalty" variant (PPO-Penalty), which is the
  same controller in the original paper — worth knowing that PPO shipped *both* a clip
  variant and a penalty variant, and RLHF ended up using both at once.

## 2-Hour Plan

### 0:00-0:06 — Cold recall of CONCLUSIONS (Days 1-4)

BT identifiability + the GRPO consequence; what \(L^{\text{CLIP}}\) guarantees and the three
gaps; what the value head regresses onto; EV vs MSE; the shaping-exactness result;
over-optimization vs optimization pathology and the instrument that splits them; the
lengthening-pays inequality; what your zero-reward control proved.

### 0:06-0:16 — Recall Quiz (Day 4, graded)

1. The taxonomy table from memory: four failures, mechanism and instrument for each. Which
   are invisible to KL?
2. Derive the lengthening-pays condition. You *predicted* a turnover \(\beta\) yesterday and
   then measured it — quote both numbers and the gap.
3. Off-policy-by-accident: give the mechanism, then your measured bias table, then explain
   why the IS fix worked for temperature and failed for top-\(p\). (This drill has run five
   times; today it should come out as numbers.)
4. Why does adversarial selection make held-out RM accuracy optimistic? What does that argue
   for?
5. Gao's functional form: what is on each axis, what shape, and what moves the turnover?

### 0:16-0:38 — Concept: what a real RLHF loop looks like

- **The loop, in order, with the reason for each ordering constraint.** Sample a batch of
  prompts → generate completions with the *current* policy at \(T{=}1.0\) untruncated →
  score with the RM (one scalar at EOS) → compute reference logprobs (**cache them: the
  reference never changes**, so this is one forward pass you never repeat) → shape the
  per-token reward with the KL term → compute values, then GAE advantages, then whiten →
  freeze `theta_old` and the advantages → \(K\) epochs × \(M\) minibatches of
  \(L^{\text{CLIP}} + c_1L^{\text{VF}} - c_2H\) → update \(\beta\) from the measured KL →
  log the panel → periodically run the gold eval. Note which quantities are computed
  **once per batch** (advantages, old logprobs, reference logprobs, value targets) versus
  **per minibatch step** (ratios, losses); getting that split wrong is the most common
  implementation bug and it is silent.
- **The instrument panel, and what each line is for.** Not a list — a set of paired
  readings. Mean RM reward *and* gold eval (divergence = Goodhart). KL to reference *and*
  \(\beta\) (both moving = the controller working; KL up with \(\beta\) pinned = drift).
  Entropy *and* unique-completion count (collapse). Mean length *and* length-controlled win
  rate (exploitation). Clip fraction *and* KL to old policy (step size health). Value EV
  *and* value loss (critic health). **A dashboard with only reward on it cannot detect any
  of the four failures**, which is the punchline of Day 4 stated operationally.
- **Checkpointing is part of the algorithm, not ops.** Because gold reward is concave in
  \(\sqrt{\mathrm{KL}}\) and bends down, the run's *final* policy is systematically the wrong
  one to ship. You checkpoint along the KL ladder and select on the gold eval. Say this
  plainly: **in RLHF, early stopping is not regularization, it is the estimator for the
  optimum.**
- **What this toy has and what it doesn't.** Has: real token-level logprobs, EOS and variable
  length, a learned RM from noisy preferences, a learned critic, exact ground truth. Lacks:
  a shared trunk (so \(c_1\) coupling is simulated rather than real), sampling cost (so the
  epoch-reuse economics are stated rather than felt), distributed anything, and a real
  gold eval (yours is a known function, which is a *luxury* no production run has — say so,
  because "we had ground truth so unbiasedness was an assertion" is the honest framing of
  the whole two weeks).

### 0:38-1:03 — Derivation: adaptive \(\beta\) as dual ascent

1. **Write the constrained problem.**
   \(\max_\theta \mathbb{E}[r]\) s.t. \(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\le\delta\).
   Lagrangian \(\mathcal L(\theta,\beta)=\mathbb{E}[r]-\beta(\mathrm{KL}-\delta)\).
   Observe that for **fixed** \(\beta\) this is exactly the W3 D5 objective up to the
   constant \(\beta\delta\) — so "penalty form" and "constraint form" are primal and dual
   views, and \(\beta\) was always a dual variable.
2. **Dual ascent.** \(g(\beta)=\max_\theta\mathcal L(\theta,\beta)\) is convex in \(\beta\)
   (a max of affine functions — say why that matters) and by the envelope theorem
   \(\dfrac{\partial g}{\partial\beta} = \delta - \mathrm{KL}(\pi^*_\beta\|\pi_{\text{ref}})\).
   This is the *same* envelope computation as \(dJ^*/d\beta=-\mathrm{KL}\) which you verified
   numerically to rtol 1e-4 — reuse it, don't rederive it. So the dual **descent** step is
   \[
     \beta \leftarrow \beta + \eta\big(\mathrm{KL}-\delta\big),
   \]
   i.e. **raise \(\beta\) when you are over budget, lower it when under.** That is the
   controller, in additive form.
3. **Recover InstructGPT's multiplicative form** by setting \(\eta = K_\beta\beta/\delta\):
   \(\beta\leftarrow\beta\big(1+K_\beta(\mathrm{KL}/\delta-1)\big)\). Two things to say about
   why the multiplicative version is the one that ships: it is **scale-free** (\(\beta\)
   spans orders of magnitude across tasks, and a multiplicative step is a step in
   \(\log\beta\)), and it keeps \(\beta>0\) automatically, which the additive form does not
   — and \(\beta\) is a multiplier on an inequality constraint, so \(\beta<0\) is not merely
   numerically bad, it is *infeasible*. Note the standard implementation also clips the
   relative error to \(\pm\)something for the same reason.
4. **Behaviour, and where it bites.** \(K_\beta\) is a **proportional gain**, so the usual
   control-theory reading applies: too small and the KL overshoots for many iterations
   before \(\beta\) catches up; too large and you get oscillation (\(\beta\) chases a *noisy*
   KL estimate — and recall from D5 that k1's estimate is negative on 20-44% of samples, so
   the measurement noise is not small). Pure proportional control also leaves **steady-state
   error**; the fix in control is an integral term, and the RLHF analogue is simply that
   nobody needs \(\mathrm{KL}=\delta\) exactly. Then the honest caveat: dual ascent's
   convergence guarantees assume the inner problem is solved to optimality, and PPO takes
   \(K\) clipped epochs. So this is a *justification*, not a proof — name that, because
   claiming more is the kind of overreach an interviewer will test.
5. **Fixed vs adaptive, when to prefer which.** Fixed \(\beta\) is a statement about the
   *price* of drift (\(d\mathbb{E}[r]/d\mathrm{KL}=\beta\), your measured identity) and lets
   the KL land wherever the reward makes it worth it. Adaptive \(\beta\) is a statement about
   the *budget* and gives comparable runs across tasks and RMs. Preference: adaptive when
   you are comparing runs or trust a KL budget more than a price; fixed when you are
   deliberately mapping the frontier (which is exactly what your D5 \(\beta\) sweep did).

### 1:03-1:45 — Coding: `week-04-rlhf-ppo/code/ppo_rlhf.py`

The week's capstone artifact. Staged harness; imports all four earlier files, so most stages
are integration + ablation rather than new math.

**TODOs:**

1. `sft_reference(env, ...)` — produce \(\pi_{\text{ref}}\) by supervised fitting to a small
   "demonstration" set, so the reference is *trained*, not handed over. Its gold reward is
   the bar every later run must clear.
2. `kl_controller(beta, kl, kl_target, K_beta, clip)` — the multiplicative update, with the
   relative-error clip and a positivity guard.
3. `rlhf_iteration(...)` — one full loop iteration in the order given above, returning a
   log record of every panel quantity as **floats**.
4. `train_rlhf(...)` — the outer loop, with checkpointing along the KL ladder.
5. `select_checkpoint(history, gold_eval)` — the shipping decision: pick the checkpoint at
   the gold turnover, and return *why*.
6. `ablate(...)` — the comparison grid.

**Checks and experiments:**

- **Chained regressions back through Week 3** (the pattern that has paid off every day):
  set \(\beta=0\), \(\epsilon=\infty\), one epoch, \(\lambda=1\), critic off — assert the
  update equals **W3 D2's REINFORCE gradient** elementwise. Then re-enable the baseline and
  assert W3 D3; \(\lambda<1\) and assert W3 D4's GAE. Four days of Week 3 and three of
  Week 4 must collapse into one code path, and each switch must recover the previous day
  exactly. If any link breaks, that is the bug.
- **Controller correctness.** (i) At \(\mathrm{KL}=\delta\) exactly, \(\beta\) is unchanged
  (fixed point). (ii) \(\beta\) is **strictly positive** for any input, including a negative
  KL estimate (k1 can go negative — assert the guard). (iii) Sign: over budget ⇒ \(\beta\)
  rises. Getting this backwards produces a run that looks like it is working and diverges,
  so assert the direction, not just the magnitude.
- **The controller vs the theory.** Run the controller to convergence at several
  \(\delta\), then assert the converged \(\beta\) matches the \(\beta\) that D5's *closed
  form* frontier assigns to that KL — i.e. \(\beta_\infty \approx d\mathbb{E}[r]/d\mathrm{KL}\)
  at \(\mathrm{KL}=\delta\). **This is the day's best check**: dual ascent's fixed point and
  the envelope identity are the same number arrived at two structurally independent ways
  (one iterative and sampled, one closed-form and exact).
- **Gain sweep.** \(K_\beta\in\{0.01,0.1,0.5,2\}\): measure settling time and overshoot;
  assert small gain ⇒ sustained overshoot, large gain ⇒ oscillation (variance of \(\beta\)
  over the last \(N\) iterations). Two failure modes of the same knob, both measured.
- **The headline ablation grid** — hold rollout budget fixed (W1 D5 discipline) and report
  gold reward, proxy reward, KL, entropy, length, EV, mean ± spread over ≥3 seeds:

  | run | expect |
  |---|---|
  | full loop (clip + KL + critic + adaptive \(\beta\)) | best gold; KL parked at \(\delta\) |
  | fixed \(\beta\) at the D5-optimal value | comparable gold, KL wanders |
  | no KL term (clip only) | proxy up, gold humps then falls; KL walks (Day 2's result) |
  | no clip (KL only, many epochs) | high-variance / unstable updates |
  | critic off (\(\lambda=1\), baseline only) | works, higher variance — quantify against Week 3's ladder |
  | REINFORCE + baseline, matched budget | the honest baseline PPO must beat |
  | rollouts at top-\(p\) 0.9 | Day 4's collapse, reproduced inside the full loop |

  Assert the full loop beats the SFT reference on **gold**, and that the no-KL run's *proxy*
  score exceeds the full loop's while its *gold* score is worse. That single inversion is the
  week in one assertion.
- **Checkpoint selection.** Assert `select_checkpoint` does **not** return the final
  iteration on the no-KL run (the whole point), and that it returns something near the gold
  argmax on all runs. Then the honest note in the deliverable: this selection used a gold
  eval you only have because the environment is synthetic.
- **`diagnose()` on every run** (Day 4's artifact): assert it fires the right failure for
  each ablation and stays silent on the healthy run.

### 1:45-1:55 — Interview Drill

Three, and treat these as final answers to be written down verbatim in the deliverable.

1. **"Design an RLHF system from scratch."** Ten minutes of material compressed to three:
   stages, the four models and the memory bill, what gets cached, the loop order, the
   instrument panel in *pairs*, checkpoint selection on a gold eval, and \(\beta\) fixed vs
   adaptive with a reason for your choice.
2. **"Why does PPO look the way it does?"** The chain, each link a *why*: reward is not
   differentiable and the sampling distribution depends on \(\theta\) ⇒ score function ⇒
   variance ⇒ reward-to-go, baselines, GAE ⇒ rollouts are expensive ⇒ reuse ⇒ off-policy ⇒
   importance ratios ⇒ ratio variance over \(T\) tokens ⇒ per-token ratios ⇒ ratios must
   stay near 1 ⇒ TRPO's constraint ⇒ too expensive ⇒ clip. Then, separately: the reward is a
   proxy ⇒ KL to reference ⇒ it must be dense ⇒ put it in the reward ⇒ its price should be a
   budget ⇒ adaptive \(\beta\). **Every link is something you derived and most are things
   you measured.**
3. **"When would you not use PPO?"** Wants the honest cost accounting — four models,
   critic bias, hyperparameter surface, and a rollout loop — versus DPO (Week 5: no
   rollouts, no RM, no critic; paid for with static off-policy data) and GRPO (Week 6: no
   critic; paid for with \(K\times\) rollouts and sequence-level credit). Name the one
   thing PPO gives that neither does: **per-token advantages from an online, on-policy
   signal**, which is what you need when the reward is dense or the horizon is long.

### 1:55-2:00 — Deliverable & Week Wrap

Write `deliverable.md` (with the three polished interview answers above), then
`week-summary.md` and `retrospective.md`. Next week: **Week 5 — DPO and preference
optimization**, which is already half-derived: W3 D5 gave you
\(r=\beta\log(\pi^*/\pi_{\text{ref}})+\beta\log Z\) and the observation that \(Z\) depends on
the prompt only, and Week 4 Day 1 gave you Bradley-Terry. Week 5 Day 2 is those two facts
composed.

## Checkpoint Questions

1. Write the RLHF loop in order and mark which quantities are computed once per batch versus
   per minibatch. Which mistake there is silent?
2. Derive the adaptive-KL controller as dual ascent. Where is the envelope theorem used, why
   is the shipped form multiplicative, and what does dual ascent assume that PPO violates?
3. Your controller converged to some \(\beta\) at KL budget \(\delta\). Predict that
   \(\beta\) from the closed-form frontier and say why agreement is meaningful.
4. Give the instrument panel in pairs and say which failure each pair detects. Why is a
   reward-only dashboard blind to all four?
5. Why is early stopping the *estimator for the optimum* in RLHF rather than a
   regularization trick?

## Frontier-Lab Interview Angle

This is the day that produces the artifact. "Walk me through a PPO-based RLHF system you
built, and tell me what broke" is the deep-dive prompt, and after this week the answer is a
real run with a real reward model, four planted-and-measured failure modes, a chained
regression proving the whole family (REINFORCE → +baseline → GAE → PPO) collapses into one
code path, and a KL controller whose fixed point you verified against a closed-form
identity. The strongest single sentence available to you: **"every unbiasedness claim in the
repo is an assertion against exact enumeration, not a plot"** — that is a statement about
research taste, which is the thing these interviews are actually screening for.
