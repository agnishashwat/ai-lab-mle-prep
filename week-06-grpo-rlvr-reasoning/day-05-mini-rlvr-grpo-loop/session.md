# Week 06 Day 05 — Mini RLVR / GRPO Loop: every `p` needs a type

## Objective

Build the week's deliverable: one mini RLVR system that generates a group of completions,
scores them with a verifier, computes group-relative advantages, updates the policy, chooses
the next prompts with a closed-loop curriculum, and evaluates the result with both an oracle
metric and a deployable selector.

Day 5 adds no new algorithm. It asks whether the four algorithms already built agree when they
share one process:

```
D1  GRPO estimator       group reward -> centered/whitened advantage -> policy update
D2  RLVR verifier        true correctness C versus verifier accept set V
D3  sampling/evaluation  pass@k versus the selector that actually emits one answer
D4  curriculum           choose prompts using a feedback estimate of current difficulty
```

The capstone is successful only if each boundary is explicit, testable, and impossible to cross
silently.

## Read this first: the collision

The same letter `p` has meant different events all week:

```
p_gold    = P(y ∈ C | x, π_θ)             actual correctness
p_obs     = P(y ∈ V | x, π_θ)             verifier acceptance; the reward process
p_oracle  = 1 − (1−p_gold)^k              pass@k; an oracle can find any success
p_deploy  = P(the real selector emits C)  filter / ranker / majority-vote outcome
```

They coincide only under extra assumptions. In particular,

```
p_obs = p_gold(1−α) + (1−p_gold)γ
      = γ + (1−α−γ)p_gold
      = γ + J p_gold
```

where `α = P(reject | correct)`, `γ = P(accept | incorrect)`, and `J = 1−α−γ`.

That one line gives the day's thesis:

> **The hard part of an RLVR stack is not writing the update. It is keeping the event types
> straight across modules.** The dead-group law consumes `p_obs`; the curriculum set point and
> pass@k derivative were derived for `p_gold`; and deployment consumes neither unless the selector
> is an oracle. A loop that feeds one into another because every array is named `p_hat` can run,
> learn, and report a better number while being wrong at all three boundaries.

If `J > 0` and the verifier rates are valid under the **current** policy, the sensor can be
calibrated:

```
p̂_gold,raw = (p̂_obs − γ̂) / Ĵ
```

That is an estimator, not a licence to hide impossible values. Finite samples can put the raw
estimate outside `[0,1]`; the controller may project it only if it records that projection.
And `α̂, γ̂` are not constants of the checker: D2 measured that they move when the policy moves.
A calibration measured once at step 0 is an open-loop controller wearing a statistical name.

## Non-negotiable system contract

1. **Truth boundary:** `gold` is held-out instrumentation. In the metamorphic test, mutating
   `gold` while holding θ, sampled indices, RNG, `accept`, and the active-prompt/data policy fixed
   must leave every training gradient and parameter update unchanged.
2. **Autograd boundary:** sampled rewards and group advantages are detached data. The gradient
   flows through sequence log-probabilities only.
3. **Policy boundary:** `π_old` generated the batch; `π_ref` anchors the KL. They are different
   policies and cannot share a variable because both are called "old."
4. **Time boundary:** panel row `t` describes the policy that generated batch `t`, before the
   update. The controller for step `t+1` may use completed rollouts through `t`, never future
   outcomes.
5. **Selector boundary:** pass@k is reported as oracle coverage. A deployable result must name
   and measure the selector that emits one answer.
6. **Prompt boundary:** means go outside nonlinear functions. Compute pass@k per prompt and then
   average; never evaluate `1−(1−mean(p))^k`.
7. **Failure boundary:** a diagnostic is silent on healthy controls, calibrated across paired
   seeds, and carries the number that fired in its reason string.

## Prerequisites

- **D1:** `group_advantage`, exact `(G−1)/G` shrinkage, annihilation of per-prompt constants,
  dead groups, `/std` reweighting, the inert clip at `μ=1`, KL estimators, and `train_grpo`.
- **D2:** `V ⊆ C` versus `C ⊆ V`, `p_obs`, policy-weighted verifier rates, the selection law,
  the Rényi form of dead groups, and the hacked-run/solved-run observational equivalence.
- **D3:** unbiased pass@k, `∂pass@k/∂p`, filter versus ranker, majority vote on answer IDs,
  best-of-n KL, and the support-preserving temperature result.
- **D4:** `p* = a/(a+b+k−1)`, the precision floor, `G*`, log-odds control, `BandTracker`,
  the implicit curriculum from whitening, and contamination as saturated dead budget.
- **W3 D2:** the seven-slot REINFORCE variance-source inventory. This is re-tested today because
  it remains a standing retrieval item, not because Day 5 introduces another estimator.

The implementation imports all four Week 6 modules. If the capstone has to copy a prior formula
instead of importing it, stop and inspect the interface. Integration debt is a finding.

## Required Reading

No new paper. Read only your own evidence:

1. `shared/numbers-card.md`, **Week 6 Day 4**, for Block 0.
2. D1-D4 `deliverable.md` files, but only after cold recall.
3. `rl-post-training-qa.md` cold cards for the mandatory interview block, but only after answering.

## Session rules

- Every block is delivered in full. Wall-clock time does not compress the lesson.
- All maths is plain Unicode or fenced text. No LaTeX.
- Recall slots are pre-printed, typed, number-first, and identify the function and varied argument.
- Cold recall tests prior material only.
- Interview answers are spoken first, then written as a three-bullet skeleton while hot.
- The learner writes the bank entries. At most one may be supplied as a worked model.
- Prefer a measured null or a seed bracket to a manufactured headline.
- Run the capstone under `-W error::RuntimeWarning`.

## Plan

### Block 0 · 0:00–0:08 — Day 4 numbers-card drill

Open `shared/numbers-card.md`, read **Week 6 Day 4** out loud from start to finish, then close it.
No notes and no scrollback for the three prompts below.

#### 1. Metric set point and group-size bill

Functions: `p*(k;a,b) = a/(a+b+k−1)` and
`P(dead | p,G) = p^G + (1−p)^G`. Vary `k`; hold `(a,b)=(1,1)`, `G=8`, and target
alive fraction `0.90` fixed.

```
p*(k=1/4/8/16/64): [five probabilities, 4 dp]
  ______  ______  ______  ______  ______

exact minimum G for 90% alive at those same k values:
  [five positive integers] ______  ______  ______  ______  ______

P(dead | k=64, G=8, p=p*): [percentage, 2 dp] ______

one-sentence consequence:
  "the metric I report sets ________________________________________________"
```

#### 2. Verifier floor and the lost signal

Functions:

```
p_floor = R/(1+R),  R = γ(1−τ)/[τ(1−α)]
retention(k) = p_floor(1−p_floor)^k / [p*(1−p*)^k]
```

Vary `k`; hold `γ=0.01`, `α=0.10`, `τ=0.10`, and `(a,b)=(1,1)` fixed.

```
J = 1−α−γ: [a probability, 2 dp] ______
p_floor: [a probability, 6 dp] ______
the exact crossover k where p_floor = 1/(k+1): [a number, 1 dp] ______
retention at k=64: [a percentage, 2 dp] ______
the constraint that binds beyond the crossover: [noun phrase] ______________________
```

#### 3. What the controller experiment actually measured

The functions are the learned progress law `Δp ∝ p^a(1−p)^b` and the controller coordinate
comparison. The stack, seeds, and prompt set are held fixed.

```
fitted (a,b): [two numbers, 3 dp] (______, ______)
fit quality R²: [a number, 4 dp] ______

50%-of-peak band narrowing from k=1 to k=64:
  in p: [factor, 2 dp] ______×
  in logit(p): [factor, 2 dp] ______×

the rule cut from diagnose_curriculum: [exact rule name] __________________
measured reason for cutting it:
  [complete sentence naming healthy and sick ordering] ______________________________
```

### Block 1 · 0:08–0:32 — Whole-week cold recall

No notes. The number is the first slot in every answer; mechanism comes second.

#### 1. D1 — the estimator

Hold binary success probability `p=0.95` and `G=8`.

```
P(dead): [percentage, 2 dp] ______
minimum G for 90% alive: [integer] ______
finite-G shrinkage of the expected centered estimator: [exact factor in G] __________
the object a group baseline annihilates: [mathematical noun phrase] _________________
why a KL term inside the reward becomes open-loop:
  [one causal sentence] _____________________________________________________________
```

#### 2. D2 — verifier semantics

Hold the policy fixed while comparing verifier sets.

```
the containment that guarantees v_pass ≤ gold_pass for every policy:
  [set relation] __________

sup(v_pass − gold_pass) when V \ C contains one reachable false positive:
  [a number, 1 dp] ______

gold pass of the measured leaky run: [probability, 4 dp] ______
verifier pass of that run: [probability, 4 dp] ______

why "97% accurate" is not a well-formed verifier claim:
  [one sentence naming both dependencies] ___________________________________________
```

#### 3. D3 — metric and selector

Hold `n=256`, `γ=0.01`, `α=0.10`, and `p_gold=0.10`.

```
unbiased pass@k estimator: [formula in n,c,k] _______________________________________
direction of the plug-in bias and theorem naming it: [direction + theorem] __________

filter hack rate at n=256: [percentage, 2 dp] ______
ranker hack rate at n=256: [percentage, 2 dp] ______
ratio: [factor, 2 dp] ______×

KL(BoN_64 || π), continuous-score idealisation: [nats, 4 dp] ______
```

#### 4. D4 — curriculum

Hold `(a,b)=(1,1)`, report pass@64, and use the measured verifier
`(α,γ,τ)=(0.10,0.01,0.10)`.

```
metric-optimal p*: [probability, 4 dp] ______
legal p after applying the precision floor: [probability, 4 dp] ______
signal retained: [percentage, 2 dp] ______
dead fraction there with G=8: [state WHICH p you use, then a percentage] ____________

the two walls and their opposite data fixes:
  low wall: [name] __________  fix: _________________________________________________
  high wall: [name] _________  fix: _________________________________________________
```

#### 5. The perennial estimator question

Function: score-function policy gradient. Hold the environment and objective fixed.

```
1. [variance source, not a cure] ____________________________________________________
2. [variance source, not a cure] ____________________________________________________
3. [variance source, not a cure] ____________________________________________________
4. [variance source, not a cure] ____________________________________________________
5. [variance source, not a cure] ____________________________________________________
6. [variance source, not a cure] ____________________________________________________
7. [variance source, not a cure] ____________________________________________________

the one that is identically zero for an autoregressive LLM environment, and why:
  [source + one causal sentence] ____________________________________________________

one item that is a variance-reduction MECHANISM rather than a source:
  [name + what source it attacks] ___________________________________________________
```

### Block 2 · 0:32–0:50 — First principles: type the events before wiring the loop

The capstone has six stages, and the output type of each stage is part of its API:

```
prompt policy q_t(x)
  -> active prompt IDs
  -> π_θ(.|x) samples G sequence IDs
  -> verifier 1[y ∈ V_x]                         reward, shape (B,G)
  -> group_advantage(reward)                     detached A, shape (B,G)
  -> policy loss using log π_θ and KL(π_θ||π_ref)
  -> θ_{t+1}

the SAME rollouts also feed:
  accept counts c_obs/G                         sensor for p_obs
  held-out labelled probe                      calibration for p_gold
  per-prompt pass@k on gold                     oracle evaluation
  deployed selector on answer_id / verifier    shippable evaluation
  -> q_{t+1}(x)                                 next prompt policy
```

#### The event audit

Every scalar gets an event subscript:

| quantity | event | allowed consumer |
|---|---|---|
| `r` | `y ∈ V` | advantage and loss |
| `p_obs` | `y ∈ V` | dead-group forecast; raw sensor |
| `p_gold` | `y ∈ C` | curriculum target; gold pass@k |
| `precision` | `y ∈ C | y ∈ V` | verifier/selector risk |
| `p_answer[a]` | `answer_id(y)=a` | majority vote |
| `π_old` | policy that sampled this batch | importance ratio |
| `π_ref` | fixed anchor policy | KL penalty |

The wrong-level trap is now executable: `p_obs` and `p_gold` are both probabilities with the same
shape, so a type checker cannot save us. Only names, contracts, and metamorphic tests can.

#### The three independent questions

Do not collapse these into "did reward improve?":

1. **Did the optimiser follow its reward?** Check update direction, loss, KL, and dead groups.
2. **Was the reward aligned with truth?** Check `V ⊆ C`, policy-weighted precision, and held-out
   gold. A solved run and a hacked run are observationally equivalent without this column.
3. **Does the eval correspond to deployment?** Compare oracle pass@k with the actual filter,
   ranker, or vote. Better oracle coverage need not produce a better emitted answer.

Each question has a different fix. A controller calibration cannot repair an unsound objective;
a sound verifier cannot repair an oracle-only eval; and a realistic selector cannot create
gradient on dead groups.

### Block 3 · 0:50–1:00 — Derivation and test design

#### 1. Sensor calibration

Starting from `p_obs = γ + J p_gold`:

```
p_gold = (p_obs − γ)/J,  J > 0
```

Three edge cases are requirements, not footnotes:

- `J = 0`: the checker is independent of truth; the inverse does not exist.
- `J < 0`: the checker is anti-informative; inverting it may estimate truth, but using its raw
  reward trains in the wrong direction. Sensor repair and objective repair are different.
- finite-sample `p̂_obs`: a raw inverse outside `[0,1]` must be surfaced; projection for scheduling
  is explicit and counted.

#### 2. A metamorphic proof of the truth boundary

For a fixed seed, `θ`, sampled indices, and `accept`:

```
update(gold = G₁, accept = V) == update(gold = G₂, accept = V)   bit-for-bit
```

If this fails, held-out truth leaked into training. The complementary non-vacuity check mutates
`accept` while keeping `gold` fixed and requires the update to change.

#### 3. A compositional dead-group check

The probability entering D1's binary law is **not** `p_gold`:

```
P(dead | verifier reward) = p_obs^G + (1−p_obs)^G
```

The capstone compares this prediction to empirical dead groups per prompt. It also prints the
incorrect `p_gold` prediction as a negative control and requires separation on an incomplete or
unsound checker.

#### 4. An evaluation check

For each prompt, compute `1−(1−p_gold,x)^k`, then average. The mean-then-transform result is
printed only as a negative control. The deployed selector is measured separately and never
aliased to the oracle column.

### Block 4 · 1:00–1:25 — Interview drill (mandatory, before coding)

Answer each out loud, cold, in 60–120 seconds. No notes or scrollback. Immediately after speaking,
write a three-bullet skeleton: **thesis / mechanism / number**. Model answers come only after
yours, as a diff.

#### Q1. How would you design the data curriculum for an RLVR run?

```
NUMBER first: [one set-point number and one group-size/dead-group number] ____________
THESIS: [one sentence using "controller"] ___________________________________________
MECHANISM: [metric derivative × progress law] _______________________________________
CONSTRAINTS: [two walls, opposite fixes] ____________________________________________
HIDDEN CURRICULUM: [what /std does] _________________________________________________
```

#### Q2. Your GRPO reward has been flat since step 0. Diagnose it, cheapest measurement first.

```
NUMBER first: [one measured dead-group or verifier number] ___________________________
FREE PANEL, in order: [three quantities] _____________________________________________
four branches: [cliff / saturation / verifier / optimiser, with discriminant] ________
EXPENSIVE LAST: [measurement requiring gold] _________________________________________
```

#### Q3. How do you know a reasoning benchmark is not contaminated, and what does contamination
do to training?

```
NUMBER first: [the k=1 and k=64 inflation at c=.05,p=.30] _____________________________
METHOD: [perturbed twin] _____________________________________________________________
CONTROL: [what must stay fixed] ______________________________________________________
TRAINING COST: [one sentence using "saturated dead group"] ___________________________
```

#### Q4. Name the REINFORCE variance sources, and keep sources separate from cures.

```
seven source slots: [seven short noun phrases]
  1. __________  2. __________  3. __________  4. __________
  5. __________  6. __________  7. __________

identically zero for an LLM environment: [source + why] ______________________________
Rao-Blackwellization is a: [source or cure, plus target] ______________________________
state-value baseline acts on: [source + expectation identity] _________________________
reward-model noise needs: [one valid mitigation, not "use a critic"] _________________
```

**Block deliverable:** complete the two `YOURS TO WRITE` D3 entries in
`interview-bank/rl-post-training-qa.md`. The D4 entries were tutor-written and have no remaining
retrieval value; today's cold answers are the repair.

### Block 5 · 1:25–1:40 — Pre-register the capstone experiment

The experiment separates three axes that are often bundled:

| axis | control | intervention |
|---|---|---|
| verifier semantics | honest (`V=C`) | starved (`V⊂C`) or leaky (`V\C≠∅`) |
| curriculum sensor | raw `p̂_obs` | calibrated `p̂_gold` |
| prompt policy | uniform/static | closed-loop log-odds band |

Use at least five **paired** seeds. Same task, initial policy, rollout budget, and random streams
within a pair.

Predictions, written before the run:

1. **Honest negative control:** raw and calibrated sensors are identical when `α=γ=0`.
2. **Starved checker:** soundness keeps the objective safe, but raw `p̂_obs` mis-targets the
   gold set point; calibration can repair prompt choice, not the lost rollout efficiency.
3. **Leaky checker:** calibration may repair the curriculum coordinate while the loss still
   optimises hacks. Gold can remain flat or fall while verifier pass rises. This is not a
   calibration failure; it is an objective failure.
4. **Static versus closed loop:** compare in-band budget share and gold metric gain per rollout.
   An effect inside paired-seed spread is a null.
5. **Oracle versus deployed selector:** pass@k is an upper bound for a selector you do not have,
   not a shipping metric. Report the gap and its direction.

Primary endpoint:

```
[mean_x pass@k_gold(x; theta_final) - mean_x pass@k_gold(x; theta_0)]
    * 1,000 / (steps * n_active * G)
```

`theta_final` is evaluated after the final update. Training-panel rows remain pre-update snapshots;
the endpoint is a separate fresh evaluation so the last update is not omitted.

Secondary endpoints:

```
verifier pass@k, deployed-selector accuracy, dead rollout share, in-band budget share,
policy-weighted precision, KL, entropy, and sensor projection rate
```

Cut rule: if a condition cannot be silent on every healthy seed or its healthy/sick
distributions overlap, remove it. Record the measured reason; do not rescue it with another
conjunct.

### Block 6 · 1:40–2:45 — Coding: `code/mini_rlvr.py`

The file is the week's deliverable. It imports, rather than reimplements:

```
grpo.py       SeqTable, group_advantage, score/log-prob helpers, exact KL/entropy
rlvr.py       Checker, verifier_rates, p_observed, dead_fraction_levels
sampling.py   pass_at_k_exact, selector machinery, answer-level evaluation
curriculum.py precision_floor, set_point, BandTracker/UniformSampler
```

#### Public objects

```
MiniTask
  env, tab, gold, accept, answer_id, theta0

MiniCfg
  steps, group, lr, beta, k_eval, n_active, seed, adv_mode,
  sensor_mode, sampler_mode, selector_mode, calibration_interval

MiniHistory
  step, p_obs, p_gold, p_hat_gold, active, verifier_passk, gold_passk,
  deployed_accuracy, precision, dead_frac, signal, kl, entropy,
  in_band, projection_frac
```

#### Learner TODO 1 — `calibrate_gold_rate`

Input: observed rates and calibrated `α,γ`. Output: raw gold-rate estimate, projected scheduling
estimate, and a projection mask.

Contract:

- validate probabilities and matching shapes;
- reject `J ≤ 0` with an explicit error;
- do not silently clip;
- exact inverse on noiseless inputs;
- honest-checker identity as a negative control;
- round-trip through `p_observed` to `1e-12`.

#### Learner TODO 2 — `evaluate_policy`

Return a typed evaluation object containing:

- per-prompt `p_obs` and `p_gold`;
- mean per-prompt oracle pass@k on both events;
- policy-weighted verifier rates;
- deployed filter/ranker/vote accuracy, with the selector named;
- exact KL and entropy.

The function must not compute pass@k at a mean probability, and a selector result must never be
filled from the oracle column.

#### Learner TODO 3 — `train_mini_rlvr`

One coherent loop:

1. record the panel **before** the update;
2. choose active prompts from information available through the prior step;
3. sample `G` completions from `π_old`;
4. score with `accept`, never `gold`;
5. compute detached group advantages;
6. update with sequence log-probabilities plus KL to fixed `π_ref`;
7. update the sensor from completed rollouts;
8. calibrate only at the configured held-out interval;
9. preserve paired randomness across ablation arms.

Dead groups skip the policy-gradient term exactly. The KL controller still remains explicit; do
not smuggle it into the centered reward, where D1 proved it can be annihilated.

#### Learner TODO 4 — `diagnose_mini_rlvr`

Calibrate, do not guess, thresholds for:

- `dead_budget` — too much rollout budget returns exactly zero advantage;
- `sensor_mismatch` — the controller's estimated gold difficulty misses the held-out probe;
- `reward_hacking` — verifier metric improves while held-out gold does not;
- `selector_gap` — oracle pass@k materially exceeds the deployable selector;
- `kl_runaway` — policy leaves the reference-supported region.

Every report is `(name, detail)` with the measured tail statistic. The panel is silent on the
healthy fixture. Fire sets must be non-collinear across the fixture bench.

#### Ten staged checks

1. `stage_contract` — shapes, binary masks, answer IDs, probability normalisation, fail-fast.
2. `stage_sensor` — `p_obs = γ + Jp_gold`, inverse round-trip, `J≤0`, explicit projections.
3. `stage_rollout` — sampled reward equals `accept[idx]`; empirical dead rate matches the
   `p_obs` law and separates from the wrong `p_gold` law.
4. `stage_truth_boundary` — mutate `gold`: update unchanged; mutate `accept`: update changes.
5. `stage_estimator` — imported group advantages, zero sum, dead-group zero update, detached
   boundary, row 0 before update.
6. `stage_controller` — honest raw/calibrated identity; incomplete-checker sensor separation;
   no future information; log-odds band and empty-band accounting.
7. `stage_selectors` — per-prompt-then-mean pass@k; filter/ranker/vote kept separate; oracle gap
   non-vacuous.
8. `stage_train` — verifier/gold trajectories, KL/entropy, no NaN or warning, deterministic seed.
9. `stage_ablation` — at least five paired seeds, pre-registered predictions, spread reported,
   null accepted.
10. `stage_diagnose` — healthy silence, each surviving rule fires somewhere, reason includes its
    number, pairwise fire sets differ.

Run:

```
./.venv/bin/python -W error::RuntimeWarning week-06-grpo-rlvr-reasoning/code/mini_rlvr.py --fast
./.venv/bin/python -W error::RuntimeWarning week-06-grpo-rlvr-reasoning/code/mini_rlvr.py
```

The private reference implementation is written and run before the TODO file is handed over, then
removed. The public harness must still reject a no-op, a gold-leaking update, a raw-sensor
controller, a mean-then-pass@k eval, and an oracle result copied into the selector column.

### Block 7 · 2:45–3:00 — Close the day and the week

1. Fill `day-05-mini-rlvr-grpo-loop/deliverable.md` with measured results and one honest null.
2. Add the capstone numbers to `shared/numbers-card.md`.
3. Complete the two D3 `YOURS TO WRITE` interview-bank entries and add the capstone question.
4. Write `week-summary.md` as a five-minute interview answer.
5. Write `retrospective.md`, including coding ownership and every cut diagnostic.
6. Update `progress.md` and `HANDOFF.md` to Week 6 complete and Week 7 Day 1 next.

## Checkpoint questions

1. Which exact event probability enters the dead-group law, which enters the curriculum set point,
   and which enters pass@k? State when they are equal.
2. Why does calibrating `p_obs` into `p_gold` not repair an unsound verifier objective?
3. Give the metamorphic test that proves held-out gold cannot affect training, plus its
   non-vacuity control.
4. Your oracle pass@64 rises while best-of-64 accuracy falls. Name two mechanisms from this week
   that can produce the split and the measurement that separates them.
5. Walk one batch through the complete system, naming `π_old`, `π_ref`, the detached boundary,
   the controller's information set, and the time at which each panel row is recorded.

## Frontier-lab interview angle

The artifact is not "I implemented GRPO." That is table stakes. The deep-dive is:

> I built a typed mini RLVR stack and found that four individually correct modules disagreed at
> their interfaces. The verifier reward rate controlled dead groups, the gold rate controlled the
> curriculum and oracle metric, and the deployed selector consumed an answer distribution. I
> enforced those boundaries with metamorphic tests, then used paired-seed ablations to separate
> curriculum miscalibration from reward hacking and selector mismatch.

Lead with one measured interface failure, one negative control, one null result, and the check that
would have prevented shipping it.
