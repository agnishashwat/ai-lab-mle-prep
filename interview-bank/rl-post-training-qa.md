# Interview Bank - RL & Post-Training Q&A

Polished, interview-ready answers. Keep each answer to something you could say out loud
in 60-120 seconds. Add entries as you finish each week.

## Format

### Q: <question>

**Short answer:** <one or two sentences>

**Deeper:** <the fuller explanation>

**Watch out for:** <the follow-up / gotcha they will probe>

---

### Q: Explain the full RLHF pipeline (SFT -> RM -> PPO -> evals).

> **WORKED MODEL** — tutor-written on W6 D3 at his request, built on his own drill answers.
> He had SFT, RM and PPO correct; the gaps were the trick (evals), two of the three
> "cannot fix" clauses, and the number. Those are what is filled in here.

**Cold card (drill this alone):**

```
SFT    optimises  log p(next token) on demonstrations
RM     optimises  log P(human preference) on PAIRED comparisons
PPO    optimises  E[learned reward] - beta * KL(pi || pi_ref)
evals  optimise   NOTHING -- and that is the point of them

the trick: three of the four are optimisations. The fourth is a MEASUREMENT, and
the moment you optimise it, it stops being one.

a preference dataset is a MATCHING (degree <= 1), so the BT gradient on an
uncompared response is EXACTLY 0.0 -- shifting it by 1e6 changes the loss by 0
```

**Short answer:**

Each stage optimises a different objective on a different dataset, and the ordering exists
because each one can only move what the previous one left in place.

**SFT** maximises the log-likelihood of the next token on demonstration data — it is supervised
learning, and it sets the *support*. **RM** maximises the likelihood of observed human
preferences under Bradley-Terry on **paired** comparisons — note it fits a *ranking*, not a
scale, so it is identified only up to an additive function of the prompt. **PPO** maximises the
learned reward minus `beta * KL` against the reference policy. **Evals optimise nothing** —
they are the only stage that is a measurement rather than a fit, and that is exactly why they
are the one you must not tune against. Goodhart is not a failure of evals; it is what happens
when you promote an eval to stage four of the pipeline.

What each stage cannot fix downstream:

- **SFT cannot fix** the reward — it never sees one. It can only fix what the data demonstrates,
  so any behaviour absent from the demonstrations is absent from the initialisation.
- **RM cannot fix** its own coverage. This is the number: a preference dataset is a **matching**,
  degree ≤ 1, so the Bradley-Terry gradient on a response that was never compared is **exactly
  0.0** — I shifted such a response by 1, by −37.5 and by **1e6** and the loss changed by
  **exactly zero**. Same data, two initialisations, and the argmax lands on **seq 0 (gold 1.0)**
  or **seq 14 (gold 0.0)**. An RM's accept set is fixed by its *initialisation* off the compared
  set; a verifier's is the same object under both, because it is data rather than a fit.
- **PPO cannot fix support.** It reweights what the policy already samples; it cannot elicit
  behaviour that is not already there. On W6 D3 I measured the sharp version of this: pass@1
  rose **0.4447 → 0.9138** while pass@1024 moved **+0.0000**. All the gain was re-ranking
  inside problems that were already solved.

The KL sits in the **PPO loss**, against **`pi_ref`**, which is the SFT checkpoint — not the
RM's data distribution and not the current policy. It can also be folded into the reward to
densify the signal, but placing it there is strictly worse: a reward-side KL is an **open-loop**
controller, and the group mean deletes most of it before it reaches the gradient.

**Deeper:**

The reason the pipeline has this shape is that each stage converts one kind of supervision into
another. Demonstrations are expensive and dense; comparisons are cheap and sparse; reward is
cheap and dense but *learned*, which is where it goes wrong. RM exists only because you cannot
collect enough demonstrations, and PPO exists only because the RM is differentiable-through-
sampling in a way the humans are not.

The identification issue in the RM is worth stating precisely: Bradley-Terry only ever sees
*differences* of rewards on the same prompt, so `r(x,y) + c(x)` fits the data identically for
any `c`. That is harmless for PPO (the KL-regularised optimum is invariant to it) and it is
exactly what cancels when you derive DPO.

**Watch out for:**

- Saying "evals optimise held-out performance." They do not optimise anything. Naming that is
  the whole point of the question, and it is the setup for a Goodhart follow-up.
- Saying the KL is against the *current* policy or against the RM's data. It is against the
  frozen SFT reference.
- Claiming PPO adds capability. It moves mass. If you are asked whether RL makes models smarter,
  the honest answer is that on every system I have measured it moved mass and left the support
  where it was — with the scope condition attached.

### Q: Derive the PPO clipped objective and explain why old logprobs are necessary.

> **WORKED MODEL** — written by the tutor on W6 D3 as a demonstration of "lead with the
> payload". The drill answer scored 4/5; the missing point was the number. Note the shape:
> **thesis sentence, then a number, then the mechanism.** Every other entry in this file that
> he writes should have that shape.

**Cold card (drill this alone):**

```
rho = pi_theta(a|s) / pi_old(a|s), per TOKEN
clip ZEROES the gradient in exactly two quadrants:  A > 0 and rho > 1+eps
                                                    A < 0 and rho < 1-eps
inert at mu = 1:  bound-token fraction 0.000e+00  vs  0.2222 at mu = 8
                  (same lr, same seed, same task)
```

**Short answer:**

The clip is not a penalty and it does not shrink the update — it **switches the gradient off**.
PPO's objective is `min(rho*A, clip(rho, 1-eps, 1+eps)*A)`, and in the two quadrants where the
update would push a token further in the direction it has already moved too far, the gradient is
**exactly zero**, not merely small. That is the entire mechanism.

The old logprobs exist because rollouts are expensive: you generate once under `pi_old` and take
several gradient steps on the same batch. The moment you take step two, the data are off-policy,
so you need the importance ratio `rho = pi_theta/pi_old` to keep the estimator honest — and once
you have a ratio you need something to stop it running away, which is the clip.

The number that shows what the clip actually is: at `mu = 1` (one inner epoch) `rho == 1` for
every token by construction, so the clip **cannot bind**. Measured bound-token fraction
**0.000e+00 at mu = 1 versus 0.2222 at mu = 8**, same lr, seed and task. So the clip is not a
property of PPO; it is a property of **reusing a batch**.

**Deeper:**

Start from the surrogate. The policy gradient wants `E_{pi_theta}[grad log pi_theta * A]`, but
the samples came from `pi_old`, so importance-sample:

```
J(theta) = E_{a ~ pi_old} [ rho * A ],      rho = pi_theta(a|s) / pi_old(a|s)
```

This is unbiased but its variance explodes when `pi_theta` drifts from `pi_old`, because `rho` is
a ratio of two small numbers. TRPO handles that with a hard KL trust region and a second-order
solve. PPO replaces the constraint with a first-order surrogate that is **cheaper and blunter**:

```
L = E[ min( rho*A,  clip(rho, 1-eps, 1+eps) * A ) ]
```

Read the `min` quadrant by quadrant, because that is where the answer lives:

```
A > 0, rho < 1+eps : min picks rho*A            gradient flows normally
A > 0, rho > 1+eps : min picks (1+eps)*A        constant in theta  =>  GRADIENT = 0
A < 0, rho > 1-eps : min picks rho*A            gradient flows normally
A < 0, rho < 1-eps : min picks (1-eps)*A        constant in theta  =>  GRADIENT = 0
```

Two observations that mark a real answer:

1. **The `min` is pessimistic, and the asymmetry is deliberate.** When `A < 0` and `rho` is
   *large* the `min` selects the **unclipped** branch, so the gradient still flows — PPO will
   happily keep pushing a bad action down no matter how far the ratio has drifted. The clip is
   one-sided *in effect*: it stops you over-committing to what looks good, not to what looks bad.
2. **It is a heuristic trust region with no guarantee.** Nothing bounds the total KL; the clip
   bounds a per-token ratio, one step at a time. That is why real RLHF adds an explicit KL term
   on top, and why KL still runs away when the reward is exploitable.

Where the KL sits, and against which policy: against the **SFT checkpoint**, and preferentially in
the **loss** rather than in the reward. Under a group baseline this stops being a preference and
becomes a correctness issue: a KL folded into the reward survives the group mean only as its
within-group deviation, measured at **0.035726 vs 0.6842 = 19.15x deleted**, and a uniform drift
is annihilated to **2.50e-16**. A KL in the reward is an **open-loop controller** — it cannot see
the drift it is supposed to correct.

**Watch out for:**

- "The clip limits how big the update can be." It does not. The update is `lr * grad`, and in the
  clipped quadrants the gradient is **zero**, not capped. A capped update and an absent update
  behave differently the moment you add momentum.
- Being asked why `min` and not `clip` alone. `clip(rho,...)*A` on its own would let a large `A>0`
  ratio keep contributing at the boundary value; the `min` makes the objective a **lower bound**
  on the unclipped surrogate, which is what makes it conservative.
- Being asked whether the clip does anything in GRPO. At `mu = 1`, **no** — 0.000e+00 versus
  0.2222 at `mu = 8`. Say the number and the scope condition together.
- The token-vs-sequence level. `rho` is per token in the standard implementation; a sequence-level
  ratio is the product over tokens and has catastrophically higher variance. If someone says
  "sequence-level importance ratio", ask what they do about the variance.
- Padding. Padded positions have `rho = 1`, not 0 (`logp` is zeroed at pad, so `exp(0-0) = 1`), so
  forgetting the mask adds a full copy of `A` per pad slot rather than nothing.

### Q: Derive DPO from KL-regularized RLHF. When is DPO the wrong tool?

> **WORKED MODEL** — tutor-written on W6 D3 at his request, built on his own drill answers.
> He had the closed-form optimum, the inversion for `r`, and the cancellation correct. The
> gaps were the one-sentence "what DPO IS" and the data-format failure mode.

**Cold card (drill this alone):**

```
optimum   pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)
invert    r(x,y)   = beta * log[ pi*(y|x) / pi_ref(y|x) ] + beta * log Z(x)
BT diff   the beta*log Z(x) CANCELS -- it depends on x only, and BT sees only
          differences of rewards on the SAME x

DPO IS:   Bradley-Terry fitted DIRECTLY on the policy, using the policy's own
          log-ratio against pi_ref as the reward. No RM, no sampling, no rollouts.

the data-format failure: a preference set is a MATCHING (degree <= 1). BT gradient
on an uncompared response is EXACTLY 0.0; shift it by 1e6, the loss moves by 0.
```

**Short answer:**

Start from the KL-regularised objective, `max_pi E[r] − beta*KL(pi || pi_ref)`. This has a
closed-form optimum — it is a Gibbs/exponential-tilt of the reference:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp( r(x,y) / beta )
```

Now read it **backwards**. Solve for the reward instead of the policy:

```
r(x,y) = beta * log[ pi*(y|x) / pi_ref(y|x) ] + beta * log Z(x)
```

Every KL-regularised optimal policy *is* a reward function, expressed as a log-ratio. The
partition function `Z(x)` is the obstruction — it is intractable. But substitute this into
Bradley-Terry, which models `P(y_w > y_l | x) = sigmoid(r(x,y_w) − r(x,y_l))`, and `beta*log
Z(x)` appears in both terms with the same sign and **cancels**. It has to: it depends on `x`
only, and BT is a model of *differences* on a fixed prompt, so nothing that is constant across
responses to the same prompt can survive. That is the same identifiability slack the RM has, and
DPO is the observation that it never mattered.

So: **DPO is Bradley-Terry fitted directly on the policy, using the policy's own log-ratio
against `pi_ref` as the implicit reward.** The RM, the rollouts and the KL penalty all disappear
— the KL is not dropped, it is *absorbed into the parameterisation*, which is why `beta` is
still there.

**When it is the wrong tool.** The failure mode that is a property of the **data format**, not
of the loss:

DPO requires **paired preferences on the same prompt**. That is a matching, degree ≤ 1 — and the
gradient on any response that was never compared is **exactly 0.0**. I measured this: shifting an
uncompared response's reward by 1, by −37.5, and by **1e6** changed the loss by **exactly zero**,
and the resulting argmax depends entirely on the initialisation — **seq 0 (gold 1.0)** under one
init, **seq 14 (gold 0.0)** under another, on identical data.

Consequences: you cannot use pointwise or unpaired data, you cannot express "both of these are
bad," you cannot use a scalar score, and you cannot use an *online* signal — the pairs are fixed
before training starts, so DPO is off-policy by construction and stays that way. If your
supervision is a **verifier** rather than a comparison, DPO is the wrong shape entirely: a
verifier gives you a per-sample binary label on *any* sample you can generate, which is the one
thing the preference format cannot represent. That is what RLVR is for.

**Deeper:**

The relative-only nature of the loss has a second consequence people notice empirically: DPO
optimises the **margin**, so it can and does drive the chosen response's absolute log-probability
*down* as long as the rejected one falls faster. Nothing in the objective pins the level. On W4
I measured the implicit reward at `theta = theta_ref` being **exactly 0.0** — which is correct
and is also why any diagnostic that normalises by `margin[0]` divides by exactly zero.

**Watch out for:**

- Dropping the `beta * log Z(x)` term from the inversion instead of *carrying it and showing it
  cancels*. Carrying it is the derivation; dropping it is a memorised formula.
- Saying "DPO removes the KL." It does not — it reparameterises so the KL is implicit. `beta`
  surviving in the loss is the tell.
- Calling DPO "on-policy." The pairs are collected in advance and never resampled.
- Saying DPO is "simpler PPO." It is a different *estimand*: PPO optimises a learned reward under
  sampling; DPO fits a preference likelihood in closed form. They coincide only under the BT
  assumption and an exactly-optimal RM.

### Q: GRPO vs PPO - what changes and why (group baseline, no critic)?

**Cold card:** `(G-1)/G` · `252 -> 140 GB (-44.4%)` · `-> 126 (-50.0%)` · `-> 112 (-55.6%)` · `G=1 => Â ≡ 0` · my null: `5.1786 ± 0.036 vs 5.1501 ± 0.109`, 8 seeds

**Short answer:** Both estimate the same policy gradient. They differ in exactly one
place — the baseline. PPO learns a value function V(s); GRPO uses the mean reward of
G samples drawn from the same prompt. Everything else is downstream of that one
substitution.

**Deeper:** Sample G responses per prompt, score them, subtract the group mean.
Centering a group of size G shrinks each advantage by `(G-1)/G`, and at G=1 the
advantage is identically zero — the baseline is the sample itself, so there is no
signal at all. That degenerate case is the cleanest way to see that the group IS the
critic.

What you buy is memory. Four rows, per 70B-scale actor:

```
PPO                    112 + 112 + 14 + 14  = 252 GB
GRPO (no critic)             112 + 14 + 14  = 140 GB   (-44.4%)
+ cached ref logprobs             112 + 14  = 126 GB   (-50.0%)
RLVR (no RM either)                    112  = 112 GB   (-55.6%)
```

The critic is the whole 112 GB — it is a second model with its own gradients and
optimizer state. The reference and the RM are inference-only at 14 GB each.

What you pay is **per-token credit assignment**. The critic produces a value at every
token, so PPO can say *which* token was good. The group mean produces one scalar per
sequence, broadcast unchanged to every token in it.

**Watch out for:**
- "GRPO reduces variance." It does not — it *changes the baseline*. Whether variance
  goes up or down depends on whether the group mean is a better predictor of return
  than the critic was, which is an empirical question, not a theorem.
- My own evidence is a **null**: removing the critic moved the loss `5.1786 ± 0.036`
  vs `5.1501 ± 0.109` over 8 seeds — inside the noise. **Scope:** this was a
  short-horizon toy task, and per-token credit assignment is exactly the thing that
  only pays off when the horizon is long enough for temporal misassignment to bite.
  So the null is what the theory predicts, not evidence that critics are useless.

---

### Q: Why KL to a reference policy? What breaks without it?

**Cold card:** loss, not reward · `19×` deleted by centering (`0.6842 -> 0.035726`) · `k3 = exp(u) - u - 1`, `u = log π_ref - log π` · `r_gold = d(α - β log d)`, `d = √KL`

**Short answer:** The KL is the only thing holding the policy near a distribution
where the reward was ever measured. Without it the policy walks off the support the
reward model was fit on, and the proxy climbs while the thing you actually want
falls.

**Deeper:** In GRPO it goes in the **loss**, not in the reward. This is not a
stylistic choice — it is forced by the group baseline. A KL term added to the reward
is a per-sequence scalar, and the group mean is subtracted from every reward in the
group, so any component of the KL that is *common to the group* is annihilated
before the gradient sees it. Measured on my own runs: of the KL signal placed in the
reward, `0.035726` survives centering against `0.6842` placed in the loss — **19×
deleted**. Uniform drift is exactly the part that is common to the group, and
uniform drift is exactly what you were trying to penalize.

Use the **k3** estimator, not k1:

```
u  = log π_ref - log π
k1 = -u                      unbiased, but SIGNED and high variance
k3 = exp(u) - u - 1          unbiased, ALWAYS ≥ 0, lower variance
```

All three properties matter, but non-negativity is the distinctive one: a signed
penalty estimate can, on some batches, *pay the policy to diverge*, and an adaptive-β
controller keyed on a signed estimate chases noise instead of tracking drift.

Without any KL, you get the Gao–Schulman–Hilton over-optimization law:

```
r_gold(d) = d · (α - β log d)      where d = √KL
```

Concave in d: gold reward rises, peaks at a finite KL, and then **falls** — while the
proxy reward keeps climbing monotonically the whole time. The KL coefficient is
therefore not a regularizer you tune for stability, it is the knob that selects where
on that curve you stop.

**Watch out for:**
- The variable is `d = √KL`, not KL. Getting this wrong changes the shape.
- "Just use k1, it's unbiased." So is k3. Unbiasedness is not the axis of comparison.
- The 19× is *my* number on *my* toy task. **Scope:** it is a statement about
  group-mean centering, so it applies to any group-relative method (GRPO, RLOO,
  GRPO-variants) and says nothing about PPO, where there is no group to center over.

---

### Q: KL spikes, reward improves, human evals degrade - diagnose it.

**Cold card:** the separator is whether gold **rose first** · peak ⇒ over-optimization, monotone fall ⇒ broken reward · leaky vs saturated: `dead 0.769/0.836`, `v_pass 0.968/0.980`, `gold 0.025/0.980`

**Short answer:** Proxy up and gold down is the definition of over-optimization, so
the first question is not *what* is wrong but *when* it started — because
over-optimization and a broken reward look identical at the end of a run and
completely different at the beginning.

**Deeper:** Ordered checklist, most diagnostic first:

```
1. shape of the spike -- one batch, or a new regime?  transient vs. real divergence
2. did the proxy reward jump at the SAME step?        reward hacking vs. optimizer blowup
3. entropy -- UP or DOWN?                             two opposite illnesses:
                                                      down = mode collapse onto a hack
                                                      up   = the policy is falling apart
4. mean completion length                             the classic length-hack signature
5. grad norm / LR schedule / a bad batch /
   sampler-vs-trainer precision mismatch              often not an RL problem at all
```

Item 5 is the one people forget: a KL spike is frequently a numerics bug, and if your
sampler and trainer disagree on precision you are off-policy even at μ = 1.

**The single measurement that separates the two diagnoses** is gold reward *as a
function of KL*, and the discriminator is the presence of a peak:

```
gold rises, peaks, then falls   -> OVER-OPTIMIZATION.  The reward was fine; you went
                                   too far. Fix: raise β / early-stop at the peak.
gold falls from KL = 0          -> BROKEN REWARD.      No KL budget saves you.
                                   Fix: the reward.
```

**Watch out for:**
- Do not put "check the KL" on the checklist. The KL spike is the *premise*. Listing
  it reads as reciting a dashboard rather than reasoning.
- "Gold reward is down" does not separate the two cases — gold is down in both. The
  discriminator is the *shape*, not the level.
- You cannot run this diagnosis on the reward panel alone. Measured on my own runs, a
  **hacked** run and a **solved** run are the same point on every column the reward
  can produce, and opposite on the held-out one:

```
                    dead_frac    v_pass    gold_pass
leaky (hacked)        0.769       0.968      0.025
saturated (solved)    0.836       0.980      0.980
```

---

### Q: What makes a reward *verifiable*, and when does RLVR fail?

**Cold card:** `V ⊆ C` sound · `C ⊆ V` complete · incompleteness bounded by COMPUTE (`22 -> 32` groups, `+45%`) · unsoundness unbounded (`sup gap = 1.000` for 1 FP *and* for 7) · dead-group law `P(dead) = Σ q^G`

**Short answer:** A reward is verifiable when its accept set is *decided* rather than
*fitted* — membership is computed by a program, and the set is fixed before training
starts. That single property is what makes its worst case knowable in advance.

**Deeper:** Write C for the set of genuinely correct responses and V for the set your
checker accepts. Two containments, and they have completely different characters:

```
V ⊆ C    SOUND       V \ C = ∅    no false positives
C ⊆ V    COMPLETE    C \ V = ∅    no false negatives
```

**Incompleteness is a compute bug.** A false negative deletes signal; it never
corrupts the objective. It is priced exactly, in rollouts. A 30% answer-extraction
failure moves the observed pass rate `0.10 -> 0.07`, and the group size needed to keep
90% of groups alive goes `22 -> 32` — **+45% rollout compute, bought by a regex and
paid in GPU-hours.**

**Unsoundness is a correctness bug, and it is unbounded.** If `V \ C ≠ ∅` then
`sup(v_pass - gold_pass) = 1`, attained by the point mass on a single false
positive — and that supremum is **1.000 for one false positive and 1.000 for seven**.
Reward hacking is a property of the *support* of V \ C, not of its measure. Their eval
accuracies under a uniform policy are 93.3% and 53.3%, so the statistic that ranks
them as evals is silent on the thing that matters under optimization.

Conversely, if the checker is sound then `v_pass ≤ gold_pass` for **every** policy at
**every** KL, with no scope condition. That is the strongest guarantee anything in
post-training gives you, and it is why the asymmetry is worth paying for.

**When RLVR fails:** when the reward is *dense in the wrong sense* — every group
all-pass or all-fail, so no gradient. `P(dead) = Σ_j q_j^G = exp(-(G-1)·H_G(q))`, a
Rényi entropy of the reward's **output distribution**. Binary reward at p = 0.05 with
G = 8 is `66.34%` dead. Splitting the failure mass into two levels (0.05, 0.40, 0.55)
takes it to `0.90%` — a **73.5×** reduction from one extra reward level. Buying the
same thing with group size alone needs `G = 92`, an **11.5× rollout bill**. Reward
design and group size are the same knob, priced differently.

**Watch out for:**
- The rule is **split the mode**. At p = 0.05, grading among failures is the 73×
  win and grading among successes does essentially nothing; at p = 0.95 it reverses
  exactly.
- The one-sentence close: *"Incompleteness is a compute bug, unsoundness is a
  correctness bug, and I will pay 45% more GPU to never ship the second one."*

---

### Q: Your math verifier is 97% accurate. Is that good enough to train against?

**Cold card:** same checker, precision `0.672 -> 0.093` under a tilted policy · at 1% FP: `P(hack|rewarded) = 1.10 / 9.09 / 35.25 / 68.86%` as `p = 0.5 / 0.1 / 0.02 / 0.005` · ask a SET question about FPs, a RATE question about FNs

**Short answer:** 97% of what, measured under which policy? Accuracy is a property of
the verifier *and the distribution it was measured on* — and RL exists precisely to
change that distribution. The number is not wrong, it is not well-formed.

**Deeper:** Two independent reasons the scalar is the wrong object.

**1. It is policy-dependent, and training moves the policy toward the errors.** Same
checker, same `gold` and `accept` arrays, two policies:

```
policy at initialisation                  precision 0.672
policy tilted toward its own false positives   precision 0.093
```

Nothing about the verifier changed. Optimization *seeks out* V \ C, because that is
where reward is cheapest — so the measured accuracy degrades as a direct consequence
of training succeeding.

**2. It is difficulty-dependent, at a fixed false-positive rate.** With false-negative
rate α and false-positive rate γ:

```
p_obs = p(1-α) + (1-p)γ
P(hack | rewarded) = γ(1-p) / [p(1-α) + γ(1-p)]
```

At a **fixed 1% false-positive rate** and α = 0.10:

```
p = 0.500  ->   1.10%
p = 0.100  ->   9.09%
p = 0.020  ->  35.25%
p = 0.005  ->  68.86%     -> 1 as p -> 0
```

A 1%-FP verifier is fine on problems the model already solves and worthless on the
frontier. And it concentrates: `P(the whole alive group's positive signal is a hack)`
is `0.33020` at p = 0.02 against `0.00060` at p = 0.5 — **554×** — with the same 1%
verifier in both runs. The pathology is difficulty-induced, not verifier-induced.

**What I ask for instead — note the asymmetry, that is the whole answer:**

```
about false positives:  is the SET empty?    a yes/no question, NOT a rate
about false negatives:  what is the RATE?    it prices my rollout budget
```

You ask a set question about FPs because their rate does not bound the damage — the
support does. You ask a rate question about FNs because their rate is *exactly* the
price, in groups.

**The deployment consequence:** *"I train against a sound-but-incomplete verifier and
pay for it in rollouts, and I hold out a gold set the reward never sees — because a
hacked run and a solved run are identical on every column the reward can produce."*

**Watch out for:**
- Do not accept the reframe "so use a 99.9% verifier." Any non-empty V \ C has
  supremum gap 1.0. The axis is empty/non-empty, not big/small.
- A false positive and a genuine success produce **bit-identical advantage vectors**
  in a group of size G with one winner — both give `(G-1)/G` to the winner. No rule
  reading the reward panel can separate them. Only a held-out gold column can.


---

### Q: Your model gets 85% pass@64 and 30% pass@1. What do you actually know?

> **LEARNER-WRITTEN on W6 D5.** The first response caught that `p=0.30` and
> `pass@64=0.85` are inconsistent for one homogeneous prompt. The repaired answer below keeps
> that catch and makes the benchmark aggregation level explicit.

**Cold card (drill this alone):**

```
you know the answer is IN THE SUPPORT for 85%. You know nothing shippable, because
pass@k = capability COMPOSED WITH AN ORACLE SELECTOR you do not have.

three selectors, three limits:
  oracle  -> 1                      (this is pass@k)
  BoN     -> the verifier PRECISION  measured 0.3216 while pass@k -> 1.0000
  vote    -> 1 if modal answer right, else 0
            measured maj@k FALLING 0.3563 (k=9) -> 0.2764 (k=101), pass@101 = 1.000000

and the estimator: plug-in 1-(1-c/n)^k is f(p_hat), f CONCAVE, so Jensen makes it
UNDER-report. n=10 p=0.1 k=10: 0.493747 vs truth 0.651322 = 24.19% thrown away
```

**Short answer:**

At benchmark level, `pass@64 = 0.85` means there is an average 85% chance that at least one of
64 rollouts is correct, **using an oracle to identify it**. `pass@1 = 0.30` is the one-draw
average; both numbers can coexist because we compute pass@k per prompt and then average, so
`mean_x f(p_x) ≠ f(mean_x p_x)`. We cannot use the oracle in the actual system, so this is not
shippable until we check what the filter, ranker, or vote actually emits against held-out gold.

**Deeper:**

The two numbers establish two different things. Pass@1 measures average one-sample correctness.
Pass@64 measures whether correct output mass is reachable within a 64-sample budget, composed
with a selector that always recognises correctness. Heterogeneous prompt difficulties are
load-bearing: applying `1−(1−p)^64` to the mean pass@1 would incorrectly predict nearly 1.

The next production check is direct: run the exact deployed selector and compare its emitted
answer with held-out gold. Report selector accuracy and verifier precision alongside oracle
pass@64. In my own fixture, pass@k approached 1 while a verifier-driven selector saturated at
precision and majority voting could get worse as k grew.

**Watch out for:**

- Do not call pass@k a capability without naming its **oracle selector**.
- Do not compute pass@k from the benchmark's mean pass@1; compute per prompt, then average.
- `dead_frac` diagnoses training signal. It does not measure whether deployment can select the
  right answer.

---

### Q: Best-of-64 against your reward model beat greedy decoding. Ship it?

> **LEARNER-WRITTEN on W6 D5.** The initial answer correctly demanded effect size and matched
> inference cost. The repair moved the circular proxy judge to the first sentence and kept those
> operational checks.

**Cold card (drill this alone):**

```
best-of-n IS a KL move you did not budget for:
  KL(BoN_n || pi) = log n - (n-1)/n
  = 0.1931 / 1.2044 / 3.1745 / 4.5491 nats at n = 2 / 8 / 64 / 256
  every DOUBLING costs ~log 2; measured +0.6853 nats from n=64 to n=128
  so BoN-64 is a 3.17-nat move against a policy you KL-constrained during training

and it is a RANKER, which is the bad half of the split. SAME verifier, SAME budget:
  FILTER  P(hack|accepted emission)
          = gamma(1-p)/[gamma(1-p)+(1-alpha)p] = 9.09%, n-INVARIANT
  FILTER  P(hack output) = [1-(1-p_obs)^n] * P(hack|accepted emission)
          0.9000% / 5.1426% / 9.0794% / 9.0909%  at n = 1/8/64/256
  RANKER  P(any top-scored hack) = 1 - [1-(1-p)gamma]^n
          = pass@n for policy hack mass (1-p)gamma
          0.9000% / 6.9772% / 43.9322% / 90.1178%  at n = 1/8/64/256
  at n=256: 9.09% vs 90.12% -- 9.91x, from the SELECTION RULE alone

"beat greedy decoding" was measured with the RM as judge. Ask for the held-out column.
```

**Short answer:**

No, because the same reward model selected the output and judged it. Evaluate greedy and the
**BoN-64 emitted output** on held-out gold labels or human annotations at matched inference cost.
BoN-64 also moves the emitted distribution by
`log 64 − 63/64 = 3.1745` nats of KL in the continuous no-tie case, so it crosses a distribution
shift that the policy's training-time KL constraint did not budget for.

**Deeper:**

If `U = F_π(score)`, then under the base policy `U ~ Uniform(0,1)`. The maximum of `n` draws has
density `n u^(n−1)`, so `q(y)/π(y) = n u^(n−1)` and
`KL(q‖π) = log n − (n−1)/n`. Score ties reduce the shift; fully tied scores with uniform
tie-breaking give `q=π` and KL exactly zero.

BoN is a **ranker**, not a filter. A filter uniformly emitting among accepted samples preserves
the verifier's posterior error rate; a max-score ranker compounds any exploitable false-positive
tail across all 64 attempts. The independent held-out column therefore decides whether the gain
is real, while matched inference cost decides whether it is worth buying.

**Watch out for:**

- Compare greedy output with the **BoN-emitted output**, not greedy with `pass@64`; pass@64 still
  uses the unavailable oracle.
- "The reward model score improved" is circular when that reward model also chose the winner.
- Include the 64× rollout bill and latency in the comparison, not only accuracy.

---

### Q: How would you design the data curriculum for an RLVR run?

**Cold card:** `p* = a/(a+b+k−1) → 1/(k+1)`; 0.5 only at k=1 · exact 90%-alive `G` → 5/6/11/20/38/149 at k=1/2/4/8/16/64 (`G ≈ 2.3026k` is one-sided) · **88.34% dead** at (report pass@64, train at p*, G=8) · `p_floor = γ(1−τ)/[τ(1−α)]/(1+·) = 1/11`, crosses `1/(k+1)` at **k=10 exactly** · `/std` tilts by `G/4` = **16× at G=64** · training at 0.5 while reporting pass@8 costs **22.17×** · band narrows **19.07× in p, 1.43× in logit-p**

**Short answer:** A curriculum is a **controller**, so I need three things before I write any
code: a **measured** quantity to control, a **set point**, and the **constraints that make the
set point infeasible**. The measured quantity is the current per-prompt success rate `p`,
estimated from recent rollouts — not a static difficulty label assigned once, because `p` is a
property of the *policy*, and the policy is the thing I am changing. The set point is **not 0.5**.
It is `1/(k+1)`, where `k` is the pass@k budget I report. And the constraints are two walls with
**opposite fixes**, which is why this is a control problem and not an optimisation.

**Deeper:**

**1. The set point falls out of a product, not a preference.** The value of a rollout is
(how much the policy moves) × (what a unit of movement is worth to the metric):

```
progress:  Δp ∝ p^a (1−p)^b            (a = b = 1 for the plain group-advantage mass, 2(G−1)p(1−p))
weight:    ∂pass@k/∂p = k(1−p)^(k−1)   ← the METRIC's derivative, in this direction
value(p) ∝ p^a (1−p)^(b+k−1)   ⇒   p* = a/(a + b + k − 1)   →   1/(k+1) at a=b=1
```

`p* = 1/2` **only at k = 1**. The folk rule "train at 50%" is the `k=1` special case of a
`k`-dependent law, repeated after everyone stopped reporting `k=1`. And I would **measure**
`(a, b)` for my stack rather than assume `(1,1)` — the exponents are the empirical part; the
`+k−1` is not, it is arithmetic.

The size of the mistake, at pass@8, training at 0.5 instead of 1/9:

```
progress  2.53× BETTER at p=0.5      weight  56.12× WORSE      net  22.17× of the available signal, gone
```

**2. Wall one — dead groups. The metric you report sets your group size.** A group is dead
(all-0 or all-1) with probability `p^G + (1−p)^G`, and every advantage in it is exactly zero:

```
G to keep 90% alive at p*:  min G with p*^G + (1-p*)^G <= 0.1
                            5 / 6 / 11 / 20 / 38 / 149  at k = 1/2/4/8/16/64
one-sided approximation:    ln(0.1)/ln(1-p*) -> 2.3026*k asymptotically
P(dead | G=8, p=p*):        0.78 / 3.92 / 16.78 / 38.97 / 61.57 / 88.34 %
```

**Report pass@64, train at the difficulty that maximises it, use the G=8 everyone uses, and 88%
of your groups return zero gradient.** This is the number I would open with, because it is the
one that reframes group size from a compute knob into a *consequence of your eval choice*.

**3. Wall two — the precision floor. Below some `p`, your signal is mostly false positives.**
With false-negative rate α and false-positive rate γ, demanding `P(hack | rewarded) ≤ τ`:

```
p_floor = R/(1+R),  R = γ(1−τ)/[τ(1−α)]
γ=1%, α=10%, τ=10%  ⇒  p_floor = 1/11 = 0.0909, and it crosses 1/(k+1) at k = 10 EXACTLY
```

So for `k ≤ 10` the metric's optimum is legal and the verifier is not binding; for `k > 10` the
verifier is binding and I train *above* the metric's optimum and eat the loss:

```
value retained at the floor:  0.9793 (k=8) / 1.0000 (k=10) / 0.8872 (16) / 0.3804 (32) / 0.0358 (64)
```

**At pass@64, a 1% false-positive rate costs 96.4% of the signal the metric was asking for.** And
raising the learning rate does not help: the noise floor is `γ(1−γ) = 0.0099` and does not shrink
with `p`, so at `p = 0.001` roughly 92% of gradient magnitude is false positives. **You cannot
raise the LR out of a bad SNR — it scales signal and noise identically.**

**4. Which wall binds is a measurement, and only one of them is purchasable.**

```
γ = 0.01,  G = 8  ->  floor 0.0909  vs  dead-50% wall 0.0830  ->  PRECISION binds
γ = 0.001, G = 8  ->  floor 0.0099  vs  dead-50% wall 0.0830  ->  DEAD GROUPS bind
```

The dead-group wall is **bought with compute** (raise `G`). The precision floor is not — no
amount of sampling makes a false positive into a true one. So I check which one binds *before*
asking for GPUs.

**5. What I would actually build.** Estimate `p̂` per prompt from an EMA of recent rollouts —
the sensor is `ĉ/G` and **I have already paid for it**, it is the group I just rolled out. Sample
prompts whose `p̂` sits in a band around `clip(p*, p_floor, ·)`, control in **logit-`p` space**
(the value curve is roughly symmetric there and the band is multiplicative, not additive), and
re-estimate continuously. **Logit space is not cosmetic** — the 50%-of-peak band narrows
**19.07×** in `p` from k=1 to k=64 (0.7071 → 0.0371) but only **1.43×** in logit-`p`
(3.526 → 2.465 nats). Controlling the wrong coordinate makes the problem **13.3× harder than it
is**, and a fixed additive tolerance like `p* ± 0.05` is meaningless at k=64, where the entire
legal band is 0.037 wide. Second operational note: the two walls need
**opposite data moves** — scaffolding/hints at the bottom, harder items at the top. A hint that
lifts a prompt from `p = 0.01` to `p = 0.15` buys **12.88× the signal** and drops dead groups
92.27% → 27.25%, which is what you would otherwise get by raising `G` from **8 to 130**. The hint
is free. The 16× rollout budget is not.

**Watch out for:**
- **You already have a curriculum, whether you designed one or not.** GRPO's `/std` reweights
  groups by `1/σ`, which favours near-dead groups — it moves the effective set point by up to
  **4.2×** (at G=64, k=4: 0.2000 → 0.0474). *Any per-group rescaling is a reweighting of the
  problem distribution, i.e. a curriculum.* Say this before proposing a new one.
- Do not defend `p* = 1/2` by symmetry. `p(1−p)` is symmetric; `pass@k` is **not**, and the
  product is what you optimise. The asymmetry enters entirely through the metric.
- `∂pass@k/∂p`, not its reciprocal, and the exponent is `k−1`, not `k`. Getting the exponent
  wrong turns `1/(k+1)` into `1/(k+2)` — small at k=64, but it is the kind of error that reads
  as "recited" rather than "derived."
- Static difficulty labels are **stale by construction**: an item at `p = 0.1` when you labelled
  it is at `p = 0.4` two thousand steps later, and it is now the wrong item.

---

### Q: Your reward curve is flat for 2000 steps. Diagnose it — cheapest measurement first.

**Cold card:** dead groups give **exactly 0**, not small · `dead_frac` is invariant under `p ↔ 1−p`; `p_obs` is not — that is the discriminant · `U = 2(G−1)p(1−p)·J`, `J = 1−α−γ` · `U ≡ 0` in p on the `α+γ=1` line · FREE: `dead_frac`, `p_obs`, `‖∇θ‖`. EXPENSIVE: `Cov(r,s)`, needs gold labels

**Short answer:** Flat is not one failure, it is four, and they have **opposite fixes** — so the
answer is an *ordering* of measurements, not a list of hypotheses. I order by price, and the good
news is that the first three splits are free: they read off rollouts I have already paid for.
The expensive one — the verifier check — needs held-out **gold labels**, so it goes last.

**Deeper:**

```
FREE (already in the rollouts I just did):
  1. dead_frac = fraction of groups all-0 or all-1     ─┐ these two together split
     p_obs     = mean observed success rate            ─┘ the whole space
  2. mean |A| over ALIVE groups, and ‖∇θ‖ per step

EXPENSIVE (needs a labelled slice the reward never sees):
  3. Cov(r, s) between verifier reward and true success
```

```
dead_frac HIGH, p_obs ≈ 0  ->  ZERO-REWARD CLIFF     nothing is ever solved
dead_frac HIGH, p_obs ≈ 1  ->  SATURATED SET         everything is always solved
dead_frac LOW,  |A| healthy, Cov(r,s) ≈ 0  ->  BROKEN VERIFIER   signal exists, points nowhere
dead_frac LOW,  |A| healthy, Cov ok, ‖∇θ‖ ≈ 0  ->  BROKEN LOSS/OPTIMIZER
```

**Why `p_obs` is the right second number.** `dead_frac = p^G + (1−p)^G` is **invariant under
`p ↔ 1−p`** — it cannot tell "too hard" from "too easy," and those are the two branches that
need opposite data. `p_obs` breaks exactly that symmetry, and it costs nothing.

**Why `Cov(r, s)` goes last even though it is the most diagnostic.** The useful signal is
`U = (G−1)Cov(r,s) = 2(G−1)p(1−p)(1−α−γ)`: the group mechanics give you `2(G−1)p(1−p)` and the
verifier multiplies it by Youden's `J = 1−α−γ`. On the line `α + γ = 1` this is **identically
zero in `p`** — a verifier that is exactly a coin flip produces a perfectly healthy-looking
`mean |A|`, a perfectly healthy `dead_frac`, and zero learning, forever. But measuring `Cov(r,s)`
requires the gold column the reward is not allowed to see, so I pay for it only after the free
splits fail to explain the flatness.

**The fixes, which is the half that makes the diagnosis worth anything:**

| diagnosis | fix | do NOT |
|---|---|---|
| zero-reward cliff | scaffolding / hints / easier data / raise `G` — **move `p`** | touch the loss |
| saturated set | harder data; **retire the solved prompts** | add more of the same |
| broken verifier | fix α/γ. **No amount of compute helps** | raise the LR |
| broken loss | plumbing: advantage detached, `no_grad` on the wrong branch, LR = 0 | change the data |

**Watch out for:**
- **Cliff and saturated are indistinguishable on the reward curve and on `dead_frac`, and their
  fixes are opposites.** Get it backwards — add harder data to a cliff — and you make it worse in
  a way that still looks exactly like flat. This is the trap the question is testing.
- A dead group contributes **exactly zero**, not something small. So the flat curve is not "weak
  learning," it is *a fraction of your batch that is not in the optimisation at all*. Report
  `mean |A|` over **alive** groups only, or the two effects mix and you can read neither.
- Do not open with "raise the learning rate." Three of the four causes are unaffected by it, and
  in the verifier case the LR scales the false-positive gradient identically to the true one.
- The wrong-level trap: `U` is a per-**group** expectation. Averaging `p` across prompts and then
  evaluating `2(G−1)p̄(1−p̄)` is not the mean of the per-prompt values, and the gap is exactly the
  variance you are trying to diagnose.

---

### Q: How would you detect benchmark contamination — and what does it do to *training*?

**Cold card:** eval inflation `= c·(1 − pass@k(p_true))` — **decays in k**: 3.500 / 2.450 / 1.201 / 0.288 / 0.017 / 0.000 pts at k=1/2/4/8/16/64 (c=5%, p_true=0.30) · D3's RL gain: +0.4691 at k=1 → +0.0000 at k=1024, same shape · perturbed twin: honest `Δ≈0`, memorised `Δ≈1−p_true` · **the control is the answer** · a memorised item is a **saturated dead group**

**Short answer:** Perturbed twins, **with a control** — and then the part most answers omit:
contamination is not only an eval problem, it is a **rollout-budget** problem. A memorised item
comes back unanimous, so it is a saturated dead group: it consumes its share of the batch,
returns exactly zero gradient, and scores **1.0** on the reward curve while doing it.

**Deeper:**

**1. Detection — the method.** For each benchmark item, build a matched twin that preserves the
reasoning structure and changes only instance-specific surface (numbers, names, ordering), so a
memorised answer no longer applies. Measure `Δ = p(original) − p(twin)`:

```
honest item:     Δ ≈ 0
memorised item:  Δ ≈ 1 − p_true
```

**2. Detection — the control, which is the actual answer.** Apply the **same perturbation to
items I know are clean** (post-cutoff, or privately authored). If the control shows `Δ ≈ 0`, the
perturbation is difficulty-neutral and the signal on suspect items is memorisation. If the
control moves, **my detector is measuring itself** — a perturbation that is intrinsically hard
convicts every item on the list. *A control is what converts an observation into evidence*; it is
also, verbatim, GSM1k's methodology.

**3. The signature in `k`, with a direction.** Inflation of mean pass@k is `c·(1 − pass@k(p_true))`:

```
c = 5%, p_true = 0.30:   3.500 / 2.450 / 1.201 / 0.288 / 0.017 / 0.000 points  at k = 1/2/4/8/16/64
```

**Invisible by k = 64.** Which is the same shape as yesterday's result about RL:

```
D3  RL's pass@k gain:     +0.4691 at k=1  ->  +0.0000 at k=1024
D4  contamination's lie:  +0.0350 at k=1  ->  +0.0000 at k=64
```

**One sentence generates both: a saturating metric cannot see a change in something already
saturated.** The uncomfortable corollary is that the metric that displays my improvement most
clearly is the metric that displays contamination most clearly, and I cannot have one without the
other — **reporting only pass@1 maximises my exposure to both.**

**4. The training-side consequence — the half nobody has.** A memorised item has `p ≈ 1`, so its
group comes back `c = G`, unanimous, and every advantage is `A_i = 0`.

> **A contaminated item is a saturated dead group.** A contaminated fraction `c` costs exactly `c`
> of the rollout budget for exactly zero gradient — and it is **invisible on the reward curve**,
> because those groups score 1.0. Contamination does not merely inflate the eval; it silently buys
> nothing while looking like the best-performing data in the run.

Which means contamination has a **free** detector on the training side that needs no twins at all:
prompts that are persistently in the `dead & p_obs ≈ 1` bucket from step zero. That is the same
bucket as "genuinely solved," so it is a candidate list rather than a verdict — but it is produced
by rollouts I already ran, and it costs nothing to look.

**Watch out for:**
- Twins without a control are not evidence. If asked for one number, I give the **control's Δ**,
  not the suspect's.
- n-gram / substring overlap detection misses paraphrase and translation contamination entirely,
  and it cannot be run at all against a model whose training corpus I do not have — which is
  every model I would be evaluating.
- Do not report the twin gap as a single scalar over the benchmark. Contamination is
  item-concentrated, so the **distribution** of `Δ` is bimodal and its mean hides that.
- Retiring contaminated items *improves* your measured score change while shrinking the eval set,
  so quote pass@k before and after on the **same** surviving subset, or the cleanup itself becomes
  an unfalsifiable improvement.

---

### Q: How would you validate an end-to-end RLVR/GRPO system without letting held-out gold leak into training?

> **LEARNER-WRITTEN on W6 D5.** The learner independently separated a gold-mutation software
> test from a perturbed-twin contamination test; the scope condition below was added during
> repair.

**Cold card:** every `p` needs an event type · mutate `gold`, hold θ/samples/RNG/`accept` fixed:
update bit-identical · mutate reachable `accept`: update changes · perturbed twins test data
contamination, not code dependency · clean control converts a drop into evidence · calibration
repairs the controller coordinate, not an unsound reward objective

**Short answer:**

First test the software boundary: change the gold labels while holding θ, samples, RNG, and
verifier accepts fixed; the policy update must remain bit-identical. Then change reachable
verifier labels and require the update to change, so the first result is non-vacuous. Separately,
test contamination with perturbed twins and apply the same perturbation to a clean post-cutoff or
privately authored control. A suspect-only drop is evidence of memorisation; a drop in both sets
means the perturbation itself made the task harder.

**Deeper:**

The two tests answer different questions. The metamorphic test proves that hidden truth is not a
software dependency of the reward or policy loss. The twin test asks whether benchmark content
was already in the model's data. Conflating them can produce a clean dependency graph and a
contaminated benchmark, or a clean benchmark and a gold-leaking implementation.

Gold may legitimately enter a held-out panel and a periodic calibration audit. To isolate the
policy-loss boundary, freeze the data policy or use the raw sensor with uniform prompt sampling;
otherwise changing gold can intentionally change calibrated curriculum selection even though it
never enters the reward. Finally, calibration only maps `p_obs` into a scheduling estimate of
`p_gold`; it cannot turn a false-positive verifier reward into a correct objective.

**Watch out for:**

- No twin drop means **no evidence of memorisation**, not proof of genuine reasoning.
- A twin drop without a clean perturbation control is not evidence; the transformation may simply
  be harder.
- Always pair the gold-mutation test with an `accept`-mutation control, or a no-op trainer passes.
