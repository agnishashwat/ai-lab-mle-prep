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

> **YOURS TO WRITE.** The cold card below is seeded with your own W6 D3 measurements so you
> are not starting from a blank page. The prose is the exercise — the drill showed MECHANISM
> slots filling 9/9 while THESIS slots went 0/3, and this is a thesis question.
> A full worked answer already exists in `week-06-.../day-03-.../deliverable.md` — write
> yours first, then compare.

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

**Deeper:**

**Watch out for:**

---

### Q: Best-of-64 against your reward model beat greedy decoding. Ship it?

> **YOURS TO WRITE.** Same deal. This one is a FILTER-vs-RANKER question wearing a
> best-of-n costume, and the number that answers it is a KL.

**Cold card (drill this alone):**

```
best-of-n IS a KL move you did not budget for:
  KL(BoN_n || pi) = log n - (n-1)/n
  = 0.1931 / 1.2044 / 3.1745 / 4.5491 nats at n = 2 / 8 / 64 / 256
  every DOUBLING costs ~log 2; measured +0.6853 nats from n=64 to n=128
  so BoN-64 is a 3.17-nat move against a policy you KL-constrained during training

and it is a RANKER, which is the bad half of the split. SAME verifier, SAME budget:
  FILTER  P(hack|emitted) = gamma(1-p)/[gamma(1-p)+(1-alpha)p] = 9.09%, n-INVARIANT
  RANKER  P(hack|emitted) = 1 - (1-gamma)^n   = pass@n FOR THE HACK SET
          1.0000% / 7.7255% / 47.4404% / 92.3685%  at n = 1/8/64/256
  at n=256: 9.09% vs 92.37% -- 10.16x, from the SELECTION RULE alone

"beat greedy decoding" was measured with the RM as judge. Ask for the held-out column.
```

**Short answer:**

**Deeper:**

**Watch out for:**
