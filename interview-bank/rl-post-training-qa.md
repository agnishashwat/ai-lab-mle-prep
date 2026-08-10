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

**Short answer:**

**Deeper:**

**Watch out for:**

### Q: Derive the PPO clipped objective and explain why old logprobs are necessary.

**Short answer:**

**Deeper:**

**Watch out for:**

### Q: Derive DPO from KL-regularized RLHF. When is DPO the wrong tool?

**Short answer:**

**Deeper:**

**Watch out for:**

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

