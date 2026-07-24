# Week 03 Day 05 — KL-Constrained RL & the Reference Policy (Week 3 capstone)

## Objective

Understand why LLM post-training maximizes reward *subject to a KL penalty to a frozen
reference policy*, derive the closed-form optimum of the KL-regularized objective
\(\pi^*(a|s)\propto\pi_{\text{ref}}(a|s)\exp(r(s,a)/\beta)\), and connect it forward to
PPO's KL penalty (Week 4) and DPO's reparameterization (Week 5).

## Prerequisites
- Days 1-4 (MDP, PG, baselines, GAE). KL divergence definition + properties.

## Required Reading
- InstructGPT §"RL" (the PPO objective with KL penalty) — the KL term only.
- Optional: the DPO paper §3–4 (derivation of the RLHF optimum) — read for the closed form.

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 4)
GAE formula + limits; bias from bootstrapping; \(\delta_t\); credit assignment for
terminal reward.

### 0:10-0:35 - Concept
Why constrain to \(\pi_{\text{ref}}\): reward models are proxies → unconstrained
maximization = **reward hacking / over-optimization** + catastrophic forgetting of the
SFT distribution. KL as a trust region / distributional leash. Forward vs reverse KL and
mode-seeking behavior. The two ways KL enters: as a penalty in the reward vs as a hard
constraint (and PPO's ratio-clip as an *implicit* local KL trust region).

### 0:35-1:00 - Derivation
1. Objective \(\max_\pi \mathbb{E}_{a\sim\pi}[r(s,a)] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})\).
2. Lagrangian + normalization ⇒ **closed form**
   \(\pi^*(a|s)=\frac1{Z(s)}\pi_{\text{ref}}(a|s)\exp\!\big(r(s,a)/\beta\big)\).
3. Interpret: exponential tilt of the reference by reward; \(\beta\to\infty\Rightarrow\pi_{\text{ref}}\),
   \(\beta\to0\Rightarrow\) greedy argmax reward. \(Z(s)\) is the intractable partition
   function (why we still need RL / or the DPO trick that cancels it).
4. Preview: rearranging the closed form to express \(r\) in terms of \(\log(\pi^*/\pi_{\text{ref}})\)
   is exactly the DPO reparameterization (Week 5) — planted here.

### 1:00-1:45 - Coding
`code/kl_rl.py`: on the Day-1 finite MDP / bandit with a known reward, (a) implement the
KL-regularized closed-form optimal policy for a sweep of \(\beta\) and verify it matches a
numerical `scipy`/gradient solve; (b) plot the reward↔KL frontier (Pareto curve) as \(\beta\)
varies; (c) demonstrate **reward hacking**: give a mis-specified proxy reward and show low
\(\beta\) chases the proxy far from \(\pi_{\text{ref}}\) while true reward drops. Self-check:
closed form == numeric optimum; monotone frontier; \(\beta\) limits recover ref/greedy.

### 1:45-1:55 - Interview Drill
"Why KL to a reference in RLHF?" "Derive the optimal KL-regularized policy." "How does
this connect to DPO?" "What is reward over-optimization and how does \(\beta\) control it?"

### 1:55-2:00 - Deliverable & Week Wrap
Write `deliverable.md`, `week-summary.md`, `retrospective.md`. Next week: **Week 4 — RLHF
& PPO for LLMs**, which assembles Days 1-5 into the full InstructGPT pipeline.

## Checkpoint Questions
1. Write the KL-regularized RLHF objective and derive its closed-form optimum.
2. What does \(\beta\) control; what are the \(\beta\to0,\infty\) limits?
3. Why constrain to \(\pi_{\text{ref}}\) at all (name two failure modes it prevents)?
4. Forward vs reverse KL — which does the penalty use and what behavior does it induce?
5. Sketch how the closed form leads to the DPO objective (cancel \(Z(s)\)).

## Frontier-Lab Interview Angle
The KL-to-reference idea is the through-line of *all* modern post-training: it's PPO's KL
penalty, DPO's implicit reward, and the reason GRPO/RLVR still regularize toward a base
model. "Derive the optimal KL-regularized policy" is a top-tier discriminating question —
it proves you understand RLHF as constrained optimization, not a recipe, and it's the
single cleanest bridge from RL (Weeks 3-4) to preference optimization (Week 5).
