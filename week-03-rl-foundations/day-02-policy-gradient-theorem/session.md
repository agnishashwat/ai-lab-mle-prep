# Week 03 Day 02 — Policy Gradient Theorem & Score-Function Estimator

## Objective

Derive \(\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\big[\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)\,G_t\big]\) from first principles via the
log-derivative (score-function/REINFORCE) trick, and understand why we can move the
gradient inside a \(\theta\)-dependent distribution and why dynamics \(P\) drop out.

## Prerequisites

- Day 1: MDP, \(p_\theta(\tau)\), return, \(V^\pi/Q^\pi/A^\pi\).
- Log-derivative identity \(\nabla_\theta p_\theta = p_\theta\nabla_\theta\log p_\theta\).

## Required Reading

- Spinning Up — "Intro to Policy Optimization" (derivation of the simplest policy
  gradient + reward-to-go + baseline).

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 1)
MDP tuple; Bellman expectation for \(V^\pi\); \(\mathbb{E}_{a\sim\pi}[A^\pi]=0\);
LLM→MDP mapping; expectation vs optimality.

### 0:10-0:35 - Concept
Score-function estimator vs pathwise/reparameterization gradient (why RL uses the
former: reward is a black box, actions are discrete tokens). REINFORCE as Monte-Carlo
policy gradient. On-policy nature: samples must come from the current \(\pi_\theta\).

### 0:35-1:00 - Derivation
1. \(\nabla_\theta J=\mathbb{E}[\,R(\tau)\nabla_\theta\log p_\theta(\tau)\,]\) via the
   log-derivative trick.
2. \(\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\) —
   **dynamics and \(\rho_0\) vanish** (no \(\theta\)). Big deal: model-free.
3. **Reward-to-go / causality:** replace \(R(\tau)\) with \(G_t\) (future reward only)
   — past rewards are independent of \(a_t\), zero-expectation.
4. Preview the \(Q/A\) forms:
   \(\nabla_\theta J=\mathbb{E}[\sum_t\nabla\log\pi\,A^\pi(s_t,a_t)]\) (bridges to Day 3).

### 1:00-1:45 - Coding
`code/reinforce.py`: REINFORCE on a toy env (e.g. the Day-1 gridworld or a bandit).
Stubs + self-check: `discount_cumsum` (reward-to-go), `policy_gradient_loss`
(`-(logp * returns).mean()`), a training loop; assert average return improves over
baseline-random and that detaching returns is correct (they're weights, not
differentiated). Include a gradient-estimator variance print.

### 1:45-1:55 - Interview Drill
"Derive REINFORCE in 4 lines." "Why does the transition model disappear from the
gradient?" "Why is vanilla REINFORCE high-variance?"

### 1:55-2:00 - Deliverable & Next
Next: **Day 3 — Baselines, advantage, variance reduction.**

## Checkpoint Questions
1. State and derive the policy gradient theorem (log-derivative trick).
2. Why do \(P\) and \(\rho_0\) drop out of the gradient?
3. What is reward-to-go and why is it valid (causality argument)?
4. Score-function vs reparameterization gradient — when each, and why RL uses SF.
5. Why is REINFORCE on-policy, and what breaks if you reuse old samples?

## Frontier-Lab Interview Angle
"Derive policy gradient" is the single most common RL whiteboard prompt. The unbiased
log-derivative estimator is also literally what PPO/GRPO compute per token
(\(\nabla\log\pi_\theta(\text{token})\)); interviewers probe whether you know the
gradient is over the *policy's own samples* (on-policy) — motivating importance
sampling + clipping next week.
