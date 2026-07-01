# Week 04 Day 02 - PPO Objective

## Objective

Understand PPO as a practical approximation to trust-region policy optimization, with
emphasis on old policy ratios, clipping, multiple epochs, and why this matters for LLM
RLHF.

## Prerequisites

- Policy gradient theorem
- Advantage functions
- KL-regularized RL intuition

## Required Reading

- PPO paper: objective section only
- Optional: Spinning Up PPO notes

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz

1. What is the score-function estimator?
2. Why do we use a baseline?
3. What does KL regularization do in post-training?

### 0:10-0:35 - Concept Explanation

Notes here.

### 0:35-1:00 - Derivation

Derive:

```text
r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

Then derive the clipped surrogate objective.

### 1:00-1:45 - Coding Task

Implement a small PPO loss function:

- input: old logprobs, new logprobs, advantages
- compute ratio
- compute unclipped objective
- compute clipped objective
- return negative mean loss

### 1:45-1:55 - Interview Drill

Explain why PPO uses an old policy and why the ratio is not always 1 after the first
gradient update.

### 1:55-2:00 - Deliverable

Commit `ppo_loss.py` and write a short explanation in `deliverable.md`.

## Checkpoint Questions

1. Why does PPO need old logprobs?
2. Why does clipping reduce destructive updates?
3. What happens if the advantage estimate is poor?
4. How does PPO differ in LLM sequence generation?
5. What can go wrong with multiple epochs over the same rollout?

## Frontier-Lab Interview Angle

You may be asked to debug an RLHF training instability where KL spikes, reward improves,
but human evals degrade. PPO mechanics are central to explaining this.
