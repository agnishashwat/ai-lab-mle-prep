# Week 03 Day 03 — Baselines, Advantage & Variance Reduction

## Objective

Prove that subtracting a state-dependent baseline \(b(s)\) from the policy gradient is
**unbiased** but **variance-reducing**, that the value function \(V^\pi\) is the natural
baseline (giving the advantage form), and understand entropy regularization as an
exploration/collapse guard.

## Prerequisites
- Day 2: \(\nabla_\theta J=\mathbb{E}[\sum_t\nabla\log\pi\,G_t]\).
- Variance of a Monte-Carlo estimator; \(\mathrm{Var}=\mathbb{E}[X^2]-\mathbb{E}[X]^2\).

## Required Reading
- Spinning Up — baselines section (EGLP lemma / "don't let the past distract you").

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 2)
Policy gradient theorem; why \(P\) drops; reward-to-go causality; SF vs reparam.

### 0:10-0:35 - Concept
Variance is the whole game in policy gradients (sparse LLM rewards ⇒ brutal variance).
Baselines, actor-critic (learn \(V_\phi\) as the baseline), the actor/critic split.
Entropy bonus \(+\beta H(\pi)\) to prevent premature determinism/collapse.

### 0:35-1:00 - Derivation
1. **EGLP lemma:** \(\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a|s)]=0\).
2. ⇒ \(\mathbb{E}[\nabla\log\pi\cdot b(s)]=0\): any state-dep baseline is **unbiased**.
3. Optimal baseline minimizing variance (the \(\nabla\log\pi\)-weighted mean of returns);
   \(V^\pi\) is the standard near-optimal choice ⇒ advantage form
   \(\nabla_\theta J=\mathbb{E}[\sum_t\nabla\log\pi\,A^\pi(s_t,a_t)]\).
4. Why \(b\) must depend only on \(s\) (not \(a\)) to stay unbiased.

### 1:00-1:45 - Coding
Extend `code/reinforce.py` → `code/actor_critic.py`: add a learned value baseline,
compute advantages \(G_t-V_\phi(s_t)\), and **measure gradient variance** with vs
without baseline on the same seeds (the headline plot/number). Add an entropy-bonus
ablation. Self-check: baseline leaves the *mean* gradient ~unchanged but shrinks
variance; return improves faster/more stably.

### 1:45-1:55 - Interview Drill
"Prove a baseline doesn't bias the gradient." "Why \(V\) and not \(Q\) as baseline?"
"What does entropy regularization buy you and how does it fail?"

### 1:55-2:00 - Deliverable & Next
Next: **Day 4 — GAE & the bias/variance tradeoff.**

## Checkpoint Questions
1. Prove \(\mathbb{E}_{a\sim\pi}[\nabla\log\pi]=0\) (EGLP) and hence baseline unbiasedness.
2. Why does subtracting \(V^\pi\) reduce variance? What's the intuition of \(A^\pi\)?
3. Why must the baseline be a function of state only?
4. Actor-critic: what does the critic regress on, and what's the bias it introduces?
5. Entropy regularization: objective term, effect, failure mode.

## Frontier-Lab Interview Angle
This is the crux linking REINFORCE→PPO→GRPO: PPO uses a learned critic + GAE for the
advantage; **GRPO drops the critic** and uses the group-relative mean reward as the
baseline. Being able to say "any state-only baseline is unbiased, so a group mean is a
legit (Monte-Carlo) baseline — GRPO trades a critic's bias for group-sampling variance"
is exactly the kind of crisp tradeoff answer labs want.
