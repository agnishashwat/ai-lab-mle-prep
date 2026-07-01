# Week 03 - RL Foundations for Post-Training

## Theme

Derive the machinery behind PPO/GRPO rather than memorizing formulas.

## Weekly Deliverable

Small policy-gradient implementation on a toy environment, with advantage/KL ablations.

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | MDPs, Returns & Value Functions | MDP / value-function derivation notes |
| Day 2 | Policy Gradient Theorem | Full PG derivation + toy gradient check |
| Day 3 | Baselines, Advantages & Variance Reduction | Variance-reduction ablation notes |
| Day 4 | GAE & the Bias/Variance Tradeoff | GAE implementation + lambda sweep |
| Day 5 | KL-Constrained RL | Toy policy-gradient run with KL ablation |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Define MDPs, returns, value functions, and Bellman equations.
2. Derive the policy gradient theorem and the score-function estimator.
3. Explain baselines, advantages, variance reduction, and entropy regularization.
4. Explain GAE and the bias/variance tradeoff of bootstrapping.
5. Explain why LLM post-training regularizes with KL to a reference policy.

## Core Readings

- OpenAI Spinning Up: Intro to Policy Optimization
- Sutton & Barto, Ch. 13 (policy-gradient methods)

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
