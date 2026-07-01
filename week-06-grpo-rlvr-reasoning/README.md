# Week 06 - GRPO, RLVR & Reasoning Post-Training

## Theme

Understand modern reasoning-focused RL: verifiable rewards, group baselines, pass@k, curriculum.

## Weekly Deliverable

Mini RLVR repo: generate N completions, score with a verifier, compute group advantages, update policy (or simulate the update math).

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | GRPO vs PPO | GRPO vs PPO comparison notes |
| Day 2 | RL with Verifiable Rewards | Verifier design + examples |
| Day 3 | Sampling, pass@k & Rejection Sampling | Sampling / pass@k experiments |
| Day 4 | Reasoning Data Curriculum | Curriculum design notes |
| Day 5 | Mini RLVR / GRPO Loop | Mini RLVR repo in code/ |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Explain GRPO vs PPO: group-relative advantages, no critic, memory/variance tradeoffs.
2. Design RL with verifiable rewards for math/code tasks.
3. Reason about sampling: group size, temperature, pass@k, rejection sampling, self-consistency.
4. Design a reasoning data curriculum and avoid zero-reward cliffs / contamination.
5. Implement a mini RLVR/GRPO-style loop.

## Core Readings

- DeepSeekMath / GRPO
- RL-with-verifiable-rewards references (math/code)

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
