# Week 04 - RLHF and PPO for LLMs

## Theme

Understand the canonical RLHF pipeline and why PPO is engineered the way it is.

## Weekly Deliverable

PPO/RLHF explainer notebook with toy code and failure-mode demonstrations.

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | InstructGPT / RLHF Pipeline | RLHF pipeline diagram + notes |
| Day 2 | PPO Objective | PPO loss implementation (ppo_loss.py) |
| Day 3 | Token-Level PPO for LLMs | Token-level logprob/value-head walkthrough |
| Day 4 | RLHF Failure Modes | Reward-hacking / KL-collapse demonstrations |
| Day 5 | Toy PPO-Style RLHF Loop | End-to-end toy RLHF loop |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Explain SFT -> RM -> PPO -> evals.
2. Derive the PPO clipped objective.
3. Explain sequence-level reward with token-level logprobs.
4. Diagnose KL collapse, reward hacking, and length exploitation.
5. Implement a minimal PPO-style loss from scratch.

## Core Readings

- InstructGPT (Ouyang et al.) - pipeline overview
- PPO (Schulman et al.) - objective section only
- Optional: Spinning Up PPO notes

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
