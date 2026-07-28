# Week 04 - RLHF and PPO for LLMs

## Theme

Understand the canonical RLHF pipeline and why PPO is engineered the way it is.

## Weekly Deliverable

PPO/RLHF explainer notebook with toy code and failure-mode demonstrations.

## Sessions

| Day | Topic | Code | Headline |
|---|---|---|---|
| Day 1 | Pipeline + Bradley-Terry reward model | `toy_rlhf.py` | RM identified only up to a per-prompt constant; RM error grows with KL |
| Day 2 | PPO objective: ratio, clip, epochs | `ppo_loss.py` | Clipping bounds the step, not the walk (measured drift vs explicit KL) |
| Day 3 | Token-level PPO: value head, sequence reward | `token_ppo.py` | A *learned* critic's bias tracks the state-varying error, not the MSE |
| Day 4 | Failure modes | `failure_modes.py` | Goodhart with a learned RM; off-policy bias measured against exact ∇J |
| Day 5 | Toy PPO-RLHF loop (capstone) | `ppo_rlhf.py` | Adaptive β = dual ascent; its fixed point matches the closed-form frontier |

## The Week's Design Decision

Week 3's results were assertions rather than claims because a finite MDP gave exact ground
truth (Bellman backups + central finite differences). Week 4 keeps that discipline in a
setting where tokens, EOS, sequence reward and length are **literal**: Day 1 builds a
token-level LM environment small enough to **enumerate every terminating sequence**, so
`exact_J`, `exact_sequence_kl` and an exact gradient are available all week. Every
unbiasedness or bias claim this week is checked against enumeration, not against a plot.

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
