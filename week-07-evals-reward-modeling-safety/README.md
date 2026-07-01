# Week 07 - Evaluations, Reward Modeling & Safety-Oriented ML

## Theme

Be unusually good at eval design; this is a frontier-lab differentiator.

## Weekly Deliverable

Evaluation harness with pairwise judge, bootstrap CI, failure taxonomy, and model-card-style report.

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | Eval Validity, CIs & Regression Tests | Eval-validity checklist + CI code |
| Day 2 | LLM-as-Judge | Pairwise judge prompt + calibration notes |
| Day 3 | Reward Model Design | RM design doc |
| Day 4 | Safety & Misuse Evals | Safety-eval taxonomy |
| Day 5 | Build an Eval Harness | Eval harness + model-card report |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Explain eval validity: construct validity, leakage, confidence intervals, regression tests.
2. Design LLM-as-judge setups and account for calibration/bias.
3. Design reward models and handle annotator disagreement and calibration.
4. Design safety/misuse evals: classifiers, red teaming, prompt injection.
5. Build a reusable eval harness for post-training experiments.

## Core Readings

- Constitutional AI / RLAIF (Anthropic)
- Selected eval papers/blogs (only what supports the exercise)

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
