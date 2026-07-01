# Week 01 - Role Calibration, LM Foundations & PyTorch Fluency

## Theme

Build a precise mental model of language modeling and refresh implementation muscle.

## Weekly Deliverable

Tiny reproducible LM training/eval harness with tests and clean logging.

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | Role Map + Baseline Diagnostic | Role map + gap diagnostic |
| Day 2 | Token-Level Language Modeling | From-scratch cross-entropy + perplexity checks |
| Day 3 | PyTorch Autograd, Shapes & Mixed Precision | Autograd + shape/masking drill notebook |
| Day 4 | Minimal Transformer Training Loop | Reusable training-loop skeleton in code/ |
| Day 5 | Evaluation Basics & Ablation Discipline | Eval/ablation harness + notes |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Explain the token-level LM objective end to end: logits -> logprobs -> cross-entropy -> perplexity.
2. Reason about tensor shapes, broadcasting, and masking without guessing.
3. Diagnose common autograd, mixed-precision, and NaN failure modes.
4. Stand up a minimal, correct decoder-only training loop from scratch.
5. Apply basic eval and ablation discipline (heldout eval, overfitting checks).

## Core Readings

- Karpathy nanoGPT / minGPT (reference training loop)
- Cross-entropy & perplexity refresher (any compact source)

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
