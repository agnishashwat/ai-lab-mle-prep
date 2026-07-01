# Week 01 Day 05 — Evaluation Basics & Ablation Discipline

## Objective

Turn the Day-4 loop into a trustworthy harness: heldout eval, loss-curve reading, overfitting detection, and one-variable-at-a-time ablations with seed variance — the habits behind credible results.

## Prerequisites

- Day 4 training loop
- Perplexity from Day 2

## Required Reading

- A short piece on train/val/test hygiene and overfitting (keep it to one)
- Optional: a note on bootstrap confidence intervals (previews Week 7)

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 4)

1. `zero_grad` / `backward` / `step` order?
2. Effective batch size formula?
3. What toggles between train and eval?

### 0:10-0:35 - Concept Explanation (First Principles)

Train/val/test split hygiene; heldout eval; loss-curve pathologies (underfit, overfit, LR-too-high divergence, LR-too-low crawl); the overfitting signature (train down / val up); early stopping. Ablation discipline: change one variable at a time, fix the controls, fix the compute budget, run >=2 seeds, and report mean +/- range. Eval loss vs downstream metric; leakage/contamination.

### 0:35-1:00 - Deconstruction

Given several described loss curves, name the pathology + fix for each. Then design one clean ablation: state the control, the single variable, the metric, and the seed count. Reason about why a single seed can lie.

### 1:00-1:45 - Coding Task

Extend the harness in `code/`:

- Add train/val split + periodic val loss + perplexity; log both curves.
- Run a small ablation (e.g., LR in {high, ok, low}, or +/- weight decay) across >=2 seeds; tabulate mean +/- range.
- Add the classic `pytest` sanity test: *overfit a single batch* — loss should collapse to ~0 over N steps.

This completes the weekly deliverable: a tiny reproducible LM training/eval harness with tests and logging.

### 1:45-1:55 - Interview Drill

"How do you know an improvement is real and not noise?" and "Train loss drops but val is flat — what now?"

### 1:55-2:00 - Deliverable & Next Session (+ week wrap)

`deliverable.md`: the ablation table + curves + the overfit-one-batch test, plus a one-paragraph conclusion written like an experiment report. Then start `week-summary.md` and `retrospective.md`. Next week: transformers + generation.

## Checkpoint Questions

1. What does the "overfit a single batch" test verify, and what does failure imply?
2. Name 3 loss-curve pathologies and their fixes.
3. Why report mean +/- variance across seeds instead of one number?
4. Eval loss vs downstream metric — when do they diverge?
5. What is the cardinal rule of ablations?

## Frontier-Lab Interview Angle

Eval taste is a top differentiator. Expect "how would you evaluate this?" and "is this result significant?" The overfit-one-batch test is a well-known signal that your pipeline (data -> loss -> optimizer) is wired correctly.
