# Week 01 Day 04 — Minimal Transformer Training Loop

## Objective

Assemble a minimal, correct, reproducible decoder-only training loop (data -> forward -> loss -> step) with clean logging — the backbone of this week's harness deliverable.

## Prerequisites

- Days 2-3 (cross-entropy, autograd, shapes, mixed precision)
- A tiny dataset (char-level tiny-Shakespeare or a toy copy task)

## Required Reading

- nanoGPT `train.py` — read the loop only (ignore the DDP/config machinery)
- Optional: AdamW + cosine schedule notes

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 3)

1. `reshape` vs `view`?
2. Why loss scaling for fp16?
3. Correct masked-mean denominator?

### 0:10-0:35 - Concept Explanation (First Principles)

Anatomy of a training loop: dataset/dataloader -> batch -> forward -> loss -> `zero_grad` -> `backward` -> grad clip -> `step` -> scheduler step. Cover AdamW, warmup+cosine LR, gradient clipping, gradient accumulation, seeding/determinism, checkpointing, train vs eval mode, and periodic logging. Keep the *model* tiny so the *loop* is the star.

### 0:35-1:00 - Deconstruction

Reduce nanoGPT's loop to its minimal invariants; rewrite it as ~20 lines of pseudocode. Derive effective batch size = `micro_batch × grad_accum × world_size` and explain why it couples to the learning rate.

### 1:00-1:45 - Coding Task

In `code/train.py`:

- Toy dataset (char-level or a copy task) + simple batching.
- Tiny model (a 1-2 layer tiny GPT, or start with an embedding+linear bigram).
- Loop: AdamW, warmup+cosine, grad clip, fixed seed, periodic eval print, loss history.
- Reproducibility check: same seed -> identical first-N losses.

This is the seed of the weekly "tiny LM harness."

### 1:45-1:55 - Interview Drill

"Write a correct training loop from scratch — what do people forget?" (zero_grad placement, `eval()` + `no_grad()` at eval, seeding, grad clip).

### 1:55-2:00 - Deliverable & Next Session

`deliverable.md`: runnable `train.py`, a short loss trace, and the reproducibility note. Next: Day 5 (eval + ablations).

## Checkpoint Questions

1. Exact order of `zero_grad` / `backward` / `step` — what breaks if reordered?
2. Effective batch size with micro-batching, accumulation, and N GPUs?
3. Why warmup *then* cosine decay — what does warmup prevent?
4. What must change between train and eval modes?
5. How do you make a run reproducible, and why might GPU runs still differ bit-for-bit?

## Frontier-Lab Interview Angle

"Implement a training loop" is a canonical ML-coding round. Common tells that lose points: forgetting `zero_grad`, evaluating without `eval()`/`no_grad()`, no seeding, and no gradient clipping.
