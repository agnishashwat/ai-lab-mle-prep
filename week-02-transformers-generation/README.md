# Week 02 - Transformer Internals & Efficient Generation

## Theme

Know transformer mechanics well enough to debug training/inference issues.

## Weekly Deliverable

Annotated decoder-only mini-implementation + generation notebook.

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | Attention from First Principles | Attention derivation + minimal implementation |
| Day 2 | Transformer Block & Numerical Stability | Transformer block in code/ with stability notes |
| Day 3 | Decoding Strategies | Decoding utilities + comparison notebook |
| Day 4 | KV Cache, Batching & Throughput | KV-cache memory calculator + notes |
| Day 5 | Generation Debug Lab | Bug taxonomy + fixes in notes/ |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Derive scaled dot-product attention (Q/K/V, scaling, masks) and explain positional encodings.
2. Implement a numerically stable transformer block.
3. Explain and implement decoding strategies (temperature, top-k/top-p, beam).
4. Do KV-cache and inference memory/throughput math.
5. Debug bad generations, NaNs, shape and masking bugs systematically.

## Core Readings

- Attention Is All You Need (attention section only)
- The Illustrated Transformer (compact reference)

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
