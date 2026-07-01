# Week 08 - Training & Inference Systems for Post-Training

## Theme

Know enough systems to reason about cost, scale, and bottlenecks.

## Weekly Deliverable

Systems design doc for a scalable SFT+DPO+eval pipeline with cost/memory estimates.

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | GPU Memory Math | Memory calculator + worked examples |
| Day 2 | Fine-Tuning: LoRA/QLoRA & FSDP/ZeRO | Fine-tuning tradeoff notes |
| Day 3 | Distributed Training Basics | Parallelism cheat-sheet |
| Day 4 | Inference Systems | Inference systems notes |
| Day 5 | End-to-End Pipeline Architecture | Systems design doc in artifacts/ |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Do GPU memory math: params, optimizer states, activations, KV cache.
2. Compare LoRA/QLoRA vs full fine-tune and FSDP/ZeRO/gradient checkpointing.
3. Explain data/model/pipeline parallelism, collectives, and failure modes.
4. Explain inference systems: continuous batching, paged attention, quantization.
5. Architect an end-to-end post-training pipeline.

## Core Readings

- FSDP / ZeRO references; LoRA / QLoRA papers
- PagedAttention / vLLM (inference systems)

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
