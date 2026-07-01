# Deliverable - Week 01 Day 01: Role Map + Baseline Diagnostic

## What I Built / Derived

- A **role map** of the frontier-lab MLE/RE interview (5 competency axes) and a self-scored **competency matrix** with evidence.
- A working **lab environment** (reused the workspace `.venv`: torch 2.12, Python 3.14) with `requirements.txt`, a passing `smoke_test.py`, and a completed `fluency_probe.py` (5/5).
- A polished **positioning statement**.

## Core Takeaway

Frontier-lab loops grade ~5 axes: ML coding, ML/research depth, ML system design, research taste, and narrative. My real edge is deeper RL understanding than I gave myself credit for; my real gaps are (1) ML systems / memory-cost reasoning and (2) making my RL depth *interview-crisp and derivable on demand*. The next 9 weeks target those.

## The 5 Competency Axes

| Axis | What it tests | Built in |
|---|---|---|
| ML coding | PyTorch fluency, custom losses, live debugging | W1-W2, W10 |
| ML/research depth | post-training, RL, evals theory | W3-W6 |
| ML system design | training / eval / reward / inference pipelines, cost/memory | W8 |
| Research taste | experiment design, ablations, valid conclusions | W7, W9 |
| Narrative | project stories, positioning | W10 |

## Competency Matrix (baseline)

| Axis | Self-score | Adjusted | Evidence |
|---|---|---|---|
| ML coding | 3 | 3 | Fluency probe 5/5 in ~20 min; gather + masking needed a nudge |
| ML/research depth | unsure | 3.5 | Policy-gradient + baseline answer was strong-hire quality; prior DPO/GRPO exposure |
| ML system design | low | 2 | **Top gap** - blanked on Adam optimizer-state memory (params + grads + 2 moments) |
| Research taste | 3 | 3 | To confirm via Day 5 ablation discipline |
| Narrative | 3 | 3 | Solid raw material (see below); needs tightening |

### Top-2 gaps -> priority

1. **ML systems / memory-cost reasoning** (Week 8; I'll also fold memory math into earlier days). Baseline fact to internalize: for a 7B model in fp32, params ~28 GB + grads ~28 GB + Adam states ~56 GB ~= 112 GB before activations.
2. **Interview-crisp RL depth**: I understand PPO/DPO/GRPO conceptually but need to *derive them from scratch on demand* (Weeks 4-6).

## Probe Baseline (metric to beat)

- **Total time:** ~20 min for 4 core primitives + 1 stretch (target by end of Week 1: <10 min, no lookups).
- **Slow spots:** `torch.gather` semantics and the masked-mean idiom.
- **Result:** 5/5 PASS. Key idiom locked in: `log_softmax(dim=-1)` + `gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)` works for both `(N,V)` and `(B,T,V)`; LM loss = `-masked_mean(seq_token_logprobs, mask)`.

## Interview Answer

Prompt:

> Tell me about yourself - why post-training, why this lab?

Answer (~45s):

> I'm an ML engineer with a background in large-scale retrieval and personalization, and
> over the last stretch I've moved from RAG into post-training. At Amazon I built a
> personalized prompt-augmentation system: for a given search query it retrieves the
> relevant slice of a user's profile by semantic similarity and rewrites the query with
> that context - so "pet food" becomes "pet food for cats" for a user who has a cat. To
> push quality past our Sonnet teacher model, I ran DPO on preference data and beat the
> teacher on our eval. That's the thread I want to pull at a frontier lab: I've felt
> exactly where retrieval and prompting hit a ceiling and where post-training plus
> trustworthy evals are what actually move the needle. I want to go deep on RL/reasoning
> and evaluation for LLMs.

### Narrative notes / to strengthen later (Week 10)

- Quantify the DPO win (metric, delta vs Sonnet teacher, dataset size) - numbers make it land.
- Have a 30s and a 3-min version; be ready for deep-dives on: why DPO over PPO here, how preferences were collected, how the eval avoided leakage.
- This maps to two `interview-bank/project-stories.md` entries: (1) personalization RAG / prompt augmentation, (2) DPO-above-teacher.

## Reusable Artifact

- `week-01-lm-foundations-pytorch/code/smoke_test.py` - env + autograd sanity check.
- `week-01-lm-foundations-pytorch/code/fluency_probe.py` - baseline primitives (softmax, cross_entropy, masked_mean, sgd_step, seq_token_logprobs).
- `requirements.txt` - reproducible env.

## Next Session

Day 2 - Token-Level Language Modeling: derive `dCE/dz = softmax(z) - onehot(y)` and build a from-scratch, unit-tested CE/perplexity module (the `softmax`/`cross_entropy` probe functions are the warm-up).
