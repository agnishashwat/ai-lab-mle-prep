# Week 01 Day 01 — Role Map + Baseline Diagnostic

## Objective

Map the AI-lab MLE/RE interview surface area and run an honest baseline self-diagnostic, so the next 9 weeks target your highest-leverage gaps instead of re-learning what you already know.

## Prerequisites

- Your own resume / project history (Amazon + LinkedIn retrieval, personalization, post-training exposure)
- A working Python 3.10+ install

## Required Reading

- 2-3 real target JDs (e.g., OpenAI "Research Engineer", Anthropic "Member of Technical Staff / ML Engineer") — read for the *implicit rubric*, not the buzzwords
- Optional: skim your own `curriculum.md` and `progress.md`

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (cold baseline)

No prior material yet — use this as a *cold* probe. Answer from memory, no notes, and mark where you stall (that is data):

1. Write next-token cross-entropy loss for a batch of sequences.
2. State the policy gradient theorem in one line.
3. What is the extra memory cost of the Adam optimizer vs plain SGD?

### 0:10-0:35 - Concept Explanation (the interview rubric)

The frontier-lab MLE/RE loop tests ~5 competency axes:

1. **ML coding** — PyTorch tensors, custom losses, dataloaders, live debugging.
2. **ML/research depth** — post-training (SFT/DPO/PPO/GRPO), RL, evals.
3. **ML system design** — training / eval / reward / inference pipelines at scale.
4. **Research taste** — experiment design, ablations, trustworthy conclusions.
5. **Narrative** — crisp project stories tying prior work to the role.

Map each axis to weeks: coding → W1-2, depth → W3-6, evals → W7, systems → W8, taste → W7/W9, narrative → W10.

### 0:35-1:00 - Deconstruction (build your competency matrix)

Take the 2-3 JDs. Extract every concrete requirement and bucket it into the 5 axes. For each axis, self-score 1-5 **with evidence** ("I've shipped X" beats "I've read about Y"). Circle your 2-3 real gaps.

### 1:00-1:45 - Coding Task (set up the lab + fluency probe)

1. Create the dev env: `python -m venv .venv`, add a `requirements.txt` (`torch`, `numpy`, `pytest`, `matplotlib`), install.
2. Write `code/smoke_test.py`: make a tensor with `requires_grad=True`, take one gradient-descent step on `f(x) = (x - 3)**2`, assert `x` moved toward 3.
3. Timed fluency probe (record wall-clock per task). From memory, implement: (a) numerically stable `softmax`, (b) `cross_entropy`, (c) masked mean over padded rows, (d) one manual SGD step. These times are your baseline.

Commit under `week-01-lm-foundations-pytorch/code/`.

### 1:45-1:55 - Interview Drill

Deliver a 60-second "tell me about yourself" + "why post-training / why this lab" positioning statement out loud. Record it; cut it to a tight 45 seconds.

### 1:55-2:00 - Deliverable & Next Session

Fill `deliverable.md` with the role map, competency matrix (scores + evidence), top-3 gaps, and positioning statement. Next: Day 2 (token-level LM).

## Checkpoint Questions

1. What are the 5 competency axes, and which is your weakest — with evidence?
2. Which curriculum weeks attack your top-2 gaps?
3. What is your one-sentence positioning statement?
4. What does "research taste" concretely look like in an interview answer?
5. Which baseline artifacts can you already show today (repos, systems, papers)?

## Frontier-Lab Interview Angle

The openers ("tell me about yourself", "why us") plus the *unstated* scoring rubric. Knowing the rubric lets you steer every deep-dive toward your strengths and name your gaps honestly — which itself signals research maturity.
