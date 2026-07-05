# Deliverable - Week 01 Day 05: Evaluation Basics & Ablation Discipline

## What I Built / Derived

`code/eval_ablation.py` (+ `test_eval_ablation.py`, 6/6 pytest, 5/5 self-check) — an
evaluation + ablation layer on top of the Day-4 loop:

- **`evaluate_loss`** — mean CE over N val batches with full eval hygiene (`eval()` + `no_grad()`, mode restore).
- **`run_with_val`** — the Day-4 loop logging BOTH train and val curves, with a `train_pool`
  (fixed finite batch set) + `label_noise` switch so overfitting is *observable*.
- **`overfit_one_batch`** — the canonical pipeline sanity test (one fixed batch → loss → ~0).
- **`ablation`** — sweep one variable × seeds, report mean / min / max of `final_val`.

## Core Takeaway

Eval taste, not loop mechanics, is the differentiator. A number is only trustworthy if (a) the
eval pipeline is verified correct, (b) the split doesn't leak, and (c) the effect exceeds the
seed-to-seed noise. The overfit-one-batch test is the cheapest proof that data→loss→optimizer is
wired correctly; ablations with ≥3 seeds and a fixed compute budget are how you tell signal from noise.

## Key Results

**LR ablation** (3 seeds, grad clipping OFF so the too-high arm can diverge):

| base_lr | mean final val | range | verdict |
|---|---|---|---|
| 3e-5 | 2.081 | [1.93, 2.22] | too low → underfit |
| 3e-3 | 0.063 | [0.06, 0.07] | best |
| 3e2  | 2.924 | [2.84, 3.08] | too high → diverges |

**Overfit gap** (fixed pool of 2 batches, 400 steps):

| pool labels | final train | final val | reading |
|---|---|---|---|
| true successor (noiseless) | 0.013 | 0.013 | no gap — rule generalizes |
| random (label noise) | 2.657 | 3.120 | classic overfit gap |

**Overfit-one-batch:** CE 2.816 → 0.0016 in 200 steps.

## Key Equations

- Gradient-estimate variance vs batch: `Var(g_B) = σ²/B` → LR scaling: linear `η∝B` (Goyal),
  sqrt `η∝√B` (holds update-noise scale constant).
- Uniform-prior loss floor = `ln V` (flat train+val here = learned nothing).

## Implementation Notes

- `F.cross_entropy` K-dim form: input `(N, C, d₁…)`, target `(N, d₁…)`. Passing `(B,T,V)` vs
  `(B,T)` does NOT crash when `T==V` — it silently treats the time axis as the class axis and
  returns a garbage loss. Always `reshape(-1, V)` / `reshape(-1)`. Square shapes hide shape bugs.
- Divide the eval sum by the `iters` argument you were passed, not by `cfg.eval_iters`.
- Dynamic-key config edit: `dataclasses.replace(base, **{var: val}, seed=s)` — you cannot write
  `replace(base, var=val)` because `var` is passed literally, not evaluated.

## Failure Modes / Debugging Notes

- **A noiseless, globally-consistent task cannot overfit** — memorization == generalization.
  To *see* an overfit gap you must inject label noise into the finite pool.
- **Grad clipping makes Adam nearly unkillable** on easy tasks: LR up to 30 still converged with
  `grad_clip=1.0`; divergence only appeared at LR≈300 with clipping off (NaN by LR≈1000).
- A broken eval harness mimics catastrophic overfitting (train 0.05 / val 5.5). **Verify eval
  before diagnosing overfitting.**

## Interview Answer

Prompt:

> "How do you know an improvement is real and not noise? And if train loss drops but val is flat?"

Answer:

First, is the measurement even real — verify the eval pipeline (same preprocessing, correct
shapes/labels, no leakage between train and val). Then treat "real improvement" statistically:
change exactly one variable, hold the compute budget fixed, run ≥3 seeds per cell, and report
mean ± std. I only claim an effect if it exceeds the seed-to-seed spread (non-overlapping
intervals) — a single-seed win smaller than run-to-run noise is not a result. For train↓/val-flat
specifically: after ruling out an eval bug or train/val mismatch, it's overfitting — reach for
early stopping first, then regularization (weight decay / dropout), then more/augmented data, and
shrink the model only as a last resort.

## Reusable Artifact

- Path: `code/eval_ablation.py`, `code/test_eval_ablation.py`
- How to reuse: `run_with_val(cfg)` for dual-curve training; `overfit_one_batch(cfg)` as a
  first-line pipeline smoke test on any new model/data; `ablation(base, var, values, seeds)` for
  seed-averaged one-variable sweeps.
