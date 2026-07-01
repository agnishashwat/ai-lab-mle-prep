# Week 05 - Preference Optimization: DPO & Variants

## Theme

Make DPO mathematically obvious and know when it is the wrong tool.

## Weekly Deliverable

DPO from-scratch trainer on a small model/dataset, with beta and data-quality ablations.

## Sessions

| Day | Topic | Deliverable |
|---|---|---|
| Day 1 | Bradley-Terry & Reward Modeling | BT / reward-model derivation notes |
| Day 2 | Deriving DPO | Full DPO derivation |
| Day 3 | DPO Implementation Details | DPO loss implementation |
| Day 4 | DPO Variants (IPO/KTO/ORPO/SimPO) | Variant comparison table |
| Day 5 | DPO Failure Modes | DPO trainer + beta/data-quality ablations |

## Must-Know Interview Outcomes

By the end of this week, I should be able to:

1. Explain the Bradley-Terry preference model and reward modeling.
2. Derive DPO from KL-regularized RLHF.
3. Explain chosen/rejected logprobs, the reference model, beta, and length effects.
4. Compare DPO to IPO/KTO/ORPO/SimPO and know when each fits.
5. Diagnose DPO failure modes (noisy prefs, distribution shift, overoptimization).

## Core Readings

- DPO (Rafailov et al.)
- Optional: IPO / KTO / ORPO / SimPO papers (skim)

## Folder Map

- `day-0X-.../` - one 2-hour session each (`session.md`, `work.ipynb`, `deliverable.md`)
- `code/` - reusable code for the week
- `notes/` - exploratory notes, paper notes, scratch derivations
- `artifacts/` - polished outputs (diagrams, notebooks, reports)
- `week-summary.md` - condensed explanation written after finishing the week
- `retrospective.md` - what clicked, what was confusing, what to review
