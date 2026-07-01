# AI Lab MLE Prep

A self-directed lab notebook for converting fragmented post-training / RL / evals
experience into frontier-lab (OpenAI / Anthropic-style) MLE & Research Engineer
crispness.

This is **not** a notes folder. It is an artifact factory: every 2-hour session must
produce something reusable - a derivation, a piece of code, an experiment, an eval
report, or an interview answer.

## Organizing principle

> Session -> Deliverable -> Interview artifact

For every 2-hour block, answer:

1. What did I derive?
2. What did I implement?
3. What failure mode did I understand?
4. What would I say in an interview?
5. What artifact can I show or reuse?

## Repo map

- `curriculum.md` - the canonical 10-week plan (source of truth, copied from the skill file).
- `progress.md` - master tracker: current week/day, status table, review queue.
- `HANDOFF.md` - living cross-chat state (a new chat is used per day); read it first, update it last.
- `interview-bank/` - polished, reusable Q&A and project stories.
- `shared/` - templates, references, and utilities (`new_day.py`).
- `week-01-...` ... `week-10-...` - one folder per week.

### Inside each week

- `README.md` - the weekly plan (goals, sessions table, deliverable, readings).
- `week-summary.md` - condensed explanation written after finishing the week.
- `day-0X-.../` - one 2-hour session each.
- `code/`, `notes/`, `artifacts/` - reusable code, exploratory notes, polished outputs.
- `retrospective.md` - what clicked, what was confusing, what to review.

### Inside each day (minimal default)

- `session.md` - the 2-hour session plan.
- `work.ipynb` - scratch / implementation notebook.
- `deliverable.md` - the most important file: forces a reusable output.

Add `derivation.md`, `notes.md`, or `checkpoint-answers.md` only when needed.

## Workflow

1. Ask the assistant: **"Generate Week N sessions."** -> flesh out that week's `README.md`.
2. Each day: **"Start Week N Day M."** -> work the session interactively.
3. Commit the day's output to its folder.
4. End of week: write `week-summary.md`, update `interview-bank/`, and `retrospective.md`.
5. Keep `progress.md` current.

## Scaffolding a new day

```bash
python shared/utils/new_day.py week-04-rlhf-ppo day-06-extra-mock "Extra Mock Interview"
```

## Session structure (2 hours)

| Time | Block |
|---|---|
| 0:00-0:10 | Recall quiz from prior material |
| 0:10-0:35 | First-principles concept explanation |
| 0:35-1:00 | Math derivation / paper deconstruction |
| 1:00-1:45 | Coding / implementation / experiment |
| 1:45-1:55 | Interview-style explanation drill |
| 1:55-2:00 | Notes, deliverable, next-session setup |
