# shared/utils

Small utilities for the lab.

## new_day.py

Scaffold a new day folder from the shared templates.

```bash
python shared/utils/new_day.py <week-folder> <day-slug> "Human Title"

# example
python shared/utils/new_day.py week-04-rlhf-ppo day-06-extra-mock "Extra Mock Interview"
```

Creates `session.md`, `deliverable.md`, and `work.ipynb` in the new day folder.
Refuses to overwrite an existing folder.
