#!/usr/bin/env python3
"""Scaffold a new day folder from the shared templates.

Usage:
    python shared/utils/new_day.py <week-folder> <day-slug> "Human Title"

Example:
    python shared/utils/new_day.py week-04-rlhf-ppo day-06-extra-mock "Extra Mock Interview"
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = REPO / "shared" / "templates"


def render(template_name: str, mapping: dict) -> str:
    text = (TEMPLATES / template_name).read_text()
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def notebook(title: str) -> str:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", "Scratch / implementation work for this session."],
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1) + "\n"


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)

    week_folder, day_slug, title = sys.argv[1], sys.argv[2], sys.argv[3]
    week_num = week_folder.split("-")[1]
    day_num = day_slug.split("-")[1]

    day_dir = REPO / week_folder / day_slug
    if day_dir.exists():
        print(f"Refusing to overwrite existing folder: {day_dir}")
        raise SystemExit(1)
    day_dir.mkdir(parents=True)

    mapping = {"WEEK": week_num, "DAY": day_num, "TITLE": title, "OBJECTIVE": "<fill in>"}
    (day_dir / "session.md").write_text(render("session-template.md", mapping))
    (day_dir / "deliverable.md").write_text(render("deliverable-template.md", mapping))
    (day_dir / "work.ipynb").write_text(notebook(f"Week {week_num} Day {day_num} - {title}"))

    print(f"Created {day_dir}")


if __name__ == "__main__":
    main()
