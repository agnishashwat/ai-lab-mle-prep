"""
TEMPLATE -- the standard shape for a daily coding exercise.

Copy this to `week-NN-<theme>/code/<topic>.py`, replace the TODO functions with the
day's content, and replace the stages with one stage per TODO.

Why this shape: the harness is STAGED, so every function is testable the moment it
is written. A stage whose function is still a TODO reports [TODO] and is skipped
instead of killing the run -- implement one function, rerun, watch the PASS count
climb. Reference implementation: week-03-rl-foundations/code/actor_critic.py.

  run everything      ./.venv/bin/python ai-lab-mle-prep/<path>.py
  one stage           ...  <path>.py <stage>
  a few stages        ...  <path>.py <stage> <stage>
  small/fast fixtures ...  <path>.py --fast
  list the stages     ...  <path>.py --list
"""

from __future__ import annotations

import sys
import traceback
from functools import cached_property

import numpy as np
import torch

# Sibling modules import plainly -- Python puts the script's directory on sys.path,
# so a day's file can build on the previous day's without package plumbing.
# from previous_day import something


# ---------------------------------------------------------------------------
# Exercises (TODO bodies -- the learner implements these)
# ---------------------------------------------------------------------------

def example_op(x: np.ndarray) -> np.ndarray:
    """One-paragraph statement of WHAT and WHY, then the shape contract.

    Say which idiom to use and which trap to avoid; do not give the solution.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Self-check harness (staged -- see the module docstring for usage)
# ---------------------------------------------------------------------------

_TALLY = {"pass": 0, "fail": 0}


def _check(name: str, cond: bool, detail: str = "") -> None:
    cond = bool(cond)
    _TALLY["pass" if cond else "fail"] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(name)


class Ctx:
    """Shared, lazily-built fixtures.

    Everything expensive is a cached_property with its OWN seeded generator, so a
    stage costs nothing unless it is actually run and the data does not depend on
    which other stages ran first.
    """

    def __init__(self, fast: bool = False):
        torch.manual_seed(0)
        self.fast = fast
        self.n_big = 1_000 if fast else 20_000

    @cached_property
    def big(self):
        print(f"  ... building the {self.n_big}-element fixture")
        rng = np.random.default_rng(0)
        return rng.standard_normal(self.n_big)


def _fail_fast(ctx: Ctx) -> None:
    """Touch every function a heavy stage needs, on a 1-element input, so a missing
    TODO surfaces BEFORE paying for the expensive fixture."""
    example_op(np.zeros(1))


# ---- individual stages ----------------------------------------------------
# One stage per TODO. The docstring is the description shown by --list.
# Each stage should assert: a hand-computable case, a shape/dtype contract, an
# analytic limit, and a structural invariant -- not just "it returned something".

def stage_example(ctx: Ctx) -> None:
    """example_op"""
    out = np.asarray(example_op(np.array([1.0, 2.0, 3.0])), dtype=np.float64)
    _check("returns the input shape", out.shape == (3,), f"got {out.shape}")
    _check("matches the hand computation", np.allclose(out, [1.0, 2.0, 3.0]),
           f"got {np.round(out, 4)}")
    _check("is floating point and not all zeros",
           np.issubdtype(out.dtype, np.floating) and not np.allclose(out, 0.0),
           f"dtype {out.dtype}")


def stage_heavy(ctx: Ctx) -> None:
    """example_op on the big fixture (statistical check)"""
    _fail_fast(ctx)
    out = example_op(ctx.big)
    tol = 0.20 if ctx.fast else 0.05
    _check("converges to the analytic value", abs(float(np.mean(out))) < tol,
           f"mean {np.mean(out):.4f}")


STAGES = [
    ("example", stage_example),
    ("heavy", stage_heavy),
]


# ---- runner (copy verbatim; nothing below needs editing) -------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, fn in STAGES:
            print(f"  {name:<12} {fn.__doc__}")
        return 0
    fast = "--fast" in argv
    names = [a for a in argv if not a.startswith("-")]

    known = {n for n, _ in STAGES}
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}\nknown: {', '.join(sorted(known))}")
        return 2

    ctx = Ctx(fast=fast)
    selected = [(n, f) for n, f in STAGES if not names or n in names]
    todo, broke = [], []

    for name, fn in selected:
        print(f"\n=== {name} ===")
        try:
            fn(ctx)
        except NotImplementedError:
            print("  [TODO] not implemented yet -- skipped")
            todo.append(name)
        except AssertionError:
            broke.append(name)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            broke.append(name)

    print(f"\n{'=' * 60}")
    print(f"checks: {_TALLY['pass']} passed, {_TALLY['fail']} failed"
          + (f" | stages TODO: {', '.join(todo)}" if todo else "")
          + (f" | stages failing: {', '.join(sorted(set(broke)))}" if broke else ""))
    if not todo and not broke:
        print("All invariants passed.")
    return 1 if broke else 0


if __name__ == "__main__":
    raise SystemExit(main())
