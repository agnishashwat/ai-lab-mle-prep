---
name: torch-question-generator
description: Generates original PyTorch tensor puzzles and compact ML coding interview questions. Use whenever the user says "Generate a torch question", asks for a Torch or PyTorch question, or requests tensor-manipulation practice.
metadata:
  version: "1.0"
---

# Torch Question Generator

Generate one self-contained PyTorch coding question at a time.

## Default behavior

When the user says `Generate a torch question` without more detail:

1. Produce exactly one new medium-difficulty question suitable for roughly 15-25 minutes of interview practice.
2. Present only the question. Do not include a solution, hints, pseudocode, implementation strategy, or leading API suggestions unless the user asks.
3. Require Python with PyTorch. Do not require NumPy, pandas, or third-party packages.
4. Prefer a vectorized tensor solution. State explicitly whether Python loops are forbidden, discouraged, or allowed.
5. Make the required behavior, tensor shapes, dtypes, device expectations, edge cases, and indexing convention unambiguous.
6. Include at least one concrete input/output example and a concise set of constraints.
7. Privately derive the expected output and check every example before presenting the question.
8. Avoid repeating a problem or core technique already used earlier in the current chat.

Honor modifiers in the request, including difficulty, topic, number of questions, time limit, desired APIs, and whether hints or solutions should be included.

## Question families

Rotate among these families rather than repeatedly rephrasing the same puzzle:

- Run, segment, and group labeling
- Boolean masks and advanced indexing
- Broadcasting and shape transformations
- Reductions, cumulative operations, and rolling windows
- `gather`, `scatter`, sorting, and top-k operations
- Ragged sequences, padding, packing, and batching
- Pairwise distances, similarity, and batched matrix operations
- Stable logits, softmax, sampling, and loss calculations
- Autograd, gradient flow, and in-place operation debugging
- Small PyTorch ML utilities for batching, sampling, scoring, or evaluation

Use the following as difficulty calibration, not as a question to copy: transforming a binary sequence so zeros become `-1` and each contiguous run of ones receives its zero-based run ID is a medium tensor puzzle.

## Required question format

Use this structure:

````markdown
## Torch Puzzle: <short title>

**Difficulty:** <Easy | Medium | Hard>

### Task
<precise problem statement>

### Function signature
```python
def function_name(...):
    ...
```

### Example
```python
<input and expected output>
```

### Constraints
- <shape, dtype, range, and device constraints>
- <edge-case behavior>

### Rules
- <loop and library restrictions>
- <expected output properties>
````

Keep the prompt concise and interview-ready. Do not add motivational filler.

## Follow-up behavior

- If the user asks for a hint, give one incremental hint without revealing the full solution.
- If the user asks for another hint, reveal the next useful idea only.
- If the user submits code, review correctness first, then vectorization, shape safety, dtype/device behavior, complexity, and readability.
- If the user asks for the solution, provide idiomatic PyTorch code, a brief explanation, time and space complexity, and focused edge-case tests.
- If multiple valid approaches exist, give the clearest idiomatic solution first and mention an alternative only when it teaches a meaningfully different technique.
