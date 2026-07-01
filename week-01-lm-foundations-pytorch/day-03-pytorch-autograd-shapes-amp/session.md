# Week 01 Day 03 — PyTorch Autograd, Shapes & Mixed Precision

## Objective

Build reflexive fluency with autograd, tensor shapes / broadcasting / masking, and the fp16/bf16 failure modes that cause NaNs — the skills tested in live-debugging rounds.

## Prerequisites

- Day 2 (cross-entropy)
- Basic torch tensor ops

## Required Reading

- PyTorch autograd mechanics (skim: leaf tensors, `requires_grad`, `no_grad`, `detach`)
- Optional: `torch.autocast` / AMP docs (skim)

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 2)

1. `∂CE/∂z = ?`
2. Perplexity of a uniform model over `V`?
3. Why subtract the max in log-softmax?

### 0:10-0:35 - Concept Explanation (First Principles)

- Autograd = reverse-mode AD over a dynamic graph; `.backward()`, `.grad`, leaf vs non-leaf, graph freeing, `retain_graph`.
- `no_grad()` vs `detach()` — different tools for different jobs.
- Broadcasting rules; `view` vs `reshape` vs `contiguous`; strides and why non-contiguous tensors break `view`.
- Precision: fp32 vs fp16 (narrow dynamic range) vs bf16 (wide range, low mantissa); why fp16 under/overflows; loss scaling; why bf16 is the default for training.

### 0:35-1:00 - Derivation

- Hand-trace backprop through `x -> Linear -> ReLU -> Linear -> CE` for a 2-neuron example; compute gradients by hand.
- Show *why* fp16 needs loss scaling: gradients below ~6e-5 underflow to 0 in fp16; multiplying the loss by `S` shifts them into representable range, then you unscale before the optimizer step.

### 1:00-1:45 - Coding Task

In `code/`:

- Shape/broadcasting drills: a function seeded with deliberate shape traps; make the assertions pass.
- Manual backward for `Linear + CE`, checked against autograd.
- Reproduce an fp16 overflow -> NaN, then fix it with bf16 `autocast` (and/or `GradScaler`).
- Masked-mean drill: correct denominator over padded tokens (show the wrong vs right version).

### 1:45-1:55 - Interview Drill

"Walk me through what happens on `loss.backward()`," and "You switched to fp16 and now get NaNs around step 50 — debug it live."

### 1:55-2:00 - Deliverable & Next Session

`deliverable.md`: the drills + a short NaN repro/fix writeup. Next: Day 4 (minimal training loop).

## Checkpoint Questions

1. `reshape` vs `view` — when does `view` raise, and why?
2. `detach()` vs `torch.no_grad()` — what is the difference?
3. Why does fp16 need loss scaling but bf16 usually does not?
4. How do you compute a correct masked mean over padding?
5. What is a leaf tensor, and when is `.grad` populated?

## Frontier-Lab Interview Angle

"Debug a run producing NaNs" and silent shape/masking bugs are the most common live-coding failure points. Interviewers watch your *debugging process* — hypotheses, isolation, checks — not just the final fix.
