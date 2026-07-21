# AI Lab MLE Prep Progress

## Current Status

- Current week: Week 02 in progress (Days 01-03 complete)
- Current day: Week 2 Day 03 done -> Day 04 next
- Current focus: transformers + generation (attention + pos-enc + TransformerBlock/TinyGPT + decoding done)

### Diagnostic (Day 01)

- Top gaps: (1) ML systems / memory-cost reasoning, (2) interview-crisp RL derivations.
- Fluency-probe baseline: 5/5 in ~20 min (slow on `gather` + masking); target <10 min.

## Weekly Progress

| Week | Theme | Status | Deliverable |
|---|---|---|---|
| 01 | LM foundations + PyTorch | Complete (5/5) | Tiny LM train/eval/ablation harness |
| 02 | Transformers + generation | In progress (D3/5) | Decoder-only mini-implementation |
| 03 | RL foundations | Not started | Toy policy-gradient implementation |
| 04 | RLHF + PPO | Not started | PPO/RLHF explainer notebook |
| 05 | DPO + preference optimization | Not started | DPO trainer |
| 06 | GRPO + RLVR | Not started | Mini RLVR repo |
| 07 | Evals + reward modeling | Not started | Eval harness |
| 08 | Training/inference systems | Not started | Systems design doc |
| 09 | Capstone | Not started | Polished repo + writeup |
| 10 | Interview conversion | Not started | Interview packet |

## Review Queue

Topics to revisit:

- [ ] Policy gradient theorem
- [ ] PPO ratio / clipping
- [ ] DPO derivation
- [ ] GRPO group baseline
- [ ] KV-cache memory math
- [ ] Mixed-precision memory (16 bytes/param; savings are activations + throughput)
- [ ] fp16-NaN debugging narrative (temporal clue → hypotheses → matched fix)
- [ ] Bootstrap confidence intervals
- [ ] Training-loop eval hygiene (`no_grad` = memory + grad-leak; `eval()` = dropout/BN)
- [ ] Masked-mean static form `(x*mask).sum(-1)/mask.sum(-1)` (soft spot, seen 3x)
- [ ] LR<->batch scaling rules (linear vs sqrt, critical batch size)
- [ ] Loop invariants cold: backward INSIDE grad-accum (memory, not math) - regressed on D5 quiz
- [ ] `cross_entropy` K-dim shape bug is silent when T==V (always reshape to (-1,V)/(-1))
- [ ] "Verify eval pipeline before diagnosing overfitting" reflex
- [ ] `1/√d_k`: divide by `d_k` over-shrinks (var ∝ 1/d_k → uniform); √d_k normalizes the STD
- [ ] Softmax Jacobian `diag(s)−ssᵀ` → 0 at one-hot (saturation = vanishing gradient)
- [ ] Causal mask after softmax → rows <1; parallel-training point; RoPE for long context (not learned)
- [ ] RoPE freq ladder + NTK/YaRN base-rescaling for context extension
- [ ] Pre-LN vs post-LN: gradient-highway `+1` vs compounding LN-Jacobian product (warmup sensitivity)
- [ ] NaN-at-step-0 = config bug (rules out LR); checklist: mask→fp16 overflow→init/eps→labels
- [ ] fp16 OVERFLOW→Inf→NaN (not bf16 underflow, which is silent/non-NaN)
- [ ] Which ops stay fp32 under bf16: LN/RMS stats, softmax, logits→CE (reduce-then-divide)
- [ ] Multi-head = independent representation subspaces (not just parallel compute)
- [ ] Top-k vs top-p: top-p adaptive (nucleus = smallest set, cumprob >= p); top-k fixed budget fails both ways
- [ ] Neural text degeneration = repetition attractor (self-attention + likelihood); fix = temp + nucleus
- [ ] Beam length-norm `1/L^α`: raw `Σ log p_t` biased short; beam right only for low-entropy tasks
- [ ] Filters return masked LOGITS (compose, one final softmax); top-p scatter-back-to-vocab-order bug
- [ ] Batched beam couples the k hypotheses (per-row top-k over k×V + beam reindex); sampling batches free

## Interview Bank Additions

Add polished answers every week (see `interview-bank/`).
