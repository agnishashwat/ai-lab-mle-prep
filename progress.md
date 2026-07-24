# AI Lab MLE Prep Progress

## Current Status

- Current week: **Week 03 IN PROGRESS** — RL foundations for post-training
- Current day: Week 3 Day 01 done (MDPs, returns, value functions, Bellman ops; `code/mdp.py` 7/7 invariants)
- Current focus: RL substrate for PPO/GRPO/DPO — value functions, advantage, Bellman fixed points

### Diagnostic (Day 01)

- Top gaps: (1) ML systems / memory-cost reasoning, (2) interview-crisp RL derivations.
- Fluency-probe baseline: 5/5 in ~20 min (slow on `gather` + masking); target <10 min.

## Weekly Progress

| Week | Theme | Status | Deliverable |
|---|---|---|---|
| 01 | LM foundations + PyTorch | Complete (5/5) | Tiny LM train/eval/ablation harness |
| 02 | Transformers + generation | Complete (5/5) | Decoder-only mini-impl + debug suite |
| 03 | RL foundations | In progress (D1/5) | Toy policy-gradient implementation |
| 04 | RLHF + PPO | Not started | PPO/RLHF explainer notebook |
| 05 | DPO + preference optimization | Not started | DPO trainer |
| 06 | GRPO + RLVR | Not started | Mini RLVR repo |
| 07 | Evals + reward modeling | Not started | Eval harness |
| 08 | Training/inference systems | Not started | Systems design doc |
| 09 | Capstone | Not started | Polished repo + writeup |
| 10 | Interview conversion | Not started | Interview packet |

## Review Queue

Topics to revisit:

- [x] MDP tuple + Markov property; Bellman expectation (E_π) vs optimality (max_a); γ-contraction ⇒ unique fixed point (W3 D1)
- [x] Value functions V^π/Q^π, advantage A^π=Q−V, and E_{a∼π}[A]=0 per state = why V is the natural baseline (W3 D1)
- [ ] EGLP lemma E_{a∼π}[∇log π]=0 ⇒ state-only baseline is unbiased (Day-1 drill gap → Day 3)
- [ ] Policy gradient theorem (log-derivative trick; why P & ρ₀ drop out; reward-to-go causality)
- [ ] PPO ratio / clipping
- [ ] DPO derivation
- [ ] GRPO group baseline
- [x] KV-cache memory math (`2·L·n_kv·d_head·T·B·bytes`; 512 KiB/tok, 2 GiB@4k, 64 GiB@8k×16; done W2 D4)
- [ ] Arithmetic intensity / roofline: decode ≈1 op/byte → memory-bound; batch to amortize weights (add BW-tied levers: quant, spec-decode)
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
- [ ] Attention-NaN checklist COLD (fastest-first): masked-row 0/0 -> fp16 overflow -> eps -> grads -> data (D5 soft spot)
- [ ] Four debugging contracts: causal / shape / numeric / train-inference; two crown-jewel invariant tests
- [ ] Teacher-forcing vs free-running; overfit-one-batch-then-greedy splits plumbing vs generation bugs
- [ ] Test-design: an invariant test is only as sensitive as the input (random model, not position-wise task)

## Interview Bank Additions

Add polished answers every week (see `interview-bank/`).
