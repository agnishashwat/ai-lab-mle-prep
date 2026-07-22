# Handoff — ai-lab-mle-prep

Living state doc for cross-chat continuity. The user starts a **new chat per day**; this
file (plus `progress.md` and each day's `session.md`) carries context forward. The
`ai-lab-mle-prep-day` skill reads this at the start of a session and updates it at the end.

## Current position

- **Week 01 — COMPLETE** (Days 01-05 done). LM Foundations & PyTorch Fluency.
- **Week 02 — IN PROGRESS.** Days 01-04 done (attention+pos-enc; TransformerBlock+TinyGPT; decoding; KV cache).
- **Next up: Week 02 Day 05 — Week-2 debug lab** (capstone: diagnose bad generations, NaNs, shape
  and masking bugs across the week's TinyGPT + decode + KV-cache stack).

## Next session

Start with the prompt: **"Start Week 2 Day 5"**. Session plan at
`week-02-transformers-generation/day-05-*/session.md` (create it if absent from `curriculum.md`).

Day 5 arc: the week's DEBUG LAB — take the TinyGPT / decoding / KV-cache code and inject/diagnose
the classic failure modes (NaN-at-step-0 vs step-N, causal-mask leakage, `T==V` shape bugs,
double-softmax, KV-cache `k` vs `k_all` / prefill-decode feeding bug), each with the matched fix.
Consolidates Week 2 for interview readiness.

Quiz Week-2 D4 cold at the start: KV-cache formula `2·L·n_kv·d_head·T·B·bytes` (7B fp16 =
512 KiB/tok → 2 GiB@4k,B1 → 64 GiB@8k,B16, dwarfs 14 GB weights; GQA n_kv=8 → 4× smaller); prefill
(compute-bound, parallel) vs decode (memory-bandwidth bound, ≈1 op/byte); why K/V cached but not Q;
why batching helps (amortize weight-streaming → intensity ↑ ~B×); BW-tied levers = batch /
continuous-batch / quantize / speculative-decode; GQA/MQA + PagedAttention. Also re-poll the D3
degeneration + top-k/top-p framing (still a soft spot: read the FULL question, name the alternative).

## Completed

- **W1 D1 — Role Map + Baseline Diagnostic.** Built role map + competency matrix; stood up
  the lab env (`requirements.txt`, `smoke_test.py`); completed `fluency_probe.py` 5/5
  (softmax, cross_entropy, masked_mean, sgd_step, + 3-D `seq_token_logprobs` stretch).
  Baseline: ~20 min, slow on `gather` + masking. Deliverable written; the derivation of
  `∂CE/∂z` was covered as a preview.
- **W1 D2 — Token-Level Language Modeling.** Built `code/token_lm.py` (log_softmax via
  logsumexp, cross_entropy, perplexity, analytic `ce_grad`) + `test_token_lm.py` (6/6 pass,
  incl. autograd AND finite-difference gradient checks). Derived `∂CE/∂z = p − e_y` cleanly on
  the whiteboard (fixed an `e^{z_k}` index slip and a gather-vs-scatter bug live). Quiz 3.5/4,
  drill 0.9/1.
- **W1 D3 — Autograd / Shapes / Mixed Precision.** Built `code/autograd_shapes_amp.py` +
  test (5/5 self-check, 6/6 pytest): masked-mean drills, **manual backward** through
  `Linear→ReLU→Linear→CE` (matched autograd first try), fp16 overflow repro + bf16-autocast fix.
  Derived bf16 memory = **16 bytes/param = 112 GB for 7B (same as fp32)**; savings are activations +
  throughput. Quiz 2.5/3, checkpoints 1.5/3, interview drills ~0.65. Live-caught 3 latent code bugs
  (in-place input mutation, CPU-hardcoded zero guard, and a `-inf`-then-sum masking bug).
- **W1 D4 — Minimal Training Loop.** Built `code/train.py` + `test_train.py` (5/5 self-check,
  5/5 pytest): `lr_at_step` (warmup→cosine), `evaluate` (no_grad + mode save/restore), and the
  full `train` loop (seed-first, per-step LR, grad-accum, clip, AdamW, periodic eval, histories).
  Toy successor task drives CE 2.82→0.022; two runs bit-identical. Quiz D3 ≈4.1/5 (masked-mean
  static form still the soft spot). Loop coding was the hard part — needed heavy scaffolding and
  hit every classic bug live: `AdamW(model)` not `.parameters()`, flat warmup (missing `*base_lr`),
  inverted eval `%`, backward-inside-accum on a summed tensor, missing grad-accum `/N`, and the
  **double-softmax** (feeding `softmax` to `F.cross_entropy`). Drill ≈3.6/5 — under-answered the
  eval-hygiene split. This is his weakest area so far; wants more loop reps.

- **W1 D5 — Eval + Ablation Discipline.** Built `code/eval_ablation.py` + test (5/5 self-check,
  6/6 pytest): `evaluate_loss` (hygiene), `run_with_val` (dual train/val curves + `train_pool`/
  `label_noise` to make overfit observable), `overfit_one_batch` (2.82→0.0016), `ablation`
  (seed-averaged one-var sweep, LR U-shape: 3e-5→2.08, 3e-3→0.06, 3e2→2.92 w/ clip off). All four
  of his impls were correct first-try on logic; the 3 failures were the scaffold's flawed task
  assumptions, which became the day's best lessons: (1) `cross_entropy` K-dim shape bug is SILENT
  when `T==V` (treats time axis as class axis); (2) a noiseless consistent rule CAN'T overfit
  (memorize==generalize) → need label noise; (3) grad clipping tames Adam so hard LR-divergence
  only shows at LR≈300 with clip off. Quiz D4 4.2/5 (regressed: said backward AFTER accum loop —
  must be INSIDE, reason=memory; missed clip in the invariant list). Drills D1 0.85 / D2 0.7
  (led with capacity/data instead of "verify eval first"). Also wrote week-summary + retrospective.

- **W2 D1 — Attention from First Principles.** Already had a correct attention core in
  `anthropic_attention/` (SDPA + causal mask + MHA, einsum/matmul parity), so the day pushed on
  derivations + the missing piece. Built `week-02/code/positional_encoding.py` (6/6 self-checks):
  sinusoidal (strided-slice interleave) + RoPE (`rope_cos_sin`/`_rotate_half`/`apply_rope`), with a
  harness proving `<rope(q,m),rope(k,n)>` is constant along constant-offset diagonals (relative-pos
  invariance, 1.4e-6). Derivations excellent: `Var(q·k)=d_k σ⁴`, softmax Jacobian `diag(s)−ssᵀ`→0 at
  one-hot. Quiz (W1) 3.85/4, derivations ~2.8/3, drill ~0.8. Gaps caught: (1) why `√d_k` not `d_k`
  (dividing by `d_k` over-shrinks var∝1/d_k→uniform — normalize the STD); (2) drill said sinusoidal
  is the hard-capped one — it's LEARNED that's hard-capped, sinusoidal is closed-form/any-length;
  (3) long-context ⇒ RoPE, learned is the WORST pick there. Add the "causal mask enables parallel
  training" systems point to the mask answer.

- **W2 D2 — Transformer Block & Numerical Stability.** Built `week-02/code/transformer_block.py` +
  `test_transformer_block.py` (7/7 self-check, 6/6 pytest): `RMSNorm` (fp32 reduction), `MLP`
  (`d→4d→d` GELU), pre-LN `TransformerBlock`, `TinyGPT` (tok+learned-pos embed, N blocks, final
  norm, LM head) with the GPT-2 `1/√(2L)` output-proj rescale wired in. Attention was provided (his
  Day-1 core). Marquee causal test passes bit-exact (`max|Δ|_past = 0.00e+00`); overfit-one-batch
  CE → 0.0010. Derivations excellent (gradient-highway `+1`, LN Jacobian removing mean+scale dirs,
  matmul-vs-normalization precision). Quiz (W2 D1) 3.4/4, drills: pre/post-LN 4.6/5, NaN-at-step-0
  1.8→4.2 after redo. Gaps caught: (1) NaN-at-step-0 he first answered with the *runtime* "find
  where NaN appears" technique — that's the step-N case; step-0 = config bug, rules out LR — needed
  the full checklist; (2) said bf16 *underflow* causes step-0 NaN — it's fp16 *overflow*→Inf; bf16
  underflow is silent/non-NaN; (3) multi-head "buys parallelism" → real answer is independent
  representation subspaces. Latent bug live-caught: `TransformerBlock(d, n_heads, n_heads)` passed
  `n_heads` for `mult`, silent because both were 4 (same class as W1 `T==V` CE bug).

- **W2 D3 — Decoding Strategies.** Built `week-02/code/decoding.py` + `test_decoding.py` (13/13
  self-check, 9/9 pytest): `apply_temperature`, `top_k_filter` (kth-value threshold), `top_p_filter`
  (sort→cumsum→shift-right→scatter-back), `sample` (seeded multinomial), `generate` (eval+no_grad,
  crop-to-cfg.T, full-seq accumulator), `beam_search` (expand k×V → prune, length-normalized, B=1).
  Evidence: temp entropy `0→1.744→2.079(=ln V)`; top-p peaked=1/flat=5 (adaptive); greedy & beam
  both recover successor `[6..10]`. Derivations excellent (temp log-odds monotone reparam, beam
  length-bias, top-p peaked-vs-flat walk, degeneration attractor). Coding was heavily
  question-driven (many "why this line?" — good sign, but needed lots of idiom-level help): the
  whole session was Q&A on advanced-indexing-vs-scatter, the `None`/`-1:` broadcast axis,
  `>=` vs `>` at the nucleus boundary (he CAUGHT that `>` keeps an extra token — I'd mis-stated it),
  batched-beam coupling, `new_tensor`, `*c` unpack. Live bugs caught: `cat` defaulting to `dim=0`
  (missing `dim=-1`); bare `cfg.T` instead of `model.cfg.T`; returning the cropped context instead
  of a full accumulator; missing eval/no_grad. Drills ~0.7/0.7 — both had the core idea but missed
  the second half (top-k's symmetric failure modes / prod combo; and "what to use instead" + the
  low-entropy exception). Gap to reinforce: read the FULL question (name the alternative + the
  exception), and don't invert the top-k-vs-top-p "when" framing.

- **W2 D4 — KV Cache, Batching & Inference Throughput.** Built `week-02/code/kv_cache.py` +
  `test_kv_cache.py` (13/13 self-check, 6/6 pytest): cache-aware attention (`sdpa_cached` with an
  **offset causal mask** via `tril(diagonal=pos_offset)`), block/model cached forwards (positions =
  `pos_offset+arange(S)`), `generate_cached` (prefill@0 then 1-token decode@len-1), and
  `kv_cache_bytes`. Correctness anchor: cached greedy **token-identical** to uncached Day-3 generate;
  prefill matches full forward at **0.00e+00**; incremental matches each step; **4.15× decode speedup**
  at N=200. Derivation regressed then recovered: first pass had `d_head=128/32=4` (double-divide) +
  dropped the `n_heads` multiply + a `2^20 bytes = 1024 GB` unit slip → landed the formula cold on the
  drill (64 GiB @ 8k×16, dwarfs 14 GB weights; GQA 4×). Drill 1 **1.0** (clean number + "KV dwarfs
  weights" insight), Drill 2 **0.75** (had the symptom — GPU idles on weight loads — but under-stated
  the *mechanism*: arithmetic intensity ≈1 op/byte, batching amortizes weight-streaming; PagedAttention
  is a capacity fix not a BW lever — wants quant/speculative-decode as the direct BW levers). Two live
  code bugs I mis-passed first then caught via the harness: (1) `scores = q @ k` not `q @ k_all`
  (silent in prefill P=0, crashes first decode step); (2) `generate_cached` fed the whole running
  context + past cache into decode instead of just `next_tok` (defeats the cache). Quiz (W2 D1-D3)
  4.6/5.

## Learner profile (quick facts)

- Strong: PhD-level math/optimization, senior ML eng, real post-training experience
  (SFT/DPO/GRPO, retrieval, personalization). RL intuition is genuinely good.
- **Top gaps:** (1) ML systems / memory-cost reasoning; (2) making RL depth
  interview-crisp and derivable on demand.
- Prior work to reference in examples: Amazon personalized prompt-augmentation (RAG:
  "pet food" → "pet food for cats"), DPO to beat a Sonnet teacher via HF.
- Teaching style that works: derive from first principles, quiz from prior day, give code
  stubs + self-check harness (don't pre-solve), connect to his projects and later weeks.

## Conventions

- Run code: `./.venv/bin/python ai-lab-mle-prep/<path>` from `/Users/sshivam/1point3acres`
  (torch 2.12, Python 3.14, pytest installed).
- **Auto-commit + push is authorized** (`agnishashwat` identity; see
  `.cursor/rules/auto-commit.mdc`). Confirm only for destructive git ops or secrets.
- Each day ends by writing that day's `deliverable.md` and updating `progress.md` + this file.

## Running facts to reinforce (spaced repetition)

- LM loss: `−(1/BT) Σ log p_θ(y_{b,t} | x_{b,<t})`; one-hot label collapses the vocab sum.
- `∂CE/∂z = softmax(z) − onehot(y)` = "prediction − target"; sums to 0 (mass is moved).
- Stable softmax: subtract row-max, `keepdim=True` on both max and sum.
- Token log-prob idiom (works for `(N,V)` and `(B,T,V)`):
  `log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)`.
- Masked mean denominator is `mask.sum()`, not `numel()` (matters for variable-length seqs).
- Optimizer memory: params + grads + **2 Adam moments**; 7B fp32 ≈ 28+28+56 ≈ 112 GB
  before activations. (D1 gap — reproduced cold on D2. Adam `m`=1st moment, `v`=2nd moment.)
- `log_softmax(z) = z − logsumexp(z)`; NEVER `softmax(z).log()` (underflow → −inf).
- gather = *read* value at index (loss reads log p at target); scatter = *write* value at index
  (build the one-hot for the gradient). `scatter` scalar form uses `value=`, not `src=`.
- Mean-CE gradient carries a **1/N**: per-position grad is `p − e_y`, averaged over N positions.
- Perplexity = exp(CE) = effective branching factor. Floor 1, uniform baseline = V,
  **unbounded above** (PPL > V ⇒ worse than random). PPL 30 ⇔ CE ≈ 3.4 nats.
- CE vs KL: identical gradient (H(q) const in θ), but KL is the right *metric* for soft targets
  (distillation/label smoothing) — it reads 0 at a perfect match, CE bottoms out at H(q).
- Mixed-precision (bf16 Adam) = **16 bytes/param** (fp32 master 4 + fp32 m 4 + fp32 v 4 + bf16 W 2 +
  bf16 grad 2) = **112 GB for 7B, identical to fp32**. Savings = activations (~½) + ~2× matmul throughput,
  NOT optimizer/weight state. (D3 slip: forgot the 2nd bf16 grad copy → 98 vs 112.)
- fp16: exponent 5 bits (range ~6e-5..65504), training-killer = gradient **underflow** → loss scaling /
  `GradScaler`. bf16: exponent 8 bits (fp32 range), mantissa 7 → no loss scaling needed.
- Two numerics safeguards: max-subtraction ⇒ no `exp` overflow; `z−logsumexp(z)` ⇒ no log-of-zero underflow.
- Masking: `masked_fill(-inf)` **before** max/softmax (within reduction axis); multiply/boolean-index
  **after** (exclude positions from a mean). **Never `-inf` a value you then sum.**
- `detach` scopes by tensor (DPO reference logprobs); `no_grad` scopes by code region (inference,
  optimizer step). `no_grad` on the branch you optimize ⇒ silent no-learning.
- `.grad` populated only on **leaf** tensors; backward **accumulates** (`+=`) → need `zero_grad`; graph
  freed after `.backward()` unless `retain_graph=True`.
- `view` = zero-copy, raises on non-contiguous; `reshape` = copy-fallback. `x[bool_mask]` flattens to 1-D,
  data-dependent (bad for compile); `(x*mask).sum()/mask.sum()` is the static-shape form.
- Interview debug habit: **NaN at step 0 = config bug; NaN after stable steps = something grew** (check
  grad norms first, `set_detect_anomaly`, then matched fix).
- Training-loop five invariants + order: **seed→(model/opt)→ per step: set LR into param_groups →
  accumulate (`backward` INSIDE accum loop) → clip → step → zero_grad → eval under `eval()`+`no_grad()`.**
- `F.cross_entropy` eats **raw logits** (does log_softmax internally). Feeding `softmax(logits)` =
  double-softmax = SILENT underfit (no crash). Flatten `(-1,V)` + class-id targets, no unsqueeze.
- Grad-accum: divide each micro-loss by `grad_accum` (backward SUMS grads → "sum of means" must
  become "mean of means"). Effective batch = micro × accum × world_size.
- Eval hygiene = TWO bugs: `no_grad` off → activation graph kept (OOM) + `.grad` leaks into next
  train step; `eval()` off → dropout stays on (noisy metric) + BatchNorm uses/corrupts running stats.
- LR override lives in `opt.param_groups[i]['lr']` (optimizer reads it live each `step()`); the
  constructor `lr=` is inert in a hand-rolled loop but becomes `base_lr` under `lr_scheduler`.
- Reproducibility needs **data `torch.Generator`s seeded explicitly** (fresh Generator = system
  entropy); `set_seed`/`torch.manual_seed` only covers global RNG (model init).
- AdamW vs Adam: Adam folds L2 `λθ` into `g` so it passes through `1/√v̂` (uneven decay). AdamW
  applies `−ηλθ` directly after the adaptive step → uniform decay. Warmup guards the step-0 spike
  (Adam's `v̂` is garbage early); cosine anneals to settle. LR↔batch: `Var(g_B)=σ²/B` → linear rule
  `η∝B` (Goyal) / sqrt rule `η∝√B`, up to a critical batch size.
- `F.cross_entropy` K-dim form: input `(N,C,d₁…)`, target `(N,d₁…)`. Passing `(B,T,V)` vs `(B,T)`
  does NOT crash when `T==V` — silently treats time axis as classes → garbage loss. ALWAYS
  `reshape(-1,V)`/`reshape(-1)`. Square shapes hide shape bugs.
- Overfitting needs NON-generalizing signal: a noiseless, globally-consistent rule can't overfit
  (memorize == generalize). To see a train≪val gap you must inject label noise into a finite pool.
- Grad clipping (norm cap) makes Adam nearly unkillable on easy tasks: LR≤30 still converged with
  `grad_clip=1.0`; divergence only at LR≈300 clip-off, NaN by ~1000. Adam step is ~lr-bounded.
- Ablation: change ONE var, FIX the compute budget, ≥3 seeds, report mean ± spread; claim a result
  only if the effect beats seed noise (non-overlapping intervals). One seed can lie.
- Debug reflex: VERIFY THE EVAL PIPELINE before diagnosing overfitting — a broken eval mimics
  catastrophic overfit (train 0.05 / val 5.5 was a shape bug, not the model). Senior tell.
- `dataclasses.replace(base, **{var: val}, seed=s)` for a dynamic-key config edit; `replace(base,
  var=val)` fails (passes literal kwarg `var`).
- Attention = soft differentiable retrieval: `softmax(QKᵀ/√d_k + M)V`; softmax replaces a
  non-differentiable argmax over keys. Score is a dot product (→ scale), softmax-normalized
  (→ saturation), permutation-equivariant (→ needs positional encoding).
- `Var(q·k)=d_k σ⁴`; scale by `1/√d_k` to make logit STD `d_k`-independent. Dividing by `d_k`
  over-shrinks (var ∝ 1/d_k → uniform attention, can't discriminate). Big `d_k` unscaled →
  saturated softmax → near one-hot rows → zero Jacobian → vanishing gradient (a trainability bug).
- Softmax Jacobian `∂s_i/∂z_j = s_i(δ_ij − s_j)` = `diag(s) − ssᵀ`; at one-hot `s=e_k`,
  `diag(s)=ssᵀ` ⇒ Jacobian = 0.
- Causal mask = additive `−∞` BEFORE softmax (contributes 0 to denominator, rows stay normalized) +
  enables training all positions in parallel in one pass. After softmax → rows sum <1; re-normalizing
  just reimplements mask-before. RoPE applied to Q,K only, after head split, before scores.
- Positional encodings: sinusoidal = parameter-free, closed-form/any-length, weak extrapolation;
  learned = expressive but HARD-CAPPED at max_positions (worst for long context); RoPE = relative
  via rotations, `<rope(q,m),rope(k,n)>` depends only on `m−n`, extrapolates best. Freq ladder
  `θ_i=base^(−2i/D)`: fast hands = local resolution, slow hands = global range (slowest wavelength
  bounds context). Long context ⇒ RoPE + NTK/YaRN base-rescaling to extend without retraining.
- RoPE convention consistency is the #1 silent bug: `_rotate_half` interleaved pairs
  (`x[...,0::2]`/`x[...,1::2]`) must match `repeat_interleave(angles, 2)` in the cos/sin cache.
- Transformer = residual stream: `x = x + f(Norm(x))` per sublayer. Attention = ONLY cross-position
  mixing; MLP (`d→4d→d`) = per-position. Two sublayers/block: causal MHA then MLP.
- Pre-LN `x+f(LN(x))` vs post-LN `LN(x+f(x))`. Pre-LN keeps the residual an identity ⇒ backward has
  a standalone `+1` (gradient highway) that survives even if every `∂f≈0`. Post-LN puts `J_LN` on
  the highway ⇒ `Π_{2L} J_LN` compounds ⇒ warmup-sensitive divergence at depth. Post-LN slightly
  better final quality IF tamed; everything modern is pre-LN ("it just trains").
- Gradient highway: `∂L/∂x_ℓ = ∂L/∂x_L Π(I+∂f_k)` → `I + Σ∂f_k + …`; the additive `I` can't be
  killed by a product of small Jacobians (contrast plain deep net = pure product = vanishing).
- RMSNorm = `γ⊙x/√(mean(x²)+ε)`: drops mean-centering AND bias, keeps scaling. Bet (LLaMA): the
  re-scaling is what matters, not re-centering. Cheaper, ~same quality, now standard.
- LN Jacobian `(1/σ)(I − (1/d)11ᵀ − (1/d)x̂x̂ᵀ)`: projects out the mean direction (shift-invariance)
  AND the radial direction (scale-invariance) ⇒ LN removes the scale d.o.f. (activations can't drift
  bigger to cut loss; only direction changes pass through).
- Depth init: rescale each sublayer OUTPUT proj (`Wo`, MLP `fc2`) by `1/√(2L)` so the residual var
  stays `O(1)` over `L` layers (else `2L` additive writes → var ~2L → blow-up). GPT-2 trick.
- fp32-under-bf16-autocast: LN/RMS statistics, softmax, and logits→CE (logsumexp). Rule: matmul =
  accumulate-in-fp32-then-round-ONCE (safe in bf16); normalization = reduce-THEN-DIVIDE (unsafe, the
  imprecise variance gets amplified by the division).
- NaN AT STEP 0 = config bug, not dynamics (no weight moved yet) ⇒ RULES OUT lr (only bites step 1+).
  Bisect forward→loss→backward. Checklist by prob: (1) mask/loss (fully-masked row → softmax over
  all −∞ → 0/0 → NaN), (2) fp16 OVERFLOW→Inf (deep residual > 65504; fix bf16 or GradScaler),
  (3) init/eps (unscaled out-proj var ~2L, or LN eps=0), (4) bad data/labels (target id ≥ V). Senior
  tell = temporal contrast: step-0 ⇒ forward/config; step-N-then-NaN ⇒ "something grew" (grad norms,
  set_detect_anomaly). bf16 UNDERFLOW→0 is SILENT/non-NaN, a different failure.
- Multi-head vs one wide head = independent representation subspaces (each head its own
  Wq/Wk/Wv + softmax → attend to different relations), then `Wo` mixes. Parallel compute is a bonus,
  not the point.
- Silent-arg-bug pattern (seen twice now): a wrong positional arg passes because two config numbers
  are equal (`n_heads==mult==4`; earlier `T==V` in CE). Vary a config value to expose it.
- nanoGPT-idiom: name transformer sublayers (`fc1`/`fc2`, `Wo`) rather than `nn.Sequential` —
  required for the `1/√(2L)` rescale and any surgical weight access (LoRA/tying/freeze).
- Decoding axis: search (greedy/beam) MAXIMIZES likelihood → bland+repetitive on open text; sampling
  (temp+nucleus) matches human-text entropy. Task type picks the tool.
- Temperature = `softmax(z/T)`; log-odds `log(p_i/p_j)=(z_i−z_j)/T` scale by `1/T` (monotone → greedy
  is T-invariant). `T→0` argmax, `T→∞` uniform, entropy monotone `0→ln V`. Return RESCALED LOGITS.
- Top-k = fixed budget k: FAILS both ways (peaked → drags junk tail; flat → amputates real mass).
  Top-p (nucleus) = smallest set with cumprob `>=` p → ADAPTIVE (few when confident, many when
  uncertain). Boundary is `>=` (`>` keeps ONE extra token at exact ties = the HuggingFace quirk).
- Decode filters return MASKED LOGITS (−inf), not probs, so temp→top_k→top_p compose and a SINGLE
  final softmax renormalizes once. top-p's #1 bug: build remove-mask in sorted order then MUST
  `scatter_(-1, sort_idx, mask)` back to vocab order (+ shift-right-by-one keeps the crossing token).
- Beam raw score `Σ_t log p_t ≤ 0` → every token lowers it → length-biased short; fix = length-norm
  `(1/L^α) Σ log p_t` (`α≈0.6–0.7`; α=1 per-token avg). Length-norm is a no-op when all beams are
  equal length (only bites with EOS/variable length) — normalize each hyp by ITS OWN length.
- Neural text degeneration: max-likelihood decoding = repetition attractor (transformer attends to
  its own output → each repeat is lower-entropy/higher-prob, positive feedback); human text is in the
  TYPICAL SET not at the mode. Fix = temp+nucleus. Beam only for LOW-entropy (MT, code, ASR).
- `generate` MUST crop context to last `cfg.T` INSIDE the loop (learned abs pos-table hard-capped at
  T) and keep a SEPARATE full-sequence accumulator (don't return the cropped ctx). eval()+no_grad().
- `torch.cat` default is `dim=0` — appending a token needs `dim=-1`. `x.new_tensor([[v]])` inherits
  dtype+device (GPU-safe). Threshold `values[..., -1:]` (or `-1, None`) keeps size-1 axis for `(B,V)`
  broadcast; bare `[..., -1]` → `(B,)` misaligns unless `V==B` (square-shape silent bug, seen again).
- Sampling batches trivially (rows independent); BEAM does not — top-k prune couples a prompt's k
  hypotheses → batched beam needs `(B,k)` layout, per-row top-k over `k×V`, and a beam-reindex
  (`//V`, `%V`) each step. Hence B=1 scope for a from-scratch beam.
- KV cache: past tokens' K/V never change (causal) → cache them; new token computes only its own
  q,k,v and attends to cached K/V. Cache Q? NO — a query is used only at its own step. Attention
  per step `O(T²)→O(T)`, total gen `O(N³)→O(N²)`. Store per layer `(B,n_kv,T,d_head)` ×2 (K,V).
- KV-cache bytes = `2·L·n_kv·d_head·T·B·dtype_bytes` (2=K&V; `n_kv` is what GQA/MQA shrink;
  `d_head=d/n_heads`). 7B fp16 MHA = **512 KiB/token** → **2 GiB@4k,B1**, **64 GiB@8k,B16** (dwarfs
  the ~14 GB fp16 weights → KV is the binding serve constraint). GQA `n_kv=8` → **4× smaller**.
  Byte ladder: `2^30=1 GiB`; don't call `2^20` bytes a GB (it's 1 MiB). `d_head` is already 128 —
  don't re-divide by heads; and MULTIPLY by n_heads (both were the D4 derivation slips).
- Prefill vs decode: **prefill** = whole prompt in one parallel pass, fills cache, COMPUTE-bound;
  **decode** = 1 token/step, MEMORY-BANDWIDTH-bound. Decode arithmetic intensity ≈ **1 op/byte**
  (stream all weights+KV per token, do ~one token of math) ⇒ GPU idles on HBM. Batching reads each
  weight ONCE for B tokens → FLOPs↑, bytes~const → intensity ↑ ~B× → climb roofline toward compute.
- BW-tied throughput levers (name the mechanism, not just the name): batch (amortize weights),
  continuous/in-flight batch (keep batch full), quantize weights (fewer bytes to stream),
  speculative decode (draft proposes k, big model verifies in ONE pass → multiple tokens/weight-sweep),
  GQA/MQA (smaller KV + fits bigger batch). PagedAttention = capacity/fragmentation fix (INDIRECT).
- KV-cache impl invariants: (1) score `q @ k_all` (ALL cached+new keys) not `q @ k` — the `q@k` bug
  is SILENT in prefill (P=0) and crashes the first decode step (`S` vs `P+S`). (2) Model sees each
  token ONCE: prefill feeds `(B,L)`@`pos_offset=0`; each decode step feeds `(B,1)`@`pos_offset=len-1`
  — feeding the whole running context WITH the past cache double-counts and defeats the cache.
  (3) Offset causal mask = `tril(ones(S,P+S), diagonal=pos_offset)` → keep iff `key_j ≤ pos_offset+q_i`
  (prefill→lower-tri; decode S=1→keep all). (4) Learned pos-table ⇒ positions `pos_offset+arange(S)`.
  (5) `cat` past+new on time axis `dim=-2`; build mask on `x.device`.
