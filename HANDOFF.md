# Handoff — ai-lab-mle-prep

Living state doc for cross-chat continuity. The user starts a **new chat per day**; this
file (plus `progress.md` and each day's `session.md`) carries context forward. The
`ai-lab-mle-prep-day` skill reads this at the start of a session and updates it at the end.

## Current position

- **Week 01 — COMPLETE** (Days 01-05 done). LM Foundations & PyTorch Fluency.
- **Week 02 — COMPLETE** (Days 01-05 done). Transformers + generation: attention+pos-enc;
  TransformerBlock+TinyGPT; decoding; KV cache; **generation debug lab**.
- **Week 03 — IN PROGRESS.** All five Day session.md plans fleshed out from curriculum
  (MDPs → policy gradient → baselines/advantage → GAE → KL-constrained RL). **Days 01-03 DONE.**
- **Next up: Week 03 Day 04 — GAE & the Bias/Variance Tradeoff.** RL depth is a
  DECLARED TOP GAP — derive.

## Next session

Start with the prompt: **"Start Week 3 Day 4"**. Session plan at
`week-03-rl-foundations/day-04-gae-bias-variance/session.md`.

Day 4 = GAE. It slots straight onto D3's machinery: `δ_t = r_t + γV(s_{t+1}) − V(s_t)` and
`GAE(γ,λ) = Σ (γλ)^l δ_{t+l}` are just two more `weight_fn`s for
`actor_critic.batch_gradient_statistics`, so the finite-difference `exact_gradient` will
**quantify the bias** for each λ — the headline experiment is a λ-sweep table (bias vs variance,
λ=0 → pure TD, λ=1 → Monte-Carlo reward-to-go, and it should reproduce D3's numbers at both ends).
D3 already previewed the bias source: critic error. Derive the telescoping identity
`Σ_l γ^l δ_{t+l} = G_t − V(s_t)` (λ=1) and the geometric-average interpretation.

**Also worth building on D4 (D3 left it open):** a TIME-INDEXED finite-horizon baseline `V_t(s)`
(`T−t` Bellman backups → a `(T,S)` table). It should beat both the pooled `b(s)` and `b*`, and it
converts D3's "mismatched V^π is unbiased but WORSE" cautionary result into a constructive fix.

Quiz D3 cold: EGLP (should now be instant — if so, retire it); the **h_t-measurability table**
(which quantities qualify and the consequence of each); why `b=Q^π` DELETES the gradient rather
than biasing it; `b*` vs `V^π` (weighted vs unweighted mean return) and why nobody uses `b*`;
the GRPO `(K−1)/K` identity and why the STD division is the real bias.
**Re-drill the two D3 drill misses:** (1) **"unbiased ≠ variance-reducing"** — the mismatched-critic
scenario, he answered "biased" (0.25); (2) **entropy regularization's failure mechanism** —
`∂H/∂z_k = −p_k(log p_k+H)` decays as π sharpens ⇒ preventative not cure, plus KL-to-ref as the
LLM alternative (0.5). And the perennial: **six sources of REINFORCE variance** (still ~2.5/6 on
the third attempt — drill the estimator-walk table in Running facts). Keep the Week-2 spaced item
alive: attention-NaN fastest-first ORDER.

**Recurring behavioural gaps to police:** he still answers the first clause and stops (D3: gave
what entropy buys but not how it fails, and did NOT cite the β-sweep numbers that were on his
screen). Demand the second half of every question, and demand he quote his OWN measured numbers.

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

- **W2 D5 — Generation Debug Lab (Week-2 capstone).** Built `week-02/code/debug_lab.py` — six
  INVARIANT regression tests (causal-no-leakage, cached==uncached, masked-row-not-all-inf,
  no-nan-fwd-bwd, shape-contract, overfit-one-batch) + a planted-bug demo that monkeypatches Bug A
  (mask `diagonal=0`) and proves the suite fires. 6/6 pass (harness + pytest). Wired the Week-2
  generation notebook (`work.ipynb`): causal attention heatmaps (0 mass above diagonal), all 5
  decoding strategies, cached==uncached + **2.85× decode speedup**, live NaN repro/fix. Taught the
  **four-contract debug map** (causal/shape/numeric/train-inference) and the two crown-jewel tests.
  Quiz (D1-D4) 5.2/6 (KV math now automatic; D3 decoding solid). Bug hunt A-E ~4.1/5 (slipped on the
  CE reshape fix and the crop-the-INPUT axis for E). **Two genuine TEST-DESIGN bugs surfaced live:**
  (1) `cached==uncached` was TOOTHLESS on the successor-trained model — `next=current+1` is
  position-wise so a broken attention mask doesn't change output; fix = use a RANDOM model (logits
  depend on all positions). (2) `torch.zeros_like(bool_mask)` is bool → `masked_fill(-inf)` casts to
  True → `isfinite` trivially passes; fix = float scores + check finiteness AFTER softmax. Drills:
  train/inference-mismatch ~0.75 (needs to LEAD with teacher-forced vs free-running); **attention-NaN
  fastest-first ~0.4 — came out as guesses, the clear remaining soft spot.** Checkpoints ~4.4/5.
  Wrote deliverable + week-summary + retrospective. WEEK 2 COMPLETE.

- **W3 D1 — MDPs, Returns & Value Functions.** Fleshed out all 5 Week-3 session.md plans first.
  Built `week-03-rl-foundations/code/mdp.py` (7/7 self-check invariants PASS): `policy_matrices`
  (einsum contractions), `bellman_expectation_backup`, iterative policy eval (γ-contraction in
  code) == exact linear solve `(I−γP_π)^{-1}r_π`, `q_from_v`, `advantage`, `value_iteration`
  (max-backup + one-hot greedy). 2-state chain: `V^*=[10,10]=1/(1−γ)`, `π^*`="reach & stay",
  random `V^π=[4.75,5.41]`<V^*. Derived Bellman expectation cleanly on the whiteboard (via the
  Q-then-V-substitution route — correct; noted the Q line is where Markov + law-of-total-expectation
  live). Week-2 cold quiz **~4.15/5** (KV math now automatic + exact; NaN checklist came out as
  knowledge not guesses — drill the fastest-first ORDER + remember fp16 overflow is bucket-agnostic;
  teacher-forcing decisive-because-same-weights was the missing half). Interview drills: Q1 (why not
  SL) 0.9; **Q2 (critic/unbiasedness) 0.6 — knew critic=V^π but missed the EGLP unbiasedness
  argument → explicit Day-3 seed.** Code was strong (all correct, only nits: unseeded V init;
  argmax on last-iter Q has a one-step lag, recompute Q after loop). Deliverable written.

- **W3 D2 — Policy Gradient Theorem & REINFORCE.** Built `week-03/code/reinforce.py` (9/9
  self-checks, ~27 s): tabular softmax policy, `episode_logprobs` (LM `log_softmax`+`gather`
  idiom), `discount_cumsum`, the two `Psi_t` modes, `episode_gradient` via `autograd.grad`,
  `collect_batch`/`gradient_statistics`, and the REINFORCE loop. **Design win: the env is the
  Day-1 finite MDP, so `exact_J` (T Bellman backups) + central finite differences give a
  GROUND-TRUTH gradient** — unbiasedness became an assertion, not a claim. Results: both
  estimators match FD truth (max|err| 0.0069 / 0.0038); **reward-to-go = 3.15× variance
  reduction at identical mean**; learned π == Day-1 π* ("reach & stay"), return 7.916/8.0.
  Noticed unprompted-by-him structure: gradient rows sum to zero (softmax over-parameterization,
  same null space as `diag(s)−ssᵀ`) and state-0 grad is 2.7× state-1's (discounted state
  visitation). Cold quiz (D1) **4.5/6** — clean Bellman + advantage whiteboards; **Q5 LLM→MDP
  0.55: put the policy's randomness into the DYNAMICS** (action = sampled TOKEN not logits;
  dynamics = deterministic append; γ≈1 not 1/length). NaN checklist up to 0.75 from 0.4.
  Derivation **1.8/4**: Part 1 clean; **Part 2 he omitted ρ₀ and P from `p_θ(τ)` entirely =
  assuming the conclusion**; **Part 3 invalid — claimed `∇_θ E[r_{t'}]=0` (false; early rewards
  DO depend on θ) by treating the sampled trajectory as θ-independent.** Correct proof = tower
  rule on `h_t` (past reward is `h_t`-measurable) + EGLP on the inner `E_{a_t}`. Part 4 gave 2 of
  7 Ψ_t choices. Live code bugs: **`np.zeros(dtype=np.long)` gradient buffer (int truncation →
  silent all-zero gradients, the week's most dangerous bug)**; `sum` instead of `mean`; variance
  un-normalized; `T = rewards.shape` (tuple) ×2; `actions.unsqueeze` on a numpy array; **an
  episode-internal `G_t − G.mean()` "baseline" (BIASED — b must be `h_t`-measurable)**; a
  `reward_to_go_baseline` branch silently identical to `reward_to_go`. Heavy idiom Q&A
  (gather-vs-advanced-indexing incl. the batched `arange(B)[:,None]` case, `autograd.grad` tuple
  return, `detach` not in-place, optimizer forms for ASCENT). **Drills weak: 0.35 / 0.0 / 0.3** —
  named 1 of 6 variance sources, no answer at all on the temp/top-p off-policy scenario, and
  Drill 3 restated the conclusion without the argument or the 3.15× number that was on screen.
  Recurring pattern (also in the W2 retro): **answers the first clause of a question and stops.**
  **RE-DRILLED same session (he asked for it) — big recovery:** LLM→MDP 0.55→**0.85**;
  factorization 0.4→**0.7** (derivation now complete; couldn't name model-based RL as the class
  needing P); causality proof 0.3→**0.85** (clean, via the slick `∇E_{a∼π}[1]=∇1=0` route);
  Ψ ladder 0.3→**0.6**; variance sources 0.35→**0.6**; off-policy 0.0→**0.72** (got PPO's
  motivation unaided: "still off-policy after epoch 1, fix = importance sampling"); pushback
  0.3→**0.5**. Two residual tells: (1) repeated the "depends on θ" slip when saying why `t'≥t`
  survives — the discriminator is `a_t`-MEASURABILITY, not θ-dependence; (2) **inflated his own
  measured 3.15× to "~4×"** — flag hard, never round a measured number in your favor.

- **W3 D3 — Baselines, Advantage & Variance Reduction.** Built `week-03/code/actor_critic.py`
  (**39/39 staged checks, ~31 s**): `fit_state_baseline` (two-`bincount` tabular critic),
  `oracle_state_baseline` (Day-1 linear solve), `optimal_state_baseline` (`b*`),
  `weights_with_baseline`, the deliberately-buggy `weights_within_episode_baseline`,
  `batch_gradient_statistics` (takes a `weight_fn` closure so the DATA is held fixed across
  estimators), `policy_entropy`/`entropy_gradient`, and `train` with a β sweep.
  **Variance ladder (identical rollouts): total_return 20.033 → reward_to_go 6.317 (3.17×,
  reproducing D2's 3.15× = free regression test) → +fitted baseline 1.765 (11.35× cumulative,
  3.58× on top) → +`b*` 1.476 (13.57×, the best unbiased, as theory predicts).**
  Learned π == Day-1 π*, return 7.966/8.
  **Two results that became the day's lessons.** (1) The oracle **`V^π` baseline is unbiased
  (max|err| 0.0046) but INCREASES variance 27%** (7.769 vs 6.317) — `V^π≈3.4` is the
  infinite-horizon value while `Ψ_t` uses the T-step reward-to-go pooled over t (mean ≈1.27), so
  it DEcentres. **My harness assertion "oracle V^π also stacks" was the thing that was wrong, not
  his code** (3rd time a scaffold claim became the best lesson: W1-D5 noiseless-overfit, W2-D5
  toothless cached==uncached). ⇒ unbiasedness and variance reduction are INDEPENDENT; this is the
  unconverged-PPO-critic failure mode. (2) The **biased** within-episode baseline has the LOWEST
  variance of all (0.784, 25.56×) with its mean 38% wrong (0.561 vs 0.902) ⇒ low variance alone is
  worthless; only "variance reduction at FIXED mean" is a claim.
  **Entropy β sweep:** β 0/0.5/2/5 → H 0.074/0.118/0.294/0.522, return 7.966/7.925/7.634/6.375
  (monotone both ways). β=0.05 did nothing (ΔH=0.0024) because `∂H/∂z_k = −p_k(log p_k+H)` DECAYS
  as π sharpens ⇒ needs β~O(1); entropy bonus is a collapse PREVENTATIVE, not a cure.
  **Cold quiz (D2) 6.25/8:** EGLP finally produced cold and unprompted (**1.0** — retire it);
  PG derivation clean but **regularity conditions unknown (0.7)**; ρ₀/P now written out in full
  (1.0, fixed from D2); causality proof correct via tower rule (0.8); Ψ ladder **7/7 rungs + bias
  column** (0.85, ordering principle only half-named); NaN checklist 0.8.
  Still weak: **six variance sources ~0.4 (third attempt)** and off-policy 0.7.
  **Derivations:** D1(a) and the `b*` minimization both correct first try (only a dropped square in
  a denominator); needed 3 rounds of hints on D1(b) (`b=Q^π` ⇒ mean exactly 0) — the block ran
  ~2× long. Taught **measurability as "if I tell you h_t you can tell me X"**, which he had
  explicitly asked for. Gave him the GRPO `(K−1)/K`-times-RLOO identity + the std-division-is-the-
  real-bias point. **Interview drills: D1 0.85, D2 0.9 (best answer — separated Q-as-Ψ from
  Q-as-b unprompted), D3 0.5, D4 0.25 (answered "biased" to the mismatched-critic scenario — the
  exact thing his own harness had just measured, and he did not cite his own β-sweep numbers).**
  **Live code bugs (all caught by the harness):** `np.bincount` on 2-D input + scalar `weights`;
  `torch.softmax` with no `dim`; `entropy_history.append(tensor)` (graph leak); calling
  `entropy_gradient` INSIDE `no_grad`; and the notable one — **`theta += numpy_array` is not
  in-place**: numpy takes over, rebinds the name, and `theta` silently becomes a fresh tensor with
  `requires_grad=False` (probed and confirmed). **Process change (he asked): rebuilt the harness as
  STAGED and promoted it to a repo-wide convention** (skill file + `shared/templates/
  exercise-harness-template.py`) so he can test each TODO the moment he writes it.

## Learner profile (quick facts)

- Strong: PhD-level math/optimization, senior ML eng, real post-training experience
  (SFT/DPO/GRPO, retrieval, personalization). RL intuition is genuinely good.
- **Top gaps:** (1) ML systems / memory-cost reasoning; (2) making RL depth
  interview-crisp and derivable on demand.
- Prior work to reference in examples: Amazon personalized prompt-augmentation (RAG:
  "pet food" → "pet food for cats"), DPO to beat a Sonnet teacher via HF.
- Teaching style that works: derive from first principles, quiz from prior day, give code
  stubs + self-check harness (don't pre-solve), connect to his projects and later weeks.

### Running facts — Week 3 (RL foundations)

- RL vs SL: `J(θ)=E_{τ∼p_θ}[R(τ)]` — the SAMPLING DISTRIBUTION depends on θ. This (not just
  non-differentiable reward) is the FUNDAMENTAL reason you can't plain-backprop; forces the
  log-derivative/score-function trick `∇E_{p_θ}[f]=E_{p_θ}[f ∇log p_θ]`.
- MDP tuple `(S,A,P,r,γ,ρ₀)`. Markov property = future ⊥ past | present ⇒ it's EXACTLY what makes
  V^π(s) a function of s alone (no path dependence ⇒ the Bellman recursion closes).
- Trajectory factorization `p_θ(τ)=ρ₀(s₀)Π_t π_θ(a_t|s_t)P(s_{t+1}|s_t,a_t)`: only the π terms
  carry θ ⇒ dynamics & ρ₀ vanish from ∇log p_θ (⇒ model-free, the D2 punchline).
- Return `G_t=Σ_k γ^k r_{t+k}`; γ sets effective horizon `≈1/(1−γ)` (verified: V^*=10 at γ=0.9).
- `V^π(s)=E_π[G_t|s]`, `Q^π(s,a)=E_π[G_t|s,a]`, `V^π=E_{a∼π}[Q^π]`. Advantage `A^π=Q^π−V^π`;
  `E_{a∼π}[A^π]=0` per state (proven in code) = why V^π is the natural (variance-reducing) baseline.
- Bellman EXPECTATION `V^π(s)=Σ_a π(a|s)[r+γΣ_{s'}P V^π(s')]` (linear in V, fixed π = evaluation)
  vs Bellman OPTIMALITY `V^*(s)=max_a[r+γΣ P V^*]` (nonlinear, greedy-in-Q^* is optimal). Both are
  γ-contractions in sup-norm ⇒ unique fixed point ⇒ iterative eval / value iteration converge.
- Deriving Bellman: peel one step `Q^π(s,a)=r(s,a)+γΣ_{s'}P(s'|s,a)V^π(s')` — that line silently
  uses Markov (future depends only on s') + law of total expectation; then V^π=Σ_a π Q^π, substitute.
- EGLP lemma (Day-1 GAP, seed for D2/D3): `E_{a∼π}[∇_θ log π_θ(a|s)]=∇Σ_a π=∇1=0` ⇒ any STATE-ONLY
  baseline b(s) is UNBIASED: `E[∇log π · b(s)]=b(s)·0=0`. Reduces variance, not mean. b=V^π ⇒
  advantage form `∇J=E[Σ_t ∇log π · A^π]`.
- LLM→MDP: state=prompt+tokens, action=next token, π=LM softmax, dynamics DETERMINISTIC (append),
  reward SPARSE & TERMINAL (RM/verifier @EOS), γ≈1. ⇒ long horizon + sparse reward = credit
  assignment is the crux (why baselines/GAE matter). PPO critic estimates V^π; GRPO uses a group
  mean as the (still-unbiased) baseline instead of a learned critic.
- Numpy idioms locked: `np.eye(A)[idx]` = one-hot rows (= scatter `M[arange, idx]=1`); solve linear
  systems with `np.linalg.solve`, NEVER `inv`; explicit `V[:,None]` when subtracting a per-state
  vector across the action axis (S==A square-shape silent-bug trap, seen again).
- LLM policy notation: net maps state→**logits** `z_t∈R^V`; `π_θ(·|s_t)=softmax(z_t)` is the
  DISTRIBUTION; `π_θ(a_t|s_t)` is ONE SCALAR coordinate at the SAMPLED token. Logits ≠ action.
  Dynamics = deterministic append (`s'=s⊕a`), so ALL trajectory randomness lives in π_θ.
- **PG = cross-entropy on your OWN samples, reweighted.** Same `log_softmax→gather` pass as LM
  training; differences are (a) gathered token is sampled not gold, (b) weight is `Ψ_t` not `1/BT`.
  `∇_z log π(a|s) = e_a − p` = exact NEGATIVE of W1's `∂CE/∂z = p − e_y` ⇒ `Ψ_t<0` pushes mass AWAY
  from the sample, which SFT structurally cannot express. Same off-by-one shift as LM training.
- Score-function `E[f ∇log p]` (needs only to EVALUATE f; discrete/black-box OK; high variance,
  f is a scalar weight) vs pathwise/reparam `x=g_θ(ε)` (needs ∇f + continuous x; low variance).
  LLM RL is forced into SF on BOTH counts. **DPO = the attempt to get back to pathwise** (closed
  form over fixed pairs ⇒ low variance, paid for with off-policy static data).
- PG derivation regularity conditions: exchange ∇ and Σ (dominated convergence), **support of p_θ
  must not depend on θ** (fine for softmax), R(τ) not a function of θ.
- Reward-to-go causality (COLD-DRILL THIS): for `t'<t`, condition on `h_t=(s_0,a_0,…,s_t)`; `r_{t'}`
  is `h_t`-measurable ⇒ pulls out; inner `E_{a_t∼π}[∇log π]=0` by EGLP ⇒ term is 0. **NOT** because
  `∇E[r_{t'}]=0` — that's false, early rewards do depend on θ. For `t'≥t` it fails (`r_{t'}` depends
  on `a_t`) — that asymmetry IS causality. Effect: bias UNCHANGED, variance strictly lower.
- Ψ_t ladder (high→low variance): `R(τ)` → `G_t` → `G_t−b(s_t)` → `Q^π` → `A^π` → `δ_t` (biased if
  V≠V^π) → `GAE(γ,λ)`. Every algorithm REINFORCE→PPO differs ONLY in Ψ_t.
- **What Ψ_t IS:** `∇log π(a_t|s_t)` is the DIRECTION ("make a_t more likely at s_t"); `Ψ_t` is the
  signed scalar CREDIT for that decision. **Ψ_t MUST depend on the action** — a state-only Ψ gives
  an IDENTICALLY ZERO gradient by EGLP (that's the definition of a baseline). Division of labor:
  Ψ_t depends on `a_t` (all the signal); the baseline `b` does not (keeps it unbiased). `A^π=Q^π−V^π`
  states both at once.
- Ladder ORDERING PRINCIPLE (he got this wrong — said "shrink |Ψ|"): shrinking Ψ by `c` scales mean
  AND std by `c`, so SNR is unchanged. It's **lower variance at FIXED mean**, via two mechanisms:
  (1) delete provably zero-mean terms = control variate (reward-to-go, baselines); (2) replace a
  SAMPLE with its EXPECTATION = Rao-Blackwellization, `Var(E[X|Y]) ≤ Var(X)` (G_t → Q^π → bootstrap).
  Mechanism 2 needs a critic you don't have ⇒ critic error is the source of the bias in δ_t/GAE.
- Cross-term count: `ĝ = (Σ_t ∇log π_t)(Σ_{t'} γ^{t'} r_{t'})` = `T²` pairs; the `T(T−1)/2` pairs with
  `t'<t` are provably zero-mean = pure noise. **CORRECTION to an earlier over-claim: the variance
  GAP does NOT grow like T².** Measured on the 2-state chain (γ=0.99, 3000 eps): ratio 1.84× @T=2,
  2.70× @T=8, 3.09× @T=32, **4.09× @T=128** (both variances grow ~T^2.3; the ratio grows slowly and
  monotonically). Interview claim = "free, consistent, horizon-growing win," with the table.
- `t'≥t` survives because `r_{t'}` depends on **`a_t`** ⇒ NOT `h_t`-measurable. NOT because it
  "depends on θ" (past rewards depend on θ too — that's not the discriminator). He has slipped on
  this exact point twice.
- Model-based RL is the class that NEEDS the dropped `P` term: DP/value iteration (his own Day-1
  `mdp.py` uses `mdp.P` in every backup), Dyna/MBPO (learn `P̂`, roll out synthetic data), Dreamer /
  differentiable sims (need `∂s'/∂a` for a PATHWISE gradient), AlphaZero MCTS (known P for planning).
  Trade = sample efficiency vs. the policy exploiting model error.
- LLM determinism consequences: all trajectory randomness is in π_θ; the transition "model" is exact
  and free (model error is not an LLM-RL failure mode); environment stochasticity is NOT a variance
  source here — gradient noise is policy sampling + reward noise only.
- Interview hygiene: **never round a measured number in your favor** (3.15× is not "~4×"). The
  credibility of a quantitative answer rests on the number being exact and reproducible.
- Sources of REINFORCE variance (D2 drill gap; he got 3 of 5 on the re-drill): (1) SF estimator is
  zeroth-order — sees no ∇f, only a scalar score; (2) temporal credit misassignment → reward-to-go;
  (3) all-positive returns, no centering → baseline/advantage; (4) horizon: `T(T−1)/2` zero-mean
  cross terms; (5) compounding ρ₀/π/P stochasticity over an exponentially large trajectory space →
  batch size, the EXPENSIVE lever (only `1/n`, so halving noise costs 4× compute); (6) reward-model
  NOISE (not reward SCALE — he correctly noted scaling Ψ moves mean and std together, so SNR is
  unchanged; scale is an effective-LR issue, and advantage whitening helps mostly via CENTERING);
  (7) LLM-specific: sparse terminal reward, one scalar shared across all T tokens.
- **Off-policy-by-accident (the #1 RLHF sampling bug):** rollouts at temp 0.7 / top-p 0.9 come from
  μ≠π_θ; both distortions oversample already-likely tokens ⇒ reinforce the mode ⇒ ENTROPY COLLAPSE
  ("reward up, then degeneration"). Fix order: sample RL rollouts at T=1.0 untruncated → IS ratio
  (variance explodes over a product of T ratios ⇒ why PPO clips a PER-TOKEN ratio) → note clipping
  can't repair top-p (**support** mismatch, can't estimate mass you never sample) → instrument
  entropy + `KL(π_θ‖π_ref)`.
- Baseline validity: `b` must be `h_t`-MEASURABLE (known before `a_t` is drawn). An episode-internal
  `G_t − G.mean()` depends on `a_t` and the future ⇒ BIASED. Baselines are a BATCH-level quantity.
- The surrogate `Σ_t Ψ_t log π_t` is a **gradient generator, not an objective** — its value is
  meaningless (watch mean reward, never `policy_loss`). `.sum()` not `.mean()` over t, and `/batch`
  when using `.backward()` (the grad-accum "sum of means → mean of means" fact again).
- Optimizers DESCEND ⇒ ascent needs `theta.grad = −g`, or `SGD(..., maximize=True)`, or minimize
  `loss = −surrogate`. `autograd.grad` returns a TUPLE (one per input) and leaves `.grad` untouched
  (right for per-sample measurement loops); `.backward()` accumulates. `.detach()` on an
  `autograd.grad` result is redundant at `create_graph=False`; `.detach()` is NOT in-place.
- **Accumulator dtype is part of correctness**: `np.zeros(..., dtype=np.long)` for a gradient buffer
  silently truncates floats to 0 and every downstream check "passes" on zeros. Same family as W2-D5
  `zeros_like(bool_mask)`. Also `theta[:, states]` gives `(B,B,T,A)` — pair with `arange(B)[:,None]`.
- Indexing rule: whole-slice lookup → advanced indexing `x[idx]` / `index_select` / `F.embedding`
  (backward is a SCATTER-ADD, so repeated states/tokens accumulate); one-element-per-position →
  `gather` (index must be the FULL output shape); `take_along_dim` broadcasts the index for you.
- Softmax over-parameterization: adding a constant to a logit row leaves π unchanged ⇒ J is flat
  along `1` ⇒ **every policy-gradient row must sum to zero** (same null space as `diag(s)−ssᵀ`).
  A row that doesn't sum to zero is a bug. Gradient magnitude tracks DISCOUNTED STATE VISITATION
  (early states get bigger gradients — in LLMs, early tokens).
- The `γ^t` factor on the reward-to-go weight is required for exact unbiasedness w.r.t. the
  discounted objective; PPO/GRPO DROP it (formally biased, empirically better conditioned).
- Finite-MDP ground-truth pattern (reusable all week): `J_T` = T Bellman backups from V=0, then
  central finite differences ⇒ an exact gradient to check ANY new estimator's unbiasedness
  numerically. `np.var` already divides by n; per-episode variance (estimator property) ≠ variance
  of the batch mean (`/n`) — batch size buys only `1/n`, baselines/GAE attack the numerator.
- **MEASURABILITY, in one line: "`X` is `h_t`-measurable" = if I tell you `h_t`, you can tell me
  `X`** (X is a deterministic function of what's already known). This single notion settles
  everything: `r_{t'}` for `t'<t` qualifies (⇒ reward-to-go valid); `r_{t'}` for `t'≥t` does NOT
  (depends on `a_t`, AND is correlated with `∇log π`, so it can't factor out); `b(s_t)`/`V^π`/`V_φ`
  qualify (⇒ unbiased regardless of ACCURACY); episode-internal `G_t−mean(G)` does not (⇒ biased,
  measured 38% low). The discriminator is measurability, **NOT θ-dependence** (past rewards depend
  on θ too) — he has slipped on this 3×.
- **Baseline unbiasedness holds for ANY `h_t`-measurable `b`** (history-dependent is fine, not just
  `b(s)`): tower-rule on `h_t` → `b` pulls out → EGLP kills `E_{a_t}[∇log π]`. Always finish with
  the second half: **mean unchanged, VARIANCE reduced** — that's the whole point.
- **An action-dependent baseline DELETES the gradient, it doesn't merely bias it.** With `Ψ_t=G_t`
  and `b=Q^π(s_t,a_t)`: condition on `(s_t,a_t)` — now `∇log π` is the CONSTANT and `G_t` the random
  thing, so `E[G_t−Q^π|s_t,a_t]=0` by definition of `Q^π` ⇒ mean is the ZERO matrix, bias `=−∇J`.
  Per-sample the weights look perfectly centred; only a ground-truth check finds it. Generally
  `E[∇log π·b(s,a)] = ∇E_{a∼π}[b(s,a)]` ⇒ you subtract the gradient of "make b large" (Q-Prop must
  add a correction back).
- Optimal baseline `b*(s) = E[||∇log π||²G]/E[||∇log π||²]` = the score-magnitude-WEIGHTED mean
  return; `V^π` is the UNWEIGHTED one; equal iff score magnitude ⊥ return. Nobody uses `b*`: ratio
  of noisy estimates, wants a per-COORDINATE baseline for vector θ, and `V^π` is reusable (gives
  bootstrapping + GAE). Measured: `b*` 1.476 vs fitted `V` 1.765 — real but small.
- **UNBIASED ≠ VARIANCE-REDUCING (D3's headline, D4 re-drill).** A horizon-mismatched oracle `V^π`
  was unbiased (0.0046) and yet +27% variance (7.769 vs 6.317 with NO baseline) — `V^π≈1/(1−γ)=3.4`
  vs pooled reward-to-go ≈1.27 ⇒ it DEcentres `Ψ_t`. This is the unconverged/mis-targeted PPO
  critic: no crash, no NaN, advantage still ~zero-mean after whitening, reward stalls, noise up, and
  **the gradient is NOT biased** (a critic is a function of `s_t` ⇒ `h_t`-measurable by
  construction; how you FIT it is irrelevant to that). Diagnostic = critic **explained variance** vs
  realized return-to-go. **Decisive experiment = zero the baseline and re-measure gradient variance
  on the SAME rollouts; if variance FALLS, the critic is worse than nothing ⇒ mismatch, not bias.**
- **Low variance alone is worthless** (`Ψ_t=0` has zero variance). The biased within-episode
  baseline had the LOWEST variance of everything (25.56×) with a 38%-wrong mean. Only
  "variance reduction at FIXED mean" is a claim — so pair every variance number with an
  unbiasedness assertion.
- Baselines STACK with reward-to-go (3.58× on top of 3.17× = **11.35× cumulative**) because they
  kill different noise: reward-to-go deletes zero-mean cross terms, the baseline centres the rest.
- **GRPO group mean = `(K−1)/K` × RLOO.** Identity: `R_k − (1/K)Σ_j R_j = (1−1/K)(R_k −
  (1/(K−1))Σ_{j≠k}R_j)`. The leave-one-out baseline IS unbiased, so the group mean's "bias" is a
  pure SCALE factor absorbed into the LR ⇒ unbiased in DIRECTION. The real bias is the **std
  division**: nonlinear in the group, doesn't factor out, blows up on degenerate all-correct/
  all-wrong groups (σ≈0) — what Dr. GRPO says to drop.
- PPO-vs-GRPO in one sentence: PPO buys per-token low-variance advantages from a learned critic and
  pays with the critic's approximation bias + a second network's memory/compute; GRPO drops the
  critic (no critic bias, no critic memory) and pays K× rollout cost + loss of per-token credit
  assignment (one sequence-level advantage broadcast to every token).
- **Entropy bonus:** `H(π)=−Σπlogπ`, `∂H/∂z_k = −p_k(log p_k + H)` (rows sum to zero, same softmax
  null space). Force **DECAYS as π sharpens** ⇒ weakest exactly where most needed ⇒ it is a collapse
  **PREVENTATIVE, not a cure**, and needs `β~O(1)` (the PG's own scale) — β=0.05 moved H by 0.0024.
  Measured tradeoff β 0/0.5/2/5 → H .074/.118/.294/.522, return 7.97/7.93/7.63/6.38 (monotone both
  ways). LLM RLHF barely uses it: a UNIFORM-entropy target is wrong for text (you want high entropy
  only where genuinely uncertain); the shipping regularizer is `KL(π_θ‖π_ref)`, anchored to a
  known-good distribution instead of to uniform.
- **`theta += numpy_array` is NOT in-place.** numpy takes over, returns a new object, Python rebinds
  the name ⇒ `theta` silently becomes a fresh tensor with `requires_grad=False`, `is_leaf=True`
  (values and dtype are fine — only the property you depend on dies). Also
  `numpy + torch` RAISES while `torch + numpy` works (`Tensor.__array_priority__=1000` only wins on
  the left), and the error names **concatenation**, an op nowhere in the code. Rule: numpy = storage/
  statistics layer, torch = autograd layer, every crossing gets an explicit
  `torch.as_tensor(..., dtype=theta.dtype)`. Heuristic: **an error naming an operation you never
  called = a type-coercion fallback; look at operand TYPES.**
- `no_grad` scopes by CODE REGION ⇒ never build a graph you need inside it (calling
  `entropy_gradient` there raises "does not require grad"). Also `beta * f(x)` still evaluates
  `f(x)` when `beta==0.0`. Log **floats** (`.item()`/`float()`), never tensors — appending a
  0-dim tensor retains that iteration's whole graph. On GPU `.item()` forces a device sync: fine per
  logging interval, a pipeline stall in a hot loop.
- `np.bincount` needs **1-D** input (`.ravel()` both indices and weights) and `weights` must be an
  ARRAY not a scalar; guard the empty bucket or unvisited states become `nan`. Two-`bincount` idiom:
  `sums/counts` = unweighted mean (`V^π`), `Σ(wG)/Σw` = weighted mean (`b*`).
- Indexing, one more time: ONE index array → whole rows (`pi[states]` → `(n,T,A)`); TWO aligned
  index arrays → one element each (`pi[states, actions]` → `(n,T)`, numpy's `gather`). Better still,
  expand the norm `||e_a−p||² = 1 − 2p_a + ||p||²` to avoid materializing an `(n,T,V)` one-hot — at
  V=128k that's the reason production code never builds one-hots.
- Grad of a SCALAR w.r.t. a matrix has the MATRIX's shape (Jacobian-layout questions only arise for
  non-scalar outputs). `torch.softmax` has **no default `dim`**. Use `log_pi = log_softmax(...)`
  then `pi = log_pi.exp()` — one stable pass, and the two are exactly consistent.
- **Harness convention (repo-wide as of D3):** staged self-checks, one stage per TODO,
  `NotImplementedError` → `[TODO]`+skip, `--fast`/`--list`/named-subset CLI, lazy `Ctx` fixtures
  with their own seeded RNGs, `_fail_fast` probes before expensive fixtures. Skeleton:
  `shared/templates/exercise-harness-template.py`. Two design rules earned here: **assert that the
  BUG is detectably a bug** (relative threshold, e.g. `>5×` the unbiased error, so it survives batch
  -size changes), and **when a well-motivated check fails, check the CLAIM before the code** — the
  scaffold's assertion has now been the wrong one three times, each time producing the day's best
  lesson.

## Conventions

- Run code: `./.venv/bin/python ai-lab-mle-prep/<path>` from `/Users/sshivam/1point3acres`
  (torch 2.12, Python 3.14, pytest installed).
- **Every coding exercise uses the STAGED harness** (one stage per TODO; `NotImplementedError`
  → `[TODO]` and skip; `--fast` / `--list` / `<stage names>` CLI; lazy `Ctx` fixtures;
  `_fail_fast` before expensive stages). Skeleton: `shared/templates/exercise-harness-template.py`.
  Reference: `week-03-rl-foundations/code/actor_critic.py`. Rationale: he must be able to test
  each function the moment he writes it, not after finishing the file.
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
- Four debugging CONTRACTS (every transformer/gen bug = one violation): CAUSAL (no future leakage),
  SHAPE/AXIS (each axis means one thing; worst when two dims equal), NUMERIC (reductions stay finite
  + normalized), TRAIN/INFERENCE (token seen the same at gen as train). Method: symptom → contract →
  cheapest EXCLUSION test → decisive LOCALIZATION test.
- Two crown-jewel INVARIANT tests (weights-independent → run pre-training in CI): (1) causal-leakage
  = perturb tokens > t, assert logits ≤ t are BIT-IDENTICAL; (2) cached==uncached = KV-cache greedy is
  token-identical to naive path. Leakage is dangerous because the model CHEATS (peeks at the answer) →
  train loss collapses to ~0 while learning nothing → total failure at inference.
- Teacher forcing (train: gold prefix, 1 parallel pass, causal mask makes it valid) vs free-running
  (inference: feed model's own outputs back, T sequential steps). "Trains fine but generates garbage"
  ⇒ feed a known example: teacher-forced correct + free-running garbage ⇒ bug is 100% in the gen path
  (positions/cache/mask/eval/crop/preprocessing skew), NOT the weights.
- Overfit-one-batch = the #1 plumbing smoke test: a correct model MUST drive one fixed batch to ~0
  (memorize == solve). Fail ⇒ bug (detached graph, no_grad on the optimized branch, wrong .parameters(),
  LR, shape). Then GREEDY-the-overfit to also test the gen path (learning works vs using-what-it-learned).
- Attention-NaN fastest-first (SOFT SPOT — memorize the order): temporal clue (step-0=config, rules out
  LR; step-N=grew) → (1) fully-masked softmax row `softmax([-inf…]) = exp→0 everywhere → 0/Σ0 = 0/0 = NaN`
  (fix: every row keeps ≥1 key) → (2) fp16 overflow→Inf→Inf−Inf=NaN (bf16/fp32 softmax) → (3) LN/RMS
  eps=0 → (4) exploding grads (clip/warmup) → (5) bad data / target id ≥ V.
- TEST-DESIGN lesson: an invariant test's power = the input's SENSITIVITY to the thing under test. A
  cached==uncached test on a POSITION-WISE task (successor `next=cur+1`, needs no attention) is TOOTHLESS
  to a broken attention mask — use a RANDOM model (logits depend on all positions). And a passing test
  can be wrong for the wrong reason (`zeros_like(bool_mask)` → isfinite trivially True; use float + check
  AFTER softmax).
