# AI Lab MLE Prep Progress

## Current Status

- Current week: **Week 04 — RLHF & PPO for LLMs** (Days 01-03 of 5 done)
- Previous day: Week 4 Day 02 (the PPO objective; `code/ppo_loss.py` **112 staged checks in
  5.3 s**, `--fast` 109 in 1.5 s). **Headline: clipping bounds the STEP, never the WALK** —
  25 iterations at a mean clip fraction of **5.3%**, every step inside the trust region, and the
  policy still walked **4.68 nats** from `π_ref`, against **0.75** for a β-penalised run that
  plateaued (6.2×). Also: `grad L_IS` at θ=θ_old equals W3's score-function gradient
  **elementwise at 0.0** and the enumerated finite difference at 1.9e-11; the β frontier's three
  secant slopes each **bracket their β pair** and fall as KL grows (concavity measured);
  `σ(T) = σ₁√T` verified to **1.1%** ⇒ at T=500 a 1024-sequence batch has ESS **0.38**.
  Correction to the plan: clipping makes the update DIRECTION worse (43.8° vs 11.8° at K=64)
  while cutting KL travelled **29.3×** — it protects step size, not direction.
- Day-1 interview drills were run on D2 (0.6 / 0.6 — structure right, **zero measured numbers**
  on the second). D2's own two drills are carried to D3's recall quiz at his request, so they
  are asked cold a day later.
- Current day: Week 4 Day 03 done (token-level PPO + learned value head; `code/token_ppo.py`
  **144 staged checks in 7.6 s**, `--fast` 144 in 6.9 s). **Headline: a critic cannot bias you
  as a BASELINE, only as a BOOTSTRAP** — at λ=1 *every* critic (zero, pure noise, anti-correlated,
  V\*+25) reproduces the enumerated finite-difference gradient to **2.8e-10**; at λ=0.5 the same
  zero critic is **0.796** off. Bias tracks the state-VARYING error: Spearman(bias, ‖ẽ‖) =
  **+0.995**, Spearman(bias, EV) = **−0.995**, and a real regression's error is almost entirely
  the expensive kind (‖ẽ‖ ≈ ‖e‖ at every checkpoint).
  **Correction to the plan, and the day's best result:** `session.md` claimed `V+c` shows
  near-zero bias generally. It shifts advantages by exactly `−c·λ^(L−1−t)` (asserted to 3.6e-15),
  so with EOS SUPPRESSED a wrong LEVEL costs **1.5e-10** (the FD floor — literally nothing) while
  a wrong RANKING of identical ‖e‖ costs **0.97**; with EOS live the same level error costs
  **0.447**. "Wrong level is free" is a statement about the BASELINE role — under bootstrapping it
  needs FIXED length, and an LLM never has that. Also: per-token KL shaping = **0.405113796560**
  vs `β·KL_seq` = **0.405113796560** (1e-11, both k1 and k3); EV converges to an analytic ceiling
  of **0.4280**, not 1; zero-baseline probe reads **−42.6%** (exact V\*) / **+128.1%**
  (anti-critic); c₁ is **non-monotone** (neutral at 0.1–0.5, keeps only **2.5%** of the gain at 5).
- D2's two interview drills were carried into D3's recall quiz and asked cold as planned.
- Next: **Week 4 Day 4 — RLHF failure modes**: the D1 reward model gets over-optimized on purpose
  (Goodhart hump from finite noisy preference data), off-policy-by-accident finally gets MEASURED
  against exact ∇J, and length exploitation shows up because the env has an EOS token. D3 already
  seeded all three: the rare-state tail (critic still off by **0.942** where occupancy < 1e-4,
  which is exactly where an over-optimizing policy goes), `reduce="mean"` moving weight on len≤2
  from **0.3371 → 0.5522**, and level-error-costs-only-with-EOS.

### Diagnostic (Day 01)

- Top gaps: (1) ML systems / memory-cost reasoning, (2) interview-crisp RL derivations.
- Fluency-probe baseline: 5/5 in ~20 min (slow on `gather` + masking); target <10 min.

## Weekly Progress

| Week | Theme | Status | Deliverable |
|---|---|---|---|
| 01 | LM foundations + PyTorch | Complete (5/5) | Tiny LM train/eval/ablation harness |
| 02 | Transformers + generation | Complete (5/5) | Decoder-only mini-impl + debug suite |
| 03 | RL foundations | Complete (5/5) | Toy policy-gradient implementation (213 checks) |
| 04 | RLHF + PPO | In progress (3/5) | PPO/RLHF explainer notebook |
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
- [x] EGLP lemma E_{a∼π}[∇log π]=0 ⇒ state-only baseline is unbiased (produced COLD and unprompted on W3 D3, 1.0)
- [x] Policy gradient theorem (log-derivative trick; why P & ρ₀ drop out; reward-to-go causality) (W3 D2)
- [x] Reward-to-go causality proof COLD: tower rule on h_t → past reward is h_t-measurable → EGLP kills the inner term (W3 D3 quiz 0.8)
- [x] The Ψ_t ladder (R(τ) → G_t → G_t−b → Q^π → A^π → δ_t → GAE) with bias/variance for each (W3 D3, 7/7 rungs + bias column; ordering principle still needs both mechanisms named)
- [ ] PG regularity conditions: ∇/Σ swap; **support of p_θ must not depend on θ** (top-p violates it); R(τ) not θ-dependent (D3 quiz gap)
- [ ] Score-function vs pathwise/reparameterization; why RL is forced into SF (discrete actions + black-box reward)
- [ ] Six sources of REINFORCE variance + which technique kills each (1/6 cold on W3 D2, 3/5
  re-drill, ~2.5/6 on D3, 4/7 on D4, **0.2 on W4 D1 (filed Rao-Blackwellization as a source)**,
  **6 sources at 0.85 on W4 D2 — best yet, and RM noise finally present**. Discriminator that
  fixed it: *a source is something you would still have if you never wrote any variance-reduction
  code.* Still missing: the policy's own ENTROPY; ENVIRONMENT stochasticity (worth naming because
  it is ZERO for LLMs — deterministic append); OFF-POLICY-ness itself. And the kill for RM noise is
  NOT "use expectation" — it is ensembling / reward whitening / multiple samples per prompt)
- [ ] `Ψ_t` need NOT be `h_t`-measurable — that requirement is on the **subtracted baseline** only. A state-only `Ψ` gives an identically zero gradient by EGLP, so non-measurability of `G_t` is the point, not a defect (D4 question — he applied the baseline test to the signal)
- [ ] Why the policy gradient carries `γ^t`: it is the leftover from re-basing `Σ_{t'≥t} γ^{t'} r_{t'}` into `γ^t G_t`; the textbook `d^π_γ` form absorbs it into the state measure. PPO/GRPO drop it (D4 question)
- [ ] `Σ_t` in a policy gradient always traces back to the product over timesteps in `p_θ(τ)` — log of a product; `ρ₀` and `P` vanish because they are θ-free (D4 question — reproducible on demand but not yet automatic)
- [ ] Off-policy-by-accident (W3 D3 0.7 → W4 D1 0.65 → **W4 D2 0.72**): support-vs-reweighting now
  clean and unprompted (temperature reweights, top-p TRUNCATES); the per-token half now lands too.
  **Still owed: the FIX ORDER.** (1) sample at T=1.0 untruncated — the only actual fix; (2) IS-weight
  by `pi_theta/q` using the ACTUAL sampler — works for temperature, **provably fails for top-p
  because IS needs `supp(pi_theta) ⊆ supp(q)` and truncation violates it in the wrong direction: no
  estimator recovers mass you never observed**; (3) mitigation only (clip, short reuse). Keep the
  three clip-like things separate: gradient-NORM clipping (optimizer), PPO RATIO clipping
  (deliberate off-policy from reuse), top-p TRUNCATION (accidental off-policy from sampling)
- [x] A valid baseline must be h_t-measurable; an episode-internal G_t mean is BIASED (W3 D3: measured, mean 38% too small)
- [x] Measurability = "if I tell you h_t you can tell me X"; it (not θ-dependence) is the discriminator (W3 D3)
- [x] Action-dependent baseline b=Q^π DELETES the gradient (mean exactly 0), it doesn't merely bias it (W3 D3)
- [x] Optimal baseline b* = score-magnitude-weighted mean return; V^π is the unweighted one (W3 D3, measured 1.476 vs 1.765)
- [x] **Unbiased ≠ variance-reducing**: a horizon-mismatched V^π is unbiased but +27% variance = the unconverged-PPO-critic failure mode (W3 D4 re-drill 0.8 — got "not biased" + the measurability argument cold; still owes the diagnostic + decisive experiment + his own numbers)
- [ ] Mismatched-critic SECOND HALF: diagnostic = critic **explained variance**; decisive experiment = **zero the baseline, re-measure variance on the same rollouts** (D4 B3 scored 0.1 — answered "compare to ground-truth gradients", which a real stack does not have)
- [x] GAE = geometric mixture of k-step rungs; telescoping `Σ γ^l δ_{t+l} = G_t − V_t(s_t)` assumes NOTHING about V; reverse scan + `dones` reset (W3 D4)
- [ ] **Bootstrap bias = the STATE-VARYING part of the critic's error only.** `e` enters δ as `γe(s′)−e(s)`, so a uniform offset is a constant = a baseline = free. Measured: wrong-LEVEL critic (off by +2.174 everywhere) bias 0.0010 at λ=0; wrong-RANKING critic (±1 per state) bias 1.5557. **Wrong level → variance; wrong ranking → bias.**
- [ ] With an EXACT critic every λ is unbiased (the `l≥1` terms are zero-mean) ⇒ λ=0 strictly wins; **λ>0 exists only as a hedge against a critic you don't trust**. Interior MSE optimum appears only when the critic is wrong (λ*=0.9 measured)
- [ ] γ vs λ: **γ is part of the problem** (changes J, V, the optimal policy, the horizon); **λ is part of the estimator** (target identical, only bias/variance move). Schulman: γ biases regardless of critic accuracy, λ only when the critic is inaccurate (D4 Part 3 gap)
- [ ] Time-indexed `V_t(s)` is the value function a finite-horizon objective actually has (`T−t` Bellman backups, `V_T=0`); beats pooled `b(s)` by 1.55× and matches the exact oracle (1.142 vs 1.147)
- [ ] GRPO group mean = (K−1)/K × RLOO ⇒ unbiased in direction; the STD division is the real bias + degenerate-group blowup (Dr. GRPO)
- [ ] Entropy bonus: ∂H/∂z_k = −p_k(log p_k+H) DECAYS as π sharpens ⇒ preventative not cure; needs β~O(1); KL-to-ref is the LLM regularizer (D3 drill 0.5 — no mechanism, didn't cite own sweep)
- [x] **An entropy bonus IS a KL penalty to a UNIFORM reference**: `KL(π‖Unif) = log A − H(π)` (asserted by identity in W3 D5). RLHF's one change is replacing "uniform" with "known-good"
- [x] KL-regularized optimum `π*(a|s) = π_ref e^{r/β}/Z`, `J* = β log Z` = a log-sum-exp / soft max at temperature β (W3 D5 derivation, first try)
- [x] `dJ*/dβ = −KL(π*‖π_ref)` (envelope theorem) and **`dE[r]/dKL = β`** — β is the exchange rate in reward-per-nat, and the frontier is concave in KL (W3 D5, rtol 1e-4)
- [ ] **β limits, COLD** (D5 gap): `β→∞ ⇒ π_ref`, NOT uniform — uniform is the entropy-bonus special case; `β→0 ⇒` uniform over `argmax{r(s,a) : a ∈ supp(π_ref)}` — the argmax is over the SUPPORT
- [ ] **Reverse KL confines support** (`supp(π_θ) ⊆ supp(π_ref)` at every β, asserted exactly at β=2/0.5/0.01) ⇒ RLHF ELICITS, cannot teach ⇒ SFT must precede RL ⇒ top-p rollouts are unrepairable (support, not reweighting). Needed 2 prompts to land
- [ ] Reverse KL is mode-seeking / forward is mass-covering: measured low-reward-mode mass ref 0.500 → forward 0.095 → reverse 0.011 (8.6×); forward puts 0.775 on an action the reference never emits
- [ ] k1/k2/k3 KL estimators: k1 unbiased but negative on 20-44% of samples; k2 biased and the bias is INVISIBLE at small KL; k3 unbiased + ≥0 pointwise. **var(k3)/var(k1) crosses over: 423× better at KL=0.004, 4.4× WORSE at KL=0.65** (ρ is a likelihood ratio)
- [ ] Per-token KL as reward shaping == the sequence KL: `E[Σ_t penalty] = β·KL(p_θ(τ)‖p_ref(τ)) = β Σ_t E_{d_t}[KL_t]` (ρ₀ and P cancel — same reason PG is model-free). Measured 0.77787 vs 0.78974 exact. k3 shaping: same mean, 1.47× less variance, and k1 pays a BONUS on 24.5% of tokens
- [ ] Over-optimization = Goodhart: proxy monotone in 1/β while true reward peaks interior (1.1286 @ β=0.16) then falls BELOW the reference (−2.3250 vs 0.0270). Negative control: tilting by the TRUE reward is monotone (D5 drill 0.35 — did not cite the numbers)
- [x] **Bradley-Terry NLL + gradient `sigma(-delta)`**: gradient magnitude is monotone
  decreasing in how right the model already is (→0 confidently correct, exactly 1/2 at a tie,
  →1 confidently wrong) ⇒ the fit is dominated by near-ties and misrankings, which is exactly
  where annotators disagree ⇒ RM accuracy has a noise ceiling in the 70s (W4 D1, derivation 0.8)
- [x] **RM identified only up to a per-prompt constant `c(x)`** — hypothesis consumed: both
  completions in a pair share the same prompt. Verified by refitting from a different init
  (centered agree 8.2e-15, raw differ 1.74). Three consequences: cross-prompt scores
  incomparable (0.445 = worse than chance), GRPO's per-prompt group mean is free, per-prompt
  centering is the only safe normalization (W4 D1)
- [ ] **The KL penalty is a CONFIDENCE REGION for the reward model**, not a regularizer:
  `pi_ref = pi_SFT` = the RM's own sampling distribution, so KL literally measures distance from
  the RM's training data. Measured 0.177 → 4.89 (27.6×) along a KL ladder. ⇒ beta is a property
  of the RM's DATA (better RM earns a wider trust region ⇒ retune beta after improving the RM;
  or retrain the RM on current-policy samples = iterated RLHF) (W4 D1 Part 3 — was "not sure")
- [ ] Over-estimation is the asymmetry that matters: under-estimates are harmless, over-estimates
  are SOUGHT by an argmax. Collapses 86× from 300 → 20 000 comparisons (W4 D1)
- [ ] Four RLHF networks + permissions: only `pi_theta` can WRITE; 7B bill 112/112/14/14 = 252 GB
  naive → ~126 GB by sharing the trunk and caching reference logprobs at rollout time (they never
  change); LoRA makes `pi_ref` "the same weights with the adapter off". PPO is a training job AND
  an inference job in one loop, which is the real source of RLHF infra complexity (W4 D1)
- [ ] All `C(K,2)` comparisons from one prompt must share a batch: statistical (else each
  completion is seen K−1 times/epoch ⇒ overfits after one epoch) AND compute (K forwards serve
  all pairs ⇒ ~(K−1)× cheaper) (W4 D1)
- [x] **`grad L_IS` at θ=θ_old IS the Week-3 policy gradient** (ρ=1 is an EVALUATION POINT, not an
  assumption; consumed in exactly one place — the ρ_t factor multiplying ∇log π). ⇒ PPO is
  REINFORCE-with-advantages plus permission to keep going. Asserted elementwise at 0.0 vs W3's
  estimator and 1.9e-11 vs the enumerated FD (W4 D2, derivation 0.9)
- [x] **The 2×2 clip table** (value / gradient / intent per cell), gradient EXACTLY 0 in the two
  clipped cells, one-sided per cell, `L_CLIP ≤ L_IS` pointwise by cases (W4 D2, 0.85 — the only
  slip was naming TWO unbounded cells; it is only `A<0, rho>1+eps`, which dual-clip patches)
- [ ] **Clipping bounds the STEP, never the WALK — the three gaps, one per level of aggregation:**
  (i) per TOKEN (T tokens at the boundary ⇒ sequence KL `T eps²/2`, 10 nats at T=500, eps=0.2);
  (ii) per SAMPLE and only where the clip BINDS (says nothing about `E[KL]`); (iii) relative to
  `pi_old`, which RESETS every iteration ⇒ N bounded steps in one direction is an unbounded walk.
  **Measured: 4.68 nats of drift at a 5.3% clip fraction vs 0.75 for the β run (6.2×).** He had
  1 of 3 on D2 (and his (i) was really the soft-vs-hard point) — RE-ASK
- [ ] **PPO approximates a constraint by DELETING THE GRADIENT, not by enforcing a bound.**
  Nothing stops rho from being 5; the clip only declines to pay for the trip. "Clipping keeps the
  policy close to the old one" is FALSE as stated — that is the discriminating interview follow-up
- [ ] **Clipping protects step SIZE, not DIRECTION** (measured: 43.8° vs 11.8° angle to the true
  ∇J at K=64, while cutting KL travelled 29.3×). Those degrees ARE the deliberate bias. The
  no-clip run also scores HIGHER J on the same batch = Goodhart from the inside
- [ ] **Why per-token and not per-sequence ratios**: `prod_t rho_t` is log-normal, so mean is
  EXACTLY 1 (likelihood ratio) while the median collapses. `sigma(T) = sigma_1 sqrt(T)` verified
  to 1.1%, `ESS/n = exp(-sigma²)` to 1.6%, `ESS/n = 1/(1+chi²)` exactly ⇒ **at T=500 a
  1024-sequence batch has an effective size of 0.38**, with 88.5% of tokens inside the eps=0.2
  band. Cost of the per-token choice: it is no longer an unbiased IS estimator of the
  sequence-level objective — same trade as clipping
- [ ] **The PPO loss curve is uninformative BY CONSTRUCTION** (`L_IS(theta_old) = E[A] ≈ 0`), and
  the residual is `sum_b w_b len(y_b) A_b` — nonzero only through LENGTH/advantage covariance
  (measured +0.010993; exactly 0 in a one-step bandit). Read clip fraction, `KL(pi||pi_ref)`,
  entropy, explained variance
- [ ] **`clip_frac` ≠ `oob_frac`** (0.237 vs 0.570 measured): every out-of-band token is a
  candidate, only those whose sign of A points further out lose the gradient. `frac_high` rising =
  trying to escape upward; `frac_low` rising = entropy collapse. Equal ⇒ you clipped both sides
- [ ] **TRPO constrains `KL(pi_old||pi_theta)` — FORWARD, mass-covering** ("don't abandon what the
  old policy did"); InstructGPT penalises `KL(pi_theta||pi_ref)` — REVERSE, support-confining.
  Different objects; a real interview contrast
- [x] **`dE[r]/dKL = +beta` at the optimum, and the frontier is CONCAVE** — measured on four
  β-plateaus: every secant slope brackets its β pair (0.793∈[0.7,1.0], 1.203∈[1.0,1.5],
  1.892∈[1.5,2.5]) and the slopes FALL as KL grows. Concavity argument: KL is convex in π ⇒
  `{KL ≤ κ}` is convex and growing ⇒ R(κ) is a max of a linear functional over it = a perturbation
  function. β is a PRICE, and the price falls as you buy more nats (W4 D2 — sign fixed, shape was
  the gap)
- [ ] **The KL penalty is REWARD SHAPING, not a loss term**: `r(y) - beta[log p_old(y) - log
  p_ref(y)]` applied before the advantages ⇒ no estimator knows it exists (consistent with "the KL
  term is not a Psi — it modifies r"). Uses only rollout-time quantities ⇒ `pi_ref` can be dropped
  after rollout. Stale after step 1, exactly like the ratio
- [ ] DPO derivation — `r = β log(π*/π_ref) + β log Z`; `Z(x)` intractable because the action is a whole COMPLETION (`|V|^T`), not because logits are unavailable; it depends on the PROMPT only ⇒ cancels pairwise (W3 D5 Part 4, seen unprompted)
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
