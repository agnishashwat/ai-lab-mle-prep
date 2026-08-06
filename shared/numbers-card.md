# The Numbers Card

**How to use this.** Retrieval has failed seven consecutive sessions under a diagnostic
framing while derivation and mechanism recall are fine. So stop trying to retrieve.
**Read this out loud at the START of every recall block**, then answer. Recognition works
where retrieval does not. Add a line at the end of every session; never round in your favour.

Format: `quantity = value (condition)`.

---

## Week 1-2 — LM foundations, transformers, inference

- bf16 + Adam trainable = **16 bytes/param**; 7B ⇒ **112 GB** before activations
- frozen inference-only = **2 bytes/param**
- KV cache = `2·L·n_kv·d_head·T·B·bytes` = **512 KiB/token**, **2 GiB @ 4k**, **64 GiB @ 8k×16**
  (vs **14 GB** of weights — the cache dwarfs the model)
- decode arithmetic intensity ≈ **1 op/byte** ⇒ memory-bound
- KV-cache decode speedup measured = **4.15×** @ N=200; **2.85×** in the debug lab
- temperature entropy ladder: **0 → 1.744 → 2.079** (= ln V)

## Week 3 — RL foundations

- variance ladder, identical rollouts: total_return **20.033** → reward-to-go **6.317** (3.17×)
  → +fitted baseline **1.765** (11.35×) → +`b*` **1.476** (13.57×) → +time-indexed `V_t`
  **1.142** (**17.5× cumulative**)
- oracle `V^π` baseline is unbiased but **increases** variance **27%** (7.769 vs 6.317)
- biased within-episode baseline: lowest variance (**0.784**, 25.56×) with mean **38% wrong**
- GAE: uniform critic offset **+2.174** biases by **0.0010**; ±1 state-varying skew biases by
  **1.5557** (~**1500×** worse) ⇒ wrong LEVEL → variance, wrong RANKING → bias
- entropy β sweep: β 0/0.5/2/5 → H **0.074/0.118/0.294/0.522**, return **7.966/7.925/7.634/6.375**
- reverse KL confines support: `π*[a] == 0.0` exactly; forward KL puts **0.775** there
- `var(k1)/var(k3)` = **423×** @ KL 0.004, **19.9×** @ 0.065, **3.0×** @ 0.23, **0.23×** @ ~0.65
  ⇒ **it REVERSES**; k3's win is a small-KL phenomenon

## Week 4 — RLHF & PPO

- off-policy-by-accident vs exact ∇J: on-policy **2.95e-10**, temp 0.7 **0.2284**, temp 0.5
  **0.4466**, top-p 0.9 **0.1169**, stacked **0.3127**
- IS repairs temperature to **2.953e-10**; **cannot** repair top-p (**0.1169 → 0.1267, WORSE**)
  because **7.2%** of π's mass is unreachable
- top-p ratchet: reachable sequences **40 → 39 → 18 → 17 → 12 → 9** across iterations
- clipping ≠ trust region: **token** `T·ε²/2` ≈ **10 nats** @ T=500; **sample** clip_frac
  **0.237** vs oob_frac **0.570**; **iteration** **4.68 nats @ 5.3% clip fraction** vs **0.75**
  with a penalty (**6.2×**) — π_old RESETS
- `∏ρ_t` log-normal: `σ(T)=σ₁√T` to 1.1%, `ESS/n = exp(−σ²)` to 1.6%; **T=500, batch 1024
  ⇒ ESS 0.38**; **88.5%** of tokens inside the ε=0.2 band
- `dE[r]/dKL = β`, frontier **concave in KL**: secants **0.793∈[0.7,1.0]**, **1.203∈[1.0,1.5]**,
  **1.892∈[1.5,2.5]**; concave at **100%** of secant slopes
- adaptive β fixed point vs `dE[r]/dKL`: **5.00% / 1.05% / 2.27% / 4.95%** at δ = 0.25/0.5/1.0/2.0
- dual-ascent caveat measured: exact primal **β=0.6173** vs PPO primal **β=0.1563** at the same
  KL 0.8646 ⇒ **74.7% gap**
- chained regression REINFORCE→PPO elementwise to **7.749e-15**; EGLP check non-vacuous by **1.3e11×**
- ablation grid (7 arms × 3 seeds): π_ref gold **4.4191**; full **5.1501 ±0.109**;
  **no_kl gold 4.9098 with proxy 0.8728** (proxy UP, gold DOWN); reinforce **4.7112**;
  top-p **4.8559**; **no_clip ≈ full (5.1508)** — the PENALTY was binding, not the clip;
  **no_critic BEAT full: 5.1786 ±0.036 vs 5.1501 ±0.109** ⇒ the critic's value is a function
  of HORIZON, and this horizon is 5
- early stopping: gold peaks **5.1894 @ iter 8 (KL 0.503)**, ends **3.9372 @ KL 3.605** —
  below π_ref's **4.4191**. Shipping the last checkpoint costs **1.2522**
- Goodhart is a VARIANCE result: seed spread amplifies **89× / 259× / 667×**; at 30 comparisons
  spread **±1.60** exceeds the entire mean improvement **+0.95**; max spurious RM weight
  **32.07 → 4.29**
- RM error along a KL ladder: **0.177 → 4.89** (**27.6×**) with the RM unchanged
- length: confound **κ = +0.6276** vs bounded per-token resistance **max 0.1411** ⇒ **no β stops
  it**; weak control κ **0.0504 < 0.2205** ⇒ stopped at 3.95
- gold reward with a **−0.4/token** explicit penalty has marginal slope **+0.1751** ⇒ win rate
  must be length-CONTROLLED
- deadly triad scalar `A`: **−3.70 converges**, **+1.07 diverges to 4.4e13**, EV **−1.9e27**
- `diagnose` fired **value_divergence (EV → −5.19e9)** on the top-p arm unprompted
- silent-bug probe: `std(log ρ)` = **0.098999** on a real 4-epoch update, exactly **0** if
  `logp_old` is recomputed in the loop; `clip_frac` reads **0.000** under BOTH hypotheses
- entropy collapse caught at KL **0.30**, length exploitation at KL **0.55** ⇒ **KL is
  NECESSARY, not SUFFICIENT**

## Week 5 — DPO & preference optimization

*(D1 values are from `week-05-.../code/preference_models.py`, **74 checks**, 38 s.)*

- **existence.** 8 completions, strict ordering. Deterministic annotator, all 28 pairs ⇒
  wins-digraph **not strongly connected**. Fit re-centred every step, so the location gauge
  is fixed and the divergence is not the additive constant:
  ‖r̂‖ = **6.22 / 14.42 / 17.77 / 26.80 / 31.03 / 41.21 / 45.66** at t = 100 … 2e5.
  Monotone, unbounded. nll **2.086e-04**, pairwise accuracy **exactly 1.0**, max margin **49.97**
- **the contrast.** *Same items, same 28 comparisons*, labels flipped at ε=0.10 ⇒ **strongly
  connected**, ‖r̂‖ **converges to 4.6967**. Relative change over the last decade:
  clean **1.081e-01** (still moving) vs noisy **0.000e+00** (converged). Norms **45.66 vs 4.70**
- **why it is invisible.** cos(r̂_t, r̂_2t) = **0.9999873 / 0.9999938 / 0.9999985** at the
  1e3→2e3, 1e4→2e4 and 1e5→2e5 decades, while ‖r̂_2t‖/‖r̂_t‖ = **1.2321 / 1.1580 / 1.1081** ⇒
  direction converges to the max-margin separator, norm grows logarithmically (Soudry et al.).
  Nothing errors; accuracy reads 100%. **Plot ‖r̂‖ against steps, not the loss**
- **the DPO shape.** Perfect matching (each completion appears once): **not strongly
  connected**; ‖r̂‖ **8.78 → 12.04 → 15.30**, **min r̂_i = −3.10 → −4.26 → −5.41**, min margin
  **6.21 → 8.52 → 10.82**. *Flipping every label does not create a cycle* — in a matching,
  annotator noise has nowhere to act
- **the ε-cap, saturated** (2 items, exact win counts): fitted margin == `logit(1−ε)` to
  **1e-14**. **3.8918 / 2.9444 / 2.1972 / 1.3863 / 0.6190** at ε = .02/.05/.10/.20/.35
- **the ε-cap, parametric** (8 items, 28 pairs — BT is *misspecified* under flip noise, so
  the fit compromises): spread **12.29 / 7.49 / 5.00 / 2.76 / 1.22**, monotone decreasing.
  ε → 0 *is* the transitive tournament ⇒ the two results are one result
- **scale ≡ β.** `tilt(c·r, c·β) == tilt(r, β)` to **3.9e-16**; location free in both the tilt
  and the NLL to **<1e-15**; support confinement exact (`π*[a] == 0.0`). Consequence, **same
  β=0.5**, only the reward's scale changing ×0.5/1/2/4 ⇒ KL = **1.0156 / 1.8047 / 2.6198 / 2.8802**
- **Plackett–Luce at K=2 == Bradley-Terry to 0.000e+00**
- **the accuracy ceiling.** Bayes ceiling **0.7675** (n=11200, SE **0.0040**); the *gold* reward
  itself scores **0.7747** and the fitted RM scores **0.7747** — it reaches Bayes and not a hair
  more. ⇒ the ceiling is set by the ANNOTATOR. Accuracy depends only on the SIGN of the margin,
  so it is scale-invariant and cannot detect a diverging fit
- (D3, algebra) cDPO optimal implicit margin = `logit(1−ε)/β` — **29.4444 / 21.9722 / 10.9861**
  at β=0.1, ε = .05/.10/.25
- **useful conversions**: `logit(0.68) = 0.7538`, `logit(0.91) = 2.3137` ⇒ moving inter-annotator
  agreement 68% → 91% is a **3.07×** larger achievable margin at fixed β
- **the asymmetric probe for a sign flip** (W5 D1): 80/20 wins ⇒ MLE margin `logit(0.8)` =
  **1.3863**; a gradient-ASCENT bug returns **−317.66**. On separable data both signs give
  ‖r‖ → ∞ and are indistinguishable — use data whose MLE is finite and known
- **the scrambled fixture's own baseline**: raw column 0 of `all_pairs` has gold accuracy
  **0.571** ⇒ "return column 0" is measurably wrong, so the `noisy_labels` check is non-vacuous

### D2 — deriving DPO (`code/dpo.py`, **42 checks**, 24 s; 15 sequences, 7 states, β=0.5)

- **THE EQUIVALENCE.** `train_dpo` == `bt_then_tilt` **elementwise to 7.772e-16** over all 15
  sequences — two algorithms sharing no code path. Wrong-β tilts differ by
  **6.15e-02 / 8.25e-02 / 1.63e-01** at β = 0.25/1.0/2.0 ⇒ **non-vacuous by 2.10e+14**
- **the gauge.** `Σ_y π_ref e^{r̂/β} = 1.000000000000000` identically; `r̂ = r − β log Z(x)` to
  **4.441e-16** with offset `β log Z = +1.425614`. Implicit reward == BT MLE **after** centering
  (**1.332e-15**) and **not** before (**0.7454**, = the offset). ⇒ the simplex FIXES the gauge,
  so you cannot lower `r̂(y_l)` without raising `r̂` elsewhere
- **the gradient.** FD **1.161e-10**; loss at θ=θ_ref is **exactly +log 2 = 0.693147**;
  `‖g‖ = β·σ(−M)·‖score_w − score_l‖` to **3.331e-16**; best-ranked decile weight **0.049334**
  vs worst **0.951741** = **19.3×** ⇒ mislabels get weight → 1, the MAXIMUM in the batch
- **π_ref changes the gradient's SCALE, not its DIRECTION**: two references with weights
  **0.448705 vs 0.673414** give gradients equal after dividing by their weights to **2.776e-17**
- **divergence on a matching** (7 pairs): accuracy hits **1.0 at step 2** with KL **0.1129**;
  step 20 KL **0.8486**; step 20 000 KL **1.4660**, `log π(y_l) = −25.9537`, loss **4.834e-05**,
  losers' mass **0.343396 → 1.649e-09**. **KL is BOUNDED by max_y −log π_ref(y) = 4.6894** (the
  point-mass bound) — "grows without bound" is FALSE in a finite space
- **both logprobs fall AT THE CONVERGED OPTIMUM** (strongly-connected fixture): margin
  **0 → +0.6155**, mean log π(y_w) **−3.1249 → −4.0003 (−0.8754)**, log π(y_l) **−3.3174 →
  −5.4238**. Not a divergence artifact
- **the μ ≠ π_ref tax is paid in KL**: same 8 sequences, columns swapped ⇒ winners in π_ref's
  bulk KL **0.6640** vs tail KL **3.3932** = **5.11×**, at margins **9.1364 vs 8.3982** and
  comparable loss. The SIGN of Δlog π(y_w) is a transient (positive on `tail` until ~step 5000)
  — |Δw|/|Δl| = **0.0941 / 0.0056** is the stable statement
- **shared prefixes cancel EXACTLY** (0.000e+00 at the root state): in a TABLE the shared prefix
  of a pair contributes nothing to the DPO gradient ⇒ "y_l's pushdown drags y_w down" needs
  shared PARAMETERS, not shared tokens
- at θ = π_ref the implicit reward is **exactly 0.0** everywhere ⇒ every pair starts at weight
  **σ(0) = 0.5**. The ratio does NOT start large; the ratio the LABELS require is what is large

### D3 — implementation details (`code/dpo_trainer.py`, **51 checks**, TinyGPT)

- **cDPO finite optimum at β=0.3:** `Δ̂*=logit(1−ε)/β` =
  **9.814797 / 7.324082 / 4.620981 / 2.063464** at ε = .05/.10/.20/.35; gradient at each
  analytic optimum below **3e-17**, curvature positive
- **length prior:** constant per-token drift +0.2 ⇒ sum margin-vs-length slope
  **+0.200000**, mean slope **0.000000**. Sum asks `d_new>0`; mean asks `d_new>d̄`
- **reference cache:** chosen + rejected = **2 fp32 scalars/pair = 8 bytes/pair**; exactly
  **2 preprocessing forwards**; cached vs fresh **bitwise exact**; trainer reference calls **0**
- **precision:** at T=256, `|fp32−bf16 sum| = 1.026245` vs representative
  `β·margin = 0.050000` ⇒ numerical error **20.525× the signal**
- **SFT initialization:** token NLL **2.071305→0.396558**; chosen mean logprob/token
  **−2.081233→−0.474107**; clone warmed checkpoint into BOTH roles ⇒ initial log-ratios
  **exactly 0** and DPO NLL **+log 2 = 0.693147**
- **TinyGPT full run:** loss **0.693147→0.549205**; margin **0→4.3369**; `|r̂|`
  **0→0.7751**; exact subset KL **0→0.92182**; expected length **2.1227→2.3867**
- **chosen-logprob sign is not a diagnostic:** fast fixture Δchosen **−0.59155**,
  Δrejected **−2.06221**; full fixture Δchosen **+2.21402**, Δrejected **−2.12287**.
  Use Day 2's **5.11× KL tax at matched margin/loss** for μ≠π_ref
- **correct boundaries:** gradient no-op is **β→0**; ε=.5 means **zero-margin target**, not
  constant loss; fp32 naive form saturates at **z=+40** and underflows at **z=−100**
