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

### D4 — DPO variants (`code/variants.py`, **126 checks**, 48.8 s; 5×5 grid, β=0.5)

- **the grid:** `L = mean_k ℓ( f(y_w), f(y_l) ) [+ λ_SFT·mean_k −log π(y_w)/|y_w| ]`.
  5 rewards × 5 links = **25 cells**, 4 named. Grid == 5 direct implementations elementwise at
  **max dev 0.000e+00**. The **LINK** decides if the margin is finite; the **REWARD** decides
  where the length goes
- **separable (D1's matching, 8000 steps, lr 0.5, point-mass bound 6.3216):**
  DPO margin **5.9758**, KL **0.9511**; cDPO **2.1972 = ln 9** exact, KL **0.4704**;
  IPO **2.0000 = (2f−1)τ** exact, KL **0.2332**. **acc = 1.000 for all three** — the metric
  that reports nothing (saturates at step ~20, then 7 980 uninformative steps)
- **DPO's divergence signature is a CONSTANT increment per DOUBLING**, not % growth:
  **+0.7163** then **+0.7078** ⇒ `margin ~ log t`, the separable-logistic rate (a theorem about
  **GD** — Adam erases it while still printing a table)
- **four link optima:** `logit(f)` / `logit((f−ε)/(1−2ε))` / `(2f−1)τ` / **no closed form**.
  Only **2 of 5** are finite at f=1. Margin-DPO still diverges (**21.2233** at f=1−1e−9);
  naive `logit(f)+γ = 0.905465` vs **TRUTH 0.533227** (γ breaks BT's antisymmetry)
- **IPO's price is COMPRESSION, not a uniform cap:** strong/weak ratio **11.3330 → 4.9000**
  (**2.31×** squeeze), margins **4.5951/0.4055 → 1.9600/0.4000** at f=.99 vs .60
- **SimPO's 1/|y| AMPLIFIES length bias** at matched margin — MEAN moves E[|y|] further than
  SUM in **all four** cells: **+1.834→+2.210, +2.520→+2.672, −0.876→−1.135, −1.091→−1.196**.
  `d(r)/d(logπ) = β/|y|` is largest at |y|=1 ⇒ short seqs are the cheapest lever.
  **Run to convergence and both saturate at t_max, hiding it completely**
- **reference-freeness is a SEPARATE trade** (matched margin 3.0, equal-length pairs):
  noref KL **2.0042** vs ref **1.2091**; H **1.3131** vs **1.4627** (H(π_ref)=2.0657)
- **ORPO 2×2** (strongly-connected, 3000 steps): chosen NLL/tok **1.302213** baseline →
  DPO **1.209615**, DPO+SFT **1.042314**, ORPO **1.029723**, ORPO−noSFT **1.078406**.
  Decomposition: **odds −0.131210**, then SFT **−0.048683**. P2 measured directly as
  `Δlogp_l = −0.1161` under plain DPO
- **the odds answer to P2 FLIPS SIGN with β** (**−1.899, −0.457, +0.184, +0.385** at
  β=.25/.5/1/2) while **SFT holds at every β** (**+0.566, +0.528, +0.522, +0.557**)
  ⇒ run every ablation at more than one hyperparameter before naming a mechanism
- **common-shift invariance (c=+0.37):** ref_sum **5.55e-17**, noref_sum **0.00e+00**,
  ref_mean **1.2333e-01** `= β·c·(1/|y_w| − 1/|y_l|)` exactly, odds **2.1883e-01**.
  4 of 5 links are functions of `d` alone (**2.220e-16**) — that invariance **IS** P2, and it
  is why they all require pairs. Only `prospect` sees the marginals (**5.800e-01**)
- **KTO's bounded loss does NOT fix P1.** `prospect ∈ [0, λ_D+λ_U] = [0, 2.5]` at |r|=1e6 vs
  logsigmoid's **2e+06** — but **inf σ(−u) = 0 only at u=+∞**, so KTO climbs **+0.9294** per
  doubling next to DPO's **+0.7207** — *faster*. **A bounded loss is not a finite minimiser**
- **what boundedness buys is bounded INFLUENCE:** at d=−20, `|grad| −log σ` = **0.9999999979**
  (saturates at 1, never lets go) vs `|grad| prospect` = **2.061e-09** (abandoned).
  4/40 labels flipped ⇒ DPO clean margin falls **1.2411** vs KTO **0.6684**; pull toward the
  wrong labels **+2.8343** vs **+1.5870**. **KTO answers D3's NOISE problem, not P1**
- **dropping pairs is EXACT:** `KTO(unpaired) == 0.5 × mean(prospect link)` =
  **0.475211814705432** both sides. The 2 is 2n examples vs n pairs; prospect was already a sum
  of two independent per-example terms
- **z_ref's roll is a NO-OP without prompt conditioning:** on TabEnv a roll is a **permutation**
  and the mean is permutation-invariant (**−0.61677074380460184** vs **−0.61677074380460173**).
  On `PromptEnv` the gap is **0.020983** and the multisets differ. Load-bearing only when the
  reward depends on the **PAIRING**
- **the empty cell** (`ref_mean × squared_target`, no paper): both squared cells stop dead at
  the **identical margin 1.619048** (a free matched-margin comparison) while both logsigmoid
  cells climb (**+1.1744**, **+0.9671**); at that matched margin mean E[|y|] **4.8287** vs sum
  **3.8282**. Inherits τ's cap from IPO and the length amplification from SimPO ⇒ **the grid is
  a mechanism, not a taxonomy**
- **uniform π_ref is not a trust region:** under SUM it is reference-free **plus β·log V per
  token** = **0.549306** at V=3, **5.4125** at V=50257; under MEAN the constant cancels and it
  is bitwise reference-free (**1.110e-16**)

### D5 — DPO failure modes (`code/dpo_lab.py`, **92 checks**, 65.6 s / 34 s fast; `TabEnv(3,3)` = 15 seqs, β=0.5, lr=0.5)

- **THE THESIS:** the four modes are consequences of the **data format**, not of the algorithm.
  The loss is **exactly constant** along every direction the data do not constrain
- **Ford, measured over 12 seeds — P(wins digraph strongly connected):**
  `matching` (deg 1) **0.00 / 0.00 / 0.00 / 0.00** at ε = 0 / .10 / .25 / .35;
  `cycle` (deg 2) **0.00** at every ε;
  `all_pairs` (deg 14) **0.00 → 0.58 → 0.83 → 1.00**.
  ⇒ **clean data are NEVER strongly connected at any degree** (a transitive tournament is a
  DAG), and **noise needs DENSITY to act on**. In a matching, **2 of 7 labels flipped and it
  was still a DAG** — annotator noise has nowhere to act. **A real DPO dataset is the top row**
- **more comparisons ⇒ SMALLER final margin** (ε=.10, 6/6 seeds):
  **2 pairs 6.0470 > 7 pairs 4.1916 > 105 pairs 1.6687** (seed 0). Density creates the cycles
  that pin the MLE down. *Step-by-step monotonicity holds in only 3/6 seeds — assert the
  ENDPOINT ordering, not the fragile version*
- **the β/ε COLLAPSE, confirmed.** Two structurally independent sweeps matched only through
  `Δ̂* = logit(1−ε)/β`: |KL diff| **9.59e-05 / 6.56e-05 / 2.78e-16 / 2.47e-04** at
  Δ̂* = 1.0986 / 2.1972 / 4.3944 / 8.7889. Non-trivial: KL itself varies **0.0711 → 0.6805**
- **NEW LAW — steps-to-target ∝ 1/β².** Steps to 0.995·M*: **76 / 307 / 1237 / 4978** at
  β = 2 / 1 / .5 / .25, ratios **4.04, 4.03, 4.02**. *Halving β costs 4× the compute, not 2×*
  (gradient scale ∝ β, distance ∝ 1/β ⇒ time ∝ 1/β²). Δ̂* only went up 8× while steps went 65×
- **acc = 1.000 at step 1 of 6000** while the margin runs **0.0245 → 5.8439**
- **M2 measured:** `log π(y_w)` **−3.1862 → −3.3715 (−0.1852)** AND `log π(y_l)`
  **−3.8037 → −4.2863 (−0.4826)** while **the margin rose +0.5947 and told you neither**.
  The SFT term fixes it: **logp_w −0.1852 (DPO) → +0.0213 (DPO+SFT)**
- **M4's blind spot:** with 6 of 15 sequences covered, π_θ puts **19.4%** of its mass where the
  loss is **exactly constant** (**49.83%** at init). **Off-data mass FALLS during training**
  (0.4983 → 0.1943) because pushing losers down moves mass onto covered winners ⇒ **a delta
  rule NEVER fires. Support is a LEVEL**
- **the proxy/gold inversion, with a CONTROL** (6001 steps):
  `uncovered (M4)` gold peaks **@0**, gives back **0.4313**;
  `length confound (M3)` peaks **@2**, gives back **1.2103**;
  `separable (M1)` peaks **@77**, gives back **0.1036**;
  **CONTROL healthy** peaks **@1270**, gives back **0.0000**. All broken arms at **acc 1.000**
- **THE SHARPEST LINE OF THE WEEK: M3 gold +0.7514 → −0.4532.** DPO training made the model
  **worse than the reference it started from**, at 100% train accuracy, loss falling monotonically
- **early stopping as an ESTIMATOR** (never sees gold): stops at **step 105**, where
  **margin 2.2050 vs cap 2.1972**. gold **selected 0.4963 | last 0.3300 | oracle 0.6737** ⇒
  recovers **0.1663 of the 0.3437** that shipping the last checkpoint costs.
  On clean data ε=0 ⇒ **cap = +inf ⇒ no natural stopping rule** — that IS M1
- **a healthy DPO+SFT run moves E[|y|] by +8.9%** ⇒ a length move ALONE is not a failure.
  M3's gate (elen rise > 0.25) has **2.8× air below (healthy 0.089)** and **4.4× above
  (fixture 1.110)**
- **step-0 anchors, exact:** loss = **log 2 = 0.693147**, KL = **0**, r̂ = **0** everywhere
  (π and π_ref seeded from the same checkpoint). Uniform-policy logprob anchor
  **log(1/15) = −2.7081**. Point-mass KL bound **6.3216**
- **`logp_w`/`logp_l` are PLAIN logprobs.** Logged as ratios, `logp_w − logp_l` equals
  `margin/β` to **1.39e-16** — the pair would be perfectly redundant with the margin it exists
  to check. Plain carries π_ref's own bias: **+0.617490** of extra signal
- **CORRECTION to W4 D4 — "seed spread grows with pressure" is a 3-SEED ARTIFACT.**
  Same runs: gold spread **6.09× at 3 seeds** vs **1.04× at 5 seeds**. On those same runs
  `margin` spread **contracts to exactly 0.000000** and `kl` spread grows **71×** (KL is
  measured against each seed's OWN π_ref). **≥3 seeds is a FLOOR**
- **pairing the seeds** (DataCfg.seed AND RunCfg.seed together) shrinks the worst KL spread
  **2.07×** (0.7511 → 0.3621) — *a change to the DESIGN, not to the code*
- **the metric decides whether an ablation has power.** Same runs, same seeds:
  KL effect **0.4038** vs worst spread **0.3621** ⇒ **intervals OVERLAP, "β lowers KL" is NOT
  supported**. margin effect **0.1201** vs spread **0.0282** ⇒ **4.3×, resolved**
- **the units decide the CONCLUSION.** Over β = 0.25 → 2.0: margin **0.4746 → 0.5947 (UP)**
  while **Δ̂ = margin/β 1.8984 → 0.2974 (DOWN)**. *Report the wrong one and you report the
  opposite result*
- **DPO ≡ tilt(BT MLE)** re-verified here: max |dev| **1.164e-13** over all 15 sequences,
  non-vacuous by **4.40e+13×**
