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
  **no\_critic COST NOTHING MEASURABLE: 5.1786 ±0.036 vs 5.1501 ±0.109** — effect **0.0285**
  vs `full`'s OWN spread **0.109** (**3.8×** the effect), and no\_critic's interval is
  *contained* in full's ⇒ **a NULL, not a win** (corrected W6 D1: "BEAT" violated this card's
  own non-overlapping-intervals rule, and the spread comparison is a 3-seed one — see the W5 D5
  6.09×-vs-1.04× artifact). The critic's value is a function of HORIZON, and this horizon is 5.
  **The NULL is the stronger argument for GRPO** — you do not delete a component that is earning
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

## Week 6 — GRPO & RL with verifiable rewards

*(All from `week-06-grpo-rlvr-reasoning/code/grpo.py`, **96 checks**, 5.9 s.)*

- **the dead-group law.** P(a group carries zero signal) = **p^G + (1−p)^G**. At G = 8:
  **0.78% / 5.77% / 43.05% / 66.34% / 92.27%** dead at p = 0.50 / 0.70 / 0.90 / 0.95 / 0.99.
  N_eff/N at p = 0.95 is **33.658%**. Raising the pass rate 0.90 → 0.95 kills a further
  **23.3%** of the batch — *improving the model degrades its own training signal*
- **the price of fixing it.** Keeping 90% of prompts alive at p = 0.95 needs **G = 45**
  (G = 44 gives **89.533%**; closed form ln(0.1)/ln(0.95) = **44.891**). GRPO trades
  training memory for **inference compute**, and the bill grows as the model improves
  - **CORRECTION (learner's, W6 D1; confirmed by measurement W6 D4).** The closed form
    as written, `ln(0.1)/ln(1−p)`, is **single-sided** — it counts only the all-FAIL
    tail. The honest criterion is `p^G + (1−p)^G ≤ 0.1`, i.e. the two-branch
    `G = ln(0.1)/ln(max(p, 1−p))`. It happens to be right at p = 0.95 because the
    easy tail is negligible there, and it is **wrong by a factor of 2 in the dead
    mass exactly at p = 1/2**, where it returns 4 instead of **5**. See W6 D4 for the
    `G*` table, where this recurs at precisely the one k whose set point is 1/2.
- **seen in a real run**, two fixtures differing only in the verifier: healthy
  **dead 0.0354 at pass 0.4963** vs saturated **dead 0.8042 at pass 0.9742** — **23×** more
  of the batch silent on the run with the *better* reward
- **the exact shrinkage.** E[ĝ] = **((G−1)/G)·∇J**, exactly at finite G. Measured
  **0.4994 / 0.7496 / 0.8755** at G = 2/4/8 (want 0.5/0.75/0.875), cos ≥ **0.999992** so the
  DIRECTION is untouched. LOO ratios **0.9987 / 0.9994 / 1.0005**. At **G = 1 the gradient
  is identically 0** — the defining property, not an edge case
- **LOO is a rescaling, not a variance reduction.** Â_loo = Â_centered · G/(G−1) *pathwise*
  (max dev **0.00e+00**), so Var(ĝ_loo) = (G/(G−1))²·Var(ĝ_centered) — *larger* absolutely,
  identical relatively. **LOO removes the shrinkage, NOT the G = 1 death**
- **the 1/std bias.** Binary reward, k of G: Â_succ = **+√((G−k)/k)**, Â_fail = **−√(k/(G−k))**
  (every G cancels). Group weight Σ|Â|: centered = **2G·p(1−p)** = the *variance* = Fisher
  information; normalized = **2G·√(p(1−p))** = the *std*. Amplification is exactly 1/std:
  **2.0000 / 2.3094 / 3.0237** at k = 4/2/1, so k=1 is boosted **1.5119×** over k=4 and the
  balanced:extreme ratio collapses **2.2857 → 1.5119**. *Control:* balanced still outweighs
  extreme under both (1.75<4.00, 5.29<8.00) — **the bias COMPRESSES, it does not invert**
- **annihilation is EXACT at finite G, not in expectation.** c(x) removed to max dev
  **0.000e+00 / 2.220e-16 / 1.210e-14 / 2.039e-10** for c = 0 / 1 / −37.5 / 1e6. A KL folded
  into the reward survives only as its within-group deviation: **0.035726 vs 0.6842 full =
  19.15× DELETED**; a uniform drift penalty is invisible to the gradient (**2.50e-16**).
  *This is why DeepSeekMath puts KL in the LOSS, outside the advantage*
- **k1 vs k3 is FEASIBILITY, not variance.** Both unbiased to 10 dp (**0.1517183046** vs
  **0.1517183046**). k1 = −ℓ is **negative on 50.0% of the probability mass**; k3 = e^ℓ−1−ℓ
  on **0.0%** (min **+6.130e-02**). A negative reading in a minimised loss *pays* the policy
  to leave π_ref. (var(k1)/var(k3) REVERSES: **423×** at KL 0.004, **0.23×** at ~0.65)
- **the clip is inert at μ=1**, exactly: bound-token fraction **0.000e+00** at every step,
  vs **0.2222** at μ=8 with the SAME lr/seed/task. ⇒ *GRPO at μ=1 is REINFORCE + a group
  baseline + a KL term in PPO's notation*
- **memory, 7B bf16+Adam** (16 B/param trainable, 2 frozen): PPO **252 GB**
  (112+112+14+14) → GRPO **140 GB** (−44.4%) → cached ref logprobs **126 GB** → RLVR, no RM
  **112 GB** (−55.6%)
- **the `token` length norm is a PER-TOKEN GRADIENT fact.** At ρ=1 with constant Â the
  `token` and `batch` reductions are equal *by construction*, so it must be asserted on
  ∂/∂logp: a token in a 3-long sequence is pushed **exactly 3.0×** less hard than one in a
  1-long sequence; under `batch` the ratio is **exactly 1.0**
- **diagnostic thresholds, chosen from the gap between DISTRIBUTIONS (8 seeds/fixture):**
  dead_frac **0.185 | 0.50 | 0.783**, kl **0.300 | 1.00 | 1.288**, entropy
  **0.121 | 0.50 | 0.701**. **0 of 32 fixture-seeds** disagree with the seed-0 verdict

### Week 6 Day 2 — RL with verifiable rewards

*(All from `week-06-grpo-rlvr-reasoning/code/rlvr.py`, **112 checks**, 5.9 s.)*

- **a verifier is a SET, not a rate.** `V ⊆ C` = SOUND (no false positives),
  `C ⊆ V` = COMPLETE (no false negatives). Same checker, same two arrays, two policies:
  precision **0.6720 → 0.0926** and gamma **0.1515 → 0.7820** once the policy tilts toward
  the responses it wrongly accepts. Mass-weighted gamma **0.1515** vs count-weighted
  **0.2222** — *"our verifier is 97% accurate" is not a well-formed sentence*
- **the soundness theorem.** `V ⊆ C` ⇒ `v_pass ≤ gold_pass` for **all 415 policies at every
  KL**, no scope condition (max gap **−2.000e-12**). If `V \ C ≠ ∅`, sup(v_pass − gold_pass)
  = **1.000000** for **1 false positive AND for 7** — reward hacking is a **support**
  property, not a rate property. Their eval accuracies under uniform are **93.3% vs 53.3%**,
  i.e. the eval statistic ranks them *opposite* to what matters under optimization
- **the selection law.** `p_obs = p(1−α) + (1−p)γ`;
  `P(hack | rewarded) = γ(1−p)/[p(1−α) + γ(1−p)]`. At a **fixed 1% FP rate**, α = 0.10:
  **1.0989% / 9.0909% / 35.2518% / 68.8581%** at p = 0.500 / 0.100 / 0.020 / 0.005, → 1 as
  p → 0. *Precision is a function of DIFFICULTY, at a fixed verifier*
- **conditioning on ALIVE does not move the rate** (**0.34902 vs 0.34902**) — what moves is
  CONCENTRATION. `P(the whole alive group's positive signal is a hack)` = **0.33020** at
  p = 0.02 vs **5.9568e-04** at p = 0.5 = **554×**, same 1% verifier in both runs.
  *(NOT 2.4e7× — that is the conditional-on-k comparison h¹ vs h⁴, not the marginal)*
- **the dead-group law generalises to a RÉNYI ENTROPY.**
  `P(dead) = Σ_j q_j^G = exp(−(G−1)·H_G(q))` over the reward's OUTPUT LEVELS. At p = 0.05,
  G = 8: binary **66.3420%** → split the failure mass to (0.05, 0.40, 0.55) → **0.9029%** =
  **73.5×** from ONE extra reward level. Buying the same with G alone needs **G = 91.77 → 92**,
  an **11.5×** rollout bill — *reward design and group size are the same knob*
- **the rule is SPLIT THE MODE, and it is asymmetric.** At p = 0.05 splitting the FAILURE
  mass gives 73.5×; splitting the SUCCESS mass gives **66.3420% → 66.3420%** (nothing).
  Exactly reversed at p = 0.95. **P(dead) = 1 exactly at G = 1**, for any reward
- **false negatives are a COMPUTE bug, priced exactly.** A 30% extraction failure moves
  p_obs **0.10 → 0.0700**; G for 90% of groups alive goes **22 → 32** = **+45.5% rollout
  compute**, bought by a regex and paid in GPU-hours. A 5% FP rate instead takes **32 → 19**
  — *unsoundness BUYS compute, which is why the cheap fix is the dangerous one*
- **composite rewards SELF-ANNEAL.** `E[Σ_i |Â_i|] = 2(G−1)·c(1−c)` per group (emp
  **1.26493 / 3.50591 / 1.27215** vs closed **1.26 / 3.50 / 1.26** at c = 0.10/0.50/0.90).
  **Exactly zero at c ∈ {0,1} for ANY λ**, including λ = 1000; the same component at c = 0.5
  carries **0.5722**. Decay **3.49869 / 1.24724 / 0.13178** = **26.55×** from c = 0.50 → 0.99.
  *Influence tracks INCONSISTENCY, not the coefficient — a mastered component cannot be
  re-weighted back into relevance*
- **the λ trap.** A weighted sum is lexicographic only while λ·max(partial) < min(correctness
  gap): at **λ = 0.99** the argmax is correct, at **λ = 1.01** it is a wrong-but-formatted
  sequence, tied at λ = 1. `|A_norm|/|A_cent| = **2.850**`, so a λ tuned under `centered` is
  NOT tuned under `normalized`
- **the RM contrast.** A preference dataset is a **MATCHING** (degree ≤ 1), so the BT gradient
  on an uncompared response is **exactly 0.0** and shifting it by 1 / −37.5 / **1e6** changes
  the loss by exactly 0. Same data, two inits: argmax **seq 0 (gold 1.0)** vs **seq 14
  (gold 0.0)**. *An RM's accept set is set by its INITIALISATION; a verifier's is the same
  object under both — data, not a fit*
- **THE DAY'S RESULT — four real runs, differing only in the checker:**

```text
                    dead_frac    v_pass    gold_pass
  honest              0.0361     0.4400      0.4400    sound+complete: ONE column (dev 0.0)
  leaky (hacked)      0.7694     0.9681      0.0251
  starved             0.7833     0.0319      0.1383    gold ABOVE v = incompleteness
  saturated (solved)  0.8361     0.9803      0.9803
```

  **The HACKED run and the SOLVED run agree on every column the reward can produce** and
  differ **0.025 vs 0.980** on the held-out one. A k=1 group gives `Â = (G−1)/G = 0.875000`
  whether the acceptance was a hack or a real success — **bit-identical advantage vectors**.
  *The gold column is the only column carrying the distinction*
- **fixture calibration (8 seeds each), thresholds from the gap between DISTRIBUTIONS:**
  honest dead **0.017–0.042** (Δv **+0.306…+0.435**), leaky dead **0.728–0.822** (Δgold
  **−0.115…−0.086**), starved dead **0.650–0.844** at v_pass **0.022–0.050**, saturated dead
  **0.814–0.919** at v_pass **0.980–0.983**. Diagnostic is silent on **8/8** healthy and
  fires `verifier_hacked` ONLY on **8/8** leaky

## Week 6 Day 3 — sampling, pass@k & rejection sampling

- **THE THESIS.** *pass@k is not a capability — it is a capability COMPOSED WITH A SELECTOR you
  do not have.* Three selectors, three limits, on the same policy: **oracle → 1**,
  **BoN/filter → the verifier's PRECISION**, **majority vote → 1 if the modal answer is right
  else 0**. Measured: unsound verifier saturates at **0.321655** while pass@k → **1.000000**
- **the unbiased estimator.** `1 − C(n−c,k)/C(n,k)`, unbiased to **7.772e-16** over 9 points, by
  **exchangeability** (compute P(a random k-subset is all-wrong) two ways). Product form
  `Π_{i=0}^{k−1} (n−c−i)/(n−i)` — never form `C(200,100) = 9.05e58`. At **k = n** it collapses to
  exactly `1[c ≥ 1]`
- **the plug-in's bias.** `1 − (1−c/n)^k` is `f(p̂)` with **f concave** ⇒ Jensen ⇒ **UNDER**-reports.
  n=10, p=0.1, k=10: **0.493747 vs truth 0.651322**, bias **−0.157575 = 24.19% thrown away**.
  Aggregate order matters: same data, mean-then-estimate **0.9960** vs estimate-then-mean **0.6250**
- **the identity.** `P(dead) = [1 − pass@G] + p^G` to **5.551e-17** on a 501-point grid. At G=8,
  dead = **66.3420% at BOTH p=0.05 and p=0.95**, failure-side share **100.00% vs 0.00%**.
  `alive(0.95) = pass@8(0.05) = 0.3365796` — *the same term, read twice*
- **the reweighting.** `∂pass@k/∂p = k(1−p)^(k−1)`; hard/easy ratio **exactly 4.75^(k−1)** =
  **1× / 5.456e+04× / 1.414e+10×** at k=1/8/16. At **k=1 every problem is weighted EXACTLY
  equally** (1/6 each); at **k=256 the two hardest carry 100.0000%** and the easiest **0.000e+00**.
  *Changing k changes which benchmark you are running*
- **temperature × k, and its SCOPE CONDITION.** Heterogeneous set: T* = **0.20/0.20/0.20/3.00/
  3.00/3.00** at k=1/2/4/8/16/64, transition **between k=4 and k=8**. Homogeneous set: T* =
  **0.20 at EVERY k**. *T\* moves with k ONLY if p is heterogeneous across problems.* For one
  problem the whole **ordering** of temperatures is k-invariant (monotone maps preserve argmax)
- **the two decoding knobs are not the same kind of object.** Temperature preserves the
  **support** at every finite T > 0; **top-p 0.90 leaves 11 of 15** sequences alive. *Only
  truncation can lower pass@infinity*
- **correlation: the gap is UNIMODAL, not monotone.** i.i.d. over-states pass@k at every k>1:
  **+0.0706 / +0.2313 / +0.4176 / +0.4749 / +0.1921 / +0.0040 / +0.0000** at k=2/4/8/16/64/256/1024.
  Peak **0.4749 at k=16**, back to 0 by k=1024 because both curves saturate. *Largest exactly
  where people report*
- **maj@k goes to ZERO, non-monotonically, while pass@k goes to ONE.** Modal answer wrong:
  **0.3500 / 0.3517 / 0.3563 / 0.3508 / 0.3231 / 0.2764** at k=1/3/9/21/51/101 vs pass@101 =
  **1.000000**. Control (modal answer right): **0.4500 → 0.8701**. *pass@k is monotone in k by
  construction and CANNOT do this — maj@k lives on ANSWERS, pass@k on SEQUENCES*
- **FILTER vs RANKER — same verifier (γ=0.01, α=0.10), same p=0.10, same budget:**

```text
  n         FILTER P(hack output)       RANKER P(hack output)
  1               0.9000%                      0.9000%
  8               5.1426%                      6.9772%
  64              9.0794%                     43.9322%
  256             9.0909%                     90.1178%
```

  The filter's **accepted-output composition** is n-invariant at **9.0909%**. Its unconditional
  risk under first-sample fallback is `[1−(1−p_obs)^n]` times that posterior, so both selectors
  equal **0.9000% at n=1**. Ranker hack probability is `1 − [1−(1−p)γ]^n` = **pass@n for policy
  hack mass `(1−p)γ`**. At n=256: **9.09% vs 90.12% = 9.91×, from the selection rule alone**
- **best-of-n IS a KL knob.** `KL(BoN_n ‖ π) = log n − (n−1)/n` = **0.1931 / 1.2044 / 3.1745 /
  4.5491** nats at n=2/8/64/256; **√KL at n=64 = 1.7817**; each doubling ≈ log 2, **measured
  +0.6853 nats** from n=64→128. **TIES make BoN LESS selective**, so a discrete score gives
  strictly less KL: **2.0507 < 3.1745** at n=64 (convergence to the continuous law verified,
  err **4.27e-07** at L=20000)
- **THE DAY'S RESULT — the rule that was measured and NEVER FIRED.** `coverage_collapse`
  (pass@1 rises while pass@k_max falls) is the standard claim about RL on LLMs. Over **24
  configurations** (2 flavours × 4 init scales × 3 (lr,β)) then 8 seeds × 3 fixtures:

```text
  min over ALL runs of [ pass@k_max(final) − pass@k_max(0) ]  =  +0.000e+00
```

  **Never negative, anywhere.** Mechanism: on a prompt whose gold is never sampled every rollout
  scores the same ⇒ group-relative advantage **identically zero** ⇒ zero gradient. *The
  algorithm cannot push mass off coverage it is not sampling* — **D2's dead group as a SAFETY
  property**. SCOPE: 15-sequence tabular softmax, sound verifier, group baseline — *the collapse
  needs a MECHANISM; it is not automatic*
- **what training DID do.** pass@1 **0.4447 → 0.9138**, gain **decaying in k and exactly zero at
  k_max**: `+0.4691 / +0.3557 / +0.1945 / +0.0779 / +0.0216 / +0.0002 / +0.0000 / +0.0000` at
  k=1/2/4/8/16/64/256/1024. ***1024× the samples buys 0e+00 of the improvement***
- **a number that refused to be a number.** Crossover k (where the gain drops under the seed
  noise floor) = **[16, 64, 16, 64, 64]** over 5 seeds — a **4× range**. Seed 0 alone says
  "k = 16". *Report the BRACKET, not the point estimate*
- **fixture calibration (8 seeds × 3 fixtures), thresholds from the gap between DISTRIBUTIONS:**
  `no_coverage` final pass@k_max broken **≤ 0.3845** / healthy **≥ 1.0000** → **0.70**;
  `mode_locked` final entropy broken **≤ 0.6597** / healthy **≥ 0.9739** → **0.80**;
  `mass_without_support` Δpass@1 broken **≥ 0.3697** / healthy **≤ 0.1130** → **0.25**.
  `mode_locked` **LOST a conjunct to the data** — locked spread 0.123–0.379 vs healthy
  0.323–0.556 **OVERLAP**, so it cannot be thresholded. *Fewer conditions, because that is what
  the measurement supports*

## Week 6 Day 4 — reasoning data curriculum (`code/curriculum.py`, **106 checks**; 15 seqs, G=8, 16 prompts)

**The set point is a function of the metric you report.**

- **the family.** `p* = a/(a + b + k − 1)`, from `Δp ∝ p^a(1−p)^b` and
  `∂pass@k/∂p = k(1−p)^(k−1)`. At a = b = 1 it is **`p* = 1/(k+1)`**:
  **0.5000 / 0.2000 / 0.1111 / 0.0588 / 0.0154** at k = 1/4/8/16/64
- **`p* = 1/2` ⟺ `a = b + k − 1`, NOT ⟺ `k = 1`.** (2,1,2) also gives 1/2; (2,2,2) gives
  exactly **2/5**. "Train where the model is at 50%" is one point of a family
- **measure (a,b), do not assume them.** Measured on the stack: **(1.900, 1.911),
  R² = 0.9979** — exact ascent moves p by `~lr‖∇p‖² ∝ p²(1−p)²`, so (2,2) is the law,
  not the advantage-mass (1,1). Moves the k=8 set point **0.1111 → 0.1758 (1.58×)**.
  The `+k−1` is arithmetic; the exponents are not
- **the cost of the folk answer.** Train at 0.5, report pass@8: progress rate **2.53×
  better**, metric weight **56.12× worse**, net **22.17× of the useful signal lost**

**Group size at the set point.**

- `G* ≈ 2.3026·k`, approached from above. Both-tails exact integers:
  **5 / 6 / 11 / 20 / 38 / 149 / 591** at k = 1/2/4/8/16/64/256
- the single-sided `ln(0.1)/ln(1−p*)` gives **4 at k=1 and is wrong there and only
  there** (at p* = 1/2 the easy tail is half the deaths) — the W6 D1 correction again
- **with G = 8 anyway**, P(dead | p*) = **0.78 / 3.92 / 16.78 / 38.97 / 61.57 / 88.34 /
  96.93 %** at k = 1/2/4/8/16/64/256. A correctly-targeted curriculum is *supposed*
  to look mostly dead

**The verifier is a multiplier, not a preference.**

- `U = E[Σ A_i s̃_i] = 2(G−1)p(1−p)J = J·M_gold` under **s̃ = 2s−1**, where
  `M_gold=2(G−1)p(1−p)` is counterfactual gold-reward advantage mass. With s ∈ {0,1} you get half
- observed verifier-reward mass is `M_obs=2(G−1)p_obs(1−p_obs)`, so its useful fraction is
  `U/M_obs`, not generally `J`
- **`argmax_p U = 1/2` for EVERY (α,γ) with J > 0** — verified at (0,0), (.1,.01),
  (.3,.05), (.45,.50). A verifier **cannot tilt a curriculum**
- **α + γ = 1 ⇒ U ≡ 0 in p**, exactly. Healthy |A|, healthy dead_frac, zero learning,
  forever. **α + γ > 1 ⇒ U < 0 everywhere**: the gradient points backwards
- **the useful fraction** `U/mass = p(1−p)J / [p_obs(1−p_obs)]` (mass is at `p_obs`,
  U at `p`) = **0.8973 / 0.8980 / 0.6454 / 0.3109 / 0.0825** at p = .5/.1/.02/.005/.001.
  Collapse slope `J/[γ(1−γ)]` = **89.90**; the noise floor γ(1−γ) = 0.0099 **does not
  shrink with p**, so you cannot raise the LR out of it
- **γ = 0 ⇒ ratio → 1 as p → 0, at any α.** Soundness is *exactly* the property that
  stops the collapse — which is why the FP question is "is the set empty", not "how big"

**...it floors the curriculum instead.**

- `p_floor = γ(1−τ) / [τ(1−α) + γ(1−τ)]` = **0.090909** at (γ,α,τ) = (.01,.10,.10);
  **0.174312** at τ=.05; **0.230769** at γ=.03; **0.500000** at γ=.10
- **crossover with 1/(k+1) at k = 10.0000 EXACTLY** (solved, not scanned) — beyond
  pass@10 the verifier binds, not the metric
- **value retention v(floor)/v(p*)** = **0.9793 / 1.0000 / 0.8872 / 0.3804 / 0.0358**
  at k = 8/10/16/32/64 — at pass@64 a **1% FP rate costs 96.4%** of the signal
- **γ = 0.10 has J = 0.80 and p_floor = 0.50.** A respectable Youden index *and*
  unusable below the median: **J is a multiplier on magnitude, p_floor is a constraint
  on the domain.** Different objects — hence base-rate neglect
- **which wall binds depends on your alive-tolerance** (learner's finding): the dead
  wall at G=8 is **p = 0.2501 / 0.1591 / 0.0830** at a 10% / 25% / 50% tolerance, so
  the dead wall binds at 10% and the precision floor binds at 50%. Both correct

**Control in log-odds, not in p.**

- 50%-of-peak band: **[0.1464, 0.8536] at k=1 → [0.0036, 0.0407] at k=64**
- width in p **0.70711 → 0.03709 = 19.07× narrower**; in logit-p **3.5255 → 2.4654 =
  only 1.43×**. **Ratio 13.33×** — controlling in p makes it 13× harder than it is
- hi/lo *grows* **5.83 → 11.33**: the band is not shrinking uniformly

**`/std` is itself a curriculum.**

- centered `ΣA² = c(G−c)/G`: at G=64, **c=1 → 0.9844 vs c=G/2 → 16.0 = 16.25×**.
  Whitened, **both → 64**. That equalisation *is* the tilt
- **TILT is a ratio of ratios**: `[G²/(G−1)] / [4G(1−2^(1−G))/(G−1)] = G/[4(1−2^(1−G))]
  ≈ G/4` = **2.0157 at G=8, 16.0000 at G=64.** The (G−1) cancels
- the tilt is **symmetric** — it lifts c=1 and c=G−1 equally; **pass@k picks which tail
  wins**, which is why the whitened set point moves *down*
- whitened p* / metric p*: G=8 → **1.00 / 0.75 / 0.82 / 0.88 / 0.96**;
  G=64 → **0.24 / 0.32 / 0.44 / 0.71** at k = 4/8/16/64. Up to **4.2×**
- **at (G=64, k=1) the whitened set point does not exist** — flat to 1e-15 over
  **p ∈ [0.4166, 0.5834]**. Report "no preference", not a third significant figure
- **negative control:** at G=2, whitened argmax == metric argmax (P(alive) = 2p(1−p))

**Contamination: the eval lie decays, the training cost does not.**

- inflation = `c(1 − pass@k(p_true))` = **3.500 / 2.450 / 1.201 / 0.288 / 0.017 / 0.000**
  points at k = 1/2/4/8/16/64 (c = 0.05, p_true = 0.30) — same shape as D3's RL gain
- a memorised prompt is **dead on the SUCCESS side, ~100%**: contaminated fraction c
  costs exactly c of the rollout budget for zero gradient, and it is **invisible on the
  reward curve** because those groups score 1.0
- **twin gap 0.7000 = 1 − p_true, control +0.000000**, populations fully separated.
  But the detector is an **estimator**: `se = sqrt(p(1−p)/n_contaminated)`; at n = 4
  that is **0.23** and it cannot tell a clean set from a quarter-rotten one

**Calibration laws (from the diagnostic panel).**

- **a threshold is calibrated for a REGIME**: the alive-fall gap is **0.35 wide at 150
  steps and 0.014 wide at 60**. Re-derive, never borrow
- **a change statistic must not overlap the estimator's burn-in**: with the head window
  at step 0, prompts whose true p never left 1e-4 read as having moved **0.047–0.276**
- **three fixtures cannot separate four rules** — `budget_dead`/`saturated_band` looked
  collinear until the bench went 3 → 6 fixtures
- **cut rule `mean_p_rising`, reason: fires on healthy HARDER than on sick** —
  **+0.796…+0.938 healthy vs +0.373…+0.410 sick.** Rising p *is* learning

**The shipped panel** (TODO 4, calibrated over 5 paired seeds × 6 fixtures at 150 steps).

| threshold | value | healthy / uniform | sick |
| --- | --- | --- | --- |
| `CLIFF_SHARE` | 0.50 | 0.06–0.13 | `too_hard` 0.94–1.00 |
| `ALIVE_FALL` | 0.40 | 0.04–0.24 | `open_loop` 0.59–0.67 |
| `DEAD_LEVEL` | 0.60 | 0.36–0.54 | `open_loop` 0.67–0.78 (thin) |
| `FLOOR_SHARE` | 0.125 | 0.01–0.07 | `too_hard` 0.83–1.00 |

Fire-sets, and they are all distinct — which is what non-collinearity actually means:

- `healthy` **{}** · `uniform` **{}** — silent is a *check*, not an aspiration
- `open_loop` **{saturated_band, budget_dead}** — stable, 5/5
- `too_hard` **{cliff, budget_dead, floor_violated}** — stable, 5/5
- `coin_flip` **{budget_dead}** or **{+floor_violated}** — borderline, 3/5
- `bad_verifier` **{}** or **{floor_violated}** — borderline, 3/5

**`floor_violated` is a SHARE, not a MEAN — and that is the whole rule.** Written as
"mean realised p below the floor" it reads **0.237…0.400 ABOVE** the floor on
`bad_verifier` and goes permanently silent. Written as "share of the rollout budget
below the floor" the same runs read **0.000…0.172** and straddle. The mean is dominated
by the prompts you are training on *correctly*; the question "are we training where the
verifier cannot be trusted" is about the **tail of the budget**, not the average prompt.

## Week 6 Day 5 — mini RLVR/GRPO loop (`code/mini_rlvr.py`, **75 checks**)

- **THE THESIS: every `p` needs an event type.** `p_obs` drives reward-group deaths;
  `p_gold` drives oracle pass@k and curriculum value; `p_answer[a]` drives voting.
- **sensor identity:** `p_obs = γ + (1−α−γ)p_gold = γ + Jp_gold`; inverse
  `p̂_gold = (p̂_obs−γ)/J`, with raw inverse retained, scheduling estimate projected, and
  **`J≤0` rejected**.
- **truth boundary:** mutate `gold` with θ/samples/RNG/`accept`/data policy fixed ⇒ final θ
  **bit-identical**; mutate reachable `accept` ⇒ θ changes; all-zero `accept` ⇒ exact no-op.
- **paired randomness:** one uniform for every `(step,prompt,draw)`, including inactive prompts;
  separate sampler RNG. Different controller arms never shift each other's rollout stream.
- **honest negative control:** raw vs calibrated final θ difference **0.000000 exactly** over
  5 seeds.
- **calibration repairs the coordinate:** starved sensor-error reduction mean **0.076343**
  (range **0.063569–0.090097**); leaky mean **0.365958**
  (**0.311036–0.401402**).
- **calibration does not repair the objective:** leaky calibrated gold pass@8 gain
  **−0.005075** over 1,344 rollouts, final verifier–gold gap **0.240452**.
- **controller result is a NULL at the final-policy endpoint:** per 1,000 rollouts, band gain
  **+0.014309**, uniform gain **+0.010175**; paired effect **+0.004134**, spread **0.002114**.
- **one policy, six non-interchangeable numbers** on the leaky step-0 panel:
  verifier pass@8 **0.999956**, oracle gold pass@8 **0.764608**, filter **0.391510**,
  ranker **0.419953**, vote **0.332450**, precision **0.383829**.
- **diagnostic thresholds, measured:** dead budget **0.584583**, sensor mismatch **0.166659**,
  reward hacking **0.110000**, selector gap **0.150000**, KL runaway **0.430424**.
  Every rule is strict `measured > threshold`, reads a tail level, and is silent on health.

## Week 8 Day 1 — GPU memory math (`code/gpu_memory.py`, **9 tests**)

- 7B BF16 weights = **14 GB = 13.04 GiB**.
- classic mixed-precision Adam = `2 + 2 + 4 + 8` = **16 bytes/parameter**:
  **112 GB = 104.31 GiB** before saved activations, workspaces, communication, or runtime.
- KV cache = `2·L·n_kv·d_head·B·T·bytes`; 32-layer MHA example =
  **512 KiB/token**, **2 GiB at 4k×B1**, **64 GiB at 8k×B16**.
- simplified 24 GiB budget: `24 − 13 − 2 = 9 GiB` for KV ⇒
  **18,432 aggregate token slots**, or **4,608 tokens/sequence at B=4**.
- at `B=8,T=4096,H=4096,L=32`, one saved BF16 `[B,T,H]` tensor per layer =
  **8 GiB**; naive materialized attention matrices = **256 GiB**.
  Doubling context gives the scaling ladder **2× hidden / 4× attention / 2× KV**.
- global batch 128 on 8 workers at microbatch 2 ⇒ accumulation **8**; microbatch 1 ⇒
  accumulation **16**, roughly half the batch-linear activation memory, unchanged model state.

## Week 8 Day 2 — fine-tuning systems (`code/finetuning_memory.py`, **31 checks**)

- four buckets: **LoRA → trainable state**; **QLoRA → frozen-base storage too**;
  **ZeRO/FSDP → replicated state**; **checkpointing → saved activations**.
- rank-16 `q_proj` + `v_proj` over 32 hidden-size-4096 layers =
  **8,388,608 adapter parameters**; classic adapter state = **0.125 GiB**.
- scoped 7B persistent state: full FT **104.308 GiB**; BF16-base LoRA
  **13.164 GiB**; idealized 4-bit-base QLoRA **3.385 GiB**.
- full-FT world-size-8 ZeRO-0/1/2/3 lower bounds =
  **104.308 / 35.856 / 24.447 / 13.039 GiB per rank**.
- `16P/N` is persistent ZeRO-3 state, **not peak**: add current full FSDP unit,
  possible prefetch, communication buffers, activations, temporaries, and headroom.
- checkpoint chain model `ceil(L/k)+k`: at `L=32`, minimum **12 slots** for the
  entire interval **k=4..8**, not one rounded `sqrt(L)` answer.

## Week 8 Day 3 — distributed training (`code/distributed_training.py`, **29 checks**)

- equal-chunk ring, `N=8, M=1 GiB`: chunk **128 MiB**; all-gather or
  reduce-scatter **896 MiB sent/rank**; all-reduce **1.75 GiB sent/rank**.
- ring time baseline at **5 us/round** and effective **100e9 bytes/s**:
  1 GiB all-reduce over 8 ranks = **18.860482 ms**.
- at `N=8, W=G=1 GiB`: DDP all-reduce **1.750 GiB/rank**; FSDP with one
  parameter gather **1.750 GiB/rank**; FSDP with two gathers
  **2.625 GiB/rank = 1.5x DDP**.
- pipeline `S=4`: bubble-only efficiency **8/11 = 72.73%** at `m=8` and
  **32/35 = 91.43%** at `m=32`.
- stage times `[1,1,1,2]`: balance efficiency **62.5%**; at `m=8`, composed
  utilization **45.45%**.
- only **DP** multiplies independent examples:
  `global_batch = DP * microbatch/replica * accumulation`; TP/PP ranks
  cooperate on those examples.
