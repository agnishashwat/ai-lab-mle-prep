# AI Lab MLE Prep Progress

## Current Status

- Current week: **Week 06 — GRPO & RL with Verifiable Rewards — COMPLETE (5/5)**.
  **462 staged checks** across the week: `grpo.py` **96**, `rlvr.py` **112**,
  `sampling.py` **73**, `curriculum.py` **106**, `mini_rlvr.py` **75**. Fast and full Day 5
  suites are clean under `-W error::RuntimeWarning`.
- Current day: **Week 6 Day 05 DONE** — *mini RLVR/GRPO capstone*. The composed loop preserves
  the truth boundary, calibrates only the scheduling sensor, evaluates deployable selectors
  separately from oracle pass@k, and uses paired random events across ablation arms. Full result:
  final-policy controller effect **+0.0041 per 1,000 rollouts**, paired spread
  **0.0021 → null**; leaky calibrated gold gain **−0.0051**, verifier–gold gap **0.2405**.
  **Thesis: every `p` needs an event type.**
- Historical Day 3: **pass@k is not a capability; it is a capability COMPOSED WITH A SELECTOR you do
  not have.** Three selectors, three limits, same policy: **oracle → 1**, **BoN/filter → the
  verifier's PRECISION** (measured saturating at **0.321655** while pass@k → **1.000000**),
  **majority vote → 1 if the modal answer is right else 0**.
  **Headline 1 — THE RESULT OF THE DAY: a rule that was measured and never fired.**
  `coverage_collapse` (pass@1 rises while pass@k_max falls) is *the* standard claim about RL on
  LLMs — well-posed, discriminative in principle, would have passed code review. Over **24
  configurations** (2 flavours × 4 init scales × 3 (lr,β)) then 8 seeds × 3 fixtures, the
  minimum of `pass@k_max(final) − pass@k_max(0)` was **+0.000e+00. Never negative, anywhere.**
  Mechanism: on a prompt whose gold is never sampled every rollout scores the same ⇒
  group-relative advantage **identically zero** ⇒ zero gradient. *The algorithm cannot push
  mass off coverage it is not sampling* — **D2's dead group read as a SAFETY property**.
  SCOPE: 15-sequence tabular softmax, sound verifier, group baseline — it says the collapse
  needs a MECHANISM, not that it cannot happen. Three days, three cut rules, three *different*
  reasons: D1 the **instrument** couldn't support it, D2 the **semantics** didn't, D3 the
  **system does not exhibit it**.
  **Headline 2 — maj@k goes to ZERO, non-monotonically, while pass@k goes to ONE.**
  **0.3500 / 0.3517 / 0.3563 / 0.3508 / 0.3231 / 0.2764** at k=1/3/9/21/51/101 against
  pass@101 = **1.000000**; control (modal answer correct) **0.4500 → 0.8701**. pass@k is
  monotone in k *by construction* and cannot do this — maj@k lives on **answers**, pass@k on
  **sequences**, and the map between them is many-to-one.
  **Headline 3 — FILTER vs RANKER: 9.91× from the selection rule alone.** Same verifier
  (γ=.01, α=.10), same p=.10, same n=256. The filter's accepted-output composition
  `γ(1−p)/[γ(1−p)+(1−α)p]` is **9.0909%, n-INVARIANT**; its unconditional fallback risk is
  **0.9000 / 5.1426 / 9.0794 / 9.0909%** at n=1/8/64/256. Ranker
  `1 − [1−(1−p)γ]^n` = **pass@n for policy hack mass `(1−p)γ`** =
  **0.9000 / 6.9772 / 43.9322 / 90.1178%**. *Equal at n=1; 9.09% vs 90.12% at n=256.*
  **Headline 4 — the estimator, and Jensen.** Unbiased `1 − C(n−c,k)/C(n,k)` to **7.772e-16**
  by **exchangeability**; the plug-in is `f(p̂)` with f **concave** so it **UNDER**-reports —
  **0.493747 vs truth 0.651322 = 24.19% thrown away** at n=10,p=.1,k=10. Aggregate order
  matters: **0.9960** mean-then-estimate vs **0.6250** estimate-then-mean on identical data.
  **Headline 5 — the metric REWEIGHTS the benchmark.** `∂pass@k/∂p = k(1−p)^(k−1)`; at **k=1
  every problem is weighted EXACTLY equally**, at **k=256 the two hardest carry 100.0000%** and
  the easiest **0.000e+00**. Hard/easy ratio exactly **4.75^(k−1)**. *Changing k changes which
  benchmark you are running.* Related scope condition: T\* moves with k **only if p is
  heterogeneous** — homogeneous set gives T\* = 0.20 at every k, heterogeneous transitions
  **between k=4 and k=8**.
  **Headline 6 — best-of-n IS a KL knob, and ties are the whole exercise.**
  `KL(BoN_n ‖ π) = log n − (n−1)/n` = **0.1931/1.2044/3.1745/4.5491** nats at n=2/8/64/256;
  measured **+0.6853 nats** per doubling at n=64→128. A **discrete** score is strictly less
  selective: **2.0507 < 3.1745** at n=64.
  **Coding headline — the private reference run caught SIX defects, the most of any day**: a
  vacuous stability check (`1 − 1.1e-59` rounds to exactly 1.0), a threshold with the wrong
  sign, an argmax destroyed by saturation at k=1024, an operator-precedence bug
  (`np.abs(a - b if False else c)` parses as `(a-b) if False else c`), two byte-identical
  fixtures (`max(2, 15//2) == 15//2`), and the rejection above. Also: the crossover k came out
  **[16, 64, 16, 64, 64]** across five seeds — a **4× range** — so it is reported as a
  **BRACKET**; seed 0 alone would have said "k = 16".
  **Pedagogical headline — the skeleton contract was revised FOUR times in one session**:
  slots must (1) name the FUNCTION and which argument varies, (2) not bundle an authorship
  judgement with a technical skill (the MY-OWN-EVIDENCE slot, blank 6 sessions running, was
  retired), and (4) not contain a label that contradicts their own formula (my "FAILURE-side
  term p^G" — p^G is the success side; he answered the formula correctly and did **not flag
  the contradiction**, which is now a coaching point).
  **Second-clause gap split TWICE.** Block 4: MECHANISM slots **9/9**, NUMBER slots **0/2**,
  THESIS slots **0/3**. Block 5 split it again into **MECHANISM vs NAME** — he derived
  `F(u)=u^n`, `q(u)=n·u^(n−1)` and the integral `log n − (n−1)/n` correctly while leaving
  **"probability integral transform"** and **"Uniform(0,1)"** blank. *He can execute the object
  and cannot name it* — identical to Jensen on Block 3, which he repaired 40 min later.
  **Wrong-level family: 14th sighting**, and the two sharpest yet. (13) He substituted the
  filter's **posterior** (9.09%) where the ranker's **per-draw hack mass**
  (`(1−p)γ=0.9%`) was required — arithmetic flawless, modelling one level off.
  (14) `pass_at_k` returned `product` =
  P(all k wrong) instead of `1 − product`, inverting both edge cases; the docstring warned
  about this exact error. Note the trap: at c=1 it returns **0.5, which is correct**, because
  0.5 is the fixed point of x ↦ 1−x.
  Block 0 **14/17 (82.4%)**, Block 1 **26.5/37 (71.6%)**, Block 2 check **2/3**, Block 3
  **11.5/19**, Block 4 drill **11/20 (55%)**, Block 5 **20.25/32 (63.3%)** — total **85.25/128
  = 66.6%**, his best day of Week 6. **Block 5 §(3) scored 5/5 — the first perfect section of
  the week**, and it was earned by *spontaneously* reaching back to `∂pass@k/∂p = k(1−p)^(k−1)`
  to explain why T\* moves with k. First unprompted mechanism link in six weeks.
- Previous day: **Week 6 Day 02 DONE** — *RL with verifiable rewards: what "verifiable" buys and
  what it does not*. `code/rlvr.py`, **112 staged checks**, ten stages, 5.9 s full / 4.9 s
  `--fast`, 0 failures, clean under `-W error::RuntimeWarning`. All four TODOs his.
  **Thesis — a verifier is a SET, not a rate.** Everything that matters is a property of `V`
  (what the checker accepts) and its two containments with `C` (what is correct): `V ⊆ C` =
  SOUND, `C ⊆ V` = COMPLETE. The scalar people quote is a property of `V` **and the policy it
  was measured under**, and RL exists to change that policy.
  **Headline 1 — the soundness theorem, and the asymmetry of the two failures.** `V ⊆ C` ⇒
  `v_pass ≤ gold_pass` for **all 415 policies at every KL** (max gap **−2.000e-12**), no scope
  condition — the strongest guarantee anything in post-training offers. If `V \ C ≠ ∅`,
  sup(v_pass − gold_pass) = **1.000000** for **1 false positive AND for 7**: hacking is a
  **support** property, not a rate property. Their eval accuracies are **93.3% vs 53.3%**, so
  the eval statistic ranks the two checkers *opposite* to what matters under optimization.
  Incompleteness by contrast is bounded and priced: a 30% extraction failure takes p_obs
  **0.10 → 0.0700** and G **22 → 32** = **+45.5% rollout compute**. A 5% FP rate takes
  **32 → 19** — *unsoundness BUYS compute, which is why the cheap fix is the dangerous one*.
  **Headline 2 — the day's result: the hacked run and the solved run are identical on every
  column the reward can produce.** Four real runs differing only in the checker: leaky
  **dead 0.7694 / v_pass 0.9681 / gold 0.0251** vs saturated **0.8361 / 0.9803 / 0.9803**. A
  k=1 group gives `Â = (G−1)/G = 0.875000` whether the acceptance was a hack or a genuine
  success — **bit-identical advantage vectors**. The held-out gold column is the only column
  carrying the distinction; `starved` shows the other sign (**gold 0.1383 ABOVE v_pass
  0.0319** = incompleteness), and `honest` collapses to **one column** (dev exactly 0.0).
  **Headline 3 — "97% accurate" is not well-formed, and precision is a function of
  DIFFICULTY.** Same checker, same arrays, tilt the policy toward the responses it wrongly
  accepts (what optimization does): gamma **0.1515 → 0.7820**, precision **0.6720 → 0.0926**.
  Mass-weighted gamma **0.1515** vs count-weighted **0.2222**. And at a **fixed 1% FP rate**,
  `P(hack | rewarded) = γ(1−p)/[p(1−α)+γ(1−p)]` runs **1.0989 / 9.0909 / 35.2518 / 68.8581 %**
  at p = .500/.100/.020/.005 → 1 as p → 0. *Your verifier gets worse as your model gets
  better, with no change to the verifier.* Conditioning on ALIVE leaves the per-sample rate
  **unchanged** (0.34902 vs 0.34902) — what moves is CONCENTRATION: **0.33020** vs
  **5.9568e-04** = **554×**.
  **Headline 4 — D1's dead-group law generalises to a RÉNYI ENTROPY of the reward's OUTPUT
  levels.** `P(dead) = Σ_j q_j^G = exp(−(G−1)·H_G(q))`. At p = .05, G = 8: binary **66.3420%**
  → split the failure mass to (.05,.40,.55) → **0.9029%** = **73.5× from one extra reward
  level**; matching that with G alone needs **G = 92**, an **11.5×** rollout bill ⇒ *reward
  design and group size are the same knob, priced differently*. **The rule is SPLIT THE MODE**
  and it is strictly asymmetric — splitting the SUCCESS mass at p = .05 does **nothing**
  (66.3420% → 66.3420%), exactly reversed at p = .95.
  **Headline 5 — a composite reward is a self-annealing curriculum.** Centering is linear so
  `A_total = Σ_j A_j` **exactly at finite G** (max dev 2.22e-16), and
  `E[Σ_i |Â_i|] = 2(G−1)·c(1−c)` per group (emp **1.26493 / 3.50591 / 1.27215** vs closed
  **1.26 / 3.50 / 1.26**). The share is **exactly zero at c ∈ {0,1} for ANY λ, including
  λ = 1000**, while the same component at c = 0.5 carries **0.5722**; decay c = .50→.90→.99 is
  **3.49869 / 1.24724 / 0.13178** = **26.55×**. *Influence tracks INCONSISTENCY, not the
  coefficient — a mastered component has left the gradient and cannot be re-weighted back in.*
  Corollary: a weighted sum is lexicographic only while λ·max(partial) < min(correctness gap)
  — argmax is correct at **λ = 0.99**, wrong-but-formatted at **λ = 1.01**.
  **Headline 6 — the RM contrast, W5's identifiability result pointed at reward design.** A
  preference dataset is a **MATCHING** (degree ≤ 1), so the BT gradient on an uncompared
  response is **exactly 0.0** and shifting it by 1 / −37.5 / **1e6** changes the loss by
  exactly 0; same data, two inits → argmax **seq 0 (gold 1.0)** vs **seq 14 (gold 0.0)**. *An
  RM's accept set is decided by its INITIALISATION; a verifier's is the same object under both
  — data, not a fit — so its worst case is computable BEFORE training starts.*
  **Coding headline — a NEW bug family, six sightings in one session: unguarded denominators,
  every one producing a NaN rather than a crash.** `component_shares` returned `[nan]` on the
  `_fail_fast` probe **with the stage still green**; `increase_percent` hit "mean of empty
  slice". NaN never compares True, so a threshold built on one **silently never fires** — the
  worst possible behaviour for a diagnostic. Standing rule: run with
  `-W error::RuntimeWarning`; "passes with a warning" is a failure.
  **Pedagogical headline — pre-printed slots must be TYPED.** He said the Block 1 slots were
  "vague"; the defect was real. `____________` scores like an open question; `[a number]` /
  `[a set-theoretic definition]` / `[which containment fails]` recovers the format's value.
  Second finding: cold recall must test PRIOR material only, and must never name candidate
  answers in the stem. **Second-clause gap, 9th sighting** — in drill Q4/Q5 he filled **5/5
  definitional** slots and left **5/5 number-and-consequence** slots blank.
  Block 0 **4.0/6**, Block 1 **8.0/14 (57%)**, Block 2 check **2.5/4**, Block 3 **5.0/6**,
  Block 4 drill **14.0/25 (56%)**, coding **112/112 checks**. The drill **repaired a same-day
  regression**: `d = √KL` failed in Block 1 and passed under drill conditions 40 minutes later.
  **First unforced SCOPE CONDITION in six weeks** (the `no_critic` null holds only at short
  horizon), volunteered without being asked.
- Previous day: **Week 6 Day 01 DONE** — *GRPO vs PPO: what replacing the critic with a group
  mean actually costs*. `code/grpo.py`, **96 staged checks**, ten stages, 5.9 s full / 3.2 s
  `--fast`, 0 failures. All four TODOs his.
  **Thesis — GRPO does not delete the value function; it replaces a LEARNED estimate of E[r|x]
  with a MONTE-CARLO one, and unlike a critic that estimate vanishes exactly where the policy
  succeeds.** Every property follows from the baseline being computed *from the same G samples
  it subtracts from*.
  **Headline 1 — the dead-group law, the one consequence with no PPO analogue.**
  P(group carries zero signal) = **p^G + (1−p)^G**; at G = 8 that is **0.78 / 5.77 / 43.05 /
  66.34 / 92.27 %** at p = .50/.70/.90/.95/.99. N_eff/N = **33.658%** at p = .95, and raising
  the pass rate .90 → .95 kills a further **23.3%** of the batch — *improving the model degrades
  its own training signal*. Seen live in two fixtures differing only in the verifier: healthy
  **dead 0.0354 at pass 0.4963** vs saturated **dead 0.8042 at pass 0.9742** = **23×** more of
  the batch silent on the run with the better reward. The fix is G and it is expensive:
  **G = 45** for 90% alive at p = .95 (ln(0.1)/ln(0.95) = **44.891**). GRPO trades training
  memory (**252 → 140 → 112 GB**, −44.4% then −55.6% with RLVR) for **inference compute**, and
  the bill grows as the model improves.
  **Headline 2 — the shrinkage is exact and directionless.** E[ĝ] = **((G−1)/G)·∇J** at finite
  G; measured **0.4994 / 0.7496 / 0.8755** at G = 2/4/8 with cos ≥ **0.999992**. At **G = 1 the
  gradient is identically 0** — the defining property, not an edge case. LOO = centered ·
  G/(G−1) **pathwise**, so it removes the shrinkage and *cannot* reduce variance.
  **Headline 3 — the group mean is a high-pass filter, EXACTLY at finite G.** c(x) removed to
  max dev **0.000e+00 / 2.220e-16 / 1.210e-14 / 2.039e-10** for c = 0/1/−37.5/1e6. A KL folded
  into the reward survives only as its within-group deviation: **0.035726 vs 0.6842 full =
  19.15× deleted**, and a *uniform* drift is invisible (**2.50e-16**). ⇒ a β inside the
  advantage is an **open-loop controller**. Companion result: k1 and k3 are both unbiased to 10
  dp (**0.1517183046**), but **k1 is negative on 50.0% of the probability mass** and k3 on
  **0.0%** — so the preference for k3 is **feasibility, not variance** (var ratio *reverses*:
  **423×** at KL 0.004, **0.23×** at ~0.65).
  **Headline 4 — ÷std swaps information weighting for its square root.** Group weight Σ|Â| is
  centered **2G·p(1−p)** (= the variance = Fisher information) vs normalized **2G·√(p(1−p))**;
  amplification exactly 1/std (**3.0237** at k=1 vs **2.0000** at k=4 ⇒ **1.5119×** boost, and
  the balanced:extreme ratio collapses **2.2857 → 1.5119**). *Control: the bias COMPRESSES the
  weighting, it does not invert it* (1.75 < 4.00 and 5.29 < 8.00 both hold).
  **Headline 5 — the clip is inert at μ=1, exactly.** Bound-token fraction **0.000e+00** every
  step at μ=1 vs **0.2222** at μ=8 with the *same* lr/seed/task ⇒ **GRPO at μ=1 is REINFORCE +
  a group baseline + a KL term wearing PPO's notation**.
  **Headline 6 — a correction carried in from W4 D5: `no_critic` never BEAT `full`.**
  **5.1786 ±0.036 vs 5.1501 ±0.109** — the effect is **3.8× smaller than `full`'s own spread**.
  Corrected at 5 sites. It is a *null result*, and it is a stronger argument for GRPO than the
  wrong reading was.
  **Pedagogical headline — the skeleton-first boundary condition, found by direct test.**
  *Telling* him to use skeleton-first produced **zero compliance**; making him emit the empty
  slots before any content produced **4/4 (100%)** on a question he had scored **1/2** on
  thirty minutes earlier. The operative variable is emitting the slots, not being told to.
  Standing rule instituted: **the tutor pre-prints the skeleton, he fills it.**
  Cold recall **4/7 (57%)**, D-recall **10/18 (56%)** then **4/4 on the re-ask**, derivations
  **10.75/16 (67%)**, mechanism **2.5/3**, design **13.5/27 (50%)**, coding **19/23.5 (81%)** —
  his best block. Total **63.75/98.5 = 64.7%**. **All four coding losses were calibration or
  numpy/torch numerics; the mathematics was correct on first submission every time.**
  Block 5 (interview drills) deferred by agreement.
- Previous day: **Week 5 Day 05 DONE — WEEK 5 COMPLETE** — *DPO failure modes: what the data
  format can and cannot identify*. `code/dpo_lab.py`, **92 staged checks**, nine stages,
  65.6 s full / 34 s `--fast`, 0 failures. His output is **byte-identical to the private
  reference implementation** in `--fast` mode (`diff` clean across all 89 shared lines).
  **Thesis — the four failure modes are consequences of the DATA FORMAT, not of the
  algorithm.** The training loss is *exactly constant* along every direction the data do not
  constrain, so the loss going down is not evidence of anything. Every mode is a **fixture**:
  one dataset function, four knobs (`shape` / `eps` / `length_confound` / `n_pairs`), one knob
  per mode.
  **Headline 1 — Ford's condition, measured over 12 seeds.** P(wins digraph strongly
  connected): `matching` (deg 1) **0.00 at every ε ∈ {0, .10, .25, .35}**; `cycle` (deg 2)
  **0.00** likewise; `all_pairs` (deg 14) **0.00 → 0.58 → 0.83 → 1.00**. So **clean data are
  never strongly connected at any degree** (orienting any graph by a total order gives a DAG),
  and **noise needs density to act on** — in a matching, **2 of 7 labels flipped and it was
  still a DAG**. A real DPO dataset is the top row, so the finite-margin regime is
  structurally unavailable. Corollary, 6/6 seeds: **more comparisons ⇒ SMALLER final margin**
  (2 pairs **6.0470** > 7 pairs **4.1916** > 105 pairs **1.6687**).
  **Headline 2 — the β/ε collapse, predicted on D1 and confirmed on D5.** Because the policy
  is identified only by `r/β`, a β-sweep and an ε-sweep matched on `Δ̂* = logit(1−ε)/β` must
  produce the same KL. Two structurally independent sweeps agreed to **9.59e-05 / 6.56e-05 /
  2.78e-16 / 2.47e-04** at Δ̂* = 1.0986 / 2.1972 / 4.3944 / 8.7889, and the collapse is
  non-trivial (KL itself varies **0.0711 → 0.6805**).
  **Headline 3 — a NEW law, unplanted: steps-to-target ∝ 1/β².** Steps to 0.995·M*:
  **76 / 307 / 1237 / 4978** at β = 2 / 1 / .5 / .25 — ratios **4.04, 4.03, 4.02**.
  *Halving β costs four times the compute, not two* (gradient scale ∝ β, distance ∝ 1/β).
  **Headline 4 — the proxy/gold inversion, with a healthy CONTROL.** Broken fixtures reach
  **acc 1.000** while gold peaks at steps **0 / 2 / 77 of 6001** and gives back **0.4313 /
  1.2103 / 0.1036**; the healthy control peaks at step **1270** and gives back **exactly
  0.0000**. Sharpest single line of the week: **M3's gold goes +0.7514 → −0.4532** — DPO
  training made the model *worse than the reference it started from*, at 100% train accuracy,
  with the loss falling monotonically the whole way.
  **Headline 5 — early stopping is an ESTIMATOR, and it needs no held-out set.**
  `select_checkpoint` never sees gold; it stops at the first step whose margin reaches
  `logit(1−ε)` — step **105**, margin **2.2050** vs cap **2.1972** — recovering **0.1663 of
  the 0.3437** that shipping the last checkpoint costs. Rejected: epochs (accuracy hits 1.000
  at **step 1 of 6000**) and KL (bounded at **6.3216**, so it saturates while the reward
  diverges, and M4 is invisible in it). On clean data ε = 0 ⇒ cap = **+inf** ⇒ no natural
  stopping rule — *that is not a limitation of the code, it is M1*.
  **Headline 6 — a correction to W4 D4: "seed spread grows with optimisation pressure" is a
  3-SEED ARTIFACT.** Same runs: gold spread amplifies **6.09× at 3 seeds** and **1.04× at 5**.
  On those same runs `margin` spread *contracts to exactly 0.000000* while `kl` spread grows
  **71×**. **≥3 seeds is a FLOOR.** Also measured: pairing the data and run seeds shrinks the
  worst KL spread **2.07×**; and the *metric* decides whether an ablation has power (KL effect
  **0.4038** vs spread **0.3621** ⇒ intervals overlap, unsupported; margin effect **0.1201**
  vs spread **0.0282** ⇒ **4.3×**, resolved).
  Cold recall **3.85/6 (64%)**, D1–D4 recall **9.0/20 (45%)**, derivation/design **5.0/12
  (42%)**, interview drills **not attempted** (model answers given at his request), coding
  **92/92** with three of six TODOs his. **Pedagogical headline: skeleton-first did NOT
  transfer unprompted** — D4's 3.7× intervention was not reused and the recall quiz hit the
  week's low. W6 D1 should test prompted-vs-withheld directly.
- Previous day: **Week 5 Day 04 DONE** — *DPO variants as a 5×5 grid, not a paper zoo*.
  `code/variants.py`, **126 staged checks**, ten stages, 48.8 s full / 12.1 s `--fast`, under
  `-W error::RuntimeWarning`. He wrote all four TODOs; his output is **byte-identical to the
  private reference implementation** (`diff` clean across all ten stages).
  **Headline 1 — every named method is one cell of one grid.**
  `L = mean_k ℓ(f(y_w), f(y_l)) [+ λ_SFT·mean_k −log π(y_w)/|y_w|]`, with 5 implicit rewards ×
  5 links = **25 cells**, 4 of them named by papers. The grid reproduces five independently
  hand-written implementations **elementwise at max dev 0.000e+00**. The **link** decides
  whether the margin is finite; the **reward** decides where the length goes. Proved by
  occupying the empty cell `ref_mean × squared_target`, which inherits τ's cap from IPO and the
  mean reduction's length amplification from SimPO exactly as predicted.
  **Headline 2 — DPO's divergence signature is a constant increment per DOUBLING.**
  On D1's non-strongly-connected matching: DPO margin **5.9758** with increments **+0.7163**
  then **+0.7078** ⇒ `margin ~ log t`. cDPO stopped at **2.1972 = ln 9 = logit(1−ε)** and IPO
  at exactly **2.0000 = (2f−1)τ**, both hitting their closed forms. KL **0.9511 → 0.4704 →
  0.2332** against a point-mass bound of 6.3216. **Accuracy read 1.000 for all three** —
  it saturates at step ~20 and reports nothing for the remaining 7 980 steps.
  **Headline 3 — three received claims were overturned by measurement.**
  (a) IPO does not cap uniformly; it **compresses**, squeezing the strong/weak margin ratio
  **11.3330 → 4.9000** (2.31×) — that is its price. (b) SimPO's `1/|y|` **amplifies** length
  bias rather than braking it: at matched margin, MEAN moved `E[|y|]` further than SUM in all
  four cells, because `d(r)/d(logπ) = β/|y|` is largest at `|y|=1`, making short sequences the
  cheapest lever. (c) ORPO's odds term **flips sign with β** (−1.899 → +0.385) while the SFT
  term holds at every β (+0.566 → +0.557) — run every ablation at more than one hyperparameter
  before naming a mechanism.
  **Headline 4 — a bounded loss is not a loss with a finite minimiser.** Built `stage_kto` +
  `PromptEnv` to test the claim that KTO gets P1 immunity free. It does not: `inf σ(−u) = 0`
  only at `u = ∞`, and KTO's margin climbed **+0.9294** per doubling next to DPO's **+0.7207**
  — *faster*. What boundedness buys is bounded **influence**: at `d = −20`, `−log σ`'s gradient
  is **0.9999999979** while prospect's is **2.061e-09**. With 4/40 labels flipped DPO's clean
  margin fell **1.2411** vs KTO's **0.6684**. KTO answers D3's noise problem, not D1's P1.
  **Headline 5 — dropping pairs is exact, not approximate.** The prospect link is a sum of two
  independent per-example terms, so `KTO(unpaired) == 0.5 × mean(prospect link)` =
  **0.475211814705432** on both sides. Learner-initiated finding: `z_ref`'s mismatch roll is a
  **no-op on unconditional data** (a roll is a permutation, the mean is permutation-invariant,
  agreement to 1e-16); it only becomes load-bearing under prompt conditioning, where the gap is
  **0.020983**. This is why `PromptEnv` had to exist.
  Cold recall **3.6/6 (60%)**, D1–D3 recall **9.0/14 (64%)**, invent-the-variants **4.2/10
  (42%)**, interview drills **2.0/9 (22%)** → **3.7/4 (93%)** after a retrieval reset,
  coding **126/126**. **Pedagogical headline: "count the parts" FAILED and "type the numbered
  skeleton FIRST, then fill it" WORKED** — same question 15 minutes apart, 1.0/4 → 3.7/4.
- Previous day: **Week 5 Day 03 DONE** — *DPO implementation details: length, reference caching,
  cDPO, SFT initialization, and numerics*. `code/dpo_trainer.py`, **51 staged checks**, seven
  stages, built on Week 2's `TinyGPT`, under `-W error::RuntimeWarning`.
  **Headline 1 — cDPO is Day 1's ε-cap put back from the loss side.**
  `dL/dΔ̂ = β[σ(βΔ̂)−(1−ε)]`, so
  `Δ̂*=log((1−ε)/ε)/β`, finite for every ε>0 and divergent at ε=0. At β=0.3, the analytic
  optima **9.814797 / 7.324082 / 4.620981 / 2.063464** for ε=.05/.10/.20/.35 had gradients
  below 3e-17 and positive curvature. Past the optimum, cDPO reverses the margin rather than
  merely slowing it. Corrected boundary: ε=.5 is a zero-margin target, not constant loss;
  β→0, not uniformly β→∞, is the no-op boundary.
  **Headline 2 — sum vs mean is a measured length prior.** On constant per-token drift +0.2,
  sum-reduced implicit reward had slope **+0.200000** against length while mean-reduced reward
  had slope **0.000000**. Sum asks the marginal token to beat zero; mean asks it to beat the
  running average. DPO has no explicit on-policy `−β KL_t` resistance term.
  **Headline 3 — the reference is a preprocessing artifact on fixed data.** Cached chosen and
  rejected reference logprobs were **bitwise exact**, required exactly **2 forwards**, and cost
  **2 fp32 scalars = 8 bytes/pair**. The trainer has no `model_ref` argument, so reference calls
  after preprocessing are impossible by construction.
  **Headline 4 — fp32 accumulation is part of the algorithm.** At T=256, the bf16 path differed
  from fp32 by **1.026245**, versus representative `β·margin=0.050000`: numerical error was
  **20.525× the signal**. Stable probes: fp32 `z=+40` makes naive `log(sigmoid(z))` exactly 0;
  `z=−100` makes naive NLL infinite while `−logsigmoid(z)=100`.
  **Headline 5 — SFT must seed BOTH roles.** Chosen-token NLL **2.071305→0.396558** and chosen
  mean logprob/token **−2.081233→−0.474107**. Cloning that checkpoint into policy and reference
  restored exactly zero initial log-ratios and DPO NLL **+log 2=0.693147**.
  **Headline 6 — the TinyGPT trainer exposes what loss/accuracy hide.** Full run loss
  **0.693147→0.549205**, margin **0→4.3369**, exact subset KL **0→0.92182**, expected length
  **2.1227→2.3867**. On the fast fixture chosen logprob fell **−0.59155** while rejected fell
  **−2.06221**; on the full fixture chosen rose **+2.21402** while rejected fell **−2.12287**.
  The chosen-logprob sign is fixture/phase-sensitive and is not a μ≠π_ref diagnostic.
  Cold recall **3.9/6 (65%)**, D1-D2 recall **8/14 (57%)**, cDPO derivation **3/4**,
  interview drills **4/9** before resets, coding **51/51**. He wrote the core tensor/math
  bodies with hints; final train snapshot bookkeeping was refactored for him at his request.
- Previous day: **Week 5 Day 02 DONE** — *deriving DPO, and then past where the paper stops*.
  `code/dpo.py`, **42 staged checks**, eight stages, 24 s full / 12 s `--fast`, under
  `-W error::RuntimeWarning`. He implemented all four TODOs. Imports D1's `fit_bt` / `tilt`.
  The derivation itself was produced **cold and graded 3.55/4** in the first block; the day's
  content is the four things AFTER the cancellation. One thesis:
  **DPO is a reward model whose parameterisation happens to be a policy** — the gradient weight
  `σ(−M)` is the same function of the same object as W4 D1's Bradley-Terry weight, so everything
  true of RM fitting is true of DPO training, including D1's existence theorem.
  **Headline 1 — the equivalence, VERIFIED.** `train_dpo` vs `bt_then_tilt` (fit BT over
  sequences, then tilt π_ref exactly) — two algorithms sharing no code path — agree
  **elementwise to 7.772e-16** over all 15 sequences. Wrong-β tilts differ by
  **6.15e-02 / 8.25e-02 / 1.63e-01** at β = 0.25/1.0/2.0 ⇒ **non-vacuous by 2.10e+14**.
  "DPO is equivalent to RLHF" is now a number, not a citation. **The three conditions are
  literally the three steps of the proof**: (1) the policy class represents the tilt exactly,
  (2) the BT MLE exists (Ford, D1), (3) π_ref is the distribution the pairs came from — plus the
  silent fourth, that the PPO side is run to convergence. Condition 1 bites hardest, and not as
  "approximation error": once the class misses the tilt, DPO's optimum is **not even the
  projection** of it in any divergence, because `L_BT` is a finite sum over *dataset pairs*
  weighted by `σ(−M)` ⇒ **DPO allocates approximation error by the DATA distribution, PPO by the
  POLICY's occupancy.**
  **Headline 2 — the gauge is fixed by the simplex** (came out of *his* objection that
  `Σ_y π_ref e^{r̂/β} = 1` is a tautology — it is, and that is the point).
  `r̂ = β log(π/π_ref) = r − β log Z(x)` to **4.441e-16**, offset `β log Z = +1.425614`, and
  `Σ π_ref e^{r̂/β} = 1.000000000000000`. So the parameterisation returns the UNIQUE member of
  the BT equivalence class whose normaliser is 1 — you do not get to choose the gauge, and
  therefore **you cannot lower `r̂(y_l)` without raising `r̂` somewhere else.** Confirmed
  independently: implicit reward == BT MLE **after centring (1.332e-15)** and **not before
  (0.7454 = exactly the offset)**.
  **Headline 3 — the divergence, and a plan claim falsified.** On a 7-pair perfect matching:
  accuracy hits **1.0 at step 2** (KL **0.1129**), step 20 KL **0.8486**, step 20 000 KL
  **1.4660** with `log π(y_l) = −25.9537` (monotone at all 20 000 steps), loss **4.834e-05**,
  losers' mass **0.343396 → 1.649e-09**. **But "the KL grows without bound" is FALSE in a finite
  space** — it is capped by the point-mass bound `max_y −log π_ref(y) = 4.6894`. The harness
  asserts the true version; "unbounded" is the practical statement, not the mathematical one.
  **Headline 4 — both logprobs fall AT THE CONVERGED OPTIMUM**, not on the divergent fixture:
  strongly-connected data, margin **0 → +0.6155** while mean log π(y_w) goes **−3.1249 →
  −4.0003** and log π(y_l) **−3.3174 → −5.4238**. Nothing in the gradient mentions the marginal
  likelihood, so this is what convergence looks like.
  **Headline 5 — the μ ≠ π_ref tax is paid in KL and the loss never reports it.** Same eight
  sequences, columns swapped (one line of the fixture): winners in π_ref's **bulk** reach KL
  **0.6640**, winners in its **tail** KL **3.3932** = **5.11×**, at margins 9.1364 vs 8.3982 and
  comparable loss. Second plan claim falsified: the chosen logprob does **not** rise when
  μ = π_ref — it falls both ways, and on `tail` the sign is a *transient* (positive until ~step
  5000). The stable statement is |Δw|/|Δl| = **0.0941 / 0.0056**. Mechanism for "SFT on the
  chosen completions first": at θ = π_ref the log-ratio is **exactly 0** and every pair starts at
  σ(0)=0.5, so it is not that the ratio *starts* large — the ratio the LABELS require is large,
  and phase one of training is spent moving mass onto the data manifold.
  **Headline 6 — not in the paper: shared prefixes cancel EXACTLY (0.000e+00).** The score-
  function terms on a shared prefix are identical and subtract to zero, so in a TABULAR policy
  the shared prefix of a pair contributes nothing to the DPO gradient ⇒ "pushing y_l down drags
  y_w down" needs shared **parameters**, not shared tokens. Also: **π_ref changes the gradient's
  SCALE but not its DIRECTION** (2.776e-17) ⇒ at gradient level π_ref is a per-pair learning-rate
  multiplier and nothing else.
- Previous day: **Week 5 Day 01 DONE** — *what preference data can and cannot identify*.
  `code/preference_models.py`, **74 staged checks**, nine stages, 38 s full / 8 s `--fast`,
  under `-W error::RuntimeWarning`. He implemented all eight TODOs.
  Day 1 was re-scoped away from re-deriving Bradley-Terry (W4 D1 already did that) toward the
  identification question, and the whole day has one thesis: **the scale of a BT reward is not
  a property of quality, it is a measurement of annotator disagreement.**
  **Headline 1 — Ford/Zermelo: the BT MLE is finite IFF the wins-digraph is strongly
  connected.** So a noiseless annotator on a strictly ordered item set gives a transitive
  tournament — a DAG — and **the MLE is at infinity with infinite data; more data adds edges
  but never a cycle.** Measured with the fit RE-CENTRED every step, so the divergence provably
  is not the additive constant: ‖r̂‖ = **6.22 / 14.42 / 17.77 / 26.80 / 31.03 / 41.21 / 45.66**
  across t = 1e2..2e5, monotone, nll **2.086e-04**, accuracy **exactly 1.0**, max margin
  **49.97**. Control: the *same* items and the *same* 28 comparisons relabelled at ε=0.10 are
  strongly connected and converge to **4.6967** — relative change over the last decade
  **1.081e-01 vs 0.000e+00**.
  **Headline 2 — and it is invisible.** cos(r̂_t, r̂_2t) = **0.9999873 / 0.9999938 / 0.9999985**
  while ‖r̂_2t‖/‖r̂_t‖ = **1.2321 / 1.1580 / 1.1081** ⇒ the direction converges to the max-margin
  separator while the norm grows logarithmically (Soudry et al.). Nothing errors, the loss curve
  is beautiful, accuracy reads 100%.
  **Headline 3 — noise is the regularizer, and it is the SAME result.** Flip noise ε caps the
  saturated per-pair margin at `logit(1−ε)`, matched to **1e-14**: **3.8918 / 2.9444 / 2.1972 /
  1.3863 / 0.6190** at ε = .02/.05/.10/.20/.35. **ε → 0 IS the transitive tournament**, so the
  existence theorem is the ε→0 limit of the cap, not a separate result. Honest scope: with n−1
  parameters against C(n,2) pair frequencies BT is MISSPECIFIED under flip noise, so the
  parametric fit compromises (spread **12.29 / 7.49 / 5.00 / 2.76 / 1.22**) — finite and
  monotone is the claim, not the cap.
  **Headline 4 — RM scale and β are ONE knob.** `tilt(c·r, c·β) == tilt(r, β)` to **3.886e-16**
  ⇒ only `r/β` is identified by the policy. Chain: agreement rate ⇒ scale of r̂ ⇒ meaning of β ⇒
  the KL you land at. Same β=0.5, reward scaled ×0.5/1/2/4 ⇒ KL **1.0156 / 1.8047 / 2.6198 /
  2.8802**. **β is a property of the ANNOTATION PROCESS** — strictly stronger than W4 D5's "β is
  a property of the RM's data" (that was about WHERE the data is; this is about how noisy its
  LABELS are — independent axes, both retune β).
  **Headline 5 — the DPO corollary, one week early.** A preference dataset is a **perfect
  matching**: not strongly connected, min r̂ᵢ **−3.10 → −4.26 → −5.41**, min margin **10.82**,
  nll 2.0e-05, accuracy 1.0. And **in a matching, annotator noise cannot help at all** —
  flipping a label just swaps the winner, and a matching has no cycles under ANY labelling.
  Cycles need an item in ≥2 comparisons ⇒ this is the real argument for K completions per
  prompt and all C(K,2) pairs. Under D2's identification that min r̂ᵢ IS `β log(π/π_ref)`, i.e.
  the rejected logprob ⇒ **"DPO drives log π(y_l) to −∞" is Ford's theorem, not a bug.**
  **Headline 6 — the accuracy ceiling.** Bayes ceiling **0.7675** (n=11200, SE 0.0040); the GOLD
  reward itself scores **0.7747** and the fitted RM scores **0.7747** — it reaches Bayes and not
  a hair more. Accuracy depends only on the SIGN of the margin ⇒ scale-invariant ⇒ structurally
  unable to detect a diverging fit. **A high reported accuracy is the alarm, not the
  reassurance.**
- Previous week: Week 04 — RLHF & PPO for LLMs — COMPLETE (5/5).
- Current day: **Week 4 Day 05 DONE — WEEK 4 COMPLETE** (toy PPO-RLHF loop; `code/ppo_rlhf.py`
  **90 staged checks**, ten stages, under `-W error::RuntimeWarning`, full suite 26 s). He
  implemented five of six TODOs; `train_rlhf` logging, `select_checkpoint` and `ablate` were
  written for him late in the session at his request.
  **Headline 1 — adaptive β IS dual gradient ascent, and its FIXED POINT is the envelope
  theorem.** `g(β)=max_θ L` is a max of AFFINE functions of β ⇒ convex *regardless of the
  primal*; `∂g/∂β = δ − KL` ⇒ `β ← β(1+K_β(KL/δ−1))`. Controller fixed point vs `dE[r]/dKL`
  measured as a secant slope of two enumerated tilts (no β anywhere in its computation):
  δ=0.25 **1.23079 vs 1.17216 (5.00%)**, δ=0.5 **0.83637 vs 0.82765 (1.05%)**, δ=1.0
  **0.56914 vs 0.55652 (2.27%)**, δ=2.0 **0.37505 vs 0.35735 (4.95%)**. Frontier concave at
  **100%** of secant slopes.
  **Headline 2 — the caveat MEASURED.** Dual ascent assumes the inner max is exact; PPO takes K
  clipped epochs. Same controller, exact primal **β=0.6173** vs PPO primal **β=0.1563** at the
  same KL 0.8646 ⇒ **gap 74.7%**. A justification, not a proof, with a number on it.
  **Headline 3 — the chained regression closes.** REINFORCE → +baseline → +critic → GAE(λ) →
  PPO → KL shaping all recovered from ONE code path by flipping switches, elementwise to
  **7.749e-15**. The EGLP/baseline check is non-vacuous by **1.3e11×** (advantages moved
  9.897e-04, gradient moved 7.749e-15).
  **Headline 4 — the ablation grid, 7 arms × 3 seeds, fixed budget.** π_ref gold **4.4191**.
  full **5.1501 ±0.109** (proxy 0.6153, KL 0.361); **no_kl gold 4.9098 ±0.407 with proxy
  0.8728** ⇒ **the inversion: proxy UP, gold DOWN**. reinforce 4.7112; top-p 0.9 **4.8559**.
  Two honest surprises: **no_clip ≈ full** (5.1508) — under a tight KL budget the PENALTY is
  the binding constraint, not the clip — and **no_critic cost NOTHING MEASURABLE** (5.1786 vs
  5.1501, effect **0.0285** against full's own spread **±0.109** ⇒ **a NULL**; corrected
  W6 D1) ⇒ **the critic's value is a function of HORIZON, and this horizon is 5**.
  **Headline 5 — early stopping is the ESTIMATOR.** No-KL run: gold peaks **5.1894 at iter 8
  (KL 0.503)**, ends **3.9372 at KL 3.605** — **below π_ref's 4.4191**. Shipping the final
  checkpoint costs **1.2522**. `select_checkpoint` returns #8 of 40. (This is the case D4 said
  a learned RM did NOT produce; **starving the RM to 30 comparisons produces it**.)
  **Headline 6 — `diagnose` survived contact with real runs**: SILENT on the healthy arm, and
  it fired **`value_divergence` (EV → −5.19e9) on the top-p arm unprompted** — the deadly triad
  assembling itself with the sampler as its off-policy leg. Also measured the **top-p ratchet**
  as a crash: reachable sequences **40 → 39 → 18 → 17 → 12 → 9** until `B < n_minibatches`.
  **The silent-bug probe:** `std(log ρ)` = **0.098999** on a real 4-epoch update and exactly 0
  if `logp_old` is recomputed in the loop — `clip_frac` reads a healthy 0.000 under BOTH
  hypotheses and cannot discriminate.
  **Two `diagnose` calibration findings:** `rel()` divides by a first value that is exactly 0
  (the RM is standardised under π_ref and the policy starts there) ⇒ reason strings read
  "+111362994095%"; and the −20% entropy threshold sits INSIDE the range a healthy budgeted run
  occupies (monotone-gold run at δ=0.4 dropped 26%).
- Previous day: Week 4 Day 03 (token-level PPO + learned value head; `code/token_ppo.py`
  **144 staged checks**). **Headline: a critic cannot bias you as a BASELINE, only as a
  BOOTSTRAP** — at λ=1 *every* critic reproduces the enumerated FD gradient to **2.8e-10**;
  at λ=0.5 the same zero critic is **0.796** off. Bias tracks ‖ẽ‖: Spearman **+0.995**
  (and **−0.995** vs EV). "Wrong level is free" needs FIXED length: **1.5e-10** with EOS
  suppressed vs **0.447** with EOS live. Per-token KL shaping = **0.405113796560** vs
  `β·KL_seq` = **0.405113796560**; EV ceiling **0.4280**; zero-baseline probe **−42.6%** /
  **+128.1%**; c₁ **non-monotone** (keeps **2.5%** of the gain at 5.0).
- Current day: **Week 4 Day 04 done** (RLHF failure modes; `code/failure_modes.py`
  **96 staged checks**, ten stages, all under `-W error::RuntimeWarning`). He implemented
  nine of the ten TODOs; `diagnose` was written for him at his request.
  **Headline 1 — off-policy-by-accident is MEASURED and the perennial is CLOSED.** Against
  the exact enumerated ∇J: on-policy **2.95e-10**, temp 0.7 **0.2284**, temp 0.5 **0.4466**,
  top-p 0.9 **0.1169**, stacked **0.3127**. IS repairs temperature to **2.953e-10** and
  **cannot** repair top-p — **0.1169 → 0.1267, it got WORSE** — with **7.2%** of π's mass
  unreachable under the sampler. `E_μ[ρ] = 1.000000000000` on the temperature arm.
  **Headline 2 — the bug has a nonzero drift with NO reward.** Uncentered constant reward:
  on-policy **9.72e-17** (the EGLP identity), temp 0.7 **7.46e-02**, and the step moves π
  TOWARD μ (**KL(μ‖π) 0.0666 → 0.0587**) — a pure MLE/distillation step. The identity
  `E[∇log π]=0` needs the expectation under π; taken under μ it is false, and that is the
  whole bug. Design note: with centering the advantages are identically zero and every arm
  reports "no bug" — **the control must be run uncentered or it silently passes**.
  **Headline 3 — Goodhart is a VARIANCE result.** Seed spread in gold amplifies
  **89× / 259× / 667×** from weakest to strongest pressure, and at n=30 the spread
  (**±1.60**) **exceeds the entire mean improvement over π_ref** (+0.95) — the outcome is
  seed-determined. Mechanism printable: max |spurious RM weight| **32.07 → 5.82 → 4.29**.
  **Headline 4 — the length derivation PREDICTS a measurement, in both directions.**
  `β*` is a FIXED POINT, not a formula, and the resistance `β·KL̄_t` is **BOUNDED**
  (max **0.1411**/token). κ = **+0.6276** exceeds it ⇒ predicted runaway at every β ⇒
  measured length **5.0000 of 5.0**. Negative control: weak confound κ = **0.0504** <
  **0.2205** ⇒ stopped at **3.9549**. Debiasing: coefficient → **1.71e-16** (algebraic),
  excess length **+0.8656 → +0.1790**, gold **5.7706 → 5.9311**, within-length ranking
  bit-identical.
  **Headline 5 — the deadly triad, with off-policy-ness as its THIRD leg.** Same data,
  same features, only the visitation changed: A = **−3.70** converges (EV +1.000000),
  A = **+1.07** diverges to **4.366e13** (EV **−1.9e27**). MC-target control isolates
  bootstrapping. The value clip bounds the step (**0.010000**) and not the walk (**−6.0**
  after 600 steps).
  **Two corrections to the session plan**, both from measurement: the over-optimization
  turnover does NOT move right with data for a TABULAR RM (no generalization — the fixture
  was rebuilt as a linear model over 23 features), and gold does not fall below π_ref's own
  gold in this env because this gold also pays for length indirectly.
- **Prediction he got wrong, and it was the most valuable part of the day:** he predicted
  top-p would collapse entropy faster. Temperature 0.7 costs **1.8× the entropy and ~2× the
  bias** of top-p 0.9. **Severity and repairability are ORTHOGONAL** — top-p is the gentler
  distortion per step and the unrepairable one.
- Next: **Week 4 Day 5 — the toy PPO-RLHF loop** (capstone), with adaptive β derived as
  dual gradient ascent on the W3 D5 Lagrangian. D4's `diagnose` is imported by it, and D4's
  bounded-resistance result constrains what an adaptive-β controller can achieve.

### Diagnostic (Day 01)

- Top gaps: (1) ML systems / memory-cost reasoning, (2) interview-crisp RL derivations.
- Fluency-probe baseline: 5/5 in ~20 min (slow on `gather` + masking); target <10 min.

## Weekly Progress

| Week | Theme | Status | Deliverable |
|---|---|---|---|
| 01 | LM foundations + PyTorch | Complete (5/5) | Tiny LM train/eval/ablation harness |
| 02 | Transformers + generation | Complete (5/5) | Decoder-only mini-impl + debug suite |
| 03 | RL foundations | Complete (5/5) | Toy policy-gradient implementation (213 checks) |
| 04 | RLHF + PPO | Complete (5/5) | Toy PPO-RLHF loop + failure-mode demos (515 checks) |
| 05 | DPO + preference optimization | Complete (5/5) | DPO trainer + failure-mode lab (385 checks) |
| 06 | GRPO + RLVR | Complete (5/5) | Mini RLVR capstone + cheat sheet (462 checks) |
| 07 | Evals + reward modeling | Not started | Eval harness |
| 08 | Training/inference systems | Not started | Systems design doc |
| 09 | Capstone | Not started | Polished repo + writeup |
| 10 | Interview conversion | Not started | Interview packet |

## Review Queue

Topics to revisit:

- [ ] **pass@k IS A CAPABILITY COMPOSED WITH A SELECTOR YOU DO NOT HAVE** (W6 D3). Three
  selectors, three limits on one policy: oracle **→ 1**, BoN/filter **→ the verifier's
  PRECISION** (measured **0.321655** while pass@k → **1.000000**), majority vote **→ 1 if the
  modal answer is right else 0**. maj@k measured **falling** 0.3563 (k=9) → 0.2764 (k=101)
  while pass@101 = 1.000000 — *the vote gets worse with more compute, and pass@k cannot show
  it because it is monotone in k by construction*.
- [ ] **THE COLLAPSE THAT NEVER HAPPENED, AND ITS SCOPE CONDITION** (W6 D3). `coverage_collapse`
  measured over **24 configurations + 8 seeds × 3 fixtures**: min `Δpass@k_max` = **+0.000e+00**,
  never negative. Mechanism: unsampled gold ⇒ uniform group ⇒ **advantage identically zero** ⇒
  zero gradient. Scope: 15-seq tabular softmax, sound verifier, group baseline. *Drill saying
  the scope condition in the same breath as the number* — the claim is "the collapse needs a
  mechanism", NOT "RL never narrows a policy".
- [ ] **FILTER vs RANKER — the same verifier is two different instruments** (W6 D3). Filter
  accepted-output composition `γ(1−p)/[γ(1−p)+(1−α)p]` = **9.0909%, n-INVARIANT**; its
  unconditional fallback risk reaches **9.0909%** at n=256. Ranker hack probability
  `1 − [1−(1−p)γ]^n` = **pass@n for policy hack mass `(1−p)γ`** = **90.1178%** at n=256.
  **9.91× from the selection rule alone.** Pairs with D2's selection law.
- [ ] **THE PLUG-IN pass@k IS BIASED LOW BY JENSEN** (W6 D3). `1 − (1−c/n)^k` is `f(p̂)` with f
  **concave** ⇒ under-reports: **0.493747 vs 0.651322 = 24.19%** at n=10,p=.1,k=10. Unbiased
  form `1 − Π_{i=0}^{k−1}(n−c−i)/(n−i)`, proved by **exchangeability** (one probability, two
  ways) — *not* by expanding the binomial sum. Also: mean **outside**, not inside (0.6250 vs
  0.9960 on identical data).
- [ ] **BEST-OF-n IS A KL YOU DID NOT BUDGET** (W6 D3). `log n − (n−1)/n` = **0.1931 / 1.2044 /
  3.1745 / 4.5491** nats at n=2/8/64/256, **+0.6853 nats per doubling** measured. **Ties make
  BoN less selective** so a discrete score is strictly cheaper (**2.0507 < 3.1745** at n=64).
- [ ] **A VERIFIER IS A SET, NOT A RATE — and the two containments are not two sides of one
  dial** (W6 D2). `V ⊆ C` = SOUND, `C ⊆ V` = COMPLETE. **Unsoundness: ask whether the set is
  EMPTY** (sup gap = **1.000000** for 1 false positive AND for 7 — a SUPPORT property), and the
  eval statistic ranks the two checkers **backwards** (93.3% vs 53.3%). **Incompleteness: ask
  for the RATE**, because that is the price in rollouts (p_obs .10→.0700, G **22→32**,
  **+45.5%** compute). A 5% FP rate takes G **32→19**: *the two failures move the same number
  in opposite directions, so no single scalar can express verifier quality*
- [ ] **THE HACKED RUN AND THE SOLVED RUN AGREE ON EVERY COLUMN THE REWARD CAN PRODUCE**
  (W6 D2). leaky **0.7694 / 0.9681 / gold 0.0251** vs saturated **0.8361 / 0.9803 / gold
  0.9803**; a k=1 group gives **Â = 0.875000** either way — **bit-identical advantage
  vectors**. No rule reading v_pass, dead_frac, entropy or loss can separate them. **The
  held-out gold column is not a nice-to-have, it is the only column carrying the
  distinction**, and `starved` shows the other sign (gold **0.1383** ABOVE v_pass **0.0319**)
- [ ] **PRECISION IS A FUNCTION OF DIFFICULTY, AT A FIXED VERIFIER** (W6 D2). At a fixed 1% FP
  rate, `P(hack | rewarded)` = **1.10 / 9.09 / 35.25 / 68.86 %** at p = .500/.100/.020/.005.
  And the rates are POLICY-dependent: same checker, tilt π toward its false positives and
  precision goes **0.6720 → 0.0926**. *"Our verifier is 97% accurate" is not a well-formed
  sentence.* Conditioning on ALIVE does NOT move the per-sample rate (0.34902 vs 0.34902) —
  what moves is **concentration** (**554×**)
- [ ] **DEAD GROUPS ARE A RÉNYI ENTROPY OF THE REWARD'S OUTPUT LEVELS** (W6 D2).
  `P(dead) = Σ q_j^G = exp(−(G−1)·H_G(q))`. **SPLIT THE MODE**: at p=.05, G=8, splitting the
  FAILURE mass takes **66.34% → 0.90%** (**73.5×**) from one extra reward level, vs **G=92**
  (**11.5×** rollouts) to buy the same thing. Splitting the SUCCESS mass does **nothing**
  (66.3420% → 66.3420%), reversed at p=.95 ⇒ *reward design and group size are the same knob*
- [ ] **A COMPOSITE REWARD IS A SELF-ANNEALING CURRICULUM** (W6 D2).
  `E[Σ|Â|] = 2(G−1)c(1−c)` per group (emp 1.26493 / 3.50591 / 1.27215 vs closed 1.26/3.50/1.26).
  **Exactly zero share at c ∈ {0,1} for ANY λ, including 1000**; same component at c=.5 carries
  **0.5722**. *Influence tracks INCONSISTENCY, not the coefficient — a mastered component
  cannot be re-weighted back into relevance*, and "a component's share is decaying to zero" is
  the **health** signature, not a fault. λ is lexicographic only below the correctness gap
  (correct at λ=0.99, wrong-but-formatted at λ=1.01)
- [ ] **AN RM'S ACCEPT SET IS DECIDED BY ITS INITIALISATION; A VERIFIER'S IS DATA** (W6 D2).
  A preference dataset is a **MATCHING**, so the BT gradient on an uncompared response is
  **exactly 0.0** and a shift of **1e6** changes the loss by exactly 0 — inits ±50 converge to
  argmax **seq 0 (gold 1.0)** vs **seq 14 (gold 0.0)**. A verifier is the same object under
  both ⇒ **its worst case is computable before training starts**. Do NOT cross-wire the two
  taxonomies (sound/complete vs underdetermined/extrapolating)
- [ ] **UNGUARDED DENOMINATORS PRODUCE NaN, NOT A CRASH — six sightings in W6 D2 alone.**
  `component_shares` returned `[nan]` **with the stage still green**; `increase_percent` hit
  "mean of empty slice". **NaN never compares True, so a threshold built on one silently never
  fires.** Guard every denominator with a plain `if`; run with `-W error::RuntimeWarning` and
  treat "passes with a warning" as a failure. (Companion to W6 D1's
  `np.divide(..., where=)`-without-`out=` finding — same family, third form)
- [ ] **`_tail` READS A LEVEL. A CHANGE NEEDS TWO POINTS FAR APART** (W6 D2). Measuring a trend
  *inside* the tail window cut the leaky run's signal from **+0.2981 to +0.00562** — a **53×**
  loss of signal-to-noise against a threshold of exactly zero
- [ ] **THE DEAD-GROUP LAW — the one GRPO consequence with no PPO analogue** (W6 D1).
  P(zero signal) = **p^G + (1−p)^G**; at G=8 that is **43.05% at p=.90 and 66.34% at p=.95**,
  so N_eff/N = **33.658%**. Raising the pass rate .90→.95 kills a further **23.3%** of the
  batch: **a SUCCESS-driven failure — improving the model degrades its own training signal**,
  and it looks like success in every other column. Confirmed live at **23×** separation
  (dead 0.0354 @ pass 0.4963 vs 0.8042 @ pass 0.9742). Fix is G and it costs: **G=45** for 90%
  alive at p=.95. ⇒ **GRPO trades training memory for INFERENCE compute, and the bill grows as
  the model gets better**
- [ ] **Centring by a statistic OF THE SAME SAMPLES shrinks the gradient EXACTLY** (W6 D1):
  E[ĝ] = **((G−1)/G)·∇J** at finite G — the j=i term survives, the j≠i term dies by
  independence + EGLP. Direction untouched (cos ≥ **0.999992**) so it is absorbable into the
  lr. **At G=1 the gradient is identically zero**, which is the defining property of a
  CONTRAST estimator, not a bug. LOO = centered · G/(G−1) **pathwise** ⇒ removes the bias,
  **cannot** reduce variance, still dies at G=1
- [ ] **A group mean is a HIGH-PASS FILTER, and it is exact at finite G** (W6 D1): anything
  constant within the group is deleted to **2e-16**, so a KL folded into the reward keeps only
  its within-group deviation (**19.15× deleted**) and a UNIFORM drift is **invisible**
  (2.50e-16). ⇒ **a β inside the advantage is an OPEN-LOOP controller — it integrates an error
  it cannot see, saturates, and never errors.** Hence KL goes in the LOSS. Companion: k1 and
  k3 are both unbiased (**0.1517183046** to 10 dp), but **k1 is negative on 50.0% of the
  probability mass** — a negative reading in a minimised loss PAYS the policy to leave π_ref.
  **The k3 preference is FEASIBILITY, not variance** (var ratio reverses 423× → 0.23×)
- [ ] **÷std replaces the VARIANCE weighting with the STD weighting** (W6 D1): Σ|Â| is
  **2G·p(1−p)** centered (= Fisher information for a Bernoulli) vs **2G·√(p(1−p))** normalized;
  amplification is exactly 1/std, boosting k=1 by **1.5119×** over k=4 and collapsing the
  balanced:extreme ratio **2.2857 → 1.5119**. **CONTROL: it COMPRESSES, it does not INVERT.**
  State the control — "the bias reverses the ordering" is the overclaim to avoid
- [ ] **A ratio-based objective is INERT when the ratio is identically 1** (W6 D1): at μ=1 the
  clip binds on **0.000e+00** of tokens at every step vs **0.2222** at μ=8, same lr/seed/task.
  ⇒ **GRPO at μ=1 is REINFORCE + a group baseline + a KL term in PPO's notation.** Generalises:
  before attributing behaviour to a mechanism, check the mechanism is ACTIVE
- [ ] **`np.divide(a, b, where=m)` WITHOUT `out=` returns UNINITIALISED MEMORY** (W6 D1;
  numpy says so explicitly in a warning). It usually *looks* right because fresh pages arrive
  zeroed — **a 20,000-row all-dead batch still returned exactly 0.** ⇒ **when a value cannot be
  tested reliably, assert the CONTRACT the library reports**: run under
  `warnings.catch_warnings` and fail on any warning. Same check also catches a discarded 0/0
  from `np.where` (**seventh sighting** of the not-short-circuit family)
- [ ] **`np.exp` on a torch tensor SILENTLY returns a numpy array** when `requires_grad=False`
  and RAISES when True (W6 D1) — so it passes every value check and then severs the graph.
  Related: **padded positions have ρ = 1, not 0** (logp is zeroed at pad, exp(0−0)=1), so
  forgetting the mask does not add zero, it adds **a full copy of Â per pad slot**
- [ ] **Calibrate a threshold against the GAP BETWEEN DISTRIBUTIONS, not between one healthy
  and one broken observation** (W6 D1). 8 seeds/fixture: dead **0.185 | 0.50 | 0.783**, kl
  **0.300 | 1.00 | 1.288**, ent **0.121 | 0.50 | 0.701** ⇒ 0 of 32 disagree, where a seed-0
  threshold gave 2 of 20 false verdicts. Corollaries: **CUT a rule the instrument cannot
  support** (`length_drift` — E|y| can only travel 1.985→3.0 and every learning run
  lengthens); and **fix a fixture with the MECHANISM, not the knob** (lowering lr destroyed the
  runaway; raising **G 8→32** fixed it, because the day's own dead-group law was contaminating
  it)
- [ ] **A "beat" that is smaller than the baseline's OWN spread is a NULL RESULT** (W6 D1
  correcting W4 D5): `no_critic` **5.1786 ±0.036** vs `full` **5.1501 ±0.109** was recorded as
  a win; the effect is **3.8× smaller than `full`'s own error bar**. Corrected at 5 sites.
  Read the ± before reading the mean

- [x] **FORD/ZERMELO: the BT MLE is finite IFF the wins-digraph is strongly connected**
  (W5 D1). Equivalently: BT is logistic regression on `e_w − e_l`, so this is exactly linear
  SEPARABILITY. ⇒ a noiseless annotator on a strictly ordered set gives a transitive tournament
  (a DAG) ⇒ **MLE at infinity with INFINITE data; more data adds edges but never a cycle.**
  Location invariance is a FLAT direction (constant likelihood, infinitely many finite
  maximisers, gauge-fixable); separability is an ASCENT direction (**zero** maximisers,
  centring cannot help). Measured with the fit re-centred every step: ‖r̂‖ **6.22 → 45.66**
  monotone, nll **2.086e-04**, accuracy **exactly 1.0**
- [ ] **The divergence is INVISIBLE** (W5 D1): cos(r̂_t, r̂_2t) → **0.9999985** while
  ‖r̂_2t‖/‖r̂_t‖ stays **> 1.10** at every decade ⇒ direction converges to the max-margin
  separator, norm grows logarithmically (Soudry et al., implicit bias of GD). Nothing errors,
  loss curve is beautiful, accuracy 100%. **Plot ‖r̂‖ against steps, not the loss**
- [ ] **Noise is the regularizer, and it is the SAME result** (W5 D1): flip noise gives
  `p = (1−2ε)σ(Δ*) + ε`, not σ of anything; the saturated MLE caps at **`logit(1−ε)`**
  (matched to 1e-14: **3.8918 / 2.9444 / 2.1972 / 1.3863 / 0.6190**). **ε → 0 IS the transitive
  tournament** ⇒ existence is the ε→0 limit of the cap. Scope: BT is MISSPECIFIED under flip
  noise (n−1 params vs C(n,2) frequencies), so a parametric fit compromises — finite and
  monotone is the claim, not the cap
- [ ] **RM scale and β are ONE knob** (W5 D1): `tilt(c·r, c·β) == tilt(r, β)` to **3.886e-16**
  ⇒ only `r/β` is identified. Chain: agreement rate ⇒ scale of r̂ ⇒ meaning of β ⇒ the KL you
  land at (same β=0.5, scale ×0.5/1/2/4 ⇒ KL **1.0156 / 1.8047 / 2.6198 / 2.8802**).
  **β is a property of the ANNOTATION PROCESS.** Strictly stronger than W4 D5's "β is a property
  of the RM's data" — that was about WHERE the data is, this is about how noisy its LABELS are.
  `logit(0.68)=0.7538 → logit(0.91)=2.3137` is a **3.07×** larger margin ⇒ cleaning your data
  triples your learning rate in reward-space without editing a config file
- [ ] **A preference dataset is a PERFECT MATCHING** (W5 D1): not strongly connected,
  min r̂ᵢ **−3.10 → −4.26 → −5.41**. And **noise cannot help at all in a matching** — flipping a
  label just swaps the winner; cycles require an item in ≥2 comparisons ⇒ the real argument for
  K completions per prompt and all C(K,2) pairs. ⇒ **"DPO drives log π(y_l) → −∞" is Ford's
  theorem, not a bug**
- [ ] **The RM accuracy ceiling is set by the ANNOTATOR** (W5 D1): Bayes = `E[max(p,1−p)]`;
  measured **0.7675** with the gold reward itself at **0.7747** and the fitted RM at **0.7747**
  — reaches Bayes and not a hair more. Accuracy depends only on the SIGN of the margin ⇒
  scale-invariant ⇒ **cannot** detect a diverging fit. **96% is the alarm, not the reassurance**
  (it implies ε ≲ 4%, i.e. the near-separable regime)
- [ ] **`**` is ELEMENTWISE, not matrix power** (W5 D1): on a bool array `c ** 2.0` upcasts to
  float64 and returns c UNCHANGED — silent, no warning, no shape change. Boolean `@` is what you
  want and numpy does it logically (no overflow). Third sighting of *the wrong operation is
  type-compatible* (W3 D2 `np.zeros(dtype=np.long)`, W2 D2 `TransformerBlock(d, n_heads, n_heads)`)
- [ ] **A symmetric probe cannot detect a sign flip — SECOND sighting** (W5 D1): gradient ASCENT
  on an NLL is undetectable on separable data, where ascent and descent both send ‖r‖ → ∞ and
  differ only in sign. The asymmetric probe is data whose MLE is **finite and known in closed
  form**: 80/20 wins ⇒ `logit(0.8) = 1.3863`, and the buggy fit returned **−317.66**
- [ ] **When a function's output has the same type and shape as one of its inputs, test it on an
  input where "return the input" is WRONG** (W5 D1). `noisy_labels` had a dead store that
  discarded `gold_r` entirely and survived six stages, because the fixture emitted pairs as
  (i, j) with i > j so column 0 was already the gold winner. Fourth sighting of *silent because
  two things that should differ happen to coincide* (W1 D5 `T==V`, W2 D2 `n_heads`,
  W3 D4 `Batch(..., n_iters, rng)`). **A fixture on which the naive answer is correct is not a
  fixture**
- [x] **Adaptive β IS DUAL GRADIENT ASCENT** (W4 D5): `g(β)=max_θ L` is a max of AFFINE
  functions of β ⇒ **convex regardless of the primal being non-convex** — a 1-D convex problem
  sitting on a hopeless one, which is why K_β needs no tuning intuition. Envelope theorem gives
  `∂g/∂β = δ − KL` ⇒ the sign is FORCED. Multiplicative because it is a step in **log β**
  (scale-free) and because **β<0 is INFEASIBLE**, not merely bad — β is the multiplier on an
  inequality constraint. **Fixed point verified against `dE[r]/dKL` on the closed-form frontier
  to 1.05–5.00% at four budgets, by two routes sharing no code.**
- [ ] **The caveat, with a number** (W4 D5): dual ascent's guarantee assumes the inner max is
  solved to optimality; PPO takes K clipped epochs. Exact primal β **0.6173** vs PPO primal
  **0.1563** at the same KL ⇒ **74.7% gap**. Say "justification, not proof" and quote it.
- [ ] **K_β is a GAIN with two opposite failure modes** (W4 D5): 0.01 never settles (overshoot
  1.97 sustained), 0.50 settles in 8, **2.00 never settles with std(log β) = 6.00** (β ringing
  ~400×) and overshoot **6.60**. And the input is a SAMPLED KL — **k1 is negative on 20–44% of
  samples**, so at high gain β chases a sign-flipping quantity. ⇒ **fix the sensor (k3) before
  detuning the gain**; lowering K_β low-passes the noise and permanently costs response speed.
- [ ] **`std(log ρ)` is the probe `clip_frac` cannot be** (W4 D5): `clip_frac = 0.000` is
  produced by BOTH "logp_old recomputed in the loop" and "policy barely moved" — a SYMMETRIC
  statistic. `std(log ρ)` is exactly 0 under the bug and **0.098999** on a real 4-epoch update.
  And the loss cannot help: `L_IS(θ_old) = E[A] ≈ 0` **by construction**.
- [ ] **Early stopping is the ESTIMATOR for the optimum, not regularization** (W4 D5): in SL the
  training optimum is a place you decline to reach; in RLHF the PROXY's optimum is in the wrong
  place and gold's is at a finite interior KL, so convergence OVERSHOOTS. Measured: gold peaks
  **5.1894 at KL 0.503**, ends **3.9372 at KL 3.605** — **below π_ref's 4.4191**, costing
  **1.2522** to ship the final checkpoint.
- [ ] **The ablation grid's two honest surprises** (W4 D5): **no_clip ≈ full** (5.1508 vs
  5.1501) because under a tight KL budget the **penalty**, not the clip, is binding; and
  **no_critic cost NOTHING MEASURABLE** (5.1786 vs 5.1501 — effect **0.0285** vs full's spread
  **±0.109**, intervals nested ⇒ **a NULL**, corrected W6 D1) because **the critic's
  value is a function of HORIZON** and this horizon is 5. Do not generalize "PPO needs a critic"
  from a short-horizon terminal-reward task.
- [ ] **A test is only as sensitive as its input — THIRD SIGHTING** (W4 D5): the check named
  "beta stays strictly positive" PASSED while β could reach exactly 0, because it ran at
  `k_beta·clip = 0.06` and the failure needs `k_beta·clip = 1.0`. **When you write a guard, ask
  what input drives it to its boundary and test THAT.** (D4: `sigmoid(0)=0.5`; W2 D5: the
  toothless cache test.)
- [ ] **A defensive guard that alters SHAPE is worse than no guard** (W4 D5): adding an
  unnecessary `[batch.mask]` to the KL flattened the batch axis and produced a **2.16×** error
  on a 3-row batch — a clean factor of **B** on any sampled batch, which would pin β at the clip
  ceiling forever and look like a working budget. Use `np.where` when you want a mask.
- [ ] **Zero is a MULTIPLICATIVE ABSORBING STATE** (W4 D5): the β floor is not numerical
  hygiene — a dead multiplier means the inequality constraint has silently ceased to exist.
  1e-12 recovers (`→ 2e-12` on the next over-budget step); 0 never does.

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
- [x] Off-policy-by-accident (W3 D3 0.7 → W4 D1 0.65 → W4 D2 0.72 → **MEASURED on W4 D4;
  RETIRED**). Against the exact enumerated ∇J: on-policy **2.95e-10**, temp 0.7 **0.2284**,
  temp 0.5 **0.4466**, top-p 0.9 **0.1169**, top-p+temp **0.3127**. IS repairs temperature to
  **2.953e-10**; on top-p it **cannot** — **0.1169 → 0.1267, it gets WORSE** — with **7.2%**
  of π's mass unreachable. `E_μ[ρ] = 1.000000000000` where the support condition holds.
  Fix ORDER, now owed nothing: (1) sample at T=1.0 untruncated — the only actual fix;
  (2) IS-weight by `π_θ/q` with the ACTUAL sampler — exact for temperature, **impossible for
  top-p** (`supp(π_θ) ⊄ supp(q)`); (3) mitigation only. Keep the three clip-like things
  separate: gradient-NORM (optimizer), PPO RATIO (deliberate off-policy from reuse), top-p
  TRUNCATION (accidental off-policy from sampling). **New and non-obvious: SEVERITY and
  REPAIRABILITY are ORTHOGONAL** — top-p 0.9 is the *gentler* per-step distortion (it removes
  only the last 10% of mass, and entropy is mass-weighted) and the unrepairable one, while
  temperature reshapes the bulk and repairs exactly. He predicted the opposite. What IS
  intrinsic to top-p: the excluded tail is a **one-way ratchet** — it takes the `−p_k` share
  of every update and never the `+1[k=a]` share, so there is no recovery path across iterations
- [ ] **Entropy collapse has a nonzero drift at ZERO reward** (W4 D4). With `r ≡ c` and no
  baseline, `ĝ = c·∇E_μ[log π_θ]` is a **maximum-likelihood step fitting π to μ's samples** —
  pure distillation of the sharpened sampler, no reward information present. Measured: on-policy
  **9.72e-17**, temp 0.7 **7.46e-02**, and KL(μ‖π) **0.0666 → 0.0587** (moves TOWARD μ). It is
  nonzero *because* `E[∇log π] = 0` needs the expectation under **π** and we took it under **μ**.
  General form: the bug is exactly equivalent to optimizing `r̃(y) = w(y)·r(y)` with `w = μ/π`.
  **Design trap:** with centering the advantages are identically 0 and every arm reports "no
  bug" — a control that cannot fail is not a control
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
- [x] **DPO derivation (W5 D2)** — `r = β log(π*/π_ref) + β log Z`; `Z(x)` intractable because the action is a whole COMPLETION (`|V|^T`), not because logits are unavailable; it depends on the PROMPT only ⇒ cancels pairwise. **Produced cold and graded 3.55/4**, and the cancellation was stated *as a reason* on the re-derivation
- [ ] **DPO ≡ tilt of the BT MLE, and the three conditions** (W5 D2) — verified **elementwise to
  7.772e-16**, non-vacuous by **2.10e+14** against a wrong-β control. The three conditions ARE
  the three steps of the proof: policy class represents the tilt / BT MLE exists / π_ref is the
  pair distribution (+ the silent fourth: the PPO side run to convergence). **Could not
  enumerate them on demand (0.25) forty minutes after writing the audit unprompted** — a
  STRUCTURE gap, recover by replaying the proof and asking "what did I use here?"
- [ ] **Where did the KL go — all THREE clauses** (W5 D2, scored 0.6 on two): (i) consumed in the
  algebra, surviving only as π_ref in the ratio and in the gradient only as a scalar weight;
  (ii) off the dataset **nothing** enforces it; (iii) **the MEASURE** — PPO's is reverse KL under
  π_θ on generated samples, DPO's is evaluated under the DATA distribution at fixed completions.
  *The KL in the derivation is not the KL that gets enforced.*
- [ ] **The gauge is fixed by the simplex** (W5 D2) — `r̂ = r − β log Z(x)`, the unique class
  member with normaliser 1; `Σ π_ref e^{r̂/β} = 1` identically ⇒ **cannot lower `r̂(y_l)` without
  raising `r̂` elsewhere.** In reward space `r → r+c` is free; in policy space it is not
- [ ] **Both logprobs fall at the CONVERGED optimum** (W5 D2) — margin **+0.6155** while chosen
  logprob falls **0.8754**; no gradient term mentions the marginal likelihood
- [ ] **The μ≠π_ref tax is paid in KL** (W5 D2) — **5.11×** for the same margin, same eight
  sequences, columns swapped. Hence SFT-on-chosen-first, with a mechanism
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
- [ ] **The four-way failure taxonomy, with the split FIRST** (W4 D4): over-optimization =
  the OBJECTIVE is wrong (hacking, length); optimization pathology = the OPTIMIZER is broken
  (entropy collapse, value divergence). **Identical on a reward curve, OPPOSITE fixes.**
  And **two of the four do not move the KL** — entropy collapse happens INSIDE π_ref's
  support, length exploitation changes one decision (EOS) many times. Caught at KL **0.30**
  and **0.55** by his own `diagnose`. ⇒ KL is NECESSARY, not SUFFICIENT
- [ ] **Goodhart is a VARIANCE result, not just a hump** (W4 D4): seed spread in gold
  amplifies **89× / 259× / 667×** with optimization pressure, and at 30 comparisons the
  spread (**±1.60**) exceeds the entire mean improvement over π_ref (**+0.95**) ⇒ the outcome
  is SEED-DETERMINED and no reward curve shows it. Mechanism printable: max |spurious RM
  weight| **32.07 → 4.29**. ⇒ ≥3 seeds is mandatory, not advisory
- [ ] **Why pessimism works and averaging does not** (W4 D4 check, 0.6 — gave only the
  "averaging fails" half): `E[max] > max E`, so averaging SHRINKS the error but never changes
  its SIGN and the optimizer harvests it anyway. **The minimum converts DISAGREEMENT into a
  PENALTY** — uncertainty becomes cost, so the incentive to go there disappears. It is an LCB
- [ ] **When does lengthening pay** (W4 D4): `Δ = κ + Δr_content − β·E[KL_t]`. **β\* is a
  FIXED POINT, not a formula** — the resistance `β·KL̄_t` depends on β through the policy it
  produces, so it vanishes at both ends and is **BOUNDED** (max **0.1411**/token measured).
  κ = **+0.6276** exceeds it ⇒ **NO β stops the lengthening** (measured 5.0/5.0); weak-confound
  control κ = **0.0504** < **0.2205** ⇒ stopped at **3.95**. ⇒ fix length in the DATA. Also
  (ii) in-distribution **filler has KL_t ≈ 0 ⇒ the penalty gives NO resistance at any β**,
  which is why RLHF'd models are verbose but unsurprising
- [ ] **Length-controlled win rate, and why raw win rate cannot answer** (W4 D4 drill, 0.4 —
  never named the metric): measured a gold reward whose FORMULA penalizes length at −0.4/token
  but whose MARGINAL slope is **+0.1751**, because longer completions contain more content
  tokens. Length correlates with quality even when explicitly penalized ⇒ the metric must be
  length-CONTROLLED, not merely length-aware
- [ ] **Sum vs mean is a LENGTH PRIOR, not a normalization** (W4 D4): mean form lengthens iff
  `Δ > R̄` (beat the running average) vs sum's `Δ > 0`. PPO uses **sum over tokens, mean over
  sequences** ⇒ no built-in brake. Same `Σ_t` vs `(1/T)Σ_t` choice that makes GRPO
  length-biased and that Dr. GRPO changes. Third sighting after D3's `reduce="mean"`
- [ ] **The deadly triad, and its third leg is OFF-POLICY-ness** (W4 D4): function
  approximation + bootstrapping + off-policy distribution; any two are safe. Divergence
  condition is one scalar `A = Σ visitation·φ(γφ′ − φ)`: **−3.70 converges, +1.07 diverges to
  4.4e13** with EV **−1.9e27**, same data and features. MC-target control isolates
  bootstrapping. ⇒ **a distorted sampler does not only bias the ACTOR; it is also the leg that
  lets the CRITIC diverge**
- [ ] **`np.where` is NOT short-circuit — SANITIZE INPUTS, NOT OUTPUTS** (W4 D4, and this is
  now the **sixth sighting in three weeks**: W3 D5's unguarded `np.log(pi_ref)`, W4 D3's
  `nan < 1e-5` guard that INVERTED, and four separate instances on D4). Both branches are
  fully evaluated, so a guard downstream of a dangerous op only LAUNDERS the result. Idiom:
  outer `where` to select, **inner `where` to make the input safe** —
  `np.log(np.where(p>0, p, 1.0))`, `np.where(d>0, n/np.where(d>0, d, 1.0), 0.0)`
- [ ] **`argsort(axis=k)` → `take_along_axis` → `put_along_axis` are a MATCHED SET** (W4 D4).
  A 2-D index array in plain fancy indexing addresses **axis 0 only** (result grows a
  dimension); `axis=` names the one axis whose coordinate is REPLACED by the index, all others
  matched positionally. A sort permutation is a **BIJECTION** ⇒ no collisions ⇒ `np.add.at`
  was never the right tool. Rule: collisions → `add.at`; permutation → `put_along_axis`
- [ ] **Sign errors need an ASYMMETRIC probe** (W4 D4, two of them): `H(uniform over V) = log V`
  catches the entropy sign instantly, but **`sigmoid(0) = 0.5` for BOTH the right and the wrong
  sign** — the symmetric point cannot detect a flip. Same principle as W2 D5's toothless
  cached-vs-uncached test. **Check the input whose answer you know in closed form, and pick the
  one where the two hypotheses disagree most**
- [ ] **Read the parameter list as a CHECKLIST before writing the body** (W4 D4): three
  parameters (`visitation`, `clip`, `mc_targets`) went unused and the signature still
  satisfied the fail-fast probe. Unweighted = uniform = the ON-POLICY case, which is exactly
  where "forgot the weights" and "did it right" agree

## Interview Bank Additions

Add polished answers every week (see `interview-bank/`).

**STATUS AFTER SIX WEEKS AND TWO DAYS: the bank is NO LONGER EMPTY.** On W6 D2 the drill ran
in its new slot and **five entries in `rl-post-training-qa.md` were filled (+233 lines)** — the
first non-empty bank in the project's history. Three entries remain blank (RLHF pipeline, PPO
clipped objective, DPO).

**The caveat, recorded because it changes what the artifact is worth.** He answered the first
three questions himself (D1 **11.0/15**, D2 Q4/Q5 **3.0/10**) and then asked me to *"please
fill it out yourself here."* I complied and said so once. The **reference** value is intact;
the **retrieval** value of those five entries is spent — you cannot be tested on prose you
watched someone else write. Mitigation adopted: every entry now opens with a **cold card** —
the entry's numbers, alone, with no surrounding argument — which is the drillable artifact.
The rule below is unchanged and the next entries are his.

**The evidence that the drill format itself works** (also W6 D2): `d = √KL` failed in Block 1
cold recall and **passed under drill conditions 40 minutes later, same day, no teaching in
between**. The drill is not just an output format; it is a better probe than the quiz.

**Fixed on W6 D1, structurally:** the drill now runs at **1:00-1:15, before the coding block**,
in `shared/templates/session-template.md` and in all 25 remaining session plans (W6 D2-D5, W7,
W8, W9, W10). Rules:

- The drill is **mandatory**. If time runs short, coding depth gets cut, not the drill.
- Questions are asked **cold** - no notes, no scrollback - and answered **out loud**.
- **He** writes the bank entry; model answers come after his, as a diff, never instead.
- **An empty entry means the block did not happen.**

**First back-fill owed (W6 D2): DONE.** The three questions stubbed in
`rl-post-training-qa.md` - "GRPO vs PPO: what changes and why", "Why KL to a reference policy",
and "KL spikes, reward improves, human evals degrade - diagnose it" - are now written, plus the
two W6 D2 questions ("what makes a reward verifiable" and "your math verifier is 97% accurate").

---

## Week 6 Day 4 — Reasoning Data Curriculum (complete)

**Thesis: a curriculum is a controller, and its set point is `p* = 1/(k+1)`, not 0.5.**
Second thesis: **the verifier is a multiplier, not a preference — it does not tilt the
curriculum, it floors it.**

Blocks: 0 (11.0/16) · 1 cold recall (23.15/32) · 2 (1.8/4) · 3 (2.3/4) · 4 interview
drill (8.9/15) · 5 derivations (**3.55/4 — best block of the day**) · 6 code
(`curriculum.py`, 10 stages, **106 checks**) · 7 close-out.

**Learner model.** Numbers are solid; **formulas — especially exponents — are the weak
point.** Wrote `∂pass@k/∂p` with `^k` instead of `^(k−1)` in Block 0 and inverted it to
`∂p/∂pass@k` in Block 4. Block 5 was a step change: correct mechanism *and* correct
attribution (the pass@k asymmetry, not the whitening, breaks the symmetry), plus
base-rate neglect named unprompted.

**Two findings credited to the learner.**
1. **Which wall binds is a consequence of your alive-tolerance, not a fact about the
   system.** He used the 90%-alive criterion (`p_wall = 0.2501`) where I had used
   50%-dead (`0.0830`) and got the *opposite* binding answer. Both internally correct.
   State the tolerance or two people argue past each other for a week while both right.
2. His **W6 D1 two-branch `g_for_alive` correction recurred** in the D4 `G*` table:
   the single-sided `ln(0.1)/ln(1−p*)` gives 4 at k=1 where the both-tails criterion
   needs **5** — wrong at *precisely* the one k whose set point is 1/2. Both
   `session.md` and `shared/numbers-card.md` now carry the correction with credit.

**Rule violation, logged.** I wrote all three D4 interview-bank entries myself after he
asked me to. Their retrieval value is spent. **D5 must re-test those three cold cards
from memory**, plus the REINFORCE variance-sources skeleton.

**Standing debt, now 3 days old:** the two W6 D3 entries in `rl-post-training-qa.md`
(lines ~537–596) still have blank `Short answer:` / `Deeper:` / `Watch out for:`.

**The cut rule, and a refuted prediction.** `mean_p_rising` is cut — but *not* for the
redundancy I pre-registered. Measured, it fires **twice as hard on healthy runs
(+0.796…+0.938) as on sick ones (+0.373…+0.410)**, because rising p *is* learning.
Reason: fires-on-healthy. The same measurement showed the `rise` conjunct inside
`saturated_band` was dead weight; alive-fall carries all the discrimination. Four days,
four cut rules, four distinct reasons: instrument / semantics / the system does not
exhibit it / **it fires on healthy, harder than on sick**.

**New results this day.** `(a,b)` measured at **(1.900, 1.911), R² = 0.9979** — the (2,2)
prediction written *before* the run was confirmed, and it moves the k=8 set point 1.58×.
The 50%-of-peak band narrows **19.07× in p but only 1.43× in logit-p** (ratio 13.33×) —
the quantitative case for logit-space control. At **(G=64, k=1) the whitened set point
does not exist**: flat to 1e-15 over a 0.167-wide plateau. And the whitening tilt is a
**ratio of ratios**, `G/[4(1−2^(1−G))]`, in which the `(G−1)` cancels.

**Eight harness defects** from the private reference run (seventh consecutive day it has
paid). The worst: `_tilt_to_target` bisected a **non-monotone** map, so **40% of fixture
difficulties were silently wrong, with errors up to 0.97**, while every stage quoted them
to four decimals. That one would have invalidated the day. All eight are in `HANDOFF.md`.

**W6 D4 follow-up — the four TODOs implemented, `curriculum.py` at 106/106.** The learner
wrote `signal_curve`, `useful_signal` and `set_point` across several review rounds; the tutor
wrote the `whitened_argmax` helper and `diagnose_curriculum` on request (both logged — that
is **three** tutor-written artefacts this day, after the three interview-bank entries, and the
retrieval value of all of them is spent). Defects the review caught, in order: binomial
coefficients wrong three ways, then **int64 overflow at G=64** after the fix; dead groups
credited `G` instead of `0` under whitening, which flattens `E[ΣA²]` to a constant and deletes
the day's headline; `p_obs` written as a **product** of the two reward cells instead of a sum,
which made the useful fraction ~99 (a share above 1) and **constant in p**; `np.where` used as
a division guard. Every one was found by a check the learner could have run unaided —
`combs.sum() == 2**G`, `ratio ≤ 1`, `p_obs → γ as p → 0`.

**New numbers.** The shipped diagnostic panel calibrates to `CLIFF_SHARE 0.50 / ALIVE_FALL
0.40 / DEAD_LEVEL 0.60 / FLOOR_SHARE 0.125`, all four rules with distinct fire-sets over 6
fixtures × 5 paired seeds. `floor_violated` only works as a **share of budget**, not a mean —
as a mean it is silent on the fixture built for it.

---

## Week 6 Day 5 — Mini RLVR / GRPO Loop (complete)

**Thesis: every `p` needs an event type.** `p_obs` is verifier acceptance and controls reward
deaths; `p_gold` is held-out correctness and controls oracle evaluation/curriculum value;
`p_answer[a]` controls voting. The capstone composes all four prior days without allowing these
events to collapse into one unnamed "success rate."

`code/mini_rlvr.py` is complete at **75/75 checks** in both fast and full regimes. Its ten stages
cover contracts, sensor inversion, rollout laws, truth-boundary mutation, the imported estimator,
controller timing, selector separation, deterministic training, paired ablation, and calibrated
diagnostics.

**Pre-registered ablation result.** Across five paired full-regime seeds, calibrated band tracking
improved final-policy gold pass@8 by **+0.014309 per 1,000 rollouts** versus **+0.010175** for
uniform sampling. The paired effect was **+0.004134** with spread **0.002114**. Conclusion:
**null / unresolved**, not a controller win.

**Calibration result.** Honest raw/calibrated policies are bit-identical (**max θ difference
0.000000**). Calibration reduces sensor error by **0.076343** on the starved fixture and
**0.365958** on the leaky fixture. It does not repair the leaky objective: calibrated gold
pass@8 changes **−0.005075** over 1,344 rollouts and the final verifier–gold gap is **0.240452**.

**Truth boundary.** Mutating `gold` while holding θ, samples, RNG, `accept`, and the data policy
fixed leaves final θ bit-identical. Mutating reachable `accept` changes θ, and all-zero `accept`
is an exact no-op. Perturbed twins remain a separate contamination test and require a clean
post-cutoff/private control.

**Learner model.** The learner supplied the calibration core, correctly explained why gold can
evaluate but not construct a ranker, repaired the KL-level diagnostic, caught the homogeneous-p
inconsistency in the pass@64 interview prompt, and independently separated truth-boundary from
contamination testing. The tutor completed TODO 2 on request and wrote TODOs 3–4 after conceptual
checks; this ownership limitation is recorded in the retrospective.

**Closeout complete.** Day 5 deliverable, numbers card, two outstanding Day 3 interview entries,
new capstone interview entry, week summary, retrospective, progress, and handoff are updated.
Next: **Week 7 Day 1**.
