# Handoff — ai-lab-mle-prep

Living state doc for cross-chat continuity. The user starts a **new chat per day**; this
file (plus `progress.md` and each day's `session.md`) carries context forward. The
`ai-lab-mle-prep-day` skill reads this at the start of a session and updates it at the end.

## Current position

- **Week 01 — COMPLETE** (Days 01-05 done). LM Foundations & PyTorch Fluency.
- **Week 02 — COMPLETE** (Days 01-05 done). Transformers + generation: attention+pos-enc;
  TransformerBlock+TinyGPT; decoding; KV cache; **generation debug lab**.
- **Week 03 — COMPLETE** (Days 01-05 done). RL foundations: MDPs → policy gradient →
  baselines/advantage → GAE → KL-constrained RL. **213 property-based checks across the week**,
  all against exact or finite-difference ground truth. `week-summary.md` + `retrospective.md`
  written.
- **Week 04 — COMPLETE (Days 01-05 done). RLHF & PPO for LLMs. 515 checks across the week**,
  every one against exact enumeration or a central finite difference, all under
  `-W error::RuntimeWarning`. Day 1 pipeline + Bradley-Terry RM (`toy_rlhf.py`, **73/73**);
  Day 2 PPO objective (`ppo_loss.py`, **112/112**); Day 3 token-level PPO + learned value head
  (`token_ppo.py`, **144/144**); Day 4 failure modes (`failure_modes.py`, **96/96**); Day 5
  capstone loop + adaptive β as dual ascent (`ppo_rlhf.py`, **90/90**, 26 s full suite).
  `week-summary.md` + `retrospective.md` written.

- **Week 05 — COMPLETE (Days 01-05 done). Preference optimization: DPO & variants. 385 checks
  across the week.** Day 1 re-scoped
  away from re-deriving Bradley-Terry (W4 D1 did that) toward **what preference data can and
  cannot identify**; `preference_models.py`, **74 checks**. Thesis: *the scale of a BT reward
  is not a property of quality, it is a measurement of annotator disagreement.* Day 2 derived
  DPO cold (**3.55/4**) and spent the day past the cancellation; `dpo.py`, **42 checks**, all
  four TODOs his. Thesis: *DPO is a reward model whose parameterisation happens to be a policy.*
  Headline: **DPO ≡ tilt of the BT MLE, elementwise to 7.772e-16, non-vacuous by 2.10e+14.**
  Day 3 implemented the training contracts on Week 2's TinyGPT; `dpo_trainer.py`, **51 checks**.
  Headlines: cDPO's optimum is exactly D1's ε-cap divided by β; sum-vs-mean slopes
  **0.200000 vs 0.000000**; the exact reference cache costs **8 bytes/pair**; bf16 accumulation
  error at T=256 was **20.525×** a representative `β·margin`; SFT seeded both policy and
  reference; and the chosen-logprob sign changed across fixtures, confirming it is not a
  reference-mismatch diagnostic.
  Day 4 built the variants as **one 5×5 grid** (`variants.py`, **126 checks**, ten stages,
  48.8 s / 12.1 s fast); all four TODOs his, and his output is **byte-identical to the private
  reference implementation**. Thesis: *the link decides whether the margin is finite; the reward
  decides where the length goes.* Headline: **three received claims overturned by measurement**
  — IPO compresses (11.3330 → 4.9000) rather than capping uniformly; SimPO's `1/|y|` AMPLIFIES
  length bias at matched margin; ORPO's odds term flips sign with β while its SFT term does not.
  Plus a fourth found mid-session: **a bounded loss is not a loss with a finite minimiser**, so
  KTO does *not* fix P1 — it climbed `+0.9294` per doubling vs DPO's `+0.7207`.
  Day 5 assembled the failure-mode lab (`dpo_lab.py`, **92 checks**, nine stages, 65.6 s /
  34 s fast). Thesis: *DPO's failure modes are consequences of what the data format can
  identify; the loss is exactly constant along every direction the data do not constrain.*
  Headline: **the β/ε collapse confirmed** — a β-sweep and an ε-sweep matched on
  `Δ̂* = logit(1−ε)/β` land on the same KL to **2.8e-16** — plus an unplanted law,
  **steps-to-target ∝ 1/β² (4.04× / 4.03× / 4.02×)**, and a run whose gold goes
  **+0.7514 → −0.4532** (worse than its own reference) at 100% train accuracy with the loss
  falling monotonically. `week-summary.md` + `retrospective.md` written.

- **Week 06 — IN PROGRESS (Days 01-03 done). GRPO & RL with verifiable rewards.** Day 1
  `grpo.py`, **96 checks**, ten stages, 5.9 s full / 3.2 s `--fast`, **all four TODOs his**.
  Thesis: *GRPO does not delete the value function; it replaces a LEARNED estimate of E[r|x]
  with a MONTE-CARLO one, and unlike a critic that estimate vanishes exactly where the policy
  succeeds.* Headline: **the dead-group law, P(zero signal) = p^G + (1−p)^G** — 43.05% dead at
  p=.90 and **66.34% at p=.95** for G=8, so improving the model *degrades its own training
  signal*; confirmed live at **23×** separation (dead 0.0354 @ pass 0.4963 vs 0.8042 @ pass
  0.9742), and the fix needs **G=45** for 90% alive at p=.95. Supporting: shrinkage exactly
  **(G−1)/G** with direction untouched (cos ≥ 0.999992) and **identically 0 at G=1**;
  annihilation exact at finite G (**2e-16**) so a KL in the reward is an **open-loop
  controller** (19.15× deleted, uniform drift invisible); **k1 negative on 50.0% of the
  probability mass** vs k3's 0.0% ⇒ the k3 preference is *feasibility, not variance*; ÷std
  swaps **2G·p(1−p)** for **2G·√(p(1−p))** and **compresses without inverting**; and the clip
  is inert at μ=1 (**0.000e+00** vs 0.2222 at μ=8, same lr).
  Day 2 `rlvr.py`, **112 checks**, ten stages, 5.9 s full / 4.9 s `--fast`, clean under
  `-W error::RuntimeWarning`, **all four TODOs his**. Thesis: *a verifier is a SET, not a rate
  — everything that matters is a property of V and its two containments with C, and the scalar
  people quote is a property of V AND the policy it was measured under.* Headline: **the hacked
  run and the solved run agree on every column the reward can produce** (leaky 0.7694 / 0.9681
  / gold **0.0251** vs saturated 0.8361 / 0.9803 / gold **0.9803**; a k=1 group gives
  Â = 0.875000 either way, so the advantage vectors are **bit-identical**) — the held-out gold
  column is the only column carrying the distinction. Supporting: `V ⊆ C` ⇒ `v_pass ≤ gold_pass`
  for **all 415 policies at every KL** (max gap −2e-12) with **no scope condition**, while
  `V \ C ≠ ∅` gives **sup gap = 1.000000 for 1 false positive AND for 7** ⇒ hacking is a
  **support** property and the eval statistic ranks the checkers backwards (93.3% vs 53.3%);
  incompleteness is bounded and priced in rollouts (p_obs .10→.0700, **G 22→32 = +45.5%
  compute**, and a 5% FP rate takes 32→19 — *unsoundness BUYS compute*); precision is a
  function of **difficulty** (**1.10 → 68.86%** hack-given-rewarded as p goes .500 → .005 at a
  fixed 1% FP rate) and of the **policy** (precision **0.6720 → 0.0926** under a tilt);
  D1's dead-group law generalises to a **Rényi entropy of the reward's output levels**
  (**SPLIT THE MODE**: 66.34% → 0.90% = **73.5×** from one extra level vs **G=92** for the
  same thing); a composite reward **self-anneals**, share ∝ **2(G−1)c(1−c)**, exactly zero at
  c ∈ {0,1} **for any λ including 1000**; and an RM's accept set is decided by its
  **initialisation** (BT gradient on an uncompared response **exactly 0.0**, a 1e6 shift costs
  0 loss) where a verifier's is **data**.
  Day 3 `sampling.py`, **73 checks**, ten stages, clean under `-W error::RuntimeWarning`,
  TODO 1 and two-thirds of TODO 2 his. Thesis: *pass@k is a CAPABILITY composed with a SELECTOR
  you do not have — the three selectors have three different limits, and only one of them is the
  one pass@k reports.* Oracle **→ 1**, filter **→ the verifier's PRECISION** (**0.321655**
  measured while pass@k → **1.000000**), majority vote **→ 1 iff the modal answer is right**;
  **maj@k measured FALLING** 0.3563 (k=9) → 0.2764 (k=101) against pass@101 = 1.000000, so more
  compute makes the vote worse and pass@k structurally cannot show it. Headline:
  **`coverage_collapse` NEVER FIRED** — min `Δpass@k_max` = **+0.000e+00** across **24
  configurations + 8 seeds × 3 fixtures** — because unsampled gold ⇒ uniform group ⇒ advantage
  **identically zero** ⇒ zero gradient, so **D2's dead group is a safety property** at this
  scope (15-seq tabular softmax, sound verifier, group baseline). Supporting: the unbiased
  estimator `1 − Π_{i=0}^{k−1}(n−c−i)/(n−i)` unbiased to **7.772e-16** by **exchangeability**
  (one probability, two ways) while the plug-in is `f(p̂)` with f **concave** and under-reports
  **24.19%** (0.493747 vs 0.651322) and the mean must go **outside** (0.6250 vs 0.9960 on
  identical data); **`∂pass@k/∂p = k(1−p)^(k−1)` reweights the benchmark** — uniform at k=1, the
  two hardest problems carrying **100.0000%** at k=256, so *changing k changes which benchmark
  you are running*; **T\* moves with k only under heterogeneity** (0.20 → 3.00, transition
  bracketed between k=4 and k=8; homogeneous set gives 0.20 at every k); the dead-group law and
  pass@G are the **same algebra** (`P(dead) = p^G + (1 − pass@G)`, identity to 5.551e-17, dead
  **66.3420%** at p=.05 and p=.95 with failure shares 100%/0%); **filter vs ranker = 10.16×**
  from the selection rule alone (filter **9.0909% n-INVARIANT**, ranker = **pass@n for the hack
  set** = 92.3685% at n=256); best-of-n costs **log n − (n−1)/n** = 0.1931/1.2044/3.1745/4.5491
  nats at n=2/8/64/256 (**+0.6853 per doubling**, and **ties make BoN cheaper**: 2.0507 <
  3.1745 at n=64); and the correlation gap is **UNIMODAL, not monotone** — peaks **0.4749 at
  k=16** and returns to 0, overturning my own session-plan claim. Crossover k reported as the
  **bracket [16, 64, 16, 64, 64]** across five seeds, never as a point estimate.

## SESSION-FORMAT PREFERENCE (he stated this explicitly on W4 D1 — obey it)

**Give FULL blocks every day, regardless of how long the session is taking**, unless he
explicitly asks for something to be shortened or moved to the next day. Do not compress a block
because of the clock. He works through sessions across a whole day (work, then home) and has
plenty of time; wall-clock elapsed is NOT a reason to trim. On D1 an over-eager compression of
Block 3 had to be re-delivered in full.

**RENDERING (new, W5 D2): he reads raw text, LaTeX does NOT render.** Write all maths in plain
Unicode / fenced code blocks — `β`, `π_θ`, `σ(−M)`, `Σ_y`, `∇`, `‖r̂‖`. He asked for this
explicitly during Block 0. Do not emit `\(...\)` or `$$...$$`.

**INTERVIEW DRILLS ARE MANDATORY (he stated this explicitly on W6 D1 — obey it).** They were
skipped on W5 D5 and again on W6 D1 and he called it out. **Every day gets its drill block, and
it is never the block that gets dropped for time.**

- **The fix is STRUCTURAL, and it is already applied.** The drill sat at **1:45–1:55** in
  `shared/templates/session-template.md`, i.e. last before wrap-up, so it was always what the
  clock ate. It has been moved to **1:00–1:15, BEFORE the coding block**, in the template and in
  every remaining day (W6 D2–D5, W7, W8, W9, W10 — 25 files). **When you flesh out a session
  plan, keep that order.** If something must be cut for time, cut coding depth, not the drill.
- **`interview-bank/*.md` is the drill's deliverable, and after W6 D2 it is NO LONGER EMPTY.**
  `rl-post-training-qa.md` now has **five filled entries (+233 lines)** — the first non-empty
  bank in the project's history. The other three files (`evals-qa.md`, `systems-qa.md`,
  `project-stories.md`) are still fully blank. **An empty entry means the block did not
  happen.** At the end of each drill, write his answer into the matching file.
- **Do NOT write the answers for him — and note that on W6 D2 this rule was BROKEN, at his
  request.** He answered the first three questions himself (**11.0/15**), then Q4/Q5 came back
  **3.0/10** with 5 of 10 slots blank, then he said *"please fill it out yourself here."* I
  complied, stated the caveat once, and added a **cold card** (the entry's numbers alone, no
  surrounding argument) to the top of each entry so there is still something drillable. **The
  reference value is intact; the retrieval value of those five entries is spent.** Do not
  repeat this: next time, hold the line, or at most write ONE entry as a worked model and make
  him write the next. On W5 D5 the same request was granted and the block scored nothing.
- **`rl-post-training-qa.md`'s W6 questions are now WRITTEN** — "GRPO vs PPO", "Why KL to a
  reference policy", "KL spikes / reward improves / human evals degrade", "what makes a reward
  verifiable", and "your math verifier is 97% accurate". **The back-fill debt is cleared.**
  Three entries remain blank: RLHF pipeline, PPO clipped objective, DPO. **Drill those next**,
  and make him write them.

## Next session

Start with the prompt: **"Start Week 6 Day 4"**. Read `week-06-*/day-04-*/README.md` and build
the session. **W6 D3 is CLOSED** — `deliverable.md` written, `progress.md`, the Review Queue and
the numbers card updated, `sampling.py` at **73/73**, and **two more interview-bank entries
filled** (Q1 RLHF pipeline, Q3 DPO).

**FOUR MANDATORY RULES FOR D4 (all standing, all earned by direct test):**

1. **Pre-print the answer skeleton with TYPED slots — and NAME THE FUNCTION.** D1 said emit
   slots; D2 said type them; **D3 said that is still not enough.** He pushed back on Block 0
   with *"it is not clear what function you want"* and he was right. A slot must state the
   **function**, **which argument varies**, **which are held fixed**, and **units/precision**:
   not `P(dead): ______` but `P(dead) for q = (0.05, 0.95) [binary reward] : [a percentage,
   4 dp]`. Re-issued that way, the same block went from stalled to **14/17**.
   Two further D3 corrections: **(a)** the "MY OWN EVIDENCE" slot is **retired in its open
   form** — blank six sessions running, because it bundles an *authorship judgement* with a
   *technical skill*. **Assign** a specific result, then ask only number → provenance → scope →
   caveat. **(b)** Never let a label contradict its own formula. My D3 stem said "the
   **FAILURE-side** term `p^G`" — `p^G` is the *success* side. He answered the formula
   correctly and did **not flag the contradiction**; coach him to say *"your label and your
   formula disagree — which do you want?"*, which is a scoring move in an interview.
2. **RUN THE INTERVIEW DRILL at 1:00–1:20, before the coding block.** Non-negotiable. It is a
   better probe than the quiz: D2 it repaired a same-day `√KL` regression, D3 it produced the
   sharpest diagnostic of the week (below).
   **The "he writes the bank entries" rule has now failed twice** (D2 and D3, both at his
   explicit request late in the session). Stop re-litigating it at midnight — **change the
   mechanism**: have him write a **3-bullet skeleton during the drill itself**, while the
   material is hot, and expand it into prose afterwards. Entries written at close-out do not
   get written. D3's two D3-specific stubs are seeded with **cold cards only** and are marked
   `YOURS TO WRITE` — check whether that half-measure worked before deciding.
3. **WRITE AND RUN A PRIVATE REFERENCE IMPLEMENTATION OF EVERY TODO BEFORE HANDING THE FILE
   OVER, then delete it.** Eight days running. **D3 was the highest-yield yet: six defects**,
   listed in the deliverable. The one to internalise: `np.abs(a - b if False else c)` parses as
   `(a-b) if False else c`, because the conditional binds looser than `-`. Calibrate every
   fixture across **8 seeds** before handover.
4. **PREFER THE NULL, AND PREFER THE BRACKET.** D3 produced both in one session. The rule
   `coverage_collapse` was **measured 24 configurations × 8 seeds and never fired** (min
   `Δpass@k_max` = **+0.000e+00**), so it was cut and the *rejection* became the day's headline.
   And the crossover k came out **[16, 64, 16, 64, 64]** across five seeds, so it is reported as
   a **bracket**, not as "k = 16". Seed 0 alone would have produced a slide-ready lie.

**What W6 D4 inherits — measured, in the numbers card:**

- **THE SELECTOR THESIS.** *pass@k is a capability composed with a selector you do not have.*
  Oracle **→ 1**, filter **→ the verifier's PRECISION** (**0.321655** measured while pass@k →
  **1.000000**), vote **→ 1 iff the modal answer is right**. maj@k measured **falling**
  0.3563 (k=9) → 0.2764 (k=101) against pass@101 = 1.000000. Any D4 eval claim must name its
  selector before it names a capability.
- **FILTER vs RANKER is the reusable frame.** Filter composition is **n-invariant**
  (`γ(1−p)/[γ(1−p)+(1−α)p]` = 9.0909%); ranker composition is **`1 − (1−γ)^n` = pass@n for the
  hack set** (92.3685% at n=256). **10.16× from the selection rule alone.** This is D2's
  selection law at inference, and it generalises to anything that picks an argmax of a noisy
  score — which is most of Week 7.
- **BEST-OF-n IS A KL KNOB.** `log n − (n−1)/n` = **0.1931/1.2044/3.1745/4.5491** nats at
  n=2/8/64/256, **+0.6853** per doubling measured. Ties make BoN **less** selective, so a
  discrete score is strictly cheaper (**2.0507 < 3.1745** at n=64). Price any inference-time
  search against the KL budget the training run was constrained to.
- **THE METRIC REWEIGHTS THE BENCHMARK.** `∂pass@k/∂p = k(1−p)^(k−1)`: at **k=1 every problem
  weighs exactly the same**, at **k=256 the two hardest carry 100.0000%**. *Changing k changes
  which benchmark you are running* — directly relevant to Week 7 (evals).
- **THE UNBIASED ESTIMATOR AND ITS PROOF.** `1 − Π_{i=0}^{k−1}(n−c−i)/(n−i)`, unbiased to
  **7.772e-16**, proved by **exchangeability** — one probability computed two ways, never by
  expanding the binomial. The plug-in is `f(p̂)` with f concave and under-reports by **24.19%**
  at n=10,p=.1,k=10. And the mean goes **outside** (0.6250 vs 0.9960 on identical data).
- **THE COLLAPSE THAT NEVER HAPPENED, WITH ITS SCOPE.** GRPO could not reduce pass@k_max in any
  of 24 configurations, because unsampled gold ⇒ uniform group ⇒ **advantage identically zero**
  ⇒ zero gradient. **D2's dead group is a safety property.** Scope: 15-sequence tabular softmax,
  sound verifier, group baseline. It says the collapse needs a **mechanism** — not that RL never
  narrows a policy. Reinforce the scope clause every time he quotes the result.
- **`sampling.py`, `rlvr.py` and `grpo.py` are all importable from `code/`.** New in D3:
  `pass_at_k` / `pass_at_k_exact` / `dpass_dp` / `temper` / `top_p_truncate` /
  `selector_curves` / `bon_distribution` / `bon_kl_continuous` / `correlated_pass_at_k` /
  `train_and_track`. `make_sample_task` flavours `easy`/`mixed`/`sharp`/`leaky`. The third
  array `answer_id[p,y]` beside `gold`/`accept` is what makes maj@k definable at all.
- **RUN EVERYTHING UNDER `-W error::RuntimeWarning`.** Still the single highest-yield check.
- **≥3 seeds is a FLOOR; D1–D3 all used 8 per fixture.** Thresholds come from the *gap between
  distributions*. D3 added the corollary: **if the distributions overlap, DROP THE CONDITION.**
  `mode_locked` lost its spread conjunct because locked spread (0.123–0.379) and healthy spread
  (0.323–0.556) overlap. Fewer conditions, because that is what the measurement supports.

### W6 D3 grading, and the two diagnostic findings

Block 0 **14/17 (82.4%)** · Block 1 **26.5/37 (71.6%)** · Block 2 check **2/3** · Block 3
**11.5/19** · Block 4 drill **11/20 (55%)** · Block 5 **20.25/32 (63.3%)**.
**Total 85.25/128 = 66.6% — his best day of Week 6** (D1 and D2 both sat at 57%).

**FINDING 1 — the second-clause gap is not one gap, it is three, and D3 split it twice.**
Block 4: **MECHANISM slots 9/9 filled, NUMBER slots 0/2, THESIS slots 0/3.** Block 5 split it
again into **MECHANISM vs NAME**: he derived `F(u) = u^n`, `q(u) = n·u^(n−1)` and the integral
`log n − (n−1)/n` correctly, while leaving **"probability integral transform"** and
**"Uniform(0,1)"** blank. *He can execute the object and cannot name it.* Identical to Jensen on
Block 3 — which he then **repaired 40 minutes later** once the name was said aloud. The names
are retrievable; they are just not *first*. **Prescription: drill naming BEFORE deriving.**

**FINDING 2 — wrong-level is now at 14 sightings, and D3 produced the two sharpest.**
*(13)* He substituted the filter's **posterior** (9.09%) where the ranker's **per-sample rate**
(γ = 1%) was required. His arithmetic was flawless — `1 − 0.909091^8 = 53.35%` is exactly right
*for the input he chose*. The modelling was one level off and the arithmetic faithfully
propagated it. **The error family is not carelessness: it is failing to check what KIND of
object a slot wants before filling it.**
*(14)* `pass_at_k` returned `product` = P(all k wrong) instead of `1 − product`, inverting both
edge cases — and the docstring warned about this exact error. **Note the trap: at c = 1 it
returns 0.5, which is correct**, because 0.5 is the fixed point of `x ↦ 1−x`. Spot-check only
that value and it ships. He also wrote `c.any() < 0`, which reduces the array to one bool and
then compares it to 0 — **always False, so the validation never fired on any input** and
`c = 99` silently returned `−206235.25`.

**THE POSITIVE, and it is significant.** Block 5 §(3) scored **5/5 — the first perfect section
of the week** — and it was earned by *spontaneously* reaching back to `∂pass@k/∂p = k(1−p)^(k−1)`
to explain why T\* moves with k. **First unprompted mechanism link in six weeks.** He also
produced the correct non-obvious argument that no clip is needed on the product's numerator
(`n−c ∈ range(k)` whenever `k > n−c`, so the exact zero is hit before any negative factor).

### W6 D2 grading, and what the day established about the format

Block 0 (numbers card) **4.0/6**; Block 1 (cold recall) **8.0/14 (57%)**; Block 2 check
**2.5/4**; Block 3 (selection law) **5.0/6**; Block 4 (interview drill) **14.0/25 (56%)**;
coding **112/112 checks, all four TODOs his**.

- **THE PEDAGOGICAL RESULT OF THE DAY — pre-printed slots must be TYPED, and he found the
  defect, not me.** He wrote: *"A lot of these questions are vague, not sure what you want
  here... please be more clear in the future."* He was **right**, and the diagnosis is precise:
  D1 established that *emitting* the slots beats *asking* for structure, but an untyped
  `____________` still scores like an open question because it does not say what kind of object
  goes in it. Re-issuing Q3 and Q6 with typed slots (`[a number]`, `[the SCOPE condition]`,
  `[which containment fails]`) took Q3 to **3.0/4** and got Q6 filled **on the fifth ask**. A
  `> ## RESULT` box recording this is in D2's `session.md`.
- **Two corollary rules, both from D2 failures.** (a) **Cold recall must test PRIOR material
  only** — several Block 1 questions asked for things the day had not yet taught, which is a
  quiz defect, not a recall failure. (b) **Never name the candidate answers in the stem**; a
  stem that offered three options got exactly those three back.
- **THE SECOND-CLAUSE GAP, 9th sighting, and now with a clean signature.** On drill Q4/Q5 he
  filled **5 of 5 definitional slots** and left **5 of 5 number-and-consequence slots blank**.
  Five for five, both ways. He knows the definitions and does not produce the evidence. **The
  fix is to make the number slot the FIRST slot in the skeleton**, not the last.
- **`[a number]` slots come back as gestures** — twice in the drill, both times with the number
  in hand from earlier the same day. When a slot wants a number, say so *and* say where it
  comes from (file, stage, seeds).
- **A NEW BUG FAMILY, six sightings in one session: unguarded denominators, every one a NaN
  rather than a crash.** `component_shares` returned `[nan]` on the `_fail_fast` probe **with
  the stage still green**; `increase_percent` hit "mean of empty slice". NaN never compares
  True, so **a threshold built on one silently never fires** — the worst possible behaviour for
  a diagnostic. Same family as W6 D1's `np.divide(..., where=)`-without-`out=`, third form.
  Prescription taught: guard every denominator with a plain `if`, and treat "passes with a
  warning" as a failure.
- **A NEW ERROR FAMILY: the TAXONOMY CROSS-WIRE.** In Block 2 he put the *reward model's*
  failure modes (underdetermined / extrapolating) into the *checker's* slots (unsound /
  incomplete) — two orthogonal classifications, one paragraph apart. Watch for it whenever two
  taxonomies are taught adjacently.
- **A REGRESSION, AND ITS REPAIR, ON THE SAME DAY.** `d = √KL` (W4 D5) failed in Block 1 cold
  recall and **passed under drill conditions 40 minutes later**, no teaching in between. This
  is the strongest evidence yet that the drill format extracts what the quiz format cannot.
- **HIS OWN CODING PATTERN HELD: the mathematics was right on first submission; the losses
  were numerics and contract-reading.** TODO 2 was **16/16 first try**. TODO 1 first attempt
  was a no-op (`alpha_all_seq` reduced to `1 − accept`, with `probs` and `gold` missing from
  the numerator); TODO 3 first attempt summed components *before* centering (|·| is not
  linear, so the attribution is destroyed); TODO 4 first attempt measured the change *inside*
  the tail window (signal +0.2981 → +0.00562, a **53×** SNR loss) and had no gold conjunct.
  Each was one review cycle from correct.
- **FIRST UNFORCED SCOPE CONDITION IN SIX WEEKS.** Unprompted, on his own `no_critic` null:
  *"holds only at short time horizon."* This has been asked for repeatedly since W4. Reinforce.

### W6 D1 grading, process gaps, and coding ownership

### W6 D1 grading, process gaps, and coding ownership

Cold recall **4/7 (57%)**; D-recall **10/18 (56%)** and then **4/4 (100%) on a re-ask under
skeleton-first**; derivations **10.75/16 (67%)**; mechanism **2.5/3 (83%)**; design
**13.5/27 (50%)**; coding **19/23.5 (81%) — his best block of the day**. Total
**63.75/98.5 = 64.7%**. **All four TODOs were his**, reversing three consecutive sessions of
declining coding ownership.

- **THE PEDAGOGICAL RESULT OF THE DAY — the skeleton-first boundary condition, found by
  direct test, and it is now a STANDING RULE.** W5 D5 left the open question: is skeleton-first
  a prompted technique or a retrieval-reset one? W6 D1 ran the experiment. *Telling* him to use
  it in Block 0 produced **zero compliance** across Block 1. Then the same question was re-asked
  with the empty slots **pre-printed by me**: **4/4 (100%)**, against **1/2** thirty minutes
  earlier on the identical question. **The operative variable is emitting the slots before any
  content exists — not being told to.** ⇒ **The tutor pre-prints the skeleton every time. Never
  ask him to structure his own answer.**
- **The maths was correct on FIRST submission every single time.** All four coding losses were
  **calibration or numpy/torch numerics**, never derivation: a discarded `0/0`, a `where=`
  without `out=`, `np.exp` on a torch tensor, and a diagnostic threshold set from one observed
  broken value. This is a genuine change in the failure profile — through W5 the losses were
  split between mechanism and mechanics. **Grade his numerics harder than his derivations now.**
- **He found a threshold better than my reference's.** His entropy bound (`ent < 0.5`) beat my
  1.0, which misfired on the nokl fixture; I adopted his into the reference. Worth telling him
  — he does not yet believe his calibration instincts.
- **MY OWN EVIDENCE is still unfilled — FOURTH ask.** The design-block slot asking for a number
  *he* measured, plus a scope condition and a caveat, was skipped again. He now has three strong
  candidates from today alone (the `no_critic` null, the 23× dead_frac separation, the G=8→32
  fixture repair). **W6 D2: pre-print this slot too, and do not accept a general claim in it.**
- **Block 5 (interview drills) was SKIPPED, and he called it out afterwards.** Second
  consecutive week the drills block went unattempted, and the `interview-bank/` files are still
  empty after six weeks. **He has since made drills mandatory — see SESSION-FORMAT PREFERENCE
  above.** The structural fix is applied: the drill now runs at **1:00–1:15, before coding**,
  in the template and in all 25 remaining session plans. The root cause was scheduling, not
  intent — a block placed after the coding block is a block that will be cut.

### Bug families from W6 D1

- **`np.where` / `where=` is NOT short-circuit — SEVENTH sighting, and this one was
  undetectable by value.** `np.divide(a, b, where=m)` **without `out=`** returns *uninitialised
  memory* in the masked lanes; numpy warns explicitly. It passed 12/12 and then a 20,000-row
  all-dead batch still returned **exactly 0**, because fresh pages arrive zeroed. ⇒ **when a
  value cannot be tested reliably, assert the CONTRACT the library reports** — run the check
  under `warnings.catch_warnings` and fail on any warning. That one line catches both the
  `where=`-without-`out=` and the discarded `0/0`.
- **Silent numpy/torch boundary crossings.** `np.exp(torch_tensor)` returns a **numpy array**
  when `requires_grad=False` (graph severed, six checks passed on it) and **raises** when True.
  Related trap he avoided but should be reminded of: **padded positions have ρ = 1, not 0**
  (logp is zeroed at pad ⇒ exp(0−0)=1), so forgetting the mask adds **a full copy of Â per pad
  slot**, not zero.
- **Thresholds set from a single broken observation** (the `kl > 1.5` bound was silent on a
  real KL of 1.231). Rule taught: **air belongs between the healthy MAXIMUM and the broken
  MINIMUM across seeds**, not between healthy and the one broken value you happened to see.
- **Second-clause gap — still unretired, now 7 weeks.** Q3's `Δ̂`/`M` answer was correct but
  stopped at the invariance without saying what it *implies*. Same shape as every prior
  sighting.
- **Wrong LEVEL — 9th sighting** (B1: answered token-level when the question was
  sequence-level). Still the most persistent family in the log.
- **Dead code left in a signature**: `G = adv.shape[-1]` in `grpo_surrogate`, overwritten two
  lines later by `G = logp.shape[0]`. Cosmetic, but it is the same "read the parameter list as
  a checklist" family from W4 D4.

### Harness bugs *I* found by running a private reference before handover (seventh day, still paying)

`_ref_grpo.py` monkeypatches the four TODOs and calls `grpo.main()`. Before he ever ran the
file it found **five real defects**, three of which would have made a stage pass for the wrong
reason:

1. **`seq_token_logprobs_torch` indexed the wrong axis.** It assumed `(n_states, vocab)` but
   `train_grpo` passes `(P, n_states, vocab)`, so it indexed the *prompt* axis with state ids.
   **Every training stage crashed.**
2. **The length-bias check was VACUOUS.** At ρ=1 with a constant Â, the `token` and `batch`
   reductions are equal *by construction* — the check could not fail. Re-expressed as a
   per-token **gradient** ratio via autograd: exactly **3.0** vs exactly **1.0**.
3. **`clip_frac` was read AFTER `opt.step()`**, making the μ=1 claim true for the wrong reason.
   Now measured at loss time, keyed on the **sign condition** (did the clip actually bind), and
   accumulated as a max over inner epochs.
4. **History rows were recorded post-update**, so `h.kl[0] = 7.9e-06` instead of exactly 0 —
   which silently destroys row 0's usefulness as the reference baseline.
5. **All five `diagnose` fixtures were miscalibrated, and `length_drift` had to be CUT.** With
   t_max = 3, E|y| can only travel 1.985 → 3.0 and every learning run lengthens, so healthy
   (1.14) and broken (1.35) have no gap. **Cutting a rule the instrument cannot support is the
   correct move**, and the day says so out loud.

Plus one bug in my own *grading* scaffold: `_fail_fast` touched **every** TODO, so
`stage_surrogate` was blocked by an unrelated unimplemented `kl_penalty`. It now takes the
specific names a stage uses.

And one fixture repaired **by mechanism rather than by knob**: the KL-runaway fixture fired
`dead_groups` at 2 of 5 seeds because `sharp` drives the pass rate off 0.5 and the day's own
dead-group law contaminated it. Lowering `lr` *destroys the runaway* (KL falls to 0.83);
raising **G from 8 to 32** fixed it (dead 0.152–0.544 → 0.004–0.265, clean at 8/8). **8-seed
stress across 4 fixtures: 0 of 32 disagree.**

### W5 D5 grading, process gaps, and coding ownership

Cold recall **3.85/6 (64%)**; D1–D4 recall quiz **9.0/20 (45%)**; derivation/design
**5.0/12 (42%)**; interview drills **not attempted — model answers given at his request**;
coding **92/92**, three of six TODOs his (`rel`, `make_dataset`, and the `train_dpo`
optimiser loop), three written for him at his request (`train_dpo`'s panel, `diagnose_dpo`,
`select_checkpoint`, `ablate`).

- **SKELETON-FIRST DID NOT TRANSFER — and this is now the finding to act on.** D4's headline
  intervention (3.7× on the identical question) was **not used unprompted on D5**, and the
  recall quiz fell to **45%**, the lowest of the week. Two possible readings and they have
  different fixes: (a) it is a *prompted* technique that needs re-issuing every session, or
  (b) it only works under retrieval-reset conditions. **W6 D1: issue it explicitly in Block 0,
  then withhold it in Block 1 and compare.** That is a cheap, decisive experiment.
- **Recall across days is the binding constraint, not comprehension.** He built a 92-check lab
  correctly while scoring 45% on recalling what D1–D4 measured. The gap is retrieval of
  *numbers and named results*, not of mechanism. Consider opening W6 sessions with a 5-minute
  numbers-card drill rather than a cold-recall quiz.
- **Coding ownership DROPPED, third consecutive request.** D3 and D4 he asked for the loop
  plumbing; D5 he asked for `diagnose_dpo`, `select_checkpoint` and `ablate` outright. The
  three he wrote he wrote *well* — `make_dataset` went 20/20 and reproduced every calibrated
  number bit-for-bit. **The pattern is that he writes the mathematical core and delegates the
  scaffolding.** That is a defensible split, but it means the *design* skills (thresholds with
  air on both sides, negative controls, dispatch tables) are being taught rather than practised.
  W6 should hand him one **diagnostic** TODO specifically.
- **Two genuine unprompted catches, matching D4's positive signal.** (1) He asked
  *"`logp_w`, `logp_l` — is this logprob ratio or plain logprob?"* — exactly the right
  question, and the answer is load-bearing (ratios make the pair redundant with the margin to
  1.39e-16). (2) He challenged `len(xs)` on an ndarray. Both are the *"does this actually do
  anything?"* move. **Keep reinforcing it by name.**
- **He fixed `rel()` completely and correctly**, including the guard-ordering subtlety
  (`len(xs) < 2` is total; `denom == 0` requires evaluation; they cannot be combined with `or`).
  6/6 on the port stage.
- **Second-clause gap — unchanged, six weeks.** Oldest open item. Still worth a standing probe.

### Bug families from W5 D5

1. **WRONG LEVEL — now at SEVEN sightings and it is the signature failure.** D5 added two:
   `np.where(cond, reward_a, reward_b)` selecting **reward values** where **sequence indices**
   were required; and `rng.random(cfg.n_pairs)` asking the **config** for a count when
   `cfg.n_pairs` is the knob (default 0) and `len(edges)` is the realisation.
   **The unifying shape of the last two: asking the CONFIG for something only the DATA knows.**
   Standing probes: *"any number with a β in it — divide by β and check whether the story
   flips"*; *"when you write `np.where(cond, A, B)`, ask what TYPE the output must be"*;
   *"never ask the config for a count — measure the array you are holding."*
2. **`cfg.eps` vs `data.cfg.eps`.** Two independent ε in the same file (cDPO smoothing vs
   annotator flip rate). Reading the wrong one makes the cap move with the training config, so
   a β-sweep silently sweeps its own target. Same family as (1).
3. **A copy-paste index that produces a HEALTHY-LOOKING panel.** `delta` indexed `pairs[:,0]`
   twice, so `margin ≡ 0` and `acc ≡ 0` **forever** — not a crash, a panel that reports every
   broken fixture as healthy. Defence: **write the two lines as a pair on adjacent lines**
   (`r_w, r_l = r[pairs[:,0]], r[pairs[:,1]]`) so the asymmetry is visible.
4. **A count where a rate was required.** `(delta > 0).sum()` for `acc`, range `[0, 105]`
   instead of `[0, 1]`. His own docstring listed the range. **This is D4's range probe, and it
   still needs to be run.**
5. **`ndarray` treated as an accumulator.** `history.step.append(...)` — `ndarray` has no
   `.append`, and `np.append` reallocates the whole array every call. The record/builder
   distinction: accumulate in lists, `np.asarray` once at the end.
6. **numpy/torch mixing at no fixed boundary.** Recurred from earlier days. Taught the
   **single conversion boundary** discipline: convert once, at the edge, and let `float()` on
   a multi-element tensor act as a free assertion.
7. **`requires_grad` is an ATTRIBUTE; `requires_grad_` (trailing underscore = in-place) is the
   method.** Generalises: in torch, a trailing underscore means in-place.
8. **Missing `unsqueeze(1)` on a `(K,)` mask against a `(K,2)` tensor.** At K=2 it **silently
   succeeds with a wrong answer** — a pair comparing a sequence with itself. Flipping *rows*
   instead of selecting *values* is closed on the type (long → long), which makes the
   reward/index confusion of family (1) unrepresentable.

### Harness bugs *I* found by running a private reference before handover (sixth day, still paying)

**Always write and run a reference implementation of every TODO before handing the file over.**
On D5 this caught two **wrong fixtures** — failures that would have been blamed on his code:

1. **M4's fixture did not exhibit M4 under a delta rule.** Off-data mass *falls* during
   training (0.4983 → 0.1943), because pushing losers down moves mass onto covered winners.
   The rule had to key on the **LEVEL**, not the change. Re-seeded to `matching, n_pairs=4,
   seed=3` so gold falls monotonically from step 0.
2. **M3's fixture had gold RISING on the original seed.** Searched the ladder seeds and chose
   **seed 11** (corr(gold, |y|) = −0.572, gold drops 0.6311).
3. **`stage_inversion` had vacuous asserts** — rewritten with real assertions *and a healthy
   control*. The control is what turns "gold peaks early" from a statement about training
   length into a statement about the data (healthy gives back **exactly 0.0000**).
4. **W4 D4's `rel()` was ported with two calibration bugs** and fixed as a diff, deliberately,
   so the fix is code and not a memory. The zero-denominator case is **not hypothetical**: at
   θ = θ_ref the implicit reward is exactly 0.0, so `margin[0]` is exactly 0.0 in *every* DPO
   run and the old form returned `1e9 · x[-1]`.
5. **W4 D4's "seed spread grows with pressure" turned out to be a 3-SEED ARTIFACT** (6.09× at
   3 seeds vs 1.04× at 5, same runs). Found by running the reference at two seed counts.

### W5 D4 grading, process gaps, and coding ownership

Cold recall **3.6/6 (60%)**; D1–D3 recall quiz **9.0/14 (64%)**; invent-the-variants **4.2/10
(42%)**; interview drills **2.0/9 (22%)**, then **3.7/4 (93%)** on a retrieval reset; coding
**126/126**, all four TODOs his, output byte-identical to the reference.

- **THE FINDING OF THE DAY — the intervention that finally moved retrieval.** Drill 1 scored
  **1.0/4** with the standing "count the parts" instruction. Fifteen minutes later, the
  *identical question* with a different instruction — **"type the numbered skeleton FIRST, with
  blank slots, THEN fill each slot"** — scored **3.7/4**. Same material, same session, 3.7×.
  **"Count the parts" FAILED; "write the empty structure first" WORKED.** This supersedes the
  counting instruction from D3. Use skeleton-first on every multi-part question from now on,
  and check on D5 whether it transfers without being told.
- **Two genuine unprompted catches — a new positive signal.** (1) He invented **IPO's exact
  symmetric squared cost unaided** in Block 2, before being told the name. (2) He challenged the
  `z_ref` roll — *"if we're just taking a mean, why roll at all?"* — and he was **right**: a roll
  is a permutation, the mean is permutation-invariant, so on unconditional `TabEnv` the mismatch
  is a no-op to 1e-16. That catch is what forced `PromptEnv` and `stage_kto` into existence.
  Both catches came from asking *"does this actually do anything?"* — the exact opposite of the
  known-result reflex. **Reinforce this move by name.**
- **Known-result reflex — FIFTH sighting.** Still the most reliable failure. Keep the probe:
  *"is that the obstruction, or one you already know?"*
- **MINUS-SIGN STREAK BROKEN, then it migrated.** Four consecutive days of a missing `−` on a
  loss; on D4 the loss functions were **clean** (`implicit_reward`, `link`, `variant_loss` all
  correct first time, including the SFT sign, which is algebraically the subtle one). But the
  sign reappeared in a **metric** (`ent` computed as `+Σ p log p`). New standing probe, and it
  is cheaper than any test: **every metric has a RANGE — check it.** A length ∈ `[1, t_max]`, an
  entropy ≥ 0, an NLL > 0. All three of his wrong metrics were *outside the range of the
  quantity they claimed to be*, catchable without running a single step.
- **Wrong LEVEL (loss vs reward) did NOT recur.** `implicit_reward` was correct first time.
  This was Block 2's error on D4 morning and it did not survive into the code.
- **Second-clause gap — unchanged, five weeks.** Still the oldest open item.
- **REINFORCE variance perennial — partial movement.** He named environment stochasticity for
  the first time; policy entropy and off-policy importance-weight variance still missing.
- **Coding ownership: the strongest of Week 5.** He wrote `implicit_reward`, `link`,
  `variant_loss`, and the `train_variant` optimiser loop and the nine-metric `update_record`
  himself. Only the loop *plumbing* (step-0 snapshot, `stop_margin` bookkeeping, return dict)
  was written for him, at his explicit request — the same bookkeeping that was refactored for
  him on D3. **That is the second time; re-ask it cold on D5.**

### Bug families from W5 D4

1. **Python builtins doing object/sequence semantics on things meant as arithmetic — TWICE.**
   `max(tensor, 0)` returned the Python `int` `0` (no `.detach()`), and `lr * grad` where
   `grad` was `autograd.grad`'s **tuple**. The first is the nastiest bug of the day: at
   `π = π_ref` the mean is exactly `0.0`, the comparison ties, `max` returns the Tensor, so it
   **passes step 0 and dies at step 1**. The second is worse if `lr` is an `int` — `2 * (g,)`
   silently *replicates the list*. Use `.clamp(min=0.0)` and `g, = ...`.
2. **Indexing one tensor by `pairs` and not the other.** `lengths` (15 seqs) against
   `seq_logp[pairs[:,0]]` (20 pairs). Per-sequence and per-pair are different objects.
3. **An optimiser chosen inside a comparison function.** `torch.optim.Adam(theta, spec.lr)` —
   two violations at once: an adaptive optimiser (which erases the `margin ~ log t` result while
   still printing a table) and a **per-arm learning rate** in a function whose job is comparing
   arms at a matched one. **The only bug of the day that produces plausible numbers instead of a
   traceback, and therefore the expensive one.**
4. **Metrics outside the range of their own quantity.** `E[|y|] = −10.93`, `H = −0.107`,
   `nll_w < 0`. See the range probe above.
5. **Re-implementing helpers the file already provides.** `exact_kl`, `entropy`, and
   `expected_length` were all present; he hand-rolled all three and got two wrong.
   **Grep before you write.**
6. **`.double()` applied one line above the line that needed it.** `(r_w - r_l)` was already
   floating point; `(r_w > r_l)` was the Bool. Fix the line the traceback names.
7. **Appending tensors instead of `float(...)`.** Converting at the boundary turns the
   conversion into an assertion — `float()` on a 20-element tensor raises, which is exactly the
   `nll_w` missing-`.mean()` bug, caught for free.

### Harness bugs *I* found by running a private reference before handover (the rule keeps paying)

Fifth day running, and it caught five more. **Always write and run a reference implementation
of every TODO before handing the file over.**

1. `prospect` asserted bounded on the **open** interval — false in float (`σ(−50)` underflows to
   exactly `0`). Closed interval at `|r| = 1e6`.
2. `closed_form_optimum` mixed **units** — link-input `M` vs D3's log-ratio `Δ̂ = M/β`.
   `M* = 2.197225` but `Δ̂* = 7.324082`. Added `logratio_optimum()`.
3. **`logsigmoid_margin` has no closed form.** γ is subtracted whichever way the label points,
   destroying BT's antisymmetry. Solved by bisection; the natural guess `logit(f)+γ = 0.905465`
   against the truth `0.533227`.
4. **`ref_mean`/`noref_mean` are not shift-invariant** — a common shift moves `d` by exactly
   `β·c·(1/|y_w| − 1/|y_l|)`, zero only on equal-length pairs.
5. **IPO at `lr = 2.0` → NaN** while DPO is fine at the same lr. An optimiser must never be the
   confound in a comparison stage; `stage_separable` uses `lr = 0.5` for all arms.

### W5 D3 grading, process gaps, and coding ownership

Cold recall **3.9/6 (65%)**; D1-D2 recall quiz **8/14 (57%)**; cDPO derivation **3/4**;
interview drills **4/9** before both retrieval resets; coding **51/51**.

- **Second-clause gap persists.** On sum-vs-mean he gave the sum condition and omitted the mean
  condition. The four-clause SFT question came back as two clauses. Counting the requested parts
  still helps, but does not yet transfer without prompting.
- **Numbers retrieval regressed.** D2 agreement came back only as "within 1e-12"; KL at first
  perfect accuracy was absent. Card reinstalled as **7.772e-16 / 0.1129**. Keep numbers first on
  D4: cDPO cap and length slopes.
- **Three DPO≡RLHF conditions improved but remain imprecise.** He produced all three slots, but
  called representability "full state space reachable" and omitted the silent fourth (RL side
  optimized to convergence). Re-ask by replaying the three proof steps.
- **`tilt(c r,cβ)` recovered**, but the annotation-process consequence required prompting:
  **1.5/2**. This is movement from D2's zero.
- **REINFORCE variance perennial unchanged.** Environment stochasticity was named; policy entropy
  and off-policy importance-weight variance were not. Score **1/3**.
- **Good conceptual objection:** he noticed "91% annotator agreement" is ambiguous. Preserve the
  distinction: adjudicated correctness gives ε=.09 directly; raw pairwise agreement does not,
  absent a restrictive independent-equal-noise model.
- **Coding ownership:** he wrote the sequence-logprob, loss, cache, and SFT cores with hints and
  review. He attempted the full train loop. At his request, the final optimization/snapshot
  separation and nine-field logging bookkeeping were refactored for him. Re-ask that separation
  cold later; code written for him has historically not returned.

### Bug families from W5 D3

1. **Missing `−` on an NLL — THIRD CONSECUTIVE DAY.** The first cDPO expression was a
   log-likelihood. Standing asymmetric fixture: `M=0 ⇒ NLL=+log 2`.
2. **Correct token values, missing sequence reduction.** The first logprob body returned
   `(B,T−1)`, not `(B,)`. The explicit `Σ_t` is what crosses token→sequence.
3. **`finally: return None` overrides the real return and suppresses exceptions.** `finally` is
   restoration only (`model.train(was_training)`); return after it.
4. **Discarded `.reduced(...)` return.** Functional helpers do not mutate a frozen dataclass.
5. **NumPy/PyTorch dtype mismatch in mental model.** `torch.where(cond,1,0)` is int64 and
   `.mean()` refuses it; use `cond.float().mean()`.
6. **Logging state and operator precedence.** `step+1 % k` parsed as `step+(1%k)`; worse, stale
   pre-update records were relabelled as final. Separate differentiable optimization from fresh
   no-grad snapshots.
7. **Absolute policy logprob ≠ implicit reward.** The metric is
   `β·|log π_θ−log π_ref|`, not `β·|log π_θ|`.

### W5 D2 grading, and the state of the process gaps

Block 0 cold recall **3.95/6 (66%)**, up from D1's 65%. Block 1 (the derivation, cold, graded
on four axes) **3.55/4 (89%) — the best graded block of Week 5.** Coding **42/42**, all four
TODOs his own.

- **THE FINDING OF THE DAY — a STRUCTURE gap, distinct from the second-clause gap.** He produced
  the assumption audit **unprompted, as a margin annotation, while deriving** in Block 1. Forty
  minutes later, asked to enumerate the three conditions for DPO≡RLHF, he scored **0.25**. The
  material is present; the retrieval *structure* is not. **He can emit it while the algebra is
  flowing and not on demand as a list.** Fix given, keep enforcing it: **the three conditions ARE
  the three steps of the proof** — recover them by replaying the proof and asking "what did I use
  at this step?" Ask this again on D3 and D5.
- **RETIRED: the Z(x) cancellation stated as a REASON.** Block 1 scored 0.7 (used it, did not
  state it); on the re-derivation later the same session he wrote *"Z(x) cancels as it depends on
  prompt only"* on the page unprompted. Closed within the session. The D1 carry-forward
  (`Z = Z(x)`, prompt only) is also closed — answered correctly, cold, in Block 0.
- **NUMBERS GAP — first real movement in eight sessions.** He read the numbers card and Q4 came
  back with **both** numbers before the prose (45 vs 4.5; actuals 45.66 / 4.6967). Recognition is
  working where retrieval did not. **Keep the read-aloud instruction; it is the intervention that
  moved.** But Q3 (the `tilt(cr,cβ)` identity) was a **REGRESSION to 0.0** — it was D1's stage
  `scale`, one day old. The card had it; he did not use it for that item.
- **Known-result reflex — THIRD sighting, and the standing probe fired correctly.** Asked what
  the DPO gradient does *not* control, he answered "shift invariance" — yesterday's location
  gauge, a result he already owns — instead of the marginal likelihood. Same shape as D1's "does
  the MLE exist?" → location invariance. **Keep asking: "is that the obstruction, or just an
  obstruction you already know?"** It works: it produced the best teaching moment of the day
  (the simplex-fixes-the-gauge result came directly out of correcting this).
- **Second-clause gap — FIFTH week, unchanged.** Block 2 asked for the gradient **plus (a) plus
  (b)**; he delivered the gradient. Block 4's Q1 asked for three clauses; he gave two. Keep
  enforcing "count the parts in the question, write the count in the margin, then answer."
- **He asked for Block 3's answers rather than deriving them.** Given, with the trade named. The
  retention was moved to the interview drill and to the coding checks, which verify Parts 1 and 3
  numerically. Watch whether Block 3's content comes back cold on D4 — if it does not, the W5 D1
  rule ("the code he writes comes back cold; the code written for him does not") generalises
  from code to derivations and should be enforced for both.

### Perennials — status after W5 D2

- **Sources of REINFORCE variance: still the SAME three missing** on the fifth ask — the
  policy's own **entropy**, **environment stochasticity** (name it *because it is exactly zero
  for LLMs*: deterministic append ⇒ all LLM-PG variance is policy-side), and
  **off-policy-ness itself**. He offered "model variance itself" only.
- **`tilt(cr, cβ) == tilt(r, β)` REGRESSED to 0.0 one day after measuring it to 3.9e-16.** Ask it
  every session this week. The consequence clause is what matters: only `r/β` is identified, so
  **β is a property of the annotation process** and "just use 0.1" is a confession.
- **k1/k2/k3 engineering half: not asked on W5 D2.** Still first-clause-only as of D1.
- **`dE[r]/dKL = β`: not asked on W5 D2.** Sign retired, shape (concave in **KL**) was the D1 gap.
- **Ψ ladder: not asked.** Occasional check only.

### Bug families from W5 D2 (four of five are repeats)

1. **Missing `−` on an NLL — SECOND SIGHTING IN TWO DAYS.** `L = logsigmoid(M).mean()` is a
   log-*likelihood*. Diagnostic signature: `analytic == −FD` to **1.161e-10** ⇒ the loss and
   gradient are inconsistent by exactly a constant, so **exactly one of them is wrong**; a random
   disagreement would mean the gradient. Resolved by the **asymmetric probe**: at θ=θ_ref every
   margin is 0 so the NLL is `+log 2 = +0.693147`; the bug read **−0.693147**. Yesterday's
   version of this was *undetectable* because separable data sends ‖r‖→∞ under both signs. **The
   fixture with a known finite answer is what makes a sign bug cheap.**
2. **`pi_final = softmax_np(theta)` — states ≠ sequences, SECOND SIGHTING.** Shape (7,3) summing
   to **7.0** instead of (15,) summing to **1.0**. The trap that lets it survive: **the gradient
   of a sequence quantity has TOKEN shape** (`grad_sequence_logprob` correctly returns (7,3)), so
   shape tells you the level of the *parameters*, not of the *value*. Same family as W3 D5's
   per-state KL 1.053 vs sequence KL 5.056 and W4 D3's "the RM scores completed text, not a
   state". **The one-line test given to him: does it sum to 1, or to n_states?** Also given: only
   `Σ_t` crosses the token→sequence boundary; `softmax` never does.
3. **`history = []` then `history[t].append(...)`** → IndexError. It is a dict keyed by step and
   you assign. **LATENT**: no stage passes `record`, so the suite would have gone 42/42 with the
   bug in it. Untested branch — same reason W3 D5 passed 94/94 on a wrong number.
4. **Returned `pi` where the contract said `(pi_star, r_bt)`** — the only thing between 35/35 and
   42/42. **Read the RETURN annotation as a checklist**, the same way W4 D4 taught for the
   parameter list.
5. **Re-implemented a provided helper** (`implicit_reward` hand-rolled what
   `all_sequence_logprobs` already is). Standing rule restated: *if you finish a file without
   calling a provided helper, that is a bug report about your own code.*
6. *(harness-side, mine)* `‖g_pair‖ ∝ σ(−M)` is FALSE — it is `β·σ(−M)·‖score_w − score_l‖`, and
   the score norm varies **1.81×** across pairs. A proportionality check must divide out
   *everything* that varies.

### Retired on W5 D1

- **CLIPPING ≠ TRUST REGION — 3/3 mechanisms, first time** (was 1/3 on D2, 2/3 on D4 and D5).
  He produced token (`O(Tε²)`), sample (the sign condition, written out exactly: gradient dies
  only when `A>0, ρ>1+ε` or `A<0, ρ<1−ε`), and iteration (**π_old resets ⇒ drift**). The
  **token / sample / iteration** mnemonic is what moved it. Consider the concept closed; only
  the numbers are still missing.
- **RM NOISE as a REINFORCE variance source — named for the first time in five asks.** (His
  proposed *fix* was wrong — "use a critic" — a critic reduces return-sampling variance, not
  reward-signal noise. Fixes are ensembling with a MINIMUM, more data, a better RM.)

### Perennials — status after W5 D1

- **Sources of REINFORCE variance: still the SAME three missing** on the fifth ask — the
  policy's own **entropy**, **environment stochasticity** (worth naming *because it is exactly
  zero for LLMs*: deterministic append ⇒ all LLM-PG variance is policy-side), and
  **off-policy-ness itself**. He now names six of nine including RM noise.
- **k1/k2/k3: definitions and derivations RETIRED** (`k3 ≥ 0` via the minimisation, `E[k3]=KL`
  — both clean, cold). **The engineering half is still first-clause-only**: he said "use k3, it
  is non-negative and unbiased", but **k1 is also unbiased**, so that cannot be the
  discriminator. The two real reasons: (i) k3's variance win is a SMALL-KL phenomenon that
  **reverses** (var(k1)/var(k3) = 423× at KL 0.004 but **0.23×** at ~0.65, and a controller runs
  at δ = 0.5–1.0, i.e. in the reversed regime — so "k3 has lower variance" is FALSE where the
  controller lives); (ii) a controller **integrates** its input, and a negative KL reading is
  not noise around a true value, it is **infeasible** — with k1 negative on 20–44% of samples,
  up to 44% of controller steps are pushed toward the feasibility boundary. **Ask it that way
  again.**
- **`dE[r]/dKL = β`: sign retired, SHAPE still wrong.** He said "the frontier is concave in β";
  it is concave in **KL**. `R(κ) = max{E[r] : KL ≤ κ}` is a max of a linear functional over a
  growing convex set — a perturbation function — and β is its *slope*, which falls as you buy
  more nats. Also worth re-teaching: **the frontier (proxy vs KL, monotone and concave) and the
  Goodhart hump (gold vs KL, rises then falls) are DIFFERENT CURVES.** He asked which one it
  was, which was the right question.
- **Ψ ladder: not asked on W5 D1.** Occasional check only.

### Bug families from W5 D1 (four of six are repeats)

1. **`**` is ELEMENTWISE, not matrix power — NEW, and silent.** On a bool array `c ** 2.0`
   upcasts to float64 and returns c *unchanged*: no warning, no shape change, no error. Boolean
   `@` is the right tool and numpy does it logically. Third sighting of **"the wrong operation
   is type-compatible"** after W3 D2's `np.zeros(dtype=np.long)` and W2 D2's
   `TransformerBlock(d, n_heads, n_heads)`.
2. **Gradient ASCENT on an NLL (`+=` for a loss) — and it is undetectable on the day's data.**
   On separable data an ascent bug and a descent bug both send ‖r‖ → ∞ and differ only in
   **sign**. **Second sighting of W4 D4's "a symmetric probe cannot detect a flip."** The
   asymmetric probe is data whose MLE is finite and known in closed form: 80/20 wins ⇒
   `logit(0.8) = 1.3863`, and the buggy fit returned **−317.66**.
3. **A DEAD STORE that discarded a whole parameter.** `noisy_labels` computed the winner from
   `gold_r` and then *overwrote* it from the raw columns. It survived six stages because the
   fixture emitted pairs as (i, j) with i > j, so column 0 was already the gold winner and the
   dead code and the live code agreed. **Fourth sighting of "silent because two things that
   should differ happen to coincide"** (W1 D5 `T==V`, W2 D2 `n_heads`, W3 D4
   `Batch(..., n_iters, rng)`). New rule: **when a function's output has the same type and
   shape as one of its inputs, test it on an input where "return the input" is wrong.**
4. **Off-by-one between a test and a key**: `if step+1 in record` then `history[step] = ...`.
   Loop from 1 instead of fixing it up at the use site.
5. **Unseeded `np.random.random` init** in a function whose whole purpose is a reproducible
   trajectory comparison (two identical calls: −238.75 vs −239.53). Should be `zeros` — and not
   merely for determinism: zeros makes every initial margin 0 and every initial gradient weight
   exactly σ(0)=½, so the run-away direction is purely a property of the DATA. Same property as
   DPO's initialisation.
6. **Returned a value computed before the last update** (stale `nll`). Same family as W3 D1's
   argmax-on-last-iteration-Q lag.

### Harness lesson — FIFTH time a scaffold design became the day's lesson

`all_pairs` was emitted as (i, j) with i > j, which made "return column 0" a *correct*
implementation on two of three fixtures and let a dead-store bug survive to stage 9. **A fixture
on which the naive answer is right is not a fixture.** Fixed: the column order is now scrambled
(raw column 0 has gold accuracy **0.571**) and `graph` asserts both non-vacuity and that a
deterministic annotator labels every comparison correctly. Verified by reinstating the bug —
the suite now fails at **stage 2 instead of stage 9**. Prior four: W1 D5 noiseless-overfit,
W2 D5 toothless cached==uncached, W3 D3 oracle-baseline, W3 D4 "λ=0 is biased".

### Process win — repeated, and it paid again

The private reference implementation was written and run **before** handover. It caught three
dishonest thresholds — a `rel()` divide-by-zero when the converged fit's change is exactly 0
(**the same family as W4 D5's own `diagnose` calibration bug**), a ceiling tolerance that the
*gold* reward violated by 1.7 SE, and a 2m49s runtime cut to 28 s — plus a fourth during
handover verification (a "loss → 0" threshold unreachable in `--fast`, now expressed relative
to the noisy fit's loss rather than as a constant). **Do this for every future harness.**

### Session-length note

W5 D1 ran ~09:45 to ~22:15, about **12.5 hours**. Front-loading the graded blocks worked — Block
0 and Block 1 were both complete before 11:15, and unlike W4 D5 nothing was skipped for the
clock. **Keep the ordering: cold recall → graded drills → concept → derivation → drills →
coding → deliverable.**

## Completed

- **W6 D2 — RL with verifiable rewards: what "verifiable" buys and what it does not.** Built
  `week-06/code/rlvr.py` (**112 staged checks**, ten stages, 5.9 s full / 4.9 s `--fast`, under
  `-W error::RuntimeWarning`); **all four TODOs his.**
  **Thesis — a verifier is a SET, not a rate.** Everything that matters is a property of `V`
  (what the checker accepts) and its two containments with `C` (what is correct): `V ⊆ C` =
  SOUND, `C ⊆ V` = COMPLETE. The scalar people quote is a property of `V` **and the policy it
  was measured under**, and RL exists to change that policy.
  **Headline — the hacked run and the solved run agree on EVERY column the reward can
  produce.** Four real runs differing only in the checker: leaky **dead 0.7694 / v_pass 0.9681
  / gold 0.0251** vs saturated **0.8361 / 0.9803 / 0.9803**; a k=1 group gives
  **Â = (G−1)/G = 0.875000** whether the acceptance was a hack or a genuine success, so the
  **advantage vectors are bit-identical** and no rule reading v_pass / dead_frac / entropy /
  loss can separate them. `starved` shows the other sign (**gold 0.1383 ABOVE v_pass 0.0319** =
  incompleteness) and `honest` collapses to **one column** (dev exactly 0.0). ⇒ **the held-out
  gold column is not instrumentation hygiene, it is the only column carrying the distinction.**
  **The soundness theorem, and the asymmetry of the two failures:** `V ⊆ C` ⇒
  `v_pass ≤ gold_pass` for **all 415 policies at every KL** (max gap **−2.000e-12**), **no
  scope condition** — the strongest guarantee anything in post-training offers. If
  `V \ C ≠ ∅`, **sup(v_pass − gold_pass) = 1.000000 for 1 false positive AND for 7** ⇒ hacking
  is a **SUPPORT** property, not a rate property, and the eval statistic ranks the two checkers
  **backwards** (93.3% vs 53.3%). Incompleteness by contrast is **bounded and priced in
  rollouts**: 30% extraction failure ⇒ p_obs **0.10 → 0.0700**, G **22 → 32 = +45.5% compute**;
  a 5% FP rate instead takes **32 → 19**, so **unsoundness BUYS compute** — the two failures
  move the same number in opposite directions and **no single scalar can express verifier
  quality**.
  **"97% accurate" is not well-formed, twice over.** Policy-dependence: same checker, same
  arrays, tilt π toward the responses it wrongly accepts (what optimization does) ⇒ gamma
  **0.1515 → 0.7820**, precision **0.6720 → 0.0926**; mass-weighted gamma **0.1515** vs
  count-weighted **0.2222**. Difficulty-dependence (the selection law,
  `P(hack|rewarded) = γ(1−p)/[p(1−α)+γ(1−p)]`): at a **fixed 1% FP rate**, **1.0989 / 9.0909 /
  35.2518 / 68.8581 %** at p = .500/.100/.020/.005 → 1 as p → 0. *Your verifier gets worse as
  your model gets better, with no change to the verifier.* Conditioning on ALIVE leaves the
  per-sample rate **unchanged** (0.34902 vs 0.34902) — what moves is **CONCENTRATION**,
  **0.33020 vs 5.9568e-04 = 554×**. *(Recorded as a correction: 2.4e7× was mis-stated live;
  that is the conditional-on-k comparison h¹ vs h⁴, not the marginal.)*
  **D1's dead-group law generalises to a RÉNYI ENTROPY of the reward's OUTPUT levels:**
  `P(dead) = Σ_j q_j^G = exp(−(G−1)·H_G(q))`. **The rule is SPLIT THE MODE** — at p=.05, G=8,
  splitting the FAILURE mass to (.05,.40,.55) takes **66.3420% → 0.9029% = 73.5× from one extra
  reward level**, where G alone needs **G=92, an 11.5× rollout bill** ⇒ *reward design and
  group size are the same knob, priced differently*. Strictly asymmetric: splitting the SUCCESS
  mass at p=.05 does **nothing** (66.3420% → 66.3420%), exactly reversed at p=.95.
  **A composite reward is a SELF-ANNEALING CURRICULUM.** Centering is linear so
  `A_total = Σ_j A_j` **exactly at finite G** (max dev 2.22e-16) and
  `E[Σ_i |Â_i|] = 2(G−1)·c(1−c)` per group (emp **1.26493 / 3.50591 / 1.27215** vs closed
  **1.26 / 3.50 / 1.26**). Share is **exactly zero at c ∈ {0,1} for ANY λ, including 1000**,
  while the same component at c=0.5 carries **0.5722**; decay c=.50→.90→.99 is
  **3.49869 / 1.24724 / 0.13178 = 26.55×**. *Influence tracks INCONSISTENCY, not the
  coefficient — a mastered component has left the gradient and cannot be re-weighted back in*,
  and "a component's share is decaying to zero" is the **health** signature, not a fault.
  Corollary: a weighted sum is lexicographic only below the correctness gap (argmax correct at
  **λ=0.99**, wrong-but-formatted at **λ=1.01**), and `|A_norm|/|A_cent| = 2.850` so a λ tuned
  under `centered` is not tuned under `normalized`.
  **The RM contrast (W5's identifiability result pointed at reward design):** a preference
  dataset is a **MATCHING** (degree ≤ 1), so the BT gradient on an uncompared response is
  **exactly 0.0** and shifting it by 1 / −37.5 / **1e6** changes the loss by exactly 0; same
  data, inits ±50 ⇒ argmax **seq 0 (gold 1.0)** vs **seq 14 (gold 0.0)**. ⇒ **an RM's accept
  set is decided by its INITIALISATION; a verifier's is the same object under both — data, not
  a fit — so its worst case is computable BEFORE training starts.**
  Fixtures calibrated over **8 seeds each**; two real defects caught by the private reference
  pass — `starved` **escaped starvation** at lr 0.4 (retuned to lr 0.1, beta 0.10) and my own
  reference `saturated` rule **fired on the LEAKY run** until a `gold_pass > 0.70` conjunct was
  added, which is the day's thesis failing in my own code. Diagnostic is silent on **8/8**
  healthy and fires `verifier_hacked` **only** on **8/8** leaky.
  Graded: Block 0 **4.0/6**, Block 1 **8.0/14**, Block 2 **2.5/4**, Block 3 **5.0/6**, drill
  **14.0/25**, coding **112/112**.
  **Pedagogical headline: pre-printed slots must be TYPED — and HE found the defect.** He
  called the Block 1 slots "vague" and was right; `____________` scores like an open question.
  Typed slots (`[a number]`, `[which containment fails]`, `[the SCOPE condition]`) took a
  re-asked Q3 to 3.0/4 and got Q6 filled on the fifth ask. **Second-clause gap, 9th sighting,
  clean signature: 5/5 definitional slots filled, 5/5 number-and-consequence slots blank.**
  **New bug family: unguarded denominators — six sightings, every one a silent NaN**, including
  one that left a stage GREEN while returning `[nan]`. **New error family: the taxonomy
  cross-wire** (RM's failure modes placed in the checker's slots). **A same-day regression AND
  its repair:** `d = √KL` failed cold recall and passed under drill conditions 40 min later.
  **First unforced scope condition in six weeks** (`no_critic` null holds only at short
  horizon). **The interview bank became non-empty for the first time (+233 lines, 5 entries)**
  — but he asked me to write them; see the caveat under the drill rules.

- **W6 D1 — GRPO vs PPO: what replacing the critic with a group mean actually costs.** Built
  `week-06/code/grpo.py` (**96 staged checks**, ten stages, 5.9 s / 3.2 s `--fast`); he
  implemented **all four TODOs**, reversing three sessions of declining coding ownership.
  **Thesis — GRPO does not delete the value function; it replaces a LEARNED estimate of E[r|x]
  with a MONTE-CARLO one, and unlike a critic that estimate vanishes exactly where the policy
  succeeds.** Every property follows from the baseline being computed from the same G samples
  it subtracts from.
  **Headline — the dead-group law, the one consequence with no PPO analogue.** P(a group
  carries zero signal) = **p^G + (1−p)^G**; at G=8 that is **0.78 / 5.77 / 43.05 / 66.34 /
  92.27 %** at p = .50/.70/.90/.95/.99, so N_eff/N = **33.658%** at p=.95 and raising the pass
  rate .90 → .95 kills a further **23.3%** of the batch. **This is a SUCCESS-driven failure —
  improving the model degrades its own training signal, and it looks like success in every
  other column.** Confirmed on two fixtures differing only in the verifier: healthy
  **dead 0.0354 @ pass 0.4963** vs saturated **dead 0.8042 @ pass 0.9742** = **23×** more of
  the batch silent on the run with the better reward. The fix is G and it is expensive:
  **G = 45** for 90% alive at p=.95 (**ln(0.1)/ln(0.95) = 44.891**; G=44 gives 89.533%).
  ⇒ **GRPO trades TRAINING MEMORY for INFERENCE COMPUTE, and the bill grows as the model
  improves.** Memory bill, 7B bf16+Adam: PPO **252 GB** → GRPO **140** (−44.4%) → cached ref
  logprobs **126** → RLVR **112** (−55.6%).
  **The shrinkage is exact and directionless:** E[ĝ] = **((G−1)/G)·∇J** at finite G, measured
  **0.4994 / 0.7496 / 0.8755** at G=2/4/8 with cos ≥ **0.999992**; **at G=1 the gradient is
  identically 0**, the defining property of a contrast estimator, not an edge case. LOO =
  centered · G/(G−1) **pathwise**, so it removes the bias and **cannot** reduce variance.
  **The group mean is a HIGH-PASS FILTER, exactly at finite G:** c(x) deleted to
  **0.000e+00 / 2.220e-16 / 1.210e-14 / 2.039e-10** for c = 0/1/−37.5/1e6, so a KL folded into
  the reward keeps only its within-group deviation (**0.035726 vs 0.6842 = 19.15× deleted**)
  and a *uniform* drift is **invisible** (2.50e-16). ⇒ **β inside the advantage is an OPEN-LOOP
  controller** — it integrates an error it cannot see, saturates, and never errors. Hence KL
  goes in the LOSS. Companion: k1 and k3 are both unbiased against exact enumeration to 10 dp
  (**0.1517183046**), but **k1 is negative on 50.0% of the probability mass** and k3 on
  **0.0%** (min +6.130e-02) — a negative reading in a minimised loss *pays* the policy to leave
  π_ref. **The k3 preference is FEASIBILITY, not variance** (var(k1)/var(k3) reverses: **423×**
  at KL 0.004, **0.23×** at ~0.65).
  **÷std swaps the information weighting for its square root:** Σ|Â| is centered
  **2G·p(1−p)** (= the Bernoulli variance = Fisher information) vs normalized
  **2G·√(p(1−p))**; amplification is exactly 1/std (**3.0237** at k=1 vs **2.0000** at k=4 ⇒
  k=1 boosted **1.5119×**, balanced:extreme collapsing **2.2857 → 1.5119**). **Control: the
  bias COMPRESSES, it does not INVERT** (1.75<4.00 and 5.29<8.00 both hold).
  **The clip is inert at μ=1, exactly:** bound-token fraction **0.000e+00** at every step vs
  **0.2222** at μ=8 with the same lr/seed/task ⇒ **GRPO at μ=1 is REINFORCE + a group baseline
  + a KL term wearing PPO's notation.**
  **A correction carried in from W4 D5: `no_critic` never BEAT `full`** — **5.1786 ±0.036 vs
  5.1501 ±0.109**, an effect **3.8× smaller than `full`'s own spread**. Fixed at 5 sites. It is
  a null result, and a null is still the right evidence for GRPO.
  Diagnostic thresholds calibrated against the **gap between distributions** over 8 seeds per
  fixture (dead **0.185|0.50|0.783**, kl **0.300|1.00|1.288**, ent **0.121|0.50|0.701**) ⇒
  **0 of 32 fixture-seeds disagree**; a fourth rule, `length_drift`, was **cut** because the
  instrument (t_max=3) cannot support it.
  Graded **63.75/98.5 = 64.7%**; coding **19/23.5 (81%)** was his best block.
  **Pedagogical headline: the skeleton-first BOUNDARY CONDITION.** Being *told* to use it gave
  zero compliance; **pre-printing the empty slots gave 4/4 (100%) on a question he scored 1/2
  on thirty minutes earlier.** Standing rule instituted.

- **W5 D1 — What Preference Data Can and Cannot Identify.** Fleshed out all five Week-5
  session plans first (D1 re-scoped off "re-derive Bradley-Terry", which W4 D1 already did).
  Built `week-05/code/preference_models.py` (**74 staged checks**, nine stages, 38 s / 8 s
  `--fast`, under `-W error::RuntimeWarning`); he implemented all eight TODOs.
  **Headline — Ford/Zermelo: the BT MLE is finite IFF the wins-digraph is strongly connected**,
  which is exactly linear separability in the difference features `e_w − e_l`. ⇒ **a noiseless
  annotator on a strictly ordered set gives a transitive tournament (a DAG) and the MLE is at
  INFINITY with infinite data — more data adds edges but never a cycle.** Measured with the fit
  RE-CENTRED every step so the divergence provably is not the additive constant: ‖r̂‖
  **6.22 → 45.66** monotone across t = 1e2..2e5, nll **2.086e-04**, accuracy **exactly 1.0**.
  Control: same items, same 28 comparisons, ε=0.10 labels ⇒ strongly connected, converges to
  **4.6967** (relative change over the last decade **1.081e-01 vs 0.000e+00**).
  **And it is invisible:** cos(r̂_t, r̂_2t) → **0.9999985** while ‖r̂_2t‖/‖r̂_t‖ stays **> 1.10**
  at every decade (max-margin direction, log-growing norm — Soudry et al.).
  **Noise is the regularizer, and it is the same result:** flip noise caps the saturated
  per-pair margin at `logit(1−ε)` to **1e-14** (**3.8918 / 2.9444 / 2.1972 / 1.3863 / 0.6190**),
  and **ε → 0 IS the transitive tournament**. Honest scope: BT is misspecified under flip noise,
  so a parametric fit only gives finite-and-monotone (**12.29 / 7.49 / 5.00 / 2.76 / 1.22**).
  **Scale ≡ β:** `tilt(c·r, c·β) == tilt(r, β)` to **3.886e-16** ⇒ only `r/β` is identified;
  same β=0.5 with the reward scaled ×0.5/1/2/4 lands at KL **1.0156 / 1.8047 / 2.6198 / 2.8802**
  ⇒ **β is a property of the ANNOTATION PROCESS**, and `logit(0.68) → logit(0.91)` is a
  **3.07×** margin increase, i.e. cleaning your data triples your learning rate in reward-space.
  **The DPO corollary, one week early:** a preference dataset is a **perfect matching** — not
  strongly connected, min r̂ᵢ **−3.10 → −4.26 → −5.41** — and **noise cannot help in a matching
  at all** (flipping a label just swaps the winner; cycles need an item in ≥2 comparisons).
  **Ceiling:** Bayes **0.7675** (SE 0.0040) with the gold reward itself at **0.7747** and the
  fitted RM at **0.7747**.
  Cold recall **4.55/7 (65%)**, up from 62.5%/56%. The three unattempted W4 D5 drills, re-asked
  cold: **1.45/3**. Live bugs: `**` used for matrix power (silent no-op on bools), gradient
  ASCENT on an NLL (undetectable on separable data — second "symmetric probe can't detect a
  flip"), a dead store that discarded `gold_r` and survived six stages, an off-by-one between a
  record test and its key, an unseeded init, and a stale return value.
  **Two retirements: clipping ≠ trust region at 3/3 mechanisms, and RM noise as a variance
  source.** Created `shared/numbers-card.md` after the numbers gap hit seven sessions — the
  mechanism switches from retrieval to **recognition** (read it aloud before every recall
  block).

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

- **W3 D4 — GAE, Bootstrapping & the Bias/Variance Tradeoff.** Built `week-03/code/gae.py`
  (**64/64 staged checks, 60 s**): `td_errors`, `k_step_advantage`, `gae_naive`, `gae_reverse`
  (with `dones` reset), `exact_time_indexed_values`, `fit_time_indexed_values`, `weights_gae`,
  `train_gae`. **Design win: the λ-sweep is run against THREE critics chosen along interpretable
  axes** (exact / wrong-LEVEL / wrong-RANKING), which is what made the day's headline findable.
  **Headline (new, and it corrected my own harness): bootstrap bias comes ONLY from the
  state-VARYING part of the critic's error.** `e` enters `δ_t` as `γe(s′)−e(s)`, so a uniform
  offset is a constant, a constant is a baseline, and a baseline is free. Measured: the
  infinite-horizon `V^π` is off by **+2.174 in every row** yet bootstraps at bias **0.0010**
  (λ=0), while a `±1`-per-state skew with the *right* average level biases by **1.5557** — ~1500×
  worse. ⇒ **wrong LEVEL → variance; wrong RANKING → bias**; diagnose a PPO value head by whether
  it RANKS states correctly, not by `|V−G|`.
  **Second result: with an exact critic EVERY λ is unbiased** (bias flat ~0.003 while variance
  rises 5.3×) because the `l≥1` terms are zero-mean ⇒ **λ=0 strictly wins and λ>0 exists only as a
  hedge against a critic you don't trust**. An interior MSE optimum appears only when the critic
  is wrong (λ*=0.9: MSE 3.059 vs 6.954 at λ=0, 3.930 at λ=1).
  **Third: time-indexed `V_t(s)` is the constructive fix for D3's mismatched baseline** — ladder
  at λ=1 is 6.317 (V=0, reproducing D2 to the digit) → 1.765 (D3 pooled `b(s)`) → **1.142 (fitted
  `V_t`)** → 1.147 (exact oracle), i.e. 1.55× on top of D3 and statistically indistinguishable
  from the oracle. **17.5× cumulative** vs D2's `total_return`. λ=1 reproduces D3 and D2
  *elementwise* (`~1e-16`) because telescoping is algebraic, not statistical.
  Cold quiz (D3) **5.70/9**: strong on Q6 entropy (0.85), Q7 variance sources (0.95, 4/7, best of
  four attempts), Q8 NaN (full marks — retire); weak on Q3 `b*` (0.6), Q4 GRPO (0.5 — never
  mentioned the std division), Q5 mismatched critic (0.8, up from 0.25 — got "not biased" cold).
  **Derivation 2.3/4:** Parts 1-2 (telescoping, geometric collapse) clean; **Part 3 0.25** — said
  `Â=0` at λ=0 (mis-evaluated `(γλ)^0`), called `G_t` "infinite horizon" (the D3 horizon slip
  again), skipped γ-vs-λ; **Part 4 0.3 — dropped `∇log π` from the estimator entirely** (same
  failure as D2 Part 2). Taught the slicker `∂H/∂z_k = −p_k(z_k − E_p[z])` via
  `H = lse(z) − E_p[z]`. His best question of the day: *"`G_t` is not `h_t`-measurable, so why
  isn't it biasing?"* ⇒ measurability is required of the **subtracted baseline**, never of `Ψ_t`.
  **Live code bugs:** `gae_reverse` never assigned `A[T-1]` and did `return A[t]` (a scalar —
  `np.allclose` broadcast it silently); `torch.softmax(...)` handed to a numpy einsum
  (`Can't call numpy() on Tensor that requires grad`, traceback naming `.numpy()`, an op he never
  called — D3 heuristic fired again); `return V[:T]` dropping the terminal zero row; and the
  notable one — **`Batch(..., n_iters, rng)` where `batch_size` belongs**, 6.25× too many episodes,
  **which PASSED the test because less noise means better convergence** (tell = wall clock, 102 s
  vs 60 s). Also `critic="exact"` was a `TypeError` that survived a 64/64 green run because no
  stage exercised it, and `history.append(batch.total_returns)` appended arrays not floats.
  ⇒ **green tests ≠ correct code; check the paths the tests don't cover.**
  **My harness was wrong three times again** (unvisitable `(t=0, s≠s0)` bucket; "λ=0 with the
  mismatched critic IS biased" — it isn't, which is how the headline was found; a false
  interpolation claim). Fourth session running where checking the CLAIM beat checking the code.
  Interview drills NOT run (out of clock) — carried to D5.

- **W3 D5 — KL-Constrained RL & the Reference Policy (Week-3 capstone).** Built
  `week-03/code/kl_rl.py` (**94/94 staged checks, 16 s**; `--fast` 6 s): `kl_divergence`
  (support conventions), `tilted_policy` (closed form in log space), `regularized_objective`,
  `forward_kl_objective`, `frontier`, `overoptimization`, `kl_estimators` (k1/k2/k3),
  `shaped_rewards` (InstructGPT per-token mechanism), `train_kl`. **Design win: most of the day
  has a CLOSED FORM, so ground truth became algebra** — the harness checks the closed form
  against `solve_by_ascent`, a black-box finite-difference simplex maximizer that knows nothing
  about exponential tilts (the D2 ground-truth trick moved up a level: there the reference was a
  gradient, here an argmax).
  **Results.** (1) Three exact identities: `J* = β log Z`, `dJ*/dβ = −KL` (envelope theorem),
  and **`dE[r]/dKL = β`** to rtol 1e-4 ⇒ β is the exchange rate in reward-per-nat and the
  frontier is concave in KL. (2) **Reverse KL confines support**: `π*[a] == 0.0` EXACTLY at
  β=2/0.5/0.01 for the highest-reward action when `π_ref` gives it zero mass; forward KL puts
  **0.775** there. Mode-seeking vs mass-covering measured: low-reward-mode mass ref 0.500 →
  forward 0.095 → reverse 0.011. (3) **k3's variance win is a small-KL phenomenon and it
  REVERSES** — var(k1)/var(k3) = 423× at KL=0.004, 19.9× at 0.065, 3.0× at 0.23, **0.23× at
  KL=0.65** (ρ is a likelihood ratio). k2's bias is invisible at small KL (0.1 se) and obvious at
  6.4 se by KL=0.23; k1 reads NEGATIVE on 20-44% of samples. (4) **Goodhart, with a negative
  control**: proxy monotone in 1/β (0.027→1.767) while true reward peaks at **1.1286 @ β=0.16**
  then collapses to **−2.3250**, BELOW the reference's own 0.0270; tilting by the TRUE reward is
  monotone with no hump. (5) Per-token KL shaping IS the sequence KL: measured 0.77787 vs
  0.78974 exact (β=0.3), because `ρ₀` and `P` cancel — the same cancellation that makes PG
  model-free. **k1 pays a BONUS on 24.5% of tokens**; k3 same mean, 1.47× less variance. (6)
  `train_kl` β sweep 0/0.2/1/5 → return 7.928/7.875/5.897/2.966, sequence KL
  5.056/4.834/1.758/0.072; β=0 reproduces D3/D4's "reach & stay" (a regression chaining back
  three days) and every β beats the reference's 2.109.
  **Cold quiz (D4) 4.60/6:** Q1 GAE derivation **1.0** (clean order-swap + geometric collapse);
  Q4 **γ vs λ 1.0 — retire, was missed twice before**; Q3 limits 0.85; Q2 telescoping 0.85
  (dropped `γ^l` off the reward sum twice); Q5 unbiasedness 0.75 (`l≥1` case perfect; `l=0` had a
  false equality `δ_t = G_t − V(s_t)` and neither case named where `V = V^π` is consumed);
  **Q6 0.15 — stated the λ/bias relation BACKWARDS**, i.e. inverted his own Day-4 headline two
  pages after tabling it correctly. Perennials: P1 variance sources **0.80, 6/7, best of five**
  (got the LLM sparse-terminal one at last; missed reward-model noise; filed Rao-Blackwellization
  as a source); **P2 off-policy 0.45, REGRESSED from 0.70**.
  **Derivation 2.85/4:** Part 1 (Lagrangian → closed form) **1.0**, including the `+1` from
  `∇(π log π)`; Part 2 0.8 (algebra right, `dJ*/dβ < 0` noticed, but did not name `β log Z` as a
  log-sum-exp / soft max, nor the Week-1 link); **Part 3 0.25 — BOTH limits wrong**: `β→∞` given
  as uniform (that's the entropy-bonus special case) and `β→0` as an unrestricted argmax (must be
  over `supp(π_ref)`), with the consequence chain absent; Part 4 0.8 — **saw the DPO cancellation
  unprompted**, but said `Z` is intractable because "we lack `π_ref` for all actions" rather than
  because the action is a whole COMPLETION (`|V|^T`).
  **Live code bugs, both in named recurring classes.** (1) `kl_divergence` left `log_q`
  unguarded: right answer for `p>0,q=0` by IEEE accident, **`0*inf = nan` when BOTH are zero** —
  and **94/94 passed because no fixture had `p` and `q` vanishing at the same index**, though
  that is the NORMAL case in RLHF (every tilted policy inherits the reference's zeros, so
  `KL(π*_{β1}‖π*_{β2})` was `nan`). Then inverted `np.where((p>0)&(q==0), res, np.inf)`, stamping
  `inf` on every healthy row. (2) **`kl_history` measured the wrong quantity and every check
  still passed**: unweighted mean of per-state KL (1.053) instead of `Σ_t E_{d_t}[KL_t]` (5.056),
  appended as a `(2,)` ARRAY not a float (D4's `history.append` bug one container over). All four
  assertions tested monotonicity in β, which a wrong-by-a-factor quantity preserves. **The
  available tell: a provided helper (`state_distributions`) the code never called — an unused
  fixture in a file that told you to use it is a bug report.** Coding block 0.8 → 0.9 after fixes.
  **Interview drills: D1 (why a reference) 0.6** — had both failure modes, β monotonicity, the
  support mechanism and KL logging, but never named the KL DIRECTION or mode-seeking, gave no
  elicitation-vs-capability consequence, and omitted entropy from the instrumentation;
  **D2 (reward up / eval down) 0.35** — named reward hacking and "raise β" and nothing else, no
  numbers despite being asked twice; **D3 (credit at EOS) 0.5** — named both mechanisms
  (estimator-side GAE and reward-side per-token KL) but not how they interact, and no λ numbers.
  Wrote deliverable (with three polished interview answers) + week-summary + retrospective.
  **WEEK 3 COMPLETE.**

- **W4 D1 — The InstructGPT Pipeline & the Reward Model.** Built
  `week-04-rlhf-ppo/code/toy_rlhf.py` (**73/73 staged checks, 13.3 s; `--fast` 3.4 s**), the
  week's infrastructure: `TokenEnv` (vocab 4 with a real EOS, `t_max=4`, 40 states,
  **121 enumerable terminating sequences** = 40 eos + 81 truncated), `enumerate_sequences`,
  `sequence_logprob`, `exact_J`, `exact_sequence_kl`, `tilted_sequence_dist`, plus the RM half:
  `bt_loss_and_grad`, `fit_reward_model`, `sample_preferences`. Days 2-5 import the first five.
  **Ground-truth pattern transplanted intact from Week 3**: enumeration replaces Bellman backups,
  so `exact_J` is exact and unbiasedness stays an assertion.
  **Headline 1 — per-prompt identifiability, asserted as an experiment.** Refit from a different
  init on identical data: within-prompt differences agree to **8.22e-15**, raw scores differ by
  **1.7355**; the per-prompt mean drifts **6.9e-15** from its init, i.e. data cannot move it at
  all. Cross-prompt pairwise accuracy on raw scores **0.445 — worse than chance** — vs **0.974**
  within-prompt and 0.976 cross-prompt after centering. Raw recovery err 8.6088 → **0.2743**
  centered (**31.4×**).
  **Headline 2 — the KL penalty is a confidence region for the RM.** `pi_ref = pi_SFT` = the RM's
  own sampling distribution, so KL literally measures distance from its training data. Error
  **0.177 on-distribution → 4.89 at sequence KL 4.44 (27.6×)**, monotone, on BOTH a gold-tilted
  and a proxy-tilted ladder. Data-size sweep: over-estimation **3.231 → 0.038 (86×)** from 300 to
  20 000 comparisons ⇒ **β is a property of the RM's DATA, not of the policy**.
  Other verified numbers: sum-to-one `1.000000000000000` under two unrelated thetas; `exact_J` vs
  12k-rollout MC mean z **−0.01** over 5 seeds; enumerated sequence KL **0.96993705** == the
  occupancy-weighted route (unweighted per-state mean would read 0.50890693 — **D5's actual bug
  wired in as an assertion**); BT analytic grad vs central FD **1.785e-10**; annotator noise
  0→0.40 drops RM accuracy 0.969→0.844 (±0.034, 3 seeds).
  **Cold recall (new block) 4.15/6**, quiz **2.6/5**, derivation **2.35/4**. Clean: closed form +
  `J* = β log Z` (1.0), why `Z` is intractable (1.0), wrong level vs wrong ranking (0.95), Eq. (2)
  terms (1.0), BT NLL + gradient calculus (1.0 on the algebra). **Two sign/limit errors worth
  re-asking: he gave `dE[r]/dKL = −β` (it is +β; he had collapsed it with `dJ*/dβ = −KL`, which
  have OPPOSITE signs — mnemonic: β is a price, prices are positive) and did not name the
  frontier as CONCAVE in KL.** Also `1/V` for `1/K` on the β→0 tie set.
  **Live code review (his implementations were strong — every stage passed first or second try):**
  `tilted_sequence_dist` first shipped `beta==0` / `beta==inf` special cases — ~7 lines, **4 bugs,
  zero test coverage** (`len()` of a scalar argmax, zeroing forbidden rewards instead of `-inf`,
  `list > int`, returning log-probs) — the W3 D4 `critic="exact"` lesson exactly; and it
  exponentiated an unnormalised log-weight, correct for every β the ladders use and `nan` only at
  the one β the check was placed at. Punchline taught: **the stable log-space form makes both
  special cases unnecessary** (softmax gives 1/K on ties free, `softmax(log p_ref) = p_ref`, and
  `log 0 = -inf` gives exact support confinement). Also caught: `enumerate_sequences` rebinding
  `frontier` *inside* the loop iterating it (right answer, works only via Python's iterator
  semantics), `np.long` (version- and platform-dependent; and the spelling of the W3 D2 bug), and
  a `list + ndarray` that would silently concatenate if the caller passed a list.
  **MY HARNESS WAS WRONG FOUR TIMES BEFORE ANY LEARNER CODE RAN** (fifth session running):
  (1) a correlation assertion over **6 prompts** (read −0.666) → replaced with the deterministic
  two-init refit, which is strictly better; (2) "over-estimation grows with KL" — it falls, and
  `E[max(err,0)]` DEGENERATES as β→0 because the tilted distribution collapses onto one sequence
  → moved to a data-size axis where the effect is real and 86×; (3) "off-distribution error >2×
  at every data size" fitted to one dataset (across 4 seeds the middle sizes reach 1.6×) →
  direction everywhere, magnitude only where converged; (4) regret asserted monotone — it hangs
  on a **discrete argmax over 121 items**, is non-monotone across seeds and sometimes 0 at the
  smallest size → **reported, not asserted**. Also: `--fast` was cutting gradient STEPS, making a
  true assertion false for unrelated reasons; it now shrinks data and loosens tolerances only.
  Closed the W3 chore (`tilted_policy` log guard, 94/94 under `-W error::RuntimeWarning`).
  Deliverable written with a polished five-part answer to the 78%-RM drill.
  **Interview drills NOT run (out of clock) — carried to D2.**

- **W4 D2 — The PPO Objective: Ratio, Clipping, Epochs.** Built
  `week-04-rlhf-ppo/code/ppo_loss.py` (**112 staged checks, 5.3 s; `--fast` 109 in 1.5 s**):
  `token_logratios`, `is_surrogate`, `clipped_surrogate` (+ `clip_frac`/`oob_frac`/`frac_high`/
  `frac_low`), `surrogate_gradient` (analytic, with the clip GATE), `ppo_update` (epoch ×
  minibatch reuse), `sequence_ratio_stats` (exact by enumeration), `drift_experiment` (4 modes).
  **Design win: `exact_batch`** returns all 85 sequences with `weights = p_θold(y)` instead of
  `1/B`, so an estimator on it is not an estimate but the exact expectation — that is what makes
  the ρ=1 identity an assertion at 1e-11 rather than a claim about standard errors.
  **Headline — clipping bounds the STEP, never the WALK.** 25 iterations, β=1.0: clip-only reached
  **KL(π‖π_ref) = 4.6836 nats at a mean clip fraction of 5.3%** (max 16.5%) while the β-penalised
  run **plateaued at 0.7497** (last 8 iterations span 0.034 nats; clip-only rose another 1.33 over
  the same window). **6.2×.** `both` tightest at 0.7488, `none` loosest at 5.6133. Mean
  per-iteration step KL 0.017 vs a 4.68 walk = **270×**, the arithmetic of the whole result.
  **Second — the β frontier, and `dE[r]/dKL = +β` MEASURED.** β 0.7/1.0/1.5/2.5 → KL
  1.890/0.849/0.342/0.115, J 2.910/2.085/1.475/1.046. All three secant slopes **bracket their β
  pair** (0.793∈[0.7,1.0], 1.203∈[1.0,1.5], 1.892∈[1.5,2.5]) and **fall as KL grows** ⇒ concavity,
  measured. This closes the D1 sign error and the D2 cold-recall shape gap in one experiment.
  **Third — the session plan was WRONG about clipping and the code corrected it.** Clipping makes
  the angle to the true ∇J **worse** (43.8° vs 11.8° at K=64) while cutting KL travelled **29.3×**
  ⇒ **it protects step SIZE, not DIRECTION**, and those 32° ARE the deliberate bias of Part 4,
  measured. The no-clip run also scores higher J on the same batch (2.46 vs 0.58) = Goodhart from
  the inside. Negative control clean: K=1/1 minibatch, clip == no-clip at **0.00e+00** and the
  direction is **exactly 0°** from ∇J.
- **W4 D3 — Token-Level PPO: Value Heads, Sequence Rewards, Per-Token KL.** Built
  `code/token_ppo.py` (**144 staged checks in 7.6 s**; 8 TODOs, 12 stages, fixture env
  vocab=5/t_max=4 so V=5, T=4, S=85, B=341 are four different numbers and value arrays are
  `(B, T+1)` against everything else's `(B, T)`). He implemented all of it bar
  `zero_baseline_probe` and the `instructgpt_loss` plumbing, which were finished for him at
  his request when he ran out of time.
  **First — the reconciliation, measured.** At λ=1 the critic enters ONLY as a subtracted
  `V(s_t)`, so it is `h_t`-measurable, so it is a baseline: **every** critic — zero, pure
  noise σ=6, anti-correlated −V\*, V\*+25 — reproduces the enumerated finite-difference
  gradient to **2.8e-10**. At λ=0.5 the same zero critic is **0.796** off. Same critic, same
  rollouts; only the estimator's USE of it changed. That IS checkpoint question 3.
  **Second — bias tracks the state-VARYING error.** Occupancy-weighted `e = ē + ẽ` along a
  real 4000-step regression: Spearman(bias, ‖ẽ‖) = **+0.995**, Spearman(bias, EV) =
  **−0.995**, Pythagoras asserted. A learned critic's error is almost entirely the EXPENSIVE
  kind (‖ẽ‖ ≈ ‖e‖ at every checkpoint; ē dies in the first 200 steps) — gradient descent
  fixes the free component first, so what remains is the part that bites.
  **Third — the session plan was WRONG about the negative control, and the correction is the
  day's best result.** `session.md` claimed `V+c` gives near-zero bias. Truth: it shifts
  advantages by exactly **`−c·λ^(L−1−t)`** (asserted elementwise to 3.6e-15). Suppress EOS so
  every rollout runs to `t_max` and the remaining horizon is a function of `t` ALONE ⇒
  `h_t`-measurable ⇒ a baseline ⇒ a wrong LEVEL costs **1.5e-10**, the FD floor, *literally
  nothing*, while a wrong RANKING of identical ‖e‖=1.5 costs **0.9724**. Put EOS back and the
  same level error costs **0.4468**. ⇒ **"Wrong level is free" is a claim about the BASELINE
  role; under bootstrapping it additionally requires FIXED length, and an LLM never has that.**
  The pinned terminal row (V=0, unshiftable) is what breaks the uniformity. This is the fourth
  session running where checking the CLAIM before the code was right.
  **Fourth — EV vs MSE, exactly.** `V\*+25`: MSE grows by **exactly c²=625**, EV changes by
  **0.0** (bit-identical, 0.428000392162 both), gradient at λ=1 bit-identical to 1.7e-15.
  EV converges to an **analytic ceiling of 0.4280, not 1** (irreducible within-state return
  variance) — a critic at EV 0.43 in this env is perfect. EV can be negative (−3.65 at
  noise 2.5) = worse than predicting the mean.
  **Fifth — the decisive experiment, wired in and owned** (closes the standing D4 drill gap he
  twice scored ~0.1 on): `zero_baseline_probe` recomputes gradient variance on the SAME
  rollouts — none 4.582, depth-only 4.306 (**−6.0%**), exact V\* 2.632 (**−42.6%**),
  anti-correlated 10.453 (**+128.1%**), all four unbiased at λ=1 so the comparison is pure
  variance. Note this env has NO horizon mismatch (every prefix state has a unique depth), so
  W3 D3's +27% is not reproducible here — the anti-critic replaces it.
  **Sixth — shaping exactness at the EOS boundary.** `E[Σ_t penalty_t]` = **0.405113796560**
  vs `β·exact_sequence_kl` = **0.405113796560** (1e-11), for **both** k1 and k3, now with an
  honest terminal state and variable length. k1/k3 same mean to 1e-12; k1 pays a BONUS on
  **28.2%** of token weight, k3 never; variance ratio **k3/k1 = 0.865**. MC at n=20k: 0.40376.
  **Also measured:** value target = `Â + V_old` equals `G_t` at λ=1 to **8.9e-16** and has
  lower variance at λ<1 (0.9016 vs 1.9649); the value clip PUNISHES a critic that jumps
  straight to the right answer (plain 0 → clipped **16**); centering leaves the exact gradient
  identical (2.8e-17) while per-group std division amplifies a low-variance group **5×**
  (Dr. GRPO, in two lines); c₁ is **NON-monotone** — neutral at 0.1–0.5, keeps only **2.5%**
  of the gain at 5.0, so "we tuned c₁ and saw nothing" is not evidence of safety;
  `reduce="mean"` moves weight on len≤2 completions from **0.3371 → 0.5522**.
  **Fourth — the log-normal law, exactly.** `E_old[ρ]=1` to 1e-15 at every T while p99/median goes
  1.06→2.50 and ESS/n 0.9991→0.8387 over T=1..12; `σ(T)=σ₁√T` to **1.1%**, `ESS/n=exp(−σ²)` to
  **1.6%**, `ESS/n = 1/(1+χ²)` exactly. Extrapolated with σ₁=0.1257 measured while **88.5% of
  tokens sit INSIDE the ε=0.2 band**: at T=500, ESS/n = 3.7e-4 ⇒ **0.38 effective sequences out of
  1024.** Other verified: grad == W3's score-function gradient **elementwise at 0.0** and the
  enumerated FD at 1.9e-11; rows sum to zero 2.1e-16; centring moves the gradient 9.7e-17 (a
  baseline is free, exactly); `L_CLIP ≤ L_IS` pointwise; ε→∞ reproduces `is_surrogate` exactly;
  gradient EXACTLY 0.0 in the two clipped cells; only ONE cell unbounded (−4.0→−12.0, linear in ρ).
  **Cold recall 5.1/7** (Ψ ladder **7/7 recovered** from 5/7 — the new block is working; `dE[r]/dKL`
  sign fixed; **numbers still blank on the RM-error item, 2nd day running**; "two reasons pairs
  batch together" answered with VARIANCE — it is OVERFITTING). **Quiz 2.52/4** (BT derivation and
  σ(−Δ) clean; **Q2 sequence-KL two routes was "not sure" — 0.15**, taught in full since it is D3's
  load-bearing lemma). **Derivation 3.15/4 — his best derivation block yet**, up from 2.35 on D1:
  ρ=1 identity clean, 2×2 table fully correct with values/gradients/intents, k1→ε and k2→ε²/2 both
  right with 0.2/0.02 nats. Slips: called the frozen advantages `π_ref`-computed (they are
  `π_old`'s — the very distinction the day is about); named TWO unbounded cells; pulled `min`
  outside an expectation (`E[min] ≤ min(E,E)`, Jensen); **3 gaps 1 of 3.**
  **Live code review — strong.** Every function correct within 1-2 iterations. His single-`np.add.at`
  whole-row scatter is **cleaner than my reference's two scatters** and sidesteps the tuple-index
  trap entirely. Bugs caught: `array[actions]` indexes axis 0 (4th "name the axis" event);
  `take_along_axis` needs the new axis on the gathered axis (`[..., None]` not `[:, None]`); two
  missing broadcast axes — **both RAISED instead of hiding, purely because V=5, T=3, B=85 are
  distinct** (with D1's `vocab=4, t_max=4` the second would have silently broadcast time against
  vocab); **`frac_high`/`frac_low` SWAPPED, which MY HARNESS DID NOT CATCH** (the only check was
  `frac_high + frac_low == clip_frac`, symmetric in them) ⇒ added semantic checks, and the general
  rule: *a check on an aggregate cannot validate a decomposition*; logratios computed once per
  EPOCH instead of per minibatch (the concept block's "ρ≠1 even within epoch 1" as a live bug);
  `list - list`; `drift_experiment` returning a tuple instead of a dict.
  **MY HARNESS WAS WRONG 4× BEFORE ANY LEARNER CODE RAN (sixth session running), all caught by
  building a private reference first and running it:** (1) asserted the A<0 cell has an unbounded
  **gradient** — false in a tabular softmax, since driving ρ up saturates π and `e_a − p → 0`, so it
  read exactly 0.0; the unboundedness lives in the OBJECTIVE and in ρ, not in this
  parameterisation ⇒ restated as "value is exactly linear in ρ, other three cells flat";
  (2) two exact-equality checks ignoring the 4e-7 of mass that still terminates early with eos at
  −14 rather than −inf ⇒ one-step-bandit fixture instead; (3) the concavity assertion had the
  inequality **backwards** (iterating by increasing β is *decreasing* KL); (4) a `vocab=2` spread
  fixture that was a **silent null experiment** — with one real token its probability is already ~1,
  so perturbing the logit changes nothing and σ read 0.0000 at every T. Also `--fast` truncating
  the T ladder made a true extrapolation assertion false ⇒ restated as an effective sample COUNT.
  **D2's own two interview drills deferred to D3's recall quiz at his request** (cold next-day
  retrieval, which is the right call given his error mode). Polished answers are in the deliverable.

- **W4 D4 — RLHF Failure Modes: Hacking, Length, Entropy Collapse, Drift.** Built
  `code/failure_modes.py` — **96/96 staged checks**, ten stages, all under
  `-W error::RuntimeWarning`, all against enumeration or exact finite-difference ground truth.
  **He implemented nine of the ten TODOs**; `diagnose` was written for him at his request at
  the end. Deliberate design choice worth reusing: **no Monte Carlo anywhere** — the distorted
  sampler's sequence distribution is ENUMERATED, so "the gradient under a broken sampler" is an
  exact expectation and every bias number is an assertion rather than a plot.
  **Five headlines, all measured.** (1) **Off-policy-by-accident, the perennial, CLOSED**:
  on-policy **2.95e-10** vs exact ∇J; temp 0.7 **0.2284**; temp 0.5 **0.4466**; top-p 0.9
  **0.1169**; stacked **0.3127**. IS repairs temperature to **2.953e-10** and on top-p makes it
  **WORSE** (0.1169 → 0.1267) with **7.2%** of π's mass unreachable — reweighting the visible
  part to compensate for the invisible part over-corrects, so *a fix that does not apply is not
  neutral*. (2) **Zero-reward control**: uncentered constant reward gives **9.72e-17**
  on-policy but **7.46e-02** at temp 0.7, and the step moves π TOWARD μ (KL(μ‖π)
  **0.0666 → 0.0587**) — a pure MLE/distillation step, entropy falling with no reward
  information present. The identity `E[∇log π]=0` needs the expectation under π; under μ it is
  false, and that one measure IS the bug. (3) **Goodhart is a VARIANCE result**: seed spread in
  gold amplifies **89× / 259× / 667×** with pressure, and at n=30 the spread (**±1.60**)
  exceeds the entire mean improvement over π_ref (**+0.95**) ⇒ the outcome is seed-determined;
  max |spurious RM weight| **32.07 → 5.82 → 4.29** is the mechanism printed. (4) **The length
  derivation predicts a measurement in BOTH directions**: `β*` is a FIXED POINT, the resistance
  `β·KL̄_t` is **BOUNDED** (max **0.1411**/token), κ = **+0.6276** beats it ⇒ predicted runaway
  at every β ⇒ measured **5.0000 of 5.0**; weak-confound control κ = **0.0504** < **0.2205** ⇒
  stopped at **3.9549**. Debiasing drove the coefficient to **1.71e-16** (algebraic — residuals
  are orthogonal to the regressor), excess length **+0.8656 → +0.1790**, gold
  **5.7706 → 5.9311**, within-length ranking bit-identical. (5) **Deadly triad with
  off-policy-ness as its THIRD leg**: same data, same features, only the visitation changed —
  A = **−3.70** converges (EV +1.000000), A = **+1.07** diverges to **4.366e13** (EV
  **−1.9e27**); MC-target control isolates bootstrapping; the value clip bounds the step
  (**0.010000**) and not the walk (**−6.0** after 600 steps).
  **MY HARNESS WAS WRONG 3× BEFORE ANY LEARNER CODE RAN (seventh session running), all caught
  by building a private reference solution first and running it:** (1) asserted the
  over-optimization turnover moves right with RM data — **false for a TABULAR RM**, which has
  no generalization, so its starved tail never improves; fixture rebuilt as a linear model over
  23 features (3 real + 20 spurious) and the claim restated as the variance result, which is
  stronger; (2) asserted gold falls below π_ref's own gold — false here because this gold also
  pays for length indirectly, so the claim was restated around tokens-per-nat (**+1.42** vs
  **−0.72**); (3) the entropy step size in `zero_reward` was large enough to leave the linear
  regime and moved KL the wrong way. Also fixed a **runner bug inherited from D3**: the
  all-clear was gated on whether a stage THREW, not on the check TALLY, so "All invariants
  passed" could print underneath a failing check — a status line that lies.
  **His D4 numbers are in `day-04-rlhf-failure-modes/deliverable.md`, which is built as tables
  BECAUSE the tables are the flashcards and the prose is not.**

- **W4 D5 — The Toy PPO-RLHF Loop, and Adaptive β as Dual Gradient Ascent (WEEK 4 CAPSTONE).**
  Built `week-04-rlhf-ppo/code/ppo_rlhf.py` — **90/90 checks**, ten stages, under
  `-W error::RuntimeWarning`, full suite **26 s**. Six TODOs; he implemented `sft_reference`,
  `kl_controller`, `rlhf_iteration` and the skeleton of `train_rlhf`; the record/logging block,
  `select_checkpoint` and `ablate` were written for him at 00:00–00:20 at his request.
  **The day's theory:** β was never a hyperparameter — it is the Lagrange multiplier of
  `max E[r] s.t. KL ≤ δ`, `g(β)=max_θ L` is a **max of AFFINE functions of β** hence convex
  *regardless of the primal*, the envelope theorem gives `∂g/∂β = δ − KL` (the same computation
  as W3 D5's `dJ*/dβ = −KL`, verified there to rtol 1e-4), and InstructGPT's controller is
  proportional dual ascent with `η = K_β β/δ`. Multiplicative because it is a step in **log β**
  and because **β<0 is INFEASIBLE**, not merely numerically bad.
  **THE CHECK:** controller fixed point vs `dE[r]/dKL` computed as a **secant slope of two
  enumerated tilted distributions** (no β anywhere in its computation) — δ=0.25 **1.23079 vs
  1.17216 (5.00%)**, δ=0.5 **0.83637 vs 0.82765 (1.05%)**, δ=1.0 **0.56914 vs 0.55652
  (2.27%)**, δ=2.0 **0.37505 vs 0.35735 (4.95%)**; frontier concave at **100%** of secant
  slopes. **THE CAVEAT, MEASURED:** exact primal β **0.6173** vs PPO primal **0.1563** at the
  same KL 0.8646 ⇒ **74.7% gap** — a justification, not a proof, with a number attached.
  **THE CHAINED REGRESSION CLOSES:** REINFORCE → +baseline → +learned critic → GAE(λ) → PPO →
  KL shaping, all from ONE code path by flipping switches, elementwise to **7.749e-15**; the
  EGLP/baseline check is non-vacuous by **1.3e11×**.
  **THE GRID (7 arms × 3 seeds, fixed budget):** π_ref gold **4.4191**; full **5.1501 ±0.109**
  (proxy 0.6153); **no_kl 4.9098 ±0.407 with proxy 0.8728** ⇒ **proxy UP, gold DOWN — the week
  in one assertion**; reinforce 4.7112; top-p 0.9 4.8559. Two honest surprises: **no_clip ≈
  full** (the PENALTY is binding under a tight budget, not the clip) and **no_critic cost NOTHING
  MEASURABLE** (5.1786 vs 5.1501 — effect **0.0285** vs full's own spread **±0.109**, and
  no_critic's interval is *contained* in full's ⇒ **a NULL, not a win**; corrected W6 D1)
  ⇒ **the critic's value is a function of HORIZON**.
  **CHECKPOINTING:** gold peaks **5.1894 at iter 8 (KL 0.503)** and ends **3.9372 at KL 3.605**
  — **below π_ref** — so shipping the final policy costs **1.2522**. Required starving the RM to
  **30 comparisons at 15% noise**; D4's 400-pair RM was too accurate to show it.
  **`diagnose` SURVIVED CONTACT WITH REAL RUNS** (silent on healthy) and fired
  **`value_divergence`, EV → −5.19e9, on the top-p arm UNPROMPTED** — the deadly triad with the
  sampler as its off-policy leg. Also measured the **top-p ratchet as a crash**: reachable
  sequences **40 → 39 → 18 → 17 → 12 → 9**.
  **THE SILENT-BUG PROBE:** `std(log ρ)` = **0.098999** on a real 4-epoch update, exactly 0 if
  `logp_old` is recomputed in the loop; `clip_frac` reads a healthy **0.000** under BOTH
  hypotheses and is therefore useless. The loss cannot help either — `L_IS(θ_old) = E[A] ≈ 0`
  by construction.
  **His D5 numbers are in `day-05-toy-ppo-rlhf-loop/deliverable.md`, which also contains the
  three polished interview answers verbatim** (design an RLHF system / why PPO looks the way it
  does / when NOT to use PPO). Those three were **not attempted cold** — re-ask them.

## Learner profile (quick facts)

- Strong: PhD-level math/optimization, senior ML eng, real post-training experience
  (SFT/DPO/GRPO, retrieval, personalization). RL intuition is genuinely good.
- **Top gaps:** (1) ML systems / memory-cost reasoning; (2) making RL depth
  interview-crisp and derivable on demand.
- Prior work to reference in examples: Amazon personalized prompt-augmentation (RAG:
  "pet food" → "pet food for cats"), DPO to beat a Sonnet teacher via HF.
- Teaching style that works: derive from first principles, quiz from prior day, give code
  stubs + self-check harness (don't pre-solve), connect to his projects and later weeks.
- **What he reliably does and does not do** (stable across six weeks, re-confirmed W6 D2):
  **the mathematics is right on first submission almost every time** — every coding loss on
  W6 D1 and D2 was numerics, calibration, or contract-reading, never the derivation. What he
  does *not* do without a typed slot is produce the **second clause**: the number, the scope
  condition, the consequence. W6 D2's signature is exact — **5/5 definitional slots filled,
  5/5 number-and-consequence slots blank** (9th sighting). **Put the number slot FIRST in the
  skeleton**, and type it (`[a number]`, `[the SCOPE condition]`, `[which containment fails]`).
- **He will tell you when a question is defective, and he is usually right.** On W6 D2 he
  wrote *"a lot of these questions are vague"* — the slots were untyped, which is a real
  format defect, and fixing it moved a re-asked question from 1/2 to 3/4. Treat that feedback
  as a bug report, not as resistance.

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
- **TD error / GAE:** `δ_t = r_t + γV_{t+1}(s_{t+1}) − V_t(s_t)`; `Â^(k)_t = Σ_{l<k} γ^l δ_{t+l}
  = (Σ_{l<k} γ^l r_{t+l} + γ^k V(s_{t+k})) − V(s_t)` ("k real rewards, then bootstrap, minus the
  baseline") — **assumes NOTHING about V, it is algebra on one sampled trajectory**. At `k=T−t`
  with `V_T=0` it telescopes to `G_t − V_t(s_t)`. `Â^GAE_t = (1−λ)Σ_{k≥1}λ^{k−1}Â^(k)_t =
  Σ_{l≥0}(γλ)^l δ_{t+l}` (total weight on `δ_{t+l}` is `(1−λ)Σ_{k≥l+1}λ^{k−1} = λ^l`).
  Reverse scan `A_t = δ_t + γλ(1−done_t)A_{t+1}`, `A_T=0` — O(T) **and** numerically stable; the
  cumsum form needs `γ^{−t}` (`0.9^{−999}≈1e45`) which is why every library scans backwards.
- **The bias theorem:** with an EXACT critic, `Ψ_t = γ^t Â^GAE_t` is unbiased for **every** λ.
  `l=0` terms give `∇J_T` (Bellman peel ⇒ `E[δ_t|s_t,a_t] = A_t`); `l≥1` terms are **exactly
  zero** (condition on `h_{t+l}` — up to and including `s_{t+l}` but NOT `a_{t+l}`; `∇log π_t` is
  `h_{t+l}`-measurable since `t<t+l`, and `E_{a∼π}[A_{t+l}|s_{t+l}]=0`). Same
  tower-rule-plus-EGLP as D3, conditioned later. ⇒ the `l≥1` terms are pure zero-mean noise.
- **Where it breaks:** `E[δ_t|s_t,a_t] = A_t + γE_{s'|s_t,a_t}[e_{t+1}(s')] − e_t(s_t)` with
  `e_t(s)=V(s)−V_t(s)`. `−e_t(s_t)` is state-only = a **baseline** = free. `γE_{s'}[e_{t+1}]` is a
  deterministic function of `(s_t,a_t)` ⇒ an **action-dependent baseline** = D3's catastrophic
  case in mild dose. Bias vanishes iff `E_{s'|s,a}[e_{t+1}(s')]` doesn't depend on `a`; the clean
  sufficient case is `e` constant across states.
- **γ vs λ: γ is part of the PROBLEM** (changes `J`, `V`, the optimal policy, the horizon
  `1/(1−γ)`); **λ is part of the ESTIMATOR** (identical target, only bias/variance move; there is
  no λ-optimal policy). Schulman: *γ introduces bias regardless of the value function's accuracy;
  λ introduces bias only when the value function is inaccurate.*
- **Two different powers of γ in `weights_gae`:** `γ^t` out front (the discounted-objective
  factor, = the leftover from re-basing `Σ_{t'≥t}γ^{t'}r_{t'}` into `γ^t G_t`; the textbook
  `d^π_γ` form hides it in the state measure; PPO/GRPO drop it because it crushes late-token
  signal) vs `γ^l` inside `Â^GAE` (offset from `t`). Conflating them is the likeliest silent bug.
- **`Ψ_t` must NOT be `h_t`-measurable** — that requirement is on the subtracted baseline alone.
  A state-only `Ψ` is identically-zero-gradient by EGLP. `Ψ` = signal (must see `a_t`);
  `b` = control variate (must not).
- **Value tables are `(T+1,S)` with `V[T,:]=0`.** Materializing the boundary as DATA (same as
  `A = np.zeros(T+1)` in the scan) makes the classic GAE horizon off-by-one structurally
  impossible, and means `s_T` never has to be recorded.
- **Keep the slow reference structurally independent of the fast one.** `gae_naive` /
  `k_step_advantage` must not call `gae_reverse`, or they can only ever agree with it — this
  caught the unassigned `A[T-1]`, which was invisible to any scan-vs-scan comparison.
- **Silent positional-arg bug, 3rd occurrence** (`Batch(..., n_iters, ...)` for `batch_size`;
  earlier `TransformerBlock(d, n_heads, n_heads)` and `T==V`). **Use keywords when two adjacent
  parameters share a type.** And note the new twist: this one **passed** because the error
  improved the metric. A bug that helps is the hardest to find; the tell was the wall clock.
- **Untested branches are broken until proven otherwise** (`critic="exact"` was a `TypeError`
  under a 64/64 green run). Coverage, not pass count.
- **KL-CONSTRAINED RL (D5, the week's capstone).** `J(π) = E_{a∼π}[r] − β KL(π‖π_ref)`; the
  optimum is an EXPONENTIAL TILT of the reference, `π*(a|s) = π_ref(a|s) e^{r(s,a)/β}/Z(s)`,
  `Z(s) = Σ_a π_ref e^{r/β}`, with optimal value **`J* = β log Z(s)`**. Two failure modes force
  the term, and they are DIFFERENT: reward hacking (the RM is a proxy, valid only near the SFT
  distribution — the KL term encodes where you trust it) and catastrophic forgetting (`π_ref` is
  the only thing in the loop that can WRITE; the RM only scores).
- **`β log Z` is a LOG-SUM-EXP = a soft max of `r` at temperature β with `π_ref` as base
  measure** — Week 1's operator, and the "soft value function" of MaxEnt RL / soft Q-learning
  (which is why that literature and RLHF are the same math). It interpolates `max_a r` (β→0) and
  `E_ref[r]` (β→∞). Implementation consequence: compute `Z` in LOG SPACE — `π_ref exp(r/β)`
  overflows to `inf` at β=1e-3 and normalizes to `nan`.
- **β LIMITS (D5 gap, cold-drill these):** `β→∞ ⇒ π* → π_ref` — **NOT uniform**; uniform is the
  entropy-bonus special case (`π_ref` uniform). `β→0 ⇒ π* →` uniform over
  `argmax{ r(s,a) : a ∈ supp(π_ref) }` — **over the SUPPORT**, with `1/K` on K-way ties.
- **REVERSE KL CONFINES SUPPORT: `supp(π_θ) ⊆ supp(π_ref)` at every β > 0.** Two proofs: the tilt
  MULTIPLIES `π_ref` so a zero survives any finite exponential; and `KL(π‖π_ref) = ∞` off the
  support, so those policies are infinitely penalized regardless of reward. Asserted EXACTLY
  (`== 0.0`) at β=2/0.5/0.01. Three consequences: (1) **RLHF ELICITS, it cannot teach** — it
  reweights what the base model can already produce, so a weak base can't be fixed with more RL;
  (2) **SFT must precede RL** — SFT MOVES the support, RL reweights within it; (3) **top-p
  rollouts are unrepairable** — nucleus truncates SUPPORT (temperature is only a reweighting), so
  no IS ratio and no clipping recovers mass you never sampled.
- **Forward vs reverse KL.** Reverse `KL(π_θ‖π_ref)` = **mode-seeking** (dropping a mode costs
  only the mass you moved) and support-CONFINING; forward `KL(π_ref‖π_θ)` = **mass-covering**
  (diverges as `π→0` anywhere `π_ref` has mass) and support-EXPANDING, with no closed form.
  Measured: low-reward-mode mass ref 0.500 → forward 0.095 → **reverse 0.011** (8.6×); forward
  puts **0.775** on an action the reference never emits. Alignment WANTS mode-seeking: commit to
  preferred behaviour, don't average over everything the base model does.
- **β IS AN EXCHANGE RATE, not a regularization strength.** Two exact identities (envelope
  theorem / Danskin): **`dJ*/dβ = −KL(π*‖π_ref)`** and **`dE[r]/dKL = β`** (verified rtol 1e-4).
  The second makes the reward-KL frontier CONCAVE in KL — the first nats of drift buy far more
  reward than the last — and makes β the price in reward-per-nat.
- **The tilt is invariant to `r → r + c`** (absorbed by `Z`) ⇒ a reward model is identified only
  up to a per-prompt constant ⇒ **that is exactly why GRPO's per-prompt group mean is free.**
- **KL ESTIMATORS (you cannot sum over sequences).** With `lr = log π(a) − log π_ref(a)`, `a∼π`,
  and `ρ = π_ref/π = e^{−lr}` (so `E_π[ρ] = 1` exactly): `k1 = lr` (unbiased, high variance,
  **NEGATIVE on 20-44% of samples** — a logged KL below zero is k1 working, the #1 RLHF-dashboard
  false alarm); `k2 = ½lr²` (≥0, BIASED); `k3 = ρ − 1 − log ρ = expm1(−lr) + lr` (unbiased via
  the `E[ρ]=1` control variate AND ≥0 pointwise; what TRL/GRPO use). **Use `expm1`** — at
  `|lr|~1e-12`, `exp(−lr) − 1` cancels to garbage and can go negative, killing the one property
  k3 exists for.
- **k3's variance win is a SMALL-KL phenomenon and it REVERSES** (measured, 200k samples):
  var(k1)/var(k3) = **423× at KL=0.004, 19.9× at 0.065, 3.0× at 0.23, 0.23× (i.e. 4.4× WORSE) at
  KL=0.65** — `ρ` is a likelihood ratio and ratios blow up once the distributions separate. k3 is
  the right DEFAULT because production runs at small per-token KL, not because it dominates.
  Correspondingly **k2's bias is INVISIBLE where you run** (0.1 se at KL=0.004) and obvious only
  once the KL is real (6.4 se at KL=0.23) — that is how a biased estimator survives in the wild.
- **SEQUENCE KL = TRAJECTORY-DISTRIBUTION KL, exactly.**
  `KL(p_θ(τ)‖p_ref(τ)) = E_τ[Σ_t log(π_θ/π_ref)] = Σ_t E_{d_t}[KL(π_θ(·|s)‖π_ref(·|s))]`
  because both policies share `ρ₀` and `P`, which cancel term by term — **the same cancellation
  that makes the policy gradient model-free.** Needs the STATE-OCCUPANCY weighting `d_t` (forward
  recursion `d_{t+1} = d_t P_π`): local disagreement at a state you never visit costs no KL
  budget. Units: nats per COMPLETION, growing ~linearly in T — when a dashboard says "KL = 12",
  always ask per-token or per-sequence (a factor of T hides there, and it is the commonest way
  people misread each other's β).
- **KL-AS-REWARD-SHAPING IS THE SHIPPED MECHANISM (InstructGPT):**
  `r_t ← r_t − β(log π_θ(a_t|s_t) − log π_ref(a_t|s_t))`. Its expected sum is EXACTLY β × the
  sequence KL (measured 0.77787 vs 0.78974 at β=0.3) ⇒ reward shaping and a sequence-level KL
  constraint are the SAME OBJECT, one sampled and one exact. Why shape rather than add a loss
  term: DENSITY — the RM reward is one scalar at EOS but the KL term exists at every token, so it
  densifies the critic's regression target and hence every `δ_t` (D4's credit-assignment problem
  attacked from the reward side). Use k3: same mean, **1.47× lower variance**, and ≥0 everywhere,
  whereas **k1 pays a BONUS on 24.5% of tokens** (any token made less likely than the reference).
- **OVER-OPTIMIZATION IS GOODHART, NOT AN RL ARTEFACT.** Optimizing a mis-specified proxy: the
  proxy rises MONOTONICALLY in 1/β (0.027 → 1.767) while the true reward peaks at an INTERIOR β
  (**1.1286 at β=0.16**) and then collapses to **−2.3250, BELOW the reference's own 0.0270** —
  over-optimizing is worse than not doing RL at all, and the proxy reports success all the way
  down. ⇒ **"our reward went up" is evidence of nothing.** **Negative control that makes it
  attributable: tilting by the TRUE reward is monotone with no hump** ⇒ the hump is proxy
  mis-specification, not RL and not the KL penalty. General form = Gao et al.: true reward
  concave in `√KL`, eventually bending down. Fix: pick β by taking checkpoints along a KL ladder,
  scoring each on a held-out GOLD/human eval, and landing at the turnover — plus never gating on
  RM score.
- **A leash is not a muzzle.** `train_kl` β sweep 0/0.2/1/5 → return 7.928/7.875/5.897/2.966 with
  sequence KL 5.056/4.834/1.758/0.072 against a reference whose own return is 2.109: every β
  improves on the reference, and β=0 reproduces D3/D4's optimum exactly.
- **DPO IS PLANTED HERE.** `r(s,a) = β log(π*(a|s)/π_ref(a|s)) + β log Z(s)`. `β log Z` is
  intractable **because the action is an entire COMPLETION** — `Z(x) = Σ_y π_ref(y|x)e^{r(x,y)/β}`
  over `|V|^T` sequences — **NOT because logits are unavailable** (for one token you have all
  `|V|`). It is combinatorial, not sparse. But `Z` depends on the **PROMPT ONLY**, so any two
  completions of the same prompt share it and a DIFFERENCE kills it:
  `r(x,y₁) − r(x,y₂) = β log(π*/π_ref)|_{y₁} − β log(π*/π_ref)|_{y₂}`. That converts
  "maximize r subject to a KL constraint" (needs rollouts, an RM, and an intractable partition
  function) into supervised classification on preference pairs. Week 5 adds Bradley-Terry on top.
- **`np.where` IS NOT LAZY, and guarding the RESULT is not guarding the ARGUMENT.**
  `np.where(p>0, p*np.log(p/q), 0.0)` still computes `p/q` everywhere, warns, and leaves
  correctness resting on the bad slot and the masked slot coinciding. Guard INSIDE the log:
  `np.log(np.where(p>0, p, 1.0))`. **But guarding then BREAKS the `+inf` convention** (a support
  violation returns a finite wrong number), so the convention must be ASSERTED:
  `np.any((p>0)&(q==0), axis=-1) → np.inf`. Leaving `log_q` unguarded instead gives `0*inf = nan`
  whenever p AND q both vanish. Rule: **never let the answer depend on infinity arithmetic
  working out.** Also `np.where(cond, a, b)` returns `a` where cond is TRUE — writing the
  VIOLATION as the condition and the good value as the true-branch inverts the whole function.
- **A METRIC CAN BE WRONG IN A DIRECTION NO TEST CAN SEE** (D5, and D4's positional-arg bug):
  `kl_history` logged the unweighted per-state mean (1.053) instead of the occupancy-weighted
  sequence KL (5.056) and 94/94 still passed, because every assertion tested MONOTONICITY IN β,
  which a wrong-by-a-factor quantity preserves. Two tells: it appended an ARRAY not a float, and
  **a provided helper the code never called — an unused fixture in a file that told you to use it
  is a bug report.**
- **Harness convention (repo-wide as of D3):** staged self-checks, one stage per TODO,
  `NotImplementedError` → `[TODO]`+skip, `--fast`/`--list`/named-subset CLI, lazy `Ctx` fixtures
  with their own seeded RNGs, `_fail_fast` probes before expensive fixtures. Skeleton:
  `shared/templates/exercise-harness-template.py`. Two design rules earned here: **assert that the
  BUG is detectably a bug** (relative threshold, e.g. `>5×` the unbiased error, so it survives batch
  -size changes), and **when a well-motivated check fails, check the CLAIM before the code** — the
  scaffold's assertion has now been the wrong one three times, each time producing the day's best
  lesson.

### Running facts — Week 4 (RLHF & PPO)

- **The three stages as three optimization problems.** SFT = MLE on demonstrations (moves the
  SUPPORT and the format); RM = MLE on comparisons (turns a comparison into a scalar FIELD);
  PPO = `max_theta E[r_psi - beta log(pi_theta/pi_ref)]` on **prompts only, no labels**. Stage 3
  is where the estimator class changes from pathwise to SCORE-FUNCTION — which is the whole
  reason Week 3 existed. The economic argument for RLHF: stages 1-2 buy a FIXED human budget,
  stage 3 converts it into unbounded gradient steps. Every downstream failure is a consequence.
- **THE LOAD-BEARING COINCIDENCE: `pi_ref = pi_SFT` = the RM's training distribution.** The RM
  was fit on completions sampled from `pi_SFT`, and the KL penalty measures distance from
  `pi_SFT`. These are taught as two facts and are ONE. ⇒ **the KL penalty is a confidence region
  for the reward model**, measured in nats, not a regularizer for its own sake. Consequences:
  (1) beta is a property of the RM's DATA, so a bigger/better-fit RM justifies a SMALLER beta —
  which is testable, and is what Gao et al. measured (the turnover moves out); (2) the fix is
  often to retrain the RM on CURRENT-policy samples, i.e. iterated/online RLHF is *moving the
  confidence region to where the policy now is*; (3) Gao's law is
  `r_gold(d) = d(alpha - beta log d)` with `d = sqrt(KL)` — DISTANCE, not divergence, is the
  natural axis.
- **Four networks, and only `pi_theta` can WRITE.** Policy (trainable, writes), value head
  (trainable, scores, Day 3), RM (frozen, scores), reference (frozen, could write but is only a
  measuring stick). ⇒ **no amount of RM quality creates a behaviour**; the RM can only re-rank
  what the policy already emits. 7B bf16-Adam bill: trainable **16 B/param = 112 GB**, frozen
  **2 B/param = 14 GB** ⇒ naive 112+112+14+14 = **252 GB** before activations and the rollout KV
  cache. Three cuts: share the trunk (−112 GB, but the two losses then compete for one
  representation); **cache reference logprobs at rollout time and drop `pi_ref` entirely** (free
  — the weights are frozen and you know which tokens were sampled; the one people forget);
  LoRA (`pi_ref` = "the same weights with the adapter off"). InstructGPT used a **6B RM for a
  175B policy** and found a 175B RM less stable — ranking is easier than generation.
- **PPO is a training job AND an inference job in one loop**: rollout (memory-bandwidth-bound,
  wants a paged KV cache and continuous batching) → score → optimize (compute-bound, wants
  ZeRO/FSDP). Those want different memory layouts, and the weights move between them every
  iteration. That impedance mismatch, not the math, is why RLHF frameworks are complicated.
- **Bradley-Terry.** `P(y_w > y_l|x) = sigmoid(r_w - r_l)`;
  `L = E[logaddexp(0, -delta)]`; `dL/dr_w = -sigmoid(-delta)`, `dL/dr_l = +sigmoid(-delta)`
  (equal and opposite — the differences-only structure is visible in the gradient itself).
  Gradient magnitude `sigmoid(-delta)` is **monotone decreasing in how right the model already
  is**: →0 confidently correct, exactly **1/2 at a tie**, →1 confidently WRONG. (The max is at
  `delta -> -inf`, NOT at `delta ~ 0`.)
- **⇒ THE FIT IS DOMINATED BY THE LEAST RELIABLE LABELS.** Confidently-correct pairs are switched
  off, so what drives the gradient is near-ties and misrankings — and near-ties are exactly where
  human annotators coin-flip. Hence (a) **RM accuracy has a NOISE CEILING in the 70s**, and a 95%
  RM on real preference data is evidence of a leak, not of quality; (b) Llama-2's **margin loss**
  `-log sigmoid(delta - m(r))` exists to restore signal on confident pairs; (c) filter near-ties
  or spend extra annotators on them.
- **IDENTIFIABILITY: `r_psi(x,.)` and `r_psi(x,.) + c(x)` fit IDENTICALLY.** Hypothesis consumed:
  **both completions in a pair share the same prompt** — pair across prompts and the free
  direction collapses to a single GLOBAL constant, which is why anchor completions and deliberate
  cross-prompt comparisons are a real fix. Verified as an EXPERIMENT (better than a statistic):
  refit from a different init ⇒ centered scores agree to **8.22e-15**, raw differ by **1.7355**,
  per-prompt mean drifts **6.9e-15** from its init. Three consequences: raw scores are **not
  comparable across prompts** (cross-prompt accuracy **0.445, worse than chance**, vs 0.974
  within-prompt — a dashboard averaging RM score over a mixed prompt set is measuring the prompt
  mix); **GRPO's per-prompt group mean is free** because subtracting it moves along exactly the
  unidentifiable direction; per-prompt centering is the only safe normalization.
- **ONE degree of freedom, THREE appearances**: unidentifiable in the RM's fit (today), invisible
  in the KL-regularized optimum (`pi* ∝ pi_ref e^{r/beta}` is invariant to `r -> r+c`, W3 D5),
  and free to subtract in GRPO. Saying all three connects the week.
- **The error asymmetry that makes over-optimization dangerous.** MSE is symmetric; the failure is
  not. An UNDER-estimate is harmless (the policy avoids that completion); an OVER-estimate is
  actively SOUGHT, because to an argmax an over-estimate is indistinguishable from a genuinely
  good completion. So the statistic that matters is the UPPER TAIL of the error over the
  reachable region, not the mean — and it grows with optimization pressure at fixed KL.
  Measured: over-estimation collapses **86×** (3.231 → 0.038) from 300 to 20 000 comparisons.
- **Preference data: `K` completions per prompt → `C(K,2)` pairs, all in the SAME batch.**
  Statistical reason: otherwise each completion is used in `K-1` gradient steps per epoch and the
  RM memorizes it (InstructGPT overfits after one epoch treating pairs as independent). Compute
  reason: `C(K,2)` pairs reference only `K` distinct completions, so one batch = `K` forwards
  instead of up to `K(K-1)` ⇒ **~(K-1)× cheaper, 8× at K=9**. The statistically correct choice is
  also the cheap one. Why rank rather than pick: annotator time is dominated by READING, so
  ranking K costs barely more than comparing 2 and yields `C(K,2)` labels.
- **InstructGPT Eq. (2) in one sentence:** `r_psi` is what you want, the `beta` KL term is where
  you are ALLOWED to look for it, and the `gamma` **ptx** term is what you promise not to break.
  ptx mixes PRETRAINING gradients back in to fight the **alignment tax** (RLHF'd models regressed
  on SQuAD/DROP/translation while winning human preference) — note it is an **MLE** term, i.e.
  the one part of the objective that can move SUPPORT back.
- **Enumeration is Week 4's ground truth.** `TokenEnv(vocab=4, t_max=4)`: states are prefixes with
  a decision still PENDING (lengths 0..t_max-1 ⇒ 1+3+9+27 = 40; the root `()` is a state because
  `theta[0]` holds the first token's logits), and **121 terminating sequences** = 40 eos-terminated
  + 81 truncated. **Sum-to-one over all terminating sequences under an arbitrary theta is the
  invariant that proves you enumerated a PARTITION** — it catches a dropped EOS branch and a
  double-counted prefix, the two likely bugs. There is no separate "stop probability": EOS is an
  ACTION with its own logit and truncation is forced.
- **Sequence KL two ways.** By enumeration, and by occupancy `sum_s d(s) KL(pi(.|s)||pi_ref(.|s))`
  (each state has a unique time index here, so W3 D5's `sum_t E_{d_t}[KL_t]` collapses to this).
  Agreement to 1e-10 across structurally independent routes. **The unweighted per-state mean reads
  0.50891 against the true 0.96994 — D5's actual bug, now wired in as an assertion.**
- **Never exponentiate an unnormalised log-weight.** `exp(log p_ref + r/beta)` still overflows
  (the exponent is ~6400 at beta=1e-3); build `z` and hand it to a softmax that subtracts the max.
  **Doing this makes the beta→0 and beta→inf special cases UNNECESSARY**: softmax gives exactly
  `1/K` on K-way ties, `softmax(log p_ref) = p_ref` exactly, and `log 0 = -inf` propagates to an
  exact `0.0` (= reverse-KL support confinement, for free). **When a formulation needs `if`
  statements at its limits, the formulation is usually wrong, not the limits.**
- **`np.add.at` for any scatter-add with repeated indices.** `grad[p,w] -= g` GATHERS the RHS then
  SCATTERS it back, so duplicates keep only the last write. The index must be a **TUPLE** `(p, w)`
  — a stacked `(n,2)` array indexes axis 0 only, which RAISES on a rectangular array and
  **silently returns `(n,2,n_items)` on a square one** (square shapes hiding an indexing bug:
  4th occurrence). Cheap structural test that needs no ground truth: **a correct BT gradient's
  rows sum to EXACTLY zero**, because the loss is flat along `c(x)`; the `-=` bug leaves 0.2437.
  Same argument as W3's "every policy-gradient row sums to zero" under softmax
  over-parameterization — different free direction, identical logic.
- **A "safe-looking" learning rate can silently fail to converge.** The BT loss is a MEAN over
  pairs, so each item's gradient is `O(1/n_items)` NO MATTER how much data there is; the top
  Hessian eigenvalue here is ~0.02, so the stable range is `lr < ~100` and 0.5 does nothing.
  Diagnose with the loss curve, not with intuition about "large" learning rates.
- **`np.long` is version- AND platform-dependent** (removed in numpy 1.24, reintroduced in 2.0 as
  C `long` = 32-bit on Windows). Write `np.int64`. It is also the spelling of the W3 D2
  gradient-buffer bug. Relatedly `list + ndarray` ADDS but `list + list` CONCATENATES — never let
  `+` straddle the two when one side is caller-supplied.
- **Statistics that depend on an ARGMAX cannot carry an assertion without seed-averaging.**
  Regret (`gold.max() - gold[argmax(psi)]`) over 121 items moves by a whole reward gap when one
  noisy estimate reaches the top: non-monotone across seeds and sometimes 0 at the SMALLEST data
  size. Report it, don't assert it. Same family: a correlation over 6 points is not a test.
- **`--fast` must shrink DATA and loosen TOLERANCES, never gradient STEPS.** Cutting steps made a
  true assertion false for reasons unrelated to the assertion, which is the worst kind of test
  failure — it sends you debugging correct code. (D2 hit the same family again: `--fast` truncating
  a T ladder moved a measured σ₁ and broke a threshold. Fix = state the claim in units that do not
  move with the fixture — an effective sample COUNT, not a fraction.)
- **PPO in one sentence: the answer to "how many gradient steps can I take on data I already
  collected?"** Rollouts cost 10-100× a gradient step (T sequential forwards, bandwidth-bound) so
  vanilla PG uses the expensive thing once. Reuse ⇒ off-policy at step 2 ⇒ ratio ⇒ variance ⇒ clip.
  **`grad L_IS` at θ=θ_old IS the W3 policy gradient; ρ=1 is an EVALUATION POINT, not an
  assumption**, consumed in exactly one place (the ρ_t factor). Everything past the first step is a
  different algorithm. Structural payoff of the IS form: **π_old is θ-free, so θ leaves the MEASURE
  and enters the INTEGRAND** — the surrogate is honestly differentiable where `J(θ)` is not, which
  is what all of Week 3's machinery existed to work around.
- **DELIBERATE vs ACCIDENTAL off-policy are different situations with the same words.** Deliberate
  (reuse): you chose it, you know π_old exactly, its logprobs are cached, you can correct it.
  Accidental (temp/top-p): the sampler silently isn't the policy, q may not be in closed form, and
  for top-p it is **provably uncorrectable**. PPO's ratio fixes the first; it is not a fix for the
  second, and conflating them is the standard confusion.
- **`Var(ρf) ~ 1 + χ²(π_θ‖π_old)`, and `ESS/n = 1/E[ρ²] = 1/(1+χ²)` because `E[ρ]=1` always.** IS is
  unbiased for ANY θ — that is not the problem; it is *correct and worthless* simultaneously once
  the distributions separate. Same shape as W3 D5's k3 crossover (423× better at KL 0.004, 4.4×
  worse at KL 0.65) and for the same reason: everything built from a likelihood ratio degrades
  exponentially in the log-difference.
- **Why PER-TOKEN ratios: `prod_t ρ_t` is LOG-NORMAL.** Mean exactly 1 (likelihood ratio), median
  `e^{−σ²/2}`, so the mean is held up by a vanishing fraction of samples. `σ(T)=σ₁√T` (verified to
  1.1%) ⇒ `ESS/n = exp(−σ₁²T)`, **exponential in sequence length**. Measured: at T=500 with 88.5% of
  tokens inside the ε=0.2 band, a 1024-sequence batch has ESS **0.38**. Cost of going per-token: it
  is no longer an unbiased IS estimator of the sequence-level objective — a correct high-variance
  estimator swapped for a biased usable one, **the identical trade as clipping**. Don't clip the
  product; clip each factor — the same instinct as working in log space.
- **The 2×2 clip table.** `A>0, ρ>1+ε`: value `(1+ε)A`, **grad 0**. `A>0, ρ<1−ε`: `ρA`, grad flows
  (pull back UP). `A<0, ρ<1−ε`: `(1−ε)A`, **grad 0**. `A<0, ρ>1+ε`: `ρA`, grad flows, **UNBOUNDED**
  (value exactly linear in ρ; dual-clip patches it with an outer `max(., cA)`, c≈3). Clipping is
  one-sided PER CELL, and the two cells where the gradient flows are cells where ρ is ALREADY
  outside the band. `L_CLIP ≤ L_IS` pointwise by cases — the `min` makes it a pessimistic bound,
  which is the honest content of "trust region". **In a tabular softmax the θ-gradient stays bounded
  even in the unbounded cell**, because saturating π sends `e_a − p → 0`; the unboundedness is in
  the objective and in ρ.
- **CLIPPING BOUNDS THE STEP, NEVER THE WALK.** Three gaps, one per level of aggregation: (i) per
  TOKEN ⇒ `T ε²/2`, unbounded in T (10 nats at T=500, ε=0.2); (ii) per SAMPLE and only where the
  clip BINDS ⇒ nothing about `E[KL]`; (iii) relative to **π_old, which RESETS every iteration** ⇒ N
  bounded steps in a consistent direction is an unbounded walk. Measured: **4.68 nats at a 5.3% clip
  fraction vs 0.75 for the β run, 6.2×**; mean step KL 0.017 against that 4.68 walk = 270×.
  ⇒ **log the WALK, not just the step**, and this is why InstructGPT ships clip AND penalty.
- **PPO approximates a constraint by DELETING THE GRADIENT, not by enforcing a bound.** Nothing
  stops ρ from being 5; the clip only declines to pay for the trip. "Clipping keeps the policy close
  to the old one" is FALSE as stated and is the discriminating interview follow-up. Contrast
  **TRPO: `KL(π_old‖π_θ) ≤ δ` — FORWARD, mass-covering, second-order (Fisher-vector product + line
  search)** vs **InstructGPT's `KL(π_θ‖π_ref)` — REVERSE, mode-seeking, support-confining.**
- **Clipping protects step SIZE, not DIRECTION** (43.8° vs 11.8° to the true ∇J at K=64, while
  cutting KL travelled 29.3×). Those degrees ARE the bias: the clip gate depends on ρ hence on
  `a_t`, so it is **action-dependent** — a baseline is unbiased only because it is constant in `a`
  and therefore FACTORS OUT of `E_{a∼π}[∇log π]`; an action-dependent multiplier does not factor
  out, so no theorem forces its contribution to vanish. It is also θ-dependent ⇒ `L_CLIP` is not
  differentiable at the boundary and you are doing SUBGRADIENT ascent. **PPO is deliberately biased
  and the bias is the product.** And the unclipped run scores higher J on the same batch = Goodhart
  from the inside.
- **The PPO loss curve is uninformative BY CONSTRUCTION**: `L_IS(θ_old) = E[Â] ≈ 0`. Read clip
  fraction, `KL(π_θ‖π_ref)`, entropy, explained variance (D3) — never the loss. Sharper: with
  `Σ_t`, the value is `Σ_b w_b len(y_b) Â_b`, so the residual is pure **LENGTH/advantage
  covariance** (measured +0.010993; exactly 0.0 in a one-step bandit). RLHF length bias, visible two
  days before D4 names it.
- **`clip_frac` ≠ `oob_frac`** (0.237 vs 0.570 measured). Every out-of-band token is a candidate;
  only those whose sign of A points further out lose the gradient. **`frac_high` rising = trying to
  escape upward; `frac_low` rising = entropy collapse** — opposite responses, so a transposed
  dashboard is worse than none. `ε=0.0` is NOT `ε=None`: at ε=0 the band is `{1}` and the gate still
  closes on two of four cells. **Epochs × MINIBATCHES is how far off-policy you are, not epochs** —
  "we only do 4 epochs" is not a safety argument; the clip fraction is the measurement.
- **The KL penalty is REWARD SHAPING, not a loss term**: `r(y) − β[log p_old(y) − log p_ref(y)]`
  applied before the advantages, so no estimator knows it exists (consistent with "the KL term is
  not a Ψ — it modifies `r`, changing every rung at once"). Uses only rollout-time quantities ⇒
  **π_ref can be dropped after rollout** (14 GB of a frozen 7B never resident). Stale after step 1,
  exactly like the ratio.
- **`E[min(X,Y)] ≤ min(E X, E Y)`, never `=`** (Jensen, `min` concave). Same family as pulling a
  `max` out of an expectation, which shows up constantly in RL appendices.
- **Don't switch an instrument off with the mechanism it measures.** Reporting `oob_frac = 0` when
  clipping is disabled asserts "nothing drifted" — false exactly in the run where you need the
  reading. Compute it at a fixed reference ε regardless. Same failure as logging only `kl_step`.
- **A check on an AGGREGATE cannot validate a DECOMPOSITION.** `frac_high`/`frac_low` swapped passed
  112 checks because the only assertion touching them was their SUM. When the content is in how a
  total splits, assert the split.
- **Freeze by CONSTRUCTION, not by discipline.** Putting `logp_old` and `advantages` in an immutable
  `Batch` makes "θ_old and the advantages are frozen" unforgeable: the reuse loop has access to
  nothing else. In a real stack `logp_old` is the only thing kept from the rollout worker.

- **The failure taxonomy, and the split that goes FIRST:** over-optimization = the OBJECTIVE is
  wrong (hacking, length); optimization pathology = the OPTIMIZER is broken (entropy collapse,
  value divergence). **Identical on a reward curve, OPPOSITE fixes.** And **two of the four do
  not move the KL** — entropy collapse happens INSIDE π_ref's support, length exploitation
  changes one decision (EOS) many times ⇒ **KL is NECESSARY, not SUFFICIENT** (caught at KL
  0.30 and 0.55). Best single line for "reward up, eval down."
- **The KL penalty is a CONFIDENCE REGION for the RM, not a regularizer.** `π_ref = π_SFT` IS
  the RM's own sampling distribution, so KL literally measures distance from the RM's training
  data ⇒ β is a property of the RM's DATA (retune it after improving the RM), or retrain the RM
  on current-policy samples = iterated RLHF.
- **Why pessimism works and averaging does not:** `E[max] > max E`, so averaging SHRINKS the
  error but never changes its SIGN and the optimizer harvests it anyway. **The minimum converts
  DISAGREEMENT into a PENALTY** — uncertainty becomes cost. It is an LCB from offline RL.
- **`β*` for length is a FIXED POINT, not a formula**, and the resistance `β·KL̄_per_token` is
  **BOUNDED** (it vanishes at both ends of the β ladder: large β = a policy that never moved,
  small β = a saturated one). Measured max **0.1411**/token vs κ **+0.6276** ⇒ **no β stops the
  lengthening**. Plus: **in-distribution FILLER has KL_t ≈ 0 ⇒ the penalty gives no resistance
  at any β**, because the penalty is proportional to SURPRISE. That is why RLHF'd models are
  verbose but unsurprising — the house style derived from a two-line inequality.
- **Sum vs mean is a LENGTH PRIOR, not a normalization**: mean lengthens iff `Δ > R̄` (beat the
  running average) vs sum's `Δ > 0`. PPO uses sum-over-tokens, mean-over-sequences ⇒ no brake.
  Same choice that makes GRPO length-biased and that Dr. GRPO changes. Third sighting.
- **Marginal ≠ partial, and it is why the metric is LENGTH-CONTROLLED win rate:** a gold reward
  that penalizes length at −0.4/token still has a marginal slope of **+0.1751**, because longer
  completions contain more content tokens. Length correlates with quality even when explicitly
  penalized.
- **Severity and repairability are ORTHOGONAL.** top-p 0.9 is the GENTLER per-step distortion
  (removes only the last 10% of mass; entropy is mass-weighted) and the UNREPAIRABLE one;
  temperature reshapes the bulk and repairs to 3e-10. Not intrinsic (top-p 0.5 beats temp 0.9);
  what IS intrinsic is top-p's **one-way ratchet** — the excluded tail takes the `−p_k` share of
  every update and never the `+1[k=a]` share.
- **The bug, in one sentence:** sampling from μ and attributing to π is exactly equivalent to
  optimizing `r̃(y) = w(y)·r(y)` with `w = μ/π` — you are paying a bonus for BEING LIKELY.
  Entropy collapse stated as an objective rather than as a vibe.
- **A control that CANNOT fail is not a control.** With advantage centering, a constant reward
  gives `Â ≡ 0` exactly and every arm reports "no bug" — the zero-reward experiment has to be
  run UNCENTERED or it silently returns a false negative. Same family as D3's `nan < 1e-5`
  guard that inverted.
- **`np.where` is NOT short-circuit — SANITIZE INPUTS, NOT OUTPUTS.** Sixth sighting in three
  weeks. Both branches are fully evaluated, so a guard downstream of a dangerous op only
  LAUNDERS the result (and still fires the warning). Outer `where` selects; **inner `where`
  makes the input safe**: `np.log(np.where(p>0, p, 1.0))`.
- **`argsort(axis=k)` → `take_along_axis` → `put_along_axis` are a MATCHED SET.** A 2-D index
  array in plain fancy indexing addresses **axis 0 only**; `axis=` names the one axis whose
  coordinate is REPLACED, all others matched positionally. A sort permutation is a BIJECTION ⇒
  no collisions ⇒ `np.add.at` was never the tool. Collisions → `add.at`; permutation →
  `put_along_axis`.
- **`&` binds TIGHTER than `==` and `>`** — `a==0 & b>0` parses as `a == (0 & b) > 0`, a chained
  comparison on a bitwise-and. Parenthesize every comparison combined with `&`/`|`. (In pure
  Python `and` binds looser and reads fine; NumPy forces `&`, which sits on the other side of
  the precedence table.)
- **Sign errors need an ASYMMETRIC probe.** `H(uniform over V) = log V` catches an entropy sign
  instantly, but **`sigmoid(0) = 0.5` for BOTH signs** and cannot catch a flipped BT likelihood.
  Check the input whose answer you know in closed form — and pick the one where the two
  hypotheses DISAGREE MOST. Same lesson as W2 D5's toothless invariant test.
- **Read the parameter list as a CHECKLIST before writing the body.** Three unused parameters in
  one function still satisfied the fail-fast probe. And note the specific trap: **unweighted =
  uniform = the ON-POLICY case**, i.e. exactly where "forgot the weights" and "did it right"
  agree.
- **Probabilities and logits are the same TYPE** (`ndarray (S, V)`) and Python cannot tell them
  apart. `log_softmax(0)` is FINITE, so feeding a probability table to a logprob function
  silently converts top-p's exact zeros into ordinary small numbers and every support check
  stops firing. When a docstring shouts PROBABILITIES in caps, that is load-bearing.
- **A status line that lies is worse than no status line.** D3's runner gated "All invariants
  passed" on whether a stage THREW rather than on the check TALLY, so it printed underneath a
  failing check. Fixed on D4.

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

- **A VERIFIER IS A SET, NOT A RATE.** `V ⊆ C` = SOUND (no false positives), `C ⊆ V` = COMPLETE
  (no false negatives). The two failures are not two sides of one dial: **for unsoundness you
  ask whether the set is EMPTY** (any non-empty `V \ C` has sup gap **1.000000**, one FP or
  seven — a SUPPORT property), **for incompleteness you ask for the RATE** (it is the price in
  rollouts: **G 22 → 32 = +45.5%**). They move the same number in opposite directions, so **no
  single scalar can express verifier quality** — a 5% FP rate *buys* compute (G 32 → 19).
- **"Our verifier is 97% accurate" is not a well-formed sentence.** Every rate is over
  probability MASS under π, so it is a joint property of the checker AND the policy — and RL
  exists to change the policy (precision **0.6720 → 0.0926** under a tilt toward the FPs). It
  is also a function of DIFFICULTY: at a fixed 1% FP rate, `P(hack | rewarded)` runs
  **1.10% → 68.86%** as the true pass rate goes .500 → .005.
- **A HACKED RUN AND A SOLVED RUN AGREE ON EVERY COLUMN THE REWARD CAN PRODUCE.** A k=1 group
  gives `Â = (G−1)/G` whether the acceptance was a hack or a real success — **bit-identical
  advantage vectors** (0.7694/0.9681 vs 0.8361/0.9803, gold **0.025 vs 0.980**). **Always hold
  out a gold set the reward never sees.** It is not hygiene; it is the only column carrying the
  distinction.
- **Dead groups are a RÉNYI ENTROPY of the reward's OUTPUT levels**, `P(dead) = Σ q_j^G`, so
  **SPLIT THE MODE**: one extra reward level took 66.34% → 0.90% (**73.5×**) where G alone
  needed **92** (**11.5×** the rollout bill). Reward design and group size are the same knob.
  Asymmetric — splitting the mass you *already* separate does nothing.
- **A composite reward SELF-ANNEALS: share ∝ 2(G−1)·c(1−c), exactly zero at c ∈ {0,1} for ANY
  λ.** Influence tracks INCONSISTENCY, not the coefficient; a mastered component has left the
  gradient and **cannot be re-weighted back into relevance**. So "a component's share is
  decaying to zero" is the HEALTH signature, not a fault.
- **An RM's accept set is decided by its INITIALISATION; a verifier's is DATA.** A preference
  dataset is a MATCHING, so the BT gradient on an uncompared response is **exactly 0.0** and a
  **1e6** shift costs zero loss. A verifier's worst case is computable *before* training starts.
- **pass@k IS A CAPABILITY COMPOSED WITH A SELECTOR YOU DO NOT HAVE.** Three selectors, three
  limits on the SAME policy: oracle **→ 1**, filter/BoN **→ the verifier's PRECISION**
  (**0.321655** measured while pass@k → **1.000000**), majority vote **→ 1 iff the modal answer
  is right**. maj@k measured **falling** 0.3563 (k=9) → 0.2764 (k=101) with pass@101 = 1.000000
  — *more compute makes the vote worse, and pass@k cannot show it because it is monotone in k by
  construction.* **Name the selector before you quote a capability.**
- **CHANGING k CHANGES WHICH BENCHMARK YOU ARE RUNNING.** `∂pass@k/∂p = k(1−p)^(k−1)`: at k=1
  every problem weighs exactly the same, at k=256 the two hardest carry **100.0000%** of the
  gradient. pass@1 and pass@256 are two different metrics over two different effective test
  sets, not one metric at two budgets.
- **THE PLUG-IN pass@k IS BIASED LOW, AND IT IS JENSEN.** `1 − (1−c/n)^k` is `f(p̂)` with f
  **concave**, so it under-reports: **0.493747 vs 0.651322 = 24.19%** at n=10,p=.1,k=10. Use
  `1 − Π_{i=0}^{k−1}(n−c−i)/(n−i)` (unbiased to **7.772e-16**, never forms a number > 1, and all
  three edge cases fall out of the product). The proof is **exchangeability** — one probability
  computed two ways — not a binomial expansion. Corollary: **the mean goes OUTSIDE** (0.6250 vs
  0.9960 on identical data).
- **THE SAME VERIFIER IS TWO INSTRUMENTS: FILTER vs RANKER, 10.16× apart.** A filter emits
  *every* accepted sample, so its composition `γ(1−p)/[γ(1−p)+(1−α)p]` = **9.0909% is
  n-INVARIANT**. A ranker emits the argmax, so its composition is **`1 − (1−γ)^n` = pass@n over
  the HACK SET** = **92.3685%** at n=256. Same checker, same policy; only the selection rule
  differs. Scaling n scales *selection pressure on the flaw*.
- **BEST-OF-n IS A KL YOU DID NOT BUDGET:** `log n − (n−1)/n` = **0.1931/1.2044/3.1745/4.5491**
  nats at n=2/8/64/256, **+0.6853 per doubling**. And **ties make BoN LESS selective**, so a
  discrete score is strictly cheaper (**2.0507 < 3.1745** at n=64) — the continuous formula is
  an upper bound. Price inference-time search against the KL the training run was held to.
- **THE COLLAPSE THAT NEVER HAPPENED — AND SAY THE SCOPE IN THE SAME BREATH.** GRPO could not
  reduce pass@k_max in **24 configurations × 8 seeds** (min Δ = **+0.000e+00**): unsampled gold
  ⇒ every rollout scores alike ⇒ **advantage identically zero** ⇒ zero gradient. **D2's dead
  group is a SAFETY property.** Scope: 15-sequence tabular softmax, sound verifier, group
  baseline. The claim is *the collapse needs a mechanism*, NOT *RL never narrows a policy*.
- **An unguarded denominator gives you a NaN, not a crash — and NaN never compares True, so a
  threshold built on one SILENTLY NEVER FIRES.** Six sightings in one session, one of them with
  the stage still green. Guard with a plain `if`; run under `-W error::RuntimeWarning`; treat
  "passes with a warning" as a failure.
- **A tail window reads a LEVEL. A CHANGE needs two points FAR APART.** Measuring a trend inside
  the tail cut a real signal from **+0.2981 to +0.00562** — a 53× SNR loss against a
  zero threshold.
- **GRPO's baseline is a STATISTIC OF THE BATCH, not a function approximator.** Everything
  follows: it shrinks the gradient by exactly `(G−1)/G`, it deletes anything constant within
  the group, and **it vanishes when all G agree** — which is precisely when the policy is
  doing well. `P(dead) = p^G + (1−p)^G`.
- **A group mean is a high-pass filter.** Put a control term inside the advantage and you are
  controlling only its *within-group deviation*; a uniform drift is invisible. **KL goes in the
  loss, outside the advantage, and uses k3** (k1 is negative half the time, which pays the
  policy to leave π_ref).
- **`Â = 0` at G = 1 is the algorithm, not a bug.** Any estimator built on contrasts within a
  sample is identically zero at sample size 1.
- **Before attributing behaviour to a mechanism, check the mechanism is ACTIVE.** At μ=1 the
  PPO clip binds on exactly 0% of tokens.
- **`np.divide(a, b, where=m)` without `out=` reads UNINITIALISED MEMORY.** It usually looks
  like zeros. When a value cannot be tested reliably, assert the *contract* — run under
  `warnings.catch_warnings` and fail on any warning.
- **Read the ± before the mean.** An effect smaller than the baseline's own spread is a null.
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
- **The critic bias rule, in one sentence:** a critic can never bias you as a BASELINE (it is
  `h_t`-measurable by construction) but can bias you through BOOTSTRAPPING (λ<1 makes its error
  a function of `(s_t, a_t)` — an action-dependent baseline). Measured W4 D3: at λ=1 EVERY
  critic hits the FD gradient to 2.8e-10; at λ=0.5 the same zero critic is 0.796 off.
- **Bias tracks ‖ẽ‖ (state-VARYING error), not ‖e‖ and not ē.** Spearman +0.995 vs ‖ẽ‖, −0.995
  vs EV (W4 D3, a real 4000-step regression). A learned critic's error is almost entirely the
  expensive kind — GD fixes the free constant first.
- **"Wrong LEVEL is free" needs a caveat: FIXED length.** A constant critic offset shifts GAE
  advantages by exactly `−c·λ^(L−1−t)`. With L fixed that is a function of t alone ⇒
  h_t-measurable ⇒ a true baseline (measured 1.5e-10 = FD floor). With EOS live, L is an ACTION,
  so the shift is action-dependent and it biases (0.4468 at identical ‖e‖). **An LLM always has
  variable length.** Corrects the naive W3 D4 statement.
- **EV not MSE, and say WHY:** Var is shift-invariant and the shift is exactly the free
  component, so EV's invariance MATCHES the theorem's invariance. `V+25` → MSE +625 exactly,
  EV bit-identical. General habit: choose a diagnostic whose invariances match the invariances
  of the thing you are detecting. EV can be negative = worse than predicting the mean. EV also
  has an analytic ceiling below 1 (irreducible within-state return variance; 0.4280 in W4 D3's
  env) — do not read EV<1 as a broken critic.
- **The decisive critic experiment (say it as ONE sentence):** zero the baseline and recompute
  the gradient VARIANCE on the SAME rollouts. Healthy critic reduces it (−42.6%), mis-specified
  increases it (+128.1%). Decisive because it needs NO ground-truth gradient — which a real
  stack does not have, which is why "compare against true gradients" is unusable.
- Value target is `Â_t + V_old(s_t)`, NOT `G_t`: at λ=1 they are equal by telescoping (8.9e-16,
  algebra); at λ<1 the target is bootstrapped with lower variance (0.9016 vs 1.9649). The critic
  rides the same bias/variance dial as the actor.
- The value clip is a `max` = pessimism applied to a regression, and it **punishes correctness**:
  a critic that jumps straight to the right answer has plain loss 0 and clipped loss 16. No
  theoretical justification; in every implementation. `clip_frac` counts where the max SELECTED
  the clipped branch, not where V left the band (overshooting AWAY from the target leaves the
  plain branch larger) ⇒ `clip_frac ≤ oob_frac`, same as L^CLIP.
- Advantage whitening: CENTERING is a batch-level constant baseline (exact gradient identical,
  2.8e-17). STD DIVISION within one batch is only a per-batch LR rescale; **across groups it
  reweights them** — two groups with a 5× spread, per-group scaling amplifies the low-variance
  one 5×. That is Dr. GRPO's critique of GRPO, same bug different letterhead.
- `c₁` (shared trunk) is **NON-MONOTONE**: neutral at 0.1–0.5, then keeps only 2.5% of the gain
  at 5.0. "We swept c₁ and saw no difference" is NOT evidence the coupling is harmless.
- k1 vs k3 per-token KL: same mean EXACTLY (1e-12), k1 pays a BONUS on 28.2% of token weight,
  k3 is ≥0 pointwise (`e^x ≥ 1+x`), variance ratio k3/k1 = 0.865. `E[e^{lr}] = 1` is what makes
  `e^{lr}−1` a zero-mean control variate. Per-token shaping IS the sequence-KL objective sampled:
  0.405113796560 both ways, at the EOS boundary.
- **GATHER vs SCATTER are adjoints.** `V = φ[ids]` forward (one read per token, `phi[ids]`),
  `dφ[ids] += dV` backward (many tokens collide on one state ⇒ **`np.add.at`**, never `+=`).
  Test: "can two inputs land on the same output slot?" Gather no, scatter yes. (W4 D3: used
  `np.add.at` for a gather — it raised; then the reverse trap is the silent one.)
- `np.zeros_like(index_array)` inherits **int64** — a float critic/gradient stored there
  truncates silently (0.7 → 0). Third sighting after W3 D2's `np.long` buffer. Allocate
  accumulators with an explicit float dtype, never off the shape of an index array.
- `δ_t = r_t + γ(1−done_t)V(s_{t+1}) − V(s_t)`: the baseline subtraction is **UNCONDITIONAL**;
  `(1−done)` attaches ONLY to the bootstrap term. Pulling `V(s_t)` inside the paren zeroes it at
  every terminal token. (W4 D3 — and the correct formula was three lines above in the docstring:
  the standing "state the theorem, then point at where each hypothesis is consumed" gap, in code.)
- **States ≠ sequences.** The reward model scores COMPLETED TEXT, not a state — there is no
  `r(s)` in RLHF. The final state is the prefix WITHOUT the last token, so reconstructing a
  sequence from a state id also drops the EOS that `ends_clean` tests. If you are inverting an
  index to recover something the batch already carries, you reached for the wrong object.
- **`bool("scale") is True`** — a bool parameter fed a mode string fails SILENTLY and both
  branches take the same path. Pass `center_only=(mode == "center")`, and raise on unknown modes.
- **`E[A²] − E[A]²` is numerically unusable** when the variance is ~0: catastrophic cancellation
  gave −1.78e-15 ⇒ `sqrt` ⇒ nan. Use the two-pass `Σw(A−μ)²`, non-negative by construction.
  AND: **`nan < 1e-5` is False** — NaN compares False to everything, so a guard placed AFTER the
  nan is created INVERTS and passes through exactly the case it was written to catch.
- `log.append(phi)` appends a REFERENCE; with in-place `phi -= ...` every snapshot aliases the
  final value. `.copy()`, or rebind with `phi = phi - lr*g`. W3 D3's `history.append(tensor)`.
- A staged harness's `_fail_fast` must touch ONLY the TODOs its own stage needs. Touching all of
  them makes a finished function report "[TODO] not implemented" — a guard that emits a FALSE
  failure is worse than no guard. (Fixed mid-session on W4 D3.)
- **A BOUNDED LOSS IS NOT A LOSS WITH A FINITE MINIMISER.** `inf σ(−u) = 0` is attained only at
  `u = +∞`, so KTO's bounded prospect loss still diverges on separable data — measured at
  `+0.9294` per doubling against DPO's `+0.7207`, i.e. FASTER. cDPO and IPO stop because their
  losses have an **interior** minimum. Check `inf ℓ` *and where it is attained*. (W5 D4 — this
  overturned a claim I had taught earlier the same day.)
- **Every metric has a RANGE; check it before you check anything else.** A length ∈ `[1, t_max]`,
  an entropy ≥ 0, an NLL > 0, a probability ∈ `[0,1]`, a KL ≥ 0. On W5 D4 three hand-rolled
  metrics were all outside their own range (`E[|y|]=−10.93`, `H=−0.107`, `nll_w<0`) — catchable
  without running a single training step. This is the successor probe to the minus-sign fixture,
  because the sign error migrated from the loss to the metrics.
- **Convert at the boundary so the conversion becomes an assertion.** `float(x)` on a
  20-element tensor raises; appending the raw tensor does not. W5 D4's missing `.mean()` would
  have been caught for free.
- **`torch.autograd.grad` returns a TUPLE.** Unpack with `g, = ...`. `float * tuple` throws, but
  `int * tuple` silently *replicates the list* — `lr=2` builds a 2-element tuple. Same family:
  `max(tensor, 0)` returns the Python `int` when the tensor is negative, and *ties return the
  first argument*, so at `π = π_ref` it passes step 0 and dies at step 1. Use `.clamp(min=0.0)`.
- **Never let the optimiser be a free variable in a comparison stage.** Plain GD, one shared lr,
  passed in as a parameter. `margin ~ log t` is a theorem about **gradient descent**; Adam erases
  it while still printing a plausible table. And IPO NaNs at `lr=2.0` where DPO is fine, so a
  per-arm lr would silently confound every arm. (W5 D4.)
- **Run every ablation at more than one hyperparameter before naming a mechanism.** ORPO's odds
  term answers P2 at β=1 (`+0.184`) and is its own opposite at β=0.25 (`−1.899`); its SFT term
  holds at every β. One β would have produced a confident, wrong story. (W5 D4.)
- **Never run a length experiment to convergence** — both arms saturate at `t_max` and the effect
  disappears entirely. Use `stop_margin` and compare at a **matched margin**; that is also the
  only honest way to compare KL across two different objectives. (W5 D4.)
- **A permutation cannot change a mean.** If a shuffle/roll/mismatch step feeds only into a mean,
  it is a **no-op** — verify it does something before believing the mechanism it claims to
  implement. On W5 D4 KTO's `z_ref` roll agreed to `1e-16` on unconditional data; it only becomes
  load-bearing when the reward depends on the PAIRING. (Learner-initiated catch.)
- **A DIAGNOSTIC NEEDS ITS NEGATIVE HALF.** Assert it is **silent on a healthy run**, and that
  each rule does *not* fire on fixtures it does not own. A rule that fires on everything is not
  a rule. W4 D4's −20% entropy threshold sat *inside* the range a healthy run occupies; W5 D5's
  M3 gate was calibrated to have **2.8× air below (healthy 0.089) and 4.4× above (fixture
  1.110)**. **A threshold needs air on BOTH sides, and you only know that by measuring healthy.**
- **Build rules from MECHANISMS, not thresholds.** W5 D5's M1 fires on a *conjunction* — accuracy
  saturated AND margin still rising AND `logp_l` still falling, all over the run's second half —
  not on "the margin is large", because on clean data the cap is `+inf` and a large margin is the
  correct answer to a badly-posed question.
- **Ask whether the signature is a LEVEL or a CHANGE.** W5 D5's M4 looked absent under every
  delta rule because off-data mass *falls* during training (0.4983 → 0.1943) — pushing losers
  down moves mass onto covered winners. Support is a **level**. Getting this wrong makes a real
  failure mode look like it does not exist.
- **A panel needs at least one LEVEL next to every DIFFERENCE.** Log `logp_w`/`logp_l` as plain
  logprobs, never as ratios: as ratios their difference equals `margin/β` to **1.39e-16**, so
  the pair becomes perfectly redundant with the instrument it exists to check. The difference is
  exactly what hides the level. (W5 D5, learner-initiated question.)
- **≥3 seeds is a FLOOR, not a sufficiency condition.** W5 D5 measured gold-spread amplification
  at **6.09× on 3 seeds and 1.04× on 5 — identical runs**, flipping the conclusion completely,
  and it retroactively downgraded a W4 D4 headline. `margin` and `kl` spreads were stable at both
  counts, so the instability is specific to **tail-sensitive expectations over a distribution**.
  Use ≥5 seeds for anything of that shape. Also: **pair the seeds** — vary the data seed and the
  run seed *together*, or you are re-running one fixture (worth **2.07×** tighter spreads).
- **Report the effect against the seed noise, or do not report it.** Same runs, same seeds:
  KL effect **0.4038** vs worst spread **0.3621** ⇒ intervals overlap ⇒ *the honest report is
  "underpowered", not a number*. Margin effect **0.1201** vs spread **0.0282** ⇒ **4.3×**,
  resolved. **The metric decides whether an ablation has power**, not the experiment. (W5 D5.)
- **ANY NUMBER WITH A β IN IT — divide by β and check whether the story flips.** Over the same
  sweep W5 D5 measured margin going **UP** (0.4746 → 0.5947) while `Δ̂ = margin/β` goes **DOWN**
  (1.8984 → 0.2974). This is the **WRONG-LEVEL family, now at SEVEN sightings**, and it is the
  single most reliable failure in this repo.
- **Never ask the CONFIG for something only the DATA knows.** `cfg.n_pairs` is a knob (default 0)
  while `len(edges)` is the realisation; `RunCfg.eps` is cDPO smoothing while `DataCfg.eps` is the
  annotator flip rate that sets the cap. Both W5 D5 bugs have this exact shape. **The config is a
  request; the data are the realisation, and they disagree.**
- **`np.where(cond, A, B)` — ask what TYPE the output must be.** Selecting reward *values* where
  sequence *indices* were required is unrepresentable if you instead flip rows (`long → long`).
  **Prefer the operation that is closed on the type.** (W5 D5.)
- **Write index-pair lines ADJACENTLY** — `r_w, r_l = r[pairs[:,0]], r[pairs[:,1]]`. W5 D5's
  copy-paste that indexed column 0 twice made `margin ≡ 0` and `acc ≡ 0` *forever*: not a crash,
  a panel that reported every broken fixture as healthy. Two lines differing only by an index
  must sit next to each other so the asymmetry is visible.
- **A RECORD IS NOT AN ACCUMULATOR.** `ndarray` has no `.append`, and `np.append` reallocates the
  whole array every call. Accumulate in Python lists inside a `@torch.no_grad()` **closure** over
  the training scope (not an N-argument function — that is a closure written the hard way), then
  `np.asarray` once at the end. Without `no_grad` every logged tensor retains a live autograd
  subgraph. (W5 D5.)
- **`np.nonzero(cond)[0]` for "first index where", never `np.argmax(cond)`** — `argmax` on an
  all-False array returns **0**, silently claiming the *first* element satisfied the condition.
  Test `hit.size` to distinguish never-reached from reached-immediately. (W5 D5.)
- **In torch a TRAILING UNDERSCORE means in-place.** `requires_grad` is an attribute;
  `requires_grad_()` is the method. And `.mean()` on a bool tensor **raises** in torch where numpy
  allows it — use `.double().mean()`. (W5 D5.)
- **A `(K,)` mask against a `(K,2)` tensor needs `unsqueeze(1)`** — a bare `(K,)` broadcasts to
  the **last** axis. At K=2 it *silently succeeds with a wrong answer*. Check broadcasting on a
  shape where the dimensions differ. (W5 D5.)
- **The fix for a zero denominator is the RIGHT denominator, not an epsilon on the wrong one.**
  W4 D4's `rel()` divided by `max(|x[0]|, 1e-9)`, and at θ = θ_ref the implicit reward is
  *exactly* 0.0 — so `margin[0]` is exactly 0.0 in **every** DPO run and it returned `1e9·x[-1]`,
  making every downstream threshold meaningless. Divide by `max_t |x_t|` (bounded in `[−2, 2]`)
  or pass the problem's own scale. **Also guard `scale = inf`**, which returns 0.0 silently.
- **Verify a fixture EXHIBITS the phenomenon before building a rule on it.** W5 D5's private
  reference run caught **two wrong fixtures** — M4's off-data mass *fell* rather than rose, and
  M3's gold *rose* on the original seed. Both would have surfaced as "the learner's code is
  broken". Sixth consecutive day the reference-implementation rule paid.
- **A CONTROL converts an observation into evidence.** "Gold peaks early" is a statement about
  training length until a healthy run peaks at step 1270 and gives back **exactly 0.0000**; then
  it is a statement about the **data**. One extra run. (W5 D5.)
- **An absolute tolerance is only meaningful on a quantity you have checked is O(1).**
  W6 D4 asserted `< 1e-15` on `2(G−1)p(1−p)`, which is **31.5** at G=64, p=0.5 — a demand for
  `1e-16` *relative*, below machine epsilon. The exact enumeration was fine; the harness was
  wrong. Scale the tolerance, or state the magnitude next to it.
- **The closed form is not automatically the reference.** `1 − p^G − (1−p)^G` cancels
  catastrophically at small p — at G=64 it loses ~13 significant figures near p=1e-4, and the
  64-term Binomial *enumeration* was the accurate one. Rewrite with `expm1`/`log1p` and the
  disagreement vanishes. Whichever expression is better conditioned is the reference. (W6 D4.)
- **"The smallest point in my grid" is not "→ 0".** W6 D4 measured a `p → 0` slope at
  `grid[1] = 5e-4` and got 85.34 against a predicted 89.90 — a 5% "failure" that was really
  "5e-4 is not small compared to the `γ/J = 0.0112` scale where the linear regime starts".
  Evaluate a limit as a limit. And **query exact values, not the nearest grid point**: at
  p = 0.001 a 2.5e-4 snap is a 25% error. A grid is a plotting device, not a lookup table.
- **Never set a threshold equal to the quantity it tests.** `|p* − 1/2| > 0.1` against a true
  value of *exactly* 0.1 fails on floating point (`0.09999999999999998 > 0.1` is False). Assert
  the value, not a hand-picked distance from a value. (W6 D4.)
- **VERIFY THE BRACKET, or do not use a root-finder that needs one.** W6 D4's `_tilt_to_target`
  bisected a **non-monotone** map: the scalar logit bonus went on every `(state, action)` pair of
  a correct sequence, and those pairs are *shared* with incorrect sequences through the same
  prefix — so for ~30% of correct-sets a bonus of **−40 drove p to 1.0**. Bisection silently
  returned an endpoint, and the fixture's advertised "exact known difficulty" was wrong on **40%
  of prompts with errors up to 0.97**, while every stage downstream quoted it to four decimals.
  Fix: dense scan + local refinement (no monotonicity assumption), **return the achieved value**
  so the caller must check it, and **reject** fixtures whose target is unattainable. The
  twin-gap control went from `+0.097` to `+0.000000`. Seventh consecutive day the reference run
  paid — and this one would have invalidated the whole day.
- **A change statistic must not overlap the estimator's burn-in.** `p_hat` is an EMA from a
  prior, so a head window at step 0 measures the **sensor** converging, not the system moving:
  prompts whose true p never left 1e-4 read as having moved **0.047–0.276**, and the `cliff`
  rule could not fire on the fixture built for it. Start the head window after burn-in. (W6 D4.)
- **A threshold is calibrated FOR A REGIME.** The same rule's healthy/sick gap was **0.35 wide at
  150 steps and 0.014 wide at 60** — at the fast setting *no* threshold separates. Run the panel
  at its calibrated regime even under `--fast`; you may re-derive thresholds for a faster one,
  but you may not borrow them. (W6 D4.)
- **Identical fire-sets can be a BENCH defect, not a rule defect.** W6 D4's collinearity check
  reported `budget_dead` and `saturated_band` as the same rule. They are not — **three fixtures
  cannot separate four rules**, and `cliff` and `floor_violated` had never fired at all. The
  bench went 3 → 6 fixtures and all four rules separated. Non-collinearity measured on a
  fixture set that does not span the failure modes is measuring the bench.
- **When a fixture lands inside a threshold's uncertainty band, REPORT IT, do not tune it away.**
  Two W6 D4 fixtures gave seed-dependent verdicts (3/5 and 2/5). Moving the threshold until they
  behave is fitting the panel to the bench. A threshold has a band; say so.
- **A detector is an estimator.** The contamination twin-gap test has
  `se = sqrt(p(1−p)/n_contaminated)`; at n = 4 that is **0.23**, and a measured 0.4667 against an
  expected 0.7000 is **0.9 se** — not a disagreement, an unpowered estimate. Size the fixture to
  the claim before reading its output. (W6 D4.)
