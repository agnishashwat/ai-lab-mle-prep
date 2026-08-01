# W4 D4 — RLHF Failure Modes: Hacking, Length, Entropy Collapse, Drift

**Code:** `week-04-rlhf-ppo/code/failure_modes.py` — **96/96 checks**, ten stages, all
against enumeration or exact finite-difference ground truth, all under
`-W error::RuntimeWarning`. Fixture env `vocab=4, t_max=5` (V=4, T=5, S=121, B=364).

**Session score:** Block 0 cold recall 4.45/8, Block 1 quiz 3.55/7, drills 0.52 / 0.44.

**No Monte Carlo anywhere.** The distorted sampler's sequence distribution is
*enumerated*, so "the gradient under a broken sampler" is an exact expectation, not an
estimate. Every bias number below is an assertion against ground truth.

---

## The frame — say this first, always

> **Over-optimization means the OBJECTIVE is wrong. Optimization pathology means the
> OPTIMIZER is broken.** Hacking and length are the first kind; entropy collapse and
> value divergence are the second. They present **identically on a reward curve** and
> their fixes are **opposite** — the first wants a tighter leash, the second wants it
> *loosened* and the sampler fixed.

And the follow-up that catches people:

| failure | shows in KL? | why |
|---|---|---|
| reward hacking | **yes** | a move off-distribution by definition |
| policy drift | **yes** | it *is* the KL |
| **entropy collapse** | **often no** | a *sharpened* policy sits **inside** π_ref's support |
| **length exploitation** | **barely** | one decision (when to emit EOS), made many times — cheap in nats |

⇒ **KL is a NECESSARY instrument, not a sufficient one.** `diagnose` catches both of the
bottom two at KL **0.30** and **0.55**. A dashboard with only KL on it is blind to half
the taxonomy.

---

## Headline 1 — off-policy-by-accident, finally MEASURED

Six asks across three weeks. It is no longer a drill. Bias is the relative L2 error of
the computed gradient against the **exact** ∇J from enumeration.

| sampler | grad bias | angle to ∇J | H(μ) | ΔH |
|---|---|---|---|---|
| **on-policy** (τ=1, no truncation) | **2.95e-10** | 0.00° | 1.1698 | — |
| temp 0.7 | **0.2284** | 11.17° | 1.0237 | −0.1461 |
| temp 0.5 | **0.4466** | 21.01° | 0.8402 | −0.3296 |
| top-p 0.9 | **0.1169** | 6.62° | 1.0866 | −0.0832 |
| top-p 0.9 + temp 0.7 | **0.3127** | 14.16° | 0.9301 | −0.2397 |

2.95e-10 is the finite-difference floor. Every distorted sampler is **bounded away from
zero**, exactly, with no MC error to hide in.

### The repair, and the impossibility

| | raw bias | after IS | verdict |
|---|---|---|---|
| **temperature 0.7** | 0.2284 | **2.953e-10** | **REPAIRED** (7.7e8× reduction) |
| **top-p 0.9** | 0.1169 | **0.1267** | **FLOOR — it got WORSE** |

`E_μ[ρ] = 1.000000000000` on the temperature arm — the same `E[ρ]=1` identity that made
k3 a control variate on D3. When it holds, IS is exact. When the support condition fails
it cannot even be stated: **7.2% of π's mass is unreachable under top-p**.

> **An IS ratio can REWEIGHT mass you observed; it cannot RESURRECT mass you never
> observed.** Support, not reweighting. And note applying IS to top-p made it *worse* —
> reweighting the visible part to compensate for the invisible part **over-corrects**.
> A fix that doesn't apply is not neutral.

### The fix ORDER (the part that was owed)

1. **Sample at T=1.0, untruncated.** The only actual fix.
2. **IS-weight by `π_θ/q`** with `q` the ACTUAL sampler — works for temperature,
   **provably fails for top-p** because IS needs `supp(π_θ) ⊆ supp(q)` and truncation
   violates it in the wrong direction.
3. Mitigation only.

Three different clippings, do not conflate: **gradient-NORM** clipping (optimizer),
**PPO RATIO** clipping (deliberate off-policy from reuse), **top-p TRUNCATION**
(accidental off-policy from sampling).

### Severity ≠ repairability — the prediction I got wrong

Predicted top-p would collapse entropy faster. **Wrong: temperature 0.7 costs 1.8× the
entropy and ~2× the bias of top-p 0.9.** top-p 0.9 removes only the last 10% of mass by
construction, and entropy is mass-weighted; temperature reshapes the whole distribution.

> **Top-p is the GENTLER distortion per step and the UNREPAIRABLE one. Temperature is
> the HARSHER one and repairs to 3e-10.** The two axes are orthogonal — which is why the
> fix order is not a ranking of how bad each looks.

Not intrinsic: top-p 0.5 *would* beat temp 0.9. The honest answer is "measure both, don't
rank the mechanisms." What *is* intrinsic: top-p's excluded tail is a **one-way ratchet**
— it eats the `−p_k` share of every update and never receives the `+1[k=a]` share.

---

## Headline 2 — the bug has a nonzero drift with NO reward at all

The decisive control, and the experiment that makes the mechanism undeniable.

| arm | max\|g\| | verdict |
|---|---|---|
| **centered**, constant reward, every sampler | **0.00e+00** | a control that CANNOT fail |
| uncentered, constant reward, **on-policy** | 9.72e-17 | `E_π[∇log π] = 0`, an identity |
| uncentered, constant reward, **temp 0.7** | **7.46e-02** | NOT zero |
| uncentered, constant reward, **top-p 0.9** | **1.70e-02** | NOT zero |

After one step, with a **constant reward**: entropy 1.1698 → 1.1617, and
**KL(μ‖π) 0.0666 → 0.0587** — the policy moved **toward the sampler**.

Because with `r ≡ c` and no baseline,

\[
\hat g \;=\; c\,\mathbb E_\mu[\nabla\log\pi_\theta] \;=\; c\,\nabla_\theta\,\mathbb E_\mu[\log\pi_\theta]
\]

> **is a maximum-likelihood step fitting π to μ's samples. Pure distillation of the
> sharpened sampler into the policy, with zero reward information present.**

It is not zero even though `E_π[∇log π] = 0` — **that identity needs the expectation
under π, and we took it under μ.** One measure, one line, the whole bug.

**Experimental-design lesson:** with centering, a constant reward gives `Â ≡ 0` exactly
and every arm reports "no bug." **The control must be run uncentered or it silently
reports a false negative.** A control that cannot fail is not a control.

**General form:** the bug is exactly equivalent to optimizing `r̃(y) = w(y)·r(y)` with
`w = μ/π`. You are not optimizing `r`; you are optimizing `r` times *how much the
sampler likes this sequence*. Entropy collapse stated as an objective.

---

## Headline 3 — Goodhart with a LEARNED RM: a VARIANCE result

RM is **linear over 23 features (3 real + 20 spurious)**, not a table — deliberately.
A tabular RM has **no generalization**, so its starved tail never improves with data and
Gao's mechanism is inexpressible. π_ref's own gold = **0.9104**; best possible = 8.0.

### Optimization pressure amplifies estimation error into outcome variance

| n_pairs | spread @ β=24 | spread @ β=0.05 | amplification |
|---|---|---|---|
| 30 | ±0.0179 | ±1.5970 | **89×** |
| 300 | ±0.0097 | ±2.5105 | **259×** |
| 3000 | ±0.0020 | ±1.3205 | **667×** |

> **At n=30 and full pressure the seed spread (±1.60) EXCEEDS the entire mean improvement
> over π_ref (+0.95).** The outcome of RLHF is *seed-determined* — and no reward curve
> shows you that. This is stronger than the hump, because it is what you would actually
> hit in production.

At low pressure more data buys **reproducibility**: ±0.0179 → ±0.0020.

### The mechanism, printable

| n_pairs | max \|spurious weight\| |
|---|---|
| 30 | **32.07** |
| 300 | 5.82 |
| 3000 | **4.29** |

**The policy was climbing those.** The shrinkage *is* Gao's scaling law. And the n=3000
arm has an **interior gold maximum** (peak at β=0.15, 2.2580, falling to 2.2196), with
gold-vs-√KL rising then bending down: first slope **+0.7072**, last **−1.2622**.

**Negative control:** tilting by the **gold** reward is monotone, no hump, no spread —
0.9814 → **8.0000**, the true optimum. Without it the hump is not attributable.

### Two corrections to what the session plan claimed

- **The turnover does NOT move right with data here.** That result is about *function
  approximators*, and requires the RM to generalize into its own thin tail.
- **Gold does not fall below π_ref's own gold in this fixture**, because this gold also
  pays for length indirectly. Whether over-optimization goes *below* baseline depends on
  the geometry of the true reward, not just on the proxy being wrong.

---

## Headline 4 — when does lengthening pay? A derivation that PREDICTS a measurement

\[
\Delta \;=\; \kappa \;+\; \Delta r_{\text{content}} \;-\; \beta\,\mathbb E[\mathrm{KL}_t]
\qquad\Longrightarrow\qquad
\text{lengthen iff } \kappa > \beta\,\overline{\mathrm{KL}}_{\text{per token}} - \Delta r_{\text{content}}
\]

**A competition between two PER-TOKEN quantities.** The confound is **learned, not
injected** — annotators label with utility `gold + κ·L`:

| length coefficient (standardized, per token) | value |
|---|---|
| gold reward (marginal slope) | **+0.1751** |
| clean RM | **−0.0323** |
| **confounded RM** | **+0.6276** |

### β\* is a FIXED POINT, not a formula

The resistance `β·KL̄_t` is *itself* a function of β — larger β means a policy that moved
less, hence less KL. So it **vanishes at both ends** and is **BOUNDED**:

| | κ | max over β of `β·KL̄_t` | predicted | measured |
|---|---|---|---|---|
| **strong confound** | **+0.6276** | **0.1411** (at β=0.158) | runs to the ceiling at EVERY β | **5.0000 / 5.0** ✓ |
| **weak confound** *(control)* | +0.0504 | 0.2205 | does **not** run away | **3.9549 / 5.0** ✓ |

> **Consequence (iii), measured: the length GAIN is linear in L while the KL RESISTANCE
> is BOUNDED. A large enough κ cannot be priced out by ANY β.** Length must be fixed in
> the DATA, not in the penalty.

**Consequence (ii), the best line in the derivation:** in-distribution **filler** has
`KL_t ≈ 0`, so the KL penalty provides **no resistance at all, at any β** — the penalty
is proportional to *surprise*, and filler is by definition unsurprising. **The cheapest
hack is verbose but unsurprising text**, which is exactly what RLHF'd models produce.

### Sum vs mean — the normalization IS a length prior

\[
R_{\text{mean}}(L{+}1) - R_{\text{mean}}(L) \;=\; \frac{1}{L+1}\Big(\Delta - \overline R\Big)
\]

**Sum asks the marginal token to beat zero; mean asks it to beat the running average.**
PPO uses **sum over tokens, mean over sequences** ⇒ no built-in length brake. Same
`Σ_t` vs `(1/T)Σ_t` choice that makes GRPO length-biased (a *long wrong* answer is
punished less per token) and that Dr. GRPO changes. **Third sighting**, after D3's
`reduce="mean"` moving weight on len≤2 from 0.3371 → 0.5522.

### Why KL alone misses it, and the fix

At **matched KL ≈ 1.0**: confounded length **4.7250** vs clean **2.5376**.
**Length bought per nat: +1.4227 vs −0.7228.** It converts KL into *length*, not quality.

**Fix, measured** (`debias_length_rm`): length coefficient **+0.6276 → 1.71e-16** (an
*algebraic identity* — least-squares residuals are orthogonal to the regressor, not an
empirical finding); mean excess length over the ladder **+0.8656 → +0.1790 tokens**; at
β=0.83, **+1.0404 → −0.1994**; gold at the aggressive end **5.7706 → 5.9311**.
**Within-length ranking is bit-identical** — it removes a confound, not information.

### Why the metric is LENGTH-CONTROLLED win rate

Gold's coefficient **in its own formula** is −0.4/token, but its **marginal slope** is
**+0.1751**: longer completions contain more content tokens, so **length correlates with
quality even when it is explicitly penalized.** Raw win rate cannot separate "better
content" from "more content."

---

## Headline 5 — the deadly triad, with off-policy-ness as its THIRD leg

**function approximation + bootstrapping + an off-policy distribution.** Any two are
safe; all three can diverge. Divergence condition, one scalar:

\[
A \;=\; \sum_i \text{visitation}_i\;\phi_i\,\big(\gamma\phi_i' - \phi_i\big)
\qquad A>0 \Rightarrow w \leftarrow (1+\eta A)\,w
\]

| visitation | A | result |
|---|---|---|
| on-policy (1/3, 1/3, 1/3) | **−3.7000** | converges, w = 1.000000, EV **+1.000000** |
| off-policy (0.90, 0.09, 0.01) | **+1.0748** | **diverges to 4.366e+13**, EV **−1.9e+27** |

**Same data, same features, same rewards — only the visitation distribution changed.**
The true value function is representable exactly (`w=1`), so capacity explains nothing.

The terminal transition (`φ' = 0`) is the **only anchor** — the only one whose target
does not depend on `w`. Off-policy weighting starves it 33× and the amplifying terms win.

**Negative control:** off-policy visitation + **Monte-Carlo targets** converges to
w = 1.000000 ⇒ the divergence is **bootstrapping's** fault, not the weighting's or the
feature tying's.

**The value clip bounds the STEP, not the WALK** — largest single step **0.010000**, and
after 600 such steps **w = −6.0000**. D2's lesson, now about the critic.

> **A distorted sampler does not only bias the ACTOR. It is also the leg that lets the
> CRITIC diverge.** One bug, two victims.

---

## The reusable artifact — `diagnose(history)`

Returns `(failure_name, one_line_reason)` pairs. Design constraints that are lessons:

1. **Reports NOTHING on a healthy run.** The false-positive check IS the artifact.
2. **Not gated on KL.** Entropy collapse fires at KL 0.30, length at KL 0.55.
3. **`gold` used in exactly ONE place** — separating the two branches, the one thing
   training-time instruments cannot settle. Everything else runs on what a real stack logs.
4. **Reasons quote NUMBERS.** `"per-token entropy 1.050 -> 0.200 (-81%) at KL 0.30"`,
   not `"entropy collapse detected"`.
5. **Length is tested BEFORE hacking.** Every length-exploiting run is *also* an
   over-optimizing run, so testing hacking first would shadow the more actionable
   verdict. **When two diagnoses nest, the specific one goes first.**
6. All thresholds are **relative change**, so they transfer across reward scales — the
   same reason D3 preferred EV over MSE.

---

## The taxonomy table, with my numbers in it

| Failure | Mechanism | Instrument | My measurement |
|---|---|---|---|
| **Reward hacking** | RM fit on π_ref-distributed data; error grows off-distribution. The policy climbs the **argmax of the error** | gold vs proxy, plotted against **√KL** | spread **89×→667×**; spurious weights **32.07→4.29**; RM error **0.177→4.89 (27.6×)** along a KL ladder |
| **Length exploitation** | Preference data has a length confound ⇒ RM acquires **κ>0** ⇒ EOS suppressed | length; **reward-vs-length slope**; length-controlled win rate | κ **+0.6276** vs bounded resistance **0.1411** ⇒ ceiling **5.0/5.0**; **+1.42 tokens/nat** |
| **Entropy collapse** | Distorted rollouts oversample the mode ⇒ gradient reinforces it ⇒ positive feedback | per-token **entropy**; unique completions; **frac_low** | bias **0.2284** (temp) / **0.1169** (top-p) vs exact ∇J; drift is **nonzero at zero reward** |
| **Policy drift** | Clipping bounds the **step**, not the **walk**; π_old resets each iteration | exact KL **over iterations** | D2: **4.68 nats at 5.3% clip fraction** vs 0.75 with explicit β (**6.2×**) |
| **Value divergence** | Deadly triad — the third leg is **off-policy-ness** | **EV** (not MSE); value loss | A **−3.70 → +1.07** flips it; **4.4e13**, EV **−1.9e27** |

---

## Three cross-cutting points

**(a) "Reward went up" is evidence of nothing.** W3 D5: proxy monotone 0.027 → 1.767
while true reward peaked at **1.1286** (β=0.16) and collapsed to **−2.3250**, *below*
the reference's own **0.0270**. Over-optimizing was worse than not doing RL at all, **and
the proxy reported success the whole way down.**

**(b) The KL penalty is a CONFIDENCE REGION for the RM, not a regularizer.**
`π_ref = π_SFT` **is the RM's own sampling distribution**, so KL literally measures
distance from the RM's training data. ⇒ **β is a property of the RM's DATA** — a better
RM earns a wider trust region, so **retune β after improving the RM**; or retrain the RM
on current-policy samples, which is **iterated RLHF**.

**(c) RM error is not noise — it is ADVERSARIALLY SELECTED.** The policy is an optimizer
pointed at the RM; it finds the **highest-error** region, not an average one. So
held-out accuracy systematically **overstates** RM usefulness under optimization pressure,
and *"our RM has 78% accuracy"* answers nothing. **Why pessimism works and averaging
doesn't:** `E[max] > max E`, so averaging shrinks the error but never changes its **sign**
— the optimizer harvests it anyway. The **minimum converts disagreement into a penalty**,
so uncertainty becomes cost and the incentive to go there disappears. That is an LCB.

---

## Bugs hit, and what each generalizes to

| bug | lesson |
|---|---|
| `np.where(pi>0, pi*log(pi), 0)` | **`np.where` is not short-circuit** — both branches are fully evaluated. **SANITIZE INPUTS, NOT OUTPUTS:** `log(np.where(pi>0, pi, 1.0))`. **Sixth sighting in three weeks** |
| `np.add.at(remove, idx_sorted, ...)` | a 2-D index array indexes **axis 0 only**. `argsort(axis=k)` → `take_along_axis` → `put_along_axis` are a **matched set**. A sort permutation is a **bijection** ⇒ no collisions ⇒ `add.at` was never the tool |
| `cumsum(logits)` vs `cumsum(probs)` | top_p is a threshold on **probability mass**; logits are unbounded and shift-invariant |
| `np.sort` defaults to **ascending** | the nucleus is the **top**; the shift-right pattern assumes descending |
| `a==0 & b>0` | **`&` binds TIGHTER than `==`/`>`** — parenthesize every comparison combined with `&`/`\|` |
| `sequence_logprob(env, mu, ...)` on **probabilities** | `log_softmax(0)` is **finite**, so top-p's zeros vanish silently. "Probabilities" and "logits" are the same type; the docstring is the only contract |
| `np.zeros(n)` → float in an index array | **mirror image** of D3's `zeros_like(index)` → int64. Allocate indices int, accumulators float, **always explicitly** |
| `H = +Σ p log p` | dropped minus. **`sigmoid(0)=0.5` for BOTH signs** — the symmetric point cannot detect a flip. Probe where the hypotheses disagree most |
| `w - lr*(-grad)` | mixed ascent and descent conventions ⇒ the sign of `A` flipped ⇒ on-policy diverged and off-policy converged |
| `visitation`, `clip`, `mc_targets` unused | **read the parameter list as a checklist.** Unweighted = uniform = the on-policy case, which is exactly where "forgot the weights" and "did it right" agree |

---

## Interview answers

### "Reward is up 40%, human eval is down 15%. Debug it."

1. **The split first:** is the objective wrong or is the optimizer broken? They look
   identical on a reward curve and the fixes are opposite.
2. **The instrument:** a gold eval the RM never trained on, plotted against √KL. And
   entropy + length alongside KL, because **two of the four failures don't move the KL**
   — I catch them at KL 0.30 and 0.55.
3. **The mechanism:** the RM is a proxy fit on π_ref-distributed data; KL *is* distance
   from its training set; the policy climbs the argmax of the error.
4. **My numbers:** gold peaks and falls below the reference's own reward while the proxy
   rises monotonically; RM error grows 27.6× along a KL ladder; and the one I'd lead with
   — **seed spread amplifies 89×→667× with pressure, and at 30 comparisons it exceeds
   the entire mean improvement.** A single seed at high pressure tells you nothing.
5. **Picking β:** KL ladder of checkpoints, gold eval at each, land at the turnover.
   Gate on gold, never on the proxy.
6. **Day one I would have logged:** a gold eval; mean length **and the RM's length slope,
   which is measurable before a single PPO step**; KL per iteration; entropy and unique
   completions; EV and clip fraction; ≥3 seeds per config.

### "Answers got 3× longer and win rate went up. Did it get better?"

**You can't tell from that number.** Raw win rate confounds better content with more
content — I measured a gold reward whose formula *penalizes* length at −0.4/token but
whose **marginal** slope is **+0.1751**, because longer completions contain more content.
Report **length-controlled win rate**. The RM's length coefficient came from the data
(**+0.6276** confounded vs **−0.0323** clean). **Fix it in the data, because the penalty
cannot:** the gain is linear in L while the resistance `β·KL̄_t` is **bounded at 0.1411**,
so at κ=0.6276 **no β stops it** — measured, length ran to the ceiling at every β, while
a weak confound (0.0504 < 0.2205) stopped at 3.95. And filler has KL_t ≈ 0, so it evades
the penalty entirely. Debiasing drove the coefficient to 1.7e-16, cut excess length
+0.87 → +0.18 tokens, and recovered gold 5.77 → 5.93 **with within-length ranking
untouched**.

---

## Carried into Day 5

1. **`diagnose` is imported by the capstone loop.** It must keep reporting nothing on a
   healthy run once the runs are real rather than synthetic.
2. **Adaptive β as dual gradient ascent** on the W3 D5 Lagrangian — β is the dual
   variable and always was. Check its fixed point against `dE[r]/dKL = β`.
3. **The bounded-resistance result constrains what adaptive β can achieve**: a controller
   that targets a KL setpoint cannot fix a length confound, because the resistance it
   commands is bounded regardless of the β it chooses.
4. **≥3 seeds is now mandatory, not advisory** — 89× to 667× spread amplification.
