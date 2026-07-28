# Week 04 Day 04 — RLHF Failure Modes: Hacking, Collapse, Length, Drift

## Objective

Turn four war stories into four measured experiments, each with an **instrument that
detects it** and a **negative control that makes it attributable**. The organizing
distinction — and the thing to say first in any interview answer on this topic:

> **Over-optimization means the objective is wrong. Optimization pathology means the
> optimizer is broken.** Reward hacking and length exploitation are the first kind; entropy
> collapse and value-head divergence are the second. They present identically on a reward
> curve and they have *opposite* fixes: the first wants a tighter leash or a better reward,
> the second wants the leash loosened and the sampler fixed.

The day's two headline experiments:

- **Goodhart with a learned reward model.** Week 3 D5 found the hump with a hand-specified
  proxy reward. Today the proxy is the Bradley-Terry model you fit on Day 1 from finitely
  many noisy preferences, so the hump's location becomes a *function of the preference data*
  — and you can show it moves when you add data. That is the version that answers "how do I
  know my \(\beta\) is right for *this* RM."
- **Off-policy-by-accident, finally measured.** This has been your weakest recurring drill
  (0.45 on D5, regressed from 0.70). Today it stops being a drill: you sample rollouts at
  \(T\!=\!0.7\) / top-\(p\!=\!0.9\), compute the resulting gradient, and compare it to the
  *exact* \(\nabla J\) from Day 1's enumeration. The bias becomes a number and the entropy
  collapse becomes a curve.

## Prerequisites

- Days 1-3 of this week (env + RM, clip + drift, value head + full loss).
- W3 D5 (Goodhart hump, support confinement, k1/k2/k3), W3 D3 (entropy bonus decay),
  W1 D5 (ablation discipline: fix the compute budget, ≥3 seeds, one seed can lie).

## Required Reading

- **Gao, Schulman & Hilton, "Scaling Laws for Reward Model Overoptimization" (2022)** —
  §3 and the figures only. The functional form is the thing:
  \(r_{\text{gold}}(d) = d(\alpha-\beta\log d)\) with \(d=\sqrt{\mathrm{KL}}\), i.e. **true
  reward is concave in \(\sqrt{\mathrm{KL}}\) and eventually bends down**, with the turnover
  moving right as the RM gets bigger / better. You measured this curve's shape on W3 D5;
  today you get the \(\sqrt{\mathrm{KL}}\) axis and the data-size dependence.
- Optional: Singhal et al., "A Long Way to Go: Investigating Length Correlations in RLHF" —
  abstract + the one plot showing most of RLHF's win being length.

## 2-Hour Plan

### 0:00-0:06 — Cold recall of CONCLUSIONS (Days 1-3)

The full InstructGPT loss; what the value head regresses onto; EV vs MSE and why; the two
ways a critic's error can and cannot bias you; the shaping-exactness number; the three ways
clipping isn't a trust region; BT identifiability.

### 0:06-0:16 — Recall Quiz (Day 3, graded)

1. Write the full objective from memory, term by term, with each term's job.
2. Value target \(\hat A + V_{\text{old}}\): why not \(G_t\)? Where do they coincide, and is
   that a statistical or an algebraic fact?
3. Your learned critic's error decomposition — what did \(\bar e\) and \(\tilde e\) do over
   training, and which one tracked the measured gradient bias? Quote the numbers.
4. Why does whitening's *centering* keep you unbiased while the *std division* doesn't, and
   what is that same object called in the GRPO literature?
5. **Perennial — off-policy-by-accident.** Last cold ask before you measure it. Give the
   whole causal chain, the fix order, why top-\(p\) is unrepairable, and the instruments.
   Then state your *prediction* for today's measurement: which direction does the bias go,
   and does entropy fall faster with temperature or with top-\(p\)?

### 0:16-0:40 — Concept: a taxonomy with mechanisms and instruments

Build the table live — symptom → mechanism → instrument → fix. Insist on the mechanism
column; naming the failure is not the answer.

| Failure | Mechanism | Instrument | Fix |
|---|---|---|---|
| **Reward hacking / over-optimization** | RM is a proxy fit on \(\pi_{\text{ref}}\)-distributed data; error grows off-distribution (your Day-1 curve). Policy finds the argmax of the *error*, not of the reward. | RM score **vs a gold/held-out eval**, plotted against \(\sqrt{\mathrm{KL}}\) | raise \(\beta\); pick the checkpoint at the turnover; more/better RM data; RM ensembles + pessimism |
| **Length exploitation** | Preference data has a length confound ⇒ the RM has a positive length gradient. If lengthening pays more than the per-token KL costs, EOS gets suppressed. | length histogram + reward-vs-length regression; length-controlled win rate | length-normalize or debias the RM; length penalty; length-balanced preference data |
| **Entropy / KL collapse** | Mode reinforcement: distorted rollouts (temp<1, top-\(p\)) oversample already-likely tokens ⇒ gradient reinforces the mode ⇒ entropy falls ⇒ less exploration ⇒ tighter modes. A positive feedback loop. | per-token entropy, distinct-\(n\)/unique-completion count, ratio distribution, clip fraction | sample at \(T=1.0\) untruncated for RL; entropy bonus (preventative only); lower LR/epochs; KL floor |
| **Policy drift** | Clipping bounds the **step**, not the **walk** (Day 2, measured). Many small in-region steps compose into arbitrary displacement from \(\pi_{\text{ref}}\). | exact \(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\) over iterations, not per-step | explicit KL in the reward; adaptive \(\beta\) (Day 5); KL-triggered early stop |
| **Value-head divergence** | Shared trunk + large \(c_1\), or a bootstrapped target chasing its own prediction. | EV, value-loss curve, target-vs-prediction drift | clip the value loss; lower \(c_1\); separate trunk; warm up the critic |

Then the three cross-cutting points:

- **"Reward went up" is evidence of nothing.** Your D5 numbers: proxy monotone
  0.027→1.767 while true reward peaked at 1.1286 (\(\beta=0.16\)) and collapsed to
  \(-2.3250\), *below* the reference's own 0.0270. Over-optimizing was worse than not doing
  RL at all, and the proxy reported success the whole way down.
- **Two of these failures raise KL and two don't.** Hacking and drift show up in the KL;
  entropy collapse can happen at *low* KL (a sharpened policy inside the reference's
  support), and length exploitation can look like modest KL because it changes one decision
  (when to emit EOS) many times. **So KL is a necessary instrument, not a sufficient one** —
  you need entropy and length alongside it. This is the single best thing to say in the
  "reward up, eval down" interview question.
- **The RM error is not noise, it is adversarially selected.** The policy is an optimizer
  pointed at the RM; it will find the RM's highest error region, not an average one. So RM
  accuracy on an i.i.d. held-out set systematically overstates the RM's usefulness under
  optimization pressure. This is why pessimism (ensemble minimum, uncertainty penalties)
  works and why "our RM has 78% accuracy" answers nothing.

### 0:40-1:05 — Derivation: when does lengthening pay?

Do this one properly; it is the day's cleanest piece of math and it is not in any paper in
this form.

1. **Setup.** Sequence \(y\) of length \(L\). Shaped reward:
   \(R(y) = r_\psi(y) - \beta\sum_{t=1}^{L}\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\text{ref}}(a_t|s_t)\big)\).
   Model the RM's length confound as \(r_\psi(y) = r_{\text{content}}(y) + \kappa L\)
   (\(\kappa>0\) if annotators preferred longer answers, which they did).
2. **The marginal token.** Compare emitting EOS now against emitting one more content token
   then EOS. The expected gain is
   \[
     \Delta = \kappa + \Delta r_{\text{content}} - \beta\,\mathbb{E}\big[\mathrm{KL}_t\big],
   \]
   so **lengthening pays iff \(\kappa > \beta\,\overline{\mathrm{KL}}_{\text{per token}} - \Delta r_{\text{content}}\)**.
   Read off the three consequences: (i) it is a *competition between two per-token
   quantities*, so \(\beta\) directly buys length control; (ii) if the extra token is
   "in-distribution filler," \(\mathrm{KL}_t\approx 0\) and the KL penalty provides **no
   resistance at all** — the cheapest hack is verbose but unsurprising text, which is
   exactly what RLHF'd models do; (iii) a *constant* \(\kappa\) per token means the gain is
   linear in \(L\) while the resistance is only as large as the KL you actually incur.
3. **Sum vs mean.** Redo the comparison with the per-sequence objective normalized by
   \(1/L\). Show that the mean form penalizes length whenever the marginal token is below
   the current average, i.e. **the normalization silently encodes a length prior**. State
   which one PPO implementations use (sum over tokens, mean over sequences) and what that
   implies. Then connect: this is the same \(\sum_t\) vs \(\frac1T\sum_t\) choice that makes
   GRPO length-biased and that Dr. GRPO changes.
4. **The entropy-collapse loop, written as a recursion.** Let \(\pi^{(k)}\) be the policy at
   iteration \(k\) and \(\mu^{(k)}\) the *sampler* (temp \(\tau<1\) or top-\(p\)). Show
   \(\mu^{(k)}\) puts more mass on the mode than \(\pi^{(k)}\); the gradient computed on
   \(\mu\)-samples but attributed to \(\pi\) is
   \(\mathbb{E}_{\mu}[\hat A\nabla\log\pi_\theta] \neq \nabla J\), and the discrepancy is a
   *systematic* reweighting toward high-probability tokens. Therefore \(H(\pi^{(k+1)}) <
   H(\pi^{(k)})\) even at zero reward signal — **the bug has a nonzero drift with no reward
   at all**, which is the sharpest way to state it and the prediction you will test with a
   zero-reward control run. For top-\(p\) add the support argument: mass outside the nucleus
   is never sampled, so no importance weight can recover it — an IS ratio can reweight, it
   cannot resurrect.

### 1:05-1:47 — Coding: `week-04-rlhf-ppo/code/failure_modes.py`

Four planted failures, each: **instrument that fires**, **negative control**, **fix that
provably works**. Staged harness; ground truth from Day 1's enumeration throughout.

1. **`stage: hacking`** — Goodhart with a *learned* RM.
   Fit RMs on \(n_{\text{pairs}} \in \{50, 500, 5000\}\) noisy preference pairs sampled from
   \(\pi_{\text{ref}}\). For each, sweep \(\beta\) down a ladder and record proxy reward,
   **gold** reward, and exact sequence KL. Assert: proxy is **monotone** in \(1/\beta\);
   gold has an **interior maximum**; at small \(\beta\) gold falls **below the reference's
   own gold reward**; and — the new result — the turnover \(\beta^*\) moves toward
   over-optimization-resistance as \(n_{\text{pairs}}\) grows. Plot gold against
   \(\sqrt{\mathrm{KL}}\) and check the Gao concave-then-bending shape.
   **Negative control:** tilt by the *gold* reward instead; assert monotone, no hump. Without
   this the hump is not attributable to the proxy.
2. **`stage: length`** — length exploitation, built not injected.
   Generate preference labels with a deliberate length confound (label the longer completion
   as preferred with probability \(\sigma(\Delta r_{\text{gold}} + \kappa \Delta L)\)). Fit
   the RM; assert it acquires a positive length coefficient. Run PPO; assert mean length
   rises and **gold reward falls** while proxy rises. Instruments: length histogram,
   RM-score-vs-length slope, EOS probability at each position.
   **Then verify the derivation numerically**: predict the \(\beta\) at which lengthening
   stops paying from \(\kappa\) and the measured per-token KL, and assert the empirical
   turnover is within tolerance of the prediction. A derivation that predicts a measurement
   is worth ten that don't.
   **Fix, measured:** length-debias the RM (regress out \(L\)) and assert length stabilizes
   *and* gold recovers.
3. **`stage: entropy_collapse`** — off-policy-by-accident, measured. *(Your standing gap.)*
   Rollout samplers: `on_policy` (\(\tau{=}1\), no truncation), `temp07`, `topp09`,
   `topp09_temp07`. For each, over PPO iterations, log per-token entropy, unique-completion
   count, exact \(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\), and — the decisive one —
   the **angle and relative error between the computed update and the exact \(\nabla J\)**.
   Assertions: on-policy bias ≈ 0 (within MC error); every distorted sampler has bias
   **bounded away from zero** and entropy strictly lower at matched iteration count.
   **Negative control #1 (the good one):** set the reward to a **constant**. On-policy, the
   policy must not move (assert \(\mathrm{KL}\) stays at ~0 — advantages are zero-mean); with
   the distorted samplers, assert entropy **still falls**. That isolates the bug from the
   reward entirely and is the experiment that makes the mechanism undeniable.
   **Negative control #2:** apply the correct IS ratio \(\pi_\theta/\mu\) for the temperature
   case and assert the bias shrinks; then apply it to the top-\(p\) case and assert it
   **does not** — support, not reweighting. This is the assertion that finally pins the part
   of the drill you keep half-answering.
4. **`stage: value_divergence`** — sweep \(c_1\) with a shared trunk; assert a large \(c_1\)
   degrades policy return and that clipping the value loss bounds the target chase. Instrument
   EV and the value-loss curve.
5. **`stage: instruments`** — a single `diagnose(history)` function that takes a run's logs
   and returns which of the four failures fired, with a one-line reason. Test it against all
   four planted runs **and** against a healthy run (must report nothing — false-positive
   check). This is the reusable artifact; Day 5's loop imports it.

**Discipline reminders:** ≥3 seeds and fixed compute per configuration; report mean ± spread;
claim an effect only if it beats seed noise (W1 D5). Never round a measured number in your
favour. Log floats, not tensors. Keyword-arg every call where two adjacent parameters share
a type — the silent positional bug has appeared four times now, and once it *passed* because
the error improved the metric.

### 1:47-1:57 — Interview Drill

1. **The canonical one, third attempt (0.35 on W3 D5, re-asked Day 1).** *"Reward is up 40%,
   human eval is down 15%. Debug it."* Required structure now: (a) the two-hypothesis split
   — objective wrong vs optimizer broken — and the instrument that distinguishes them;
   (b) the mechanism, in terms of where the RM is valid; (c) your numbers (the hump, the
   RM-error-vs-KL curve, the length slope); (d) how you pick the new \(\beta\) — KL ladder,
   gold eval at each checkpoint, land at the turnover; (e) what you would have logged on day
   one to catch it earlier. Five clauses; five parts.
2. **"Your RLHF'd model's answers got 3× longer and win rate went up. Did it get better?"**
   Wants: length-controlled win rate, the confound in the preference data, the \(\kappa\) vs
   \(\beta\overline{\mathrm{KL}}\) inequality with your measured turnover, and the fix at the
   data level rather than the penalty level.

### 1:57-2:00 — Deliverable & Next Session

`deliverable.md`: the failure taxonomy table with *your* measured numbers in it, the
gold-vs-\(\sqrt{\mathrm{KL}}\) plot at three RM data sizes, the length-turnover prediction vs
measurement, and the off-policy bias table. Next: **Day 5 — the toy PPO-RLHF loop**, which
assembles the week and adds adaptive \(\beta\) (which you will derive as dual gradient ascent
on the W3 D5 Lagrangian, i.e. \(\beta\) is the dual variable and always was).

## Checkpoint Questions

1. State the over-optimization vs optimization-pathology split, and give the one instrument
   that distinguishes them. Which of the four failures raise KL and which don't?
2. Derive the condition under which lengthening pays. Why does in-distribution filler evade
   the KL penalty entirely?
3. Why does entropy collapse under top-\(p\) rollouts even with a constant reward? What did
   your zero-reward control show, and why can't an IS ratio fix the top-\(p\) case?
4. Why does held-out RM accuracy overstate RM usefulness under optimization pressure, and
   what does that imply about ensembles?
5. Given a KL ladder of checkpoints and a gold eval, describe exactly how you choose
   \(\beta\) — and what you refuse to gate on.

## Frontier-Lab Interview Angle

Failure-mode debugging is the highest-signal RLHF interview format, because it cannot be
crammed: the interviewer describes a symptom and watches whether you reason from mechanisms
or pattern-match to blog posts. Three things make an answer read as senior. First, leading
with the *split* (is the objective wrong or is the optimizer broken) instead of jumping to a
fix. Second, naming the instrument before the remedy — KL, entropy, length, EV, clip
fraction — and knowing which failures are invisible to each. Third, quoting your own
numbers. After today you have a gold-vs-KL hump from a reward model you fit yourself, a
length turnover you *predicted* before measuring, and an off-policy gradient bias table.
Very few candidates have measured any of that.
