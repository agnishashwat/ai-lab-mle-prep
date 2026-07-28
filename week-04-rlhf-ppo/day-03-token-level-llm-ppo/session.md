# Week 04 Day 03 — Token-Level PPO: Value Heads, Sequence Rewards, Per-Token KL

## Objective

Assemble the full InstructGPT loss and make every term *measured*. The one structural
addition to Week 3 is the **value head trained by regression**: on D4 of Week 3 you
injected critic error by hand along interpretable axes (exact / wrong-level /
wrong-ranking) and discovered that only the state-*varying* part biases the gradient.
Today the critic is *learned*, so its error is earned — and you get to check whether a
real value head's error looks like the wrong-level kind (free) or the wrong-ranking kind
(expensive), and how that changes over training.

By the end you can write, from memory and with every term's job named:
\[
  L(\theta,\phi) = -\,L^{\text{CLIP}}(\theta) \;+\; c_1 L^{\text{VF}}(\phi) \;-\; c_2\,\mathbb{E}[H(\pi_\theta)],
  \qquad
  r_t \leftarrow \underbrace{r_\psi(y)\,\mathbb 1[t=T]}_{\text{sparse, from Day 1's RM}}
       \;-\; \underbrace{\beta\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\text{ref}}(a_t|s_t)\big)}_{\text{dense, every token}}
\]
and say precisely why the KL lives in the *reward* rather than in the loss.

## Prerequisites

- Day 1 (`toy_rlhf.py`: enumeration, `exact_J`, exact sequence KL, the fitted RM),
  Day 2 (`ppo_loss.py`: ratio, clip, epochs).
- W3 D4 (GAE, the bias theorem, wrong-level vs wrong-ranking), W3 D3 (baselines,
  explained variance, entropy), W3 D5 (per-token KL shaping ≡ sequence KL; k1/k2/k3).

## Required Reading

- **InstructGPT §3.5, Equation (2)** again — now read it as three separate design
  decisions and be able to say who each one is protecting against.
- Optional: the "value function loss clipping" and "advantage whitening" paragraphs of
  *The 37 Implementation Details of PPO* (Huang et al.) — two paragraphs, not the article.
  Both are unprincipled and both matter; know which is which.

## 2-Hour Plan

### 0:00-0:06 — Cold recall of CONCLUSIONS (Days 1-2)

One line each: what \(L^{\text{CLIP}}\) guarantees and what it does not; the three ways
clipping isn't a trust region; the gradient identity at \(\rho=1\); BT identifiability;
your measured clip-only drift number; wrong LEVEL vs wrong RANKING.

### 0:06-0:16 — Recall Quiz (Day 2, graded)

1. Reproduce the 2×2 clip table cold, with gradients. Which cell is unbounded?
2. Show \(\nabla L^{\text{IS}}|_{\theta_{\text{old}}}=\nabla J\) and **name the line where
   \(\rho=1\) is consumed**. (Standing gap: you state theorems without pointing at where
   the hypothesis is used — expect this question every day.)
3. Sequence ratio vs per-token ratio: the argument, and your measured percentile spread.
4. **Perennial — off-policy-by-accident**, one more time. Tomorrow you measure it, so
   today's answer should already contain the prediction you will test.
5. \(\hat A\) is action-dependent; \(b\) must be \(h_t\)-measurable. Which side of that
   line does the *clip* fall on, and therefore is PPO biased?

### 0:16-0:40 — Concept: the three additions, and who each protects

- **The value head.** A scalar head on the policy's trunk, \(V_\phi(s_t)\), regressed onto
  returns. Three things to be precise about:
  - **What it regresses onto.** Not the Monte-Carlo return. In practice the target is
    \(\hat A^{\text{GAE}}_t + V_{\phi_{\text{old}}}(s_t)\) — the same telescoping identity
    you proved on W3 D4 (\(\sum_l(\gamma\lambda)^l\delta_{t+l} = G_t - V(s_t)\) at
    \(\lambda=1\)), so at \(\lambda=1\) the target *is* the empirical return and at
    \(\lambda<1\) it is a bootstrapped, lower-variance target. The critic is trained on the
    same bias/variance dial as the actor.
  - **Shared vs separate trunk.** Sharing halves the memory (this is the actual reason) and
    couples the two losses through \(c_1\), which is why \(c_1\) is a real hyperparameter
    and not a formality: a large value loss can dominate the shared trunk's gradient and
    degrade the policy. Separate trunks make \(c_1\) irrelevant and cost a second model.
  - **Initialization.** At step 0, \(V_\phi\) is untrained, so every \(\delta_t\) is noise
    with a *state-varying* error. By your own W3 D4 theorem, that is the **bias-inducing**
    kind. Prediction to test today: early PPO steps are biased, and the bias decays as the
    critic's explained variance rises. This is also the argument for a critic warm-up phase
    and for starting the value head from the RM's weights (which already scores sequences).
- **Sequence reward, token-level everything else.** The RM emits one scalar at EOS. The
  logprobs, the ratios, the values, the KL are all per-token. Everything hard about LLM RL
  is this mismatch: **one number, \(T\) decisions.** Two attacks, and you own both:
  estimator-side (GAE spreads the terminal signal backwards) and reward-side (per-token KL
  densifies the reward, which densifies the critic's regression target, which densifies
  every \(\delta_t\)). Name how they interact — the KL term gives the critic something to
  predict at every \(t\), which raises explained variance, which makes GAE's bootstrap
  trustworthy at smaller \(\lambda\).
- **Why the KL goes in the reward and not the loss.** Three reasons, in order of
  importance: (i) **density**, as above; (ii) it makes the KL flow through the *advantage*,
  so it is automatically credit-assigned and clipped along with everything else; (iii) it
  keeps the objective a single scalar reward, so the RL machinery does not need to know the
  KL exists. And the exactness result you already measured: its expected sum is
  \(\beta\times\) the sequence KL (0.77787 vs 0.78974 at \(\beta=0.3\)), so this is not an
  approximation to the D5 objective — it *is* the D5 objective, sampled.
- **Advantage whitening.** Standardize \(\hat A\) per batch. Centering is a
  batch-level baseline shift (safe, and it is where most of the benefit is); **dividing by
  the std is not** — it is nonlinear in the batch, it does not factor out, and it degenerates
  when the batch is uniform. You already know this object under another name: it is exactly
  the Dr. GRPO critique of GRPO's std division. Same bug, different letterhead.
- **The ptx term.** \(\gamma\,\mathbb{E}_{\text{pretrain}}[\log\pi_\theta]\): a pretraining
  gradient mixed in to fight the alignment tax. Note it is a *third* anchor, distinct from
  the KL (which anchors to the SFT model in distribution) — this one anchors to the
  pretraining data in likelihood. Modern stacks often drop it and rely on KL + data mixing.
- **Sum vs mean over tokens** (seeds Day 4). Per-sequence, do you use \(\sum_t\) or
  \(\frac1T\sum_t\)? These differ by a length-dependent factor, and the choice silently
  encodes a preference over lengths. Derive it tomorrow; today just make sure your code
  makes the choice *explicitly* and records it.

### 0:40-1:05 — Derivation

1. **The value loss and its clipped variant.** \(L^{\text{VF}}=\mathbb{E}[(V_\phi(s_t)-\hat R_t)^2]\)
   with \(\hat R_t=\hat A_t+V_{\phi_{\text{old}}}(s_t)\). Then the clipped version,
   \(\max\big((V_\phi-\hat R)^2, (\mathrm{clip}(V_\phi,V_{\text{old}}\pm\epsilon)-\hat R)^2\big)\),
   and the honest verdict: it is a `max` (pessimism) applied to a regression, has no
   theoretical justification, and is in every implementation. Know it because it will be in
   the code you are asked to read.
2. **Critic error → gradient bias, re-derived in the learned case.** Restate W3 D4's
   result: with \(e_t(s)=V(s)-V_t(s)\), the error enters \(\delta_t\) as
   \(\gamma\,\mathbb{E}_{s'|s,a}[e_{t+1}(s')] - e_t(s_t)\); the second term is state-only,
   hence a baseline, hence free; the first is a function of \((s_t,a_t)\), hence an
   **action-dependent baseline**, hence bias. Now add the LLM specialization: transitions
   are **deterministic** (append), so \(\mathbb{E}_{s'|s,a}[e_{t+1}(s')] = e_{t+1}(s_t\oplus a_t)\)
   exactly — no expectation to hide behind. The bias at token \(t\) is literally *how much
   the critic's error varies across the tokens you might emit next*. That is a sharper
   statement than the general MDP one, and it is worth having ready.
3. **Explained variance as the diagnostic.** \(\mathrm{EV} = 1 - \mathrm{Var}(\hat R - V_\phi)/\mathrm{Var}(\hat R)\).
   Show why EV, not MSE, is the right instrument: it is invariant to the additive constant
   that W3 D4 proved is free, so **EV measures exactly the ranking part that actually
   biases you**. (This closes the D4 drill you scored 0.1 on. The decisive experiment
   remains: zero the baseline and re-measure gradient variance on the *same* rollouts.)
4. **Entropy in an LLM.** \(H\) here is per-token entropy averaged over positions. Recall
   the W3 D3 result: \(\partial H/\partial z_k = -p_k(\log p_k + H)\), the force **decays**
   as \(\pi\) sharpens, so it is a collapse *preventative*, not a cure, and needs
   \(\beta\sim\mathcal O(1)\). Then the LLM-specific caveat: a uniform-entropy target is
   *wrong* for text — you want high entropy where genuinely uncertain, not everywhere —
   which is why \(c_2\) is often 0 in RLHF and the KL term does the work.

### 1:05-1:47 — Coding: `week-04-rlhf-ppo/code/token_ppo.py`

Staged harness; imports Day 1's env/RM and Day 2's clip.

**TODOs, one stage each:**

1. `value_table_init(env)` / `value_predict(phi, states)` — the tabular stand-in for a
   value head, indexed by prefix state. **Boundary as data**: shape \((n_{\text{states}}+1,)\)
   with the terminal row pinned to 0, exactly as `(T+1, S)` with `V[T,:]=0` made the GAE
   off-by-one structurally impossible on W3 D4.
2. `gae_advantages(rewards, values, dones, gamma, lam)` — reuse the W3 D4 reverse scan;
   this stage is a **pure regression test** against `gae.py` and should agree to ~1e-16.
3. `shape_rewards(seq, theta, theta_ref, beta, estimator)` — per-token KL shaping with
   `k1` / `k3` selectable (reuse D5). Terminal RM reward added at EOS only.
4. `value_targets(advantages, values_old)` and `value_loss(phi, targets, clip_eps)` —
   plain and clipped.
5. `fit_value_head(...)` — regression, returning the EV trajectory.
6. `whiten(adv, center_only: bool)` — both variants, so the std division is a switch you
   can ablate rather than an assumption.
7. `instructgpt_loss(...)` — the whole thing: \(L^{\text{CLIP}}\), \(c_1L^{\text{VF}}\),
   \(c_2 H\), with the shaped rewards feeding GAE.

**Checks with teeth:**

- **Exactness of the shaping (D5 regression, one level harder).** Assert
  \(\mathbb{E}[\sum_t \text{KL-penalty}_t] = \beta\cdot\)`exact_sequence_kl` computed by
  Day 1's enumeration — now with an honest terminal state and variable length, so the check
  covers the EOS boundary that the tabular D5 version could not.
- **`beta=0` reproduces the unshaped rewards elementwise**, and `k1` vs `k3` shaping have
  the **same mean** (assert within MC error) but different variance — with the variance
  ratio recorded. Also assert k1 pays a **bonus** on a nonzero fraction of tokens and k3
  never does (\(\ge 0\) pointwise); that asymmetry is the reason for the default.
- **Value-target identity.** At \(\lambda=1\), assert `value_targets` equals the empirical
  return \(G_t\) **exactly** (telescoping is algebra, not statistics — the same
  elementwise-to-1e-16 claim that chained D2→D3→D4 last week). At \(\lambda<1\) assert it
  does not, and that its variance is lower.
- **Clipped value loss \(\ge\) plain value loss** pointwise, with a hand-computed case;
  and it is a `max` not a `min` — assert the *direction*, because getting it backwards
  still trains and just silently disables the clip.
- **The day's headline: does a learned critic's error look wrong-level or wrong-ranking?**
  Fit the value head for \(n\) steps; at checkpoints, decompose the error
  \(e = \bar e + \tilde e\) (mean + state-varying) and measure both. Then measure the
  **exact gradient bias** against `exact_J`'s finite-difference truth at each checkpoint.
  Assert: bias tracks \(\|\tilde e\|\), *not* \(\|e\|\), and not \(\bar e\). This is W3 D4's
  theorem re-derived from a real regression rather than a hand-built critic — and if the
  measurement disagrees with the theorem, **check the claim before the code** (that has now
  been the right call four sessions running).
- **EV vs bias.** Plot gradient bias against explained variance across training; assert
  monotone decreasing. Then the negative control: an intentionally offset critic
  (\(V+c\), large \(c\)) must show **bad MSE, good EV, and near-zero bias** — proving EV is
  the right instrument and MSE is the wrong one.
- **The decisive experiment, wired in** (the standing D4 drill gap): a
  `zero_baseline_probe(rollouts)` stage that recomputes gradient variance on the *same*
  rollouts with the baseline zeroed, and asserts that a healthy critic reduces variance
  while a horizon-mismatched one increases it (reproducing W3 D3's +27%).
- **Whitening ablation.** center-only vs center-and-scale: assert center-only leaves the
  gradient **direction** unchanged up to the batch baseline (unbiased), and construct a
  degenerate batch (all advantages equal) where the std division blows up — assert the
  guard returns a finite value and does not silently emit `nan`. (Empty/degenerate
  denominators giving `0.0` not `nan` is a standing house rule.)
- **Shared-trunk coupling.** Sweep \(c_1 \in \{0, 0.1, 0.5, 5\}\) and record policy return;
  assert a large \(c_1\) measurably degrades it. Then say the interview sentence out loud:
  "the value loss competes for the trunk."

### 1:47-1:57 — Interview Drill

1. **"Write the full RLHF objective and defend every term."** Then the second half, scored
   separately: *why is the KL in the reward rather than as a loss penalty?* Lead with
   density, and give the measured equivalence number.
2. **"Your value head has an \(R^2\) of 0.1 after 500 steps. Is your policy gradient
   biased? What do you do?"** Wants: a critic is a function of \(s_t\) so it is
   \(h_t\)-measurable by construction ⇒ **as a baseline it cannot bias you**; but inside
   GAE with \(\lambda<1\) it bootstraps, and *there* the state-varying part of the error
   does bias you — so the answer depends on \(\lambda\). Then: measure EV not MSE; run the
   zero-the-baseline probe; consider \(\lambda\to1\) as the hedge (your D4 result: with a
   critic you don't trust, \(\lambda>0\) exists precisely as that hedge, \(\lambda^*=0.9\)
   measured); warm up the critic; check the value target isn't horizon-mismatched.

### 1:57-2:00 — Deliverable & Next Session

`deliverable.md`: the full loss with each term's job, the learned-critic error decomposition
plot, the bias-vs-EV curve, and the shaping-exactness number. Next: **Day 4 — failure
modes**, where the RM you fit on Day 1 gets over-optimized on purpose, the off-policy
rollout bug finally gets *measured* against exact ground truth, and length exploitation
shows up because your env has an EOS token.

## Checkpoint Questions

1. Write the InstructGPT objective from memory. For each term, name the failure it prevents
   and the day you measured it.
2. What does the value head regress onto, and why is it \(\hat A + V_{\text{old}}\) rather
   than \(G_t\)? At which \(\lambda\) do these coincide, and why exactly?
3. A critic can never bias you as a baseline but can bias you through bootstrapping.
   Reconcile those two statements precisely, naming the measurability argument for the
   first and the action-dependence argument for the second.
4. Why explained variance rather than MSE? Tie it to the wrong-level/wrong-ranking result
   and to your negative control.
5. Why is the KL penalty implemented as a per-token reward? Give the density argument, the
   credit-assignment argument, and the exactness result with its number.

## Frontier-Lab Interview Angle

This is the layer where "I've used TRL" and "I understand RLHF" separate. The questions
that discriminate are all about the value head — what it regresses onto, whether its error
biases you, how you'd diagnose it — because that is the part of the pipeline that is both
genuinely subtle and genuinely load-bearing in production. It is also the most common real
bug: an unconverged or mis-targeted critic produces no crash, no NaN, a perfectly
zero-mean whitened advantage, and a run that just quietly doesn't improve. Being able to
name explained variance as the instrument and the zero-the-baseline probe as the decisive
experiment is a strong senior signal.
