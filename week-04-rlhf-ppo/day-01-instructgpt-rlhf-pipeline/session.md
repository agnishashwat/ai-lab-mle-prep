# Week 04 Day 01 — The InstructGPT Pipeline & the Reward Model

## Objective

Two things, and the second is the load-bearing one.

**(a)** Get the canonical RLHF pipeline exact: SFT → reward model → PPO-with-KL → evals,
with a precise account of *which of the four models in the loop can write, which can only
score, and which is frozen*, plus the memory/compute bill that follows.

**(b)** Build the environment Week 4 runs on: a **token-level LM small enough to enumerate
every sequence**. Week 3's ground truth came from a finite MDP (exact Bellman + central
finite differences), and that discipline is the reason its results are assertions rather
than claims. Week 4 needs the same discipline in a setting where *tokens, EOS, sequence
reward and length are literal*, so today you build `enumerate_sequences` / `exact_J` /
`exact_sequence_kl` once, and Days 2-5 spend them.

The day's substantive result: a **Bradley-Terry reward model is identified only up to a
per-prompt additive constant**, its error is concentrated *off the SFT distribution*, and
that single fact is the mechanism behind everything Day 4 calls "reward hacking."

## Prerequisites

- Week 3 in full, especially D5 (KL-constrained objective, the exponential tilt,
  \(r \to r+c\) invariance, per-token KL shaping).
- W3 D1's finite-MDP machinery (`mdp.py`) and the staged-harness convention.
- Logistic regression / MLE (for Bradley-Terry).

## Required Reading

- **InstructGPT (Ouyang et al. 2022) §3.5 "Models"** only — the three-stage description
  and Equation (2), the full PPO-ptx objective. Read Eq. (2) term by term; you will
  derive every term this week and implement all of them by Friday.
- Optional: the RM paragraph of §3.5 (the \(\binom{K}{2}\)-comparisons-per-prompt trick
  and why they batch a prompt's comparisons together — that is an overfitting *and* a
  compute argument, and it is a good interview question).

## 2-Hour Plan

> **Clock note.** Today's quiz block runs to 0:20 because it carries five outstanding
> Week-3 items. Concept is correspondingly compressed; the coding block is protected.

### 0:00-0:08 — Cold recall of CONCLUSIONS (new block, Week 3 retro action item)

*One line per result, spoken, no whiteboard, no derivations.* Week 3's error mode was
**retrieval, not comprehension** — every D5 quiz miss was a fact demonstrably known and
written down correctly elsewhere. This block trains retrieval separately from derivation.
Rapid-fire, ~40 s each:

1. Closed form of the KL-regularized optimum, and \(J^*\) in one symbol.
2. The two \(\beta\) limits. (**Known gap.** \(\beta\to\infty\Rightarrow\pi_{\text{ref}}\),
   NOT uniform; \(\beta\to0\Rightarrow\) uniform over \(\arg\max\{r: a\in\mathrm{supp}\,\pi_{\text{ref}}\}\)
   — over the *support*.) Then the three consequences of support confinement.
3. \(d\mathbb{E}[r]/d\mathrm{KL} = ?\) and what that makes the frontier's shape.
4. Why \(Z\) is intractable — one clause, and it is not "we lack logits."
5. Wrong LEVEL vs wrong RANKING in a critic: which causes variance, which causes bias?
6. The Ψ ladder, all seven rungs, in order.

### 0:08-0:20 — Recall Quiz (Week 3 backlog, graded)

1. **k1/k2/k3, with the crossover.** Define all three. Which is unbiased, which is
   \(\ge 0\), which is both, and *by what mechanism* is k3 both? Then the number:
   \(\mathrm{var}(k_1)/\mathrm{var}(k_3)\) at KL≈0.004 versus at KL≈0.65, and the
   one-sentence reason it reverses.
2. **Re-ask, D5 drill D2 (scored 0.35).** "Our RM score is up 40%; human eval is down."
   Full answer required: the mechanism *in terms of where the RM is valid*; your own
   measured hump numbers; **how you would pick the new \(\beta\)**; and how you would have
   caught it a day earlier. Four clauses in the question ⇒ four parts in the answer.
3. **Re-ask, the λ half of D5 drill D3 (scored 0.5).** Sparse reward at EOS: name both
   credit-assignment mechanisms, **how they interact**, and your measured λ numbers.
4. **Perennial — off-policy-by-accident (0.45, REGRESSED from 0.70).** The full causal
   loop: rollout distortion → what it reinforces → what collapses; then the fix order,
   why clipping is per-token, why top-p is *unrepairable*, and the two instruments.
5. **Perennial — sources of REINFORCE variance.** Best so far is 6/7. The standing miss
   is reward-model **noise**. Keep SOURCES separate from MECHANISMS.

> **Grading rubric, enforced all week.** Every multi-clause question is scored as two
> line items: *first clause* and *everything after*. Answering only the first clause caps
> the item at 0.5 regardless of how good the first clause is. Habit to apply before
> speaking: **count the clauses in the question and the paragraphs in your answer; if the
> first number is larger, you are not finished.**

### 0:20-0:42 — Concept: the pipeline as an optimization problem

- **Three stages, and what each one *buys*.**
  - **SFT** moves the *support* and the format. Recall D5: reverse KL confines
    \(\mathrm{supp}(\pi_\theta)\subseteq\mathrm{supp}(\pi_{\text{ref}})\) at every
    \(\beta>0\), so RL can only reweight what SFT made reachable. **SFT is the only stage
    that can teach; RL elicits.**
  - **RM** converts a *comparison* into a *scalar field* over completions. The reason we
    fit a model rather than asking humans online is throughput: RL needs \(10^5\)-\(10^6\)
    scores, and humans give \(10^4\) comparisons.
  - **PPO+KL** maximizes the scalar subject to staying where the scalar is trustworthy.
- **Four networks, and the permissions table.** Policy \(\pi_\theta\) (trainable, and the
  *only* thing that can write), value head \(V_\phi\) (trainable, scores, never writes —
  Day 3), reward model \(r_\psi\) (frozen, scores), reference \(\pi_{\text{ref}}\) (frozen,
  can write but doesn't — it is a *measuring stick*). Interview-grade consequence: the RM
  can only ever *rank* what the policy already produced. **No amount of RM quality creates
  a behaviour.**
- **The memory bill** (ties to W1 D3 and Week 8). Only \(\pi_\theta\) (+ \(V_\phi\)) carry
  optimizer state; \(r_\psi\) and \(\pi_{\text{ref}}\) are inference-only. Do the 7B
  arithmetic out loud: trainable ≈ 16 bytes/param under bf16-Adam vs frozen at 2. Then the
  standard tricks — share the trunk between policy and value; drop \(\pi_{\text{ref}}\)
  entirely by caching reference logprobs at rollout time (they never change!); LoRA the
  policy so \(\pi_{\text{ref}}\) is "the same weights with the adapter off."
- **Why the reward is terminal.** RM scores a *completed* response. So the MDP has one
  scalar at EOS and \(T\) actions — exactly the sparse-terminal case from W3 D1, and the
  reason GAE and per-token KL shaping (both already yours) are not optional.
- **Preference data, briefly**: \(K\) completions per prompt → \(\binom{K}{2}\) pairs; all
  pairs from one prompt must go in the **same batch** (otherwise a prompt's completions are
  seen \(K-1\) times across epochs and the RM overfits), which is also \(K\times\) cheaper
  since the forward passes are shared.

### 0:42-1:05 — Derivation: Bradley-Terry, and what a reward model cannot know

1. **The model.** \(P(y_w \succ y_l \mid x) = \sigma\!\big(r(x,y_w)-r(x,y_l)\big)\).
   Derive the NLL \(\mathcal L(\psi) = -\mathbb{E}\big[\log\sigma(r_\psi(x,y_w)-r_\psi(x,y_l))\big]\)
   and its gradient. Show the gradient magnitude is \(\sigma(-\Delta)\) — i.e. **pairs the
   model already ranks correctly contribute almost nothing**, which is why annotator
   disagreement near \(\Delta\approx 0\) dominates the fit.
2. **The invariance, which is the whole point.** \(\mathcal L\) depends on \(r\) only
   through *differences within a prompt* ⇒ \(r_\psi(x,\cdot)\) and
   \(r_\psi(x,\cdot)+c(x)\) are **exactly equivalent**. State where the hypothesis is
   consumed. Three consequences, all of which you have already met from the other side:
   - RM scores are **not comparable across prompts**. A dashboard that averages raw RM
     score over a mixed prompt set is measuring the prompt mix.
   - The D5 fact \(\pi^*\propto\pi_{\text{ref}}e^{r/\beta}\) is invariant to \(r\to r+c\)
     — the *same* degree of freedom, seen in the optimum. **That is why GRPO's per-prompt
     group mean is free**, and why per-prompt centering is the only safe normalization.
   - Practical: RMs are usually centered per prompt or given a fixed anchor, and
     "reward went up" across prompts is uninterpretable without it.
3. **Where the error lives.** The RM is fit on completions drawn from \(\pi_{\text{ref}}\)
   (or a near neighbour). Off that distribution there is *no data and no constraint*, so
   generalization error grows with distance from the fitting distribution. Write the
   informal statement you will make quantitative in the coding block:
   \(\mathbb{E}\big[(r_\psi - r_{\text{gold}})^2\big]\) grows with
   \(\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\). **The KL penalty is not a regularizer for
   its own sake — it is a confidence region for the reward model.**
4. **Preview the week's loss** (InstructGPT Eq. 2), naming which day builds each term:
   \[
     \mathbb{E}\Big[\underbrace{r_\psi(x,y)}_{\text{today}}
       - \underbrace{\beta\log\tfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}}_{\text{W3 D5, wired D3}}\Big]
     + \underbrace{\gamma\,\mathbb{E}_{\text{pretrain}}[\log\pi_\theta]}_{\text{the ptx term}},
   \]
   optimized by \(L^{\text{CLIP}}\) (Day 2) with \(L^{\text{VF}}\) (Day 3).

### 1:05-1:47 — Coding: `week-04-rlhf-ppo/code/toy_rlhf.py`

Staged harness, repo convention (skeleton `shared/templates/exercise-harness-template.py`;
reference `week-03-rl-foundations/code/kl_rl.py`). This file is **infrastructure for the
whole week** — Days 2-5 import from it — so its checks matter more than usual.

**The environment (provided, read it).** `TokenEnv`: vocabulary of `V` tokens including a
distinguished `EOS`, max length `T_max`, state = the tuple of tokens emitted so far
(deterministic append), episode ends at `EOS` or `T_max`. Sized so that the set of
terminating sequences is **fully enumerable** (\(\mathcal O(V^{T_{\max}})\), a few thousand).
Policy is tabular softmax: `theta[state_id, token]`.

**Your TODOs, one stage each:**

1. `enumerate_sequences(env)` — every terminating token sequence, with its length.
2. `sequence_logprob(theta, seq)` — \(\sum_t \log\pi_\theta(a_t|s_t)\), the literal
   token-level accumulation. (Same `log_softmax`→`gather` idiom as W1/W3, now over a tree.)
3. `exact_J(theta, reward_fn)` — \(\sum_{\text{seqs}} p_\theta(y)\,r(y)\) by enumeration.
   **This is the week's ground truth.** No sampling, no Monte Carlo.
4. `exact_sequence_kl(theta, theta_ref)` — \(\sum_y p_\theta(y)\log\frac{p_\theta(y)}{p_{\text{ref}}(y)}\)
   by enumeration.
5. `bt_loss_and_grad(psi, pairs)` — Bradley-Terry NLL + analytic gradient.
6. `fit_reward_model(pairs, ...)` — fit a tabular \(r_\psi(y)\) by gradient descent.
7. `sample_preferences(rng, gold_r, sampler, n_pairs, noise)` — draw completion pairs from
   a given sampler and label them by the gold reward through a **noisy** BT flip
   (annotator disagreement is a parameter, not an afterthought).

**Checks that must have teeth** (properties, not smoke):

- `enumerate_sequences`: probabilities of all terminating sequences sum to **exactly 1**
  under any \(\theta\) (a normalization identity — catches a dropped EOS branch or a
  double-counted prefix, which is the likeliest bug in the file).
- `exact_J` against a **large Monte-Carlo rollout** (structurally independent estimator);
  agreement within the MC standard error, and the *sign* of the discrepancy random over
  seeds. Also a **hand-computable case**: `T_max=1` reduces to a one-step bandit.
- `exact_sequence_kl`: **zero** at `theta is theta_ref` (exactly `0.0`, not `~0`);
  non-negative always; and — the good one — it must match
  \(\sum_t \mathbb{E}_{d_t}[\mathrm{KL}_t]\) computed by the *occupancy-weighted* route
  from W3 D5. (D5's real bug was logging the unweighted per-state mean and passing 94/94;
  this check is that bug wired in as an assertion.)
- `bt_loss_and_grad`: analytic gradient vs **central finite differences**; and the
  degenerate case \(r_w = r_l\) must give loss exactly \(\log 2\).
- **The identifiability assertion (the day's headline).** Fit on pairs from a gold reward.
  Assert (i) recovery is **not** exact: \(\max|r_\psi - r_{\text{gold}}|\) is large; but
  (ii) after subtracting the per-prompt mean from both, recovery is tight. Then assert the
  *negative control*: a naive cross-prompt comparison using uncentered scores has
  **worse-than-chance** pairwise accuracy on a prompt pair chosen to have a large
  constant offset. Make the bug detectably a bug.
- **The validity-decays-with-KL assertion.** Fit the RM on completions sampled from
  \(\pi_{\text{ref}}\). Evaluate \(\mathbb{E}[(r_\psi-r_{\text{gold}})^2]\) under a ladder
  of tilted policies (reuse D5's `tilted_policy` at several \(\beta\)) and assert the error
  is **monotone increasing in the measured sequence KL**. Record the numbers; Day 4 replays
  this exact curve as the mechanism behind the Goodhart hump, and having the RM *learned*
  rather than hand-specified is what makes Day 4 stronger than W3 D5.
- Annotator-noise sweep: RM pairwise accuracy vs label-noise rate, ≥3 seeds, report
  mean ± spread (W1 D5 ablation discipline: one seed can lie).

**Carried-forward chore, do it first (2 min).** `week-03-rl-foundations/code/kl_rl.py`,
`tilted_policy`: `np.log(pi_ref)` is unguarded, so it emits a `RuntimeWarning` and reaches
\(-\infty\) by IEEE accident rather than by intent. Guard inside the log
(`np.log(np.where(pi_ref > 0, pi_ref, 1.0))`) and set the forbidden entries explicitly.
Re-run `kl_rl.py` and confirm 94/94 still green.

### 1:47-1:57 — Interview Drill

Two, ~4 min each. Answer the *whole* question; the second half is scored separately.

1. **"Walk me through RLHF end to end, and tell me what each stage can and cannot fix."**
   Then: your base model refuses a capability the RM rewards. Which stage do you touch,
   and why can't RL fix it? (Wants: support confinement, elicit-vs-teach, SFT first.)
2. **"Someone hands you a reward model with 78% pairwise accuracy on a held-out set. What
   do you still not know?"** (Wants: accuracy is measured *on-distribution* and the policy
   will leave that distribution; per-prompt identifiability so raw scores don't compare;
   calibration ≠ accuracy; where the errors are concentrated — near-ties or the tail?; and
   what you would measure instead.)

### 1:57-2:00 — Deliverable & Next Session

Write `deliverable.md`: the pipeline with the permissions table, the BT derivation + the
per-prompt invariance, and the measured RM-error-vs-KL curve. Commit `code/toy_rlhf.py`.
Next: **Day 2 — the PPO objective**, where the ratio's gradient at \(\theta=\theta_{\text{old}}\)
is shown to *be* the Week-3 policy gradient, and clipping is shown to be an implicit local
KL trust region that does something different from D5's explicit penalty.

## Checkpoint Questions

1. Name the four models in an RLHF loop, which are trainable, which can *write*, and give
   the 7B memory bill for each class. Which one can you delete outright, and how?
2. Derive the Bradley-Terry NLL and show \(r\) is identified only up to a per-prompt
   constant. Give three consequences, one of which must be about GRPO.
3. Why is the KL penalty better described as a *confidence region for the reward model*
   than as a regularizer? Cite your measured error-vs-KL numbers.
4. Why must all \(\binom{K}{2}\) comparisons from one prompt sit in the same batch? Give
   both the statistical and the compute reason.
5. "RLHF elicits, it cannot teach." State the formal fact this rests on, at which \(\beta\)
   it holds, and the two independent proofs.

## Frontier-Lab Interview Angle

"Walk me through RLHF" is the single most-asked post-training question, and almost everyone
recites three boxes. The differentiators are the ones today builds: knowing that the reward
model is identified only up to a per-prompt constant (and that this is the *same* degree of
freedom that makes GRPO's group baseline free), knowing that the KL term is a statement
about where the RM's *training data* was, and being able to state the memory bill for four
models without pausing. The RM-error-vs-KL curve you measure today is also a portfolio
asset: it turns "reward hacking" from a war story into a plot.
