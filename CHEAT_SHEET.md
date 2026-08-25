# AI Lab MLE Prep - Interview Revision Guide

This is the short-form review of the substantive work committed through `9951e5b`
(2026-08-25): Weeks 1-6 and Week 8 Days 1-3.

It is organized by **ideas**, not by daily curriculum. Numerical observations marked
**Repo result** came from small controlled fixtures; use them to explain mechanisms,
not as scaling claims about production LLMs.

> **Math rendering:** GitHub supports textbook-style math in Markdown. This document
> uses fenced `math` blocks, which GitHub renders as equations. A source editor,
> including the editor canvas, will still show the underlying LaTeX; switch to a
> rendered/preview view to see typeset notation.

## The whole course in one mental model

1. Language modeling is stable token-level maximum likelihood with strict shape,
   masking, and reduction semantics.
2. Transformers add causal content retrieval and position information.
3. Generation exposes the train/inference boundary and trades compute for KV-cache
   memory.
4. Policy gradients turn sequence reward into score-weighted token updates.
5. Baselines, GAE, PPO, and GRPO are different answers to variance, credit assignment,
   and rollout reuse.
6. KL regularization defines a reference-relative reward geometry and connects RLHF
   to DPO.
7. Preference optimization and RLVR are only as trustworthy as their comparison graph,
   support, verifier, and independent evaluation.
8. Every algorithm is also a systems problem: peak live bytes and exposed
   communication decide whether it can run.

---

## Week 1 - Language modeling, PyTorch, and experimental discipline

### Core language-model objective

An autoregressive language model factorizes a sequence into next-token predictions:

```math
p_\theta(x_{1:T})
=
\prod_{t=1}^{T} p_\theta(x_t \mid x_{1:t-1})
```

Training is maximum likelihood, usually implemented as mean token cross-entropy:

```math
\mathcal L_{\mathrm{CE}}
=
-\frac{1}{N}
\sum_{n=1}^{N}
\log \mathrm{softmax}(z_n)_{y_n}
```

The central gradient is:

```math
\nabla_z \mathcal L
=
p - e_y
```

Interpretation:

- Every token position is a vocabulary classifier.
- Parameters are shared across positions.
- The gradient moves probability toward the target and away from alternatives.
- Gradient components sum to zero, so probability mass is redistributed.

Perplexity is:

```math
\mathrm{PPL}
=
\exp(\mathcal L_{\mathrm{CE}})
```

PPL is an effective branching factor. Its floor is 1; a uniform model over vocabulary
size V has PPL = V; a confidently wrong model can have PPL much larger than V.

### Numerical and tensor invariants

Stable log-softmax:

```math
\log \mathrm{softmax}(z)
=
z - \mathrm{logsumexp}(z)
```

Compute log-sum-exp after subtracting the maximum. Avoid `softmax(z).log()`, which can
underflow to `log(0)`.

Shape contract:

```text
logits:             (B, T, V)
targets:            (B, T)
target logprobs:    (B, T)
sequence logprobs:  (B,)
```

Use `gather` to read target log-probabilities. Flatten cross-entropy explicitly:

```python
F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
```

Passing `(B,T,V)` directly is dangerous because PyTorch treats dimension 1 as the
class dimension for higher-rank inputs. If T happens to equal V, the error can look
plausible.

Two masking operations must not be confused:

- Exclude entries from softmax: write `-inf` before the reduction.
- Exclude entries from a mean: multiply by a 0/1 mask and divide by valid count.

Never sum values containing `-inf`. Guard an all-false mean mask with a differentiable
zero such as `values.sum() * 0`.

### Autograd and mixed precision

`backward()` traverses the dynamic graph in reverse, applies vector-Jacobian products,
and **accumulates** into leaf `.grad` buffers.

- `detach()` cuts one branch from the graph.
- `no_grad()` suppresses graph construction in a region.
- Gradients must be cleared deliberately.
- Parameter mutation belongs under `no_grad()`.

Mixed-precision rules:

- fp16 overflows near 65,504; subtract maxima before exponentiation.
- fp16's roughly `6e-5` threshold is the minimum **normal** number; subnormals extend
  lower with less precision.
- bf16 has an fp32-like exponent range but fewer mantissa bits.
- Scale the loss before backward, then unscale before clipping and stepping.
- Keep normalization, softmax, log-sum-exp, and long reductions in fp32 when needed.

Under the repository's classic mixed-precision Adam assumption:

```text
bf16 weight       2 bytes/parameter
bf16 gradient     2
fp32 master       4
fp32 Adam m,v     8
total            16 bytes/parameter
```

This is an assumption bundle, not a universal constant.

### Training-loop checklist

Correct order:

```text
seed before model creation
create model and optimizer
create separate train/eval RNG streams

for each optimizer step:
    set scheduled LR
    for each microbatch:
        forward
        backward(loss / accumulation_steps)
    clip gradients
    optimizer.step()
    optimizer.zero_grad()
    periodically evaluate with eval() + no_grad()
```

Effective batch:

```math
B_{\mathrm{effective}}
=
B_{\mathrm{micro}}
\times
N_{\mathrm{accum}}
\times
N_{\mathrm{data\ parallel}}
```

AdamW performs decoupled weight decay:

```math
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

Common silent bugs:

- Softmax before `cross_entropy`.
- Clearing gradients inside the accumulation loop.
- Forgetting to clear gradients between optimizer steps.
- Mixing evaluation RNG into the training stream.
- Comparing pre-update train loss with post-update validation loss.
- Treating `eval()` and `no_grad()` as interchangeable.

### Evaluation and ablations

Before interpreting a metric:

1. Verify preprocessing, labels, masks, and axes.
2. Verify split isolation and contamination controls.
3. Verify evaluation mode.
4. Compare with a meaningful baseline, often `ln(V)` for uniform CE.
5. Change one variable at fixed compute.
6. Use multiple seeds.
7. Require the effect to exceed run-to-run variation.

The overfit-one-batch test is the cheapest end-to-end check of
data -> model -> loss -> backward -> optimizer.

**Repo result:** a controlled LR ablation clearly separated underfit, successful
optimization, and instability. A previously broken evaluation harness had mimicked a
huge generalization gap, reinforcing the rule: **verify the measurement system before
diagnosing the model**.

**Interview line:** "Shape correctness, reduction denominators, RNG isolation, and loop
ordering are part of model correctness, not implementation style."

---

## Week 2 - Transformers, decoding, and inference

### Attention

Scaled dot-product attention is:

```math
\mathrm{Attn}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}} + M
\right)V
```

Why scale? If query and key components are independent with variance sigma squared:

```math
\mathrm{Var}(q^\top k)
=
d_h \sigma^4
```

Dividing by the square root of head dimension keeps score scale roughly independent of
head width. Without it, softmax becomes nearly one-hot and its Jacobian approaches
zero.

Causal masking sets future scores to negative infinity **before** softmax:

```math
M_{ij}
=
\begin{cases}
0, & j \le i \\
-\infty, & j \gt i
\end{cases}
```

Every attention row must retain at least one finite key. A fully masked row produces
0/0 and NaNs after softmax.

### Position and block structure

Without position information, attention is permutation-equivariant.

RoPE rotates query and key feature pairs by position. Its defining relative-position
property is:

```math
\left\langle R_m q,\; R_n k \right\rangle
=
q^\top R_{n-m} k
```

Apply RoPE to Q and K, not V, after splitting heads and before score computation.

A pre-LN decoder block is:

```math
x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x))
```

```math
x \leftarrow x + \mathrm{MLP}(\mathrm{Norm}(x))
```

For a residual update `x_next = x + f(x)`, the Jacobian contains an identity term:

```math
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=
I + J_{f_\ell}
```

This provides a gradient highway. It is a strong stability intuition, not a proof that
deep pre-LN networks can never diverge.

RMSNorm rescales by root mean square without mean subtraction:

```math
\mathrm{RMSNorm}(x)
=
\gamma
\odot
\frac{x}
{\sqrt{\mathrm{mean}(x^2)+\epsilon}}
```

### Decoding is an objective choice

Temperature:

```math
p_i(T)
=
\mathrm{softmax}(z/T)_i
```

- T approaching zero concentrates on maxima.
- Large T approaches uniform.
- Positive T never changes greedy argmax.

Top-k retains the k largest logits. Top-p retains the smallest sorted set whose
cumulative probability reaches p. For top-p, keep the crossing token and scatter the
sorted mask back to original token IDs.

Correct composition:

```text
temperature -> top-k -> top-p -> one final softmax -> sample
```

Beam search scores complete paths by sums of log-probabilities. Raw sums favor shorter
completed sequences because every added term is non-positive. Length normalization is
a heuristic and requires meaningful EOS handling.

Use greedy/beam for low-entropy, closed-ended tasks. Open-ended text occupies a broad
typical set; mode-seeking decoding can create repetition attractors, so temperature
plus nucleus sampling is often more appropriate.

### KV cache

Cache K and V because future tokens reuse them. A past token's Q was needed only when
that token was the query.

KV memory:

```math
\text{KV bytes}
=
2
\times L
\times H_{\mathrm{KV}}
\times d_h
\times B
\times T
\times b_{\mathrm{dtype}}
```

The factor 2 is K plus V. Use KV heads, not query heads.

For a new query at local row i after P cached positions, legal key j satisfies:

```math
j \le P+i
```

Critical invariant: **feed every token exactly once**.

```text
prefill: process the whole prompt once and populate the cache
decode:  feed only the newly selected token with its absolute position
```

Feeding the full running context together with an existing cache double-counts tokens
and positions.

KV caching changes standard autoregressive attention from recomputing the full prefix
at every step to scanning cached K/V. It saves compute but grows memory linearly with
live tokens.

- Prefill is parallel and often compute-bound.
- Decode is sequential and often memory-bandwidth-bound.
- Batching, continuous batching, PagedAttention, GQA/MQA, quantization, and speculative
  decoding are major serving levers.

### Best debugging invariants

1. **Causality:** perturb future tokens; past logits must not change.
2. **Cache equivalence:** cached and uncached logits/tokens must agree.
3. **Finite rows:** no attention row is fully masked.
4. **Shape sensitivity:** use unequal B, T, and V so axis bugs cannot hide.
5. **End-to-end:** overfit a tiny batch.

Do not trust a regression whose fixture is insensitive to the bug. The repository's
successor task could ignore attention and still generate correctly, so the decisive
cache test used a model whose logits genuinely depended on legal context.

**Interview line:** "A KV cache is an optimization; the strongest test is semantic
equivalence to uncached decoding."

---

## Week 3 - Reinforcement-learning foundations

### MDPs and Bellman equations

An MDP contains states, actions, transition dynamics, rewards, a discount, and an
initial-state distribution.

```math
G_t
=
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
```

```math
V^\pi(s)
=
\mathbb E_\pi[G_t \mid s_t=s],
\qquad
Q^\pi(s,a)
=
\mathbb E_\pi[G_t \mid s_t=s,a_t=a]
```

```math
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
```

Bellman expectation:

```math
V^\pi
=
r_\pi + \gamma P_\pi V^\pi
```

```math
V^\pi
=
(I-\gamma P_\pi)^{-1}r_\pi
```

Bellman policy-evaluation and optimality operators are gamma-contractions:

```math
\lVert TV-TW\rVert_\infty
\le
\gamma \lVert V-W\rVert_\infty
```

Therefore each has a unique fixed point and iterative application converges
geometrically.

LLM mapping:

- State: prompt plus generated prefix.
- Action: next token.
- Dynamics: deterministic append.
- Reward: often one terminal scalar.
- Hard part: assigning that scalar to many token decisions.

### Policy-gradient theorem

Using the log-derivative trick:

```math
\nabla_\theta J
=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)
\nabla_\theta \log p_\theta(\tau)
\right]
```

When environment dynamics do not depend on theta:

```math
\nabla_\theta \log p_\theta(\tau)
=
\sum_t
\nabla_\theta
\log \pi_\theta(a_t\mid s_t)
```

The environment still determines sampled trajectories, but its derivative is not
needed.

The expected-gradient-log-probability lemma is:

```math
\mathbb E_{a\sim\pi}
\left[
\nabla\log\pi(a\mid s)
\right]
=
0
```

It justifies both reward-to-go and baselines:

```math
\nabla J
=
\mathbb E
\left[
\sum_t
\gamma^t
\bigl(G_t-b(h_t)\bigr)
\nabla\log\pi(a_t\mid s_t)
\right]
```

A valid baseline may depend on anything known before the current action is sampled. It
must not depend on that sampled action, and policy loss must not differentiate through
it.

### Variance reduction

Reward-to-go removes noise from past rewards. A baseline removes predictable
state/history-dependent return. These mechanisms stack.

The variance-minimizing scalar baseline is score-norm weighted:

```math
b^*(s)
=
\frac{
\mathbb E[
\lVert\nabla\log\pi\rVert^2 G_t
\mid s_t=s
]
}{
\mathbb E[
\lVert\nabla\log\pi\rVert^2
\mid s_t=s
]
}
```

The value function equals this optimum only under additional conditional
independence.

**Repo result:** reward-to-go reduced variance about 3x; adding a fitted baseline
reduced it more than 11x. An invalid action-dependent baseline had the lowest variance
but a badly wrong mean. **Low variance is not evidence of correctness.**

### GAE

TD residual:

```math
\delta_t
=
r_t
+\gamma V(s_{t+1})
-V(s_t)
```

Generalized advantage estimation:

```math
\hat A_t^{\mathrm{GAE}}
=
\sum_{\ell=0}^{\infty}
(\gamma\lambda)^\ell
\delta_{t+\ell}
```

Production recursion:

```math
A_t
=
\delta_t
+\gamma\lambda(1-d_t)A_{t+1}
```

- Gamma changes the objective.
- Lambda changes the estimator.
- Smaller lambda uses more bootstrap: usually lower variance, more critic-error bias.
- At lambda = 1, frozen critic error telescopes into a baseline and cannot bias the
  policy gradient.
- With action-dependent EOS, even a constant critic offset can become action-dependent
  through remaining length; "level error is free" is conditional.

### KL-regularized RL

For reward r, reference q, and KL price beta:

```math
\max_\pi
\left[
\mathbb E_\pi[r]
-
\beta D_{\mathrm{KL}}(\pi\Vert q)
\right]
```

The optimum is an exponential tilt:

```math
\pi^*(a)
=
\frac{
q(a)\exp(r(a)/\beta)
}{
\sum_b q(b)\exp(r(b)/\beta)
}
```

Along the optimal reward-KL frontier:

```math
\frac{dR}{dK}
=
\beta
```

Beta is the marginal reward price of one additional nat of divergence.

For a sampled action from the policy, let:

```math
\ell
=
\log\frac{\pi(a)}{q(a)}
```

Useful KL estimators:

```math
k_1 = \ell
```

```math
k_3
=
\exp(-\ell)-1+\ell
```

`k1` is unbiased but signed. `k3` is unbiased and pointwise nonnegative. Use `expm1`
near zero. `k3` was lower variance near small KL in the fixture but worse at large KL;
it does not universally dominate.

Sequence KL decomposes into expected token log-ratios, so token reward can be shaped by
`-beta * KL_estimate`.

**Interview line:** "The reference defines support and behavior; beta defines the
reward-per-nat exchange rate."

---

## Week 4 - RLHF, PPO, and failure diagnosis

### RLHF pipeline and reward models

1. **SFT** teaches format and makes useful behavior practically reachable.
2. **Reward modeling** converts same-prompt comparisons into a scalar proxy.
3. **PPO plus reference KL** optimizes the proxy while limiting distribution shift.
4. **Independent evaluation** detects what the proxy missed.

Bradley-Terry preference probability:

```math
P(y_w \succ y_l \mid x)
=
\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr)
```

Only within-prompt differences are identified:

```math
r(x,y)
\mapsto
r(x,y)+c(x)
```

Raw reward-model scores are therefore not automatically comparable across prompts.

### PPO

PPO reuses rollouts from a frozen old policy:

```math
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\mathrm{old}}(a_t\mid s_t)
}
```

Clipped surrogate:

```math
L^{\mathrm{CLIP}}
=
\mathbb E
\left[
\min
\left(
\rho A,\;
\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A
\right)
\right]
```

Clipping:

- Stops rewarding some ratio movement that already goes too far.
- Is sign-dependent: behavior differs for positive and negative advantages.
- Does not project the policy into a trust region.
- Does not directly bound expected KL.
- Resets around a new old policy each iteration, so small local moves can accumulate
  into a large walk from the fixed reference.

Whole-sequence importance ratios multiply across tokens, so effective sample size can
collapse exponentially with sequence length even when most individual token ratios
look harmless.

**Repo result:** clip-only PPO drifted far from the reference, while explicit KL
controlled the global walk. Clipping reduced local movement but introduced directional
bias.

### Token-level PPO loss and freeze line

Token reward:

```math
r_t
=
r_\psi(y)\,\mathbf 1[t=L-1]
-
\beta\,\hat k_t
```

Combined minimization loss:

```math
\mathcal L
=
-L^{\mathrm{CLIP}}
+c_1 L^{\mathrm{value}}
-c_2\,\mathbb E[H(\pi)]
```

For language, reference KL is usually a better behavioral target than a uniform
entropy bonus.

Correct order:

1. Sample on-policy without support truncation.
2. Cache old and reference logprobs.
3. Shape rewards with the current beta.
4. Compute frozen old values.
5. Compute GAE.
6. Form value targets **before** whitening.
7. Whiten only policy advantages.
8. Recompute only new logprobs and current values during PPO epochs.

Recomputing `logp_old` inside an epoch makes every ratio 1 and silently turns PPO into
repeated stale-data policy gradient.

### Failure-mode split

> **Objective wrong:** reward hacking and length exploitation.
> **Optimizer broken:** entropy collapse and value divergence.

Use paired metrics:

| Question | Paired evidence |
|---|---|
| Is the proxy being hacked? | Proxy reward + independent gold |
| Is the policy drifting? | KL to reference + beta |
| Is support collapsing? | Entropy + effective/unique support |
| Is length being exploited? | Length + length-controlled quality |
| Is the critic healthy? | Explained variance + value loss |
| Is PPO clipping active? | Clip fraction + KL to old policy |

Off-policy sampling without valid correction is a different gradient. Finite
temperature preserves support; top-p deletes support and can make exact importance
correction impossible.

Length reduction is not neutral. Summing asks whether the marginal token beats zero;
averaging asks whether it beats the current average.

The deadly triad is:

```text
function approximation + bootstrapping + off-policy visitation
```

It can make a critic diverge even when each update is clipped.

### Adaptive beta and checkpoint selection

For target KL delta, a scale-free feedback controller is:

```math
\beta
\leftarrow
\beta
\left[
1
+K_\beta
\left(
\frac{K}{\delta}-1
\right)
\right]
```

Small gain responds slowly; large gain oscillates. The controller is only as good as
its KL sensor.

**Repo result:** removing KL improved proxy reward while degrading independent gold.
The no-KL run peaked and then collapsed below the reference. The final checkpoint was
substantially worse than the selected interior checkpoint.

> Early stopping estimates an interior gold optimum; it is not merely generic
> regularization.

**Correction:** the apparent "no critic" advantage was smaller than the measured seed
spread. It is a null at horizon 5 with three seeds, not evidence that critics are
generally unnecessary.

**Interview line:** "PPO clipping controls some local updates; fixed-reference KL and
independent gold evaluation control the actual RLHF risk."

---

## Week 5 - DPO and preference optimization

### Bradley-Terry identifiability

Pairwise preferences constrain reward **differences**, not absolute levels.

Three separate issues:

1. **Location gauge:** adding a prompt-specific constant changes nothing.
2. **Scale:** stochastic disagreement can identify finite scale; deterministic
   separability can send score norm to infinity.
3. **Comparison graph:** graph structure determines whether a finite MLE exists.

**Ford's condition:** the finite Bradley-Terry MLE exists iff the directed wins graph
is strongly connected.

A deterministic strict ranking is acyclic. More perfectly consistent data can improve
accuracy while making the score norm diverge. High pair accuracy can therefore be
evidence of separability, not calibrated reward.

With symmetric label flips epsilon, a saturated pair has finite observed margin:

```math
\Delta_{\max}
=
\log
\frac{1-\epsilon}{\epsilon}
```

Disagreement acts as a finite-margin regularizer.

### DPO derivation

KL-regularized optimal policy:

```math
\pi^*(y\mid x)
=
\frac{
\pi_{\mathrm{ref}}(y\mid x)
\exp(r(x,y)/\beta)
}{
Z(x)
}
```

Invert it:

```math
r(x,y)
=
\beta
\log
\frac{
\pi^*(y\mid x)
}{
\pi_{\mathrm{ref}}(y\mid x)
}
+
\beta\log Z(x)
```

For two completions of the same prompt, the partition function cancels. Define:

```math
M
=
\beta
\left[
\log
\frac{\pi_\theta(y_w\mid x)}
{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log
\frac{\pi_\theta(y_l\mid x)}
{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]
```

DPO loss:

```math
\mathcal L_{\mathrm{DPO}}
=
-\mathbb E[\log\sigma(M)]
```

Interpretation:

> DPO is a Bradley-Terry reward model whose parameterization happens to be a
> normalized policy.

Its implicit reward is:

```math
\hat r_\theta(x,y)
=
\beta
\log
\frac{
\pi_\theta(y\mid x)
}{
\pi_{\mathrm{ref}}(y\mid x)
}
```

Exact equivalence to "fit a reward model, then tilt the reference" requires compatible
function classes, a finite reward solution, consistent data/reference assumptions, and
optimization to convergence.

### Correct implementation

Five contracts:

1. Shift causal labels exactly once.
2. Mask prompt and padding; include completion tokens and EOS.
3. Perform log-softmax and long sequence sums in fp32.
4. Initialize policy and reference from the same SFT checkpoint.
5. Log chosen and rejected marginals, KL, length, and implicit reward - not pair
   accuracy alone.

Preference margin can improve while both chosen and rejected log-probabilities fall.
Pair loss only cares that the rejected side falls more.

Mean normalization changes the length prior:

- Sum score lengthens if the new token's drift is positive.
- Mean score lengthens if its drift exceeds the current average drift.

### Variants: score axis versus link axis

Preference methods differ along two independent choices:

- **Score/reward definition:** reference-relative or reference-free, sum or mean, odds.
- **Link/loss:** logistic, smoothed logistic, squared target, margin logistic, prospect
  value.

| Method | Main distinction |
|---|---|
| DPO | Reference-relative logistic margin |
| cDPO | Smoothed labels and finite target margin |
| IPO | Squared finite target |
| KTO | Prospect-style bounded influence |
| ORPO | Odds preference plus SFT anchor |
| SimPO | Reference-free mean score with margin |

The link controls finite versus runaway margins. The score controls reference anchoring
and length behavior.

### Failure modes and corrections

- Pair accuracy can become 100% immediately while margins continue diverging.
- Sparse comparison coverage leaves off-data probability mass unconstrained.
- Preference loss can improve monotonically while independent gold gets worse.
- Chosen logprob falling is not by itself a mismatch; inspect both marginals.
- Adding SFT can protect chosen likelihood.

Important corrections:

- On finite support, DPO margin may diverge but reverse KL is bounded:

```math
D_{\mathrm{KL}}
(\pi\Vert\pi_{\mathrm{ref}})
\le
\max_y
\left[
-\log\pi_{\mathrm{ref}}(y)
\right]
```

- Mean normalization does not remove length bias.
- A bounded KTO loss does not guarantee a finite minimizer.
- IPO compresses preference strength; it does not impose one universal cap.
- For the implemented DPO gradient, beta approaching zero is the no-op boundary.

**Interview line:** "The pair loss controls observed ordering; the real safety panel is
marginals, off-data mass, length, KL, and independent gold."

---

## Week 6 - GRPO, RLVR, pass@k, and reasoning curricula

### GRPO as a Monte Carlo critic

PPO learns a token-state critic. GRPO estimates a prompt baseline from G same-prompt
completions:

```math
A_i
=
r_i-\bar r
```

With the self-including group mean:

```math
\mathbb E[\hat g]
=
\frac{G-1}{G}\nabla J
```

At G = 1, the gradient is exactly zero. Leave-one-out scaling removes this shrinkage
but not relative variance.

GRPO saves critic memory but gives every token in a completion the same sequence-level
advantage. It loses intra-sequence credit assignment.

For binary success probability p:

```math
P(\text{dead group})
=
p^G+(1-p)^G
```

Groups are dead when all samples receive the same reward. Dead groups are common both
on very hard and already-mastered prompts.

Group whitening is not merely conditioning:

- It removes every within-group constant.
- It changes prompt weighting.
- It makes alive binary groups receive similar squared advantage mass.
- Putting KL inside group-centered reward can nearly annihilate the KL signal; keep KL
  as a separate loss.

### RL with verifiable rewards

Let C be truly correct outputs and V be verifier-accepted outputs.

- Sound: V is a subset of C.
- Complete: C is a subset of V.
- False negatives waste compute.
- False positives create a reward-hacking surface.

**Soundness theorem:**

```math
V\subseteq C
\quad\Longrightarrow\quad
p_{\mathrm{observed}}
\le
p_{\mathrm{gold}}
```

If any reachable false positive exists:

```math
\sup_\pi
\left(
p_{\mathrm{observed}}
-p_{\mathrm{gold}}
\right)
=
1
```

Reward hacking is fundamentally a support property, not an average checker-accuracy
property.

With false-negative rate alpha and false-positive rate gamma:

```math
p_{\mathrm{observed}}
=
(1-\alpha)p_{\mathrm{gold}}
+
\gamma(1-p_{\mathrm{gold}})
```

Verifier precision changes as the policy changes. Even a small false-positive rate can
dominate rewarded outputs on sufficiently hard prompts.

RLVR removes a learned reward model from GPU, but adds a distributed checker system
with tail latency, timeouts, versioning, calibration, and retry semantics. Timeouts can
both slow training and create difficulty-dependent false negatives.

### pass@k and selectors

For independent samples:

```math
\mathrm{pass}@k(p)
=
1-(1-p)^k
```

Apply this transform **per prompt, then average**. Transforming mean p is wrong under
heterogeneous prompt difficulty.

Given n sampled completions and c correct ones, the unbiased finite-sample estimator is:

```math
\widehat{\mathrm{pass}@k}
=
1
-
\frac{
\binom{n-c}{k}
}{
\binom{n}{k}
}
```

The common plug-in using c/n is downward biased.

pass@k is oracle coverage, not deployment accuracy:

| Selector | Large-k behavior |
|---|---|
| Oracle | Succeeds if any correct output has support |
| Sound verifier filter | Depends on accepted-set precision |
| Majority vote | Succeeds only if the correct answer is modal |
| Learned ranker | Can concentrate extreme-score false positives |

Finite temperature preserves support; top-k/top-p delete it. Positive correlation
between samples makes the independent pass@k formula optimistic.

For continuous tie-free scores, best-of-n itself moves the output distribution:

```math
D_{\mathrm{KL}}
(\mathrm{BoN}_n\Vert\pi)
=
\log n-\frac{n-1}{n}
```

### Metric-aware curriculum

A curriculum should be a closed-loop controller over **current** policy difficulty,
not a static easy-to-hard ordering.

Assume local learning progress:

```math
\Delta p
\propto
p^a(1-p)^b
```

If evaluation uses pass@k, the metric-weighted optimum is:

```math
p^*
=
\frac{a}{a+b+k-1}
```

Under symmetric progress exponents a = b = 1:

```math
p^*
=
\frac{1}{k+1}
```

The exponents must be measured. The equality `p* = 1/2` holds generally when
`a = b + k - 1`, not simply whenever k = 1.

Curriculum design is constrained by:

- Dead-group probability.
- Group size and rollout budget.
- Verifier precision floor.
- False-negative compute tax.
- Contamination.
- Whitening-induced prompt reweighting.

Control in log-odds space, where the useful difficulty band is more stable. Add hints
when prompts are too hard; introduce harder problems when they are mastered.

### Integrated-loop boundaries

1. Hidden gold must not affect training updates.
2. Changing reachable verifier acceptance must affect the update.
3. Rewards and group advantages are detached.
4. The old policy generated the batch; the fixed reference anchors KL.
5. Controllers use completed history only.
6. Nonlinear metrics are computed per prompt before averaging.
7. Oracle coverage is never reported as selector accuracy.

Important null results:

- The tested toy runs never showed the planned pass@k coverage collapse. This does not
  prove large models cannot collapse; it means collapse needs a mechanism and direct
  measurement.
- The closed-loop curriculum's apparent gain over uniform sampling remained inside two
  paired standard deviations. It is a null, not "better."

**Interview line:** "GRPO replaces the critic with a same-prompt Monte Carlo baseline;
RLVR replaces the reward model with an accepted set. Both substitutions save model
state but introduce statistical and systems constraints."

---

## Week 8 - Training and distributed-systems math

### Peak memory, not parameter count

```text
training peak
~= model states
 + saved activations
 + temporary workspaces
 + communication buffers
 + allocator/runtime reserve

inference peak
~= weights
 + KV cache
 + temporary workspaces
 + allocator/runtime reserve
```

Classic 7B bf16 Adam persistent state:

```text
16 bytes/parameter * 7B
= 112,000,000,000 bytes
= 104.3 GiB
```

Weights alone are about 13.0 GiB. Unsharded full fine-tuning therefore does not fit on
an 80 GiB GPU before activations.

Naively materialized attention activation memory scales as T squared; hidden states and
KV cache scale linearly with T. FlashAttention avoids storing the full attention
matrix, though exact attention arithmetic remains quadratic.

KV cache:

```math
\text{KV bytes}
=
2LH_{\mathrm{KV}}d_hBTb
```

For the repository's 7B-class example, this is 512 KiB per live token:

- Batch 1, length 4096: 2 GiB.
- Batch 16, length 8192: 64 GiB.
- Reducing KV heads from 32 to 8 cuts memory 4x.

Capacity planning:

```text
usable KV = device capacity - weights - reserve
aggregate live-token slots = floor(usable KV / bytes per token)
```

For paged serving, budget the sum of active sequence lengths rather than
`batch * maximum_length`.

### LoRA, QLoRA, and checkpointing

LoRA update:

```math
W'
=
W_0
+
\frac{\alpha}{r}BA
```

Adapter parameters per target matrix:

```math
P_{\mathrm{adapter}}
=
r(d_{\mathrm{in}}+d_{\mathrm{out}})
```

For a square matrix, adapter/full ratio is `2r/d`. At d = 4096 and r = 16, it is
0.78125% per matrix.

Persistent-state approximations:

```text
full fine-tuning:  16P
LoRA:               2P + 16A
idealized QLoRA:  0.5P + 16A
```

QLoRA's `0.5P` is an ideal storage lower bound. Real systems add scales, metadata,
packing, dequantization buffers, activations, and runtime state.

Gradient checkpointing trades compute for saved activations. In a simplified L-layer
chain checkpointed every k layers:

```math
\mathrm{slots}(k)
=
\left\lceil\frac{L}{k}\right\rceil+k
```

The continuous optimum is near:

```math
k^*
\approx
\sqrt{L}
```

Checkpointing does not reduce weights, gradients, or optimizer state.

### ZeRO and FSDP

Under the 2 + 2 + 4 + 8 byte layout:

| Strategy | Per-rank persistent model state |
|---|---:|
| DDP / ZeRO-0 | `16P` |
| ZeRO-1 | `P(4 + 12/N)` |
| ZeRO-2 | `P(2 + 14/N)` |
| ZeRO-3 / full-shard FSDP | `16P/N` |

Only ZeRO-3 removes the replicated parameter floor.

`16P/N` is **not peak memory**. FSDP must materialize the active wrapped unit and may
also hold prefetched units, communication buffers, activations, and temporary kernels.

Frozen-base placement is independent of adapter sharding. Sharding only LoRA state
does not automatically shard the frozen base.

### Distributed collectives

For a ring over N ranks and message size M:

```math
\text{all-gather bytes/rank}
=
\frac{N-1}{N}M
```

```math
\text{reduce-scatter bytes/rank}
=
\frac{N-1}{N}M
```

```math
\text{all-reduce bytes/rank}
=
2\frac{N-1}{N}M
```

As N grows, ring all-reduce traffic approaches 2M per rank. Compute per rank shrinks
under strong scaling; communication retains a floor.

Every rank both sends and receives. "Gather" and "scatter" describe logical placement,
not one-way network traffic.

Approximate collective time:

```math
T_{\mathrm{collective}}
\approx
(\text{rounds})(\text{latency})
+
\frac{\text{bytes/rank}}{\text{effective bandwidth}}
```

Only communication exposed on the critical path increases wall time; overlapped
collectives may be largely hidden.

### Parallelism choices

| Axis | Partition | Main communication | Use when |
|---|---|---|---|
| DP/DDP | Examples | Gradient all-reduce | A replica fits |
| FSDP | State + examples | Parameter gather, gradient reduce-scatter | Replicated state does not fit |
| TP | Within-layer matrices | Frequent reductions/gathers | One layer/model replica does not fit; use fast links |
| PP | Layer stack | Boundary activations/gradients | Layers can be balanced across stages |
| EP | Experts/tokens | All-to-all | Sparse expert models |

Only data parallelism multiplies independent examples:

```math
B_{\mathrm{global}}
=
DP
\times
B_{\mathrm{micro/replica}}
\times
N_{\mathrm{accum}}
```

Tensor- and pipeline-parallel ranks cooperate on the same examples.

Pipeline bubble fraction for S stages and m microbatches:

```math
\text{bubble fraction}
=
\frac{S-1}{m+S-1}
```

More microbatches reduce bubbles but can hurt kernel efficiency and add launch or
communication overhead. Stage imbalance multiplies the loss.

Strong scaling:

```math
T_{\mathrm{step}}(N)
\approx
\frac{T_{\mathrm{compute}}(1)}{N}
+
T_{\mathrm{exposed\ communication}}
+
T_{\mathrm{bubble}}
+
T_{\mathrm{imbalance}}
```

### Systems debugging

| Symptom | First place to look |
|---|---|
| Collective hang | Per-rank call order, shapes, dtypes, failed rank |
| More GPUs are slower | Exposed communication after compute shrank |
| Periodic slow steps | Input/host stragglers, allocator, network contention |
| One pipeline stage is always busy | Layer or token imbalance |
| FSDP OOM only with prefetch | Multiple full wrapped units live together |
| Loss changes with world size | Double averaging or changed batch semantics |
| Multi-node TP collapses | Per-layer collectives placed on slow links |

**Interview line:** "Fit is a peak-liveness question; scaling is a critical-path
question."

---

## Rapid-recall formula sheet

| Topic | Formula |
|---|---|
| CE gradient | `grad_z L = p - one_hot(y)` |
| Perplexity | `PPL = exp(CE)` |
| Attention | `softmax(QK^T / sqrt(d_h) + mask)V` |
| KV cache | `2 * layers * KV_heads * head_dim * B * T * bytes` |
| Bellman | `V = r_pi + gamma P_pi V` |
| Policy gradient | `E[sum_t credit_t * grad log pi(a_t|s_t)]` |
| GAE | `A_t = delta_t + gamma lambda (1-done_t) A_(t+1)` |
| KL-optimal policy | `pi* proportional to pi_ref * exp(reward / beta)` |
| PPO ratio | `rho = pi_new / pi_old` |
| DPO score | `beta * [(log pi - log pi_ref)_win - (...)_lose]` |
| GRPO shrinkage | `E[g_hat] = (G-1)/G * grad J` |
| Dead binary group | `p^G + (1-p)^G` |
| pass@k | `1 - (1-p)^k` |
| Curriculum target | `p* = a / (a+b+k-1)` |
| LoRA parameters | `r(d_in + d_out)` |
| ZeRO-3 state | `16P/N` persistent bytes/rank under stated assumptions |
| Ring all-reduce | `2(N-1)M/N` bytes/rank |
| Pipeline bubble | `(S-1)/(m+S-1)` |

## Corrections worth remembering

- Pre-LN provides an identity gradient path; it does not guarantee stability.
- PPO clipping is not a global KL trust region.
- A critic's constant offset is harmless only when its residual effect remains
  history-measurable; variable EOS can break this.
- `k3` is not universally the lowest-variance KL estimator.
- The Week 4 no-critic "win" was a null.
- Clean deterministic Bradley-Terry data can cause separability and no finite MLE.
- DPO margins may diverge while finite-support reverse KL stays bounded.
- Mean sequence normalization changes length preference; it does not remove it.
- Bounded KTO loss does not imply a finite optimum.
- GRPO whitening changes prompt weighting and removes within-group constants.
- pass@k is oracle coverage, not selector accuracy.
- The tested coverage-collapse and curriculum-superiority claims were nulls.
- `16P/N` is FSDP persistent state, not peak live memory.
- Every ring rank sends and receives during all-gather and reduce-scatter.

## Final interview habit

For any post-training or systems question, state:

1. **Estimand:** what expectation or metric is actually being optimized?
2. **Support:** which outputs can the sampler, reference, or verifier represent?
3. **Credit/reduction:** per-token, per-sequence, sum, mean, centered, or whitened?
4. **Freeze line:** which quantities are rollout data and which are recomputed?
5. **Independent check:** what negative control or gold metric can falsify the story?
6. **Systems assumptions:** precision, optimizer, sequence length, batch, sharding,
   topology, and reserve.

> A good explanation does not just name the algorithm. It names the objective,
> estimator, support, failure detector, and systems cost.

## Primary repository sources

- `week-01-lm-foundations-pytorch/week-summary.md`
- `week-02-transformers-generation/week-summary.md`
- `week-03-rl-foundations/week-summary.md`
- `week-04-rlhf-ppo/week-summary.md`
- `week-05-dpo-preference-optimization/week-summary.md`
- `week-06-grpo-rlvr-reasoning/cheat-sheet.md`
- `week-08-training-inference-systems/code/gpu_memory.py`
- `week-08-training-inference-systems/code/finetuning_memory.py`
- `week-08-training-inference-systems/code/distributed_training.py`
