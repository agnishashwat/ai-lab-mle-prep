# Week 6 Cheat Sheet - GRPO, RLVR, Sampling, and Reasoning Curricula

> **One sentence:** Every `p` needs an event type. An RLVR system is correct only when its
> reward, optimizer, controller, selector, and evaluation each use the event they claim.

This sheet is the compact, end-to-end reference for Week 6. It covers the concepts, derivations,
measured results, implementation contracts, failure modes, and interview answers from all five
days.

## How to Use This Sheet

- **30-second review:** read the event ledger and the five day-level theses.
- **5-minute interview review:** read the system map, key equations, and interview cards.
- **Debugging:** jump to the decision tree and failure-mode matrix.
- **Coding:** use the implementation contracts and artifact commands.
- **Cold derivation practice:** cover the right side of each equation and reproduce it.

---

## 0. The Event Ledger - The Most Important Page

One model can have several different "success probabilities." Never write bare `p` in a design
doc, metric panel, or derivation.

| Name | Event | Used by | Must not be substituted for |
|---|---|---|---|
| `p_gold(x)` | `P(actual correctness | x)` | held-out evaluation, oracle pass@k, curriculum value | verifier reward rate |
| `p_obs(x)` | `P(verifier accepts | x)` | rollout reward, dead-group law, observed-rate sensor | actual correctness |
| `p_answer[a|x]` | mass on answer `a` | majority vote / self-consistency | sequence-level pass@k |
| `p_emit_gold(x)` | deployed selector emits a correct answer | production accuracy | oracle pass@k |
| `precision(x)` | `P(gold | accepted, x)` | filter limit, verifier monitoring | verifier "accuracy" |

The verifier identity is:

```text
alpha = P(reject | correct)          false-negative / incompleteness
gamma = P(accept | incorrect)        false-positive / unsoundness
J     = 1 - alpha - gamma            Youden's J

p_obs = (1-alpha)p_gold + gamma(1-p_gold)
      = gamma + J p_gold
```

When `J > 0`, the controller can invert the sensor:

```text
p_gold = (p_obs - gamma) / J
```

Interpretation:

- `J = 1`: honest verifier.
- `0 < J < 1`: informative but noisy verifier.
- `J = 0`: reward independent of truth; useful signal is zero.
- `J < 0`: anti-informative verifier; optimizing reward can reduce gold performance.

### When are the probabilities equal?

```text
p_obs = p_gold     at one p iff gamma(1-p) = alpha*p
p_obs = p_gold     for every policy iff alpha = gamma = 0
pass@k = filter@k  across policies when V=C, with the stated filter/fallback semantics
vote@k = pass@k    only under special answer-mass conditions, not generally
```

### The event-type test

Before using a probability, complete this sentence:

> "`p` is the probability of **[event]**, under **[distribution/policy]**, conditioned on
> **[prompt or selection event]**."

If that sentence is missing, the formula may be correct and the system still wrong.

---

## 1. The Whole System in One Diagram

```text
prompt x
   |
   v
current policy pi_theta(.|x)
   |
   | sample G completions from pi_old
   v
sequence ids y_1 ... y_G
   |
   | verifier only
   v
r_i = accept(x, y_i) in {0,1}
   |
   | group-relative baseline
   v
A_i = r_i - mean_j r_j
   |
   | A is detached
   v
policy loss  -mean_i A_i log pi_theta(y_i|x)
   +
fixed-reference k3 KL penalty
   |
   v
optimizer update

Separate paths:
  held-out gold ----------> evaluation panel and periodic verifier audit
  observed rollout rate --> EMA sensor --> calibrated scheduling coordinate
  answer_id -------------> vote / self-consistency evaluation
```

The load-bearing boundaries:

1. **Reward:** reads `accept`, never `gold`.
2. **Policy loss:** reads detached advantages and policy log-probabilities.
3. **KL:** is a separate loss term against fixed `pi_ref`.
4. **Controller:** may use a periodic held-out `alpha,gamma` audit, but only to select prompts.
5. **Evaluation:** may use gold, but gold cannot construct a production selector.
6. **Ablations:** use common random events indexed by `(step,prompt,draw)`.

---

## 2. Day 1 - GRPO vs PPO

### Thesis

> GRPO does not delete the value function. It replaces a learned estimate of `E[r|x]` with a
> Monte Carlo estimate from the same prompt group. Unlike a critic, that estimate vanishes
> exactly where the group is unanimous.

### PPO versus GRPO

| Property | PPO | GRPO |
|---|---|---|
| Baseline | learned critic `V(s)` | group mean reward |
| Credit level | usually token/state level | one sequence-level scalar broadcast to tokens |
| Extra trainable model | critic | none |
| Baseline generalization | across states/prompts | none; batch statistic only |
| Failure unique to method | critic bias/divergence | dead groups |
| Main resource cost | training memory | rollout inference compute |

### Memory bill for a 7B model

Assumptions: trainable bf16 + Adam = 16 bytes/parameter; frozen inference model = 2
bytes/parameter.

| Configuration | Policy | Critic | RM | Reference | Total |
|---|---:|---:|---:|---:|---:|
| PPO | 112 GB | 112 GB | 14 GB | 14 GB | **252 GB** |
| GRPO | 112 GB | - | 14 GB | 14 GB | **140 GB** |
| GRPO + cached ref logprobs | 112 GB | - | 14 GB | - | **126 GB** |
| GRPO + RLVR | 112 GB | - | - | - | **112 GB** |

GRPO saves **44.4%** versus PPO; RLVR can save **55.6%**. The cost moves from training memory to
rollout compute.

### Group-relative advantages

For group size `G`:

```text
centered:   A_i = r_i - r_bar

normalized: A_i = (r_i - r_bar) / std(r)
            return exactly zero for a dead group

leave-one-out:
            A_i = r_i - mean_{j != i}(r_j)
                = G/(G-1) * A_centered
```

With the estimator defined as the mean over the group:

```text
E[g_hat_centered] = (G-1)/G * grad J
```

Important consequences:

- The shrinkage is a deterministic scalar, not a directional bias.
- At `G=1`, centered and LOO advantages are exactly zero.
- LOO removes shrinkage by rescaling the same pathwise estimator.
- Therefore LOO cannot reduce absolute variance:

```text
Var(g_LOO) = [G/(G-1)]^2 Var(g_centered)
```

Measured shrinkage:

```text
G=2: 0.4994      G=4: 0.7496      G=8: 0.8755
cosine with exact gradient >= 0.999992
```

### Dead groups

For binary verifier reward with acceptance probability `p_obs`:

```text
P(dead) = p_obs^G + (1-p_obs)^G
```

A dead group is all-pass or all-fail. Every centered or normalized advantage is exactly zero.

At `G=8`:

| `p_obs` | Dead fraction |
|---:|---:|
| 0.50 | 0.78% |
| 0.70 | 5.77% |
| 0.90 | 43.05% |
| 0.95 | 66.34% |
| 0.99 | 92.27% |

At `p_obs=.95`, only **33.658%** of groups carry signal. Keeping 90% alive requires **G=45**.

The counterintuitive result:

> Improving verifier pass rate can reduce the fraction of rollout compute that participates in
> optimization.

### Centering versus whitening

For binary reward with empirical success fraction `p=k/G`:

```text
centered total |A|   = 2G p(1-p)          variance weighting
normalized total |A| = 2G sqrt(p(1-p))    standard-deviation weighting
```

Whitening compresses prompt weighting. It does not invert it, but it amplifies near-dead groups
relative to balanced groups.

At `G=8`, the centered balanced/extreme weight ratio is **2.2857**; normalized reduces it to
**1.5119**.

### Why KL belongs outside the reward

The group mean removes any within-group constant exactly:

```text
A(r_i + c(x)) = A(r_i)
```

If KL is folded into reward:

```text
r_i' = r_i - beta KL_i
```

only the within-group deviation of KL survives. Measured:

```text
full KL term:          0.6842
surviving deviation:  0.035726
deleted by centering:  19.15x
```

A uniform policy drift is invisible to the advantage. Therefore put KL directly in the
minimized loss.

### k1 versus k3

Let:

```text
l  = log pi_ref - log pi_theta
k1 = -l
k3 = exp(l) - 1 - l
```

Both estimate KL in expectation. The engineering distinction is feasibility:

- `k1` can be negative on individual samples.
- `k3 >= 0` pointwise from `log z <= z-1`.
- A negative term added to a minimized loss pays the policy to leave the reference.

Measured:

```text
k1 negative mass: 50.0%
k3 negative mass:  0.0%
```

Do not claim "k3 always has lower variance." The variance advantage reverses:

```text
Var(k1)/Var(k3) = 423x at KL .004
                 0.23x at KL about .65
```

### When PPO clipping is inert

With one inner epoch:

```text
pi_old = pi_theta at loss time
rho = pi_theta / pi_old = 1
```

so clipping cannot bind. Measured:

```text
mu=1: max binding fraction 0.000
mu=8: max binding fraction 0.2222
```

At `mu=1`, GRPO is REINFORCE + group baseline + explicit KL, written in PPO notation.

### Day 1 debugging checklist

- Is `dead_frac` rising with reward?
- Is KL outside the group-centered reward?
- Is `mu=1` while `clip_frac>0`? If yes, old logprobs or masking are wrong.
- Is advantage whitening silently changing the data weighting?
- Are rewards/advantages detached and logprobs still in the Torch graph?
- Are padded token positions masked? Padded ratios equal 1, not 0.

---

## 3. Day 2 - RL with Verifiable Rewards

### Thesis

> A verifier is a set, not an accuracy rate.

Let:

```text
C = set of actually correct responses
V = set of verifier-accepted responses
```

Then:

```text
V subset C    sound       no false positives
C subset V    complete    no false negatives
```

The two failures are asymmetric.

| Failure | Set difference | Main cost | Correct response |
|---|---|---|---|
| Incomplete | `C \ V` | lower `p_obs`, more rollout compute | expand verifier coverage |
| Unsound | `V \ C` | exploitable wrong reward support | remove false positives |

### Policy-weighted verifier rates

All rates are under the current policy, not uniform sequence counts:

```text
p_gold    = sum_y pi(y) gold(y)
p_obs     = sum_y pi(y) accept(y)
alpha     = P(reject | correct)
gamma     = P(accept | incorrect)
precision = P(correct | accepted)
recall    = 1 - alpha
```

The same checker changed under policy optimization:

```text
gamma:      0.1515 -> 0.7820
precision:  0.6720 -> 0.0926
```

"97% accurate" is incomplete without:

1. the distribution/policy used to measure it;
2. false-positive support;
3. false-negative rate;
4. task difficulty.

### Selection law and base-rate collapse

```text
p_obs = p_gold(1-alpha) + (1-p_gold)gamma

P(hack | rewarded)
  = gamma(1-p_gold)
    / [(1-alpha)p_gold + gamma(1-p_gold)]
```

At fixed `gamma=.01`, `alpha=.10`:

| `p_gold` | Hack among rewarded |
|---:|---:|
| .500 | 1.0989% |
| .100 | 9.0909% |
| .020 | 35.2518% |
| .005 | 68.8581% |

The verifier did not change. The base rate did.

Conditioning on an alive group does not change the per-accepted-sample hack probability. It
changes how hacks concentrate in the group's positive signal.

### Soundness theorem

If `V subset C`, then for every policy and every KL:

```text
p_obs <= p_gold
```

If `V \ C` is non-empty:

```text
sup_pi [p_obs - p_gold] = 1
```

One false positive and seven false positives have the same worst case: a point mass on a false
positive. Reward hacking is therefore a support property, not an average error-rate property.

### Why held-out gold is necessary

Measured training histories:

| Fixture | dead fraction | verifier pass | gold pass |
|---|---:|---:|---:|
| honest | .0361 | .4400 | .4400 |
| leaky / hacked | .7694 | .9681 | .0251 |
| starved | .7833 | .0319 | .1383 |
| saturated / solved | .8361 | .9803 | .9803 |

The hacked and solved runs are nearly identical on reward-derived columns. A one-winner group
also gives the winner the same advantage, `(G-1)/G`, whether the acceptance is genuine or a
hack.

> No reward-panel-only diagnostic can distinguish hacked success from genuine success.

### False-negative compute tax

At `p_gold=.10`, a 30% extraction failure gives:

```text
p_obs: .10 -> .07
G for 90% alive: 22 -> 32
compute increase: 45.5%
```

Adding false positives can reduce the required `G`, which is exactly why unsound shortcuts can
look computationally attractive.

### Multi-level rewards and Renyi entropy

For reward output probabilities `q_j`:

```text
P(dead) = sum_j q_j^G
        = exp(-(G-1) H_G(q))
```

Binary reward is the two-level special case. At `p=.05,G=8`:

```text
binary reward:                  66.3420% dead
split failure mass .95 -> .40+.55:
                                 0.9029% dead
reduction:                      73.5x
equivalent binary group size:  G=92
```

Rule:

> Split the dominant reward mode. Splitting the minority mode does almost nothing.

### Composite rewards self-anneal

Centering is linear:

```text
A(sum_j lambda_j r_j) = sum_j A(lambda_j r_j)
```

For a binary component with compliance `c`:

```text
E[sum_i |A_i|] = 2(G-1)c(1-c)
```

At `c=0` or `c=1`, influence is exactly zero for any coefficient, including a coefficient of
1000. Influence tracks inconsistency, not nominal weight.

Implication:

> A mastered reward component has left the gradient and cannot be weighted back into relevance.

### Day 2 debugging checklist

- Compute rates using policy mass, not sequence counts.
- Ask separately whether false-positive support is empty and what the false-negative rate is.
- Monitor held-out gold; reward-only panels cannot detect hacking.
- Run with runtime warnings as errors; NaNs often make diagnostics silently never fire.
- Do not alert on a component share decaying to zero; that can be mastery.

---

## 4. Day 3 - Sampling, pass@k, and Selection

### Thesis

> pass@k is capability composed with an oracle selector that deployment does not have.

### Exact pass@k

For one prompt with independent sample correctness probability `p`:

```text
pass@k = 1 - (1-p)^k
```

For heterogeneous prompts:

```text
benchmark pass@k = mean_x [1 - (1-p_x)^k]
```

Never use:

```text
1 - (1 - mean_x p_x)^k
```

because nonlinear transforms and averaging do not commute.

### Unbiased finite-sample estimator

Given `n` sampled completions, `c` correct, and `k<=n`:

```text
pass_hat@k
  = 1 - C(n-c,k)/C(n,k)
  = 1 - product_{i=0}^{k-1} (n-c-i)/(n-i)
```

Why unbiased: exchangeability. It is the probability that a uniformly chosen size-`k` subset
contains at least one of the observed correct samples.

Do not use the plug-in:

```text
1 - (1-c/n)^k
```

The function is concave, so Jensen makes the plug-in biased low.

Measured at `n=10,p=.1,k=10`:

```text
truth:    .651322
plug-in:  .493747
lost:     24.19%
```

Aggregation order on the same data:

```text
estimate then mean: .6250
mean then estimate: .9960
```

### pass@G and the dead-group law are the same algebra

```text
P(dead) = (1-pass@G) + p^G
```

The first term is all-fail; the second is all-pass. Dead fraction is symmetric under
`p <-> 1-p`, so it cannot tell whether prompts are too hard or already solved.

### The selector table

| Selector | Mechanism | Limit as `k -> infinity` |
|---|---|---|
| Oracle | emits a correct sample if any exists | 1 if correct support is non-empty |
| Filter | choose among verifier-accepted samples | verifier precision |
| Ranker / BoN | emit maximum score | top-score-set composition; can amplify hacks |
| Majority vote | emit modal answer | 1 if correct answer is modal, otherwise 0 |

Measured on one policy:

```text
pass@101 = 1.000000
majority:
k=9    .3563
k=101  .2764
```

More samples made the deployed vote worse while oracle pass@k stayed monotone by construction.

### Filter versus ranker

In the constructed false-positive model, the filter's accepted-output composition is:

```text
P(hack | accepted emission)
  = gamma(1-p)
    / [gamma(1-p) + (1-alpha)p]
```

It is independent of `n` because the filter emits uniformly from accepted samples. Under the
stated first-sample fallback, the comparable unconditional output risk is:

```text
P(filter outputs a hack)
  = [1-(1-p_obs)^n] * P(hack | accepted emission)
```

If every false positive receives the ranker's top score:

```text
per-draw hack mass       = (1-p)gamma
ranker hack probability = 1 - [1-(1-p)gamma]^n
```

This is pass@n for the hack event, whose policy mass is `(1-p)gamma`.

At `p=.10, alpha=.10, gamma=.01`:

| `n` | Filter hack output | Ranker hack output |
|---:|---:|---:|
| 1 | 0.90% | 0.90% |
| 8 | 5.14% | 6.98% |
| 64 | 9.08% | 43.93% |
| 256 | 9.09% | 90.12% |

The selectors agree at `n=1`. At `n=256`, where filter abstention is negligible, the selection
rule alone creates a **9.91x** gap.

### Best-of-n is a KL knob

For a continuous score with no ties, let `U=F_pi(score)`. Under the base policy,
`U~Uniform(0,1)`. The maximum of `n` draws has density `n u^(n-1)`, so:

```text
q(y)/pi(y) = n u^(n-1)

KL(BoN_n || pi)
  = E_q[log n + (n-1)log U]
  = log n - (n-1)/n
```

Values:

| `n` | KL |
|---:|---:|
| 2 | .1931 nats |
| 8 | 1.2044 nats |
| 64 | 3.1745 nats |
| 256 | 4.5491 nats |

Each doubling costs about `log 2`; measured 64 to 128 was **+.6853 nats**.

Discrete score ties make BoN less selective, so actual KL is lower. At `n=64`, a discrete score
gave **2.0507** versus **3.1745** for the continuous law.

### Temperature, truncation, and correlation

- Finite temperature `T>0` preserves support.
- Top-p truncation can set sequence probability exactly to zero.
- Importance sampling can correct a full-support temperature proposal in expectation.
- Importance sampling cannot recover support removed by top-p.
- Sequence-level importance ratios compound variance over tokens.

For one prompt, temperature ordering is independent of `k` because pass@k is monotone in `p`.
Across heterogeneous prompts, the aggregate optimum can move because changing `k` reweights
prompts.

Correlated samples violate the i.i.d. formula. The measured i.i.d. overstatement was unimodal:

```text
gap at k=2/4/8/16/64/256/1024:
.0706 / .2313 / .4176 / .4749 / .1921 / .0040 / .0000
```

### The metric reweights the benchmark

```text
d pass@k / dp = k(1-p)^(k-1)
```

At `k=1`, every prompt has equal marginal weight. As `k` rises, hard prompts dominate.

Measured hard/easy marginal-value ratio:

```text
k=1:   1x
k=8:   5.456e4x
k=16:  1.414e10x
```

Changing `k` changes which benchmark you are running.

### Day 3 debugging checklist

- Compute pass@k per prompt, then average.
- Name the selector used in deployment.
- Keep filter, ranker, vote, and oracle results in separate fields.
- Verify selector scores do not read gold.
- Check independence before using `1-(1-p)^k`.
- Treat top-p support removal as irreversible by importance weighting.
- Include BoN's KL and inference cost in deployment comparisons.

---

## 5. Day 4 - Reasoning Data Curriculum

### Thesis

> A curriculum is a controller. Its set point depends on the metric being reported, and the
> verifier constrains the legal domain.

### Deriving the set point

Suppose local learning progress scales as:

```text
Delta p proportional to p^a (1-p)^b
```

The marginal value of improving pass@k is:

```text
d pass@k/dp = k(1-p)^(k-1)
```

Ignoring constants independent of `p`:

```text
value(p) proportional to p^a (1-p)^(b+k-1)
```

Set the log derivative to zero:

```text
a/p - (b+k-1)/(1-p) = 0

p* = a / (a+b+k-1)
```

At `a=b=1`:

```text
p* = 1/(k+1)
```

`p*=1/2` iff:

```text
a = b+k-1
```

It is not equivalent to `k=1`.

Nominal set points:

| `k` | 1 | 4 | 8 | 16 | 64 |
|---:|---:|---:|---:|---:|---:|
| `p*` | .5000 | .2000 | .1111 | .0588 | .0154 |

Measured local exponents in the toy stack:

```text
(a,b) = (1.900, 1.911), R^2=.9979
k=8 set point: .1111 -> .1758
```

The `k-1` term is arithmetic. The progress exponents must be measured.

### Group size at the set point

To keep at least 90% of binary groups alive, solve:

```text
p^G + (1-p)^G <= .10
```

At `p*=1/(k+1)`, exact both-tail group sizes:

| `k` | 1 | 2 | 4 | 8 | 16 | 64 | 256 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `G*` | 5 | 6 | 11 | 20 | 38 | 149 | 591 |

Asymptotically:

```text
G* about 2.3026 k
```

With `G=8` anyway, dead fractions at the nominal set point:

```text
k=1/2/4/8/16/64/256
.78 / 3.92 / 16.78 / 38.97 / 61.57 / 88.34 / 96.93 percent
```

Reporting pass@64, targeting its nominal optimum, and using `G=8` means **88.34%** of groups are
supposed to be dead.

### The verifier is a multiplier

Using truth sign `s_tilde=2s-1`:

```text
U(p) = E[sum_i A_i s_tilde_i]
     = 2(G-1)p(1-p)J
```

Therefore:

```text
M_gold = E[sum_i |A_i^(gold)|] = 2(G-1)p_gold(1-p_gold)
U = J * M_gold
```

This is counterfactual **gold-reward** advantage mass. It is not the observed verifier-reward
advantage mass:

```text
M_obs = E[sum_i |A_i^(verifier)|] = 2(G-1)p_obs(1-p_obs)
```

The two masses generally differ. The useful signal still has:

```text
argmax_p U = 1/2 for every alpha,gamma with J>0
```

Verifier noise changes signal magnitude, not the unconstrained optimum.

At `alpha+gamma=1`:

```text
J=0 and U=0 for every p
```

At `alpha+gamma>1`, the useful signal is negative.

### Useful fraction / SNR

Observed advantage mass follows `p_obs`, while useful covariance follows `p_gold`:

```text
useful fraction
  = p_gold(1-p_gold)J
    / [p_obs(1-p_obs)]
```

For `alpha=.10,gamma=.01`, measured at `p_gold=.5/.1/.02/.005/.001`:

```text
.8973 / .8980 / .6454 / .3109 / .0825
```

If `gamma=0`, the ratio tends to 1 as `p_gold->0`, regardless of false-negative rate. Soundness
is exactly the property that prevents hard-task SNR collapse.

### Precision floor

Require:

```text
P(hack | rewarded) <= tau
```

Then:

```text
p_gold >= p_floor

p_floor = gamma(1-tau)
          / [tau(1-alpha) + gamma(1-tau)]
```

At `gamma=.01,alpha=.10,tau=.10`:

```text
p_floor = 1/11 = .090909
```

The crossover with `p*=1/(k+1)` is exactly:

```text
k=10
```

For `k>10`, the verifier floor binds before the metric optimum.

Value retained at the floor:

```text
k=8/10/16/32/64:
.9793 / 1.0000 / .8872 / .3804 / .0358
```

At pass@64, a 1% false-positive rate costs **96.4%** of the signal requested by the metric.

The two lower walls are different:

- Dead-group wall: purchasable with larger `G`.
- Precision floor: cannot be repaired with sampling compute.

Which wall binds depends on the chosen alive/dead tolerance. State the tolerance.

### Whitening is an implicit curriculum

For centered advantages with `c` successes:

```text
sum_i A_i^2 = c(G-c)/G
```

For normalized advantages, every alive group has:

```text
sum_i A_i^2 = G
```

Whitening equalizes alive groups and amplifies near-dead ones. The near-dead versus balanced
tilt is:

```text
G / [4(1-2^(1-G))] about G/4
```

Values:

```text
G=8:   2.0157x
G=64: 16.0000x
```

Any per-group rescaling changes prompt weighting and is therefore a curriculum.

### Control in log-odds

Use:

```text
logit(p) = log[p/(1-p)]
```

The 50%-of-peak band:

```text
k=1:  p in [.1464,.8536]
k=64: p in [.0036,.0407]
```

Width changes:

```text
p-space:      .70711 -> .03709     19.07x narrower
logit-space:  3.5255 -> 2.4654      1.43x narrower
```

A fixed `p +/- .05` tolerance is meaningless across `k`.

### Contamination

If fraction `c` is memorized and therefore scores 1:

```text
pass@k inflation = c [1 - pass@k(p_true)]
```

At `c=.05,p_true=.30`, inflation in percentage points:

```text
k=1/2/4/8/16/64:
3.500 / 2.450 / 1.201 / .288 / .017 / .000
```

The evaluation lie decays with `k`, but the training cost does not:

> A memorized prompt is a saturated, success-side dead group. It consumes rollout budget and
> returns zero gradient while looking perfect on reward.

Perturbed-twin method:

1. Create a matched variant that changes instance-specific surface details.
2. Measure `Delta = performance(original)-performance(twin)`.
3. Apply the same perturbation to a clean post-cutoff or privately authored control.
4. Suspect drop + clean control unchanged is evidence of memorization.
5. Both drop means the perturbation itself is harder.

No twin drop means no evidence of memorization, not proof of reasoning.

### Day 4 debugging checklist

- Is difficulty measured under the current policy or assigned statically?
- Is the target derived from the reported `k`?
- Were `(a,b)` measured or assumed?
- Is control performed in logit space?
- Which wall binds, under which dead-group tolerance?
- Is whitening changing the intended curriculum?
- Does the diagnostic threshold have air in the exact runtime regime?
- Does a change window exclude sensor burn-in?

---

## 6. Day 5 - The Composed Mini RLVR/GRPO Loop

### Thesis

> Calibration can repair the controller coordinate. It cannot repair an unsafe reward
> objective.

### Calibration contract

Given an observed-rate estimate and audited verifier rates:

```text
raw = (p_hat_obs - gamma) / J
scheduled = project raw into [0,1]
projected = mask identifying every changed element
```

Requirements:

- probabilities finite and in `[0,1]`;
- scalar inputs may broadcast;
- non-scalar shapes must match;
- reject `J<=0`;
- preserve the raw inverse;
- project only the scheduling value;
- record the projection rate.

Silent clipping destroys evidence that the sensor/model combination is inconsistent.

### Exact training-loop order

```text
pre-generate u[step,prompt,draw] for every prompt, active or inactive
create a separate RNG for prompt selection
freeze theta_ref = theta_0

for each step:
    1. evaluate and record the exact pre-update panel
    2. refresh audited alpha,gamma only on the declared interval
    3. convert prior observed-rate sensor into p_hat_gold
    4. select active prompts from that recorded prior sensor
    5. map each fixed uniform through the current policy CDF
    6. reward sampled sequence ids with accept only
    7. compute detached within-prompt group advantages
    8. minimize:
           -mean[A log pi_theta]
           + beta * mean[k3(pi_theta,pi_ref)]
    9. update the observed-rate EMA from completed rollouts
```

Why generate uniforms for inactive prompts?

- Not to cache or reuse rollouts.
- To index randomness by the event rather than control flow.
- If two controllers select different prompts, on-demand RNG would shift all future draws.
- Common random numbers isolate the controller intervention.

### Evaluation contract

For each prompt:

1. enumerate exact policy probabilities;
2. compute policy-weighted `p_obs,p_gold,alpha,gamma`;
3. compute oracle pass@k separately for each event;
4. evaluate filter, ranker, and vote independently;
5. average only after per-prompt nonlinear quantities are complete;
6. compute exact KL and entropy;
7. never let gold influence a selector score.

The fixed ranker score may use verifier acceptance and a deterministic tie-break. Gold is used
only to score the ranker's emitted distribution.

### Truth-boundary metamorphic test

To test the policy-loss dependency:

```text
mutate gold
hold theta, samples, RNG, accept, and data policy fixed
require final theta to be bit-identical
```

Non-vacuity control:

```text
mutate reachable accept values
require final theta to change
```

Scope condition:

- A calibrated curriculum intentionally uses gold-derived audit rates to choose data.
- Therefore freeze the data policy, or use raw sensor + uniform sampling, when testing the
  policy-loss boundary.

This is not the contamination test. One tests software dependency; the other tests data
memorization.

### Panel schema

| Column | Meaning |
|---|---|
| `p_obs` | exact verifier acceptance per prompt |
| `p_gold` | exact actual correctness per prompt |
| `p_hat_gold` | controller's scheduling estimate |
| `active` | prompt ids selected from prior sensor state |
| `verifier_passk` | oracle pass@k on verifier acceptance |
| `gold_passk` | oracle pass@k on actual correctness |
| `deployed_accuracy` | actual selected output accuracy |
| `precision` | policy-mass-weighted verifier precision |
| `dead_frac` | active groups with exactly zero advantage |
| `signal` | observed advantage magnitude |
| `kl` | exact sequence-level KL to fixed reference |
| `entropy` | exact sequence entropy |
| `in_band` | selected budget inside target difficulty band |
| `projection_frac` | fraction of calibrated estimates projected |

### Diagnostic statistics

All rules use tail levels or explicitly defined gains and fire only on strict threshold
exceedance.

| Rule | Statistic | Full calibrated threshold |
|---|---|---:|
| `dead_budget` | tail mean dead fraction | .584583 |
| `sensor_mismatch` | tail mean `abs(p_hat_gold-p_gold)` | .166659 |
| `reward_hacking` | verifier gain minus gold gain | .110000 |
| `selector_gap` | tail mean gold pass@k minus deployed accuracy | .150000 |
| `kl_runaway` | tail mean KL level | .430424 |

Every detail string reports:

1. tail window;
2. measured value;
3. threshold.

Why level, not trend?

> A policy can settle at dangerous KL. Requiring KL to keep increasing declares a high-KL
> plateau healthy.

### Full-regime capstone results

| Result | Measurement |
|---|---:|
| Harness | **75/75 checks** |
| Honest raw/calibrated final theta difference | **0.000000 exactly** |
| Starved sensor-error repair | mean **.076343**, range **.063569-.090097** |
| Leaky sensor-error repair | mean **.365958**, range **.311036-.401402** |
| Final-policy band gold pass@8 gain / 1,000 rollouts | **+.014309** |
| Final-policy uniform gold pass@8 gain / 1,000 rollouts | **+.010175** |
| Paired controller effect / 1,000 rollouts | **+.004134** |
| Paired spread / 1,000 rollouts | **.002114** |
| Leaky calibrated gold gain over 1,344 rollouts | **-.005075** |
| Final leaky verifier-gold gap | **.240452** |

The controller result is a **null**:

- mean effect is positive;
- the smallest paired effect is only `+.000263` per 1,000 rollouts;
- effect remains inside two paired standard deviations;
- report unresolved, not better.

Leaky step-0 panel:

```text
verifier pass@8: .999956
gold pass@8:     .764608
filter accuracy: .391510
ranker accuracy: .419953
vote accuracy:   .332450
precision:       .383829
```

One policy does not have one "pass probability."

---

## 7. Debugging Decision Tree

### A. Reward curve is flat

Start with free measurements from existing rollouts:

```text
1. dead_frac
2. p_obs
3. mean |A| on alive groups
4. gradient norm
```

Then use held-out gold:

```text
5. Cov(verifier reward, true success)
```

Interpretation:

| Observation | Diagnosis | Fix |
|---|---|---|
| high dead, `p_obs` near 0 | zero-reward cliff | scaffold, hints, easier data, or larger G |
| high dead, `p_obs` near 1 | saturated set | retire solved prompts / add harder data |
| low dead, healthy `|A|`, covariance near 0 | broken verifier | fix alpha/gamma; do not raise LR |
| low dead, useful covariance, gradient near 0 | loss/optimizer bug | inspect detach/no-grad/LR/update |

`dead_frac` alone cannot distinguish cliff from saturation because it is symmetric under
`p <-> 1-p`.

### B. Verifier improves while gold degrades

Likely unsound reward / reward hacking.

Check:

- policy-weighted false-positive support;
- verifier precision under current policy;
- held-out gold trajectory;
- whether calibration was mistaken for objective repair.

Fix the verifier. More sampling or a larger learning rate cannot turn false positives into true
positives.

### C. Oracle pass@k is high but production accuracy is low

Likely selector mismatch.

Check:

- exact production selector;
- filter/ranker/vote outputs against held-out gold;
- verifier precision;
- answer-level mass;
- inference-time BoN KL.

Do not use `dead_frac`; it is a training-signal metric.

### D. KL is high but no longer rising

Still `kl_runaway`. Read the tail level, not the slope.

Check:

- fixed reference really stays fixed;
- KL is outside the group-centered reward;
- k3 is reduced with the token mask;
- beta controller is acting on a feasible non-negative estimate.

### E. Curriculum appears ineffective

Check:

- paired seeds and common random numbers;
- per-seed paired differences;
- effect relative to paired spread;
- current-policy sensor rather than static difficulty;
- correct event (`p_gold` target versus `p_obs` sensor);
- target derived from reported `k`;
- whether a wall makes the nominal set point infeasible.

Accept a null when the effect remains inside seed spread.

---

## 8. Failure-Mode Matrix

| Symptom | Mechanism | Cheapest discriminant | Correct fix | Wrong fix |
|---|---|---|---|---|
| many all-fail groups | prompts too hard | `dead_frac` + low `p_obs` | scaffold or increase G | harder data |
| many all-pass groups | prompts solved | `dead_frac` + high `p_obs` | retire / harder data | more of same |
| reward rises, gold falls | false-positive support | held-out gold | repair verifier | calibrate sensor |
| `p_obs<p_gold` | verifier incomplete | audited alpha | expand coverage / pay G | add false positives |
| alive reward, no learning | reward uncorrelated with truth | held-out covariance | repair verifier | raise LR |
| useful reward, no update | graph/optimizer bug | gradient norm | fix detach/no-grad/LR | change curriculum |
| pass@k high, emitted low | selector mismatch | selector-vs-gold accuracy | train/change selector | quote pass@k |
| KL high | reference drift | tail KL level | explicit KL / stop | trend gate |
| static curriculum decays | stale difficulty labels | current per-prompt sensor | close loop | re-sort once |
| twin drop on both sets | perturbation harder | clean control | redesign perturbation | call contamination |
| tiny positive ablation mean | seed noise | paired effect/spread | report null / add power | claim win |

---

## 9. Interview Cards

### Q1. GRPO versus PPO: what changes and why?

GRPO changes the baseline. PPO learns a critic; GRPO subtracts the mean reward of `G` samples
from the same prompt. That saves memory - **252 GB to 140 GB** in the 7B accounting - but loses
token-level credit assignment and introduces dead groups:
`P(dead)=p_obs^G+(1-p_obs)^G`. At `p_obs=.95,G=8`, **66.34%** of groups return exactly zero
advantage. Do not claim GRPO always reduces variance; that depends on whether the group statistic
predicts return better than the critic.

### Q2. Your verifier is 97% accurate. Is it good enough?

97% under which policy, with which false-positive support? Verifier rates are policy-weighted and
RL changes the policy; measured precision fell **.672 to .093** under a tilt toward accepted
errors. Ask whether `V\C` is empty - a support question - and separately measure incompleteness,
which is a compute tax. Train with a sound verifier and retain a held-out gold panel because a
hacked and solved run can be identical on every reward-derived metric.

### Q3. The model has 85% pass@64 and 30% pass@1. What do you know?

At benchmark level, these are `mean_x f(p_x)` values and can coexist under heterogeneous prompt
difficulty. Pass@64 says a correct sample is often reachable within 64 draws **under an oracle
selector**. It does not say the deployed filter, ranker, or vote can identify it. Measure the
actual emitted output against held-out gold.

### Q4. BoN-64 beat greedy according to the reward model. Ship it?

No. The same model selected and judged the output, so the comparison is circular. Evaluate
greedy and the BoN-emitted output on independent gold/human judgments at matched inference cost.
Also account for the distribution move:
`KL(BoN_64||pi)=log 64-63/64=3.1745` nats without ties. A ranker can amplify false positives even
when a filter using the same verifier does not.

### Q5. How would you design the RLVR curriculum?

Treat it as a controller. Estimate current per-prompt success from rollouts, target
`p*=a/(a+b+k-1)` - `1/(k+1)` under the simple `(1,1)` law - and control in logit space. Then
apply two lower constraints: enough `G` to keep groups alive and the verifier precision floor.
At pass@64, nominal `p*=1/65` with `G=8` is **88.34% dead**; with a 1% FP verifier, the precision
floor retains only **3.58%** of nominal value.

### Q6. Reward is flat for 2,000 steps. Diagnose it.

Start with `dead_frac` and `p_obs`. High dead + low `p_obs` is a cliff; high dead + high `p_obs`
is saturation, and their fixes are opposite. If groups are alive, check `|A|` and gradient norm.
Only then pay for held-out covariance between verifier reward and truth: healthy advantage mass
with zero covariance is a broken verifier; useful covariance with zero gradient is optimizer
plumbing.

### Q7. How do you validate no held-out gold leaks into training?

With data selection fixed, mutate gold while holding theta, samples, RNG, and verifier acceptance
fixed; require a bit-identical update. Then mutate reachable acceptance labels and require the
update to change. Separately, test benchmark contamination with perturbed twins and a clean
post-cutoff/private control. The first test is software dependency; the second is memorization.

### Q8. Why can calibration not repair an unsound verifier?

Calibration maps an observed acceptance rate into an estimate of gold difficulty for scheduling.
It changes which prompts are selected, not which sampled responses receive reward. In the leaky
arm it reduced sensor error by **.3660**, while gold fell **.0050** and the verifier-gold gap
remained **.2404**. It steered more accurately with respect to an unsafe objective.

---

## 10. Numbers Worth Memorizing

### Day 1

```text
memory: PPO 252 GB -> GRPO 140 GB -> RLVR 112 GB
shrinkage: (G-1)/G
dead at G=8, p=.95: 66.34%
G for 90% alive at p=.95: 45
KL-in-reward deletion: 19.15x
clip binding: mu=1 -> 0.000; mu=8 -> .2222
```

### Day 2

```text
soundness: p_obs <= p_gold for every policy
unsound support: sup(p_obs-p_gold)=1 with one false positive
precision under tilt: .6720 -> .0926
hack share at p=.005, gamma=.01: 68.86%
split dominant reward mode: dead 66.34% -> .90%, 73.5x
false-negative compute tax: G 22 -> 32, +45.5%
```

### Day 3

```text
plug-in estimator loses 24.19% at n=10,p=.1,k=10
filter vs ranker hack at n=256: 9.09% vs 90.12%, 9.91x
BoN-64 KL: 3.1745 nats
majority: .3563 at k=9 -> .2764 at k=101 while pass@101=1
correlation gap peak: .4749 at k=16
```

### Day 4

```text
p*: 1/(k+1)
G* for 90% alive: 5/6/11/20/38/149 at k=1/2/4/8/16/64
dead at pass@64 set point with G=8: 88.34%
p_floor: 1/11, crossover k=10
value retained at k=64 floor: .0358
whitening tilt at G=64: 16x
band narrowing: 19.07x in p, 1.43x in logit-p
```

### Day 5

```text
full harness: 75/75
honest raw/calibrated difference: exactly 0
leaky sensor repair: .365958
leaky gold gain over 1,344 rollouts: -.005075
final verifier-gold gap: .240452
controller effect/spread per 1,000 rollouts: +.004134 / .002114 -> null
```

---

## 11. Artifact Map and Commands

| Day | File | Checks | Main reusable objects |
|---|---|---:|---|
| 1 | `code/grpo.py` | 96 | sequence table, advantages, exact KL/entropy, k3 |
| 2 | `code/rlvr.py` | 112 | verifier rates, observed-rate law, dead-level law |
| 3 | `code/sampling.py` | 73 | pass@k estimator, selector curves, BoN distribution |
| 4 | `code/curriculum.py` | 106 | set point, precision floor, band/uniform samplers |
| 5 | `code/mini_rlvr.py` | 75 | integrated task, evaluator, trainer, diagnostics |
| | **Total** | **462** | |

From the repository root:

```bash
./.venv/bin/python -W error::RuntimeWarning \
  week-06-grpo-rlvr-reasoning/code/grpo.py

./.venv/bin/python -W error::RuntimeWarning \
  week-06-grpo-rlvr-reasoning/code/rlvr.py

./.venv/bin/python -W error::RuntimeWarning \
  week-06-grpo-rlvr-reasoning/code/sampling.py

./.venv/bin/python -W error::RuntimeWarning \
  week-06-grpo-rlvr-reasoning/code/curriculum.py

./.venv/bin/python -W error::RuntimeWarning \
  week-06-grpo-rlvr-reasoning/code/mini_rlvr.py
```

List or run individual stages:

```bash
./.venv/bin/python week-06-grpo-rlvr-reasoning/code/mini_rlvr.py --list
./.venv/bin/python week-06-grpo-rlvr-reasoning/code/mini_rlvr.py truth_boundary
./.venv/bin/python week-06-grpo-rlvr-reasoning/code/mini_rlvr.py --fast
```

---

## 12. Derivations to Reproduce Cold

1. `E[g_centered] = (G-1)/G grad J`.
2. `P(dead)=p^G+(1-p)^G`.
3. `P(dead)=sum_j q_j^G=exp(-(G-1)H_G(q))`.
4. `p_obs=gamma+(1-alpha-gamma)p_gold`.
5. `P(hack|rewarded)=gamma(1-p)/[(1-alpha)p+gamma(1-p)]`.
6. Unbiased finite-sample pass@k estimator.
7. `d pass@k/dp=k(1-p)^(k-1)`.
8. `KL(BoN_n||pi)=log n-(n-1)/n`.
9. `p*=a/(a+b+k-1)`.
10. `p_floor=gamma(1-tau)/[tau(1-alpha)+gamma(1-tau)]`.
11. `U=2(G-1)p(1-p)J`.
12. Why truth mutation must not change the fixed-data-policy update.

If these twelve can be derived and every `p` can be named, the week is interview-ready.
