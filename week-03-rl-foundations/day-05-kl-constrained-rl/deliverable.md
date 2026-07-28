# Deliverable - Week 03 Day 05: KL-Constrained RL & the Reference Policy

## What I Built / Derived

Derived the KL-regularized objective's closed-form optimum from the Lagrangian, then built
`code/kl_rl.py` (**94/94 staged checks, ~16 s**; `--fast` ~6 s). Unlike Days 2-4, most of
today has a closed form, so "ground truth" stopped being a finite-difference approximation
and became algebra: the harness checks the closed form against a **black-box numerical
ascent on my own objective**, which is the Day-2 ground-truth trick moved up one level
(there the reference was a gradient, here it is an argmax).

Functions: `kl_divergence` (support conventions), `tilted_policy` (the closed form, in log
space), `regularized_objective`, `forward_kl_objective`, `frontier`, `overoptimization`,
`kl_estimators` (k1/k2/k3), `shaped_rewards` (the InstructGPT per-token mechanism), and
`train_kl` (REINFORCE-with-baseline on KL-shaped rewards).

## The Core Result

\[
\pi^*(a|s) \;=\; \frac{1}{Z(s)}\,\pi_{\text{ref}}(a|s)\,e^{r(s,a)/\beta},
\qquad Z(s) = \sum_a \pi_{\text{ref}}(a|s)\,e^{r(s,a)/\beta},
\qquad J^* = \beta \log Z(s)
\]

The optimal policy is an **exponential tilt of the reference**, and the optimal value is a
**log-sum-exp** -- a soft maximum of `r` at temperature `beta`, taken with `pi_ref` as the
base measure. It interpolates `max_a r` (as `beta -> 0`) and `E_ref[r]` (as `beta -> inf`).
That is the same operator as Week 1's `log_softmax`, and it is the "soft value function" of
MaxEnt RL / soft Q-learning -- which is why that literature and RLHF are the same math.

## Three exact identities, all measured

```
J*(beta)        = beta log Z(beta)                        checked at every beta
dJ*/dbeta       = -KL(pi*||pi_ref)                        envelope theorem, rtol 1e-4
dE[r]/dKL       = beta                                    rtol 1e-4 at beta = 0.3, 1.0, 4.0
```

The third one is the interview line: **`beta` is not a regularization strength, it is the
exchange rate between reward and KL** -- the price in nats you are willing to pay for a unit
of reward. It also makes the frontier concave in KL (the slope shrinks monotonically as KL
grows), so the first nats of drift buy far more reward than the last.

### The frontier (12-action bandit, `E_ref[r] = 0.0270`, `max_a r = 1.3665`)

```
   beta      E[r]        KL
   0.05    1.3620    3.8376     <- essentially greedy
   0.20    1.1218    1.6377
   0.80    0.6057    0.3302
   3.20    0.1986    0.0265
  12.80    0.0710    0.0017
 204.80    0.0298    0.0000     <- essentially the reference
```

## Result 2: reverse vs forward KL is the entire design

RLHF penalizes **reverse** KL, `KL(pi_theta || pi_ref)`. Measured on a bimodal reference
whose two modes are equally weighted (0.5 each) and where the reward likes one of them:

```
mass on the LOW-reward mode:  reference 0.5000 | forward-KL optimum 0.0950 | reverse 0.0110
```

Reverse KL is **mode-seeking**: dropping a mode costs only the mass you moved, so it drops
it (8.6x more aggressively than forward). Forward KL is **mass-covering**: it blows up as
`pi -> 0` anywhere `pi_ref` has mass, so every action the reference uses keeps a real share
(min mass 0.025).

And the consequence that actually matters, on a reference with a hole (`pi_ref[3] = 0` while
`r[3] = 2.5` is the best-rewarded action of all):

```
mass on the action the reference NEVER emits:  reverse 0.0000 | forward 0.7750
```

**Reverse KL enforces `supp(pi_theta) subset supp(pi_ref)` at every beta, including in the
`beta -> 0` limit.** Asserted at `beta = 2, 0.5, 0.01`: `pi*[3] == 0.0` exactly, not
approximately. Two ways to see it: the tilt *multiplies* `pi_ref`, so a zero survives any
finite exponential; and `KL(pi||pi_ref) = inf` for any `pi` that puts mass off the
reference's support, so those policies are infinitely penalized regardless of reward.

Therefore:

- **RLHF cannot teach a base model anything it cannot already produce.** It reweights
  behaviours already in the reference's support. This is the precise version of "RLHF is
  elicitation, not new capability", and it is why a weak base model cannot be fixed with
  more RL.
- **SFT must precede RL**: SFT *moves* the support, RL reweights *within* it.
- Same fact, different place in the pipeline: **top-p sampling truncates support**, so
  top-p rollouts are missing tokens that no importance ratio and no PPO clipping can
  recover. Temperature is a reweighting (repairable); nucleus is a support change (not).

## Result 3: k1/k2/k3, and the crossover nobody mentions

You cannot sum KL over sequences (`|V|^T` of them), so it is estimated from samples of
`log_ratio = log pi(a) - log pi_ref(a)`, `a ~ pi`. With `rho = pi_ref/pi` (so
`E_pi[rho] = 1` exactly):

```
k1 = log_ratio              unbiased, high variance, goes NEGATIVE on ~1/3 of samples
k2 = 0.5 log_ratio^2        always >= 0, BIASED
k3 = rho - 1 - log rho      unbiased AND >= 0 pointwise (control variate on E[rho]=1)
```

Measured across four KL regimes, 200k samples each, against the exact tabular KL:

```
 beta   exact KL   bias k1   bias k2   bias k3    var k1    var k3   ratio  neg(k1)
  8.0    0.00436  +0.00012  -0.00002  +0.00000    0.0087    0.0000  423.48    43.6%
  2.0    0.06531  -0.00005  -0.00166  -0.00020    0.1230    0.0062   19.93    32.3%
  1.0    0.22871  +0.00051  -0.00895  -0.00097    0.3870    0.1282    3.02    20.0%
  0.5    0.65363  -0.00129  -0.01356  +0.00391    0.8546    3.7677    0.23    34.1%
```

Three things worth saying out loud:

1. **k3's variance advantage is a small-KL phenomenon and it REVERSES.** 423x better at
   `KL = 0.004`, 20x at `KL = 0.065`, 3x at `KL = 0.23`, and **4.4x WORSE at `KL = 0.65`**.
   `rho` is a likelihood ratio, and ratios blow up once the distributions genuinely
   separate. k3 is the right default precisely because production RLHF runs at small
   per-token KL -- it is not a free lunch, it is a well-matched approximation.
2. **k2's bias is invisible in the regime you run in** (0.1 se at `KL = 0.004`) and becomes
   obvious only once the KL is real (6.4 se at `KL = 0.23`). That is why a biased estimator
   survives in the wild: nothing complains until the run is already drifting.
3. **k1 reads negative on 20-44% of samples.** A logged "KL" below zero is not a bug, it is
   k1 doing its job. This is the single most common false alarm in an RLHF dashboard.

## Result 4: over-optimization is Goodhart, not an RL artefact

Optimized against a proxy reward that agrees with the truth everywhere except one "hacked"
action (proxy 0.4 above the true best, true reward the worst of all, and the reference gives
it almost no mass so only aggressive optimization finds it):

```
   beta      KL     proxy      true
   0.02   9.941    1.7665   -2.3250     <- fully hacked, WORSE than doing nothing
   0.08   4.299    1.4290    0.4434
   0.16   1.971    1.1829    1.1286     <- best true reward
   0.64   0.465    0.7019    0.7003
   2.56   0.041    0.2394    0.2391
  81.92   0.000    0.0341    0.0339
  reference: KL 0.000, proxy 0.0272, true 0.0270
```

The proxy rises **monotonically** as beta falls -- that is what optimization means, and it is
why "our reward-model score went up" is evidence of nothing. The true reward **peaks at an
interior beta (0.16, at 1.1286) and then collapses to -2.3250**, i.e. *below the reference's
own 0.0270*. Over-optimizing is worse than not doing RL at all.

**Negative control, which is what makes the result attributable:** tilting by the *true*
reward instead is monotone with no hump. The hump is caused by the proxy being
mis-specified, not by the tilt, not by KL regularization, and not by RL.

## Result 5: KL in the reward == the sequence KL (the InstructGPT mechanism)

```
r_t  <-  r_t - beta * ( log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t) )
```

The sequence KL is exactly the trajectory-distribution KL, because both policies share
`rho_0` and `P`, which cancel term by term (the same cancellation that made the policy
gradient model-free on Day 2):

\[
\mathrm{KL}\big(p_\theta(\tau)\|p_{\text{ref}}(\tau)\big)
= \mathbb{E}_{\tau\sim p_\theta}\Big[\sum_t \log\tfrac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\Big]
= \sum_t \mathbb{E}_{d_t}\big[\mathrm{KL}_t\big]
\]

Verified against the exact state distribution (`state_distributions`, forward recursion
`d_{t+1} = d_t P_pi`), 2000 episodes, `beta = 0.3`:

```
k1: mean penalty 0.77787 (se 0.01030, var 0.21200)   |  beta * exact seq KL = 0.78974
k3: mean penalty 0.79774 (se 0.00850, var 0.14446)   |  same expectation, 1.47x less variance
```

So **reward shaping and a sequence-level KL constraint are the same object**, one sampled and
one exact. The reason to shape rather than add a loss term is density: the RM reward is one
scalar at EOS, but the KL penalty exists at every token, so shaping hands the critic a dense
signal -- Day 4's credit-assignment problem attacked from the reward side.

And the sign asymmetry k3 fixes: **the k1 penalty pays a BONUS on 24.5% of tokens** (any
token the policy made *less* likely than the reference has a negative log-ratio). k3 is `>= 0`
on every token with the same expectation and 1.47x lower variance. Strictly better as a
reward.

## Result 6: the frontier, by actual RL

`train_kl` on the Day-1 two-state chain, starting *at* a deliberately mediocre reference
(prefers the wrong action in `s0`; its own true return is **2.109**):

```
 beta   true return   sequence KL to ref
  0.0         7.928               5.0556     <- == Day 3/Day 4 optimum, "reach & stay"
  0.2         7.875               4.8343
  1.0         5.897               1.7582
  5.0         2.966               0.0721     <- policy stays within 0.067 of pi_ref
```

Monotone in both columns, `beta = 0` reproduces Day 3/Day 4 exactly (a free regression test
chaining back three days), and every beta beats the reference: **a leash is not a muzzle.**

## Core Takeaway

Post-training is not reward maximization, it is **constrained** reward maximization, and the
constraint is doing three jobs at once that are easy to conflate:

1. **Epistemic** -- it encodes where you trust your reward model. The hump at beta = 0.16 is
   the boundary of the RM's validity, not a property of the RL algorithm.
2. **Structural** -- reverse KL confines you to the reference's support, which caps what RL
   can do at "elicit what is already there".
3. **Numerical** -- the per-token KL term is the only dense signal in an otherwise
   sparse-terminal-reward problem.

And `beta` prices all three at once, as an exchange rate in reward-per-nat.

## Key Equations

```
objective     J(pi) = E_{a~pi}[r(s,a)] - beta KL(pi(.|s) || pi_ref(.|s))

optimum       pi*(a|s) = pi_ref(a|s) exp(r(s,a)/beta) / Z(s)
              Z(s) = sum_a pi_ref(a|s) exp(r(s,a)/beta)

value         J* = beta log Z(s)          (log-sum-exp = soft max at temperature beta)

limits        beta -> inf : pi* -> pi_ref            (NOT uniform -- uniform is the
                                                      entropy-bonus special case)
              beta -> 0   : pi* -> uniform over argmax{ r(s,a) : a in supp(pi_ref) }

support       supp(pi*) subset supp(pi_ref) at every beta > 0

identities    dJ*/dbeta = -KL(pi*||pi_ref)           (envelope theorem)
              dE[r]/dKL = beta                       (=> frontier concave in KL)

implicit r    r(s,a) = beta log( pi*(a|s) / pi_ref(a|s) ) + beta log Z(s)
              pairwise: the beta log Z(s) term CANCELS  ==> DPO

entropy       KL(pi || Uniform) = log A - H(pi)
              => an entropy bonus IS a KL penalty, to the uniform reference

estimators    k1 = lr,  k2 = 0.5 lr^2,  k3 = expm1(-lr) + lr,   lr = log(pi/pi_ref)

shaping       r_t <- r_t - beta * k_i(lr_t);  E[sum_t penalty] = beta * KL(p_theta||p_ref)

sequence KL   KL(p_theta(tau) || p_ref(tau)) = sum_t E_{d_t}[ KL(pi_theta(.|s)||pi_ref(.|s)) ]
              (rho_0 and P cancel -- same reason the policy gradient is model-free)
```

## Bridge to Week 5 (DPO), planted here

Rearranging the closed form isolates the reward:

```
r(s,a) = beta log( pi*(a|s) / pi_ref(a|s) ) + beta log Z(s)
```

`beta log Z(s)` is intractable -- **not** because logits are unavailable (for one token you
have all `|V|` of them) but because in RLHF the "action" is an entire completion, so
`Z(x) = sum_y pi_ref(y|x) exp(r(x,y)/beta)` sums over `|V|^T` sequences. It is combinatorial,
not sparse.

But `Z` depends on the **prompt only**. Any two completions of the same prompt share it
exactly, so a difference kills it:

```
r(x,y1) - r(x,y2) = beta log( pi*(y1|x)/pi_ref(y1|x) ) - beta log( pi*(y2|x)/pi_ref(y2|x) )
```

That is DPO's implicit reward. It converts "maximize `r` subject to a KL constraint" -- which
needs rollouts, a reward model, and an intractable partition function -- into a supervised
classification loss on preference pairs. Week 5 adds Bradley-Terry on top; the hard part is
above.

## Implementation Notes

- **`Z` must be computed in log space.** `pi_ref * np.exp(r / beta)` overflows to `inf` at
  `beta = 1e-3` with rewards of order 1, and normalizes to `nan`. `log pi_ref + r/beta`
  through a max-subtracting `log_softmax` is finite at `beta = 1e-3` *and* at rewards of
  magnitude 500. Week 1's logsumexp lesson, in a new costume.
- **`-inf` is how a forbidden action stays forbidden.** `log_softmax` tolerates `-inf`
  entries and maps them to exactly `0.0`, which is what makes the support property hold
  *exactly* rather than to `1e-300`.
- **The tilt is invariant to `r -> r + c`** (the constant is absorbed by `Z`). A reward model
  is only identified up to a per-prompt constant, which is precisely why a per-prompt
  baseline -- GRPO's group mean -- is free.
- **`np.expm1` for k3.** At `|log_ratio| ~ 1e-12`, `np.exp(-lr) - 1` loses every significant
  digit to cancellation and can go negative, destroying the one property k3 exists for.
- **Keep the numerical reference structurally independent.** `solve_by_ascent` uses central
  finite differences on the objective as a *black box* -- it knows nothing about exponential
  tilts. Agreement to `5e-4` therefore means something.

## Failure Modes / Debugging Notes

- **`np.where` is not lazy, and guarding the RESULT is not guarding the ARGUMENT.**
  `np.where(p > 0, p * np.log(p/q), 0.0)` still computes `p/q` for every element, emits
  `RuntimeWarning`s, and leaves correctness resting on the bad slot and the masked slot
  coinciding. Guard inside the log: `np.log(np.where(p > 0, p, 1.0))`.
- **Guarding the log then BREAKS the `+inf` convention, so the convention must be asserted.**
  With `q` sanitized, a support violation returns a finite wrong number instead of `inf` --
  the most dangerous possible outcome, since "reverse KL is infinite off the support" is the
  day's structural result. Detect it: `np.any((p > 0) & (q == 0), axis=-1) -> np.inf`.
- **`np.where(cond, a, b)` returns `a` where `cond` is TRUE.** Writing the *violation* as the
  condition and the good value as the true-branch inverts it and stamps `inf` on every
  healthy row. Every KL in the file read `inf`.
- **A latent `nan` on a path no test covered.** Leaving `log_q` unguarded gives the right
  answer when `p > 0, q == 0` (by IEEE accident) and `0 * inf = nan` when *both* are zero.
  No fixture had `p` and `q` vanishing at the same index, so 94/94 passed -- yet that
  configuration is the *normal* case in RLHF, since every tilted policy inherits the
  reference's zeros, making `KL(pi*_b1 || pi*_b2)` `nan`. **Green tests != correct code;
  check the paths the tests don't cover** (Day 4's lesson, arriving on schedule).
- **`kl_history` measured the wrong quantity and every check still passed.** Logging the
  *unweighted* mean of the per-state KL instead of `sum_t E_{d_t}[KL_t]` gave 1.053 where the
  truth is 5.056 -- because after one step the policy is in state 1 with probability 0.994,
  so weighting matters. All four assertions were about *monotonicity in beta*, which a
  wrong-by-a-factor quantity preserves. Same signature as Day 4's positional-arg bug: **the
  metric was wrong in a direction no test could see.** The available tell: a provided helper
  (`state_distributions`) that the code never called. *An unused fixture in a file that told
  you to use it is a bug report.*
- Also appended a `(2,)` array instead of a float -- Day 4's `history.append(...)` bug, one
  container over, and it survived because `np.mean` flattens a list of arrays into a number.
- `theta_ref.clone().requires_grad_(True)` wants a `.detach()` in the middle: if `theta_ref`
  ever carries a graph, `requires_grad_` on a non-leaf raises.

## Interview Answers

### "Why does RLHF constrain the policy to a reference at all?"

Because the reward is a proxy and the policy is a generator, and each of those facts breaks
in a different way.

The reward model is a network trained on finite preference data. It is accurate near the SFT
distribution and arbitrary far from it, and an unconstrained argmax will find the arbitrary
part -- that is what argmax does. So the KL term is really a statement about *where I trust
my reward model*, expressed as a leash. Separately, `pi_ref` is the only thing in the loop
that knows how to write English; the reward model scores, it does not generate. Drift far
enough and you lose fluency the RM was never asked to preserve.

The direction matters. RLHF penalizes **reverse** KL, `KL(pi_theta||pi_ref)`, which is
mode-seeking: dropping a mode of the reference costs only the mass you moved, so the policy
happily sharpens onto the high-reward mode. I measured that on a bimodal reference -- reverse
KL cut the low-reward mode from 0.50 to 0.011, about 9x more aggressively than the forward
direction, which is mass-covering and cannot drop a mode at all because it diverges as
`pi -> 0` anywhere the reference has mass. For alignment you *want* mode-seeking: the point is
to commit to preferred behaviour, not to average over everything the base model does.

The structural consequence is the part I'd lead with. Reverse KL is infinite wherever the
policy puts mass and the reference puts none, so `supp(pi_theta) subset supp(pi_ref)` at
every beta -- I asserted `pi*[a] == 0.0` exactly at beta = 2, 0.5, and 0.01 for an action with
the highest reward in the problem and zero reference mass. **So RLHF cannot add capability, it
can only reweight what the base model can already produce.** That is why SFT has to come
first -- SFT moves the support, RL reweights within it -- and it is also why a nucleus-sampled
rollout is unrepairable: top-p truncates support, and no importance ratio recovers mass you
never sampled.

For instrumentation I log the sequence KL `sum_t E_{d_t}[KL(pi_theta||pi_ref)]` and policy
entropy every step, never reward alone -- reward cannot distinguish learning from collapsing.
I'd watch KL against a target band and either early-stop or adapt beta, because the useful
question is never "is reward up" but "how many nats did that reward cost".

### "Reward-model score is up 40%, human eval win-rate is down. What happened?"

That is textbook reward over-optimization -- Goodhart on a learned proxy. The RM was trained
on completions sampled near the SFT model, so it is only calibrated there; optimizing hard
pushes the policy off that manifold, into a region where the RM's ranking is extrapolation.
The policy is not failing to optimize, it is succeeding at optimizing the wrong thing.

The shape is what makes this diagnosable. I measured it on a bandit with a mis-specified
proxy: as beta falls, the proxy score rises **monotonically** -- 0.03 at the reference up to
1.77 fully optimized -- while the true reward rises, **peaks at beta = 0.16 with 1.1286, and
then collapses to -2.3250**, which is *below the reference's own 0.0270*. So an
over-optimized policy is worse than not doing RL at all, and the proxy number reports success
the whole way down. The negative control matters too: tilting by the *true* reward is
monotone with no hump, so the hump is the proxy's mis-specification, not RL's fault and not
the KL penalty's.

Concretely I'd raise beta, but not by guessing -- the frontier is parameterized by beta and
`dE[r]/dKL = beta` exactly, so beta is the price in reward-per-nat. I'd take checkpoints along
a KL ladder from the existing run, score each on the held-out *human* (or gold-verifier)
eval, and pick the KL where that curve turns over; then set beta to land there, or just early
-stop at that KL. And I'd have caught it earlier by never trusting RM score as the stopping
criterion: gate on a small gold eval at fixed KL intervals, and watch entropy and KL for the
collapse signature. The general form is Gao et al.'s over-optimization scaling law -- true
reward as a concave function of `sqrt(KL)` that eventually bends down.

### "The reward arrives only at EOS. How do you assign credit to tokens?"

There are two mechanisms and production RLHF uses both.

Estimator side: a value head plus GAE. Train `V_phi(s_t)` to predict return from each partial
sequence, form `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`, and take
`A_t = sum_l (gamma lambda)^l delta_{t+l}` by a reverse scan. Even though the environment
reward is one scalar at EOS, the *critic's* estimate moves at every token, so `delta_t` is
nonzero everywhere -- tokens after which the expected score jumps get positive credit, tokens
after which it drops get negative. The critic manufactures density.

Reward side: fold the KL penalty into the reward per token,
`r_t <- r_t - beta(log pi_theta - log pi_ref)`. That term exists at every position by
construction. I verified that its expected sum is exactly `beta` times the sequence KL
(0.77787 measured versus 0.78974 exact at beta = 0.3), so this is not a hack -- it is the
sequence-level constraint, sampled.

The two interact, and that is the part people miss: the shaped reward is what the critic is
*trained on*, so the dense KL term also densifies the critic's regression target and feeds
back into every `delta_t`. In InstructGPT the sparse RM reward and the dense per-token KL are
summed into one `r_t` and GAE runs on the total.

Tuning is lambda, and the honest version is that lambda is a hedge against a critic you don't
trust. With an *exact* critic every lambda is unbiased -- the `l >= 1` terms have conditional
mean zero -- so lambda = 0 strictly wins, and I measured variance rising 5.3x monotonically
from lambda = 0 to 1 with no bias to show for it. An interior optimum appears only when the
critic is wrong: with a deliberately mis-ranked critic, MSE was 6.954 at lambda = 0, 3.930 at
lambda = 1, and 3.059 at lambda = 0.9. At lambda = 1 the critic appears only as `-V(s_t)`, a
state-only baseline, so you are unbiased for *any* critic -- but every token in a 2000-token
sequence then gets essentially the same sequence-level advantage, which is the credit
assignment problem you were trying to solve. So the real question is how good the value head
is on this distribution, and the diagnostic is whether it *ranks* states correctly rather than
how close `V` is to the return.

For k1 versus k3 on the KL term: use k3 (`rho - 1 - log rho`). Same expectation, `>= 0` on
every token, and 1.47x lower variance in my measurement -- whereas k1 hands out a *bonus* on
24.5% of tokens, paying the policy to deviate. One caveat I'd volunteer: k3's variance
advantage is a small-KL phenomenon. It was 423x better at KL = 0.004 and 4.4x *worse* at
KL = 0.65, because `rho` is a likelihood ratio. It is the right default because RLHF runs at
small per-token KL, not because it dominates everywhere.

## Reusable Artifact

- Path: `week-03-rl-foundations/code/kl_rl.py`
- `./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/kl_rl.py` runs all 94
  checks (~16 s); `--fast` ~6 s; `--list` shows the 10 stages; `<stage> [<stage> ...]` runs a
  subset. `kl tilt objective` need no rollouts and finish in ~1 s.
- `tilted_policy` is the reference implementation of the RLHF optimum -- useful any time you
  need "the best policy at this beta" as a ground truth to compare a real run against.
- `kl_estimators` + the four-regime sweep is a reusable answer to "which KL estimator should
  I log", including the crossover that makes the answer conditional.
- `solve_by_ascent` is a general black-box simplex maximizer (central differences + Adam on
  softmax logits) -- the "structurally independent reference" pattern, reusable for any future
  closed form.
- The over-optimization fixture (`token_bandit`) is the cheapest possible reproduction of
  Gao et al.'s curve, with a negative control built in.

## Open / Next

- **One-line fix outstanding:** `tilted_policy` still calls `np.log(pi_ref)` unguarded, which
  emits a `RuntimeWarning` and reaches the correct `-inf` by IEEE accident rather than by
  intent. Same fix as `kl_divergence`.
- Week 4 assembles Days 1-5 into the full InstructGPT pipeline: PPO's clipped ratio as an
  *implicit* local KL trust region (versus today's explicit penalty), the value head, and the
  actual loss.
- Worth building in Week 4: an **adaptive-beta** controller (InstructGPT's KL target) and the
  measured comparison of "KL as reward shaping" versus "KL as an explicit loss term with an
  analytic gradient". The prediction to test: the analytic-gradient version has strictly lower
  variance for the KL component, because on the local term you have `pi_theta` in closed form
  and do not need a score-function estimator at all -- only the *visitation* part of the KL
  gradient genuinely requires sampling, and every implementation silently drops it.
