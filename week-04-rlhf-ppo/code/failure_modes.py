"""
Week 4 Day 4 -- RLHF failure modes: hacking, length, entropy collapse, drift.

Days 1-3 built the objective and proved every term correct. Today the objective is
correct and the RUN is still bad, which is the situation you are actually paid to
diagnose. The organizing distinction, and the first sentence of any interview answer
on this topic:

    OVER-OPTIMIZATION means the OBJECTIVE is wrong.
    OPTIMIZATION PATHOLOGY means the OPTIMIZER is broken.

    hacking + length  are the first kind.  entropy collapse + value divergence
    are the second. They look IDENTICAL on a reward curve and their fixes are
    OPPOSITE: the first wants a tighter leash, the second wants it loosened.

Four planted failures. Each one gets three things, and the third is the one people
skip:

    1. an INSTRUMENT that fires
    2. a NEGATIVE CONTROL that makes the failure attributable to the cause claimed
    3. a FIX that is measured, not asserted

The two headline experiments:

  * GOODHART WITH A LEARNED RM. W3 D5 found the hump with a hand-specified proxy.
    Today the proxy is a Bradley-Terry model fit on finitely many noisy preferences,
    so the hump's location is a FUNCTION OF THE PREFERENCE DATA and moves when you
    add data. That is the version that answers "how do I know beta is right for THIS
    reward model."

  * OFF-POLICY-BY-ACCIDENT, MEASURED. Six asks, still the standing gap. Today it
    stops being a drill: the distorted sampler's gradient is compared to the EXACT
    grad J from Day 1's enumeration, with a constant-reward control that isolates the
    bug from the reward entirely, and an IS repair that works for temperature and
    provably fails for top-p.

WHY THIS FILE HAS NO MONTE CARLO. The sampler distortion is applied to the tabular
policy and the resulting sequence distribution is ENUMERATED, so "the gradient under a
distorted sampler" is an exact expectation, not an estimate. Every bias number below is
therefore an ASSERTION against ground truth, not a plot with error bars. This is the
same decision that made Weeks 3 and 4 work and it is worth protecting: `E_mu[...]` is
literally `exact_batch` with the WEIGHTS swapped, which is also the clearest possible
statement of what the bug IS.

  run everything      ./.venv/bin/python week-04-rlhf-ppo/code/failure_modes.py
  one stage           ...  failure_modes.py entropy_collapse
  a few stages        ...  failure_modes.py sampler is_repair
  small/fast fixtures ...  failure_modes.py --fast
  list the stages     ...  failure_modes.py --list

Build order (each group is runnable the moment it is written):

  1. distorted_policy                        -> stage `sampler`
  2. token_entropy                           -> stage `sampler`
  3. offpolicy_gradient                      -> stage `entropy_collapse`   <- THE HEADLINE
  4. (nothing new)                           -> stage `zero_reward`        <- the control
  5. importance_weights                      -> stage `is_repair`          <- support vs reweighting
  6. (nothing new)                           -> stage `hacking`
  7. sample_length_confounded_preferences,
     length_coefficient                      -> stage `length`
  8. predict_beta_star                       -> stage `length_prediction`  <- derivation predicts a measurement
  9. debias_length_rm                        -> stage `length_fix`
 10. td_fit                                  -> stage `value_divergence`
 11. diagnose                                -> stage `instruments`        <- the reusable artifact

TRAPS DELIBERATELY LEFT IN THE FIXTURES:

  * top-p produces EXACT ZEROS in the sampler. Every quantity that divides by mu, or
    takes log(mu), has to decide what to do there, and the RIGHT answer is different
    for each: an importance weight is UNDEFINED (that is the whole lesson), an entropy
    term is 0 by the 0*log 0 convention. If you find yourself adding 1e-12 to make a
    warning go away, you have deleted today's headline.

  * `n_pairs=50` over 364 sequences means most sequences are in NO comparison. Their
    fitted reward is whatever the init was. That is not a fixture defect, it is the
    mechanism -- an RM is unconstrained exactly where its data is thin, and an
    optimizing policy goes exactly there. `psi0 = 0` so that "unconstrained" means one
    specific, inspectable thing.

  * the gold reward PENALIZES length (-0.4/token) while the confounded labels teach a
    POSITIVE length coefficient. Proxy and gold therefore disagree in a direction you
    can read off a single scalar, which is what makes the length turnover measurable.

  * two of the four failures do NOT move the KL. If you write `diagnose` around KL you
    will pass two stages and fail two, and the two you fail are the two that matter.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from toy_rlhf import (  # noqa: E402
    TokenEnv,
    enumerate_sequences,
    exact_J,
    sequence_logprob,
    softmax_np,
    state_occupancy,
)
from ppo_loss import (  # noqa: E402
    _pack,
    finite_difference_gradient,
    score_function_gradient,
)


# ---------------------------------------------------------------------------
# Provided infrastructure -- READ IT, you build against this contract
# ---------------------------------------------------------------------------


def sequence_dist(env: TokenEnv, theta: np.ndarray) -> np.ndarray:
    """p(y) for every y in `enumerate_sequences(env)`, as a probability vector.

    Sums to 1 exactly (that is the normalization identity Day 1's `logprob` stage
    asserts), so it can be used as a set of `weights` anywhere `_pack` wants them.
    """
    return np.exp([sequence_logprob(env, theta, s) for s in enumerate_sequences(env)])


def sequence_dist_from_probs(env: TokenEnv, pi: np.ndarray) -> np.ndarray:
    """Same, but from a PROBABILITY table (n_states, vocab) rather than logits.

    Needed because a top-p sampler contains exact zeros and therefore has no finite
    logit representation. Walking probabilities directly avoids inventing a -1e30.
    """
    out = np.empty(len(enumerate_sequences(env)))
    for i, seq in enumerate(enumerate_sequences(env)):
        p = 1.0
        for t, a in enumerate(seq):
            p *= pi[env.state_id(seq[:t]), a]
        out[i] = p
    return out


def policy_from_sequence_dist(env: TokenEnv, p_seq: np.ndarray) -> np.ndarray:
    """Invert `sequence_dist`: the tabular policy (n_states, vocab) whose sequence
    distribution is `p_seq`. Returns PROBABILITIES, not logits.

    Provided because it is plumbing, not a lesson. The identity it relies on:

        pi(a | s) = subtree_mass(s + (a,)) / subtree_mass(s)

    where subtree_mass(prefix) is the total probability of every enumerated sequence
    beginning with `prefix`. This is what lets the closed-form KL-tilted sequence
    distribution `p*(y) ~ p_ref(y) exp(r(y)/beta)` be measured with the same per-token
    instruments (entropy, per-token KL, occupancy) as a policy produced by PPO --
    which is the only reason the beta ladder and the length turnover live on the
    same axes.

    Rows for states the distribution never reaches are left UNIFORM: they carry zero
    occupancy, so nothing that is weighted by occupancy can be affected by them, and
    a NaN there would poison an otherwise-fine mean.
    """
    mass: dict[tuple[int, ...], float] = {}
    for seq, p in zip(enumerate_sequences(env), p_seq):
        for t in range(len(seq) + 1):
            key = seq[:t]
            mass[key] = mass.get(key, 0.0) + float(p)

    pi = np.full((env.n_states, env.vocab), 1.0 / env.vocab)
    for s_id, prefix in enumerate(env.states):
        denom = mass.get(prefix, 0.0)
        if denom <= 0.0:
            continue
        for a in range(env.vocab):
            pi[s_id, a] = mass.get(prefix + (a,), 0.0) / denom
    return pi


def seq_kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over the ENUMERATED sequence distributions, in nats per completion.

    Terms with p == 0 contribute 0 (the 0 log 0 convention), which matters here
    because a top-p policy assigns exact zeros. Terms with q == 0 and p > 0 are
    genuinely +inf and are NOT silently clipped -- if that happens it means the
    reference cannot produce something the policy can, and you want to see it.
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    nz = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p[nz] * (np.log(p[nz]) - np.log(q[nz]))
    return float(terms.sum())


def mean_length(env: TokenEnv, p_seq: np.ndarray) -> float:
    """E[ len(y) ] under a sequence distribution. The length instrument."""
    lens = np.array([len(s) for s in enumerate_sequences(env)], dtype=float)
    return float(p_seq @ lens)


def effective_support(p_seq: np.ndarray) -> float:
    """exp(H(p_seq)) -- the perplexity of the sequence distribution, i.e. the
    "number of completions the policy effectively produces".

    The enumerable stand-in for `distinct-n` / unique-completion count, and a better
    one: it is continuous, so it can be differenced across iterations, and it does not
    depend on how many samples you happened to draw.
    """
    p = np.asarray(p_seq, float)
    nz = p > 0
    return float(np.exp(-(p[nz] * np.log(p[nz])).sum()))


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    """1 - Var(target - pred) / Var(target). The critic instrument, NOT MSE.

    Day 3's reason, in one line: Var is shift-invariant and the shift is exactly the
    component of critic error that is free, so EV's invariance matches the theorem's
    invariance. Can be negative (worse than predicting the mean).
    """
    target = np.asarray(target, float)
    var = target.var()
    if var <= 0:
        return float("nan")
    return float(1.0 - (target - np.asarray(pred, float)).var() / var)


def rm_features(env: TokenEnv, n_spurious: int, seed: int) -> np.ndarray:
    """(n_seqs, 3 + n_spurious) feature matrix the reward model is linear in.

    Columns 0-2 are `n_good`, `len(y)`, `ends_with_eos` -- which SPAN the gold reward
    exactly (`gold = 2*n_good - 0.4*len + 0.75*ends_clean`). The rest are fixed random
    directions with no relationship to gold at all.

    THIS IS WHY THE RM IS A LINEAR MODEL AND NOT A LOOKUP TABLE, and the reason is the
    single most important fixture decision on this day. A tabular RM has NO
    GENERALIZATION: a completion that appeared in no comparison keeps its initial
    value forever, and no amount of extra data changes it. Real reward models are
    function approximators, so:

      * their error in the thin tail is driven by the fit in the HEAD, which means
        more data improves the tail too -- that is what makes the over-optimization
        turnover move with data size, and a table cannot show it;
      * they have SPURIOUS CAPACITY, and with little data they put weight on it. The
        policy is an optimizer pointed at the RM, so it climbs the argmax of that
        spurious weight. "The policy finds the argmax of the ERROR, not of the
        reward" stops being a slogan and becomes a printable vector.

    The spurious weights shrinking as `n_pairs` grows IS the mechanism of Gao's
    scaling law, and stage `hacking` prints them.
    """
    seqs = enumerate_sequences(env)
    n_good = np.array([sum(1 for a in s if a == 0) for s in seqs], dtype=float)
    lengths = np.array([len(s) for s in seqs], dtype=float)
    ends_clean = np.array([1.0 if s and s[-1] == env.eos else 0.0 for s in seqs])
    spurious = np.random.default_rng(seed).standard_normal((len(seqs), n_spurious))
    return np.column_stack([n_good, lengths, ends_clean, spurious])


def fit_sequence_rm(features: np.ndarray, pairs: np.ndarray, lr: float,
                    n_steps: int) -> np.ndarray:
    """Fit a LINEAR Bradley-Terry reward model `psi(y) = features[y] @ omega` by plain
    gradient descent. Returns `omega`, shape (d,).

    Day 1 fit a TABLE by the same loss; this fits coefficients, so the chain rule adds
    exactly one line -- the scatter-add into per-sequence scores, then `features.T @`
    to push it back to the coefficients. That is gather/scatter one more time: many
    pairs collide on one sequence (scatter, needs `np.add.at`), and many sequences
    collide on one coefficient (a matrix transpose does that for you).

    features : (n_seqs, d) from `rm_features`.
    pairs    : (n_pairs, 3) rows of (0, winner_idx, loser_idx).
    lr       : step size. The BT loss is a MEAN over pairs, so each coefficient's
               gradient is O(1/n_pairs) and a "safe-looking" 0.5 does not move.
    n_steps  : full-batch steps. Not reduced by --fast: an unconverged RM breaks a
               stage's claim for a reason unrelated to the claim.

    Day 1's identifiability theorem still applies and is now visible in the features:
    there is no intercept column, because a per-prompt additive constant is exactly
    the direction the pairwise loss cannot see. Adding one would leave its coefficient
    at whatever the init was, forever.
    """
    features = np.asarray(features, float)
    omega = np.zeros(features.shape[1])
    win, lose = pairs[:, 1], pairs[:, 2]
    for _ in range(n_steps):
        psi = features @ omega
        delta = psi[win] - psi[lose]
        g = (1.0 / (1.0 + np.exp(delta))) / len(delta)      # sigmoid(-delta) / n_pairs
        g_psi = np.zeros(features.shape[0])
        np.add.at(g_psi, win, -g)
        np.add.at(g_psi, lose, g)
        omega -= lr * (features.T @ g_psi)
    return omega


def sample_preferences_over_sequences(rng: np.random.Generator, gold: np.ndarray,
                                      sampler: np.ndarray, n_pairs: int,
                                      noise: float) -> np.ndarray:
    """Preference pairs with NO length confound -- the control protocol.

    gold    : (n_seqs,) the true reward, which the RM never sees.
    sampler : (n_seqs,) the distribution completions are drawn from. This is pi_ref,
              and it is why RM error concentrates OFF the SFT distribution: a
              completion with tiny mass here appears in almost no comparison and is
              therefore almost unconstrained.
    n_pairs : number of comparisons.
    noise   : annotator flip rate applied AFTER the Bradley-Terry draw. Real
              inter-annotator agreement on preference data runs in the 70s%.

    Returns (n_pairs, 3) int rows of (0, winner_idx, loser_idx).
    """
    n = len(gold)
    a = rng.choice(n, size=n_pairs, p=sampler)
    b = rng.choice(n, size=n_pairs, p=sampler)
    resample = a == b
    while resample.any():
        b[resample] = rng.choice(n, size=int(resample.sum()), p=sampler)
        resample = a == b
    p_a = 1.0 / (1.0 + np.exp(-(gold[a] - gold[b])))
    a_wins = rng.random(n_pairs) < p_a
    a_wins ^= rng.random(n_pairs) < noise
    winner = np.where(a_wins, a, b)
    loser = np.where(a_wins, b, a)
    return np.stack([np.zeros(n_pairs, dtype=np.int64), winner, loser], axis=-1)


def standardize(proxy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Z-score a reward vector under a distribution. Provided, and load-bearing.

    A Bradley-Terry fit's output SCALE is set by how separable its training data
    happened to be, not by anything about the reward: fitting on 30 comparisons and
    on 3000 produces scores differing by ~10x in spread. So `beta` only means
    something once the proxy is in units of its own standard deviation -- otherwise
    "the same beta ladder" applies wildly different pressure to different reward
    models and no comparison across them is valid.

    Centering is free (Day 1: the RM is identified only up to an additive constant).
    Dividing by the spread is the part that matters. This is exactly what reward
    whitening does in a real stack, and it is why it is not optional there either.

    Every kappa, every beta and every beta* in this file is in these units. Mixing
    standardized and raw scales is the single easiest way to get a beta* prediction
    wrong by three orders of magnitude.
    """
    w = np.asarray(weights, float)
    w = w / w.sum()
    p = np.asarray(proxy, float)
    m = float(w @ p)
    sd = float(np.sqrt(w @ (p - m) ** 2))
    return (p - m) / max(sd, 1e-12)


@dataclass
class RunHistory:
    """One training run's logs, in the form `diagnose` consumes.

    Every field is a LIST OF FLOATS, one entry per iteration -- never a tensor, never
    an array of arrays. Day 3's rule: log floats, not tensors, because a log you
    cannot diff across iterations is not an instrument.

    proxy    : reward under the model being optimized (the RM). What the loss sees.
    gold     : reward under the held-out / gold evaluator. NOT available in a real
               stack at training time -- that is the entire problem, and it is why
               `diagnose` must be able to reach a verdict without it.
    kl       : exact KL(pi_theta || pi_ref) over sequences, per iteration.
    entropy  : occupancy-weighted per-token entropy of pi_theta, in nats.
    length   : E[len(y)] under pi_theta.
    ev       : explained variance of the critic, or NaN if there is no critic.
    label    : what actually happened, for the harness only. `diagnose` must not
               look at it.
    """

    proxy: list[float] = field(default_factory=list)
    gold: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    length: list[float] = field(default_factory=list)
    ev: list[float] = field(default_factory=list)
    label: str = ""

    def log(self, **kw) -> None:
        for k, v in kw.items():
            getattr(self, k).append(float(v))


def _gold_reward(env: TokenEnv):
    """Day 1's structured sequence reward, unchanged.

        2 * (count of token 0)  -  0.4 * len(y)  +  0.75 * [ends with eos]

    Note the SIGN of the length term: gold PENALIZES length. The confounded
    annotators in stage `length` will teach the RM the opposite, and the gap between
    those two facts is the whole of length exploitation.
    """
    def r(seq: tuple[int, ...]) -> float:
        n_good = sum(1 for a in seq if a == 0)
        ends_clean = 1.0 if seq and seq[-1] == env.eos else 0.0
        return 2.0 * n_good - 0.4 * len(seq) + 0.75 * ends_clean
    return r


# ---------------------------------------------------------------------------
# Exercises (TODO bodies)
# ---------------------------------------------------------------------------


def distorted_policy(theta: np.ndarray, temperature: float = 1.0,
                     top_p: float = 1.0) -> np.ndarray:
    """The SAMPLER mu that a rollout worker actually uses. Returns PROBABILITIES,
    shape (n_states, vocab), each row summing to 1.

    This is the function at the centre of today's headline. `theta` is the policy the
    gradient will be attributed to; the returned `mu` is what the tokens are actually
    drawn from. When the two differ, every expectation in the loop is taken under the
    wrong measure -- and nothing in the PPO machinery notices, because a rollout batch
    carries no record of how it was produced.

    theta       : (n_states, vocab) policy logits.
    temperature : divides the LOGITS before the softmax. tau < 1 sharpens, tau > 1
                  flattens, tau == 1 is the identity. Note this is a rescale of
                  logits, so it is monotone and preserves the ranking (and therefore
                  preserves greedy decoding) -- W2 D3.
    top_p       : nucleus threshold in (0, 1]. Keep the SMALLEST set of tokens whose
                  cumulative probability is >= top_p, zero the rest, renormalize.
                  top_p == 1.0 keeps everything.

    ORDER MATTERS AND IT IS NOT ARBITRARY: temperature first, nucleus second, on the
    POST-temperature distribution -- the same composition rule as W2 D3's decode
    filters, and the reason that file returned masked logits rather than probabilities.
    Applying them in the other order gives a different sampler, because temperature
    changes which tokens are inside the nucleus.

    Two things this function must do that are easy to get wrong:
      * the boundary is `>=`, not `>`. Using `>` keeps one extra token at exact ties
        (the HuggingFace quirk, W2 D3) and will make top_p=1.0 not quite an identity.
      * the nucleus is chosen in SORTED order but the zeros have to land in VOCAB
        order. If you build the keep-mask on sorted values you must scatter it back.
        This is W2 D3's #1 bug and it is silent: the row still sums to 1, it just
        keeps the wrong tokens.

    The returned array contains EXACT ZEROS when top_p < 1. That is not a numerical
    nuisance to be patched with an epsilon -- it is the entire top-p lesson, and
    `importance_weights` below is built to detect it.
    """
    raise NotImplementedError("TODO: distorted_policy")


def token_entropy(env: TokenEnv, pi: np.ndarray, occupancy: np.ndarray) -> float:
    """Occupancy-weighted mean per-token entropy of a policy, in nats.

        sum_s d(s) H( pi(. | s) )  /  sum_s d(s)

    NORMALIZED -- it is a mean PER TOKEN, so it lies in [0, log vocab] and is
    comparable across policies with different expected lengths. Leaving off the
    denominator gives the expected TOTAL entropy of a completion, which is a
    different (also useful) quantity that confounds "how uncertain is each decision"
    with "how many decisions are there" -- and today those two move independently,
    which is the entire point of separating the entropy row from the length row.

    pi        : (n_states, vocab) PROBABILITIES (not logits -- a top-p sampler has no
                finite logit form).
    occupancy : (n_states,) the state-visitation weights from `state_occupancy`. Use
                the occupancy of the POLICY BEING MEASURED, not of pi_ref: the
                question "how much entropy does this policy have" is about the states
                this policy actually reaches.

    Why weight by occupancy at all: an unweighted mean over rows counts a state the
    policy reaches with probability 1e-9 exactly as much as the root. That was the
    W3 D5 bug that let 94/94 checks pass while the logged number was wrong (1.053
    unweighted vs 5.056 weighted), and it is the reason `state_occupancy` is
    provided rather than left implicit.

    0 * log 0 is 0. With top-p rows this is not a corner case, it is most of them.
    """
    raise NotImplementedError("TODO: token_entropy")


def offpolicy_gradient(env: TokenEnv, theta: np.ndarray, mu: np.ndarray,
                       reward_fn, center: bool = True,
                       is_weights: np.ndarray | None = None) -> np.ndarray:
    """The gradient a PPO loop ACTUALLY computes when rollouts came from `mu` but the
    update is attributed to `theta`. Shape (n_states, vocab).

    This is the bug, expressed as exactly one changed line relative to a correct
    implementation, which is the point:

        correct:  weight each enumerated sequence by  p_theta(y)
        buggy:    weight each enumerated sequence by  p_mu(y)

    Everything else -- the advantages, the score function, the centering -- is
    identical. That is why the bug is invisible in code review and why it has to be
    caught by an instrument.

    env         : the token environment (supplies the enumeration).
    theta       : (n_states, vocab) policy logits. grad log pi_theta is taken here.
    mu          : (n_states, vocab) sampler PROBABILITIES from `distorted_policy`.
                  Pass `softmax_np(theta)` to get the on-policy case, which must
                  reproduce the exact gradient.
    reward_fn   : callable seq -> float, the sequence reward.
    center      : subtract the batch-mean return before forming advantages, i.e. use
                  a constant baseline. TRUE is what every implementation does.
                  FALSE exists for stage `zero_reward` and the reason is subtle
                  enough to be worth stating: with a CONSTANT reward and centering,
                  the advantages are identically zero and EVERY sampler reports "no
                  bug". A control that cannot fail is not a control.
    is_weights  : optional (n_seqs,) importance ratios to multiply into the per-
                  sequence weights before renormalizing. None = the uncorrected
                  (buggy) estimator. This is how stage `is_repair` asks whether the
                  bias is repairable.

    Build it out of the pieces that already exist: `sequence_dist_from_probs` for the
    mu-weights, `_pack` (Day 2) to assemble a Batch, `score_function_gradient` (Day 2,
    itself transplanted from Week 3) for the score function. You should not be writing
    any `np.add.at` here -- if you are, you are re-deriving something Day 2 already
    asserted against a finite difference.

    ONE SUBTLETY THAT DECIDES WHETHER THIS IS RIGHT. `_pack(env, seqs, theta_old, ...)`
    computes `logp_old` from its third argument. That argument is the policy whose
    log-probabilities you want differentiated -- so it is `theta`, never a sampler.
    The sampler enters ONLY through the weights. If you find yourself packing with a
    logit form of `mu` you have mixed up which distribution plays which role, and the
    on-policy check in stage `entropy_collapse` will not catch it because in that case
    they are equal.
    """
    raise NotImplementedError("TODO: offpolicy_gradient")


def importance_weights(env: TokenEnv, theta: np.ndarray,
                       mu: np.ndarray) -> tuple[np.ndarray, bool]:
    """Per-SEQUENCE importance ratios p_theta(y) / p_mu(y), and whether they are valid.

    Returns (weights, is_valid) where `weights` is (n_seqs,) and `is_valid` is False
    iff the support condition for importance sampling is violated, i.e. iff there is
    a sequence with p_theta(y) > 0 and p_mu(y) == 0.

    theta : (n_states, vocab) policy logits (the TARGET distribution, numerator).
    mu    : (n_states, vocab) sampler probabilities (the PROPOSAL, denominator).

    THIS FUNCTION IS THE ANSWER TO THE QUESTION YOU HAVE BEEN ASKED SIX TIMES, so
    write it so that the answer is legible in the code:

        temperature  keeps every token at positive probability  -> every ratio is
                     finite -> IS repairs the bias EXACTLY.
        top-p        zeroes tokens the policy still assigns mass to -> the ratio is
                     positive / zero -> NO importance weight exists.

    Do not return inf, do not clip, do not add an epsilon to the denominator. Detect
    the violation and report it. The reason is not numerical hygiene, it is that
    "the estimator does not exist" and "the estimator is large" are different claims
    and only one of them is true: an IS ratio can REWEIGHT mass you observed, it can
    never RESURRECT mass you never observed. An epsilon in the denominator silently
    converts an impossibility into a large finite number and the stage will then
    "pass" while asserting something false.

    Where valid, the returned weights must satisfy `sum_y p_mu(y) * w(y) == 1` to
    floating point -- that identity is what stage `is_repair` checks first, and it is
    the same `E[rho] = 1` that made k3 a control variate on Day 3.
    """
    raise NotImplementedError("TODO: importance_weights")


def sample_length_confounded_preferences(rng: np.random.Generator, gold: np.ndarray,
                                         lengths: np.ndarray, sampler: np.ndarray,
                                         n_pairs: int, kappa: float,
                                         noise: float) -> np.ndarray:
    """Preference pairs drawn from annotators who ALSO like long answers.

    Identical to `sample_preferences_over_sequences` except for the label
    probability, which becomes

        P(a preferred over b) = sigmoid( (gold_a - gold_b) + kappa * (L_a - L_b) )

    i.e. the annotator's true utility is `gold + kappa * L`, not `gold`. This is the
    honest model of the failure: nobody injects a length bonus into the reward model,
    the reward model LEARNS one because the data has it. Everything downstream --
    the positive length coefficient, the EOS suppression, the gold/proxy divergence --
    has to emerge from this one changed line, or the experiment proves nothing.

    rng     : generator.
    gold    : (n_seqs,) true reward.
    lengths : (n_seqs,) len(y) for each enumerated sequence.
    sampler : (n_seqs,) pi_ref's sequence distribution; both items are drawn from it.
    n_pairs : number of comparisons.
    kappa   : the confound strength, in reward units per token. kappa = 0 must
              reproduce the unconfounded protocol exactly given the same rng state --
              stage `length` asserts this, because a confound experiment whose kappa=0
              arm differs from the control is measuring its own plumbing.
    noise   : annotator flip rate, applied AFTER the draw.

    Returns (n_pairs, 3) int rows of (0, winner_idx, loser_idx).
    """
    raise NotImplementedError("TODO: sample_length_confounded_preferences")


def length_coefficient(psi: np.ndarray, lengths: np.ndarray,
                       weights: np.ndarray) -> float:
    """The RM's length gradient: the slope of a WEIGHTED least-squares regression of
    the reward-model score on completion length.

        psi(y) ~ intercept + kappa_hat * len(y)

    Returns kappa_hat only.

    psi     : (n_seqs,) fitted reward-model scores.
    lengths : (n_seqs,) completion lengths.
    weights : (n_seqs,) the distribution to regress under. Use pi_ref's sequence
              distribution, NOT uniform: an unweighted regression over all 364
              enumerated completions is dominated by sequences nobody ever samples,
              whose psi is unconstrained by the data. The instrument has to describe
              the RM where the RM is used.

    This is the primary length INSTRUMENT and it is the one to name in an interview
    before naming any fix. Note it is a property of the RM alone -- it can be measured
    before a single PPO step is taken, which makes it a pre-flight check rather than a
    post-mortem.

    A weighted slope is `Cov_w(psi, L) / Var_w(L)`. Use the two-pass form for both
    moments; `E[x^2] - E[x]^2` is the shape that produced a `sqrt(-1.78e-15) -> nan`
    on Day 3.
    """
    raise NotImplementedError("TODO: length_coefficient")


def predict_beta_star(kappa: float, kl_per_token: float,
                      delta_r_content: float = 0.0) -> float:
    """THE DERIVATION, AS CODE. The beta above which lengthening stops paying.

    From the marginal-token comparison (emit EOS now, versus emit one more content
    token and then EOS):

        Delta = kappa + Delta_r_content - beta * E[KL_t]

    so the incentive to lengthen vanishes at the beta where Delta crosses zero.

    kappa           : the RM's per-token length bonus, from `length_coefficient`.
    kl_per_token    : the average per-token KL(pi_theta || pi_ref) incurred by the
                      extra token, in nats. This is the RESISTANCE, and it is the term
                      that makes the whole inequality a competition between two
                      PER-TOKEN quantities.
    delta_r_content : expected change in the content part of the reward from the extra
                      token. Zero for pure filler, which is the interesting case.

    Return +inf when the resistance is zero. That is not a degenerate case to be
    guarded away -- it is consequence (ii) of the derivation and the single best line
    in it: IN-DISTRIBUTION FILLER HAS KL_t ~ 0, SO THE KL PENALTY PROVIDES NO
    RESISTANCE AT ALL, AT ANY BETA. The cheapest hack is verbose but unsurprising
    text, which is exactly what RLHF'd models produce. A function that returns a large
    finite number here would let stage `length_prediction` claim a turnover exists when
    the derivation says it does not.
    """
    raise NotImplementedError("TODO: predict_beta_star")


def debias_length_rm(psi: np.ndarray, lengths: np.ndarray,
                     weights: np.ndarray) -> np.ndarray:
    """THE FIX, MEASURED: remove the RM's length component. Returns (n_seqs,).

    Regress out length -- subtract the fitted linear-in-L part, leaving the residual:

        psi_debiased(y) = psi(y) - kappa_hat * ( len(y) - E_w[len] )

    psi     : (n_seqs,) fitted reward-model scores.
    lengths : (n_seqs,) completion lengths.
    weights : (n_seqs,) the same distribution `length_coefficient` regresses under.

    Two properties stage `length_fix` asserts, and they are worth predicting before
    you run it:
      * the debiased RM's `length_coefficient` is 0 to floating point. The residual of
        a least-squares fit is orthogonal to the regressor -- that is what least
        squares MEANS, so this is an algebraic identity and not an empirical finding.
      * the WITHIN-length ranking is untouched: for any two completions of equal
        length, psi_debiased(a) - psi_debiased(b) == psi(a) - psi(b) exactly. The fix
        removes a confound, not information, and being able to say that is the
        difference between "we length-normalized" and understanding what you did.

    Centering `lengths` by its weighted mean rather than by 0 is deliberate: it keeps
    the debiased scores on the same scale as the originals. Recall from Day 1 that the
    RM is identified only up to an additive constant anyway, so this changes nothing
    that matters -- but stage `length_fix` compares levels, and a gratuitous shift
    there would look like an effect.
    """
    raise NotImplementedError("TODO: debias_length_rm")


def td_fit(phi: np.ndarray, phi_next: np.ndarray, rewards: np.ndarray,
           weights: np.ndarray, gamma: float, n_steps: int, lr: float,
           mc_targets: np.ndarray | None = None,
           clip: float | None = None) -> np.ndarray:
    """Fit a LINEAR value head V(s) = phi(s) @ w by semi-gradient TD, and return the
    weight history, shape (n_steps + 1, d).

    This stage is the DEADLY TRIAD, and the third leg of it is today's other topic:

        function approximation  +  bootstrapping  +  an OFF-POLICY distribution

    Any two are safe. All three can diverge. The value head is a shared-trunk linear
    model (that is the function approximation), the target is bootstrapped (that is
    the `Ahat + V_old` of Day 3), and `weights` is where off-policy-ness enters --
    which is exactly the thing stages `entropy_collapse` and `is_repair` are about.
    A distorted sampler does not only bias the ACTOR; it is also the third leg that
    lets the CRITIC diverge.

    phi        : (n_transitions, d) features of the state each transition starts in.
                 Feature rows are deliberately TIED (proportional) across states --
                 that is what a shared trunk IS: one parameter serving many states,
                 so you cannot move one prediction without moving the others.
    phi_next   : (n_transitions, d) features of the successor. A row of ZEROS marks a
                 terminal successor, which is how V(terminal) = 0 becomes DATA rather
                 than an `if` -- the same decision as Day 3's pinned terminal row.
    rewards    : (n_transitions,) reward on each transition.
    weights    : (n_transitions,) how often each transition is visited. Sums to 1.
                 UNIFORM is the on-policy case here; concentrating the mass on the
                 early transitions is the off-policy case.
    gamma      : discount.
    n_steps    : number of full-batch updates.
    lr         : step size.
    mc_targets : (n_transitions,) fixed regression targets. When given, the target
                 does NOT depend on w and this is plain weighted least squares --
                 which is the NEGATIVE CONTROL: same features, same off-policy
                 weights, no bootstrap, and therefore no divergence. Without that
                 control you cannot attribute the divergence to bootstrapping rather
                 than to the feature tying or the weighting.
    clip       : if not None, limit each update so |w_new - w| <= clip componentwise.
                 The value clip, doing the one thing it is actually good for.

    SEMI-gradient means exactly one thing and it is the whole mechanism: you
    differentiate the PREDICTION and treat the TARGET as a constant, even though the
    target contains w. So the update chases a target that moves when you move -- and
    if the successor's feature is larger than the state's own, each step moves the
    target FURTHER than it moves the prediction. Write the update, then look at the
    sign of `sum_i weights_i * phi_i * (gamma * phi_next_i - phi_i)`: positive means
    every step amplifies w, and that scalar is the entire divergence condition.

    Append a COPY of w each step. Day 3's `log.append(phi)` appended a reference and
    every snapshot aliased the final value -- W3 D3's `history.append(tensor)` for
    the third time.
    """
    raise NotImplementedError("TODO: td_fit")


def diagnose(history: RunHistory) -> list[tuple[str, str]]:
    """THE REUSABLE ARTIFACT. Given a run's logs, return which failures fired.

    Returns a list of (failure_name, one_line_reason) pairs, possibly empty. Names
    must come from exactly this set, spelled exactly this way, because Day 5's loop
    imports this function and switches on them:

        "reward_hacking"      "length_exploitation"
        "entropy_collapse"    "value_divergence"      "policy_drift"

    history : a `RunHistory`. You may read any field EXCEPT `label`, which is the
              harness's answer key. You may read `gold` -- but see below.

    DESIGN CONSTRAINTS, each of which is a lesson rather than a style preference:

      1. IT MUST REPORT NOTHING ON A HEALTHY RUN. Stage `instruments` runs it against
         a healthy run and asserts the result is empty. A diagnostic that always fires
         is not a diagnostic, and the false-positive check is part of the artifact,
         not an extra.

      2. IT MUST NOT BE BUILT AROUND KL. Two of the four failures do not move the KL:
         entropy collapse happens INSIDE the reference's support (a sharpened policy
         can be very close in KL), and length exploitation changes one decision --
         when to emit EOS -- which is cheap in nats. If `diagnose` keys on KL it will
         pass the two easy stages and fail the two that matter. KL is a NECESSARY
         instrument, not a sufficient one, and this function is where you either
         believe that or find out you did not.

      3. `gold` IS A LUXURY. In production you do not have it at training time. Use it
         where it genuinely discriminates -- proxy-up-while-gold-down is the definition
         of over-optimization -- but every other verdict should be reachable from the
         instruments a real run logs: entropy, length, KL, EV, and the proxy.

      4. THE REASON STRING IS THE DELIVERABLE. "entropy fell 40%" is a reason; "entropy
         collapse detected" is a restatement. Quote the number that fired.

    Thresholds are yours to choose; the stage asserts behaviour, not your constants.
    Choose them on RELATIVE change across the run (last vs first) rather than absolute
    levels, so the function transfers to a run with a different reward scale -- the
    same reason Day 3 preferred EV over MSE.
    """
    raise NotImplementedError("TODO: diagnose")


# ---------------------------------------------------------------------------
# Self-check harness (staged)
# ---------------------------------------------------------------------------

_TALLY = {"pass": 0, "fail": 0}


def _check(name: str, cond: bool, detail: str = "") -> None:
    cond = bool(cond)
    _TALLY["pass" if cond else "fail"] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


class Ctx:
    """Lazily-built fixtures, each from its OWN seeded generator so a stage's data does
    not depend on which other stages ran first."""

    def __init__(self, fast: bool = False):
        self.fast = fast
        # vocab=4 (tokens 0,1,2 + eos=3), t_max=5 -> 121 states, 364 sequences,
        # lengths 1..5. Four different numbers again (V=4, T=5, S=121, B=364) so a
        # transposed array cannot pass silently.
        self.env = TokenEnv(vocab=4, t_max=5) if not fast else TokenEnv(vocab=4, t_max=4)
        self.n_pairs_ladder = (30, 300, 3_000) if not fast else (30, 300, 1_000)
        self.n_pairs = 20_000 if not fast else 6_000
        # NOT reduced by --fast: gradient-descent STEPS buy convergence, and an
        # unconverged RM breaks a stage's claim for a reason unrelated to the claim.
        self.rm_steps = 3_000
        self.rm_lr = 20.0
        # 20 spurious feature directions and a 25% annotator flip rate. Both are on
        # the high side ON PURPOSE: they make the RM's estimation error large enough
        # at 30 comparisons that an optimizing policy can find it, which is the only
        # way the hump is a MEASUREMENT rather than a hope. Both shrink in effect as
        # n_pairs grows, which is the point of the ladder.
        self.n_spurious = 20
        self.noise = 0.25
        self.n_seeds = 3
        self.betas = tuple(np.exp(np.linspace(np.log(24.0), np.log(0.05), 12)))

    # -- policy-side fixtures ------------------------------------------------

    @cached_property
    def theta(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(0), scale=1.2)

    @cached_property
    def theta_ref(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(1), scale=0.8)

    @cached_property
    def seqs(self) -> list:
        return enumerate_sequences(self.env)

    @cached_property
    def lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.seqs], dtype=float)

    @cached_property
    def reward(self):
        return _gold_reward(self.env)

    @cached_property
    def gold_vec(self) -> np.ndarray:
        return np.array([self.reward(s) for s in self.seqs])

    @cached_property
    def p_ref(self) -> np.ndarray:
        return sequence_dist(self.env, self.theta_ref)

    @cached_property
    def p_theta(self) -> np.ndarray:
        return sequence_dist(self.env, self.theta)

    @cached_property
    def g_true(self) -> np.ndarray:
        print("  ... building the exact finite-difference gradient (~10s, cached)")
        return finite_difference_gradient(self.env, self.theta, self.reward)

    # -- reward-model-side fixtures ------------------------------------------

    @cached_property
    def features(self) -> np.ndarray:
        return rm_features(self.env, self.n_spurious, seed=3)

    def fit_rm(self, n_pairs: int, seed: int, kappa: float = 0.0) -> np.ndarray:
        """Fit an RM on `n_pairs` comparisons drawn from pi_ref. Returns (n_seqs,)."""
        return self.features @ self.fit_omega(n_pairs, seed, kappa)

    def fit_omega(self, n_pairs: int, seed: int, kappa: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(1000 + seed)
        if kappa == 0.0:
            pairs = sample_preferences_over_sequences(
                rng, self.gold_vec, self.p_ref, n_pairs, self.noise)
        else:
            pairs = sample_length_confounded_preferences(
                rng, self.gold_vec, self.lengths, self.p_ref, n_pairs,
                kappa=kappa, noise=self.noise)
        return fit_sequence_rm(self.features, pairs, self.rm_lr, self.rm_steps)

    # -- the beta ladder, shared by `hacking` and `length` -------------------

    def tilt_ladder(self, proxy: np.ndarray, betas=None) -> list[dict]:
        """For each beta, the exact KL-regularized optimum against `proxy`:

            p*(y)  proportional to  p_ref(y) * exp( proxy(y) / beta )

        W3 D5's closed form, over whole SEQUENCES. Using the closed form instead of
        running PPO is deliberate: it removes the optimizer from the experiment
        entirely, so the hump cannot be blamed on a learning rate. Every row carries
        both rewards and every instrument, so one ladder feeds every stage.

        The proxy is STANDARDIZED under pi_ref first (zero mean, unit variance), and
        that is not cosmetic. A Bradley-Terry fit's output SCALE is set by how
        separable its training data happened to be, not by anything about the reward
        -- fitting on 30 pairs and on 3000 pairs produces scores differing by ~10x in
        spread. Without standardization the "same beta ladder" applies wildly
        different optimization pressure to each arm and the data-size comparison is
        meaningless. Centering is free (Day 1: the RM is identified only up to an
        additive constant); dividing by the spread puts beta in units of proxy
        standard deviations, which is the only unit in which betas are comparable.
        This is exactly what reward whitening does in a real stack, and it is why it
        is not optional there either.
        """
        betas = self.betas if betas is None else betas
        z = standardize(proxy, self.p_ref)
        log_ref = np.log(np.where(self.p_ref > 0, self.p_ref, 1e-300))
        rows = []
        for beta in betas:
            p = softmax_np(log_ref + z / beta)
            pi = policy_from_sequence_dist(self.env, p)
            occ = _occupancy_from_probs(self.env, pi)
            rows.append(dict(
                beta=float(beta),
                p=p,
                proxy=float(p @ z),
                gold=float(p @ self.gold_vec),
                kl=seq_kl(p, self.p_ref),
                length=mean_length(self.env, p),
                entropy=token_entropy(self.env, pi, occ),
                support=effective_support(p),
            ))
        return rows


def _occupancy_from_probs(env: TokenEnv, pi: np.ndarray) -> np.ndarray:
    """`state_occupancy` for a PROBABILITY table rather than logits (provided).

    Same forward recursion; it exists only because a top-p sampler cannot be written
    as logits without inventing a -1e30.
    """
    d = np.zeros(env.n_states)
    d[env.state_id(())] = 1.0
    for s_id, prefix in enumerate(env.states):
        if len(prefix) + 1 >= env.t_max:
            continue
        for a in range(env.vocab - 1):
            d[env.state_id(prefix + (a,))] += d[s_id] * pi[s_id, a]
    return d


def _grad_error(g: np.ndarray, g_true: np.ndarray) -> float:
    """Relative L2 gradient error -- the same measure Days 2 and 3 report."""
    return float(np.linalg.norm(g - g_true) / max(np.linalg.norm(g_true), 1e-12))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    c = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _fail_fast(ctx: Ctx, *needed: str) -> None:
    """Touch ONLY the TODOs this stage actually needs, on a tiny input, so a missing
    implementation surfaces before we pay for a finite-difference gradient or a
    3000-step reward-model fit -- and so a stage is never blocked by a TODO it does
    not use.

    Day 3's lesson, fixed mid-session: a guard that touches ALL the TODOs makes a
    finished function report "[TODO] not implemented", and a guard that emits a FALSE
    failure is worse than no guard.
    """
    tiny = TokenEnv(vocab=3, t_max=2)
    th = np.zeros((tiny.n_states, tiny.vocab))
    pi = softmax_np(th)
    n = len(enumerate_sequences(tiny))
    rng = np.random.default_rng(0)
    ones = np.ones(n) / n

    probes = {
        "distorted_policy": lambda: distorted_policy(th, 1.0, 1.0),
        "token_entropy": lambda: token_entropy(tiny, pi, state_occupancy(tiny, th)),
        "offpolicy_gradient": lambda: offpolicy_gradient(tiny, th, pi, lambda s: 1.0),
        "importance_weights": lambda: importance_weights(tiny, th, pi),
        "sample_length_confounded_preferences": lambda: (
            sample_length_confounded_preferences(
                rng, np.zeros(n), np.ones(n), ones, 4, kappa=0.0, noise=0.0)),
        "length_coefficient": lambda: length_coefficient(
            np.zeros(n), np.arange(n, dtype=float), ones),
        "predict_beta_star": lambda: predict_beta_star(0.1, 0.5, 0.0),
        "debias_length_rm": lambda: debias_length_rm(
            np.zeros(n), np.arange(n, dtype=float), ones),
        "td_fit": lambda: td_fit(
            np.ones((1, 1)), np.zeros((1, 1)), np.zeros(1), np.ones(1),
            0.9, 1, 0.1),
        "diagnose": lambda: diagnose(RunHistory()),
    }
    for name in needed:
        probes[name]()


# ---- individual stages ----------------------------------------------------


def stage_sampler(ctx: Ctx) -> None:
    """distorted_policy / token_entropy -- the sampler, and the zeros top-p creates"""
    _fail_fast(ctx, "distorted_policy", "token_entropy")
    env, th = ctx.env, ctx.theta
    pi = softmax_np(th)

    ident = distorted_policy(th, temperature=1.0, top_p=1.0)
    _check("temperature=1, top_p=1 is the IDENTITY (the `>=` boundary, not `>`)",
           np.allclose(ident, pi, atol=1e-12), f"max|d| = {np.abs(ident - pi).max():.2e}")

    for tau, tp in [(0.7, 1.0), (1.0, 0.9), (0.7, 0.9), (1.3, 1.0)]:
        d = distorted_policy(th, temperature=tau, top_p=tp)
        _check(f"rows sum to 1 at tau={tau}, top_p={tp}",
               np.allclose(d.sum(1), 1.0), f"max|sum-1| = {np.abs(d.sum(1) - 1).max():.2e}")

    hot = distorted_policy(th, temperature=1.3, top_p=1.0)
    cold = distorted_policy(th, temperature=0.7, top_p=1.0)
    occ = state_occupancy(env, th)
    h_hot = token_entropy(env, hot, occ)
    h_one = token_entropy(env, pi, occ)
    h_cold = token_entropy(env, cold, occ)
    _check("entropy is monotone in temperature: H(1.3) > H(1.0) > H(0.7)",
           h_hot > h_one > h_cold, f"{h_hot:.4f} > {h_one:.4f} > {h_cold:.4f}")
    _check("entropy is bounded by log(vocab)", h_hot < np.log(env.vocab),
           f"{h_hot:.4f} < {np.log(env.vocab):.4f}")

    _check("temperature keeps FULL support (every token still positive)",
           np.all(cold > 0), f"{int((cold <= 0).sum())} zeros")

    nuc = distorted_policy(th, temperature=1.0, top_p=0.9)
    n_zero = int((nuc == 0.0).sum())
    _check("top-p creates EXACT zeros -- the support is TRUNCATED, not reweighted",
           n_zero > 0, f"{n_zero} of {nuc.size} entries are exactly 0")
    _check("top-p never zeroes a row's argmax (the nucleus always contains the mode)",
           np.all(nuc[np.arange(env.n_states), pi.argmax(1)] > 0))

    # The ordering trap: nucleus is chosen in sorted order, zeros land in vocab order.
    kept = nuc > 0
    ranks = np.argsort(np.argsort(-pi, axis=1), axis=1)
    _check("kept tokens are exactly the TOP-ranked ones (scatter-back is correct)",
           np.all(ranks[kept] < kept.sum(1, keepdims=True).repeat(env.vocab, 1)[kept]),
           "a keep-mask left in sorted order fails here while still summing to 1")

    # Composition order: temp-then-nucleus differs from nucleus-then-temp, because
    # temperature changes which tokens are inside the nucleus.
    both = distorted_policy(th, temperature=0.7, top_p=0.9)
    _check("temp+top_p is not the same as top_p alone (order is load-bearing)",
           not np.allclose(both, nuc))
    _check("entropy falls under top-p as well as under temperature",
           token_entropy(env, nuc, occ) < h_one,
           f"{token_entropy(env, nuc, occ):.4f} < {h_one:.4f}")


def stage_entropy_collapse(ctx: Ctx) -> None:
    """offpolicy_gradient -- the bias against EXACT grad J, per sampler. THE HEADLINE"""
    _fail_fast(ctx, "distorted_policy", "token_entropy", "offpolicy_gradient")
    env, th = ctx.env, ctx.theta
    pi = softmax_np(th)
    g_true = ctx.g_true

    g_on = offpolicy_gradient(env, th, pi, ctx.reward, center=True)
    err_on = _grad_error(g_on, g_true)
    _check("ON-POLICY (mu = pi): reproduces the EXACT gradient -- no MC error to hide in",
           err_on < 1e-6, f"relative error {err_on:.3e}")

    samplers = {
        "temp 0.7": distorted_policy(th, temperature=0.7, top_p=1.0),
        "temp 0.5": distorted_policy(th, temperature=0.5, top_p=1.0),
        "top-p 0.9": distorted_policy(th, temperature=1.0, top_p=0.9),
        "top-p 0.9 + temp 0.7": distorted_policy(th, temperature=0.7, top_p=0.9),
    }
    occ = state_occupancy(env, th)
    h_on = token_entropy(env, pi, occ)

    print(f"  {'sampler':<22} {'grad bias':>10} {'angle':>8} {'H(mu)':>8} {'dH':>8}")
    print(f"  {'on-policy':<22} {err_on:>10.2e} {0.0:>8.2f} {h_on:>8.4f} {0.0:>8.4f}")
    rows = {}
    for name, mu in samplers.items():
        g = offpolicy_gradient(env, th, mu, ctx.reward, center=True)
        err = _grad_error(g, g_true)
        ang = _angle_deg(g, g_true)
        h_mu = token_entropy(env, mu, _occupancy_from_probs(env, mu))
        rows[name] = dict(err=err, ang=ang, h=h_mu)
        print(f"  {name:<22} {err:>10.3f} {ang:>8.2f} {h_mu:>8.4f} {h_mu - h_on:>8.4f}")

    for name, r in rows.items():
        _check(f"{name}: bias is BOUNDED AWAY FROM ZERO (not MC noise -- this is exact)",
               r["err"] > 100 * max(err_on, 1e-12), f"{r['err']:.4f} vs on-policy {err_on:.2e}")
        _check(f"{name}: sampler entropy is strictly BELOW the policy's",
               r["h"] < h_on, f"{r['h']:.4f} < {h_on:.4f}")

    _check("a sharper sampler biases MORE (tau 0.5 worse than tau 0.7)",
           rows["temp 0.5"]["err"] > rows["temp 0.7"]["err"],
           f"{rows['temp 0.5']['err']:.4f} > {rows['temp 0.7']['err']:.4f}")
    _check("stacking distortions compounds it (top-p + temp worst)",
           rows["top-p 0.9 + temp 0.7"]["err"] > rows["top-p 0.9"]["err"],
           f"{rows['top-p 0.9 + temp 0.7']['err']:.4f} > {rows['top-p 0.9']['err']:.4f}")


def stage_zero_reward(ctx: Ctx) -> None:
    """the decisive control: the bug has a nonzero drift with NO reward at all"""
    _fail_fast(ctx, "distorted_policy", "offpolicy_gradient")
    env, th = ctx.env, ctx.theta
    pi = softmax_np(th)
    const = lambda seq: 1.0  # noqa: E731

    # (a) With centering, a constant reward gives identically-zero advantages, so
    #     EVERY arm reports "no bug". This is the control that cannot fail -- and
    #     asserting it is what stops you from shipping it as evidence.
    for name, mu in [("on-policy", pi),
                     ("temp 0.7", distorted_policy(th, 0.7, 1.0)),
                     ("top-p 0.9", distorted_policy(th, 1.0, 0.9))]:
        g = offpolicy_gradient(env, th, mu, const, center=True)
        _check(f"centered + constant reward: {name} update is exactly 0 "
               f"(a control that cannot fail)", np.abs(g).max() < 1e-12,
               f"max|g| = {np.abs(g).max():.2e}")

    # (b) Without centering, on-policy is STILL zero -- E_pi[grad log pi] = 0 is an
    #     identity, so a constant reward cannot move an on-policy update.
    g_on = offpolicy_gradient(env, th, pi, const, center=False)
    _check("uncentered + constant reward, ON-POLICY: still exactly 0 "
           "(E_pi[grad log pi] = 0)", np.abs(g_on).max() < 1e-10,
           f"max|g| = {np.abs(g_on).max():.2e}")

    # (c) ...and that identity needs the expectation under PI. Take it under MU and it
    #     is false. What survives is c * grad E_mu[log pi_theta]: an MLE step fitting
    #     the policy to the SHARPENED sampler's samples. Pure distillation, with zero
    #     reward information present.
    occ = state_occupancy(env, th)
    h0 = token_entropy(env, pi, occ)
    print(f"  {'sampler':<22} {'|g|':>10} {'H after one step':>18} {'dH':>9}")
    for name, mu in [("temp 0.7", distorted_policy(th, 0.7, 1.0)),
                     ("top-p 0.9", distorted_policy(th, 1.0, 0.9))]:
        g = offpolicy_gradient(env, th, mu, const, center=False)
        _check(f"uncentered + constant reward, {name}: update is NOT zero",
               np.abs(g).max() > 1e-6, f"max|g| = {np.abs(g).max():.2e}")

        # An ascent step on a constant reward is exactly an MLE step toward mu.
        # Keep it SMALL: the claim is a first-order one (grad E_mu[log pi] is minus
        # the gradient of KL(mu||pi)), so a step big enough to leave the linear
        # regime can move KL the other way and does. Scale so the largest logit
        # moves by 0.05.
        th_next = th + (0.05 / max(np.abs(g).max(), 1e-12)) * g
        pi_next = softmax_np(th_next)
        h1 = token_entropy(env, pi_next, _occupancy_from_probs(env, pi_next))
        print(f"  {name:<22} {np.abs(g).max():>10.2e} {h1:>18.4f} {h1 - h0:>9.4f}")
        _check(f"{name}: entropy FALLS after a step driven by a constant reward",
               h1 < h0, f"{h1:.4f} < {h0:.4f}")

        # The mechanism, asserted directly: c * E_mu[grad log pi] is the gradient of
        # the log-likelihood of mu's samples, so it must point toward mu.
        p_mu = sequence_dist_from_probs(env, mu)
        kl_before = seq_kl(p_mu, ctx.p_theta)
        kl_after = seq_kl(p_mu, sequence_dist(env, th_next))
        _check(f"{name}: the step moves pi TOWARD mu -- it is an MLE/distillation step",
               kl_after < kl_before, f"KL(mu||pi) {kl_before:.4f} -> {kl_after:.4f}")


def stage_is_repair(ctx: Ctx) -> None:
    """importance_weights -- IS fixes temperature EXACTLY and provably cannot fix top-p"""
    _fail_fast(ctx, "distorted_policy", "importance_weights", "offpolicy_gradient")
    env, th = ctx.env, ctx.theta
    pi = softmax_np(th)
    g_true = ctx.g_true

    mu_t = distorted_policy(th, temperature=0.7, top_p=1.0)
    w_t, ok_t = importance_weights(env, th, mu_t)
    _check("temperature: the support condition HOLDS", ok_t)
    p_mu_t = sequence_dist_from_probs(env, mu_t)
    _check("temperature: E_mu[rho] == 1 exactly (the same identity that made k3 a "
           "control variate)", abs(float(p_mu_t @ w_t) - 1.0) < 1e-10,
           f"{float(p_mu_t @ w_t):.12f}")

    err_raw = _grad_error(offpolicy_gradient(env, th, mu_t, ctx.reward), g_true)
    err_fix = _grad_error(
        offpolicy_gradient(env, th, mu_t, ctx.reward, is_weights=w_t), g_true)
    _check("temperature: IS repairs the bias EXACTLY, not approximately",
           err_fix < 1e-6, f"{err_raw:.4f} -> {err_fix:.3e}")

    mu_p = distorted_policy(th, temperature=1.0, top_p=0.9)
    w_p, ok_p = importance_weights(env, th, mu_p)
    _check("top-p: the support condition FAILS -- no importance weight exists",
           not ok_p, "supp(pi) is NOT a subset of supp(mu)")

    p_th = ctx.p_theta
    p_mu_p = sequence_dist_from_probs(env, mu_p)
    lost = float(p_th[p_mu_p == 0.0].sum())
    _check("top-p: the policy assigns real mass to sequences mu NEVER emits",
           lost > 1e-3, f"{lost:.4f} of pi's mass is unreachable under mu")

    err_raw_p = _grad_error(offpolicy_gradient(env, th, mu_p, ctx.reward), g_true)
    err_fix_p = _grad_error(
        offpolicy_gradient(env, th, mu_p, ctx.reward, is_weights=w_p), g_true)
    _check("top-p: applying IS anyway does NOT repair it -- the bias has a FLOOR",
           err_fix_p > 1e-3, f"{err_raw_p:.4f} -> {err_fix_p:.4f}, not -> 0")

    print(f"  temperature   raw {err_raw:7.4f}  ->  IS {err_fix:.3e}   REPAIRED")
    print(f"  top-p         raw {err_raw_p:7.4f}  ->  IS {err_fix_p:7.4f}   FLOOR "
          f"({lost * 100:.1f}% of pi's mass never observed)")
    print("  >> an IS ratio can REWEIGHT mass you observed; it cannot RESURRECT mass")
    print("     you never observed. Support, not reweighting.")


def stage_hacking(ctx: Ctx) -> None:
    """Goodhart with a LEARNED reward model, and the mechanism printed as a vector"""
    _fail_fast(ctx, "token_entropy")
    env = ctx.env
    print(f"  env: {env}, {len(ctx.seqs)} enumerated completions, "
          f"{ctx.features.shape[1]} RM features "
          f"({ctx.features.shape[1] - ctx.n_spurious} real + {ctx.n_spurious} spurious)")
    ref_gold = float(ctx.p_ref @ ctx.gold_vec)
    print(f"  pi_ref's own gold reward: {ref_gold:.4f}   best possible: "
          f"{ctx.gold_vec.max():.4f}")

    summary = {}
    for n_pairs in ctx.n_pairs_ladder:
        per_seed = []
        spur = []
        for seed in range(ctx.n_seeds):
            omega = ctx.fit_omega(n_pairs, seed=seed)
            spur.append(float(np.abs(omega[3:]).max()))
            per_seed.append(ctx.tilt_ladder(ctx.features @ omega))

        golds = np.array([[r["gold"] for r in rows] for rows in per_seed])
        proxies = np.array([[r["proxy"] for r in rows] for rows in per_seed])
        kls = np.array([[r["kl"] for r in rows] for rows in per_seed])
        gm, gs = golds.mean(0), golds.std(0)
        best = int(np.argmax(gm))
        summary[n_pairs] = dict(gold=gm, spread=gs, kl=kls.mean(0), best=best,
                                spur=float(np.mean(spur)))

        print(f"\n  n_pairs = {n_pairs}   (mean +- spread over {ctx.n_seeds} seeds, "
              f"max |spurious weight| = {np.mean(spur):.2f})")
        print(f"    {'beta':>6} {'sqrt(KL)':>9} {'proxy':>9} {'GOLD':>16}")
        for j, rows in enumerate(zip(*per_seed)):
            b = rows[0]["beta"]
            star = "  <-- gold max" if j == best else ""
            print(f"    {b:>6.2f} {np.sqrt(kls.mean(0)[j]):>9.4f} "
                  f"{proxies.mean(0)[j]:>9.4f} "
                  f"{gm[j]:>9.4f} +-{gs[j]:<5.3f}{star}")

        for s in range(ctx.n_seeds):
            _check(f"n={n_pairs} seed {s}: proxy is MONOTONE in 1/beta -- it reports "
                   f"success the whole way down",
                   all(np.diff(proxies[s]) > -1e-9),
                   f"{proxies[s][0]:.4f} -> {proxies[s][-1]:.4f}")

    # --- the headline: optimization pressure AMPLIFIES estimation error --------
    lo, hi = ctx.n_pairs_ladder[0], ctx.n_pairs_ladder[-1]
    slo, shi = summary[lo], summary[hi]

    print(f"\n  gold SPREAD across seeds, weakest vs strongest pressure:")
    for n in ctx.n_pairs_ladder:
        s = summary[n]
        print(f"    n={n:<5d}  beta=24: +-{s['spread'][0]:.4f}   "
              f"beta=0.05: +-{s['spread'][-1]:.4f}   "
              f"({s['spread'][-1] / max(s['spread'][0], 1e-12):.0f}x)")

    for n in ctx.n_pairs_ladder:
        s = summary[n]
        _check(f"n={n}: seed spread in GOLD grows with optimization pressure -- "
               f"the same RM error, amplified by the argmax",
               s["spread"][-1] > 10 * s["spread"][0],
               f"+-{s['spread'][0]:.4f} -> +-{s['spread'][-1]:.4f}")

    _check("at LOW pressure more RM data means a more reproducible outcome",
           shi["spread"][0] < slo["spread"][0],
           f"+-{slo['spread'][0]:.4f} -> +-{shi['spread'][0]:.4f}")

    # This is the interview sentence, and it is stronger than "the reward went up":
    # with 30 comparisons the OUTCOME OF RLHF IS SEED-DETERMINED at high pressure,
    # and no reward curve shows you that.
    _check("at HIGH pressure the outcome is dominated by which seed you got -- the "
           "spread exceeds the entire mean improvement over pi_ref",
           slo["spread"][-1] > abs(slo["gold"][-1] - ref_gold),
           f"spread +-{slo['spread'][-1]:.4f} vs improvement "
           f"{slo['gold'][-1] - ref_gold:+.4f}")

    # --- the mechanism, as a number -----------------------------------------
    print(f"\n  max |spurious weight|: "
          + ", ".join(f"n={n}: {summary[n]['spur']:.2f}" for n in ctx.n_pairs_ladder))
    _check("the SPURIOUS weights shrink as the RM gets data -- that shrinking IS "
           "the mechanism, and the policy was climbing them",
           shi["spur"] < 0.5 * slo["spur"],
           f"{slo['spur']:.2f} -> {shi['spur']:.2f}")

    # --- an interior maximum somewhere on the ladder ------------------------
    interior = {n: 0 < summary[n]["best"] < len(summary[n]["gold"]) - 1
                for n in ctx.n_pairs_ladder}
    print("  interior gold maximum (in the seed MEAN): "
          + ", ".join(f"n={n}: {interior[n]}" for n in ctx.n_pairs_ladder))
    _check("at least one arm's mean gold has an INTERIOR maximum -- optimizing "
           "further makes it worse while the proxy keeps rising",
           any(interior.values()),
           ", ".join(f"n={n}: peak at index {summary[n]['best']} of "
                     f"{len(summary[n]['gold']) - 1}" for n in ctx.n_pairs_ladder))

    turn = [n for n in ctx.n_pairs_ladder if interior[n]][0]
    d, g = np.sqrt(summary[turn]["kl"]), summary[turn]["gold"]
    slopes = np.diff(g) / np.diff(d)
    _check(f"n={turn}: gold vs sqrt(KL) rises then bends DOWN (Gao's shape)",
           slopes[0] > 0 > slopes[-1],
           f"first slope {slopes[0]:+.4f}, last {slopes[-1]:+.4f}")

    # --- NEGATIVE CONTROL ---------------------------------------------------
    ctrl = ctx.tilt_ladder(ctx.gold_vec)
    ctrl_gold = [r["gold"] for r in ctrl]
    _check("NEGATIVE CONTROL: tilting by the GOLD reward is monotone, no hump, no "
           "seed spread -- so everything above is the PROXY's fault, not the tilt's",
           all(np.diff(ctrl_gold) > -1e-9),
           f"{ctrl_gold[0]:.4f} -> {ctrl_gold[-1]:.4f}, monotone increasing to the "
           f"true optimum {ctx.gold_vec.max():.1f}")


def stage_length(ctx: Ctx) -> None:
    """length exploitation, BUILT from confounded labels rather than injected"""
    _fail_fast(ctx, "sample_length_confounded_preferences", "length_coefficient",
               "token_entropy")
    kappa_ann = 0.8

    rng_a = np.random.default_rng(77)
    rng_b = np.random.default_rng(77)
    pairs_ctrl = sample_preferences_over_sequences(
        rng_a, ctx.gold_vec, ctx.p_ref, 2_000, 0.0)
    pairs_k0 = sample_length_confounded_preferences(
        rng_b, ctx.gold_vec, ctx.lengths, ctx.p_ref, 2_000, kappa=0.0, noise=0.0)
    _check("kappa=0 reproduces the unconfounded protocol EXACTLY on the same rng "
           "(otherwise the experiment measures its own plumbing)",
           np.array_equal(pairs_ctrl, pairs_k0))

    psi_clean = ctx.fit_rm(ctx.n_pairs, seed=0, kappa=0.0)
    psi_conf = ctx.fit_rm(ctx.n_pairs, seed=0, kappa=kappa_ann)

    # All three in the SAME units -- standardized under pi_ref -- because beta is.
    k_gold = length_coefficient(standardize(ctx.gold_vec, ctx.p_ref),
                                ctx.lengths, ctx.p_ref)
    k_clean = length_coefficient(standardize(psi_clean, ctx.p_ref),
                                 ctx.lengths, ctx.p_ref)
    k_conf = length_coefficient(standardize(psi_conf, ctx.p_ref),
                                ctx.lengths, ctx.p_ref)
    print(f"  length coefficient (standardized units, per token)")
    print(f"    gold {k_gold:+.4f} | clean RM {k_clean:+.4f} | confounded RM "
          f"{k_conf:+.4f}")

    # The marginal/partial gap, which is the length confound in miniature.
    print(f"  NOTE: gold's coefficient in its own FORMULA is -0.4 per token, but its "
          f"MARGINAL")
    print(f"  slope on length is {k_gold:+.4f}: longer completions contain more "
          f"content tokens, so")
    print(f"  length is correlated with quality even when it is penalized. That gap "
          f"is exactly")
    print(f"  why the metric is LENGTH-CONTROLLED win rate and not raw win rate.")
    _check("gold's MARGINAL length slope disagrees in sign with its PARTIAL "
           "coefficient (-0.4) -- a simple reward-vs-length regression is confounded "
           "by content", k_gold > 0 > -0.4, f"marginal {k_gold:+.4f} vs partial -0.4")

    _check("the clean RM recovers a NEGATIVE length coefficient once content is in "
           "the feature set (a partial coefficient, which is what the RM fits)",
           k_clean < 0, f"{k_clean:+.4f}")
    _check("the confounded RM ACQUIRES a positive length coefficient -- nobody "
           "injected it, it was LEARNED from the labels", k_conf > 0, f"{k_conf:+.4f}")
    _check("and it is a large effect, not a wobble", k_conf > 10 * abs(k_clean),
           f"{k_conf:+.4f} vs {k_clean:+.4f}")

    conf_rows = ctx.tilt_ladder(psi_conf)
    clean_rows = ctx.tilt_ladder(psi_clean)
    ref_len = mean_length(ctx.env, ctx.p_ref)
    print(f"\n    {'beta':>6} | {'CONFOUNDED':>21} | {'CLEAN':>21}")
    print(f"    {'':>6} | {'len':>6} {'gold':>7} {'KL':>6} | {'len':>6} {'gold':>7} "
          f"{'KL':>6}")
    for rc, rk in zip(conf_rows, clean_rows):
        print(f"    {rc['beta']:>6.2f} | {rc['length']:>6.3f} {rc['gold']:>7.3f} "
              f"{rc['kl']:>6.3f} | {rk['length']:>6.3f} {rk['gold']:>7.3f} "
              f"{rk['kl']:>6.3f}")

    lens = [r["length"] for r in conf_rows]
    _check("mean length RISES far above the reference as beta falls",
           lens[-1] > ref_len + 1.0, f"{lens[0]:.4f} -> {lens[-1]:.4f} "
           f"(pi_ref: {ref_len:.4f})")

    # The comparison that isolates the confound: MATCHED KL, so both runs paid the
    # same amount of divergence, and the only difference is what they bought with it.
    def at_kl(rows, target):
        return min(rows, key=lambda r: abs(r["kl"] - target))
    target_kl = 1.0
    c, k = at_kl(conf_rows, target_kl), at_kl(clean_rows, target_kl)
    print(f"\n  at MATCHED KL ~ {target_kl}: confounded len {c['length']:.3f} gold "
          f"{c['gold']:.3f}  |  clean len {k['length']:.3f} gold {k['gold']:.3f}")
    _check("at matched KL the confounded run is LONGER", c["length"] > k["length"] + 0.2,
           f"{c['length']:.4f} vs {k['length']:.4f}")
    print(f"  NOTE: the confounded run's GOLD is not lower here ({c['gold']:.3f} vs "
          f"{k['gold']:.3f}),")
    print(f"  because THIS gold reward also pays for length indirectly -- more tokens "
          f"means more")
    print(f"  content tokens. That is the finding, not a defect: whether verbosity "
          f"HURTS depends")
    print(f"  on whether the true objective shares the confound, and you cannot know "
          f"that without")
    print(f"  a gold eval. The instrument that fires either way is tokens-per-nat.")

    # Why KL alone misses it: the length run is buying LENGTH per nat, not quality.
    len_per_nat_c = (c["length"] - ref_len) / max(c["kl"], 1e-9)
    len_per_nat_k = (k["length"] - ref_len) / max(k["kl"], 1e-9)
    print(f"  length bought per nat: confounded {len_per_nat_c:+.4f}, "
          f"clean {len_per_nat_k:+.4f}")
    _check("length exploitation converts KL into LENGTH rather than into quality, "
           "which is why a KL-only dashboard cannot see it",
           len_per_nat_c > len_per_nat_k,
           f"{len_per_nat_c:+.4f} vs {len_per_nat_k:+.4f} tokens per nat")


def stage_length_prediction(ctx: Ctx) -> None:
    """predict_beta_star -- the derivation has to PREDICT the measured turnover"""
    _fail_fast(ctx, "sample_length_confounded_preferences", "length_coefficient",
               "predict_beta_star", "token_entropy")
    psi_conf = ctx.fit_rm(ctx.n_pairs, seed=0, kappa=0.8)
    kappa = length_coefficient(standardize(psi_conf, ctx.p_ref), ctx.lengths, ctx.p_ref)

    _check("infinite beta* when the marginal token is in-distribution filler "
           "(KL_t = 0 gives the KL penalty NO purchase, at any beta)",
           np.isinf(predict_beta_star(kappa, 0.0)), "the best line in the derivation")
    _check("beta* scales as 1/KL_per_token (a competition between two PER-TOKEN "
           "quantities)", np.isclose(predict_beta_star(0.2, 0.1),
                                     2.0 * predict_beta_star(0.2, 0.2)))

    # THE INEQUALITY, EVALUATED SELF-CONSISTENTLY. `beta*` is not a one-shot formula:
    # the resistance `beta * KL_per_token(beta)` is itself a function of beta, because
    # a bigger beta means a policy that moved less and therefore incurred less KL. So
    # the turnover solves a FIXED POINT, and the honest experiment is to trace the
    # resistance curve and see whether it ever reaches kappa at all.
    fine = tuple(np.exp(np.linspace(np.log(400.0), np.log(0.05), 40)))
    rows = ctx.tilt_ladder(psi_conf, betas=fine)
    ref_len = mean_length(ctx.env, ctx.p_ref)
    max_len = float(ctx.lengths.max())

    def resistance(r):
        return r["beta"] * (r["kl"] / max(r["length"], 1e-9))

    res = np.array([resistance(r) for r in rows])
    print(f"\n  kappa (measured from the RM, standardized)   {kappa:+.6f} /token")
    print(f"  MAX resistance  max_beta [ beta * KL_per_token ] {res.max():.6f} /token")
    print(f"    (attained at beta = {rows[int(res.argmax())]['beta']:.3f})")

    _check("the resistance curve is BOUNDED -- it vanishes at both ends of the ladder "
           "(large beta = a policy that never moved; small beta = a saturated one)",
           res[0] < res.max() and res[-1] < res.max(),
           f"{res[0]:.2e} .. {res.max():.4f} .. {res[-1]:.4f}")

    lens = np.array([r["length"] for r in rows])
    runaway = kappa > res.max()
    print(f"  => derivation PREDICTS length runs to the ceiling at every beta: "
          f"{runaway}")
    print(f"  => MEASURED max length {lens.max():.4f} vs ceiling {max_len:.1f}")
    _check("the derivation PREDICTS the measurement: kappa exceeds the largest "
           "resistance any beta can generate, so no beta stops the lengthening",
           runaway and lens.max() > max_len - 0.05,
           f"kappa {kappa:.4f} > max resistance {res.max():.4f}, and length reached "
           f"{lens.max():.4f} of a possible {max_len:.1f}")
    print(f"  >> consequence (iii) of the derivation, measured: the length GAIN is "
          f"linear in L")
    print(f"     while the KL RESISTANCE is bounded, so a large enough kappa cannot "
          f"be priced out")
    print(f"     by ANY beta. Length has to be fixed in the DATA, not in the penalty.")

    # NEGATIVE CONTROL: a weak confound, where the inequality predicts the OPPOSITE.
    psi_weak = ctx.fit_rm(ctx.n_pairs, seed=0, kappa=0.1)
    k_weak = length_coefficient(standardize(psi_weak, ctx.p_ref), ctx.lengths, ctx.p_ref)
    rows_w = ctx.tilt_ladder(psi_weak, betas=fine)
    res_w = np.array([resistance(r) for r in rows_w])
    lens_w = np.array([r["length"] for r in rows_w])
    print(f"\n  NEGATIVE CONTROL, weak confound: kappa {k_weak:+.6f} vs max "
          f"resistance {res_w.max():.6f}")
    print(f"    predicts runaway: {k_weak > res_w.max()}   measured max length "
          f"{lens_w.max():.4f} (ceiling {max_len:.1f})")
    _check("NEGATIVE CONTROL: a weak confound sits BELOW the resistance curve, and "
           "then length does NOT run to the ceiling -- the inequality predicts the "
           "sign of the effect in both directions",
           (k_weak < res_w.max()) and (lens_w.max() < max_len - 0.05),
           f"kappa {k_weak:.4f} < max resistance {res_w.max():.4f}, length reached "
           f"only {lens_w.max():.4f}")


def stage_length_fix(ctx: Ctx) -> None:
    """debias_length_rm -- the FIX, measured, and at the DATA level not the penalty"""
    _fail_fast(ctx, "sample_length_confounded_preferences", "length_coefficient",
               "debias_length_rm", "token_entropy")
    psi_conf = ctx.fit_rm(ctx.n_pairs, seed=0, kappa=0.8)
    psi_fix = debias_length_rm(psi_conf, ctx.lengths, ctx.p_ref)

    k_before = length_coefficient(standardize(psi_conf, ctx.p_ref), ctx.lengths, ctx.p_ref)
    k_after = length_coefficient(standardize(psi_fix, ctx.p_ref), ctx.lengths, ctx.p_ref)
    _check("the debiased RM has length coefficient 0 -- an ALGEBRAIC identity "
           "(least-squares residuals are orthogonal to the regressor), not a finding",
           abs(k_after) < 1e-9, f"{k_before:+.4f} -> {k_after:+.2e}")

    same_len = np.where(ctx.lengths == 3)[0][:2]
    if len(same_len) == 2:
        a, b = same_len
        _check("WITHIN-length ranking is untouched (the fix removes a confound, "
               "not information)",
               abs((psi_fix[a] - psi_fix[b]) - (psi_conf[a] - psi_conf[b])) < 1e-10)

    before = ctx.tilt_ladder(psi_conf)
    after = ctx.tilt_ladder(psi_fix)
    ref_len = mean_length(ctx.env, ctx.p_ref)
    print(f"    {'beta':>6} | {'len before':>10} {'gold before':>11} "
          f"| {'len after':>9} {'gold after':>10}")
    for rb, ra in zip(before, after):
        print(f"    {rb['beta']:>6.2f} | {rb['length']:>10.4f} {rb['gold']:>11.4f} "
              f"| {ra['length']:>9.4f} {ra['gold']:>10.4f}")

    ex_before = np.mean([r["length"] - ref_len for r in before])
    ex_after = np.mean([r["length"] - ref_len for r in after])
    print(f"\n  mean excess length over the ladder: {ex_before:+.4f} -> "
          f"{ex_after:+.4f} tokens")
    _check("length is CONTAINED after debiasing, averaged over the whole ladder",
           ex_after < 0.5 * ex_before, f"{ex_before:+.4f} -> {ex_after:+.4f}")

    mid = len(before) // 2
    _check(f"...and in the operating range (beta = {before[mid]['beta']:.2f}) the "
           f"excess is largely removed",
           (after[mid]["length"] - ref_len) < 0.5 * (before[mid]["length"] - ref_len),
           f"{before[mid]['length'] - ref_len:+.4f} -> "
           f"{after[mid]['length'] - ref_len:+.4f} tokens over pi_ref")

    _check("and GOLD RECOVERS at the aggressive end -- the fix is measured, "
           "not asserted", after[-1]["gold"] > before[-1]["gold"],
           f"{before[-1]['gold']:.4f} -> {after[-1]['gold']:.4f}")


def stage_value_divergence(ctx: Ctx) -> None:
    """td_fit -- the deadly triad, and off-policy-ness as its THIRD leg"""
    _fail_fast(ctx, "td_fit")
    # A 3-state chain s0 -> s1 -> s2 -> terminal with TIED features phi = 1, 2, 4
    # (one shared parameter serving three states = a shared trunk). Rewards are
    # chosen so the true value function IS representable, at w = 1 exactly -- so a
    # failure to converge cannot be blamed on the value head lacking capacity.
    gamma = 0.99
    phi = np.array([[1.0], [2.0], [4.0]])
    phi_next = np.array([[2.0], [4.0], [0.0]])          # zeros = terminal
    mc = np.array([1.0, 2.0, 4.0])                      # true reward-to-go
    rewards = mc - gamma * (phi_next @ np.ones(1))      # r = V - gamma V'

    on_policy = np.full(3, 1.0 / 3.0)
    off_policy = np.array([0.90, 0.09, 0.01])           # rarely visit the terminal step

    def cond(w):
        return float((w[:, None] * phi * (gamma * phi_next - phi)).sum())

    print(f"  divergence condition  sum_i w_i phi_i (gamma phi'_i - phi_i)")
    print(f"    on-policy   {cond(on_policy):+.4f}   (negative -> every step contracts)")
    print(f"    off-policy  {cond(off_policy):+.4f}   (positive -> every step AMPLIFIES)")
    _check("the true value function IS representable (w = 1), so capacity is not the "
           "explanation for anything below",
           np.allclose(phi @ np.ones(1), mc), f"{(phi @ np.ones(1))} vs {mc}")

    w_on = td_fit(phi, phi_next, rewards, on_policy, gamma, 600, 0.05)
    _check("ON-POLICY + bootstrap: converges, and to the right answer",
           np.isfinite(w_on[-1]).all() and abs(w_on[-1][0] - 1.0) < 1e-3,
           f"w = {w_on[-1][0]:.6f}")

    w_off = td_fit(phi, phi_next, rewards, off_policy, gamma, 600, 0.05)
    diverged = (not np.isfinite(w_off[-1]).all()) or abs(w_off[-1][0]) > 1e3
    _check("OFF-POLICY + bootstrap + shared features: DIVERGES. Same data, same "
           "features, only the visitation distribution changed",
           diverged, f"|w| = {abs(w_off[-1][0]):.3e}")

    # NEGATIVE CONTROL. Same features, same off-policy weights, no bootstrap.
    w_mc = td_fit(phi, phi_next, rewards, off_policy, gamma, 600, 0.05, mc_targets=mc)
    _check("NEGATIVE CONTROL: off-policy weights + MONTE-CARLO targets converge -- "
           "so the divergence is BOOTSTRAPPING's fault, not the weighting's",
           np.isfinite(w_mc[-1]).all() and abs(w_mc[-1][0] - 1.0) < 1e-3,
           f"w = {w_mc[-1][0]:.6f}")

    w_clip = td_fit(phi, phi_next, rewards, off_policy, gamma, 600, 0.05, clip=0.01)
    step = float(np.abs(np.diff(w_clip, axis=0)).max())
    _check("the value CLIP bounds the per-step movement by exactly the constant asked "
           "for", step <= 0.01 + 1e-12, f"largest single step {step:.6f} <= 0.01")
    _check("...but it only bounds the STEP, not the WALK -- 600 bounded steps still "
           "walk far. Day 2's lesson, now about the critic",
           abs(w_clip[-1][0] - 1.0) > 0.5, f"w = {w_clip[-1][0]:.4f} after 600 steps "
           f"of size <= 0.01")

    def ev_of(w):
        pred = phi @ np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=-1e12)
        return explained_variance(pred, mc)

    print(f"  EV   on-policy {ev_of(w_on[-1]):+.6f} | MC control {ev_of(w_mc[-1]):+.6f}"
          f" | diverged {ev_of(w_off[-1]):+.4e}")
    _check("EV is the instrument that fires: NEGATIVE (worse than predicting the mean) "
           "for the diverged critic", ev_of(w_off[-1]) < 0, f"{ev_of(w_off[-1]):.4e}")
    _check("EV is ~1 for both healthy fits", min(ev_of(w_on[-1]), ev_of(w_mc[-1])) > 0.999)


def stage_instruments(ctx: Ctx) -> None:
    """diagnose -- one function, four planted runs, and a FALSE-POSITIVE check"""
    _fail_fast(ctx, "diagnose")

    def run(label, proxy, gold, kl, entropy, length, ev) -> RunHistory:
        h = RunHistory(label=label)
        for i in range(len(proxy)):
            h.log(proxy=proxy[i], gold=gold[i], kl=kl[i], entropy=entropy[i],
                  length=length[i], ev=ev[i])
        return h

    n = 12
    t = np.linspace(0, 1, n)
    flat = np.ones(n)

    healthy = run("healthy", 0.2 + 1.4 * t, 0.2 + 1.2 * t, 0.4 * t,
                  1.05 - 0.04 * t, 2.5 + 0.05 * t, 0.42 * flat)
    hacking = run("reward_hacking", 0.2 + 2.6 * t, 0.3 + 1.0 * t - 3.2 * t ** 2,
                  6.0 * t ** 1.5, 1.05 - 0.10 * t, 2.5 + 0.1 * t, 0.41 * flat)
    length = run("length_exploitation", 0.2 + 1.9 * t, 0.3 - 1.4 * t, 0.55 * t,
                 1.02 - 0.03 * t, 2.4 + 2.6 * t, 0.40 * flat)
    collapse = run("entropy_collapse", 0.2 + 0.7 * t, 0.25 + 0.2 * t, 0.30 * t,
                   1.05 - 0.85 * t, 2.5 - 0.1 * t, 0.40 * flat)
    valdiv = run("value_divergence", 0.2 + 0.5 * t, 0.25 + 0.3 * t, 0.35 * t,
                 1.04 - 0.02 * t, 2.5 * flat, 0.42 - 1.6 * t)

    verdict = diagnose(healthy)
    _check("HEALTHY run: reports NOTHING (the false-positive check IS the artifact)",
           verdict == [], f"got {verdict}")

    for h in (hacking, length, collapse, valdiv):
        got = diagnose(h)
        names = [n for n, _ in got]
        _check(f"{h.label}: fires", h.label in names, f"got {names}")
        _check(f"{h.label}: the reason quotes a NUMBER, not a restatement",
               any(any(c.isdigit() for c in r) for n, r in got if n == h.label),
               f"got {[r for n, r in got if n == h.label]}")
        for n_, r_ in got:
            print(f"  [{h.label:<20}] -> {n_:<20} {r_}")

    # The two that KL cannot see. A `diagnose` keyed on KL passes hacking and drift
    # and fails exactly these -- which is the point of the whole taxonomy.
    _check("entropy collapse is caught at LOW KL (a sharpened policy stays INSIDE "
           "the reference's support)",
           "entropy_collapse" in [n for n, _ in diagnose(collapse)]
           and collapse.kl[-1] < 0.5, f"final KL {collapse.kl[-1]:.3f}")
    _check("length exploitation is caught at LOW KL (suppressing EOS is one decision, "
           "made many times -- cheap in nats)",
           "length_exploitation" in [n for n, _ in diagnose(length)]
           and length.kl[-1] < 1.0, f"final KL {length.kl[-1]:.3f}")

    drift = run("policy_drift", 0.2 + 1.3 * t, 0.25 + 1.0 * t, 5.5 * t,
                1.05 - 0.05 * t, 2.5 * flat, 0.42 * flat)
    _check("policy drift: caught by KL over ITERATIONS even though every other "
           "instrument looks fine",
           "policy_drift" in [n for n, _ in diagnose(drift)],
           f"got {[n for n, _ in diagnose(drift)]}")


STAGES = [
    ("sampler", stage_sampler),
    ("entropy_collapse", stage_entropy_collapse),
    ("zero_reward", stage_zero_reward),
    ("is_repair", stage_is_repair),
    ("hacking", stage_hacking),
    ("length", stage_length),
    ("length_prediction", stage_length_prediction),
    ("length_fix", stage_length_fix),
    ("value_divergence", stage_value_divergence),
    ("instruments", stage_instruments),
]


# ---- runner (copy verbatim; nothing below needs editing) -------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, fn in STAGES:
            print(f"  {name:<20} {fn.__doc__}")
        return 0
    fast = "--fast" in argv
    names = [a for a in argv if not a.startswith("-")]

    known = {n for n, _ in STAGES}
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}\nknown: {', '.join(sorted(known))}")
        return 2

    ctx = Ctx(fast=fast)
    selected = [(n, f) for n, f in STAGES if not names or n in names]
    todo, broke = [], []

    for name, fn in selected:
        print(f"\n=== {name} ===")
        try:
            fn(ctx)
        except NotImplementedError:
            print("  [TODO] not implemented yet -- skipped")
            todo.append(name)
        except AssertionError:
            broke.append(name)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            broke.append(name)

    print(f"\n{'=' * 60}")
    print(f"checks: {_TALLY['pass']} passed, {_TALLY['fail']} failed"
          + (f" | stages TODO: {', '.join(todo)}" if todo else "")
          + (f" | stages failing: {', '.join(sorted(set(broke)))}" if broke else ""))
    # `_check` tallies rather than raising, so a stage full of failed checks still
    # exits its function normally. Gate the all-clear on the TALLY, not on whether a
    # stage threw -- Day 3's runner printed "All invariants passed" underneath a
    # failing check, which is a status line that lies.
    ok = not todo and not broke and _TALLY["fail"] == 0
    if ok:
        print("All invariants passed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
