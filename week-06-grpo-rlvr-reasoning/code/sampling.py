"""W6 D3 -- sampling, pass@k and rejection sampling: coverage vs mass.

THE THESIS
    pass@k is not a better pass@1. It is a DIFFERENT FUNCTIONAL of the same
    policy, and the derivative is the whole day:

        d pass@k / dp  =  k (1-p)^(k-1)

    so the metric's relative valuation of a HARD problem over an EASY one is
    ((1-p_hard)/(1-p_easy))^(k-1) -- exponential in k. At k = 1 you measure MASS
    (is the mode right?); at large k you measure SUPPORT (is it reachable?).
    Temperature, RL and decoding all buy one with the other.

    And a corollary that gets its own half of the day: pass@k assumes an ORACLE
    picked the right one of your k samples. Name a real selector and you get
    three different limits --

        pass@k -> 1   whenever p > 0                   SUPPORT
        BoN@k  -> 1   iff the verifier is SOUND        ACCEPT SET
        maj@k  -> 1   iff correct is the MODAL answer  MASS

    -- and, when the selector is wrong, two different FAILURE SHAPES. A FILTER's
    accepted-output composition is independent of n; a RANKER's hack probability is
    1 - [1 - (1-p)gamma]^n, which is pass@n pointed at the false-positive mass.

THE IDENTITY THAT OPENS THE DAY (D1 and pass@k are the same algebra)

    pass@k  = 1 - (1-p)^k
    P(dead) = p^G + (1-p)^G  =  [1 - pass@G]  +  p^G

    A dead group is a UNANIMOUS group and pass@G asks about only ONE of the two
    unanimities. So P(dead) is symmetric under p <-> 1-p and pass@G is not:
    66.3420% dead at p = 0.05 and at p = 0.95, 100% failure-side in the first
    case and 0% in the second. That is why D1's `saturated` and `signal_starved`
    rules each needed a conjunction with the pass rate.

Everything is checked against exact enumeration. The environment, the sequence
table, the exact expectations and `group_advantage` are IMPORTED from D1's
`grpo.py`; the checker masks and the selection law are IMPORTED from D2's
`rlvr.py`. Same TokenEnv(vocab=3, t_max=3), 7 states, 15 terminating sequences,
so every pass@k is a closed form over an enumerable support -- never a sample.

What today adds structurally is a THIRD array beside `gold` and `accept`:

    answer_id[p, y] -- which ANSWER sequence y expresses.

Two sequences can express the same answer. That single addition is what makes
majority voting definable at all, and it is the difference between a metric of
the SEQUENCE distribution (pass@k) and a metric of the ANSWER distribution
(maj@k).

  run everything      ./.venv/bin/python week-06-grpo-rlvr-reasoning/code/sampling.py
  one stage           ...  sampling.py estimator
  a few stages        ...  sampling.py estimator identity
  small/fast fixtures ...  sampling.py --fast
  list the stages     ...  sampling.py --list

RUN IT UNDER  -W error::RuntimeWarning.  D2 produced six unguarded denominators
in one session, every one a silent NaN rather than a crash, one of which left a
stage GREEN while returning [nan]. This file is full of C(n-c,k)/C(n,k), of 0/0
at k > n, and of means over "the accepted samples" -- it is the highest-density
opportunity for that family yet.
"""

from __future__ import annotations

import math
import itertools
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import torch

from grpo import (                      # D1
    TokenEnv,
    SeqTable,
    build_seq_table,
    seq_logprobs_np,
    seq_token_logprobs_torch,
    exact_entropy,
    exact_expectation,
    exact_kl,
    group_advantage,
)
from rlvr import (                      # D2
    dead_fraction_levels,
    g_for_alive,
)


# ---------------------------------------------------------------------------
# The task: three masks, not two (provided)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleTask:
    """`gold[p, y]`      -- is sequence y actually CORRECT for prompt p.
       `accept[p, y]`    -- does the checker SAY it is.       (D2's V)
       `answer_id[p, y]` -- which ANSWER sequence y expresses. (new today)

    `answer_id` is what makes maj@k definable. pass@k and BoN@k are functionals
    of the SEQUENCE distribution; maj@k is a functional of the pushforward of
    that distribution onto answers. They are different objects and they have
    different limits, which is the point of `stage_selectors`.
    """
    env: TokenEnv
    tab: SeqTable
    gold: np.ndarray            # (P, n_seqs) float64 in {0.0, 1.0}
    accept: np.ndarray          # (P, n_seqs) float64 in {0.0, 1.0}
    answer_id: np.ndarray       # (P, n_seqs) int64, values in [0, n_answers)

    @property
    def n_prompts(self) -> int:
        return self.gold.shape[0]

    @property
    def sound(self) -> bool:
        """V subset of C -- no false positives anywhere."""
        return bool(np.all(self.accept * (1.0 - self.gold) == 0.0))


# ---------------------------------------------------------------------------
# Closed forms the day derives (provided -- these are the ground truth)
# ---------------------------------------------------------------------------

def pass_at_k_exact(p: np.ndarray | float, k: int) -> np.ndarray | float:
    """pass@k under the i.i.d. model, EXACTLY: 1 - (1-p)^k.

    `p` is a per-PROBLEM success probability. The benchmark number is the mean of
    THIS over problems -- never this evaluated at the mean p. (Jensen: the two
    differ, and the gap grows with k.)
    """
    return 1.0 - (1.0 - np.asarray(p, dtype=np.float64)) ** k


def dpass_dp(p: np.ndarray | float, k: int) -> np.ndarray | float:
    """d pass@k / dp = k (1-p)^(k-1). THE function of the day.

    Read it as a REWEIGHTING of the problem distribution: the metric's marginal
    valuation of progress on a problem falls off like (1-p)^(k-1), so as k grows
    the aggregate becomes a measurement of the hard tail and of nothing else.
    """
    p = np.asarray(p, dtype=np.float64)
    return k * (1.0 - p) ** (k - 1)


def bon_kl_continuous(n: int) -> float:
    """KL(best-of-n || base) = log n - (n-1)/n, for a CONTINUOUS score.

    Derivation (4 lines, and the assumption is in line 1):
        u = F(r) ~ Uniform(0,1)                      probability integral transform
        best-of-n emits max(u_1..u_n), density n u^(n-1)
        KL = int_0^1 n u^(n-1) log(n u^(n-1)) du
           = log n + (n-1) n int_0^1 u^(n-1) log u du = log n + (n-1) n (-1/n^2)
           = log n - (n-1)/n

    NOTE WHAT IS NOT IN IT: the reward, the policy, the task. It depends on n
    ALONE. The assumption that buys that is CONTINUITY -- i.e. no ties. With a
    discrete score the true KL is strictly smaller, because ties make best-of-n
    less selective than the continuous idealisation. `stage_bon_kl` measures the
    convergence.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return math.log(n) - (n - 1) / n


def temper(probs: np.ndarray, T: float) -> np.ndarray:
    """pi_T(y) proportional to pi(y)^(1/T), renormalised. Operates on the LAST axis.

    THE PROPERTY THAT MATTERS: for every finite T > 0 this preserves the SUPPORT
    exactly. The map T -> supp(pi_T) is constant on (0, inf) and discontinuous at
    0. Therefore pass@infinity is temperature-INVARIANT: temperature cannot make
    an unreachable answer reachable, it can only move the k at which you reach
    it. Truncation (top-p / top-k) can, and does, in the wrong direction.
    """
    if T <= 0.0:
        raise ValueError("temperature must be > 0; T = 0 is a different object (argmax)")
    with np.errstate(divide="ignore"):
        z = np.log(np.asarray(probs, dtype=np.float64)) / T
    z = z - z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    q = np.where(np.asarray(probs) > 0.0, q, 0.0)
    return q / q.sum(axis=-1, keepdims=True)


def top_p_truncate(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus truncation: keep the smallest prefix of the sorted mass reaching
    `top_p`, zero the rest, renormalise. Provided as the NEGATIVE CONTROL for
    `temper` -- this one deletes support, so it lowers pass@infinity."""
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p)
    csum = np.cumsum(p[order])
    keep_n = int(np.searchsorted(csum, top_p) + 1)
    out = np.zeros_like(p)
    out[order[:keep_n]] = p[order[:keep_n]]
    return out / out.sum()


def renormalise(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=-1, keepdims=True)
    if np.any(s <= 0.0):
        raise ValueError("cannot renormalise a zero vector")
    return x / s


# ---------------------------------------------------------------------------
# Config / history
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleCfg:
    steps: int = 250
    group: int = 8
    lr: float = 0.5
    beta: float = 0.02
    seed: int = 0
    flavour: str = "mixed"
    init_scale: float = 0.5
    n_prompts: int = 8
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 64, 256, 1024)


@dataclass
class SampleHistory:
    """The instrument panel. Every field is a LEVEL, recorded per step.

    `diagnose_sampling` reads ONLY this. Note which columns exist and which do
    not: `gold_at` is here because D2's headline was that a hacked run and a
    solved run agree on every column the reward can produce -- but a diagnostic
    that reads it is cheating, so the rules must be stated on `pass_at` and
    `entropy` alone. `gold_at` is for CALIBRATION and for the stage assertions.
    """
    ks: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    step: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pass_at: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (steps, n_k) on ACCEPT
    gold_at: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (steps, n_k) on GOLD
    entropy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kl: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def pass1(self) -> np.ndarray:
        return self.pass_at[:, 0]

    @property
    def passk(self) -> np.ndarray:
        """pass@k at the LARGEST k on the grid -- the coverage column."""
        return self.pass_at[:, -1]

    @property
    def spread(self) -> np.ndarray:
        """pass@k_max - pass@1. The single number separating MASS from SUPPORT."""
        return self.pass_at[:, -1] - self.pass_at[:, 0]


# ---------------------------------------------------------------------------
# Exercises -- the four TODO bodies
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: np.ndarray | int, k: int) -> np.ndarray:
    """The UNBIASED pass@k estimator, per problem. TODO 1.

    `n` samples drawn per problem, `c` of them correct (array over problems, or a
    scalar), budget `k`. Returns a float64 array of the same shape as `c`.

        SHAPES, because this is a LEVEL question and the levels are the whole point:

            n, k    scalars -- YOUR sampling budget. Same for every problem.
            c       one integer PER PROBLEM. This is the only thing that varies.
            return  one estimate PER PROBLEM, shape == c.shape.

            the reported benchmark number is  pass_at_k(n, c, k).mean()  -- and the
            mean goes OUTSIDE. See the "aggregate order matters" check: on the same
            data, mean-then-estimate gives 0.9960 and estimate-then-mean gives 0.6250.

        Vectorising is not about speed. The edge cases below fire on DIFFERENT
        problems inside the same array -- one problem has c = 0 while another has
        k > n-c. Branch with `if` and you need a mask; write the product and both
        fall out of the arithmetic with no branch at all.

        pass@k_hat = 1 - C(n-c, k) / C(n, k)

    WHY IT IS UNBIASED, in one line -- and do NOT expand the binomial sum, it
    needs a combinatorial identity you would have to prove. Compute ONE
    probability TWO ways:

        A = "a uniformly random k-subset S of [n], drawn independently of the
             labels, is entirely wrong"

        condition on S first  : P(A | S = s) = (1-p)^k for EVERY s, so P(A) = (1-p)^k
        condition on labels   : P(A | labels) = C(n-c,k)/C(n,k), so P(A) = E[...]

    Equate. The licence is EXCHANGEABILITY: the labels are exchangeable and S is
    independent of them, so WHICH positions you draw cannot matter. (You will use
    this identical argument again in `selector_curves` to show that a filter's
    output composition does not depend on n. That is not a coincidence.)

    NUMERICS -- write the PRODUCT form, never the ratio of factorials:

        C(n-c,k)/C(n,k) = prod_{i=0}^{k-1} (n-c-i) / (n-i)

    C(200,100) is 9.05e58. The product never forms a number larger than 1.

    THE THREE EDGE CASES, and read them OFF THE PRODUCT rather than special-casing:
        c = 0     -> every factor is 1  -> product 1 -> estimate 0.0
        k > n-c   -> a factor hits 0    -> product 0 -> estimate 1.0
        k > n     -> NOT a value. raise ValueError: you cannot estimate pass@k
                     from fewer than k samples.
    Also raise for k < 1, n < 1, and any c outside [0, n].

    Sanity-check the first two against the WORDS, not the algebra: c = 0 means
    nothing passed, so pass@k must be 0; k > n-c means every k-subset must
    contain a correct sample, so pass@k must be 1. On W6 D3 both came back
    inverted because the intermediate quantity was reported in the final
    quantity's slot -- the twelfth sighting of the wrong-LEVEL family.
    """
    c = np.asarray(c, dtype=np.int64)
    if k > n or k < 1: 
        raise ValueError(k)
    elif n < 1:
        raise ValueError(n)
    elif np.any(c<0) or np.any(c>n):
        raise ValueError(c)
    
    product = np.ones_like(c, dtype=np.float64)
    for i in range(k):
        product *= np.divide(n-c-i, n-i)
    return 1 - product






def pass_at_k_plugin(n: int, c: np.ndarray | int, k: int) -> np.ndarray:
    """The obvious, WRONG estimator: 1 - (1 - c/n)^k. Provided, for the contrast.

    It is f(p_hat) with f concave, so by Jensen E[f(p_hat)] <= f(p) and it
    UNDER-reports. At n = 10, p = 0.1, k = 10 it returns 0.493747 in expectation
    against a truth of 0.651322 -- a bias of -0.157575, i.e. 24.19% of the value
    thrown away. The whole loss sits in the c = 1 and c = 2 rows: you HAVE a
    passing sample and the estimator reports 0.65 instead of 1.
    """
    c = np.asarray(c, dtype=np.float64)
    return 1.0 - (1.0 - c / n) ** k


@dataclass(frozen=True)
class SelectorCurves:
    ks: np.ndarray
    oracle: np.ndarray      # pass@k    -- an ORACLE picks the right sample
    bon: np.ndarray         # BoN@k     -- a FILTER on `accept` picks
    maj: np.ndarray         # maj@k     -- a VOTE over `answer_id` picks


def selector_curves(probs: np.ndarray, gold: np.ndarray, accept: np.ndarray,
                    answer_id: np.ndarray, ks) -> SelectorCurves:
    """The three selectors' EXACT success probabilities at budget k. TODO 2.

    All inputs are (n_seqs,) for ONE prompt; `probs` sums to 1. Returns arrays
    aligned with `ks`. Everything here is exact -- closed form where one exists,
    enumeration otherwise. No sampling.

    (a) ORACLE = pass@k. Success iff at least one of the k samples is gold:

            oracle(k) = 1 - (1 - p_gold)^k,     p_gold = probs @ gold

    (b) BoN = a FILTER. Draw k, keep the accepted ones, emit a uniformly random
        accepted sample; if NONE is accepted, fall back to the first sample.
        Let a = probs @ accept. Then, by the SAME exchangeability argument as
        TODO 1 -- given at least one acceptance, the emitted sample is a uniformly
        random accepted one, so it is distributed as pi CONDITIONED on acceptance,
        independently of k:

            bon(k) = [1 - (1-a)^k] * precision  +  (1-a)^k * P(gold | rejected)

            precision          = P(gold and accept) / a
            P(gold | rejected) = P(gold and not accept) / (1 - a)

        THE RESULT THIS ENCODES: the first factor is the only place k appears.
        A filter's ACCEPTED-OUTPUT COMPOSITION is n-invariant -- raising the
        budget changes how often there is an accepted emission and therefore
        changes unconditional output risk under the fallback. Contrast a RANKER
        (`bon_distribution`), whose unconditional hack probability grows as
        1 - [1 - (1-p)gamma]^n.

        GUARD BOTH DENOMINATORS. a = 0 and a = 1 are both reachable in the
        fixtures, and each kills exactly one of them. This is D2's six-NaNs
        family and the term is multiplied by an exactly-zero factor in each case,
        so a NaN here does not crash -- it poisons the whole curve silently.

    (c) maj = a VOTE. Draw k, push forward onto `answer_id`, emit the plurality
        winner with ties broken uniformly among the tied answers; success iff the
        emitted answer is a gold answer. Exactly, by enumerating the compositions
        of k into the answer counts and weighting by the multinomial:

            maj(k) = sum over count vectors (n_1..n_A) with sum n_j = k
                       multinomial(k; n) * prod_j q_j^{n_j} * (gold share of the argmax set)

        where q_j = total probs mass of the sequences whose answer_id is j.
        (An answer counts as gold iff its sequences are gold; the fixtures keep
        that well-defined by construction.)

        THE RESULT: maj@k -> 1 iff the correct answer is the MODAL one, and
        -> 0 otherwise. Not "stops improving" -- goes to ZERO, and NON-MONOTONE
        on the way. pass@k cannot do that: it is monotone in k by construction.
        This is the sharpest statement of MASS versus SUPPORT in the file.

    Cost note: enumeration is C(k+A-1, A-1) terms. The fixtures use A = 3, so a
    k of a few hundred is fine; vectorise over the composition grid rather than
    looping in Python. Raise for A > 4 with a k over 128 rather than hanging.
    """
    p_gold = (probs * gold).sum()
    ks = np.asarray(list(ks), dtype=np.int64)
    oracle = 1 - (1-p_gold)**ks
    a = (probs * accept).sum()
    if a != 0:
        precision = (probs * accept * gold).sum() / a
    else:
        precision = 0
    if a != 1:
        p_gold_given_reject = (probs * (1-accept) * gold).sum() / (1-a)
    else:
        p_gold_given_reject = 0
    bon = (1 - (1-a)**ks) * precision  +  (1-a)**ks * p_gold_given_reject

    # (c) the VOTE. Push probs forward onto answers: q_j is the mass of answer j.
    # This projection is the whole point -- pass@k lives on SEQUENCES, maj@k lives
    # on ANSWERS, and the map between them is many-to-one and loses information.
    answer_id = np.asarray(answer_id, dtype=np.int64)
    A = int(answer_id.max()) + 1
    q = np.array([probs[answer_id == j].sum() for j in range(A)])

    # an answer is gold iff its live sequences are gold
    gold_answers = {j for j in range(A)
                    if (answer_id == j).any()
                    and probs[answer_id == j].sum() > 0
                    and np.all(gold[(answer_id == j) & (probs > 0)] == 1.0)}

    maj = []
    for k in (int(x) for x in ks):
        if A > 4 and k > 128:
            raise ValueError(f"exact maj@k is {math.comb(k + A - 1, A - 1)} terms "
                             f"at A={A}, k={k} -- refusing to enumerate")
        total = 0.0
        # enumerate compositions of k into A parts; the last part is forced
        for head in itertools.product(*[range(k + 1)] * (A - 1)):
            if sum(head) > k:
                continue
            counts = np.array(list(head) + [k - sum(head)], dtype=np.int64)
            # multinomial weight in LOG space: k is up to ~100 and q_j can be
            # small, so the product underflows long before the sum does
            logw = math.lgamma(k + 1) - sum(math.lgamma(int(x) + 1) for x in counts)
            if np.any((counts > 0) & (q <= 0)):
                continue
            logw += float(np.sum(np.where(counts > 0,
                                          counts * np.log(np.where(q > 0, q, 1.0)),
                                          0.0)))
            top = int(counts.max())
            winners = [j for j in range(A) if counts[j] == top]
            share = sum(1 for j in winners if j in gold_answers) / len(winners)
            if share:
                total += math.exp(logw) * share
        maj.append(total)

    return SelectorCurves(ks, oracle, bon, np.array(maj))



def bon_distribution(probs: np.ndarray, score: np.ndarray,
                     n: int) -> tuple[np.ndarray, float]:
    """The EXACT distribution induced by best-of-n under `score`, and its KL. TODO 3.

    `probs` and `score` are (n_seqs,). Returns `(q, kl)` with `q` the induced
    distribution over the same support and `kl` = KL(q || probs) in NATS.

    THIS IS THE RANKER, and it is the other half of the filter/ranker split.
    Where `selector_curves`'s BoN emits a uniformly random ACCEPTED sample, this
    emits the ARGMAX of a score -- the EXTREME of n draws. The extreme of a noisy
    score sits wherever the noise is largest, which is why best-of-n against a
    reward model over-optimizes and best-of-n against a unit test does not.

    Closed form. Group the sequences into DISTINCT score LEVELS. For level s with
    mass m_s and F_s = total mass of all STRICTLY LOWER levels:

        P(the max attained is level s) = (F_s + m_s)^n - F_s^n

    and given that, the emitted sample is uniform among the draws at level s,
    which by exchangeability is pi conditioned on level s. So:

        q[y] = [ (F_s + m_s)^n - F_s^n ] * probs[y] / m_s      for y in level s

    Then kl = sum_y q[y] log(q[y]/probs[y]), over y with q[y] > 0 only.

    THE TIE HANDLING IS THE WHOLE EXERCISE. If every score is distinct AND the
    support is a fine uniform grid, kl -> log n - (n-1)/n. Ties make best-of-n
    LESS selective, so the true kl is strictly SMALLER. Two exact anchors you can
    check by hand and should assert:
        all scores equal -> q == probs and kl == 0 EXACTLY, for every n
        n == 1           -> q == probs and kl == 0 EXACTLY, for every score

    Guard: drop sequences with probs == 0 before forming levels (m_s = 0 divides),
    and guard the log at q[y] == 0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    n = int(n)
    if n < 1:
        raise ValueError(f"n = {n} < 1")

    live = probs > 0
    q = np.zeros_like(probs)

    # DISTINCT score levels, ascending. Everything sharing a score is ONE level:
    # that is the tie handling, and it is why a discrete score is less selective
    # than a continuous one. F is the mass strictly below the current level.
    F = 0.0
    for s in np.unique(score[live]):
        at = live & (score == s)
        m = float(probs[at].sum())
        if m <= 0.0:
            continue
        # P(max attained is exactly this level) = P(all n <= s) - P(all n < s)
        w = (F + m) ** n - F ** n
        # given that, the emitted draw is pi conditioned on the level
        q[at] = w * probs[at] / m
        F += m

    pos = q > 0
    kl = float(np.sum(q[pos] * np.log(q[pos] / probs[pos])))
    return q, kl


def diagnose_sampling(h: SampleHistory) -> list[tuple[str, str]]:
    """Read the panel, name what is wrong. TODO 4 -- the diagnostic, again yours.

    Return `list[(name, detail)]`, EMPTY on a healthy run. It may read only
    `h.ks`, `h.step`, `h.pass_at`, `h.entropy`, `h.kl` -- NOT `h.gold_at`, which
    is the held-out column a real run does not have at training time.

    Every threshold is keyed on a LEVEL and calibrated from the gap between
    DISTRIBUTIONS over >= 8 seeds -- never between one healthy run and the one
    broken value you happened to see. Air belongs between the healthy MAXIMUM and
    the broken MINIMUM across seeds.

    RULES TO IMPLEMENT

      no_coverage -- pass@k_max at the FINAL step below its level. The answer is
          outside the support: no sampling budget you can afford will find it.
          This is the only failure on this panel that more compute cannot fix,
          and it is invisible to pass@1, which is small on healthy runs too.

      mode_locked -- final entropy below its level. Distinct from no_coverage:
          a run can be locked onto a narrow set that still CONTAINS the answer.
          (Calibration note: the obvious conjunct -- "and the spread
          pass@k_max - pass@1 is small" -- had to be DROPPED. Across 8 seeds the
          locked spread ran 0.123-0.379 and the healthy spread 0.323-0.556. They
          OVERLAP, so the conjunct cannot be thresholded. Entropy alone separates
          0.660 from 0.974. Fewer conditions, because that is what the data
          supports.)

      mass_without_support -- pass@1 rises by more than its level while pass@k_max
          does not rise at all. Not a crash: it says the entire gain was RE-RANKING
          inside problems you already solved. You bought mass, you bought no new
          coverage. This is the day's thesis, as a rule.

    THE CANDIDATE RULE THAT MUST BE REJECTED -- AND WHY IT IS DIFFERENT THIS TIME

      coverage_collapse -- "pass@1 rises while pass@k_max falls below its step-0
          value" -- do NOT implement it. It is the standard claim about RL on
          LLMs: sharpening buys pass@1 by spending pass@k.

          It was measured and it DID NOT HAPPEN. Over 24 configurations (2 task
          flavours x 4 init scales x 3 (lr, beta) settings) and then 8 seeds on
          each of the three fixtures, the minimum value of
              pass@k_max(final) - pass@k_max(0)
          was +0.000e+00. Not small. Never negative, in any run.

          The mechanism is GRPO's own advantage. On a prompt whose gold sequence
          is never sampled, every rollout in the group scores the same, so the
          group-relative advantage is identically zero and the gradient is zero.
          The algorithm cannot push mass off coverage it is not sampling. That is
          yesterday's dead group, read as a SAFETY property instead of as waste.

          SCOPE, because this is the part that matters: a 15-sequence tabular
          softmax with a sound verifier and a group baseline. Nothing here licenses
          the claim at 10^9 parameters. What it does license: the collapse is not
          automatic, it needs a mechanism, and "RL reduces pass@k" is a claim that
          has to be measured on the system in front of you, not assumed.

      Write the reason in a comment. Third day running that a rule gets cut, and
      the three reasons are DIFFERENT, which is the thing to notice:
          D1 cut a rule the INSTRUMENT could not support   (t_max = 3 left no room)
          D2 cut a rule the SEMANTICS did not support      (share -> 0 is health)
          D3 cuts a rule the SYSTEM DOES NOT EXHIBIT       (measured 0 times in 24)
      D3's is the expensive kind: the rule is well-posed, discriminative in
      principle, and would have looked completely reasonable in the code review.
      Only the measurement says it never fires. That is the deliverable for this
      TODO as much as the code is.
    """
    # Thresholds are LEVELS, calibrated from the gap between DISTRIBUTIONS over
    # 8 seeds x 3 fixtures. Air sits between the healthy MINIMUM and the broken
    # MAXIMUM in every case:
    #   final pass@k_max   broken <= 0.3845   healthy >= 1.0000   -> 0.70
    #   final entropy      broken <= 0.6597   healthy >= 0.9739   -> 0.80
    #   delta pass@1       broken >= 0.3697   healthy <= 0.1130   -> 0.25
    PASSK_FLOOR, ENT_LEVEL, D_PASS1, FLAT = 0.70, 0.80, 0.25, 1e-9

    out: list[tuple[str, str]] = []
    if h.pass_at.size == 0 or h.pass_at.shape[0] < 2:
        return out

    p1_0, p1_T = float(h.pass1[0]), float(h.pass1[-1])
    pk_0, pk_T = float(h.passk[0]), float(h.passk[-1])
    kmax = int(h.ks[-1])

    if pk_T < PASSK_FLOOR:
        out.append(("no_coverage",
                    f"pass@{kmax} = {pk_T:.4f} < {PASSK_FLOOR} at the final step -- "
                    f"the answer is outside the support and no budget can buy it"))

    if float(h.entropy[-1]) < ENT_LEVEL:
        # the obvious conjunct "and the spread is small" was DROPPED during
        # calibration: locked spread ran 0.123-0.379 and healthy spread
        # 0.323-0.556. They overlap, so it cannot be thresholded. Entropy alone
        # separates 0.660 from 0.974. Fewer conditions, because that is what the
        # measurement supports.
        out.append(("mode_locked",
                    f"entropy {h.entropy[-1]:.4f} < {ENT_LEVEL} "
                    f"(spread {h.spread[-1]:.4f} is deliberately NOT used)"))

    if (p1_T - p1_0) > D_PASS1 and (pk_T - pk_0) <= FLAT:
        out.append(("mass_without_support",
                    f"pass@1 {p1_0:.4f}->{p1_T:.4f} (+{p1_T - p1_0:.4f}) while "
                    f"pass@{kmax} moved {pk_T - pk_0:+.3e} -- pure re-ranking "
                    f"inside problems that were already solved"))

    # 'coverage_collapse' -- pass@1 rising while pass@k_max falls below its own
    # step-0 value -- is DELIBERATELY NOT IMPLEMENTED. It is the standard claim,
    # it is well-posed, and it would have passed code review. It was measured
    # over 24 configurations and then 8 seeds on each of 3 fixtures, and the
    # minimum of [pass@k_max(final) - pass@k_max(0)] was +0.000e+00. Never
    # negative, anywhere.
    #
    # The mechanism is GRPO's own advantage: on a prompt whose gold is never
    # sampled, every rollout in the group scores the same, so the group-relative
    # advantage is identically zero and so is the gradient. The algorithm cannot
    # push mass off coverage it is not sampling -- D2's dead group, read as a
    # SAFETY property instead of as waste.
    #
    # SCOPE: 15-sequence tabular softmax, sound verifier, group baseline. This
    # licenses nothing at 1e9 parameters. It says the collapse needs a MECHANISM,
    # not that it cannot happen.
    return out


# ---------------------------------------------------------------------------
# Provided: fixtures, correlated sampler, training loop
# ---------------------------------------------------------------------------

def make_sample_task(env: TokenEnv, tab: SeqTable, n_prompts: int,
                     rng: np.random.Generator, flavour: str = "mixed",
                     n_answers: int = 3) -> SampleTask:
    """Fixtures. `flavour` decides gold/accept/answer_id.

      mixed   -- a random ~half of the 15 sequences are gold; accept == gold
      sharp   -- exactly ONE gold sequence per prompt; accept == gold.
                 This is the coverage-collapse fixture: a unique optimum means
                 the policy either finds it or sharpens onto a wrong mode, and
                 across prompts you get both -- pass@1 up, pass@k_max down.
      easy    -- nearly every sequence is gold (13 of 15), so pass@k is saturated
                 at every k from step 0 and the benchmark measures nothing
      leaky   -- gold is `mixed`, but accept adds two false positives per prompt
                 (V \\ C non-empty -- D2's unsoundness, at inference this time)
    """
    n = tab.n_seqs
    gold = np.zeros((n_prompts, n))
    for p in range(n_prompts):
        if flavour == "sharp":
            gold[p, rng.integers(n)] = 1.0
        elif flavour == "easy":
            gold[p, rng.choice(n, size=max(2, n - 2), replace=False)] = 1.0
        else:
            gold[p, rng.choice(n, size=n // 2, replace=False)] = 1.0

    accept = gold.copy()
    if flavour == "leaky":
        for p in range(n_prompts):
            wrong = np.flatnonzero(gold[p] == 0.0)
            accept[p, rng.choice(wrong, size=min(2, len(wrong)), replace=False)] = 1.0

    # answer_id: gold sequences share answer 0; the rest are split over 1..A-1,
    # so "the correct answer" is a single answer and majority voting is well posed.
    answer_id = np.zeros((n_prompts, n), dtype=np.int64)
    for p in range(n_prompts):
        wrong = np.flatnonzero(gold[p] == 0.0)
        answer_id[p, wrong] = 1 + (np.arange(len(wrong)) % max(1, n_answers - 1))
    return SampleTask(env, tab, gold, accept, answer_id)


def correlated_pass_at_k(p_modes: np.ndarray, mode_probs: np.ndarray,
                         k: int) -> float:
    """pass@k when the k samples are NOT independent. Provided.

    The realistic correlation structure for an LLM is a latent APPROACH chosen
    early and then followed: with probability mode_probs[m] the model commits to
    mode m, inside which each sample succeeds independently w.p. p_modes[m].
    Marginally p = sum_m mode_probs[m] p_modes[m], identical to the i.i.d. model
    -- but

        pass@k = 1 - sum_m mode_probs[m] (1 - p_modes[m])^k

    which is BELOW the i.i.d. 1 - (1-p)^k by Jensen (x -> (1-x)^k is convex), and
    the gap grows with k. So the i.i.d. formula OVER-states pass@k, and it does so
    most in exactly the regime people quote it in -- low temperature, where the
    commitment is strongest.
    """
    p_modes = np.asarray(p_modes, dtype=np.float64)
    mode_probs = np.asarray(mode_probs, dtype=np.float64)
    return float(1.0 - (mode_probs * (1.0 - p_modes) ** k).sum())


def _seq_probs(theta_p: np.ndarray, tab: SeqTable) -> np.ndarray:
    q = np.exp(seq_logprobs_np(theta_p, tab))
    return q / q.sum()


def train_and_track(task: SampleTask, cfg: SampleCfg) -> SampleHistory:
    """A GRPO run that records the whole pass@k CURVE at every step. Provided.

    The update is D1's, at mu = 1 -- and D1 measured the clip to be inert there
    (bound-token fraction 0.000e+00 at mu = 1 versus 0.2222 at mu = 8, same lr and
    seed), so the clip is omitted rather than written and never exercised. That
    is a deliberate use of a measured result to delete code.

    Row i is the state after i updates, so row 0 is the reference checkpoint and
    h.kl[0] is EXACTLY 0 -- which is what makes h.<col>[0] usable as a baseline.
    `diagnose_sampling`'s coverage rule compares against row 0 and would be
    destroyed by an off-by-one here (D1 lost this exact thing once).
    """
    tab = task.tab
    P, n = task.n_prompts, tab.n_seqs
    ks = np.asarray(cfg.ks, dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    theta = torch.tensor(
        cfg.init_scale * np.random.default_rng(cfg.seed + 1000).standard_normal(
            (P, task.env.n_states, task.env.vocab)),
        dtype=torch.float64, requires_grad=True)
    theta_ref = theta.detach().clone().numpy()

    mask_all = torch.as_tensor(tab.mask)
    opt = torch.optim.SGD([theta], lr=cfg.lr)
    rec: dict[str, list] = {k: [] for k in ("step", "pass_at", "gold_at", "entropy", "kl")}

    for step in range(cfg.steps):
        th_np = theta.detach().numpy()
        pr = np.stack([_seq_probs(th_np[p], tab) for p in range(P)])        # (P, n)

        # the panel describes the policy that PRODUCED this batch -- read before the update
        p_acc = (pr * task.accept).sum(axis=1)                              # (P,)
        p_gold = (pr * task.gold).sum(axis=1)
        rec["pass_at"].append([float(np.mean(pass_at_k_exact(p_acc, int(k)))) for k in ks])
        rec["gold_at"].append([float(np.mean(pass_at_k_exact(p_gold, int(k)))) for k in ks])
        rec["entropy"].append(float(np.mean([exact_entropy(th_np[p], tab) for p in range(P)])))
        rec["kl"].append(float(np.mean([exact_kl(th_np[p], theta_ref[p], tab) for p in range(P)])))
        rec["step"].append(step)

        idx = np.stack([rng.choice(n, size=cfg.group, p=pr[p]) for p in range(P)])
        rewards = np.take_along_axis(task.accept, idx, axis=1)              # (P, G)
        adv = torch.as_tensor(group_advantage(rewards, "centered"))         # THE BOUNDARY

        idx_t = torch.as_tensor(idx)
        rows = torch.arange(P)[:, None]
        logp = seq_token_logprobs_torch(theta, tab)[rows, idx_t]            # (P, G, T)
        mask = mask_all[idx_t]
        logp_ref = seq_token_logprobs_torch(torch.as_tensor(theta_ref), tab)[rows, idx_t]

        seq_logp = (logp * mask).sum(-1)                                    # (P, G)
        surr = (adv * seq_logp).mean()
        k3 = (torch.exp(logp_ref - logp) - (logp_ref - logp) - 1.0)         # >= 0 pointwise
        kl_term = (k3 * mask).sum() / mask.sum()
        loss = -(surr - cfg.beta * kl_term)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return SampleHistory(
        ks=ks,
        step=np.asarray(rec["step"], dtype=np.float64),
        pass_at=np.asarray(rec["pass_at"], dtype=np.float64),
        gold_at=np.asarray(rec["gold_at"], dtype=np.float64),
        entropy=np.asarray(rec["entropy"], dtype=np.float64),
        kl=np.asarray(rec["kl"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Self-check harness (staged)
# ---------------------------------------------------------------------------

_TALLY = {"pass": 0, "fail": 0}


def _check(name: str, cond: bool, detail: str = "") -> None:
    cond = bool(cond)
    _TALLY["pass" if cond else "fail"] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(name)


class Ctx:
    def __init__(self, fast: bool = False):
        torch.manual_seed(0)
        self.fast = fast
        self.env = TokenEnv(3, 3)
        self.tab = build_seq_table(self.env)
        self.sim = 40_000 if fast else 200_000
        self.steps = 120 if fast else 250
        self.seeds = 4 if fast else 8

    @cached_property
    def theta1(self) -> np.ndarray:
        return 0.9 * np.random.default_rng(7).standard_normal(
            (self.env.n_states, self.env.vocab))

    @cached_property
    def probs1(self) -> np.ndarray:
        return _seq_probs(self.theta1, self.tab)

    def task(self, flavour: str, seed: int = 3, n_prompts: int = 8) -> SampleTask:
        return make_sample_task(self.env, self.tab, n_prompts,
                                np.random.default_rng(seed), flavour)

    def run(self, flavour: str, **kw) -> SampleHistory:
        cfg = SampleCfg(steps=self.steps, **kw)
        return train_and_track(self.task(flavour, seed=cfg.seed + 3), cfg)

    @cached_property
    def h_healthy(self) -> SampleHistory:
        print("  ... training the HEALTHY fixture (broad init, gentle lr, coverage kept)")
        return self.run("mixed", seed=0, lr=0.10, beta=0.02, init_scale=1.0)

    @cached_property
    def h_sharpened(self) -> SampleHistory:
        print("  ... training the SHARPENED fixture (mass bought, no new coverage)")
        return self.run("mixed", seed=0, lr=0.90, beta=0.0, init_scale=1.0)

    @cached_property
    def h_locked(self) -> SampleHistory:
        print("  ... training the LOCKED fixture (born sharp, never had coverage)")
        return self.run("sharp", seed=0, lr=0.10, beta=0.0, init_scale=6.0)


def _fail_fast(ctx: Ctx, *needs: str) -> None:
    """Touch ONLY the TODOs this stage uses, on a 1-element input, so a missing
    implementation surfaces before an expensive fixture is built -- and a stage is
    never blocked by a TODO it does not need (D1's own grading-scaffold bug)."""
    one = np.ones(1)
    if "passk" in needs:
        pass_at_k(1, np.zeros(1, dtype=np.int64), 1)
    if "selectors" in needs:
        selector_curves(one, one, one, np.zeros(1, dtype=np.int64), [1])
    if "bon" in needs:
        bon_distribution(one, one, 1)
    if "diagnose" in needs:
        diagnose_sampling(SampleHistory(ks=np.array([1]), step=np.zeros(1),
                                        pass_at=np.zeros((1, 1)), gold_at=np.zeros((1, 1)),
                                        entropy=np.zeros(1), kl=np.zeros(1)))


# ---- stages ---------------------------------------------------------------

def stage_estimator(ctx: Ctx) -> None:
    """pass_at_k -- unbiased by exchangeability; the plug-in is biased by Jensen"""
    _fail_fast(ctx, "passk")

    # (1) UNBIASEDNESS, against an exact Binomial sum over every c. Not a simulation.
    worst = 0.0
    for n, p, k in [(10, 0.10, 1), (10, 0.10, 5), (10, 0.10, 10),
                    (10, 0.30, 2), (10, 0.30, 5), (10, 0.50, 5),
                    (20, 0.05, 10), (20, 0.20, 4), (16, 0.60, 8)]:
        cs = np.arange(n + 1)
        pmf = np.array([math.comb(n, int(c)) * p ** int(c) * (1 - p) ** (n - int(c))
                        for c in cs])
        est = float(pmf @ pass_at_k(n, cs, k))
        worst = max(worst, abs(est - (1.0 - (1.0 - p) ** k)))
    _check("unbiased against an EXACT Binomial sum (9 points)", worst < 1e-15,
           f"max |E[est] - truth| = {worst:.3e}")

    # (2) the PLUG-IN's bias: negative everywhere, and the headline number
    biases = []
    for n, p, k in [(10, 0.10, 2), (10, 0.10, 5), (10, 0.10, 10),
                    (10, 0.30, 5), (10, 0.50, 5), (20, 0.20, 10)]:
        cs = np.arange(n + 1)
        pmf = np.array([math.comb(n, int(c)) * p ** int(c) * (1 - p) ** (n - int(c))
                        for c in cs])
        biases.append(float(pmf @ pass_at_k_plugin(n, cs, k)) - (1.0 - (1.0 - p) ** k))
    _check("plug-in bias is NEGATIVE at every point (Jensen, f concave)",
           all(b < -1e-9 for b in biases),
           "max = " + f"{max(biases):+.6f}")

    cs = np.arange(11)
    pmf = np.array([math.comb(10, int(c)) * 0.1 ** int(c) * 0.9 ** (10 - int(c)) for c in cs])
    plug = float(pmf @ pass_at_k_plugin(10, cs, 10))
    truth = 1.0 - 0.9 ** 10
    _check("the headline: n=10, p=0.1, k=10 -> plug-in 0.493747 vs truth 0.651322",
           abs(plug - 0.493747) < 5e-7 and abs(truth - 0.651322) < 5e-7,
           f"bias {plug - truth:+.6f} = {100 * (truth - plug) / truth:.2f}% of the truth")

    # (3) at k = n the unbiased estimator IS the indicator "did anything pass"
    ind = pass_at_k(10, np.arange(11), 10)
    _check("k = n collapses the estimator to 1[c >= 1], exactly",
           np.allclose(ind, (np.arange(11) >= 1).astype(float), atol=0, rtol=0),
           f"est(c=0)={ind[0]:.1f}  est(c=1)={ind[1]:.1f}  est(c=10)={ind[10]:.1f}")

    # (4) the three edge cases, read off the product -- and CHECK THEM AGAINST THE WORDS
    _check("c = 0 -> 0.0 (nothing passed, so pass@k is 0)",
           float(pass_at_k(10, 0, 3)) == 0.0)
    _check("k > n-c -> 1.0 (every k-subset must contain a success)",
           float(pass_at_k(10, 9, 3)) == 1.0 and float(pass_at_k(10, 8, 3)) == 1.0)
    for bad, why in [((10, 3, 11), "k > n"), ((10, 3, 0), "k < 1"),
                     ((10, 11, 2), "c > n"), ((10, -1, 2), "c < 0")]:
        try:
            pass_at_k(*bad)
            raised = False
        except (ValueError, IndexError):
            raised = True
        _check(f"raises on {why}", raised)

    # (5) numerical stability where the ratio of factorials cannot be formed.
    #     C(200,100) = 9.05e58: the exact big-int ratio is computable in Python and the
    #     product form must MATCH it. (Asserting "0 < v < 1" here would be vacuous --
    #     1 - 1.1e-59 rounds to exactly 1.0 in float64, which is the correct answer.)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        worst = 0.0
        for n, c, k in [(200, 20, 10), (200, 100, 5), (200, 150, 40), (200, 100, 100)]:
            exact = 1.0 - math.comb(n - c, k) / math.comb(n, k)
            got = float(pass_at_k(n, c, k))
            if not np.isfinite(got):
                raise AssertionError(f"non-finite at n={n} c={c} k={k}")
            worst = max(worst, abs(got - exact))
    _check("matches the exact big-int ratio at n=200 with no overflow",
           worst < 1e-12, f"max dev {worst:.3e} over 4 points")

    # (6) THE LEVEL: mean of the estimator != estimator at the mean c
    cs = np.array([0, 1, 9, 10])
    per_problem = float(np.mean(pass_at_k(10, cs, 5)))
    at_mean_c = float(pass_at_k(10, int(round(cs.mean())), 5))
    _check("mean_i pass@k(c_i) is NOT pass@k(mean c) -- aggregate order matters",
           abs(per_problem - at_mean_c) > 0.05,
           f"per-problem {per_problem:.6f} vs at-mean-c {at_mean_c:.6f}")


def stage_identity(ctx: Ctx) -> None:
    """the dead-group law IS pass@G: P(dead) = [1 - pass@G] + p^G"""
    G = 8
    ps = np.linspace(0.001, 0.999, 501)
    dead = ps ** G + (1 - ps) ** G
    ident = (1.0 - pass_at_k_exact(ps, G)) + ps ** G
    _check("P(dead) = [1 - pass@G] + p^G, exactly, over a 501-point grid",
           float(np.max(np.abs(dead - ident))) < 1e-15,
           f"max dev {np.max(np.abs(dead - ident)):.3e}")

    _check("P(dead) is SYMMETRIC under p <-> 1-p; pass@G is not",
           float(np.max(np.abs(dead - (dead[::-1])))) < 1e-15
           and float(np.max(np.abs(pass_at_k_exact(ps, G) - pass_at_k_exact(1 - ps, G)))) > 0.6,
           "the dead fraction cannot tell you WHICH unanimity you have")

    d05, d95 = 0.05 ** G + 0.95 ** G, 0.95 ** G + 0.05 ** G
    share05 = (1.0 - pass_at_k_exact(0.05, G)) / d05
    share95 = (1.0 - pass_at_k_exact(0.95, G)) / d95
    _check("identical 66.3420% dead at p=0.05 and p=0.95, opposite causes",
           abs(d05 - d95) < 1e-15 and abs(d05 - 0.663420) < 1e-6
           and abs(share05 - 1.0) < 1e-9 and share95 < 1e-9,
           f"failure-side share {100 * share05:.2f}% at p=0.05 vs {100 * share95:.2f}% at p=0.95")

    alive95 = 1.0 - (0.95 ** G + 0.05 ** G)
    _check("alive(p=0.95) EQUALS pass@8(p=0.05) -- the same term, read twice",
           abs(alive95 - pass_at_k_exact(0.05, G)) < 1e-10,
           f"{alive95:.7f} vs {pass_at_k_exact(0.05, G):.7f}")

    _check("consistent with D2's Renyi form on the binary case",
           abs(dead_fraction_levels(np.array([0.05, 0.95]), G) - d05) < 1e-15)
    _check("D1's G for 90% alive at p=0.95 is still 45", g_for_alive(0.95, 0.9) == 45,
           f"ln(0.1)/ln(0.95) = {math.log(0.1) / math.log(0.95):.5f}")


def stage_curvature(ctx: Ctx) -> None:
    """d pass@k/dp = k(1-p)^(k-1) -- the metric REWEIGHTS the problem distribution"""
    ps = np.linspace(0.02, 0.98, 97)
    worst = 0.0
    for k in [1, 2, 4, 8, 16, 64]:
        eps = 1e-6
        fd = (pass_at_k_exact(ps + eps, k) - pass_at_k_exact(ps - eps, k)) / (2 * eps)
        worst = max(worst, float(np.max(np.abs(fd - dpass_dp(ps, k)))))
    _check("analytic derivative vs central finite difference, 6 values of k",
           worst < 1e-6, f"max dev {worst:.3e}")

    hard, easy = 0.05, 0.80
    ratios = {k: float(dpass_dp(hard, k) / dpass_dp(easy, k)) for k in (1, 2, 4, 8, 16)}
    closed = {k: ((1 - hard) / (1 - easy)) ** (k - 1) for k in (1, 2, 4, 8, 16)}
    _check("hard/easy marginal-value ratio is exactly 4.75^(k-1)",
           all(abs(ratios[k] - closed[k]) < 1e-9 * max(1.0, closed[k]) for k in ratios),
           "  ".join(f"k={k}: {ratios[k]:.4g}x" for k in ratios))
    _check("at k=1 the two problems are worth the SAME; by k=8 the hard one is >1e4x",
           abs(ratios[1] - 1.0) < 1e-12 and ratios[8] > 1e4,
           f"k=1: {ratios[1]:.4f}x   k=8: {ratios[8]:.4g}x")

    # the effective problem distribution: normalise the marginal values.
    # At k = 1 the metric weights every problem EQUALLY -- that is what makes it a
    # benchmark. At k = 256 essentially all of its marginal value sits on the two
    # hardest problems, which is what makes it a different measurement.
    bench = np.array([0.05, 0.10, 0.30, 0.60, 0.80, 0.95])
    w1 = dpass_dp(bench, 1); w1 = w1 / w1.sum()
    _check("at k=1 the metric weights every problem EXACTLY equally",
           float(np.max(np.abs(w1 - 1.0 / len(bench)))) < 1e-15,
           f"all weights {w1[0]:.6f}")
    w2 = dpass_dp(bench, 256); w2 = w2 / w2.sum()
    _check("at k=256 the two HARDEST problems carry >99% of the marginal value",
           float(w2[:2].sum()) > 0.99,
           f"{100 * w2[:2].sum():.4f}%, and the easiest carries {w2[-1]:.3e}")


def stage_temperature(ctx: Ctx) -> None:
    """T* rises with k -- and ONLY on a heterogeneous problem set (the scope condition)"""
    rng = np.random.default_rng(0)
    n_ans = 6

    def make(kind):
        L = rng.normal(0, 1, n_ans)
        return L, (int(np.argmax(L)) if kind == "easy" else int(np.argmin(L)))

    het = [make("easy") for _ in range(8)] + [make("hard") for _ in range(8)]
    hom = [make("easy") for _ in range(16)]
    Ts = np.linspace(0.2, 3.0, 141)
    ks = [1, 2, 4, 8, 16, 64]

    def tstar(S, k):
        vals = []
        for T in Ts:
            z = np.stack([L for L, _ in S]) / T
            z = z - z.max(axis=1, keepdims=True)
            q = np.exp(z)
            q = q / q.sum(axis=1, keepdims=True)
            p = np.array([q[i, c] for i, (_, c) in enumerate(S)])
            vals.append(float(np.mean(pass_at_k_exact(p, k))))
        return float(Ts[int(np.argmax(vals))])

    t_het = [tstar(het, k) for k in ks]
    t_hom = [tstar(hom, k) for k in ks]
    _check("HETEROGENEOUS set: T* is non-decreasing in k and MOVES",
           all(b >= a - 1e-12 for a, b in zip(t_het, t_het[1:])) and t_het[-1] > t_het[0] + 0.5,
           "  ".join(f"k={k}:{t:.2f}" for k, t in zip(ks, t_het)))
    _check("HOMOGENEOUS set: T* is CONSTANT in k -- the SCOPE CONDITION",
           max(t_hom) - min(t_hom) < 1e-12,
           "  ".join(f"k={k}:{t:.2f}" for k, t in zip(ks, t_hom)))

    # the proof, on the actual objects: for ONE problem the argmax cannot move.
    # NOTE the instrument's limit -- at k large enough that pass@k saturates to 1.0
    # at EVERY temperature, argmax over a constant array is meaningless. State the
    # claim over the range where the instrument can see it, and assert the REASON
    # (the full ORDERING of temperatures is k-invariant) rather than only the argmax.
    one = [het[-1]]
    p_of_T = np.array([temper(np.exp(one[0][0] - one[0][0].max())
                              / np.exp(one[0][0] - one[0][0].max()).sum(), T)[one[0][1]]
                       for T in Ts])
    _check("for a SINGLE problem T* is k-invariant (monotone maps preserve argmax)",
           len({tstar(one, k) for k in [1, 2, 4, 8, 16, 64]}) == 1,
           f"T* = {tstar(one, 1):.3f} at k = 1, 2, 4, 8, 16, 64")
    orders = {tuple(np.argsort(pass_at_k_exact(p_of_T, k))) for k in [1, 2, 4, 8, 16]}
    _check("and the whole ORDERING of temperatures is k-invariant, not just the argmax",
           len(orders) == 1, "1 - (1-p)^k is strictly increasing in p")

    # temperature preserves support; truncation destroys it
    pr = ctx.probs1
    supp = pr > 0
    _check("temperature preserves the SUPPORT at every finite T > 0",
           all(np.array_equal(temper(pr, T) > 0, supp) for T in (0.05, 0.5, 1.0, 10.0, 50.0)))
    trunc = top_p_truncate(pr, 0.90)
    _check("top-p STRICTLY shrinks the support -> pass@infinity drops discontinuously",
           int((trunc > 0).sum()) < int(supp.sum()),
           f"{int((trunc > 0).sum())} of {int(supp.sum())} sequences survive top-p 0.90")


def stage_correlation(ctx: Ctx) -> None:
    """the i.i.d. formula OVER-states pass@k -- and the gap is UNIMODAL in k, not monotone"""
    modes = np.array([0.60, 0.02])
    w = np.array([0.30, 0.70])
    p_bar = float(w @ modes)

    ks = [1, 2, 4, 8, 16, 64, 256, 1024]
    gaps = []
    for k in ks:
        iid = float(pass_at_k_exact(p_bar, k))
        cor = correlated_pass_at_k(modes, w, k)
        gaps.append(iid - cor)
    _check("k = 1: correlation is INVISIBLE (identical marginal p)", abs(gaps[0]) < 1e-15,
           f"p = {p_bar:.4f} both ways")
    _check("i.i.d. over-states pass@k at every k > 1", all(g > 0 for g in gaps[1:]),
           "  ".join(f"k={k}:{g:+.4f}" for k, g in zip(ks[1:], gaps[1:])))

    # The gap does NOT grow without bound: BOTH curves are bounded by 1 and both
    # converge to 1, so the gap is 0 at k=1, positive in between, and returns to 0.
    # It is UNIMODAL. The peak sits in the range people actually report.
    j = int(np.argmax(gaps))
    rising = all(b > a for a, b in zip(gaps[:j], gaps[1:j + 1]))
    falling = all(b < a for a, b in zip(gaps[j:], gaps[j + 1:]))
    _check("the gap is UNIMODAL in k -- it rises, peaks, and returns to zero",
           rising and falling and 0 < j < len(ks) - 1,
           f"peak {gaps[j]:.4f} at k = {ks[j]}, back to {gaps[-1]:.4f} at k = {ks[-1]}")
    _check("both curves saturate, which is WHY the gap must come back",
           abs(1.0 - float(pass_at_k_exact(p_bar, 1024))) < 1e-6
           and abs(1.0 - correlated_pass_at_k(modes, w, 1024)) < 1e-3,
           f"iid {float(pass_at_k_exact(p_bar, 1024)):.6f}  "
           f"correlated {correlated_pass_at_k(modes, w, 1024):.6f} at k = 1024")

    single = correlated_pass_at_k(np.array([p_bar]), np.array([1.0]), 32)
    _check("NEGATIVE CONTROL: one mode reproduces the i.i.d. formula exactly",
           abs(single - float(pass_at_k_exact(p_bar, 32))) < 1e-15)


def stage_selectors(ctx: Ctx) -> None:
    """three selectors, three limits: SUPPORT vs ACCEPT SET vs MASS"""
    _fail_fast(ctx, "selectors")
    n = ctx.tab.n_seqs
    probs = ctx.probs1

    gold = np.zeros(n); gold[[1, 3, 5]] = 1.0
    acc = gold.copy()
    ans = np.ones(n, dtype=np.int64) * 1
    ans[[1, 3, 5]] = 0
    ans[[0, 2, 4, 6]] = 2

    ks = [1, 2, 4, 8, 16, 64]
    cur = selector_curves(probs, gold, acc, ans, ks)
    p_gold = float(probs @ gold)
    want = pass_at_k_exact(p_gold, np.array(ks))
    _check("oracle IS pass@k, exactly",
           float(np.max(np.abs(cur.oracle - want))) < 1e-15,
           f"p_gold = {p_gold:.6f}, pass@64 = {want[-1]:.6f}")
    _check("an EXACT verifier (V=C) makes BoN@k EQUAL pass@k at every k",
           float(np.max(np.abs(cur.bon - cur.oracle))) < 1e-12)

    # an UNSOUND verifier: BoN saturates BELOW 1 -- at the precision, not at 1
    acc_u = gold.copy(); acc_u[[0, 2]] = 1.0
    cur_u = selector_curves(probs, gold, acc_u, ans, [1, 8, 64, 256, 1024])
    prec = float((probs * gold * acc_u).sum() / (probs @ acc_u))
    _check("an UNSOUND verifier makes BoN@k -> PRECISION, not 1",
           abs(cur_u.bon[-1] - prec) < 1e-6 and cur_u.oracle[-1] > 0.999,
           f"BoN -> {cur_u.bon[-1]:.6f} = precision {prec:.6f}, while pass@k -> {cur_u.oracle[-1]:.6f}")

    # THE MAJORITY-VOTE RESULT: the mode is wrong, so maj@k -> 0, NON-monotonically
    pr = np.zeros(n); pr[1] = 0.35; pr[0] = 0.40; pr[2] = 0.25
    g2 = np.zeros(n); g2[1] = 1.0
    a2 = np.zeros(n, dtype=np.int64); a2[1] = 0; a2[0] = 1; a2[2] = 2
    cur_m = selector_curves(pr, g2, g2, a2, [1, 3, 9, 21, 51, 101])
    _check("maj@1 == p (a vote of one is a sample)", abs(cur_m.maj[0] - 0.35) < 1e-12,
           f"{cur_m.maj[0]:.6f}")
    _check("maj@k is NON-MONOTONE: it rises, then falls",
           cur_m.maj[2] > cur_m.maj[0] and cur_m.maj[-1] < cur_m.maj[2] - 0.05,
           "  ".join(f"k={k}:{v:.4f}" for k, v in zip([1, 3, 9, 21, 51, 101], cur_m.maj)))
    _check("maj@k -> 0 while pass@k -> 1 -- MASS vs SUPPORT, on one fixture",
           cur_m.maj[-1] < 0.30 and cur_m.oracle[-1] > 0.99,
           f"maj@101 = {cur_m.maj[-1]:.4f} vs pass@101 = {cur_m.oracle[-1]:.6f}")

    # and the control: when the mode IS correct, maj@k -> 1 and beats pass@1
    pr2 = np.zeros(n); pr2[1] = 0.45; pr2[0] = 0.35; pr2[2] = 0.20
    cur_c = selector_curves(pr2, g2, g2, a2, [1, 9, 51, 101])
    _check("CONTROL: modal answer correct -> maj@k rises toward 1",
           cur_c.maj[-1] > cur_c.maj[0] + 0.3 and
           all(b >= a - 1e-9 for a, b in zip(cur_c.maj, cur_c.maj[1:])),
           "  ".join(f"k={k}:{v:.4f}" for k, v in zip([1, 9, 51, 101], cur_c.maj)))


def stage_filter_ranker(ctx: Ctx) -> None:
    """a FILTER's conditional composition is invariant; a RANKER compounds flaws"""
    _fail_fast(ctx, "selectors")
    p, alpha, gamma = 0.10, 0.10, 0.01
    law = gamma * (1 - p) / (p * (1 - alpha) + gamma * (1 - p))
    _check("D2's selection law is unchanged at inference: 9.0909%",
           abs(law - 0.0909090909) < 1e-9, f"{100 * law:.4f}%")

    rng = np.random.default_rng(0)
    conditional_comps = []
    for n in [1, 2, 8, 64, 256]:
        N = ctx.sim
        correct = rng.random((N, n)) < p
        acc = np.where(correct, rng.random((N, n)) > alpha, rng.random((N, n)) < gamma)
        any_acc = acc.any(1)
        num = ((~correct) & acc).sum(1)[any_acc]
        den = acc.sum(1)[any_acc]
        if den.size == 0:
            raise AssertionError("no accepted samples -- fixture is degenerate")
        conditional_comps.append(float((num / den).mean()))
    _check("FILTER: P(hack | accepted emission) is INDEPENDENT of n",
           max(abs(c - law) for c in conditional_comps) < 0.006,
           "  ".join(f"n={n}:{100 * c:.4f}%" for n, c in
                    zip([1, 2, 8, 64, 256], conditional_comps)))

    budgets = [1, 8, 64, 256]
    p_obs = p * (1 - alpha) + (1 - p) * gamma
    filter_hack = [(1 - (1 - p_obs) ** n) * law for n in budgets]
    hack_mass = (1 - p) * gamma
    ranker = [1 - (1 - hack_mass) ** n for n in budgets]
    _check("RANKER: P(any top-scored hack) = 1-[1-(1-p)gamma]^n",
           abs(ranker[0] - 0.009) < 1e-12 and abs(ranker[-1] - 0.901177566) < 1e-9,
           "  ".join(f"n={n}:{100 * r:.4f}%" for n, r in zip(budgets, ranker)))
    _check("COMPARABLE risk: equal at n=1, filter 9.09% vs ranker 90.12% at n=256",
           abs(filter_hack[0] - ranker[0]) < 1e-12 and
           abs(filter_hack[-1] - law) < 1e-12 and ranker[-1] > 0.90,
           f"filter/ranker n=1: {100*filter_hack[0]:.2f}%/{100*ranker[0]:.2f}%; "
           f"n=256: {100*filter_hack[-1]:.2f}%/{100*ranker[-1]:.2f}% "
           f"({ranker[-1] / filter_hack[-1]:.2f}x)")


def stage_bon_kl(ctx: Ctx) -> None:
    """best-of-n is a KL knob: log n - (n-1)/n, and ties are what break it"""
    _fail_fast(ctx, "bon")
    probs = ctx.probs1
    n_seq = len(probs)
    score = np.arange(n_seq, dtype=np.float64)          # all distinct

    q1, kl1 = bon_distribution(probs, score, 1)
    _check("n = 1 is the identity: q == probs and KL == 0, EXACTLY",
           float(np.max(np.abs(q1 - probs))) < 1e-15 and abs(kl1) < 1e-15)

    qt, klt = bon_distribution(probs, np.zeros(n_seq), 64)
    _check("all scores TIED: q == probs and KL == 0, exactly, for any n",
           float(np.max(np.abs(qt - probs))) < 1e-14 and abs(klt) < 1e-14)

    kls = []
    for n in [2, 4, 8, 16, 64]:
        q, kl = bon_distribution(probs, score, n)
        _check(f"n={n}: the induced q is a distribution",
               abs(float(q.sum()) - 1.0) < 1e-12 and float(q.min()) >= -1e-15)
        kls.append(kl)
    _check("KL is monotone increasing in n", all(b > a for a, b in zip(kls, kls[1:])),
           "  ".join(f"n={n}:{k:.4f}" for n, k in zip([2, 4, 8, 16, 64], kls)))
    _check("a DISCRETE score gives strictly LESS KL than the continuous law",
           all(k < bon_kl_continuous(n) for k, n in zip(kls, [2, 4, 8, 16, 64])),
           "  ".join(f"n={n}: {k:.4f} < {bon_kl_continuous(n):.4f}"
                     for k, n in zip(kls, [2, 4, 8, 16, 64])))

    # CONVERGENCE to the continuous law as the number of distinct levels grows
    for n in [2, 8, 64]:
        conv = []
        for L in [15, 100, 1000, 20000]:
            u = np.full(L, 1.0 / L)
            q, kl = bon_distribution(u, np.arange(L, dtype=np.float64), n)
            conv.append(abs(kl - bon_kl_continuous(n)))
        _check(f"n={n}: KL -> log n - (n-1)/n as levels -> infinity",
               conv[-1] < conv[0] and conv[-1] < 2e-3,
               f"|err| {conv[0]:.5f} -> {conv[-1]:.2e} at L=20000, target {bon_kl_continuous(n):.6f}")

    _check("the headline nats: 0.1931 / 1.2044 / 3.1745 / 4.5491 at n = 2/8/64/256",
           all(abs(bon_kl_continuous(n) - v) < 1e-4 for n, v in
               [(2, 0.193147), (8, 1.204442), (64, 3.174508), (256, 4.549084)]),
           f"sqrt(KL) at n=64 is {math.sqrt(bon_kl_continuous(64)):.4f}")
    _check("each DOUBLING of n adds about 0.69 nats (it is log 2, asymptotically)",
           abs((bon_kl_continuous(128) - bon_kl_continuous(64)) - 0.6853) < 1e-3,
           f"{bon_kl_continuous(128) - bon_kl_continuous(64):.4f} nats from n=64 to n=128")


def stage_train(ctx: Ctx) -> None:
    """a real GRPO run: does RL ADD capability, or MOVE MASS? the answer is MOVE MASS"""
    base, trained, hs = [], [], []
    for seed in range(3 if ctx.fast else 5):
        h = ctx.run("mixed", seed=seed, lr=0.90, beta=0.0, init_scale=1.0)
        base.append(h.pass_at[0]), trained.append(h.pass_at[-1])
        hs.append(h)
    base_m = np.mean(base, axis=0)
    trained_m = np.mean(trained, axis=0)
    gain_m = trained_m - base_m
    ks = [int(k) for k in hs[0].ks]

    _check("pass@1 RISES: RL moves mass onto the mode", gain_m[0] > 0.02,
           f"pass@1 {base_m[0]:.4f} -> {trained_m[0]:.4f}")

    # the gain PROFILE across k, which is the whole answer to the question in the
    # docstring: the benefit is concentrated at small k and is exactly zero at k_max.
    _check("the gain DECAYS in k and is EXACTLY zero at k_max",
           all(a >= b - 1e-12 for a, b in zip(gain_m, gain_m[1:])) and abs(gain_m[-1]) < 1e-12,
           "  ".join(f"k={k}:{g:+.4f}" for k, g in zip(ks, gain_m)))

    spread = float(np.std([b[-1] for b in base] + [t[-1] for t in trained]))
    _check("pass@k_max is REPORTED with its spread -- a change inside it is a NULL",
           True,
           f"pass@{ks[-1]} {base_m[-1]:.4f} -> {trained_m[-1]:.4f} "
           f"(delta {gain_m[-1]:+.4f}, seed sd {spread:.4f})")

    # the crossover k: the k at which training stops helping. It is not that the
    # gain goes NEGATIVE (it never does -- see stage_diagnose); it is where the
    # gain falls under the seed-to-seed noise floor and the result becomes a NULL.
    #
    # MEASURED: that k is NOT seed-stable. It came out [16, 64, 16, 64, 64] over
    # five seeds -- a 4x range. So the honest report is a BRACKET, not a number.
    # Quoting "the crossover is k = 16" from seed 0 would have been a one-seed
    # artefact, and it is exactly the kind of number that survives into a slide.
    noise = max(float(np.std([t[j] for t in trained])) for j in range(len(ks)))
    cross = [next((k for k, g in zip(ks, t - b) if g <= max(noise, 1e-6)), -1)
             for b, t in zip(base, trained)]
    _check("the crossover k is NOT seed-stable -- report the BRACKET, not a number",
           len(set(cross)) > 1 and all(16 <= c <= 64 for c in cross),
           f"crossover k per seed = {cross} (noise floor {noise:.5f}) -- "
           f"the gain dies between k=16 and k=64, and no single k is quotable")

    _check("pass@1 gain EXCEEDS the pass@k_max gain -- mass bought without support",
           gain_m[0] > gain_m[-1],
           f"delta pass@1 {gain_m[0]:+.4f} vs delta pass@k_max {gain_m[-1]:+.4f} "
           f"-- {ks[-1]}x the samples buys {gain_m[-1]:.0e} of the improvement")


def stage_diagnose(ctx: Ctx) -> None:
    """diagnose_sampling -- silent on healthy, and the STANDARD rule REJECTED"""
    _fail_fast(ctx, "diagnose")
    named = lambda hh: {n for n, _ in diagnose_sampling(hh)}

    _check("SILENT on the healthy fixture", named(ctx.h_healthy) == set(),
           str(sorted(named(ctx.h_healthy))))
    _check("mass_without_support fires on the sharpened fixture",
           "mass_without_support" in named(ctx.h_sharpened),
           str(sorted(named(ctx.h_sharpened))))
    _check("mode_locked and no_coverage BOTH fire on the locked fixture",
           {"mode_locked", "no_coverage"} <= named(ctx.h_locked),
           str(sorted(named(ctx.h_locked))))
    _check("no_coverage does NOT fire on the sharpened run (coverage is intact)",
           "no_coverage" not in named(ctx.h_sharpened))
    _check("mass_without_support does NOT fire on the locked run (pass@1 barely moved)",
           "mass_without_support" not in named(ctx.h_locked))

    # THE REJECTED RULE. Not "no discriminative power" -- it never fires at all.
    # pass@k_max is measured at the final step against its own step-0 value, on
    # every fixture and every seed. If GRPO bought pass@1 by spending pass@k, this
    # number would go negative somewhere. It does not go negative anywhere.
    worst, n_runs, disagree = np.inf, 0, []
    for flav, kw, want in [
            ("mixed", dict(lr=0.10, beta=0.02, init_scale=1.0), set()),
            ("mixed", dict(lr=0.90, beta=0.0, init_scale=1.0), {"mass_without_support"}),
            ("sharp", dict(lr=0.10, beta=0.0, init_scale=6.0), {"mode_locked", "no_coverage"})]:
        for s in range(ctx.seeds):
            h = ctx.run(flav, seed=s, **kw)
            worst = min(worst, float(h.passk[-1] - h.passk[0]))
            n_runs += 1
            got = named(h)
            if (got != set() if not want else not want <= got):
                disagree.append((flav, s, sorted(got)))
    _check(f"REJECTED RULE 'coverage_collapse' never fires: {n_runs} runs, 0 negatives",
           worst >= 0.0,
           f"min delta pass@k_max = {worst:+.3e} -- the dead group is a SAFETY property "
           f"(zero advantage => zero gradient => mass cannot leave unsampled coverage)")
    _check(f"{n_runs}-seed stress: every verdict agrees with its seed-0 verdict",
           not disagree, f"{len(disagree)} disagreements {disagree[:3]}")
    _check("and it is NOT implemented",
           all("coverage_collapse" not in named(h) for h in
               (ctx.h_healthy, ctx.h_sharpened, ctx.h_locked)))
    _check("SCOPE: 15-sequence tabular softmax, sound verifier, group baseline",
           ctx.tab.n_seqs == 15,
           "nothing here licenses the claim at 1e9 parameters -- it says the "
           "collapse needs a MECHANISM, not that it cannot happen")


STAGES = [
    ("estimator", stage_estimator),
    ("identity", stage_identity),
    ("curvature", stage_curvature),
    ("temperature", stage_temperature),
    ("correlation", stage_correlation),
    ("selectors", stage_selectors),
    ("filterranker", stage_filter_ranker),
    ("bonkl", stage_bon_kl),
    ("train", stage_train),
    ("diagnose", stage_diagnose),
]


# ---- runner (copy verbatim; nothing below needs editing) -------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, fn in STAGES:
            print(f"  {name:<16} {fn.__doc__}")
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
    if not todo and not broke:
        print("All invariants passed.")
    return 1 if broke else 0


if __name__ == "__main__":
    raise SystemExit(main())
