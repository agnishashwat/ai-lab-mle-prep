"""W6 D4 -- the reasoning-data curriculum: a controller, and its set point.

THE THESIS
    A curriculum is a CONTROLLER. It has a measured variable (the current
    per-prompt success rate p), a SET POINT, and constraints that make the set
    point infeasible. The set point is not 0.5.

        value(p)  =  progress(p) x metric weight(p)
                  =  p^a (1-p)^b  x  k(1-p)^(k-1)
                  =  p^a (1-p)^(b+k-1)

        p*  =  a / (a + b + k - 1)      ->   1/(k+1)   at a = b = 1

    p* = 1/2 iff a = b + k - 1. For the (1,1) model that is k = 1 ALONE. The folk
    rule "train at 50%" is the k=1 special case of a k-dependent law, still being
    repeated after everyone stopped reporting k=1.

    (a, b) is EMPIRICAL and this file MEASURES it rather than assuming it --
    `stage_exponents`. The `+ k - 1` is not empirical; it is the metric's own
    derivative and it is arithmetic.

THE SECOND THESIS -- the verifier is a MULTIPLIER, not a preference

        M_gold = 2(G-1) p(1-p)                         counterfactual gold mass
        U = E[sum_i A_i s~_i] = M_gold (1-alpha-gamma)              s~ = 2s-1

    so U = J * M_gold with J = 1 - alpha - gamma, YOUDEN'S J. This is not J
    times the observed verifier-advantage mass, which is
    2(G-1)p_obs(1-p_obs). The verifier multiplies the gold-aligned curve by a
    CONSTANT IN p. It therefore does not TILT the curriculum at all -- it cannot
    move the argmax. What it does is FLOOR it:

        P(hack | rewarded) <= tau   <=>   p >= p_floor = R/(1+R),
        R = gamma(1-tau) / [tau(1-alpha)]

    A constraint on the DOMAIN, not a reweighting of the objective. Those are
    different kinds of object, which is why a respectable J cannot substitute for
    a usable floor: gamma = alpha = 0.10 gives J = 0.80 AND p_floor = 0.50.

THE TWO WALLS, AND ONLY ONE IS PURCHASABLE

        dead groups   P(dead) = p^G + (1-p)^G      buy your way out with G
        precision     p >= p_floor(gamma,alpha,tau)   no amount of sampling helps

    and which one binds is a consequence of the alive-TOLERANCE you pick, not a
    fact about the system -- see `stage_floor`. (Learner's finding, W6 D4.)

Everything is checked against exact enumeration. The environment, the sequence
table, `group_advantage` and the exact expectations are IMPORTED from D1's
`grpo.py`; the dead-group law, `g_for_alive` and the selection law from D2's
`rlvr.py`; `pass_at_k_exact` and `dpass_dp` from D3's `sampling.py`. Nothing this
week is re-derived, and every closed form has at least one exact-enumeration
witness.

  run everything      ./.venv/bin/python week-06-grpo-rlvr-reasoning/code/curriculum.py
  one stage           ...  curriculum.py setpoint
  a few stages        ...  curriculum.py signal covariance
  small/fast fixtures ...  curriculum.py --fast
  list the stages     ...  curriculum.py --list

RUN IT UNDER  -W error::RuntimeWarning.  This file divides by sigma on groups
that can be DEAD, by p on prompts that can sit at zero, by p_obs inside the
selection law, and takes means over "the prompts in band" -- which can be EMPTY.
Four separate members of D2's six-unguarded-denominators-in-one-session family,
and the empty band is the one that will actually bite: keeping that set non-empty
is the controller's entire job, and the failure mode under test is that it does
not.
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from grpo import (                        # D1
    TokenEnv,
    SeqTable,
    build_seq_table,
    seq_logprobs_np,
    all_score_functions,
    exact_expectation,
    group_advantage,
    log_softmax_np,
    softmax_np,
)
from rlvr import (                        # D2
    dead_fraction_levels,
    g_for_alive,
    p_observed,
)
from sampling import (                    # D3
    pass_at_k_exact,
    dpass_dp,
)


# ---------------------------------------------------------------------------
# Closed forms the day derives (provided -- these are the ground truth)
# ---------------------------------------------------------------------------

def p_alive(p: np.ndarray | float, G: int) -> np.ndarray | float:
    """P(a group of G i.i.d. Bernoulli(p) draws is NOT unanimous).

    = 1 - p^G - (1-p)^G. This is D1's dead-group law with the sign flipped, and
    D3's identity says it is symmetric under p <-> 1-p while pass@G is not.

    WRITTEN WITH expm1/log1p, NOT as the literal formula. At small p, (1-p)^G is
    1 - Gp + ... and `1 - that` cancels catastrophically: at G = 64 the naive form
    loses ~13 significant figures near p = 1e-4, which the first reference run
    caught as a 6.1e-13 RELATIVE disagreement with exact enumeration -- and the
    enumeration was the accurate one. The closed form is not automatically the
    reference; the better-conditioned expression is.
    """
    p = np.asarray(p, dtype=np.float64)
    small = -np.expm1(G * np.log1p(-p)) - p ** G          # accurate as p -> 0
    big = -np.expm1(G * np.log(np.maximum(p, 1e-300))) - (1.0 - p) ** G
    return np.where(p <= 0.5, small, big)


def dead_split(p: np.ndarray | float, G: int) -> tuple[np.ndarray, np.ndarray]:
    """(failure-side dead, success-side dead) = ((1-p)^G, p^G).

    `dead_frac` alone is symmetric and therefore CANNOT tell you which. The two
    sides need OPPOSITE fixes -- scaffolding vs retirement -- so any panel that
    reports only the sum is one number short of actionable. This is the split
    `diagnose_curriculum.budget_dead` is required to report.
    """
    p = np.asarray(p, dtype=np.float64)
    return (1.0 - p) ** G, p ** G


def signal_center_closed(p: np.ndarray | float, G: int) -> tuple[np.ndarray, np.ndarray]:
    """(E[sum|A|], E[sum A^2]) for CENTERED advantages: (2(G-1)p(1-p), (G-1)p(1-p)).

    Note E[sum A^2] = (G-1) * Var(r) with Var(r) = p(1-p): the (G-1) is Bessel's
    numerator, the SAME object as the (G-1)/G shrinkage in D1's estimator, seen
    from the other side. Both are "a group of G contains G-1 independent
    contrasts".
    """
    p = np.asarray(p, dtype=np.float64)
    v = p * (1.0 - p)
    return 2.0 * (G - 1) * v, (G - 1) * v


def signal_whiten_closed(p: np.ndarray | float, G: int) -> np.ndarray:
    """E[sum A^2] for WHITENED advantages = G * P(alive).

    Because sum_i A_i^2 = G for EVERY alive group whatever c is -- whitening
    deletes the group's difficulty from its gradient magnitude. All the
    p-dependence that survives comes from whether the group is alive at all.
    """
    return G * p_alive(p, G)


def youden(alpha: float, gamma: float) -> float:
    """J = 1 - alpha - gamma = TPR - FPR. The verifier's whole contribution to the
    useful signal, and it is a CONSTANT IN p -- hence a multiplier, not a tilt."""
    return 1.0 - alpha - gamma


def precision_floor(gamma: float, alpha: float, tau: float) -> float:
    """Smallest p at which P(hack | rewarded) <= tau. D2's selection law, solved.

        gamma(1-p) <= tau [ p(1-alpha) + gamma(1-p) ]
        p/(1-p) >= R = gamma(1-tau) / [tau(1-alpha)]
        p_floor = R/(1+R) = gamma(1-tau) / [ tau(1-alpha) + gamma(1-tau) ]

    gamma = 0 gives 0.0 (a SOUND verifier has no floor, at any difficulty -- that
    is what soundness buys you, and it is the only thing that removes this wall).
    tau = 0 with gamma > 0 gives 1.0: demanding zero hacks from an unsound
    verifier is only satisfiable where nothing is hard.
    """
    if not (0.0 <= gamma <= 1.0 and 0.0 <= alpha < 1.0 and 0.0 <= tau <= 1.0):
        raise ValueError((gamma, alpha, tau))
    if gamma == 0.0:
        return 0.0
    num = gamma * (1.0 - tau)
    den = tau * (1.0 - alpha) + num
    return float(num / den) if den > 0.0 else 1.0


def value_curve(p: np.ndarray | float, k: int, a: float = 1.0,
                b: float = 1.0) -> np.ndarray:
    """The rollout's metric-weighted value, p^a (1-p)^(b+k-1), up to a constant.

    The k in dpass_dp is dropped because it does not depend on p and therefore
    cannot move the argmax. Keep it if you want the value in metric units.
    """
    p = np.asarray(p, dtype=np.float64)
    return p ** a * (1.0 - p) ** (b + k - 1.0)


def set_point_analytic(k: int, a: float = 1.0, b: float = 1.0) -> float:
    """p* = a / (a + b + k - 1). Equals 1/2 iff a = b + k - 1."""
    return float(a / (a + b + k - 1.0))


def g_star(k: int, target: float = 0.9, a: float = 1.0, b: float = 1.0) -> int:
    """Exact integer G keeping `target` of groups alive at p*(k). D2's loop, not
    the closed form -- the closed form ln(1-target)/ln(1-p) is single-sided."""
    return g_for_alive(set_point_analytic(k, a, b), target)


def whiten_tilt_limit(G: int) -> float:
    """The factor by which whitening up-weights a near-dead group vs the middle.

    It is a RATIO OF RATIOS, not a limit of one ratio:

        whitened/centered at p -> 0   =  G^2 / (G-1)                (9.143 at G=8)
        whitened/centered at p = 1/2  =  4G(1 - 2^(1-G)) / (G-1)    (4.536 at G=8)
        TILT = quotient               =  G / [4 (1 - 2^(1-G))]  ~  G/4

    The (G-1) cancels, which is why the answer is clean. G/4 = 16 at G = 64:
    not a nudge, a different curriculum.
    """
    return float(G / (4.0 * (1.0 - 2.0 ** (1 - G))))


def contamination_inflation(c: float, p_true: float, k: int) -> float:
    """Points added to mean pass@k by a contaminated fraction c (memorised => 1.0).

        c * 1 + (1-c) * pass@k(p_true)  -  pass@k(p_true)  =  c (1 - pass@k(p_true))

    DECAYS in k, to zero. Same shape as D3's measured RL gain (+0.4691 at k=1 ->
    +0.0000 at k=1024) and for the same reason: a saturating metric cannot see a
    change in something already saturated.
    """
    return float(c * (1.0 - pass_at_k_exact(p_true, k)))


def logit(p: np.ndarray | float) -> np.ndarray:
    """log-odds, guarded at both ends. The controller's coordinate: the 50%-of-peak
    band narrows 19.07x in p from k=1 to k=64 but only 1.43x in logit-p, so
    controlling in p makes the problem 13.3x harder than it is."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Task / config / history (provided)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumTask:
    """P prompts spanning a range of DIFFICULTY, each with its own theta block.

    `gold[p, y]` -- is sequence y correct for prompt p (the truth).
    `accept[p, y]` -- does the verifier say so (D2's V). Equal to `gold` for a
    sound-and-complete checker; `stage_covariance` builds the noisy ones.
    `contaminated[p]` -- was prompt p memorised. Used only by `stage_contamination`;
    it is NOT visible to the trainer, which is the whole problem.
    """
    env: TokenEnv
    tab: SeqTable
    gold: np.ndarray             # (P, n_seqs) float64 in {0,1}
    accept: np.ndarray           # (P, n_seqs) float64 in {0,1}
    theta0: np.ndarray           # (P, n_states, vocab) float64
    contaminated: np.ndarray     # (P,) bool

    @property
    def n_prompts(self) -> int:
        return self.gold.shape[0]


@dataclass(frozen=True)
class CurrCfg:
    steps: int = 150
    group: int = 8               # G
    lr: float = 0.6
    n_active: int = 6            # prompts rolled out per step -- THE BUDGET
    seed: int = 0
    adv_mode: str = "centered"
    ema: float = 0.6             # p_hat smoothing on the FREE sensor c/G
    ks: tuple[int, ...] = (1, 8)


@dataclass
class CurrHistory:
    """The instrument panel. `diagnose_curriculum` reads ONLY this.

    Every field is a LEVEL recorded per step. `p_true` is the exact per-prompt
    pass rate by enumeration -- available here because the toy is enumerable, and
    deliberately NOT given to the diagnostic, which sees only `p_hat`: a real
    controller has the noisy estimate and nothing else.
    """
    step: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_true: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (T, P) EXACT
    p_hat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))    # (T, P) what the loop sees
    active: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (T, n_active) idx
    dead_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dead_fail: np.ndarray = field(default_factory=lambda: np.zeros(0))     # all-0 groups
    dead_succ: np.ndarray = field(default_factory=lambda: np.zeros(0))     # all-1 groups
    in_band: np.ndarray = field(default_factory=lambda: np.zeros(0))
    signal: np.ndarray = field(default_factory=lambda: np.zeros(0))        # mean |A| per ROLLOUT
    passk: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))    # (T, len(ks))
    ks: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_prompts(self) -> int:
        return self.p_true.shape[1]


# ---------------------------------------------------------------------------
# Exercises -- the four TODO bodies
# ---------------------------------------------------------------------------
def combination(c, G):
    combs = np.ones_like(c, dtype=np.float64)
    for i in range(1, int(G//2)+1):
        combs[i] = (combs[i-1] * (G-i+1))/i
    
    for i in range(int((G+1)//2)):
        combs[-i-1] = combs[i]
    
    return combs




def signal_curve(p: np.ndarray | float, G: int,
                 mode: str = "center") -> tuple[np.ndarray, np.ndarray]:
    """Exact (E[sum_i |A_i|], E[sum_i A_i^2]) over c ~ Binomial(G, p). TODO 1.

    Returns two float64 arrays shaped like `p`. Enumerate c = 0..G and weight by
    the Binomial pmf -- this is a closed-form expectation over G+1 terms, NOT a
    simulation. Vectorise over `p`: `p` is a grid, `G` is a scalar.

        SHAPES, because this is a LEVEL question:
            p       (n,) or scalar -- the DIFFICULTY axis, the thing that varies
            G       scalar         -- YOUR group size, fixed across the grid
            return  two arrays, each shape == p.shape

    Inside a group with c successes out of G, the rewards are c ones and G-c
    zeros, so everything is a function of c alone:

        centered   A_i = r_i - c/G
                   sum|A_i| = 2 c (G-c) / G          sum A_i^2 = c (G-c) / G
        whitened   A_i = (r_i - c/G) / sigma,  sigma^2 = c(G-c)/G^2
                   sum|A_i| = 2 sqrt(c (G-c))        sum A_i^2 = G

    Derive those four rather than trusting them; each is two lines, and the
    `sum A_i^2 = G` one is the day's headline (whitening deletes difficulty from
    the gradient magnitude, so ALL of the p-dependence that survives is "is the
    group alive at all").

    THE TRAP, and it is the cleanest instance of the family you will get:
    c = 0 and c = G are DEAD groups, sigma = 0, and the whitened formulas are 0/0.
    A dead group contributes EXACTLY ZERO -- not a small number, not a nan. The
    fix is THE RIGHT DENOMINATOR, NOT AN EPSILON ON THE WRONG ONE: those two terms
    are not "a division that needs guarding", they are terms that are not in the
    sum. Skip them. An epsilon here would silently make E[sum A^2] depend on your
    choice of epsilon, and the check that catches that is `signal_whiten_closed`,
    which is exact.
    (`np.where` evaluates BOTH branches, so it does not save you either -- it
    computes the nan and then discards it, which under -W error::RuntimeWarning is
    still a crash. Sanitize the INPUTS, or index, or start the loop at c = 1.)

    Contract:
        mode "center"  matches signal_center_closed  to < 1e-15
        mode "whiten"  second output matches G * p_alive(p, G) to < 1e-15
        both outputs are EXACTLY symmetric under p -> 1-p (bitwise on a mirrored
            grid is too strong; assert < 1e-15)
        p = 0 and p = 1 give exactly (0.0, 0.0) under BOTH modes
        anything else in `mode` raises ValueError
    """
    p = np.asarray(p, dtype=np.float64)
    c = np.arange(G+1)
    combs = combination(c, G)
    prob_c = (p[..., None] ** c) * (1 - p[..., None])**(G-c) * combs[None, :]

    if mode == 'center':
        sum_abs = np.asarray(2 * c * (G-c) / G, dtype=np.float64)
        sum_square = sum_abs/2
    elif mode == 'whiten':
        sum_abs = np.asarray(2 * (c * (G-c)) ** 0.5, dtype=np.float64)
        sum_square = G * np.ones_like(c, dtype=np.float64)
        sum_square[0] = 0.0
        sum_square[-1] = 0.0
    else:
        raise ValueError(mode)

    sum_abs_weighted = prob_c @ sum_abs
    sum_square_weighted = prob_c @ sum_square
    return (sum_abs_weighted, sum_square_weighted)


def useful_signal(p: np.ndarray | float, alpha: float, gamma: float,
                  G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The part of the advantage mass that points at TRUTH. TODO 2.

    Returns (U, mass, ratio), each shaped like `p`:

        U     = E[ sum_i A_i * s~_i ]     s~_i = 2 s_i - 1 in {-1, +1}
        mass  = E[ sum_i |A_i| ]          under the VERIFIER's labels
        ratio = U / mass                  guarded where mass == 0

    `A_i` is the centered advantage of the VERIFIER's reward r_i -- that is all
    the trainer ever sees, so `mass` is evaluated at p_obs, NOT at p. `s_i` is
    true success, which it never sees.

    THE +/-1 CONVENTION IS NOT COSMETIC, so make the choice deliberately. With
    s in {0,1} the failures contribute nothing to the inner product even though
    their negative advantages are doing real work (pushing DOWN wrong answers),
    and you get exactly half. With s~ = 2s-1 you get

        U = 2(G-1) p(1-p) (1 - alpha - gamma)

    (sum_i A_i = 0 identically for centered advantages, so the two conventions
    differ by the constant 2 and by nothing else: same argmax, same zeros, same
    sign. Say which one you are quoting.)

    TWO DIFFERENT CLAIMS LIVE HERE AND THEY HAVE DIFFERENT DENOMINATORS. Getting
    them the same way round is the point of the stage:

      (i) THE VERIFIER IS A MULTIPLIER. J enters U as a CONSTANT IN p, so it
          scales the whole curve and cannot move its argmax: argmax_p U = 1/2 for
          every (alpha, gamma) with J > 0. A verifier does not TILT a curriculum.

     (ii) AND YET the USEFUL FRACTION of the mass you actually spend collapses at
          low p, because at low p most of the reward variance is false positives:

              ratio = p(1-p) J / [ p_obs (1 - p_obs) ]      p_obs = p(1-a) + (1-p)g

          gamma = 0 (SOUND):  ratio -> 1 as p -> 0, at any alpha
          gamma > 0        :  ratio ~ [J / (gamma(1-gamma))] * p  ->  0, LINEARLY

          At alpha = 0.10, gamma = 0.01 that slope is 89.90, giving
          0.8973 / 0.8980 / 0.6454 / 0.3109 / 0.0825 at p = 0.5/0.1/0.02/0.005/0.001.
          SOUNDNESS IS EXACTLY THE PROPERTY THAT STOPS THE COLLAPSE, and this
          ratio is the mechanism underneath the precision floor. The noise floor
          gamma(1-gamma) does not shrink with p, so you cannot raise the learning
          rate out of it -- the LR scales numerator and denominator alike.

    COMPUTE IT BY EXACT ENUMERATION over the joint, not by Monte Carlo. Each
    sample is one of four cells:

        (s=1, r=1)  p(1-alpha)      (s=1, r=0)  p alpha
        (s=0, r=1)  (1-p) gamma     (s=0, r=0)  (1-p)(1-gamma)

    so a group is a Multinomial over those four cells; enumerate it, or -- much
    shorter -- note that A_i depends on the group only through the count of r = 1
    and use linearity. Either is fine; the second is four lines.

    THE LEVEL TRAP, eleventh sighting: U is a PER-GROUP expectation at a single p.
    The benchmark quantity is the mean over PROMPTS, and mean-then-evaluate is not
    evaluate-then-mean because U is not linear in p. Do not average p first.

    Contract:
        U = 2(G-1) p(1-p) (1-alpha-gamma) to < 1e-12 on a (p, alpha, gamma) grid
        alpha + gamma == 1  =>  U == 0 IDENTICALLY IN p (the negative control;
            it must be exactly 0.0, not "small" -- a coin-flip verifier produces a
            perfectly healthy |A| and a perfectly healthy dead_frac and learns
            nothing, forever, and that is the failure this constant detects)
        alpha + gamma  > 1  =>  U < 0 everywhere in (0,1)
        argmax_p U == 1/2 for EVERY (alpha, gamma) with J > 0
        ratio -> 1 as p -> 0 when gamma == 0; ratio -> 0 linearly when gamma > 0,
            with measured slope J/(gamma(1-gamma))
        U == 0 EXACTLY at p in {0, 1}, so ratio == 0.0 there -- but mass does NOT
            vanish unless the verifier is perfect: at p = 1, alpha = 0.1 the group
            is alive at p_obs = 0.9 and the trainer sees a healthy pile of mass,
            all of it noise. Guard the division; return 0.0, never nan.
    """
    p_obs = p * (1-alpha) + (1-p) * gamma
    mass, _ = signal_curve(p_obs, G, mode='center')
    # U = 2 * (G-1) * p * (1-p) * (1-alpha-gamma)
    m1 = p * (1-alpha) - (1-p) * gamma
    U = (G-1) * (m1 - p_obs * (2 * p - 1))

    mass_safe = np.where(mass > 0, mass, 1)
    ratio = np.where(mass>0, U/mass_safe, 0)
    return (U, mass, ratio)


@dataclass(frozen=True)
class SetPoint:
    """What a curriculum controller is actually configured with."""
    metric: float           # a/(a+b+k-1) -- what the METRIC wants
    floor: float            # precision floor -- what the VERIFIER allows
    whitened: float         # argmax under /std, numerically (no closed form)
    p_star: float           # the one you would actually target
    binding: str            # "metric" | "precision" | "whiten"
    band: tuple[float, float]   # 50%-of-peak band of the UNCONSTRAINED value curve

def whitened_argmax(k: int, G: int, rounds: int = 5, npts: int = 4001) -> float:
    """argmax_p P(alive; G) * (1-p)^(k-1). NUMERICAL -- this root has no closed form.

    Three deliberate choices, each of which is the day's lesson in miniature:

    THE LOGIT AXIS. Optimise in z = log(p/(1-p)), not in p. The 50%-of-peak band
    narrows 19.07x in p between k=1 and k=64 but only 1.43x in log-odds, so a grid
    that is well-conditioned at k=1 is useless at k=64 in p and fine in z.

    LOGS, NOT THE PRODUCT. At k=64 the raw product is ~1e-30 near the optimum.
    log P(alive) + (k-1) log1p(-p) is the same argmax, evaluated in a range float64
    can actually resolve.

    P_ALIVE, NOT `1 - p**G - (1-p)**G`. The literal form cancels catastrophically:
    at G=64, p=1e-4 it disagrees with the well-conditioned form in the 13th
    significant figure, and we are refining to 1e-6.

    COARSE-TO-FINE, NOT BISECTION ON THE DERIVATIVE. Verified unimodal (exactly one
    interior sign change of df on a 200k-point grid) for every (G, k) the stages
    exercise -- EXCEPT (G=64, k=1), where the objective is a flat plateau and the
    argmax is not identified at all. That case is deliberately absent from
    `stage_whiten`'s sweep; see the "NOT IDENTIFIED" check. A bracketing method
    would have returned a confident number there.
    """
    lo, hi = -30.0, 30.0
    best = 0.5
    for _ in range(rounds):
        z = np.linspace(lo, hi, npts)
        p = 1.0 / (1.0 + np.exp(-z))
        f = np.log(np.maximum(p_alive(p, G), 1e-300)) + (k - 1) * np.log1p(-p)
        i = int(np.argmax(f))
        best = float(p[i])
        lo, hi = float(z[max(i - 1, 0)]), float(z[min(i + 1, npts - 1)])
    return best

def find_range_ascending(low, high, threshold, k, a, b, eps):   
    while high > low + eps:
        mid = low + (high-low)/2
        midval = value_curve(mid, k, a, b)
        if midval < threshold:
            low = mid
        elif midval > threshold:
            high = mid
        else: return mid
    return mid

def find_range_descending(low, high, threshold, k, a, b, eps):
    while high > low + eps:
        mid = low + (high-low)/2
        midval = value_curve(mid, k, a, b)
        if midval < threshold:
            high = mid
        elif midval > threshold:
            low = mid
        else: return mid
    return mid


def set_point(k: int, a: float = 1.0, b: float = 1.0, G: int = 8,
              alpha: float = 0.10, gamma: float = 0.01, tau: float = 0.10,
              mode: str = "center") -> SetPoint:
    """The CONSTRAINED set point, and -- the actual deliverable -- WHICH CONSTRAINT
    BINDS. TODO 3.

    Three candidate difficulties, and they are three different KINDS of object:

        metric    a/(a+b+k-1)               an ARGMAX of the objective
        floor     precision_floor(...)      a CONSTRAINT ON THE DOMAIN
        whitened  argmax_p P(alive) (1-p)^(k-1)    a DIFFERENT objective, because
                  /std replaced p(1-p) with P(alive) behind your back

    p_star is the metric optimum pushed up to the floor when the floor is higher:
    the value curve is unimodal, so the constrained argmax is max(argmax, floor).
    Set `binding` to the reason. When mode == "whiten" the objective IS the
    whitened one, so compare the floor against `whitened` instead and report
    "whiten" when the whitened argmax is what you returned unmodified.

    THE CROSSOVER MUST BE SOLVED, NOT SCANNED. a/(a+b+k-1) = p_floor gives

        k = a/p_floor - a - b + 1

    and for (a,b) = (1,1), gamma = 0.01, alpha = 0.10, tau = 0.10 that is EXACTLY
    10.0. A grid scan reports "somewhere between 9 and 11" and that is a different
    answer to the same question -- D3's bracket lesson, where the reported
    crossover moved when the grid moved.

    `whitened` has no closed form. Get it numerically and say so; refine to at
    least 1e-6 (a two-stage grid is fine, or golden section on the log-odds, where
    the curve is far better conditioned).

    `band` is the 50%-of-peak interval of the UNCONSTRAINED value curve. Report it
    because the interesting fact is not its centre:

        hi - hi/lo   k=1: 0.70711 / 5.83     k=64: 0.03709 / 11.33
        so the band narrows 19.07x in p and only 1.43x in LOG-ODDS.

    That is the argument for controlling in logit space, and it is worth more than
    the set point itself: a tolerance of +/- 0.05 in p is WIDER THAN THE ENTIRE
    LEGAL BAND at k = 64.

    Contract:
        analytic `metric` == grid argmax at 16 (a,b,k) points, to grid resolution
        `metric` strictly decreasing in k, and == 0.5 iff a == b + k - 1
            (check (a,b) = (2,1) at k = 2, which also gives 0.5 -- that is the
            check that proves you solved it instead of pattern-matching 1/(k+1))
        `binding` flips "metric" -> "precision" at k = 10 EXACTLY for the default
            (gamma, alpha, tau), by the solved formula
        at G = 2, whitened argmax == metric argmax (P(alive) = 2p(1-p): whitening
            and centering are the SAME objective up to a constant -- the negative
            control, and if your whitened optimiser disagrees here it is broken)
        raises ValueError for k < 1, a <= 0, b <= 0, G < 2, unknown mode
    """
    if k < 1:
        raise ValueError(k)

    if a <= 0:
        raise ValueError(a)

    if b <= 0:
        raise ValueError(b)
    
    if G < 2:
        raise ValueError(G)

    metric = a/(a+b+k-1)
    p_floor = precision_floor(gamma, alpha, tau)
    whitened = whitened_argmax(k, G)
    if mode == 'center':
        if metric >= p_floor:
            p_star = metric
            binding = "metric" # "metric" | "precision" | "whiten"
        else:
            p_star = p_floor
            binding = "precision"
    elif mode == 'whiten':
        if whitened >= p_floor:
            p_star = whitened
            binding = "whiten" # "metric" | "precision" | "whiten"
        else:
            p_star = p_floor
            binding = "precision"
    else:
        raise ValueError(mode)
    
    max_objective = value_curve(metric, k, a, b)
    eps = 1e-8
    left_lim = find_range_ascending(0, metric, max_objective/2, k, a, b, eps)
    right_lim = find_range_descending(metric, 1, max_objective/2, k, a, b, eps)

    setpoint = SetPoint(metric, p_floor, whitened, p_star, binding, (left_lim, right_lim))
    return setpoint

# --- diagnose_curriculum thresholds ----------------------------------------
# Every one is a LEVEL, calibrated from the gap between DISTRIBUTIONS over 5
# paired seeds per fixture, at the 150-step regime, with the measured ranges in
# the comment. A threshold with no air on both sides is a coin flip wearing a
# number, so the ranges are the deliverable and the constant is a consequence.
CLIFF_PIN = 0.05      # p_hat level that counts as "pinned at zero"
CLIFF_MOVE = 0.05     # |tail - head| that counts as "not moving"
CLIFF_SHARE = 0.50    # healthy 0.06..0.13, coin_flip 0.06..0.25 | too_hard 0.94..1.00
ALIVE_FALL = 0.40     # healthy 0.04..0.24, uniform 0.07..0.22   | open_loop 0.59..0.67
DEAD_LEVEL = 0.60     # healthy 0.36..0.54, uniform 0.37..0.49   | open_loop 0.67..0.78 (thin)
FLOOR_SHARE = 0.125   # healthy 0.01..0.07, uniform 0.02..0.04   | too_hard 0.83..1.00


def _windows(T: int) -> tuple[slice, slice]:
    """(head, tail) for CHANGE statistics on p_hat, with the EMA burn-in excluded.

    p_hat is an EMA (0.6) from a 0.5 prior, so the first fifth of the run is the
    SENSOR CONVERGING, not the policy moving: with head = step 0 the prompts whose
    true p never leaves 1e-4 read as having "moved" 0.047..0.276 and `cliff` cannot
    fire on the fixture built for it. The head therefore starts AFTER the burn-in.

    `dead_frac` is a direct per-step count, not an EMA, so it has no burn-in and
    `saturated_band` measures its fall from step 0 -- excluding the burn-in there
    would throw away the part of the decay that carries the discrimination.
    """
    q = T // 5
    return slice(q, 2 * q), slice(T - q, T)


def diagnose_curriculum(h: CurrHistory, *, gamma: float = 0.01,
                        alpha: float = 0.10, tau: float = 0.10,
                        G: int = 8) -> list[tuple[str, str]]:
    """Read the panel, name what is wrong. TODO 4 -- yours for the fourth day.

    Returns [(name, detail)], SILENT (empty) on a healthy run. It sees `h` and the
    verifier's rates and NOTHING else -- no task, no gold, no p_true. A rule you
    cannot state in these columns is a rule you cannot ship.

    Rules to implement:

      cliff              a share of prompts pinned at p_hat ~ 0 AND not moving
                         over the run. THE CONJUNCTION IS THE RULE: hard prompts
                         that are MOVING are the entire point of the exercise, and
                         a level-only version of this rule fires on every healthy
                         curriculum run that starts hard.

      saturated_band     the share of prompts above p_hi RISING and the alive
                         fraction FALLING. This is open-loop decay: a static
                         curriculum solves its own band and then trains on nothing.

      floor_violated     the realised in-band p BELOW precision_floor(gamma,alpha,
                         tau). A LEVEL, not a change (W5 D5) -- "we are training
                         where the verifier cannot be trusted" is true of a
                         snapshot, and a delta-based version cannot say it.

      budget_dead        the share of rollouts in dead groups above a level, with
                         THE TWO SIDES REPORTED SEPARATELY in `detail`. D3's
                         identity says dead_frac alone cannot tell you which side,
                         and the two need OPPOSITE fixes. A rule that reports the
                         sum has answered a question nobody can act on.

      AND ONE CANDIDATE MUST BE REJECTED, with a MEASURED reason. The candidate is
      `mean_p_rising`. The SUSPECTED reason was REDUNDANCY: that it would fire on
      exactly the same fixtures as `saturated_band`. MEASURE THE FIRE-SETS; do not
      assume.

      MEASURED, AND THE PREDICTION WAS WRONG. The rise in the share of prompts
      above p_hi is +0.796..+0.938 on the HEALTHY closed-loop fixture and only
      +0.373..+0.410 on the sick open-loop one. It does not merely fire on healthy
      runs -- it fires TWICE AS HARD on them, because rising p IS learning. The
      cut reason is therefore FIRES-ON-HEALTHY, not redundancy.
      Two consequences, and the second is the useful one:
        * the `rise` conjunct inside `saturated_band` is dead weight and was
          removed; the alive-fraction FALL carries all the discrimination
          (healthy -0.319..+0.244, sick +0.594..+0.667)
        * the redundancy the prediction expected was real but in the wrong place:
          the collinearity check found `budget_dead` and `saturated_band` with
          identical fire-sets -- which turned out to be a FIXTURE-SET defect, not
          a rule defect. Three fixtures cannot separate four rules.
      Four days, four cut rules, four different reasons: instrument (D1) /
      semantics (D2) / the system does not exhibit it (D3) / IT FIRES ON HEALTHY,
      HARDER THAN ON SICK (D4).

    CALIBRATION, non-negotiable and inherited:
      * thresholds keyed on a LEVEL, and calibrated from the gap between
        DISTRIBUTIONS over >= 5 PAIRED SEEDS per fixture, not from one run
      * air on BOTH sides: a threshold with the healthy runs at 0.49 and the sick
        runs at 0.51 is a coin flip wearing a number
      * every threshold a module constant with the measured gap in its comment.
        The two that matter here:
            ALIVE_FALL = 0.40   healthy -0.319..0.244 | sick 0.594..0.667
            DEAD_LEVEL = 0.60   healthy 0.36..0.54    | sick 0.67..0.78  (thin)
      * A THRESHOLD IS CALIBRATED FOR A REGIME. At 150 steps the alive-fall gap is
        0.35 wide; at 60 steps it is 0.014 wide and no threshold separates. The
        panel runs at its calibrated regime even under --fast; you may re-derive
        thresholds for a faster one, but you may not borrow them.
      * A CHANGE STATISTIC MUST NOT OVERLAP THE ESTIMATOR'S BURN-IN. p_hat is an
        EMA from a prior, so a head window at step 0 measures the SENSOR
        converging: prompts whose true p never left 1e-4 read as having moved
        0.047..0.276, and `cliff` could not fire on the fixture built for it.
        Start the head window after burn-in.
      * silent on healthy is a CHECK, not an aspiration -- `stage_diagnose`
        asserts it on 5 seeds

    Read `h.p_hat`, NOT `h.p_true`. The exact rate is in the panel because the toy
    is enumerable; a real controller has c/G and an EMA, which is what `p_hat` is.
    Using `p_true` would produce a diagnostic that cannot be ported, which is the
    same defect as a rule stated in terms of the task.
    """
    out: list[tuple[str, str]] = []
    T = int(len(h.step))
    if T < 5:            # below the calibrated regime -- refuse, do not guess
        return out
    head, tail = _windows(T)

    # cliff -- LEVEL and CHANGE conjoined. The level-only version fires on every
    # healthy curriculum that starts hard, which is the whole exercise.
    pinned = h.p_hat[tail].mean(0) < CLIFF_PIN
    still = np.abs(h.p_hat[tail].mean(0) - h.p_hat[head].mean(0)) < CLIFF_MOVE
    stuck = float(np.mean(pinned & still))
    if stuck > CLIFF_SHARE:
        out.append(("cliff", f"{stuck:.0%} of prompts pinned below p_hat "
                             f"{CLIFF_PIN} AND unmoved (<{CLIFF_MOVE}) between "
                             f"steps {head.start}-{head.stop} and {tail.start}-{T}"))

    # saturated_band -- OPEN-LOOP DECAY. The `rise` conjunct was measured and cut:
    # the share above p_hi rises +0.796..+0.938 on the HEALTHY fixture and only
    # +0.373..+0.410 on the sick one, so it discriminates backwards. The
    # alive-fraction FALL carries all of it.
    alive = 1.0 - h.dead_frac
    fall = float(alive[:T // 5].mean() - alive[tail].mean())
    if fall > ALIVE_FALL:
        out.append(("saturated_band", f"alive fraction fell {fall:+.3f} "
                                      f"({alive[:T // 5].mean():.3f} -> "
                                      f"{alive[tail].mean():.3f}): the curriculum "
                                      f"solved its own band and is training on nothing"))

    # floor_violated -- a LEVEL (W5 D5). "We are training where the verifier cannot
    # be trusted" is true of a snapshot; a delta-based version cannot say it. The
    # rates are an INPUT: diagnosing a gamma=0.10 run with the default 0.01 asks
    # about a verifier that does not exist.
    p_floor = precision_floor(gamma, alpha, tau)
    act = h.active[tail].astype(np.int64)
    ph = h.p_hat[tail]
    below = float(np.mean([float((ph[t, act[t]] < p_floor).mean())
                           for t in range(act.shape[0])])) if act.size else 0.0
    if below > FLOOR_SHARE:
        out.append(("floor_violated", f"{below:.0%} of the rollout budget is spent "
                                      f"on prompts below p_floor = {p_floor:.4f} "
                                      f"(gamma={gamma}, alpha={alpha}, tau={tau}): "
                                      f"more than tau of those rewards are hacks"))

    # budget_dead -- THE TWO SIDES SEPARATELY. D3's identity says dead_frac alone
    # cannot tell you which, and the two sides need OPPOSITE fixes.
    dead = float(h.dead_frac[tail].mean())
    if dead > DEAD_LEVEL:
        fail = float(h.dead_fail[tail].mean())
        succ = float(h.dead_succ[tail].mean())
        out.append(("budget_dead", f"{dead:.3f} of groups dead -- fail-side {fail:.3f} "
                                   f"(SCAFFOLD these) vs success-side {succ:.3f} "
                                   f"(RETIRE these); the sum is not actionable"))

    # `mean_p_rising` IS DELIBERATELY ABSENT. Suspected redundant with
    # saturated_band; MEASURED to fire on healthy runs twice as hard as on sick
    # ones, because rising p IS learning. Cut for FIRES-ON-HEALTHY, not redundancy.
    return out

# ---------------------------------------------------------------------------
# Fixtures and the training loop (provided -- nothing below is a TODO)
# ---------------------------------------------------------------------------

def _seq_probs(theta_p: np.ndarray, tab: SeqTable) -> np.ndarray:
    q = np.exp(seq_logprobs_np(theta_p, tab))
    return q / q.sum()


def exact_p(theta_p: np.ndarray, tab: SeqTable, mask_p: np.ndarray) -> float:
    """The prompt's true pass rate, by enumeration. Never estimated."""
    return float(exact_expectation(theta_p, tab, mask_p))


_CONTAM_TRUE_P = 0.30    # the difficulty a memorised item ACTUALLY has


def _tilt_to_target(theta_p: np.ndarray, tab: SeqTable, mask_p: np.ndarray,
                    target: float) -> tuple[np.ndarray, float]:
    """Find a scalar logit bonus on the correct set realising exact p == target.

    Returns (theta, achieved). The caller MUST check `achieved`.

    NOT BISECTION, and this is the harness defect that cost the most to find. The
    bonus is applied to every (state, action) pair on a correct sequence's path,
    and those pairs are SHARED with incorrect sequences that pass through the same
    prefix. So `bonus -> p` is not monotone: for ~30% of correct-sets a bonus of
    -40 drives p to 1.0, because suppressing the shared prefix suppresses the
    incorrect continuations harder than the correct ones. Bisection assumes a
    monotone bracket, silently returned an endpoint when it did not have one, and
    the fixture's advertised "exact known difficulty" was wrong on 40% of prompts
    with errors up to 0.97 -- while every stage downstream quoted those difficulties
    to four decimal places.

    A DENSE SCAN plus local refinement makes no monotonicity assumption, and the
    returned `achieved` lets the caller REJECT a correct-set whose target is simply
    not attainable rather than pretend. Verify the bracket; if you cannot, do not
    use a root-finder that requires one.
    """
    hot = np.zeros_like(theta_p)
    for y in np.flatnonzero(mask_p > 0.0):
        for t in range(int(tab.lengths[y])):
            hot[tab.states[y, t], tab.actions[y, t]] += 1.0

    grid = np.linspace(-60.0, 60.0, 2401)
    ps = np.array([exact_p(theta_p + b * hot, tab, mask_p) for b in grid])
    j = int(np.argmin(np.abs(ps - target)))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    best_b, best_e = float(grid[j]), float(abs(ps[j] - target))
    for _ in range(120):
        for b in (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0):
            e = abs(exact_p(theta_p + b * hot, tab, mask_p) - target)
            if e < best_e:
                best_b, best_e = float(b), float(e)
        mid = 0.5 * (lo + hi)
        if abs(exact_p(theta_p + lo * hot, tab, mask_p) - target) < \
           abs(exact_p(theta_p + hi * hot, tab, mask_p) - target):
            hi = mid
        else:
            lo = mid
    th = theta_p + best_b * hot
    return th, float(exact_p(th, tab, mask_p))


def _prompt_at(rng: np.random.Generator, env: TokenEnv, tab: SeqTable,
               target: float, tries: int = 60, tol: float = 1e-9,
               ) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample the correct-set until `target` is ATTAINABLE, then realise it.

    Rejection sampling on the fixture: not every correct-set can express every
    difficulty, so ask several and keep one that can. Returns the best attempt
    with its achieved p even on exhaustion, so the caller can assert rather than
    silently ship a prompt that is not the difficulty it claims to be.
    """
    n = tab.n_seqs
    best = None
    for _ in range(tries):
        m = np.zeros(n)
        m[rng.choice(n, size=max(1, n // 3), replace=False)] = 1.0
        th, got = _tilt_to_target(0.5 * rng.standard_normal((env.n_states, env.vocab)),
                                  tab, m, target)
        if best is None or abs(got - target) < abs(best[2] - target):
            best = (m, th, got)
        if abs(got - target) <= tol:
            break
    return best[1], best[0], best[2]


def make_curriculum_task(env: TokenEnv, tab: SeqTable, n_prompts: int,
                         rng: np.random.Generator, p_lo: float = 0.02,
                         p_hi: float = 0.80, alpha: float = 0.0,
                         gamma: float = 0.0, contaminated_frac: float = 0.0,
                         ) -> CurriculumTask:
    """P prompts whose EXACT initial pass rates are placed on a log-odds grid.

    Placing them in log-odds rather than in p is the same argument the controller
    makes: `p_lo = 0.02, p_hi = 0.80` spans 5.4 nats, and a linear-in-p grid would
    put 12 of 16 prompts above 0.2 and call that a difficulty range.

    `alpha`/`gamma` corrupt `accept` relative to `gold` by flipping whole
    SEQUENCES (a sequence the checker mis-scores, it mis-scores every time -- a
    verifier is a function of the output, not a coin), so the realised rates are
    close to but not exactly the requested ones. `verifier_rates` from D2 is what
    measures them; this only sets them up.

    `contaminated_frac` marks a share of prompts as memorised: their theta is
    tilted to p ~ 1 - 1e-9 instead of their grid position. They still carry their
    ORIGINAL grid difficulty in `.gold`, which is what makes the twin test
    meaningful.
    """
    n = tab.n_seqs
    targets = 1.0 / (1.0 + np.exp(-np.linspace(logit(p_lo), logit(p_hi), n_prompts)))
    gold, accept, thetas, contam = [], [], [], []
    n_contam = int(round(contaminated_frac * n_prompts))
    contam_idx = set(rng.choice(n_prompts, size=n_contam, replace=False).tolist())
    worst = 0.0

    for i in range(n_prompts):
        want = 1.0 - 1e-9 if i in contam_idx else float(targets[i])
        th, m, got = _prompt_at(rng, env, tab, want)
        worst = max(worst, abs(got - want))

        acc = m.copy()
        if alpha > 0.0:
            for y in np.flatnonzero(m > 0.0):
                if rng.random() < alpha:
                    acc[y] = 0.0
        if gamma > 0.0:
            for y in np.flatnonzero(m == 0.0):
                if rng.random() < gamma:
                    acc[y] = 1.0
        gold.append(m)
        accept.append(acc)
        thetas.append(th)
        contam.append(i in contam_idx)

    if worst > 1e-6:
        raise RuntimeError(
            f"fixture could not realise its requested difficulties: worst |p - target| "
            f"= {worst:.3e}. A fixture that misses its own set point is not a fixture.")

    return CurriculumTask(env=env, tab=tab, gold=np.stack(gold),
                          accept=np.stack(accept), theta0=np.stack(thetas),
                          contaminated=np.asarray(contam, dtype=bool))


def perturb(task: CurriculumTask, rng: np.random.Generator) -> CurriculumTask:
    """The PERTURBED TWIN: same reasoning structure, different surface.

    Concretely: permute which sequences express the correct answer, keeping the
    SIZE of the correct set fixed, and re-tilt each prompt to the difficulty its
    grid position asked for. So an honest prompt keeps its p (the perturbation is
    difficulty-neutral by construction) and a MEMORISED prompt loses its
    memorisation and falls back to its true difficulty.

    Difficulty-neutral BY CONSTRUCTION is exactly the property the control
    measures, and `stage_contamination` FAILS if the control moves -- because a
    perturbation that is itself hard convicts every item on the list, and then you
    are measuring your perturbation rather than the model. That check is the
    difference between an observation and evidence (W5 D5).
    """
    P = task.n_prompts
    gold, thetas = [], []
    worst = 0.0
    for i in range(P):
        p_target = float(exact_p(task.theta0[i], task.tab, task.gold[i]))
        if task.contaminated[i]:
            p_target = _CONTAM_TRUE_P
        p_target = min(max(p_target, 1e-6), 1 - 1e-6)
        th, m, got = _prompt_at(rng, task.env, task.tab, p_target)
        worst = max(worst, abs(got - p_target))
        thetas.append(th)
        gold.append(m)
    if worst > 1e-6:
        raise RuntimeError(
            f"twin is not difficulty-matched: worst |p - target| = {worst:.3e}. "
            "The control would move and the whole comparison would be confounded.")
    return CurriculumTask(env=task.env, tab=task.tab, gold=np.stack(gold),
                          accept=np.stack(gold), theta0=np.stack(thetas),
                          contaminated=task.contaminated)





# ---- samplers -------------------------------------------------------------

class StaticSort:
    """Rank prompts by step-0 p_hat, take the middle band by RANK, never re-rank.

    This is "we sorted the data easy-to-hard and it's working" as code. It is not
    a straw man: it is what a sorted-curriculum dataloader does, and the only
    thing wrong with it is the tense -- p is a property of the POLICY, and the
    policy is the thing being changed.
    """

    name = "static_sort"

    def __init__(self, n_active: int, lo: float = 0.25, hi: float = 0.75):
        self.n_active, self.lo, self.hi = n_active, lo, hi
        self.order: np.ndarray | None = None

    def __call__(self, step: int, p_hat: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
        if self.order is None:
            self.order = np.argsort(p_hat)
        P = len(p_hat)
        band = self.order[int(self.lo * P):max(int(self.hi * P), int(self.lo * P) + 1)]
        if len(band) == 0:                       # EMPTY-BAND GUARD (see module docstring)
            band = self.order
        take = min(self.n_active, len(band))
        return rng.choice(band, size=take, replace=False)


class BandTracker:
    """Re-estimate p every step from the rollouts you ALREADY PAID FOR, and sample
    the LOG-ODDS band around the set point.

    The sensor is c/G, smoothed. It costs nothing: it is the group you just rolled
    out. That is the argument for closing the loop -- not that closed-loop is
    better in principle, but that the measurement is already on the invoice.

    `width` is in NATS of log-odds, because the legal band is multiplicative:
    +/- 1.5 nats is roughly the 50%-of-peak band at every k, whereas +/- 0.05 in p
    is wider than the entire band at k = 64 and narrower than half of it at k = 1.
    """

    name = "band_tracker"

    def __init__(self, n_active: int, p_star: float, width: float = 1.5):
        self.n_active, self.p_star, self.width = n_active, p_star, width

    def __call__(self, step: int, p_hat: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
        d = np.abs(logit(p_hat) - logit(self.p_star))
        band = np.flatnonzero(d <= self.width)
        if len(band) < self.n_active:
            # EMPTY / THIN BAND: fall back to the nearest prompts in log-odds. The
            # controller's job is to keep this set non-empty; the failure under
            # test is that it cannot, so this branch must be RECORDED, never
            # silently averaged over an empty slice (that is the nan).
            band = np.argsort(d)[:self.n_active]
        take = min(self.n_active, len(band))
        return rng.choice(band, size=take, replace=False)


class UniformSampler:
    """The null: sample prompts uniformly, no curriculum at all. Provided so that
    'the controller helped' is measured against something, not asserted."""

    name = "uniform"

    def __init__(self, n_active: int):
        self.n_active = n_active

    def __call__(self, step: int, p_hat: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(p_hat), size=min(self.n_active, len(p_hat)),
                          replace=False)


def train_with_curriculum(task: CurriculumTask, cfg: CurrCfg, sampler,
                          p_lo_band: float = 0.03,
                          p_hi_band: float = 0.60) -> CurrHistory:
    """GRPO on a numpy tabular policy, with the DATA POLICY as the variable.

    The estimator is D1's `group_advantage` and the gradient is exact given the
    sampled group -- sum_i A_i grad log pi(y_i) with grad log pi from
    `all_score_functions`. We are not re-testing the estimator today (D1 did, to
    1e-12 against enumeration); we are testing what happens when the DATA CHANGES,
    so the estimator is held fixed and cheap so that >= 5 paired seeds x 4
    fixtures runs in seconds.

    Only the ACTIVE prompts are rolled out and only they are updated -- prompts
    own disjoint theta blocks, so "not sampled" and "no gradient" are the same
    statement. That is the budget, and the whole experiment is how you spend it.
    """
    tab, P = task.tab, task.n_prompts
    rng = np.random.default_rng(cfg.seed)
    theta = task.theta0.copy()
    ks = np.asarray(cfg.ks, dtype=np.int64)

    p_hat = np.full(P, 0.5)
    rec = {k: [] for k in ("step", "dead_frac", "dead_fail", "dead_succ",
                           "in_band", "signal")}
    p_true_rows, p_hat_rows, active_rows, passk_rows = [], [], [], []

    for step in range(cfg.steps):
        p_true = np.array([exact_p(theta[i], tab, task.gold[i]) for i in range(P)])
        p_true_rows.append(p_true)
        p_hat_rows.append(p_hat.copy())
        passk_rows.append([float(np.mean(pass_at_k_exact(p_true, int(k)))) for k in ks])

        idx = np.asarray(sampler(step, p_hat, rng), dtype=np.int64)
        active_rows.append(idx)

        n_dead = n_fail = n_succ = 0
        abs_sum, n_roll = 0.0, 0
        for i in idx:
            q = _seq_probs(theta[i], tab)
            draw = rng.choice(tab.n_seqs, size=cfg.group, p=q)
            r = task.accept[i][draw]
            c = float(r.sum())
            p_hat[i] = cfg.ema * p_hat[i] + (1.0 - cfg.ema) * (c / cfg.group)

            if c == 0.0 or c == cfg.group:
                n_dead += 1
                n_fail += int(c == 0.0)
                n_succ += int(c == cfg.group)
                continue                      # dead => EXACTLY zero, so skip the update

            A = group_advantage(r, cfg.adv_mode)
            abs_sum += float(np.abs(A).sum())
            n_roll += cfg.group
            S = all_score_functions(task.env, theta[i], tab)      # (n_seqs, S, V)
            theta[i] += cfg.lr / cfg.group * np.einsum("g,gsv->sv", A, S[draw])

        n_act = max(len(idx), 1)
        rec["step"].append(step)
        rec["dead_frac"].append(n_dead / n_act)
        rec["dead_fail"].append(n_fail / n_act)
        rec["dead_succ"].append(n_succ / n_act)
        rec["in_band"].append(float(np.mean((p_true[idx] >= p_lo_band) &
                                            (p_true[idx] <= p_hi_band)))
                              if len(idx) else 0.0)
        rec["signal"].append(abs_sum / n_roll if n_roll else 0.0)

    return CurrHistory(
        p_true=np.asarray(p_true_rows), p_hat=np.asarray(p_hat_rows),
        active=np.asarray(active_rows), passk=np.asarray(passk_rows), ks=ks,
        **{k: np.asarray(v, dtype=np.float64) for k, v in rec.items()})


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
        self.fast = fast
        self.env = TokenEnv(3, 3)
        self.tab = build_seq_table(self.env)
        self.steps = 60 if fast else 150
        self.seeds = 5 if fast else 8
        self.n_prompts = 12 if fast else 16
        self.grid = np.linspace(1e-4, 1 - 1e-4, 2001 if fast else 20001)

    def task(self, seed: int = 3, n_prompts: int | None = None,
             **kw) -> CurriculumTask:
        return make_curriculum_task(self.env, self.tab,
                                    self.n_prompts if n_prompts is None else n_prompts,
                                    np.random.default_rng(seed), **kw)

    def run(self, sampler_kind: str, seed: int = 0, task_kw: dict | None = None,
            **kw) -> CurrHistory:
        cfg = CurrCfg(steps=self.steps, seed=seed, **kw)
        task = self.task(seed=100 + seed, **(task_kw or {}))
        n_act = cfg.n_active
        sampler = {"static": lambda: StaticSort(n_act),
                   "band": lambda: BandTracker(n_act, set_point_analytic(8)),
                   "uniform": lambda: UniformSampler(n_act)}[sampler_kind]()
        return train_with_curriculum(task, cfg, sampler)


def _fail_fast(ctx: Ctx, *needs: str) -> None:
    """Touch ONLY the TODOs this stage uses, on a 1-element input, so a missing
    implementation surfaces before an expensive fixture is built -- and a stage is
    never blocked by a TODO it does not need (D1's own grading-scaffold bug)."""
    one = np.array([0.3])
    if "signal" in needs:
        signal_curve(one, 4, "center")
    if "useful" in needs:
        useful_signal(one, 0.1, 0.01, 4)
    if "setpoint" in needs:
        set_point(1)
    if "diagnose" in needs:
        diagnose_curriculum(CurrHistory(
            step=np.zeros(1), p_true=np.full((1, 1), 0.5), p_hat=np.full((1, 1), 0.5),
            active=np.zeros((1, 1), dtype=np.int64), dead_frac=np.zeros(1),
            dead_fail=np.zeros(1), dead_succ=np.zeros(1), in_band=np.ones(1),
            signal=np.ones(1), passk=np.zeros((1, 1)), ks=np.array([1])))


# ---- stages ---------------------------------------------------------------

def stage_signal(ctx: Ctx) -> None:
    """signal_curve -- the two closed forms, the (G-1), and dead == exactly 0"""
    _fail_fast(ctx, "signal")
    p = ctx.grid

    worst_abs = worst_sq = 0.0
    for G in (2, 4, 8, 16, 64):
        a, s = signal_curve(p, G, "center")
        ca, cs = signal_center_closed(p, G)
        # RELATIVE, not absolute: 2(G-1)p(1-p) is 31.5 at (G=64, p=0.5), so an
        # absolute 1e-15 there is asking for 1e-16 RELATIVE -- below eps. The first
        # reference run failed exactly here at 1.24e-12 and the harness was wrong,
        # not the implementation. An absolute tolerance is only meaningful on a
        # quantity you have checked is O(1).
        worst_abs = max(worst_abs, float(np.max(np.abs(a - ca) / np.maximum(ca, 1e-300))))
        worst_sq = max(worst_sq, float(np.max(np.abs(s - cs) / np.maximum(cs, 1e-300))))
    _check("centered E[sum|A|] == 2(G-1)p(1-p) on 5 group sizes",
           worst_abs < 1e-13, f"max RELATIVE err {worst_abs:.3e}")
    _check("centered E[sum A^2] == (G-1)p(1-p)  [= Bessel with sigma^2 = p(1-p)]",
           worst_sq < 1e-13, f"max RELATIVE err {worst_sq:.3e}")

    # the (G-1) IDENTIFIED, not assumed: E[sumA^2]/p(1-p) must be affine in G with
    # slope 1 and intercept -1.
    Gs = np.array([2, 4, 8, 16, 32, 64], dtype=np.float64)
    vals = np.array([signal_curve(np.array([0.3]), int(G), "center")[1][0] for G in Gs])
    coef = np.polyfit(Gs, vals / (0.3 * 0.7), 1)
    _check("the (G-1) is MEASURED: slope 1, intercept -1",
           abs(coef[0] - 1.0) < 1e-12 and abs(coef[1] + 1.0) < 1e-10,
           f"slope {coef[0]:.12f}, intercept {coef[1]:+.12f}")

    worst_w = 0.0
    for G in (2, 4, 8, 16, 64):
        _, s = signal_curve(p, G, "whiten")
        want = signal_whiten_closed(p, G)
        worst_w = max(worst_w, float(np.max(np.abs(s - want) / np.maximum(want, 1e-300))))
    _check("whitened E[sum A^2] == G * P(alive) -- difficulty DELETED from the "
           "magnitude", worst_w < 1e-13, f"max RELATIVE err {worst_w:.3e}")

    for G in (4, 8, 64):
        for mode in ("center", "whiten"):
            a, s = signal_curve(p, G, mode)
            am, sm = signal_curve(1.0 - p, G, mode)
            _check(f"p <-> 1-p symmetry ({mode}, G={G})",
                   float(np.max(np.abs(a - am))) < 1e-12 and
                   float(np.max(np.abs(s - sm))) < 1e-12)

    # THE TRAP. Not "small". Exactly zero, from every direction.
    for G in (2, 8, 64):
        for mode in ("center", "whiten"):
            a0, s0 = signal_curve(np.array([0.0, 1.0]), G, mode)
            _check(f"dead ends contribute EXACTLY 0.0, not nan ({mode}, G={G})",
                   np.all(a0 == 0.0) and np.all(s0 == 0.0) and
                   np.all(np.isfinite(a0)) and np.all(np.isfinite(s0)),
                   f"|A| {a0}, A^2 {s0}")

    # whitened |A| lives in [2 sqrt(G-1), G] -- a range of only 1.51 at G=8, which
    # is why sum A^2 = G is the load-bearing statement and not sum|A|.
    for G in (8, 64):
        cs = np.arange(1, G)
        lo, hi = 2 * math.sqrt(G - 1), float(G)
        vals = 2 * np.sqrt(cs * (G - cs))
        _check(f"whitened sum|A| in [2sqrt(G-1), G] (G={G}), ratio {hi / lo:.2f}x",
               vals.min() >= lo - 1e-12 and vals.max() <= hi + 1e-12)

    try:
        signal_curve(p, 8, "nonsense")
        _check("unknown mode raises ValueError", False)
    except ValueError:
        _check("unknown mode raises ValueError", True)


def stage_covariance(ctx: Ctx) -> None:
    """useful_signal -- the verifier is a MULTIPLIER (constant in p), and its null"""
    _fail_fast(ctx, "useful")
    p = ctx.grid

    worst = 0.0
    for G in (2, 8, 32):
        for al in (0.0, 0.1, 0.3):
            for ga in (0.0, 0.01, 0.2):
                U, _, _ = useful_signal(p, al, ga, G)
                want = 2 * (G - 1) * p * (1 - p) * youden(al, ga)
                worst = max(worst, float(np.max(np.abs(U - want))))
    _check("U == 2(G-1)p(1-p)J on a 27-point (G, alpha, gamma) grid",
           worst < 1e-12, f"max err {worst:.3e}")

    # THE NEGATIVE CONTROL. Identically zero IN p, and exactly.
    worst_null = 0.0
    for al, ga in [(0.5, 0.5), (0.3, 0.7), (0.9, 0.1), (0.0, 1.0)]:
        U, _, _ = useful_signal(p, al, ga, 8)
        worst_null = max(worst_null, float(np.max(np.abs(U))))
    _check("alpha+gamma == 1 => U IDENTICALLY 0 in p (the coin-flip verifier)",
           worst_null == 0.0 or worst_null < 1e-15,
           f"max |U| = {worst_null:.3e} -- healthy |A|, healthy dead_frac, "
           f"zero learning, forever")

    U, _, _ = useful_signal(p[1:-1], 0.6, 0.6, 8)
    _check("alpha+gamma > 1 => U < 0 EVERYWHERE (the gradient points backwards)",
           bool(np.all(U < 0)), f"max U = {U.max():.3e}")

    # THE SECOND THESIS, as an assertion: J enters as a CONSTANT IN p, so it
    # scales the curve and cannot move its argmax. A verifier does not tilt a
    # curriculum.
    for al, ga in [(0.0, 0.0), (0.1, 0.01), (0.3, 0.05), (0.45, 0.50)]:
        U, _, _ = useful_signal(p, al, ga, 8)
        _check(f"argmax_p U == 1/2 for (alpha={al}, gamma={ga}), J={youden(al, ga):.2f}",
               abs(float(p[np.argmax(U)]) - 0.5) < 2.0 / len(p),
               "the verifier is a MULTIPLIER, not a preference")

    # ... AND YET the useful FRACTION of the mass you actually spend collapses at
    # low p, because at low p most of the reward variance is false positives.
    # Different denominator, different claim, and the file caught me conflating
    # them: `mass` is at p_obs, U is at p.
    _, _, r_sound = useful_signal(p[1:-1], 0.30, 0.0, 8)
    _check("SOUND verifier (gamma=0): useful fraction -> 1 as p -> 0, at any alpha",
           abs(float(r_sound[0]) - 1.0) < 1e-3 and float(r_sound[0]) > float(r_sound[-1]),
           f"ratio {float(r_sound[0]):.6f} at p->0 vs "
           f"{float(r_sound[len(r_sound) // 2]):.4f} at p=0.5 -- soundness is "
           f"EXACTLY the property that stops the collapse")

    al, ga = 0.10, 0.01
    # EVALUATE THE LIMIT AS A LIMIT. The first run measured this slope at the
    # grid's smallest point (p = 5e-4) and got 85.34 against a predicted 89.90 --
    # a 5% "failure" that was really "p = 5e-4 is not small compared to the
    # gamma/(1-alpha-gamma) ~ 0.0112 scale that sets where the linear regime
    # starts". "smallest point in my grid" is not "-> 0".
    tiny = np.array([1e-9, 1e-8])
    _, _, r0 = useful_signal(tiny, al, ga, 8)
    slope = float(r0[0] / tiny[0])
    want = youden(al, ga) / (ga * (1 - ga))
    _check(f"UNSOUND (gamma={ga}): useful fraction -> 0 LINEARLY, slope "
           f"{slope:.2f} == J/(gamma(1-gamma)) = {want:.2f}",
           abs(slope - want) / want < 1e-5,
           "linear regime needs p << gamma/(1-alpha-gamma) = "
           f"{ga / youden(al, ga):.4f}")
    # ...and query the exact p values, rather than the nearest point of a grid
    # that is 5e-4 coarse. At p = 0.001 a 2.5e-4 snap is a 25% error in p, which
    # is where the 0.0900-vs-0.0825 disagreement came from. A grid is a plotting
    # device, not a lookup table.
    ask = np.array([0.5, 0.1, 0.02, 0.005, 0.001])
    _, _, r = useful_signal(ask, al, ga, 8)
    got = [float(x) for x in r]
    _check("useful fraction: " + " / ".join(f"{g:.4f}" for g in got) +
           "  at p = 0.5/0.1/0.02/0.005/0.001",
           np.allclose(got, [0.8973, 0.8980, 0.6454, 0.3109, 0.0825], atol=2e-3),
           "at p = 0.001, 91.75% of the advantage mass is false positives")
    _check("the noise floor gamma(1-gamma) = 0.0099 does NOT shrink with p",
           abs(ga * (1 - ga) - 0.0099) < 1e-12,
           "so you cannot raise the LR out of a bad SNR -- it scales the useful "
           "numerator and the false-positive denominator identically")

    # A NOISY VERIFIER MANUFACTURES ALIVENESS OUT OF A MASTERED PROMPT. mass is
    # evaluated at p_obs, so at p = 1 with alpha = 0.1 we get p_obs = 0.9, the
    # group is NOT unanimous, and the trainer sees a healthy pile of advantage
    # mass -- every scrap of it noise, since U = 0 identically at p in {0,1}.
    # The first run asserted mass == 0 here; that is only true of a PERFECT
    # verifier. This is the dead-group diagnostic's blind spot: it reads the
    # verifier's labels, so it scores a saturated prompt as alive.
    ends = np.array([0.0, 1.0])
    U_e, mass_e, ratio_e = useful_signal(ends, 0.1, 0.01, 8)
    _check("saturated prompt (p=1) still shows mass "
           f"{float(mass_e[1]):.4f} > 0 under a noisy verifier",
           float(mass_e[1]) > 1.0 and float(mass_e[0]) > 0.0,
           f"p=0: mass {float(mass_e[0]):.4f} -- dead_frac reads the LABELS, so "
           "it cannot see either end")
    _check("...and U == 0 there EXACTLY, so useful fraction == 0: all mass, no signal",
           np.all(U_e == 0.0) and np.all(ratio_e == 0.0) and np.all(np.isfinite(ratio_e)))
    _, mass_perf, ratio_perf = useful_signal(ends, 0.0, 0.0, 8)
    _check("only a PERFECT verifier gives mass == 0 at the ends, ratio 0.0 not nan",
           np.all(mass_perf == 0.0) and np.all(ratio_perf == 0.0))


def stage_setpoint(ctx: Ctx) -> None:
    """set_point -- p* = a/(a+b+k-1), 1/2 IFF a = b+k-1, and the band in log-odds"""
    _fail_fast(ctx, "setpoint")
    p = ctx.grid

    worst = 0.0
    pts = [(a, b, k) for a in (1.0, 2.0) for b in (1.0, 2.0)
           for k in (1, 2, 4, 8)]
    for a, b, k in pts:
        sp = set_point(k, a, b, gamma=0.0)              # gamma=0 => no floor
        grid_arg = float(p[np.argmax(value_curve(p, k, a, b))])
        worst = max(worst, abs(sp.metric - grid_arg))
    _check(f"analytic == grid argmax at {len(pts)} (a,b,k) points",
           worst < 2.0 / len(p), f"max gap {worst:.3e} vs resolution {1 / len(p):.3e}")

    ms = [set_point(k, gamma=0.0).metric for k in (1, 2, 4, 8, 16, 64)]
    _check("p* strictly decreasing in k: " +
           " / ".join(f"{m:.4f}" for m in ms),
           all(x > y for x, y in zip(ms, ms[1:])))

    # THE CHECK THAT PROVES YOU SOLVED IT: 1/2 is not "k=1", it is "a = b+k-1".
    _check("p* == 1/2 at (a,b,k) = (1,1,1)", abs(set_point(1, 1, 1, gamma=0.0).metric - 0.5) < 1e-15)
    _check("p* == 1/2 at (a,b,k) = (2,1,2) TOO -- the condition is a = b+k-1, "
           "not k = 1", abs(set_point(2, 2, 1, gamma=0.0).metric - 0.5) < 1e-15)
    # ...and (2,2,2) gives EXACTLY 2/5. The first run wrote this as
    # `|p* - 1/2| > 0.1`, which is a threshold set exactly equal to the quantity
    # it is testing -- 0.09999999999999998 > 0.1 is False. Assert the value, not
    # a hand-picked distance from a value.
    _check("p* == 2/5 EXACTLY at (a,b,k) = (2,2,2) -- a = 2, b+k-1 = 3",
           abs(set_point(2, 2, 2, gamma=0.0).metric - 0.4) < 1e-15)

    for k in (1, 4, 8, 16, 64):
        lo, hi = set_point(k, gamma=0.0).band
        v = value_curve(np.array([lo, hi]), k)
        vmax = value_curve(np.array([set_point_analytic(k)]), k)[0]
        _check(f"band endpoints are at 50% of peak (k={k})",
               np.all(np.abs(v / vmax - 0.5) < 1e-4),
               f"[{lo:.5f}, {hi:.5f}]  hi/lo {hi / lo:.2f}x")

    lo1, hi1 = set_point(1, gamma=0.0).band
    lo64, hi64 = set_point(64, gamma=0.0).band
    shrink_p = (hi1 - lo1) / (hi64 - lo64)
    shrink_l = (logit(hi1) - logit(lo1)) / (logit(hi64) - logit(lo64))
    _check("band narrows 19.07x in p but only 1.43x in LOG-ODDS",
           abs(shrink_p - 19.07) < 0.1 and abs(shrink_l - 1.43) < 0.02,
           f"{shrink_p:.2f}x in p, {shrink_l:.2f}x in logit-p -- controlling in p "
           f"makes it {shrink_p / shrink_l:.1f}x harder than it is")

    for bad in [dict(k=0), dict(k=1, a=0.0), dict(k=1, b=-1.0), dict(k=1, G=1),
                dict(k=1, mode="nonsense")]:
        try:
            set_point(**bad)
            _check(f"rejects {bad}", False)
        except ValueError:
            _check(f"rejects {bad}", True)


def stage_exponents(ctx: Ctx) -> None:
    """MEASURE (a, b) instead of assuming (1,1) -- and report if it is not identified

    The learning law Delta p ~ p^a (1-p)^b is the EMPIRICAL half of p*. The
    `+k-1` is arithmetic; (a, b) is a property of your optimiser and your
    parameterisation, and this stage measures it in a direct-softmax model over
    the 15 sequences, where the difficulty can be dialled exactly.
    """
    rng = np.random.default_rng(0)
    n, G, lr = ctx.tab.n_seqs, 8, 0.2
    targets = np.array([0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95])
    reps = 200 if ctx.fast else 800

    dps = []
    for tgt in targets:
        base = 0.4 * rng.standard_normal(n)
        mask = np.zeros(n)
        mask[rng.choice(n, size=n // 3, replace=False)] = 1.0
        lo, hi = -40.0, 40.0                      # bisect the tilt to hit tgt exactly
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(softmax_np(base + mid * mask) @ mask) < tgt:
                lo = mid
            else:
                hi = mid
        th = base + 0.5 * (lo + hi) * mask
        q0 = softmax_np(th)
        p0 = float(q0 @ mask)

        acc = 0.0
        for _ in range(reps):
            draw = rng.choice(n, size=G, p=q0)
            r = mask[draw]
            A = group_advantage(r, "centered")
            if not np.any(A):
                continue                          # dead: exactly zero, skip
            g = np.zeros(n)
            for j, y in enumerate(draw):          # grad log pi(y) = e_y - q
                g += A[j] * (np.eye(n)[y] - q0)
            th_new = th + lr / G * g
            acc += float(softmax_np(th_new) @ mask) - p0
        dps.append((p0, acc / reps))

    P0 = np.array([d[0] for d in dps])
    DP = np.array([d[1] for d in dps])
    ok = DP > 0
    X = np.stack([np.log(P0[ok]), np.log(1 - P0[ok]), np.ones(ok.sum())], axis=1)
    coef, res, *_ = np.linalg.lstsq(X, np.log(DP[ok]), rcond=None)
    a_hat, b_hat = float(coef[0]), float(coef[1])
    pred = X @ coef
    ss_res = float(np.sum((np.log(DP[ok]) - pred) ** 2))
    ss_tot = float(np.sum((np.log(DP[ok]) - np.log(DP[ok]).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    se = math.sqrt(ss_res / max(ok.sum() - 3, 1))

    _check(f"the fit is good enough to quote (R^2 = {r2:.4f}, resid sd {se:.3f})",
           r2 > 0.90,
           "a poor fit is a RESULT: it would mean p* is not identified on this "
           "stack and the honest output is to say so, not to quote a number")
    _check(f"measured (a, b) = ({a_hat:.3f}, {b_hat:.3f}) -- NOT (1,1)",
           abs(a_hat - 2.0) < 0.45 and abs(b_hat - 2.0) < 0.45,
           "exact gradient ascent moves p by ~lr*||grad p||^2 ~ p^2(1-p)^2, so "
           "(2,2) is the PREDICTION and (1,1) -- the advantage-mass model -- is "
           "the wrong law for this optimiser")

    k = 8
    p_assumed = set_point_analytic(k, 1.0, 1.0)
    p_measured = set_point_analytic(k, a_hat, b_hat)
    _check(f"and it MOVES the set point at k={k}: assumed {p_assumed:.4f} -> "
           f"measured {p_measured:.4f} ({p_measured / p_assumed:.2f}x)",
           p_measured > p_assumed,
           "measure (a,b); the +k-1 is arithmetic, the exponents are not")


def stage_group_size(ctx: Ctx) -> None:
    """G* = ln(0.1)/ln(1-p*) ~ 2.3026k, and the 88.34% you get with G = 8 anyway"""
    ks = [1, 2, 4, 8, 16, 64, 256]
    exact = [g_star(k) for k in ks]
    approx = [2.3026 * k for k in ks]
    closed = [math.log(0.1) / math.log(1 - set_point_analytic(k)) for k in ks]

    # 4 vs 5 AT k=1, AND NOWHERE ELSE. The single-sided closed form
    # ln(0.1)/ln(1-p*) counts only the all-FAIL tail. At k=1 the set point is 1/2,
    # where the all-PASS tail is exactly as large, so it undercounts deaths by 2x
    # and returns 4 where the honest both-tails criterion p^G + (1-p)^G <= 0.1
    # needs 5. By k=2 (p* = 1/3) the easy tail is 1.5% of the total and the two
    # agree forever after. This is the learner's own W6 D1 numbers-card correction
    # -- the two-branch g_for_alive -- recurring, and it recurs PRECISELY at the
    # one k where the set point sits at 1/2. session.md's table says 4; it is 5.
    _check("G* exact integers (both tails): " + " / ".join(str(g) for g in exact),
           exact == [5, 6, 11, 20, 38, 149, 591])
    naive = [math.ceil(c) for c in closed]
    _check("single-sided ln(0.1)/ln(1-p*) disagrees at k=1 ONLY: "
           + " / ".join(str(n) for n in naive),
           naive[0] == 4 and naive[1:] == exact[1:],
           "the easy tail is only negligible once p* is; at k=1 it is half the deaths")
    _check("G* -> 2.3026k from ABOVE, and the gap closes",
           all(c >= a - 1e-9 for c, a in zip(closed, approx)) and
           abs(closed[-1] / approx[-1] - 1) < 0.01,
           f"ratio closed/2.3026k: " +
           " / ".join(f"{c / a:.3f}" for c, a in zip(closed, approx)))
    _check("the approximation is BAD at small k (44.7% high at k=1)",
           abs(closed[0] / approx[0] - 1) > 0.4,
           f"{closed[0]:.4f} vs {approx[0]:.4f} -- quote the exact integer in a "
           f"config, the asymptote in a sentence")

    dead = [100 * dead_fraction_levels(np.array([sp := set_point_analytic(k),
                                                 1 - sp]), 8) for k in ks[:-1]]
    _check("P(dead | G=8, p*) = " + " / ".join(f"{d:.2f}%" for d in dead),
           np.allclose(dead, [0.78, 3.92, 16.78, 38.97, 61.57, 88.34], atol=5e-3),
           "report pass@64, train at its optimum, use G=8 -> 88% of your groups "
           "return EXACTLY zero gradient")
    _check("and the two sides are NOT symmetric at p*: failures dominate",
           all(dead_split(set_point_analytic(k), 8)[0] >
               dead_split(set_point_analytic(k), 8)[1] for k in ks if k > 1),
           "so the fix is scaffolding, not retirement -- and dead_frac alone "
           "cannot tell you that")


def stage_floor(ctx: Ctx) -> None:
    """the precision floor, the k=10 crossover SOLVED, and which wall binds"""
    _fail_fast(ctx, "setpoint")

    rows = [(0.01, 0.10, 0.10, 0.090909, 10.0),
            (0.01, 0.10, 0.05, 0.174312, 4.7368),
            (0.03, 0.10, 0.10, 0.230769, 3.3333),
            (0.10, 0.10, 0.10, 0.500000, 1.0)]
    for ga, al, tau, want_floor, want_k in rows:
        f = precision_floor(ga, al, tau)
        _check(f"p_floor(gamma={ga}, alpha={al}, tau={tau}) = {f:.6f}",
               abs(f - want_floor) < 1e-6)
        _check(f"  crossover with 1/(k+1) at k = {1 / f - 1:.4f} -- SOLVED, not scanned",
               abs((1 / f - 1) - want_k) < 1e-3)

    _check("gamma = 0.10 has J = 0.80 and p_floor = 0.5000",
           abs(youden(0.10, 0.10) - 0.80) < 1e-12 and
           abs(precision_floor(0.10, 0.10, 0.10) - 0.5) < 1e-12,
           "a respectable J AND unusable below the median: J is a MULTIPLIER on "
           "signal, the floor is a CONSTRAINT ON THE DOMAIN -- different objects")
    _check("a SOUND verifier (gamma = 0) has NO floor at any difficulty",
           precision_floor(0.0, 0.3, 0.001) == 0.0,
           "soundness is the only thing that removes this wall -- which is why "
           "the FP question is 'is the set empty', not 'how big is the rate'")

    f = precision_floor(0.01, 0.10, 0.10)
    ret = [value_curve(np.array([f]), k)[0] / value_curve(
        np.array([set_point_analytic(k)]), k)[0] for k in (8, 10, 16, 32, 64)]
    _check("value retained at the floor: " + " / ".join(f"{r:.4f}" for r in ret),
           np.allclose(ret, [0.9793, 1.0000, 0.8872, 0.3804, 0.0358], atol=5e-4),
           "at pass@64 a 1% FP rate costs 96.4% of the signal the metric asked for")
    _check("and retention == 1.0 EXACTLY at the crossover k = 10",
           abs(ret[1] - 1.0) < 1e-12)

    for k, want in [(4, "metric"), (8, "metric"), (10, "metric"), (16, "precision"),
                    (64, "precision")]:
        sp = set_point(k)
        _check(f"binding constraint at k={k} is '{want}' (p* = {sp.p_star:.4f})",
               sp.binding == want)

    # THE LEARNER'S FINDING, W6 D4: which wall binds is a consequence of the
    # alive-TOLERANCE you choose, not a fact about the system. Bisect for the
    # wall; p^G + (1-p)^G is strictly decreasing on (0, 1/2], so the root is
    # unique and there is no excuse for scanning a grid for it.
    def dead_wall(tol: float, G: int = 8) -> float:
        f_ = lambda x: x ** G + (1 - x) ** G - tol
        lo, hi = 1e-12, 0.5
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f_(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    for tol, want in [(0.10, "dead"), (0.50, "precision")]:
        wall = dead_wall(tol)
        got = "dead" if wall > f else "precision"
        _check(f"at dead-tolerance {tol:.0%}: wall {wall:.4f} vs floor {f:.4f} "
               f"-> {got.upper()} binds", got == want,
           "both are correct; the disagreement is about the tolerance, and two "
           "people can argue past each other for a week while both are right")


def stage_whiten(ctx: Ctx) -> None:
    """/std is ALREADY a curriculum: G/4 tilt, and it moves the set point 4.2x"""
    _fail_fast(ctx, "signal", "setpoint")
    p = ctx.grid[1:-1]

    # THE TILT IS A RATIO OF RATIOS, and the first run compared the wrong pair.
    # whitened/centered at p -> 0 is G^2/(G-1) (9.143 at G=8); at p = 1/2 it is
    # 4G(1-2^(1-G))/(G-1) (4.536). Neither is "the tilt" -- the tilt is how much
    # MORE the hard tail is lifted than the middle, i.e. their quotient:
    #     [G^2/(G-1)] / [4G(1-2^(1-G))/(G-1)]  =  G/[4(1-2^(1-G))]  ~ G/4.
    # The (G-1) cancels, which is why the answer is clean. Assert the quotient.
    for G in (8, 64):
        num = signal_whiten_closed(p, G)
        den = signal_center_closed(p, G)[1]
        lim0 = G * G / (G - 1.0)
        mid = float((num / den)[np.argmin(abs(p - 0.5))])
        _check(f"whitened/centered -> G^2/(G-1) = {lim0:.4f} as p -> 0 (G={G})",
               abs(float((num / den)[0]) - lim0) / lim0 < 0.02,
               f"measured {float((num / den)[0]):.4f}; at p=1/2 it is {mid:.4f}")
        _check(f"TILT = that quotient = G/[4(1-2^(1-G))] at G={G}: "
               f"{whiten_tilt_limit(G):.4f}",
               abs(lim0 / mid - whiten_tilt_limit(G)) / whiten_tilt_limit(G) < 0.02,
               f"measured {lim0 / mid:.4f} -- the (G-1) cancels, which is why "
               "the answer is clean")
    _check("G/4 to 3 sig figs at G = 64: 16.0000", abs(whiten_tilt_limit(64) - 16.0) < 1e-9)

    # centered ScoreA^2 by c: a balanced group gets 16.25x a c=1 group at G=64.
    # Whitening equalises them. THAT is the tilt.
    G = 64
    _check("centered sum A^2 = c(G-c)/G: c=1 -> 0.9844, c=G/2 -> 16.0 (16.25x); "
           "whitened both -> 64",
           abs(1 * (G - 1) / G - 0.984375) < 1e-12 and
           abs((G / 2) * (G / 2) / G - 16.0) < 1e-12)

    # AT (G=64, k=1) THE SET POINT DOES NOT EXIST. The whitened objective is
    # G*P(alive) times the metric weight; at k=1 that weight is 1, and P(alive) at
    # G=64 is 1.0 to machine precision across the whole middle of the range. The
    # objective is flat to within 1e-15 RELATIVE over p in [0.4166, 0.5834] -- a
    # plateau 0.167 wide -- so its argmax is whatever your grid happens to return.
    # The first run quoted 0.89 and the re-run 1.11; both are grid artefacts
    # straddling the true, unidentifiable 1.00. This is the whitening thesis in
    # extremis: whitening deletes difficulty from the magnitude, and with a large
    # G and a k that asks for nothing, there is NOTHING LEFT TO PREFER.
    pg = np.linspace(1e-4, 1 - 1e-4, 20001)
    v = signal_whiten_closed(pg, 64)
    flat = pg[v >= v.max() * (1 - 1e-15)]
    _check("at (G=64, k=1) the whitened set point is NOT IDENTIFIED: flat to "
           f"1e-15 over [{flat.min():.4f}, {flat.max():.4f}]",
           float(flat.max() - flat.min()) > 0.15,
           "report 'no preference', not a third significant figure")
    v8 = signal_whiten_closed(pg, 8)
    _check("...but at G=8 it IS identified (P(alive) still curves at p=1/2)",
           float(np.ptp(pg[v8 >= v8.max() * (1 - 1e-15)])) == 0.0)

    want = {8: [1.00, 0.75, 0.82, 0.88, 0.96], 64: [0.24, 0.32, 0.44, 0.71]}
    for G in (8, 64):
        ks = (1, 4, 8, 16, 64) if G == 8 else (4, 8, 16, 64)
        got = []
        for k in ks:
            sp = set_point(k, G=G, gamma=0.0, mode="whiten")
            got.append(sp.whitened / set_point_analytic(k))
        _check(f"whitened set point / metric set point (G={G}): " +
               " / ".join(f"{g:.2f}" for g in got),
               np.allclose(got, want[G], atol=0.02),
               f"max move {1 / min(got):.1f}x -- your advantage normalisation IS "
               f"a curriculum")

    # THE NEGATIVE CONTROL: at G = 2, P(alive) = 2p(1-p), so whitening and
    # centering are the same objective up to a constant and MUST agree on argmax.
    worst = 0.0
    for k in (1, 2, 4, 8, 16, 64):
        sp = set_point(k, G=2, gamma=0.0, mode="whiten")
        worst = max(worst, abs(sp.whitened - set_point_analytic(k)))
    _check("NEGATIVE CONTROL: at G=2 whitened argmax == metric argmax",
           worst < 1e-4, f"max gap {worst:.3e} -- P(alive) = 2p(1-p) at G=2")


def stage_controller(ctx: Ctx) -> None:
    """THE EXPERIMENT: static sort vs closed-loop band tracker, >= 5 paired seeds

    PREDICTION, WRITTEN BEFORE THE RUN (D3's rule -- a prediction recorded after
    the fact is a description):
      (1) `static_sort` decays: its band is fixed at step 0, the policy solves it,
          and the alive fraction FALLS over the run. Mechanism: p is a property of
          the policy, and the sampler was configured with a property of the
          initial policy.
      (2) `band_tracker` holds its alive fraction roughly flat, because the sensor
          (c/G, already paid for) tracks the thing that moved.
      (3) The gap is in ALIVE FRACTION and in USEFUL SIGNAL PER ROLLOUT, and it
          may or may not show in pass@k -- the toy has 16 prompts and pass@8
          saturates. If it does not show, that is a NULL and it gets reported as
          one.
    """
    seeds = range(max(5, ctx.seeds))
    out = {}
    for kind in ("static", "band", "uniform"):
        runs = [ctx.run(kind, seed=s) for s in seeds]
        out[kind] = dict(
            alive_end=np.array([1 - r.dead_frac[-10:].mean() for r in runs]),
            alive_start=np.array([1 - r.dead_frac[:10].mean() for r in runs]),
            signal=np.array([r.signal[-10:].mean() for r in runs]),
            pass1=np.array([r.passk[-1, 0] - r.passk[0, 0] for r in runs]),
            passk=np.array([r.passk[-1, -1] - r.passk[0, -1] for r in runs]))

    for kind in ("static", "band", "uniform"):
        d = out[kind]
        print(f"    {kind:<8} alive {d['alive_start'].mean():.3f} -> "
              f"{d['alive_end'].mean():.3f} (sd {d['alive_end'].std():.3f})  "
              f"signal {d['signal'].mean():.4f}  "
              f"dpass@1 {d['pass1'].mean():+.4f}  dpass@8 {d['passk'].mean():+.4f}")

    _check("PREDICTION 1: static_sort's alive fraction DECAYS over the run",
           out["static"]["alive_end"].mean() < out["static"]["alive_start"].mean(),
           f"{out['static']['alive_start'].mean():.4f} -> "
           f"{out['static']['alive_end'].mean():.4f} -- it was configured with a "
           f"property of the INITIAL policy")

    gap = out["band"]["alive_end"].mean() - out["static"]["alive_end"].mean()
    spread = out["band"]["alive_end"].std() + out["static"]["alive_end"].std()
    _check(f"PREDICTION 2: band_tracker keeps more groups alive, by {gap:+.4f} "
           f"against a paired spread of {spread:.4f}",
           gap > spread,
           "an effect inside the spread is a NULL, not a result -- >= 5 paired "
           "seeds, W5 D5")

    dk = out["band"]["passk"].mean() - out["static"]["passk"].mean()
    sk = out["band"]["passk"].std() + out["static"]["passk"].std()
    verdict = "RESULT" if abs(dk) > sk else "NULL"
    _check(f"PREDICTION 3 reported honestly: dpass@8 gap {dk:+.4f} vs spread "
           f"{sk:.4f} -> {verdict}", True,
           "recorded either way: the mechanism claim is about ALIVE FRACTION and "
           "SIGNAL, and pass@8 on 16 prompts is not powered to see it")


def stage_contamination(ctx: Ctx) -> None:
    """the eval lie that decays in k, the training cost that does not, and the CONTROL"""
    c, p_true = 0.05, _CONTAM_TRUE_P
    got = [100 * contamination_inflation(c, p_true, k) for k in (1, 2, 4, 8, 16, 64)]
    _check("inflation points: " + " / ".join(f"{g:.3f}" for g in got),
           np.allclose(got, [3.500, 2.450, 1.201, 0.288, 0.017, 0.000], atol=1e-3))
    _check("and it DECAYS monotonically to invisible by k = 64",
           all(x > y for x, y in zip(got, got[1:])) and got[-1] < 1e-3,
           "same shape as D3's RL gain (+0.4691 at k=1 -> +0.0000 at k=1024): a "
           "saturating metric cannot see a change in something already saturated")

    # TRAINING side: a memorised item is a SATURATED dead group. Exactly c of the
    # budget, returning exactly zero, while scoring 1.0 on the reward curve.
    G = 8
    fail_side, succ_side = dead_split(1.0 - 1e-12, G)
    _check("a memorised prompt (p -> 1) is dead on the SUCCESS side, ~100%",
           succ_side > 1 - 1e-9 and fail_side < 1e-9,
           f"succ {succ_side:.6f}, fail {fail_side:.3e} -- so a contaminated "
           f"fraction c costs exactly c of the rollout budget for zero gradient, "
           f"and it is INVISIBLE on the reward curve because those groups score 1.0")

    rng = np.random.default_rng(11)
    # 16 prompts at c = 0.25 is 4 contaminated items; the first run used 3 and
    # measured a gap of 0.4667 against an expected 0.7000 -- which is 0.9 standard
    # errors away and therefore not a disagreement at all, just an unpowered
    # estimate. A CONTAMINATION DETECTOR IS AN ESTIMATOR AND NEEDS A POWERED
    # SAMPLE (W5 D5, again): se = sqrt(p(1-p)/n_contaminated), so at n = 4 the se
    # is 0.23 and the detector cannot distinguish "no contamination" from "a
    # quarter of the set". Size the fixture to the claim.
    task = ctx.task(seed=21, contaminated_frac=0.25, n_prompts=128)
    twin = perturb(task, np.random.default_rng(22))
    p_orig = np.array([exact_p(task.theta0[i], ctx.tab, task.gold[i])
                       for i in range(task.n_prompts)])
    p_twin = np.array([exact_p(twin.theta0[i], ctx.tab, twin.gold[i])
                       for i in range(twin.n_prompts)])
    delta = p_orig - p_twin
    con, clean = task.contaminated, ~task.contaminated
    n_con = int(con.sum())
    se = math.sqrt(0.7 * 0.3 / n_con)

    _check(f"twin gap on CONTAMINATED items: {delta[con].mean():.4f} ~ 1 - p_true "
           f"= 0.7000 (n = {n_con}, se = {se:.4f})",
           abs(float(delta[con].mean()) - 0.7) < 2 * se,
           "the estimator's se is sqrt(p(1-p)/n_contaminated) -- at n = 4 that is "
           "0.23 and the detector cannot tell a clean set from a quarter-rotten one")

    # THE CONTROL. This is the check that converts an observation into evidence,
    # and the stage FAILS if it moves.
    _check(f"THE CONTROL: same perturbation on CLEAN items gives Delta = "
           f"{delta[clean].mean():+.5f} ~ 0",
           abs(delta[clean].mean()) < 0.02 and delta[clean].std() < 0.05,
           "if the control moves you are measuring your PERTURBATION's difficulty, "
           "not memorisation, and a hard perturbation convicts every item on the "
           "list (W5 D5; GSM1k's whole methodology)")
    _check("and the two populations SEPARATE: min(contaminated) > max(clean)",
           delta[con].min() > delta[clean].max(),
           f"[{delta[con].min():.4f}, {delta[con].max():.4f}] vs "
           f"[{delta[clean].min():.4f}, {delta[clean].max():.4f}]")
    _check("report the DISTRIBUTION, not the mean: it is bimodal by construction",
           float(np.std(delta)) > float(np.std(delta[clean])) * 3,
           "contamination is item-concentrated; a benchmark-wide mean hides it")


def stage_diagnose(ctx: Ctx) -> None:
    """diagnose_curriculum -- silent on healthy, >=5 paired seeds, and one rule CUT"""
    _fail_fast(ctx, "diagnose")


    # A THRESHOLD IS CALIBRATED FOR A REGIME, AND `--fast` IS A DIFFERENT REGIME.
    # At 150 steps the open-loop fixture's alive-fraction drop separates cleanly
    # from the closed-loop one (sick 0.594-0.667 vs healthy -0.319-0.244, air on
    # both sides). At 60 steps it has not finished decaying: sick spans
    # 0.097-0.500 and uniform reaches 0.083, so the gap is 0.014 wide and any
    # threshold in it is a coin flip wearing a number. The panel therefore runs at
    # its CALIBRATED regime even under --fast. If you want a fast panel, you must
    # re-derive the thresholds at that speed -- you may not borrow them.
    dctx = ctx if not ctx.fast else Ctx(fast=False)

    # FIVE FIXTURES, BECAUSE FOUR RULES NEED FIVE. The first run used three
    # (band / static / uniform), two of them healthy, and the collinearity check
    # duly reported `budget_dead` and `saturated_band` as having identical
    # fire-sets. They are not the same rule; the FIXTURE SET was too small to tell
    # them apart, and `cliff` and `floor_violated` never fired at all. YOU CANNOT
    # ESTABLISH NON-COLLINEARITY ON A FIXTURE SET THAT DOES NOT SPAN THE FAILURE
    # MODES -- the check was measuring the bench, not the panel.
    # THE RATES ARE AN INPUT, NOT AN INFERENCE. `diagnose_curriculum` is told the
    # verifier's measured (alpha, gamma) -- that is what D2's `verifier_rates` is
    # for -- so a fixture built with gamma = 0.10 must be DIAGNOSED with gamma =
    # 0.10. The first run called it with the default 0.01 on every fixture, and
    # `floor_violated` therefore could not fire on the one fixture designed to
    # trigger it: the panel was being asked about a verifier that did not exist.
    fixtures = {
        "healthy": (lambda s: dctx.run("band", seed=s), {}),
        "open_loop": (lambda s: dctx.run("static", seed=s), {}),
        "uniform": (lambda s: dctx.run("uniform", seed=s), {}),
        # dead on the FAIL side, and NOT saturating: separates budget_dead
        "too_hard": (lambda s: dctx.run("band", seed=s,
                                        task_kw=dict(p_lo=2e-4, p_hi=5e-3)), {}),
        # J = 0: rewards are coin flips, so nothing moves and the hard prompts
        # stay pinned. This is stage_covariance's negative control turned into a
        # fixture -- healthy |A|, healthy dead_frac, zero learning, forever -- and
        # it is the only thing that exercises `cliff`.
        "coin_flip": (lambda s: dctx.run("band", seed=s,
                                         task_kw=dict(p_lo=2e-4, p_hi=5e-3,
                                                      alpha=0.50, gamma=0.50)),
                      dict(alpha=0.50, gamma=0.50)),
        # NEGATIVE CONTROL, and it turned out to be one only after measurement: a
        # verifier with gamma = 0.10 has a floor of 0.50, but this run trains at an
        # in-band p of ~0.85, which is ABOVE it. `floor_violated` is a statement
        # about WHERE YOU TRAIN relative to the floor, not about the verifier
        # alone, so it is correct for this fixture to stay silent.
        "bad_verifier": (lambda s: dctx.run("band", seed=s,
                                            task_kw=dict(gamma=0.10, alpha=0.10)),
                         dict(gamma=0.10, alpha=0.10)),
    }
    fires = {k: [{n for n, _ in diagnose_curriculum(f(s), **rates)}
                 for s in range(5)]
             for k, (f, rates) in fixtures.items()}

    # A THRESHOLD HAS A BAND, AND FIXTURES INSIDE IT GIVE SEED-DEPENDENT VERDICTS.
    # `coin_flip` fires floor_violated on 3 seeds of 5 and `bad_verifier` fires
    # saturated_band on 2 of 5. Both sit inside the uncertainty band of a threshold
    # calibrated on the other fixtures. The temptation is to move the threshold
    # until they behave; that is fitting the panel to the bench. Report them as
    # borderline, and assert stability only where the design predicts a clear
    # answer.
    STABLE = ("healthy", "open_loop", "uniform", "too_hard")
    BORDERLINE = ("coin_flip", "bad_verifier")

    _check("SILENT on the healthy (closed-loop) fixture, all seeds",
           all(f == set() for f in fires["healthy"]),
           str([sorted(f) for f in fires["healthy"] if f][:2]))
    _check("saturated_band fires on the open-loop fixture, all seeds",
           all("saturated_band" in f for f in fires["open_loop"]),
           str([sorted(f) for f in fires["open_loop"]][:2]))
    _check("verdicts are STABLE across 5 paired seeds on the 4 designed fixtures",
           all(len({frozenset(f) for f in fires[k]}) == 1 for k in STABLE),
           str({k: [sorted(f) for f in fires[k]] for k in STABLE}))
    _check("...and the 2 BORDERLINE fixtures are reported as borderline, not tuned "
           "away",
           all(len({frozenset(f) for f in fires[k]}) > 1 for k in BORDERLINE),
           str({k: sorted({n for f in fires[k] for n in f}) for k in BORDERLINE})
           + " -- moving a threshold until a fixture behaves is fitting the panel "
             "to the bench")
    _check("cliff fires on too_hard, all seeds -- and ONLY after the EMA burn-in "
           "was excluded from the head window",
           all("cliff" in f for f in fires["too_hard"]),
           "p_hat is an EMA from a prior: with head = first quintile, prompts whose "
           "true p never left 1e-4 read as having 'moved' 0.047-0.276, and the rule "
           "was measuring its own instrument settling")

    # THE CUT RULE. Measure the fire-sets; do not assume the reason.
    allf = [f for v in fires.values() for f in v]
    _check("the CUT rule `mean_p_rising` is NOT implemented",
           all("mean_p_rising" not in f for f in allf))

    # COLLINEARITY: no two surviving rules may have identical fire-sets, or the
    # panel has two names for one measurement.
    names = sorted({n for f in allf for n in f})
    sets = {n: frozenset(i for i, f in enumerate(allf) if n in f) for n in names}
    dupes = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
             if sets[x] == sets[y]]
    _check(f"no two surviving rules share a fire-set ({len(names)} rules)",
           not dupes, f"collinear pairs: {dupes}" if dupes else str(names))
    _check("SCOPE: 15-sequence tabular softmax, 16 prompts, disjoint theta blocks",
           ctx.tab.n_seqs == 15,
           "nothing here licenses the set point at 1e9 shared parameters -- it "
           "says the LAW is arithmetic once (a,b) is measured, and that measuring "
           "(a,b) is the part that does not transfer")


STAGES = [
    ("signal", stage_signal),
    ("covariance", stage_covariance),
    ("setpoint", stage_setpoint),
    ("exponents", stage_exponents),
    ("groupsize", stage_group_size),
    ("floor", stage_floor),
    ("whiten", stage_whiten),
    ("controller", stage_controller),
    ("contamination", stage_contamination),
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
