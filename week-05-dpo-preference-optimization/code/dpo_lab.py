"""
Week 5 Day 5 -- THE DPO LAB. The week's deliverable: a from-scratch DPO trainer with
beta and data-quality ablations, and the four failure modes DERIVED rather than catalogued.

Every failure below was derived somewhere in D1-D4. The point of this file is that each
one is located IN D4's GRID -- you can say which AXIS causes it and which CELL removes it
-- and that each one FIRES ON ITS OWN FIXTURE while the diagnostic stays SILENT on a
healthy run. The silence is the harder half.

    L = mean_k  l( f(y_w), f(y_l) )  [ + lam_SFT * mean_k ( -log pi(y_w)/|y_w| ) ]
                  ^ LINK  ^ REWARD

    the LINK decides whether the margin is FINITE     (P1: only 2 of 5 finite at f=1)
    the REWARD decides where the LENGTH goes          (P2/P3: sum / mean / noref)

THE FOUR MODES
--------------
  M1  DIVERGENCE / REJECTED-LOGPROB COLLAPSE.   axis: LINK.   removed by: cDPO, IPO.
      Ford (D1): the BT MLE is finite iff the wins-digraph is strongly connected. A DPO
      dataset is a PERFECT MATCHING -- every completion appears in exactly ONE comparison
      -- so no relabelling can create a cycle and ANNOTATOR NOISE HAS NOWHERE TO ACT. The
      optimum is at infinity for typical preference data, not for pathological data.
      Measured D2: log pi(y_l) -> -25.9537, losers' mass 0.343396 -> 1.649e-09, loss
      4.834e-05, and ACCURACY READ 1.0 FROM STEP 2. Silent by construction.

  M2  BOTH LOGPROBS FALL.                        axis: neither -- it is what the grid
      OMITS.   removed by: the additive SFT term (NOT the odds term, see below).
      4 of 5 links are functions of d = r_w - r_l alone (D4: 2.220e-16). That invariance
      IS the pathology: nothing in a pairwise loss asks y_w to be LIKELY, only PREFERRED.
      Measured D2 AT THE CONVERGED OPTIMUM (not a divergence artifact): margin 0 -> +0.6155
      while log pi(y_w) -3.1249 -> -4.0003 and log pi(y_l) -3.3174 -> -5.4238.
      D4 CORRECTION: ORPO's odds term FLIPS SIGN with beta (-1.899, -0.457, +0.184, +0.385
      at beta = .25/.5/1/2) while the SFT term holds at every beta (+0.566, +0.528, +0.522,
      +0.557). Only one of those two is a mechanism.

  M3  LENGTH EXPLOITATION.                       axis: REWARD.   removed by: nothing here.
      The implicit reward is EXTENSIVE in |y| with NO RESISTANCE TERM. D3: constant
      per-token drift +0.2 gives sum slope +0.200000 and mean slope 0.000000 -- sum asks
      d_new > 0, mean asks d_new > dbar. PPO at least HAD a resistance term and it was
      still insufficient: W4 D4 measured confound kappa = +0.6276 against a per-token
      resistance whose maximum over ALL beta is 0.1411. DPO is one rung worse: the
      resistance does not exist.
      D4 CORRECTION: SimPO's 1/|y| AMPLIFIES length bias rather than braking it, because
      d(r)/d(log pi) = beta/|y| is largest at |y| = 1. And NEVER run a length experiment to
      convergence -- both arms saturate at t_max and the effect vanishes entirely.

  M4  OFF-DISTRIBUTION DRIFT THE LOSS CANNOT SEE. axis: neither -- it is a property of the
      DATA's support.   removed by: on-policy data (i.e. by not being DPO).
      DPO's implied constraint is evaluated at DATASET POINTS ONLY. That is not a KL: a KL
      is an expectation over a distribution, this is a finite sum over points. pi_theta can
      do anything it likes off the support of the comparisons and THE LOSS RETURNS THE SAME
      NUMBER. This is the mode with NO INSTRUMENT IN A STANDARD TRAINING LOOP AT ALL, which
      is the point -- you have to add `offdata_mass` and the on-policy reward deliberately.

W4 D4's SPLIT, REUSED:  OBJECTIVE-WRONG (M1, M3, M4) vs OPTIMIZER-BROKEN (M2).
Identical on a loss curve. OPPOSITE FIXES: change the data/link, vs change what you
optimise against.

WHAT THIS FILE ASSERTS THAT THE WEEK ONLY CLAIMED
-------------------------------------------------
  (a) THE CHAIN. BT-fit-then-tilt -> DPO -> cDPO -> IPO -> SimPO recovered from ONE code
      path (`variants.variant_loss`), with D2's equivalence as the anchor at one end and
      D4's five independently-written direct implementations at the other. Non-vacuity is
      asserted the way W4 D5 did it: show the rung that would look IDENTICAL if a switch
      were ignored, and show that it is not.

  (b) THE COLLAPSE PREDICTION -- the week's cleanest single result if it holds.
      D1: only r/beta is identified, so beta and annotator agreement are THE SAME KNOB.
      The data caps the link-units margin at M* = logit(1-eps); the policy therefore
      targets a LOG-RATIO margin Dhat* = logit(1-eps)/beta. So KL at convergence must be a
      function of Dhat* ALONE -- a beta sweep at fixed eps and an eps sweep at fixed beta
      must trace THE SAME CURVE. That is a prediction, not a plot, and stage `collapse`
      tests it at MATCHED MARGIN (the only honest cross-objective KL comparison, D4).

  (c) ALL FOUR MODES FIRE ON THEIR OWN FIXTURES AND `diagnose_dpo` IS SILENT ON A HEALTHY
      RUN. A diagnostic that always fires is not a diagnostic.

  (d) SELECTION IS AN ESTIMATOR, AND IT NEVER SEES GOLD. W4 D5: gold peaked at iter 8 of 40
      and shipping the last checkpoint cost 1.2522. `select_checkpoint` is given ONLY the
      training panel; gold GRADES it afterwards. (The session sketch passed `gold_eval` in;
      that would make it an oracle, which is a different and much easier function.)

THE UNITS TRAP, WHICH IS ALSO A REAL BUG I HIT WRITING D4'S HARNESS
-------------------------------------------------------------------
    Dhat = log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l)     LOG-RATIO margin
    M    = r_w - r_l = beta * Dhat                        LINK-INPUT margin
    M*   = logit(1-eps)      = 2.197225  at eps = 0.10
    Dhat* = logit(1-eps)/beta = 7.324082  at beta = 0.30
A "margin" with no units attached is a bug waiting to happen. `implicit_reward` has
already applied beta, so everything this file logs as `margin` is M.

HONEST INVENTORY (say it unprompted; W4 D5's habit)
---------------------------------------------------
  TRANSFERS: everything on the LINK axis (theorems about scalars); Ford (a property of the
  digraph, i.e. of your dataset); the shift-invariance algebra; the units discipline; the
  methodology.
  DOES NOT: (i) anything needing SHARED PARAMETERS -- D2 measured shared prefixes
  cancelling EXACTLY (0.000e+00) in a table, so "pushing y_l down drags y_w down" needs
  shared parameters, not shared tokens; every M2 magnitude here is a LOWER BOUND. (ii) M4
  at scale -- supp(pi) = supp(pi_ref) always here and KL is bounded by 4.6894, so we
  validate the INSTRUMENT, not the phenomenon. (iii) length magnitude -- t_max = 3.
  (iv) optimiser reality -- plain full-batch GD, deliberately, because margin ~ log t is a
  theorem about GD that Adam erases while still printing a plausible table. (v) annotator
  reality -- eps is one i.i.d. scalar; real noise correlates with item difficulty, which
  breaks the cap's derivation. (vi) there is no explicit RM, so W4's ensembling/LCB has
  nowhere to attach.

Run:  ./.venv/bin/python ai-lab-mle-prep/week-05-dpo-preference-optimization/code/dpo_lab.py
      [--fast] [--list] [stage ...]
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from variants import (  # noqa: E402
    F_KINDS,
    L_KINDS,
    Spec,
    TabEnv,
    _direct_cdpo,
    _direct_dpo,
    _direct_ipo,
    _direct_simpo,
    closed_form_optimum,
    entropy,
    exact_kl,
    expected_length,
    implicit_reward,
    link,
    logit,
    logratio_optimum,
    named_specs,
    point_mass_kl_bound,
    variant_loss,
)

# ---------------------------------------------------------------------------
# Configs and records (provided)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataCfg:
    """ONE dataset function, FOUR knobs -- so every failure mode is a FIXTURE rather than
    an anecdote. That is the whole design: if a mode needs a bespoke dataset builder, you
    have not understood which knob causes it.

        shape             'matching' | 'cycle' | 'all_pairs'        -> DEGREE of the
                          comparison graph: 1, 2, n-1. Ford's condition (M1).
        eps               annotator flip rate                        -> the cap, and M1's escape
        length_confound   kappa: the ANNOTATOR ranks by gold + kappa*|y|, gold does not (M3)
        n_pairs           0 = whatever the shape gives; >0 truncates -> leaves sequences
                          UNCOVERED by any comparison, which is M4's fixture

    NOTE, and stage `dataset` measures it: strong connectivity of the WINS digraph is NOT
    a property of the shape. Orienting ANY graph by a total order gives a DAG, so it takes
    BOTH degree >= 2 AND noise. It is a random variable whose distribution eps controls.
    """

    shape: str = "matching"
    eps: float = 0.0
    length_confound: float = 0.0
    n_pairs: int = 0
    seed: int = 0


@dataclass(frozen=True)
class RunCfg:
    """One training run. `variant` indexes D4's grid via `named_specs`."""

    variant: str = "DPO"
    beta: float = 0.5
    eps: float = 0.1          # the LOSS's smoothing (cDPO); independent of DataCfg.eps
    tau: float = 2.0
    gamma: float = 0.5
    sft_weight: float = 0.0
    lr: float = 0.5
    n_steps: int = 2000
    seed: int = 0
    ref_scale: float = 0.7
    stop_margin: float | None = None


@dataclass
class Dataset:
    pairs: torch.Tensor          # (K, 2) long, column 0 = the ANNOTATOR's winner
    gold_r: torch.Tensor         # (n_seqs,) the TRUE reward. Never enters the loss.
    label_r: torch.Tensor        # (n_seqs,) what the annotator actually ranks by
    covered: torch.Tensor        # (n_seqs,) bool: does this sequence appear in ANY pair?
    flipped: torch.Tensor        # (K,) bool: did annotator noise flip this label?
    cfg: DataCfg = field(default_factory=DataCfg)


@dataclass
class History:
    """The instrument panel, in PAIRS. Each pair discriminates a failure the other half of
    the pair cannot see. `cap` and `beta` travel with it because a magnitude without its
    cap is not an instrument.
    """

    step: np.ndarray
    loss: np.ndarray
    margin: np.ndarray        # M = beta * Dhat, LINK units      \  pair 1 is (logp_w, logp_l)
    acc: np.ndarray           # THE METRIC THAT REPORTS NOTHING  /  never the margin alone
    kl: np.ndarray
    logp_w: np.ndarray
    logp_l: np.ndarray
    elen: np.ndarray
    ent: np.ndarray
    nll_w: np.ndarray
    r_absmax: np.ndarray      # max_y |r_hat(y)| over covered sequences  -- vs `cap`
    onpolicy_r: np.ndarray    # E_{y~pi}[r_hat(y)]      \ pair 5: the ONLY M4 instrument,
    data_r_mean: np.ndarray   # mean_{y in D} r_hat(y)  / and the loop does not compute it
    offdata_mass: np.ndarray  # 1 - sum_{y in D} pi(y)
    gold: np.ndarray          # E_pi[gold_r]  -- A LUXURY. Absent in production.
    cap: float                # logit(1-eps) in LINK units, or inf when eps == 0
    beta: float
    theta: torch.Tensor


# ---------------------------------------------------------------------------
# Provided infrastructure
# ---------------------------------------------------------------------------


def spec_for(cfg: RunCfg) -> Spec:
    """RunCfg -> one cell of D4's grid. `sft_weight` is additive, not an axis."""
    s = named_specs(beta=cfg.beta, eps=cfg.eps, tau=cfg.tau, gamma=cfg.gamma)[cfg.variant]
    return replace(s, sft_weight=cfg.sft_weight)


def raw_edges(env: TabEnv, shape: str, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """UNORDERED comparison edges for a shape. Orientation is deliberately RANDOMISED.

    W5 D1's harness lesson, fifth sighting of "a fixture whose own structure makes a wrong
    implementation look right": `all_pairs` was emitted as (i, j) with i > j, which made
    "return column 0" a CORRECT implementation on two of three fixtures and let a dead-store
    bug survive to stage 9. Randomising the orientation here is what makes stage `dataset`'s
    non-vacuity check (raw column 0 scores ~0.5 against gold) meaningful.
    """
    n = env.n_seqs
    if shape == "matching":
        perm = rng.permutation(n)
        rows = [[int(perm[i]), int(perm[i + 1])] for i in range(0, n - (n % 2), 2)]
    elif shape == "cycle":
        perm = rng.permutation(n)
        rows = [[int(perm[i]), int(perm[(i + 1) % n])] for i in range(n)]
    elif shape == "all_pairs":
        rows = [[i, j] for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(shape)
    if n_pairs:
        rows = rows[:n_pairs]
    rows = np.asarray(rows, dtype=np.int64)
    flip = rng.random(len(rows)) < 0.5
    rows[flip] = rows[flip][:, ::-1]
    return rows


def strongly_connected(pairs: torch.Tensor, n: int) -> bool:
    """Ford's condition, by boolean transitive closure over the WINS digraph."""
    p = pairs.numpy()
    adj = np.zeros((n, n), dtype=bool)
    adj[p[:, 0], p[:, 1]] = True
    np.fill_diagonal(adj, True)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        adj = adj @ adj
    return bool(adj.all())


def bt_fit(pairs: torch.Tensor, n_seqs: int, steps: int = 20_000, lr: float = 0.5,
           eps: float = 0.0) -> torch.Tensor:
    """The Bradley-Terry MLE by plain GD, RE-CENTRED every step so the location gauge is
    fixed and any divergence is genuinely in the SCALE (D1). Shares no code path with
    `variant_loss`: that independence is what makes stage `chain`'s anchor evidence.
    """
    r = torch.zeros(n_seqs, requires_grad=True)
    for _ in range(steps):
        d = r[pairs[:, 0]] - r[pairs[:, 1]]
        nll = -((1 - eps) * torch.nn.functional.logsigmoid(d)
                + eps * torch.nn.functional.logsigmoid(-d)).mean()
        g, = torch.autograd.grad(nll, r)
        with torch.no_grad():
            r -= lr * g
            r -= r.mean()
    return r.detach()


def tilt(lp_ref: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """log of pi*(y) proportional to pi_ref(y) exp(r(y)/beta). The RLHF-side closed form.

    D2's gauge result: the simplex fixes the gauge, so `sum_y pi_ref e^{r/beta}` normalises
    to exactly 1 and r_hat = r - beta log Z. You cannot lower r_hat(y_l) without raising
    r_hat somewhere else.
    """
    z = lp_ref + r / beta
    return z - torch.logsumexp(z, dim=0)


def gold_of(lp: torch.Tensor, gold_r: torch.Tensor) -> float:
    """E_pi[gold_r]. The held-out evaluation: gold is defined on EVERY sequence, including
    the ones no comparison ever touched, which is exactly what makes M4 measurable."""
    return float((lp.exp() * gold_r).sum())


# ---------------------------------------------------------------------------
# TODO 0 -- the port, and its two calibration bugs
# ---------------------------------------------------------------------------


def rel(xs, scale: float | None = None) -> float:
    """Relative change across a run. THE PORT: W4 D4's `diagnose` had this and it had two
    calibration bugs. Fix them here, as a diff, so the fix is not a memory.

    BUG 1 -- THE ZERO DENOMINATOR, and it is not hypothetical here. W4's version was

        (xs[-1] - xs[0]) / max(abs(xs[0]), 1e-9)

    and D2 measured that at theta = theta_ref the implicit reward is EXACTLY 0.0 for every
    sequence. So `margin[0]` and `r_absmax[0]` are exactly 0.0 in EVERY DPO run and this
    returns 1e9 * xs[-1]. Every threshold downstream is then meaningless.

    THE LESSON, and it generalises: the fix for a zero denominator is usually to find the
    RIGHT denominator, not to add an epsilon to the wrong one. Two right answers here:
      * pass `scale` explicitly when the problem supplies one (for a margin that is the
        CAP, logit(1-eps) -- which is the whole of instrument pair 2); or
      * when it does not, divide by max_t |x_t|, which is zero only if the entire series is
        zero and which BOUNDS the result in [-2, 2], so no threshold can ever be tripped by
        a spurious 1e9.

    BUG 2 is not in this function -- it is the -20% entropy threshold that sits INSIDE the
    range a healthy run occupies. Stage `modes` catches it by asserting silence on health.

    Contract:
        len(xs) < 2 or an all-zero series      -> 0.0
        scale given and non-zero               -> (xs[-1] - xs[0]) / abs(scale)
        otherwise                              -> (xs[-1] - xs[0]) / max_t |x_t|
        the result is FINITE for every input, including xs[0] == 0.0 exactly
    """
    if len(xs) < 2:
        return 0.0
    denom = max(abs(x) for x in xs)
    if denom == 0:
        return 0.0
    if scale is not None and scale != 0 :
        return (xs[-1]-xs[0])/abs(scale)
    else:
        return (xs[-1]-xs[0])/denom


# ---------------------------------------------------------------------------
# TODO 1 -- the fixture factory
# ---------------------------------------------------------------------------


def make_dataset(env: TabEnv, cfg: DataCfg) -> Dataset:
    """ONE function, FOUR knobs, so every failure mode is a fixture.

    Build, in this order:

      1. gold_r -- the TRUE reward on every sequence. Seed it from `cfg.seed` and CENTRE it
         (a reward's location is a gauge; D1). This never enters the loss.
      2. label_r = gold_r + cfg.length_confound * |y|. This is what the ANNOTATOR ranks by.
         kappa > 0 means the annotator systematically prefers longer completions and gold
         does not -- W4 D4's confound, and M3's fixture. At kappa = 0 they coincide.
      3. edges = raw_edges(env, cfg.shape, cfg.n_pairs, rng). ORIENT each edge so column 0
         is label_r's winner.  <-- and note raw_edges randomised the orientation on purpose,
         so "return column 0 unchanged" is measurably wrong.
      4. FLIP each label independently with probability cfg.eps. Record the mask.
      5. covered[y] = True iff y appears in some pair. With cfg.n_pairs < the full count,
         most sequences are UNCOVERED -- that is M4's fixture, and the loss is blind there.

    Use ONE `np.random.default_rng(cfg.seed)` for everything, so a fixture is reproducible.

    TRAPS, all with a check:
      * D1's DEAD STORE: compute the winner from `label_r` and then do not overwrite it from
        the raw columns. That bug survived six stages because the fixture happened to emit
        gold-ordered columns. `raw_edges` no longer does; stage `dataset` asserts raw column
        0 scores ~0.5 against gold, so "return the input" is a measurably wrong answer.
      * `flipped` is a per-PAIR mask (K,), `covered` is per-SEQUENCE (n_seqs,). D4's bug 2
        was indexing one tensor by `pairs` and not the other. They are different objects.
      * dtypes: `pairs` must be long, `covered`/`flipped` must be bool.
    """
    covered = torch.zeros(env.n_seqs, dtype=torch.bool)
    rng = np.random.default_rng(cfg.seed)
    gold_r = torch.tensor(rng.normal(size = env.n_seqs))
    gold_r -= gold_r.mean()
    label_r = gold_r + cfg.length_confound * env.lengths
    edges = torch.tensor(raw_edges(env, cfg.shape, cfg.n_pairs, rng), dtype=torch.long)
    covered[edges[:, 0]] = True
    covered[edges[:, 1]] = True
    
    reward_a, reward_b = label_r[edges[:, 0]], label_r[edges[:, 1]]
    winner = torch.where(reward_a > reward_b, edges[:, 0], edges[:, 1])
    loser = torch.where(reward_a > reward_b, edges[:, 1], edges[:, 0])
    noise = rng.random(edges.shape[0])
    flipped = torch.tensor((noise < cfg.eps), dtype=torch.bool)
    winner_noisy = torch.where(flipped, loser, winner)
    loser_noisy = torch.where(flipped, winner, loser)
    pairs = torch.stack([winner_noisy, loser_noisy], dim=1)
    return Dataset(pairs=pairs, gold_r=gold_r, label_r=label_r, covered=covered, flipped=flipped, cfg=cfg)



# ---------------------------------------------------------------------------
# TODO 2 -- the trainer
# ---------------------------------------------------------------------------
def train_dpo(env: TabEnv, data: Dataset, cfg: RunCfg) -> History:
    """Plain full-batch GD on `variants.variant_loss`, logging the FULL panel.

    Structure (D3 bug 6, D4 bug 3 -- both were about this):
      * theta_ref = env.random_theta(cfg.seed, cfg.ref_scale); theta0 = theta_ref.clone().
        SEEDING BOTH ROLES FROM THE SAME CHECKPOINT is D3's SFT result and it is what makes
        every initial log-ratio EXACTLY 0, hence every initial implicit reward exactly 0,
        hence every pair's initial gradient weight exactly sigma(0) = 1/2. The run-away
        direction is then purely a property of the DATA.
      * theta_ref is frozen and must never receive a gradient. `seq_logp_ref` once, outside
        the loop -- D3 measured the exact reference cache at 8 bytes/pair and 0 trainer-side
        reference calls.
      * PLAIN GD, fixed lr, full batch. NOT Adam. `margin ~ log t` is a theorem about GD;
        an adaptive optimiser erases it while still printing a plausible table (D4 bug 3,
        the only bug of that day that produced numbers instead of a traceback).
      * SEPARATE optimisation from logging: build the graph, step, then recompute the whole
        panel under `torch.no_grad()`. Never relabel a pre-update record as a post-update
        metric. Step 0 is measured BEFORE any update, so the panel has n_steps + 1 entries.
      * Return the FINAL theta, not a stale clone (W3 D1's argmax-lag, fifth sighting).
      * `cfg.stop_margin`: stop at the first step whose mean margin reaches it. This is the
        MATCHED-MARGIN design and it is the only honest way to compare KL across two
        objectives -- otherwise you are measuring which one trained longer.

    The panel. Five pairs, and for each the failure it discriminates:
        (logp_w, logp_l)          M1 -- NEVER the margin alone; the margin looks identical
                                  under "learned the preference" and "destroyed y_l"
        (r_absmax, cap)           M1 causally, and with NO held-out set: exceeding
                                  logit(1-eps) means fitting past what the data supports
        (kl, acc)                 M1 silently. acc saturates in ~20 steps and then reports
                                  nothing for the rest of the run
        (elen, gold)              M3 -- raw win rate cannot answer, because length
                                  correlates with quality even under an explicit penalty
        (onpolicy_r, data_r_mean) M4 -- the ONLY instrument for it, and exactly what the
        + offdata_mass            training loop does not compute
    Ranges, and CHECK THEM (D4 bug 4: three of his metrics were outside the range of the
    quantity they claimed to be, catchable without running a step):
        acc in [0,1]   ent >= 0   elen in [1, t_max]   kl in [0, point_mass_kl_bound]
        nll_w > 0      offdata_mass in [0,1]
    `cap` = logit(1 - DataCfg.eps) in LINK units, or +inf when eps == 0 -- clean data has
    NO cap, which is M1's whole point.
    """
    theta_ref = env.random_theta(cfg.seed, cfg.ref_scale)
    spec = spec_for(cfg)
    with torch.no_grad():
        lp_ref = env.seq_logprobs(theta_ref)
    theta = theta_ref.clone().requires_grad_(True)

    cov = data.covered
    w, l = data.pairs[:, 0], data.pairs[:, 1]
    keys = ("loss", "margin", "acc", "kl", "logp_w", "logp_l", "elen", "ent", "nll_w",
            "r_absmax", "onpolicy_r", "data_r_mean", "offdata_mass", "gold")
    rec: dict[str, list[float]] = {k: [] for k in keys}

    @torch.no_grad()
    def snapshot() -> None:
        lp = env.seq_logprobs(theta)
        r = implicit_reward(spec.f_kind, lp, lp_ref, env.lengths, spec.beta)
        r_w, r_l = r[w], r[l]
        p = lp.exp()
        rec["loss"].append(float(variant_loss(spec, lp, lp_ref, env.lengths, data.pairs)))
        rec["margin"].append(float((r_w - r_l).mean()))
        rec["acc"].append(float((r_w > r_l).double().mean()))
        rec["kl"].append(exact_kl(lp, lp_ref))
        rec["logp_w"].append(float(lp[w].mean()))
        rec["logp_l"].append(float(lp[l].mean()))
        rec["elen"].append(expected_length(lp, env.lengths))
        rec["ent"].append(entropy(lp))
        rec["nll_w"].append(float(-(lp[w] / env.lengths[w]).mean()))
        rec["r_absmax"].append(float(r[cov].abs().max()))
        rec["onpolicy_r"].append(float((p * r).sum()))
        rec["data_r_mean"].append(float(r[cov].mean()))
        rec["offdata_mass"].append(float(1.0 - p[cov].sum()))
        rec["gold"].append(gold_of(lp, data.gold_r))

    def reached() -> bool:
        return cfg.stop_margin is not None and rec["margin"][-1] >= cfg.stop_margin

    snapshot()
    if not reached():
        for _ in range(cfg.n_steps):
            lp = env.seq_logprobs(theta)
            loss = variant_loss(spec, lp, lp_ref, env.lengths, data.pairs)
            grad, = torch.autograd.grad(loss, theta)
            with torch.no_grad():
                theta -= cfg.lr * grad
            snapshot()
            if reached():
                break

    cap = math.inf if data.cfg.eps <= 0.0 else logit(1.0 - data.cfg.eps)
    arrays = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    return History(step=np.arange(len(arrays["loss"]), dtype=float), cap=cap,
                   beta=cfg.beta, theta=theta.detach().clone(), **arrays)



# ---------------------------------------------------------------------------
# TODO 3 -- the diagnostic
# ---------------------------------------------------------------------------


def diagnose_dpo(h: History) -> list[tuple[str, str]]:
    """THE ARTIFACT. Given a run's panel, return which of the four modes fired.

    Names must come from EXACTLY this set, spelled this way:

        "divergence"   "both_logprobs_fall"   "length_exploitation"
        "off_distribution_drift"

    FIVE DESIGN CONSTRAINTS, each of which is a lesson and not a style preference:

      1. IT MUST BE SILENT ON A HEALTHY RUN. Stage `modes` asserts an EMPTY list on a
         strongly-connected, noisy, length-neutral, fully-covered DPO+SFT run. This is the
         harder half and it is the check most likely to be missing. W4 D4's -20% entropy
         threshold sat INSIDE the range a healthy run occupies; do not repeat it.

      2. BUILT FROM MECHANISMS, NOT THRESHOLDS. Every mode above was DERIVED this week, and
         each derivation names a signature that a healthy run cannot produce:
           divergence          -- acc SATURATED at 1.0 while the margin is STILL RISING and
                                  logp_l is STILL FALLING over the last half of the run.
                                  (Not "margin is large": on clean data the cap is +inf and
                                  a large margin is the CORRECT answer to a wrong question.)
                                  Where a finite cap exists, quote margin/cap as well.
           both_logprobs_fall  -- logp_w AND logp_l both END BELOW where they started.
           length_exploitation -- elen rose A LOT *and* gold fell from its peak. A length
                                  move alone is NOT a failure: a healthy run here moves
                                  E[|y|] +8.9%. Gold is load-bearing for exactly this
                                  mode, because M3 is the objective-wrong mode.
           off_distribution    -- the LEVEL of offdata_mass, not its change. Measured: the
                                  off-data mass FALLS during training, so a delta rule
                                  never fires. Support is a level. Reachable with NO gold,
                                  which matters because this is the mode with no instrument.

      3. IT MUST NOT BE BUILT AROUND KL. Three of the four modes do not need KL and one of
         them (M4) is invisible in it -- D2 measured KL BOUNDED by max_y -log pi_ref(y) =
         4.6894, so KL saturates while the reward diverges. KL is NECESSARY, NOT SUFFICIENT.

      4. GOLD IS A LUXURY. Use it only where it genuinely discriminates (objective-wrong vs
         optimizer-broken). Every other verdict must be reachable from the training panel.

      5. THE REASON STRING IS THE DELIVERABLE. "divergence detected" is a restatement.
         "margin 5.98 = 2.72x the eps=0.10 cap 2.197 while acc has read 1.000 since step 21
         and log pi(y_l) fell 3.41" is a reason. QUOTE THE NUMBER THAT FIRED IT.

    Use `rel(...)` for series with a real scale; for the margin pass `scale=h.cap`.
    """
    out: list[tuple[str, str]] = []
    n = len(h.loss)
    if n < 4:
        return out
    mid = n // 2

    # --- M1: divergence / rejected collapse ---------------------------------
    sat = h.acc[mid:].min() >= 1.0 - 1e-12
    margin_rising = (h.margin[-1] - h.margin[mid]) > 1e-3
    loser_falling = (h.logp_l[-1] - h.logp_l[mid]) < -1e-3
    if sat and margin_rising and loser_falling:
        first_sat = int(np.argmax(h.acc >= 1.0 - 1e-12))
        cap_note = (f", {h.margin[-1] / h.cap:.2f}x the cap {h.cap:.4f}"
                    if math.isfinite(h.cap) else ", and the data supply NO finite cap")
        out.append((
            "divergence",
            f"acc has read 1.000 since step {first_sat} yet the margin still climbed "
            f"{h.margin[mid]:.4f} -> {h.margin[-1]:.4f} over the run's second half"
            f"{cap_note}; log pi(y_l) {h.logp_l[0]:.4f} -> {h.logp_l[-1]:.4f}"))
    elif math.isfinite(h.cap) and h.margin[-1] > 1.5 * h.cap:
        out.append((
            "divergence",
            f"mean implicit margin {h.margin[-1]:.4f} is "
            f"{h.margin[-1] / h.cap:.2f}x the cap logit(1-eps) = {h.cap:.4f}: "
            f"fitting past what the annotators can support"))

    # --- M2: both logprobs fall ---------------------------------------------
    dw, dl = h.logp_w[-1] - h.logp_w[0], h.logp_l[-1] - h.logp_l[0]
    if dw < -1e-3 and dl < -1e-3:
        out.append((
            "both_logprobs_fall",
            f"log pi(y_w) {h.logp_w[0]:.4f} -> {h.logp_w[-1]:.4f} ({dw:+.4f}) AND "
            f"log pi(y_l) {h.logp_l[0]:.4f} -> {h.logp_l[-1]:.4f} ({dl:+.4f}) -- the "
            f"margin rose {h.margin[-1] - h.margin[0]:+.4f} and told you neither"))

    # --- M3: length exploitation --------------------------------------------
    # A LENGTH MOVE IS NOT A FAILURE. A healthy DPO+SFT run on this env moves E[|y|] by
    # +8.9% for reasons that have nothing to do with length. The failure is a length move
    # THAT COSTS GOLD, so gold is load-bearing here -- this is the one mode where it earns
    # its keep, because M3 is the objective-wrong mode and gold is the only thing that can
    # say the objective was wrong. The 0.25 gate sits 2.8x above healthy (0.089) and 4.4x
    # below the fixture (1.110): a threshold with air on BOTH sides, which is exactly what
    # W4 D4's -20% entropy rule did not have.
    len_rel = rel(h.elen)
    len_up = len_rel > 0.25
    gold_drop = float(h.gold.max() - h.gold[-1])
    gold_down = gold_drop > 0.05 * abs(float(h.gold.max())) and gold_drop > 1e-3
    if len_up and gold_down:
        out.append((
            "length_exploitation",
            f"E[|y|] {h.elen[0]:.4f} -> {h.elen[-1]:.4f} ({len_rel * 100:+.1f}%) while gold "
            f"fell {gold_drop:.4f} from its peak {h.gold.max():.4f} to {h.gold[-1]:.4f} -- "
            f"the margin rose to {h.margin[-1]:.4f} throughout and reported none of it"))

    # --- M4: off-distribution drift -----------------------------------------
    # THE LEVEL, NOT THE CHANGE. Measured: off-data mass FALLS during training (0.498 ->
    # 0.236) because pushing losers down moves mass onto the covered winners. A delta rule
    # therefore never fires and the mode looks absent. What is true is the LEVEL: 23.6% of
    # the trained model's mass sits where the loss is exactly constant. Mode 4 is a
    # statement about the support of the objective, and support is a level.
    off = float(h.offdata_mass[-1])
    if off > 0.05:
        out.append((
            "off_distribution_drift",
            f"pi_theta puts {off:.4f} of its mass on sequences NO comparison ever touched "
            f"({h.offdata_mass[0]:.4f} at init); the loss returns the SAME value wherever "
            f"that mass lands, so {off * 100:.1f}% of the model is unconstrained by the "
            f"objective and no field of the training panel reports it "
            f"(gold {h.gold[0]:+.4f} -> {h.gold[-1]:+.4f})"))

    return out


# ---------------------------------------------------------------------------
# TODO 4 -- the estimator
# ---------------------------------------------------------------------------


def select_checkpoint(h: History) -> int:
    """Which step would you SHIP? Return an index into the panel.

    W4 D5: gold peaked at iter 8 of 40 and shipping the last checkpoint cost 1.2522. Early
    stopping is the ESTIMATOR, not regularisation.

    THIS FUNCTION NEVER SEES GOLD. It is given the training panel and nothing else; stage
    `checkpoint` grades it against gold afterwards. (The session sketch passed `gold_eval`
    in. That would make it an oracle -- a different and much easier function, and one you
    cannot run in production.)

    PPO walked a KL ladder and you checkpointed ALONG the KL. DPO has no ladder, so derive
    the analogue. Three candidates, and only one survives:
      * EPOCH COUNT -- an arbitrary constant with no units, and D2 measured accuracy hitting
        1.0 at step 2 while the run had 19 998 steps left. Rejected.
      * KL -- necessary but NOT sufficient, and BOUNDED by max_y -log pi_ref(y) = 4.6894, so
        it saturates while the reward diverges. It also cannot be computed exactly at scale.
        Rejected as the primary.
      * IMPLICIT-REWARD MAGNITUDE AGAINST ITS CAP -- monotone under GD, computed from the
        training batch alone, and the cap logit(1-eps) is a STATISTICAL statement about the
        annotators requiring NO held-out set. This is the one.

    So: stop at the FIRST step whose mean margin reaches `h.cap`. When the data supply no
    finite cap (eps = 0), there is nothing to stop against -- return the last index and let
    the caller notice that clean data has no natural stopping rule. That is not a
    limitation of the code; it is M1.
    """
    n = len(h.margin)
    if not math.isfinite(h.cap):
        return n - 1
    hit = np.nonzero(h.margin >= h.cap)[0]
    return int(hit[0]) if hit.size else n - 1


# ---------------------------------------------------------------------------
# TODO 5 -- the ablation
# ---------------------------------------------------------------------------


def ablate(env: TabEnv, dcfg: DataCfg, rcfg: RunCfg, grid: dict[str, list],
           seeds: tuple[int, ...] = (0, 1, 2)) -> dict[tuple, dict[str, tuple[float, float]]]:
    """Sweep one variable at a time, at a FIXED compute budget, over >= 3 seeds.

    `grid` maps a field name to the values it takes. A field belongs to EITHER `DataCfg`
    (shape, eps, length_confound, n_pairs) OR `RunCfg` (variant, beta, ...); dispatch on
    which dataclass has it -- that dispatch is the reason one function can run a beta sweep
    and an eps sweep, which is what the collapse prediction needs.

    Returns {(field, value): {metric: (mean, spread)}} where spread = max - min over seeds,
    for these metrics read at the LAST step:
        margin, margin_over_cap, kl, gold, elen, gold_selected, gold_last, gold_cost
    `gold_selected` is gold at `select_checkpoint(h)` and `gold_cost` is
    max(gold) - gold[last]: what shipping the last checkpoint costs.

    RULES (W3's ablation hygiene, and they are the difference between a result and a story):
      * change ONE variable, FIX the compute budget, >= 3 seeds, report mean +- spread
      * claim a result ONLY if the effect beats seed noise (non-overlapping intervals).
        ONE SEED CAN LIE.
      * the SEED must vary the dataset AND the reference/init together -- set both
        DataCfg.seed and RunCfg.seed, or you are re-running the same fixture.
      * use `dataclasses.replace(base, **{field: value})`. `replace(base, field=value)`
        passes the literal kwarg `field` and fails.
    """
    data_fields = {f for f in DataCfg.__dataclass_fields__}
    run_fields = {f for f in RunCfg.__dataclass_fields__}
    out: dict[tuple, dict[str, tuple[float, float]]] = {}

    for fieldname, values in grid.items():
        for value in values:
            per_seed: dict[str, list[float]] = {}
            for s in seeds:
                d = replace(dcfg, seed=s)
                r = replace(rcfg, seed=s)
                if fieldname in data_fields:
                    d = replace(d, **{fieldname: value})
                elif fieldname in run_fields:
                    r = replace(r, **{fieldname: value})
                else:
                    raise ValueError(f"unknown ablation field {fieldname!r}")
                data = make_dataset(env, d)
                h = train_dpo(env, data, r)
                k = select_checkpoint(h)
                vals = {
                    "margin": float(h.margin[-1]),
                    "margin_over_cap": float(h.margin[-1] / h.cap) if math.isfinite(h.cap) else float("nan"),
                    "kl": float(h.kl[-1]),
                    "gold": float(h.gold[-1]),
                    "elen": float(h.elen[-1]),
                    "gold_selected": float(h.gold[k]),
                    "gold_last": float(h.gold[-1]),
                    "gold_cost": float(h.gold.max() - h.gold[-1]),
                }
                for kk, vv in vals.items():
                    per_seed.setdefault(kk, []).append(vv)
            out[(fieldname, value)] = {
                kk: (float(np.mean(vv)), float(np.max(vv) - np.min(vv)))
                for kk, vv in per_seed.items()
            }
    return out


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
    """Fixtures, lazily built. ONE environment; every experiment changes exactly one knob."""

    BETA = 0.5
    LR = 0.5

    def __init__(self, fast: bool = False):
        self.fast = fast
        self.short = 400 if fast else 1_200
        self.long = 1_500 if fast else 6_000
        self.bt_steps = 4_000 if fast else 20_000
        self.seeds = (0, 1, 2)

    @cached_property
    def env(self) -> TabEnv:
        return TabEnv(vocab=3, t_max=3)

    @cached_property
    def ladder(self) -> TabEnv:
        """THE LENGTH LADDER: exactly one sequence per length in {1..5}, so length is the
        only thing that distinguishes two sequences and a length effect cannot be
        confounded with anything else (D4's fixture, reused)."""
        return TabEnv(vocab=2, t_max=5)

    # --- the five fixtures, one per mode plus health ------------------------

    @cached_property
    def d_separable(self) -> Dataset:
        """M1: a perfect MATCHING. Ford fails no matter what the annotators do."""
        return make_dataset(self.env, DataCfg(shape="matching", eps=0.0, seed=0))

    @cached_property
    def d_healthy(self) -> Dataset:
        """HEALTH: dense (degree 14), 25% noise, no length confound, full coverage.
        Strong connectivity is VERIFIED in stage `dataset`, not assumed."""
        return make_dataset(self.env, DataCfg(shape="all_pairs", eps=0.25, seed=0))

    @cached_property
    def d_length(self) -> Dataset:
        """M3, ON THE LADDER: the annotator ranks by gold + kappa|y|; gold has no length
        term. One sequence per length, so nothing else can explain a length move. Seed 11
        is CHOSEN, not default: it is the seed where corr(gold, |y|) = -0.572, so buying
        length has to SELL gold and the mode has a visible cost. On a seed where gold and
        length happen to agree, length exploitation is invisible -- which is itself the
        warning about single-seed evidence."""
        return make_dataset(self.ladder, DataCfg(shape="all_pairs", eps=0.10,
                                                 length_confound=1.5, seed=11))

    @cached_property
    def d_uncovered(self) -> Dataset:
        """M4: only 4 comparisons, so 7 of 15 sequences are touched by NO pair. Seed 3 is
        chosen because gold falls MONOTONICALLY from step 0 (+0.6737 -> +0.4050) while
        train accuracy reads 1.000 and the margin climbs to 3.64: every on-data instrument
        improves for the whole run and the thing you care about only gets worse."""
        return make_dataset(self.env, DataCfg(shape="matching", eps=0.10,
                                              n_pairs=4, seed=3))


def _fail_fast(ctx: Ctx) -> None:
    for fn, name in ((lambda: rel([0.0, 1.0]), "rel"),
                     (lambda: make_dataset(ctx.env, DataCfg()), "make_dataset")):
        try:
            fn()
        except NotImplementedError:
            raise
        except Exception:
            pass


# ---- individual stages ----------------------------------------------------


def stage_port(ctx: Ctx) -> None:
    """rel(): the port, with W4 D4's zero-denominator bug fixed."""
    _check("len<2 -> 0", rel([1.0]) == 0.0)
    _check("all-zero series -> 0", rel([0.0, 0.0, 0.0]) == 0.0)

    # THE BUG, reproduced: W4's version was (x[-1]-x[0])/max(|x[0]|, 1e-9).
    buggy = (5.9758 - 0.0) / max(abs(0.0), 1e-9)
    fixed = rel([0.0, 2.0, 5.9758])
    _check("zero denominator is finite and sane", math.isfinite(fixed) and abs(fixed) <= 2.0,
           f"W4's form gives {buggy:.3e}, fixed gives {fixed:.6f}")
    _check("bounded in [-2,2] for any series", all(
        abs(rel(xs)) <= 2.0 + 1e-12
        for xs in ([0.0, 1e6], [1e6, 0.0], [-3.0, 7.0], [1e-30, 1.0])))
    _check("explicit scale is used when given",
           abs(rel([0.0, 5.9758], scale=2.197225) - 5.9758 / 2.197225) < 1e-12,
           "the right fix for a zero denominator is the RIGHT denominator")
    _check("a real scale reproduces the old behaviour",
           abs(rel([4.0, 2.0]) - (-0.5)) < 1e-12)


def stage_dataset(ctx: Ctx) -> None:
    """make_dataset(): Ford per shape, the eps-cap, the length confound, non-vacuity."""
    env = ctx.env
    n = env.n_seqs

    # --- Ford's condition is NOT a property of the shape ---------------------
    # Orienting any graph by a total order gives a DAG, so it takes BOTH degree >= 2 AND
    # noise. Measure the ladder rather than asserting it: this IS D1's headline.
    print("      P(wins-digraph strongly connected) over 12 seeds:")
    table: dict[tuple[str, float], float] = {}
    for shape, deg in (("matching", 1), ("cycle", 2), ("all_pairs", n - 1)):
        row = []
        for eps in (0.0, 0.10, 0.25, 0.35):
            p = float(np.mean([
                strongly_connected(make_dataset(env, DataCfg(shape=shape, eps=eps,
                                                             seed=s)).pairs, n)
                for s in range(12)]))
            table[(shape, eps)] = p
            row.append(f"{p:.2f}")
        print(f"        {shape:<10} (degree {deg:>2})  eps 0.00/0.10/0.25/0.35 -> "
              + " / ".join(row))

    _check("degree 1: NEVER strongly connected, at ANY noise level",
           all(table[("matching", e)] == 0.0 for e in (0.0, 0.10, 0.25, 0.35)),
           "in a matching every completion appears once, so no relabelling can create a "
           "cycle -- annotator noise has NOWHERE TO ACT. That is the DPO data format.")
    _check("clean data: never strongly connected at any degree",
           all(table[(s, 0.0)] == 0.0 for s in ("matching", "cycle", "all_pairs")),
           "a transitive tournament is a DAG")
    _check("degree 2 is not enough either -- measure 0, not impossibility",
           table[("cycle", 0.35)] == 0.0 and table[("cycle", 0.0)] == 0.0,
           "a cycle CAN be strongly connected, but only if the flips orient it into a "
           "directed Hamiltonian cycle exactly. It never happened in 48 draws.")
    _check("noise is the escape hatch, and it needs DENSITY to act on",
           table[("all_pairs", 0.35)] == 1.0 and table[("all_pairs", 0.10)] > 0.0,
           f"all_pairs P: {table[('all_pairs', 0.10)]:.2f} @eps=.10 -> "
           f"{table[('all_pairs', 0.25)]:.2f} @eps=.25 -> "
           f"{table[('all_pairs', 0.35)]:.2f} @eps=.35 -- and a real DPO dataset is at "
           "degree 1, the top row")
    _check("P is monotone in eps for the dense shape",
           table[("all_pairs", 0.0)] <= table[("all_pairs", 0.10)]
           <= table[("all_pairs", 0.25)] <= table[("all_pairs", 0.35)])
    _check("the healthy fixture is verified strongly connected, not assumed",
           strongly_connected(ctx.d_healthy.pairs, n))

    m3 = make_dataset(env, DataCfg(shape="matching", eps=0.35, seed=0))
    _check("and the flips DID happen on the matching -- they just cannot help",
           int(m3.flipped.sum()) > 0,
           f"{int(m3.flipped.sum())}/{len(m3.pairs)} labels flipped, still a DAG")

    # --- the flip rate is the flip rate --------------------------------------
    big = make_dataset(env, DataCfg(shape="all_pairs", eps=0.20, seed=7))
    frac = float(big.flipped.double().mean())
    _check("realised flip rate ~ eps", abs(frac - 0.20) < 0.08, f"{frac:.4f} vs 0.20")
    ok = (big.label_r[big.pairs[:, 0]] > big.label_r[big.pairs[:, 1]])
    _check("flipped mask agrees with the labels exactly",
           bool((ok == ~big.flipped).all()),
           "the mask is per-PAIR; `covered` is per-SEQUENCE. Different objects.")

    # --- NON-VACUITY: 'return column 0 unchanged' must be measurably wrong ----
    raw = raw_edges(env, "all_pairs", 0, np.random.default_rng(7))
    raw_t = torch.tensor(raw, dtype=torch.long)
    raw_acc = float((big.label_r[raw_t[:, 0]] > big.label_r[raw_t[:, 1]]).double().mean())
    _check("raw column 0 scores ~0.5 against the labels", abs(raw_acc - 0.5) < 0.15,
           f"{raw_acc:.4f} -- so a dead store that overwrites the winner from the raw "
           "columns is DETECTABLE here (it was not on D1)")

    # --- the length confound (on the LADDER: one sequence per length, so length is the
    #     ONLY thing distinguishing two sequences and the effect cannot be confounded) ---
    lad = ctx.ladder
    lc = make_dataset(lad, DataCfg(shape="all_pairs", eps=0.0, length_confound=1.5, seed=1))
    lw = float(lad.lengths[lc.pairs[:, 0]].mean())
    ll = float(lad.lengths[lc.pairs[:, 1]].mean())
    base = make_dataset(lad, DataCfg(shape="all_pairs", eps=0.0, seed=1))
    bw = float(lad.lengths[base.pairs[:, 0]].mean())
    _check("kappa>0 makes winners systematically longer", lw - ll > 1.0,
           f"E|y_w| {lw:.4f} vs E|y_l| {ll:.4f} (kappa=0 baseline E|y_w| {bw:.4f})")
    _check("gold carries NO length term while the annotator does",
           abs(float(torch.corrcoef(torch.stack([lc.gold_r, lad.lengths]))[0, 1])) < 0.7
           and float(torch.corrcoef(torch.stack([lc.label_r, lad.lengths]))[0, 1]) > 0.9)

    # --- coverage is M4's knob -------------------------------------------------
    u = make_dataset(env, DataCfg(shape="matching", n_pairs=3, seed=3))
    _check("n_pairs truncation leaves sequences UNCOVERED",
           int(u.covered.sum()) == 6 and len(u.pairs) == 3,
           f"{int(u.covered.sum())}/{n} covered -- the loss is blind on the other "
           f"{n - int(u.covered.sum())}")
    _check("dtypes", u.pairs.dtype == torch.long and u.covered.dtype == torch.bool
           and u.flipped.dtype == torch.bool)

    # --- MORE DATA LOWERS THE MARGIN. Most people predict the opposite. --------
    # The intuition "more preference data pushes the margin further" is right ONLY when
    # the data are clean and separable: then every extra pair is another constraint in the
    # same direction and the margin runs away faster. Under noise it is BACKWARDS. At
    # eps=0.10 some pairs are flipped, so extra comparisons CONFLICT, Ford's condition
    # starts to hold, and the finite MLE pulls the margin DOWN. This is D1's theorem
    # showing up as an empirical curve: comparisons do not add pressure, they add
    # CONSTRAINTS, and constraints are what make the optimum finite.
    print("      final margin vs number of comparisons (eps=0.10, matching), 6 seeds:")
    ends = []
    for s in range(6):
        row = []
        for shape, npairs in (("matching", 2), ("matching", None), ("all_pairs", None)):
            d = make_dataset(env, DataCfg(shape=shape, eps=0.10, n_pairs=npairs, seed=s))
            h = train_dpo(env, d, RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short))
            row.append((len(d.pairs), float(h.margin[-1])))
        ends.append(row)
        print("        seed {}: ".format(s) + "   ".join(
            f"{k:3d} pairs -> {m:.4f}" for k, m in row))
    _check("more comparisons => SMALLER final margin, in every seed",
           all(r[0][1] > r[1][1] > r[2][1] for r in ends),
           f"6/6 seeds ordered {ends[0][0][0]}p > {ends[0][1][0]}p > {ends[0][2][0]}p; "
           f"e.g. seed 0: {ends[0][0][1]:.4f} > {ends[0][1][1]:.4f} > {ends[0][2][1]:.4f}. "
           f"NOTE step-by-step monotonicity along the full ladder holds in only 3/6 seeds "
           f"-- assert the endpoint ordering, which is robust, not the fragile version")


def stage_panel(ctx: Ctx) -> None:
    """train_dpo(): the exact initial conditions, and every metric inside its own range."""
    env = ctx.env
    h = train_dpo(env, ctx.d_healthy, RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short))

    _check("panel has n_steps+1 entries (step 0 measured BEFORE any update)",
           len(h.loss) == ctx.short + 1)
    _check("initial implicit reward is EXACTLY 0", h.margin[0] == 0.0 and h.r_absmax[0] == 0.0,
           "pi and pi_ref seeded from the same checkpoint (D3's SFT result)")
    _check("initial loss is EXACTLY +log 2", abs(h.loss[0] - math.log(2)) < 1e-6,
           f"{h.loss[0]:.6f} vs {math.log(2):.6f} -- the asymmetric probe for a sign flip "
           "(four consecutive days of a missing minus)")
    _check("initial KL is exactly 0", h.kl[0] == 0.0)
    _check("cap = logit(1-eps)", abs(h.cap - logit(0.75)) < 1e-12,
           f"{h.cap:.6f} at eps=0.25")

    # RANGES. D4 bug 4: three metrics were outside the range of their own quantity.
    kl_bound = point_mass_kl_bound(env.seq_logprobs(env.random_theta(0, 0.7)))
    _check("acc in [0,1]", h.acc.min() >= 0 and h.acc.max() <= 1)
    _check("ent >= 0", h.ent.min() >= 0, f"min {h.ent.min():.6f}")
    _check("elen in [1, t_max]", h.elen.min() >= 1.0 and h.elen.max() <= env.t_max,
           f"[{h.elen.min():.4f}, {h.elen.max():.4f}] in [1, {env.t_max}]")
    _check("kl in [0, point-mass bound]", h.kl.min() >= 0 and h.kl.max() <= kl_bound + 1e-9,
           f"max {h.kl.max():.6f} <= {kl_bound:.6f} -- KL is BOUNDED in a finite space")
    _check("nll_w > 0", h.nll_w.min() > 0)
    _check("offdata_mass in [0,1]", h.offdata_mass.min() >= -1e-9
           and h.offdata_mass.max() <= 1 + 1e-9)
    _check("full coverage -> offdata_mass == 0", abs(h.offdata_mass).max() < 1e-6)

    # the returned theta is the FINAL one, not a stale clone
    lp_final = env.seq_logprobs(h.theta)
    _check("returned theta reproduces the last logged row",
           abs(gold_of(lp_final, ctx.d_healthy.gold_r) - h.gold[-1]) < 1e-6,
           "W3 D1's argmax-lag, fifth sighting")

    # ACC IS THE METRIC THAT REPORTS NOTHING
    sep = train_dpo(env, ctx.d_separable,
                    RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.long))
    first = int(np.argmax(sep.acc >= 1.0 - 1e-12))
    _check("acc saturates almost immediately and then reports nothing",
           first < 60 and sep.acc[-1] == 1.0,
           f"acc hit 1.000 at step {first} of {ctx.long}; the margin then ran from "
           f"{sep.margin[first]:.4f} to {sep.margin[-1]:.4f}")


def stage_chain(ctx: Ctx) -> None:
    """The chained regression: BT-fit-then-tilt -> DPO -> cDPO -> IPO -> SimPO, one path."""
    env = ctx.env
    data = ctx.d_healthy
    beta = ctx.BETA

    # --- THE ANCHOR: DPO == tilt of the BT MLE (D2's headline) ---------------
    cfg = RunCfg(variant="DPO", beta=beta, lr=ctx.LR, n_steps=ctx.bt_steps)
    h = train_dpo(env, data, cfg)
    lp_dpo = env.seq_logprobs(h.theta)

    theta_ref = env.random_theta(cfg.seed, cfg.ref_scale)
    lp_ref = env.seq_logprobs(theta_ref).detach()
    r_bt = bt_fit(data.pairs, env.n_seqs, steps=ctx.bt_steps, lr=ctx.LR)
    lp_tilt = tilt(lp_ref, r_bt, beta)

    dev = float((lp_dpo - lp_tilt).abs().max())
    tol = 5e-3 if ctx.fast else 1e-6
    _check("DPO == tilt(BT MLE) elementwise", dev < tol,
           f"max |dev| = {dev:.3e} over all {env.n_seqs} sequences, two algorithms "
           "sharing no code path")
    _check("the tilt is normalised (the simplex fixes the gauge)",
           abs(float(lp_tilt.exp().sum()) - 1.0) < 1e-6)

    # NON-VACUITY: the wrong beta looks like the same algorithm and is not.
    worst = 0.0
    for b in (0.25, 1.0, 2.0):
        worst = max(worst, float((tilt(lp_ref, r_bt, b) - lp_tilt).abs().max()))
    _check("wrong-beta tilts are far away", worst > 1e3 * max(dev, 1e-16),
           f"nearest wrong beta differs by {worst:.4e} = {worst / max(dev, 1e-16):.2e}x "
           "the agreement -- the check is not vacuous")

    # --- the grid reproduces five independent direct implementations ---------
    lp = env.seq_logprobs(env.random_theta(1, 0.9)).detach()
    specs = named_specs(beta=beta, eps=0.1, tau=2.0, gamma=0.5)
    rungs = {
        "DPO": (specs["DPO"], _direct_dpo(lp, lp_ref, data.pairs, beta)),
        "cDPO": (specs["cDPO"], _direct_cdpo(lp, lp_ref, data.pairs, beta, 0.1)),
        "IPO": (specs["IPO"], _direct_ipo(lp, lp_ref, data.pairs, 2.0)),
        "SimPO": (specs["SimPO"], _direct_simpo(lp, env.lengths, data.pairs, beta, 0.5)),
    }
    # NOTE the shape trap: variant_loss returns a 0-d REDUCED tensor, the _direct_*
    # helpers return one entry PER PAIR. float() on the latter raises. Reduce, don't cast.
    for name, (spec, direct) in rungs.items():
        got = variant_loss(spec, lp, lp_ref, env.lengths, data.pairs)
        _check(f"grid reproduces {name}", abs(float(got - direct.mean())) < 1e-6,
               f"{float(got):.12f} vs {float(direct.mean()):.12f} "
               f"(direct returns {tuple(direct.shape)}, the loss returns a scalar)")

    # NON-VACUITY for the rung most likely to look identical under an ignored switch:
    # cDPO differs from DPO ONLY by eps. Ignore eps and the two are the same function.
    s0 = replace(specs["cDPO"], eps=0.0)
    same = variant_loss(s0, lp, lp_ref, env.lengths, data.pairs)
    dpo = variant_loss(specs["DPO"], lp, lp_ref, env.lengths, data.pairs)
    diff = variant_loss(specs["cDPO"], lp, lp_ref, env.lengths, data.pairs)
    _check("cDPO(eps=0) IS DPO, bitwise", float(same) == float(dpo),
           f"{float(same):.15f}")
    _check("cDPO(eps=0.1) is NOT DPO", abs(float(diff - dpo)) > 1e-3,
           f"|difference| = {abs(float(diff - dpo)):.6f} -- so the eps switch is "
           "load-bearing and the agreement above is evidence, not a tautology")

    # --- IPO's optimum is (2f-1)tau exactly, and it COMPRESSES ---------------
    sep = ctx.d_separable
    hi = train_dpo(env, sep, RunCfg(variant="IPO", beta=1.0, tau=2.0, lr=0.25,
                                    n_steps=ctx.long))
    target = closed_form_optimum("squared_target", 1.0, specs["IPO"])
    _check("IPO converges to (2f-1)tau", abs(hi.margin[-1] - target) < 5e-3,
           f"{hi.margin[-1]:.6f} vs {target:.6f} -- finite where DPO is not")


def stage_modes(ctx: Ctx) -> None:
    """diagnose_dpo(): all four modes fire on their own fixtures, and SILENT on health."""
    env = ctx.env
    fired: dict[str, list[str]] = {}

    # --- HEALTH FIRST. The silence is the harder half. -----------------------
    healthy = train_dpo(env, ctx.d_healthy,
                        RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short,
                               sft_weight=1.0))
    out = diagnose_dpo(healthy)
    _check("SILENT on a healthy run", out == [],
           f"fired {[n for n, _ in out]} -- a diagnostic that always fires is not a "
           "diagnostic (W4 D4's -20% entropy threshold sat inside a healthy range)")

    # --- M1: divergence -------------------------------------------------------
    h1 = train_dpo(env, ctx.d_separable,
                   RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.long))
    n1 = [n for n, _ in diagnose_dpo(h1)]
    fired["divergence"] = n1
    _check("M1 fires on the matching", "divergence" in n1, "; ".join(
        r for n, r in diagnose_dpo(h1) if n == "divergence"))

    # --- M2: both logprobs fall, and the SFT term is the fix ------------------
    h2 = train_dpo(env, ctx.d_healthy,
                   RunCfg(beta=2.0, lr=ctx.LR, n_steps=ctx.short))
    n2 = [n for n, _ in diagnose_dpo(h2)]
    fired["both_logprobs_fall"] = n2
    _check("M2 fires under plain DPO", "both_logprobs_fall" in n2, "; ".join(
        r for n, r in diagnose_dpo(h2) if n == "both_logprobs_fall"))
    h2s = train_dpo(env, ctx.d_healthy,
                    RunCfg(beta=2.0, lr=ctx.LR, n_steps=ctx.short, sft_weight=1.0))
    _check("the SFT term removes it", "both_logprobs_fall" not in
           [n for n, _ in diagnose_dpo(h2s)],
           f"logp_w {h2.logp_w[-1] - h2.logp_w[0]:+.4f} (DPO) -> "
           f"{h2s.logp_w[-1] - h2s.logp_w[0]:+.4f} (DPO+SFT)")

    # --- M3: length exploitation ---------------------------------------------
    h3 = train_dpo(ctx.ladder, ctx.d_length,
                   RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short))
    n3 = [n for n, _ in diagnose_dpo(h3)]
    fired["length_exploitation"] = n3
    _check("M3 fires on the length confound", "length_exploitation" in n3, "; ".join(
        r for n, r in diagnose_dpo(h3) if n == "length_exploitation"))

    # --- M4: off-distribution drift -------------------------------------------
    h4 = train_dpo(env, ctx.d_uncovered,
                   RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short))
    n4 = [n for n, _ in diagnose_dpo(h4)]
    fired["off_distribution_drift"] = n4
    _check("M4 fires when the data cover 6 of 15 sequences",
           "off_distribution_drift" in n4, "; ".join(
               r for n, r in diagnose_dpo(h4) if n == "off_distribution_drift"))

    # --- each mode is SPECIFIC to its own fixture ------------------------------
    _check("M3 does not fire on the length-neutral divergence fixture",
           "length_exploitation" not in fired["divergence"])
    _check("M4 does not fire on the fully covered fixtures",
           "off_distribution_drift" not in fired["both_logprobs_fall"]
           and "off_distribution_drift" not in fired["length_exploitation"])

    # --- reason strings carry numbers -----------------------------------------
    for h in (h1, h2, h3, h4):
        for name, reason in diagnose_dpo(h):
            _check(f"{name}: the reason carries a number",
                   any(ch.isdigit() for ch in reason), reason)


def stage_checkpoint(ctx: Ctx) -> None:
    """select_checkpoint(): the estimator, graded by a gold it never sees."""
    env = ctx.env
    h = train_dpo(env, ctx.d_uncovered,
                  RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short))
    k = select_checkpoint(h)
    best = int(np.argmax(h.gold))
    _check("selection never sees gold and is not the oracle", 0 <= k < len(h.gold))
    _check("the estimator beats shipping the last checkpoint",
           h.gold[k] >= h.gold[-1] - 1e-9,
           f"gold: selected(step {k}) {h.gold[k]:.4f} | last {h.gold[-1]:.4f} | "
           f"oracle(step {best}) {h.gold[best]:.4f} | shipping last costs "
           f"{h.gold.max() - h.gold[-1]:.4f}")
    _check("it stops where the margin reaches the cap",
           k == len(h.gold) - 1 or h.margin[k] >= h.cap,
           f"margin[{k}] = {h.margin[k]:.4f} vs cap {h.cap:.4f}")

    clean = train_dpo(env, ctx.d_separable,
                      RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short))
    _check("clean data supply no cap, so no natural stopping rule",
           not math.isfinite(clean.cap) and select_checkpoint(clean) == len(clean.gold) - 1,
           "that is not a limitation of the code, it is M1")


def stage_collapse(ctx: Ctx) -> None:
    """THE PREDICTION: a beta sweep and an eps sweep trace the SAME curve.

    The prediction being tested was made BEFORE it was run: beta and eps enter the optimum
    only through Dhat* = logit(1-eps)/beta, so two sweeps that move different knobs to the
    same Dhat* must land at the same KL. This is the D1 scale identity cashed out as a
    falsifiable experiment, and it is the reason "we tuned beta" and "we cleaned the labels"
    can be the same intervention wearing different clothes.
    """
    env = ctx.env
    data = ctx.d_separable          # clean & separable -> f = 1 exactly, so M* = logit(1-eps)

    def kl_at(beta: float, eps: float) -> tuple[float, float, float, int]:
        target = closed_form_optimum("logsigmoid_smoothed", 1.0,
                                     Spec("x", "ref_sum", "logsigmoid_smoothed",
                                          beta=beta, eps=eps))
        # BUDGET, NOT TIME: stop_margin exits as soon as the target is hit, so a large
        # n_steps costs nothing on the easy rungs. It has to be large because the cost of
        # a rung grows like 1/beta^2 -- see the ratio check below.
        cfg = RunCfg(variant="cDPO", beta=beta, eps=eps, lr=ctx.LR, n_steps=20_000,
                     stop_margin=0.995 * target)
        h = train_dpo(env, data, cfg)
        return target / beta, h.kl[-1], h.margin[-1] / target, len(h.margin)

    # matched Dhat* = logit(1-eps)/beta on both axes
    betas = (2.0, 1.0, 0.5) if ctx.fast else (2.0, 1.0, 0.5, 0.25)
    beta_arm = [(b, 0.10) for b in betas]
    eps_arm = []
    for b, _ in beta_arm:
        # choose eps so that logit(1-eps)/0.5 == logit(0.9)/b
        target = logit(0.90) * 0.5 / b
        eps_arm.append((0.5, 1.0 - 1.0 / (1.0 + math.exp(-target))))

    rows_b = [kl_at(b, e) for b, e in beta_arm]
    rows_e = [kl_at(b, e) for b, e in eps_arm]

    for (xb, klb, fb, _), (xe, kle, fe, _) in zip(rows_b, rows_e):
        _check("matched Dhat* on both arms", abs(xb - xe) < 1e-6, f"{xb:.6f} vs {xe:.6f}")
        _check(f"both arms reached the target margin (Dhat*={xb:.4f})",
               abs(fb - 1.0) < 0.02 and abs(fe - 1.0) < 0.02,
               f"{fb:.6f} / {fe:.6f} of target")
        _check(f"KL collapses onto ONE curve at Dhat*={xb:.4f}",
               abs(klb - kle) < 2e-3,
               f"beta-sweep KL {klb:.6f} vs eps-sweep KL {kle:.6f} "
               f"(|diff| {abs(klb - kle):.2e})")

    kls = [r[1] for r in rows_b]
    _check("and the curve is non-trivial (KL varies over the sweep)",
           max(kls) - min(kls) > 0.3,
           f"KL {min(kls):.4f} -> {max(kls):.4f} across Dhat* "
           f"{rows_b[0][0]:.4f} -> {rows_b[-1][0]:.4f} -- so the collapse is not "
           "the trivial statement that everything is equal")

    # --- THE PRICE OF A SMALL BETA: steps-to-target grows like 1/beta^2 -------
    # Halving beta doubles Dhat* AND halves the gradient scale (dL/dDhat ~ beta*sigma(.)),
    # so the traverse costs 2 x 2 = 4x. D4 found margin ~ log t at fixed beta; this is the
    # same law read along the OTHER axis, and it is the practical one: a "safer" small beta
    # under a FIXED step budget does not give you a gentler optimum, it gives you an
    # UNCONVERGED one, and the panel reports that as a small margin -- which looks healthy.
    steps = [r[3] for r in rows_b]
    ratios = [steps[i + 1] / steps[i] for i in range(len(steps) - 1)]
    print("      steps to 0.995*M*: " + "  ".join(
        f"beta={b:<5} {s:5d}" for (b, _), s in zip(beta_arm, steps)))
    _check("cost of halving beta is 4x, not 2x (steps ~ 1/beta^2)",
           all(abs(r - 4.0) < 0.25 for r in ratios),
           "ratios " + ", ".join(f"{r:.2f}" for r in ratios) +
           f" -- steps {steps[0]} -> {steps[-1]} while Dhat* only went "
           f"{rows_b[0][0]:.4f} -> {rows_b[-1][0]:.4f} ({rows_b[-1][0] / rows_b[0][0]:.0f}x)")


def stage_seeds(ctx: Ctx) -> None:
    """>=3 seeds everywhere -- and what to do when 3 seeds are NOT enough.

    W3's rule was "claim a result only if the effect beats seed noise". This stage runs
    into its corollary, which is the part nobody says out loud: when the effect does NOT
    beat seed noise, that is a fact about YOUR INSTRUMENT AND YOUR DESIGN, not about the
    knob. The same runs, the same seeds, two metrics -- and one of them resolves beta
    cleanly while the other cannot see it at all.
    """
    env = ctx.env
    base_d = DataCfg(shape="all_pairs", eps=0.25)
    base_r = RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.short)
    betas = [0.25, 0.5, 1.0, 2.0]

    # --- design A: UNPAIRED. The seed rerolls the dataset AND the init together. --------
    res = ablate(env, base_d, base_r, {"beta": betas}, seeds=ctx.seeds)
    _check("ablate ran every cell over >=3 seeds", len(res) == 4 and len(ctx.seeds) >= 3)
    for key, m in sorted(res.items(), key=lambda kv: kv[0][1]):
        print(f"      UNPAIRED beta={key[1]:<5} KL {m['kl'][0]:.4f} +-{m['kl'][1]:.4f}  "
              f"margin {m['margin'][0]:.4f} +-{m['margin'][1]:.4f}  "
              f"gold {m['gold'][0]:.4f} +-{m['gold'][1]:.4f}")
    eff_u = max(m["kl"][0] for m in res.values()) - min(m["kl"][0] for m in res.values())
    spr_u = max(m["kl"][1] for m in res.values())

    # --- design B: PAIRED. Dataset FIXED; only the init/reference seed moves. -----------
    # The dataset is a nuisance variable: rerolling it changes the TASK, so its variance
    # lands on every cell and hides the knob. Holding it fixed is free variance reduction.
    data = make_dataset(env, replace(base_d, seed=0))
    paired: dict[float, dict[str, tuple[float, float]]] = {}
    for b in betas:
        kls, mgs = [], []
        for s in ctx.seeds:
            h = train_dpo(env, data, replace(base_r, beta=b, seed=s))
            kls.append(float(h.kl[-1]))
            mgs.append(float(h.margin[-1]))
        paired[b] = {"kl": (float(np.mean(kls)), max(kls) - min(kls)),
                     "margin": (float(np.mean(mgs)), max(mgs) - min(mgs))}
        print(f"      PAIRED   beta={b:<5} KL {paired[b]['kl'][0]:.4f} "
              f"+-{paired[b]['kl'][1]:.4f}  margin {paired[b]['margin'][0]:.4f} "
              f"+-{paired[b]['margin'][1]:.4f}  Dhat {paired[b]['margin'][0] / b:.4f}")

    eff_p = (max(m["kl"][0] for m in paired.values())
             - min(m["kl"][0] for m in paired.values()))
    spr_p = max(m["kl"][1] for m in paired.values())
    _check("pairing shrinks the seed spread", spr_p < spr_u,
           f"worst KL spread {spr_u:.4f} (unpaired) -> {spr_p:.4f} (paired) "
           f"= {spr_u / spr_p:.2f}x tighter, from a change to the DESIGN and not to the code")

    # THE NEGATIVE, STATED: KL cannot resolve beta here even after pairing.
    _check("KL still CANNOT resolve beta at 3 seeds -- and we say so", spr_p > eff_p / 3.0,
           f"beta effect on KL {eff_p:.4f} vs worst seed spread {spr_p:.4f}: the intervals "
           f"OVERLAP, so 'beta lowers KL' is NOT supported by this experiment. The honest "
           f"report is 'underpowered at 3 seeds', not a number")

    # THE POSITIVE, ON THE SAME RUNS: margin resolves beta cleanly.
    m_eff = (max(m["margin"][0] for m in paired.values())
             - min(m["margin"][0] for m in paired.values()))
    m_spr = max(m["margin"][1] for m in paired.values())
    _check("margin DOES resolve beta on the very same runs", m_eff > 3.0 * m_spr,
           f"beta effect on margin {m_eff:.4f} vs worst spread {m_spr:.4f} "
           f"= {m_eff / max(m_spr, 1e-12):.1f}x -- same runs, same seeds, different "
           f"instrument. The metric decides whether the ablation has power")

    # THE UNITS TRAP, IN THE DATA: margin rises with beta, Dhat = margin/beta falls.
    mg = [paired[b]["margin"][0] for b in betas]
    dh = [paired[b]["margin"][0] / b for b in betas]
    _check("margin and Dhat move in OPPOSITE directions across the same sweep",
           mg[-1] > mg[0] and dh[-1] < dh[0],
           f"margin {mg[0]:.4f} -> {mg[-1]:.4f} (up) while Dhat = margin/beta "
           f"{dh[0]:.4f} -> {dh[-1]:.4f} (down) over beta {betas[0]} -> {betas[-1]}: "
           f"report the wrong one and you report the opposite conclusion")

    # --- W4 D4 claimed seed spread GROWS with optimisation pressure. Test it properly. --
    # Paired again (init only), so this measures optimisation and not the task reroll.
    # Two findings, and the second one retracts a W4 claim:
    #   margin CONTRACTS (every seed is pulled to the same data-determined optimum) and
    #   KL GROWS (measured against each seed's OWN pi_ref, so paths diverge). Both are
    #   stable whether you use 3 seeds or 5.
    #   gold's amplification factor is 5.96x on seeds (0,1,2) and 1.02x on seeds (0..4) --
    #   THE SAME RUNS. Seed 4 starts low (+0.1217 vs ~+0.30) and stays low, so the 10-step
    #   spread is already large once you sample it. W4 D4's claim was measured on 3 seeds
    #   in the gold column and is an artifact of the seed SET, not a property of training.
    #   The reproducible fact here is the INSTABILITY, so that is what gets asserted.
    ns = (10, ctx.long)
    seedsets = {"3 seeds": tuple(ctx.seeds), "5 seeds": tuple(range(5))}
    allseeds = sorted(set().union(*seedsets.values()))
    cache = {}
    for s in allseeds:
        for n in ns:
            h = train_dpo(env, data, replace(base_r, beta=0.5, n_steps=n, seed=s))
            cache[(s, n)] = (float(h.gold[-1]), float(h.margin[-1]), float(h.kl[-1]))

    amp: dict[str, dict[str, float]] = {}
    for label, ss in seedsets.items():
        amp[label] = {}
        for i, nm in enumerate(("gold", "margin", "kl")):
            a = [cache[(s, ns[0])][i] for s in ss]
            b = [cache[(s, ns[1])][i] for s in ss]
            sa, sb = max(a) - min(a), max(b) - min(b)
            amp[label][nm] = sb / max(sa, 1e-12)
            print(f"      {label} spread({nm:<6}) {sa:.6f} @{ns[0]} -> {sb:.6f} "
                  f"@{ns[1]} = {sb / max(sa, 1e-12):8.2f}x")

    _check("margin spread CONTRACTS with optimisation, at BOTH seed counts",
           all(amp[l]["margin"] < 0.5 for l in seedsets),
           " and ".join(f"{l}: {amp[l]['margin']:.2f}x" for l in seedsets) +
           " -- the optimum is set by the data, so running longer makes seeds AGREE")
    _check("KL spread GROWS with optimisation, at BOTH seed counts",
           all(amp[l]["kl"] > 10.0 for l in seedsets),
           " and ".join(f"{l}: {amp[l]['kl']:.0f}x" for l in seedsets) +
           " -- KL is measured against each seed's own pi_ref, so the further you travel "
           "the more the paths disagree")
    ratio = max(amp[l]["gold"] for l in seedsets) / min(amp[l]["gold"] for l in seedsets)
    _check("gold's amplification factor is NOT stable in the seed set", ratio > 3.0,
           " vs ".join(f"{l}: {amp[l]['gold']:.2f}x" for l in seedsets) +
           f" = a {ratio:.1f}x disagreement on the SAME runs. Two extra seeds flip the "
           f"conclusion from 'optimisation amplifies seed noise 6x' to 'it does not move "
           f"it at all'. W4 D4's version of this claim was a 3-seed artifact in the gold "
           f"column; >=3 seeds is a FLOOR, and it is not enough for a tail-sensitive metric")


def stage_inversion(ctx: Ctx) -> None:
    """W4's headline was proxy up / gold down. Does DPO have an analogue?

    It does, and it is sharper than W4's, because DPO's proxy is TRAIN ACCURACY -- the
    number every practitioner already watches. In all three broken fixtures accuracy goes
    0.000 -> 1.000 while gold ends below its own peak. The control is what makes this a
    result rather than an anecdote: on the healthy fixture gold ends AT its peak, so the
    inversion is a property of the BROKEN DATA and not of running DPO for a long time.
    """
    rows = []
    for name, e, data in (("uncovered (M4)", ctx.env, ctx.d_uncovered),
                          ("length confound (M3)", ctx.ladder, ctx.d_length),
                          ("separable (M1)", ctx.env, ctx.d_separable)):
        h = train_dpo(e, data, RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.long))
        peak = int(np.argmax(h.gold))
        cost = float(h.gold.max() - h.gold[-1])
        rows.append((name, h, peak, cost))
        print(f"      {name:<22} acc {h.acc[0]:.3f} -> {h.acc[-1]:.3f} | "
              f"gold {h.gold[0]:+.4f} -> peak {h.gold.max():+.4f} @{peak}/{len(h.gold)} -> "
              f"{h.gold[-1]:+.4f} | shipping last costs {cost:.4f}")

    for name, h, peak, cost in rows:
        _check(f"{name}: the proxy rises to 1.000 while gold ends below its peak",
               h.acc[-1] >= 1.0 - 1e-12 and h.acc[-1] > h.acc[0] and cost > 0.05,
               f"acc {h.acc[0]:.3f} -> {h.acc[-1]:.3f} (+{h.acc[-1] - h.acc[0]:.3f}) but "
               f"gold peaked at step {peak} of {len(h.gold)} "
               f"({100 * peak / len(h.gold):.1f}% in) and gave back {cost:.4f}")

    # THE CONTROL. Without this the stage proves only that gold is noisy.
    hc = train_dpo(ctx.env, ctx.d_healthy,
                   RunCfg(beta=ctx.BETA, lr=ctx.LR, n_steps=ctx.long, sft_weight=1.0))
    ccost = float(hc.gold.max() - hc.gold[-1])
    worst = min(c for _, _, _, c in rows)
    print(f"      {'CONTROL: healthy':<22} acc {hc.acc[0]:.3f} -> {hc.acc[-1]:.3f} | "
          f"gold {hc.gold[0]:+.4f} -> peak {hc.gold.max():+.4f} "
          f"@{int(np.argmax(hc.gold))}/{len(hc.gold)} -> {hc.gold[-1]:+.4f} | "
          f"shipping last costs {ccost:.4f}")
    _check("CONTROL: the healthy run does NOT invert", ccost < 0.5 * worst,
           f"healthy gives back {ccost:.4f} vs {worst:.4f} for the mildest broken "
           f"fixture; gold peaks at step {int(np.argmax(hc.gold))} of {len(hc.gold)}. "
           f"So the inversion tracks the DATA, not the training length")

    # The strongest form: M3 ends below where it STARTED, not merely below its peak.
    _, h3, _, _ = rows[1]
    _check("length confound: gold ends below its STARTING value",
           h3.gold[-1] < h3.gold[0],
           f"gold {h3.gold[0]:+.4f} -> {h3.gold[-1]:+.4f} "
           f"({h3.gold[-1] - h3.gold[0]:+.4f}): DPO training made the model WORSE than the "
           f"reference it started from, at 100% train accuracy, and the loss went down "
           f"the whole way")


STAGES = [
    ("port", stage_port),
    ("dataset", stage_dataset),
    ("panel", stage_panel),
    ("chain", stage_chain),
    ("modes", stage_modes),
    ("checkpoint", stage_checkpoint),
    ("collapse", stage_collapse),
    ("seeds", stage_seeds),
    ("inversion", stage_inversion),
]


# ---- runner (copy verbatim; nothing below needs editing) -------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, fn in STAGES:
            print(f"  {name:<12} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
