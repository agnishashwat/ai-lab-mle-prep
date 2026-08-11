"""W6 D2 -- RL with verifiable rewards: a verifier is a SET, not a rate.

THE THESIS
    A verifier used as an EVAL is summarised by a RATE.
    A verifier used as a REWARD is summarised by a SET.
    Optimization does not sample a checker's errors in proportion to their
    probability -- it SEARCHES for them.

    INCOMPLETENESS  (false negatives, C \\ V)  costs you COMPUTE
    UNSOUNDNESS     (false positives, V \\ C)  costs you the RUN

Everything is checked against exact enumeration. The environment, the sequence
table, the exact expectations and `group_advantage` are IMPORTED from D1's
`grpo.py` -- same `TokenEnv(vocab=3, t_max=3)`, 7 states, 15 terminating
sequences -- so today adds one structural thing and one only:

    the GOLD correctness mask C and the VERIFIER's accept mask V are two
    SEPARATE boolean vectors over the 15 sequences.

That single split is what turns "soundness" from an adjective into a checkable
set relation (V subset of C), and it is what lets `train_rlvr` reward the policy
with V while scoring it with C -- which is the only way a hacked run and a
healthy run can be told apart at all (see `stage_selection`: every other scalar
on the panel is bit-identical between them).

THE NUMPY/TORCH BOUNDARY -- unchanged from D1, and it is not arbitrary:
    rewards, masks and advantages are NUMPY   (constants w.r.t. theta)
    log-probabilities are TORCH               (they carry the graph)

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-06-grpo-rlvr-reasoning/code/rlvr.py
  one stage           ...  rlvr.py <stage>
  a few stages        ...  rlvr.py rates selection
  small/fast fixtures ...  rlvr.py --fast
  list the stages     ...  rlvr.py --list
"""

from __future__ import annotations

import math
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import torch

from grpo import (                      # D1, and this import IS the point
    TokenEnv,
    SeqTable,
    build_seq_table,
    seq_logprobs_np,
    seq_token_logprobs_torch,
    all_score_functions,
    exact_expectation,
    exact_entropy,
    exact_kl,
    group_advantage,
    log_softmax_np,
    softmax_np,
)


# ---------------------------------------------------------------------------
# The checker: two masks, not one (provided)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Checker:
    """`gold[p, y]` = is sequence y actually CORRECT for prompt p.
       `accept[p, y]` = does the checker SAY it is.

    The whole day is the difference between these two arrays. `gold` is never
    visible to training -- `train_rlvr` rewards with `accept` and only RECORDS
    `gold`, exactly as a real run can only see its verifier and its held-out eval.
    """
    gold: np.ndarray        # (P, n_seqs) float64 in {0.0, 1.0}
    accept: np.ndarray      # (P, n_seqs) float64 in {0.0, 1.0}

    @property
    def n_prompts(self) -> int:
        return self.gold.shape[0]

    @property
    def fp(self) -> np.ndarray:
        """V \\ C -- the unsoundness set. The entire reward-hacking surface."""
        return self.accept * (1.0 - self.gold)

    @property
    def fn(self) -> np.ndarray:
        """C \\ V -- the incompleteness set. Priced in rollouts, not in correctness."""
        return self.gold * (1.0 - self.accept)

    @property
    def sound(self) -> bool:
        """V subset of C. The ONE property that survives being optimized against."""
        return bool(np.all(self.fp == 0.0))

    @property
    def complete(self) -> bool:
        """C subset of V. Costs compute when false, never correctness."""
        return bool(np.all(self.fn == 0.0))


@dataclass(frozen=True)
class RlvrTask:
    env: TokenEnv
    tab: SeqTable
    checker: Checker

    @property
    def n_prompts(self) -> int:
        return self.checker.n_prompts


# ---------------------------------------------------------------------------
# Closed forms the day derives (provided -- these are the ground truth)
# ---------------------------------------------------------------------------

def dead_fraction_levels(q: np.ndarray, G: int) -> float:
    """P(a group of G i.i.d. draws carries ZERO signal) = sum_j q_j^G.

    D1 derived the L=2 case, P = p^G + (1-p)^G. The generalisation is one line:
    a group is dead iff every member lands in the SAME reward level.

    Identify it: sum_j q_j^G = exp( -(G-1) * H_G(q) ) with H_G the RENYI ENTROPY
    of order G. So the effective batch size is set by the Renyi entropy of the
    REWARD's output distribution -- and the verifier is the only thing that
    controls that distribution.
    """
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(q ** G))


def renyi_entropy(q: np.ndarray, order: float) -> float:
    """H_alpha(q) = log(sum q^alpha) / (1 - alpha), in nats."""
    q = np.asarray(q, dtype=np.float64)
    return float(np.log(np.sum(q ** order)) / (1.0 - order))


def g_for_alive(p: float, target: float = 0.9) -> int:
    """Smallest G with 1 - p^G - (1-p)^G >= target. D1's closed form is the
    single-sided approximation G ~ ln(1-target)/ln(1-p); this is the exact
    integer, which is what you would actually put in a config."""
    G = 1
    while 1.0 - p ** G - (1.0 - p) ** G < target:
        G += 1
        if G > 100_000:
            raise ValueError("no feasible G")
    return G


def p_observed(p: float, alpha: float, gamma: float) -> float:
    """The pass rate the TRAINER sees, given the true rate and the two error rates.

        p_obs = p(1-alpha) + (1-p)gamma

    This is the composition that prices a bad answer-extractor in rollouts:
    p_obs is the ONLY input to the dead-group law, so alpha does not "add noise",
    it moves the operating point.
    """
    return p * (1.0 - alpha) + (1.0 - p) * gamma


# ---------------------------------------------------------------------------
# Config / history
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RlvrCfg:
    steps: int = 300
    group: int = 8
    lr: float = 0.5
    beta: float = 0.02
    seed: int = 0
    n_prompts: int = 6
    init_scale: float = 0.5
    adv_mode: str = "centered"


@dataclass
class RlvrHistory:
    """The instrument panel. `v_pass` is what training can see; `gold_pass` is the
    held-out eval and is the ONLY column that separates a hacked run from a
    healthy one -- see `stage_selection` for the proof that the others cannot."""
    step: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_pass: np.ndarray = field(default_factory=lambda: np.zeros(0))     # EXACT
    gold_pass: np.ndarray = field(default_factory=lambda: np.zeros(0))  # EXACT, held out
    dead_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    adv_absmean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kl: np.ndarray = field(default_factory=lambda: np.zeros(0))         # EXACT
    entropy: np.ndarray = field(default_factory=lambda: np.zeros(0))    # EXACT, nats
    loss: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# Exercises -- the four TODO bodies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rates:
    p_true: float       # P(output is actually correct)
    p_obs: float        # P(the checker accepts)     <- what training sees
    alpha: float        # P(reject | correct)        = INCOMPLETENESS
    gamma: float        # P(accept | incorrect)      = UNSOUNDNESS
    precision: float    # P(correct | accepted)
    recall: float       # P(accepted | correct)      = 1 - alpha


def verifier_rates(gold: np.ndarray, accept: np.ndarray,
                   probs: np.ndarray) -> Rates:
    """The checker's confusion UNDER A POLICY. TODO 1.

    `gold`, `accept` are (n_seqs,) in {0,1}; `probs` is (n_seqs,) and sums to 1.

    THE LEVEL TRAP, and it is the tenth sighting of the family waiting to happen:
    every rate here is over PROBABILITY MASS under `probs`, not over the 15
    sequences. Counting sequences gives you a policy-INDEPENDENT number, which
    looks like a property of the verifier and is not one. A verifier's confusion
    is a joint property of the checker AND the policy being trained, and that is
    exactly why "our verifier is 97% accurate" is not a well-formed sentence.

    Contract:
        p_true    = sum_y probs[y] gold[y]
        p_obs     = sum_y probs[y] accept[y]
        alpha     = P(reject | correct)      -- 0.0 when p_true    == 0
        gamma     = P(accept | incorrect)    -- 0.0 when 1 - p_true == 0
        precision = P(correct | accepted)    -- 1.0 when p_obs     == 0
        recall    = 1 - alpha

    Guard every denominator EXPLICITLY. D1 lost a stage to `np.divide(..., where=)`
    without `out=`, which returns uninitialised memory; a plain `if` is the fix.
    """
    
    p_true = (probs * gold).sum()
    p_obs = (probs * accept).sum()
    p_reject = np.ones_like(accept) - accept
    if p_true > 0:
        alpha = (p_reject * probs * gold).sum()/p_true
    else:
        alpha = 0
    if p_true < 1:
        gamma = (accept * probs * (1-gold)).sum()/(1-p_true)
    else:
        gamma = 0
    if p_obs == 0:
        precision = 1
    else:
        precision = (probs * gold * accept).sum()/p_obs
    recall = 1-alpha
    rate = Rates(p_true, p_obs, alpha, gamma, precision, recall)
    
    return rate


def hack_share(gold: np.ndarray, accept: np.ndarray, probs: np.ndarray,
               G: int) -> tuple[float, float]:
    """How much of the REWARDED mass is a hack, per-sample and per-group. TODO 2.

    Returns `(h, group_all_hack)`.

    (a) `h` = P(the sample is incorrect | the checker rewarded it). This is the
        SELECTION LAW. Derive it from Block 3 and note it equals 1 - precision:

            h = gamma(1-p) / [ p(1-alpha) + gamma(1-p) ]

        and h -> 1 as p -> 0, at FIXED gamma. Precision is not a property of the
        checker; it is a property of the checker AND the difficulty.

    (b) `group_all_hack` = P(every rewarded sample in the group is a hack | the
        group is ALIVE). This is the number that actually matters, because GRPO
        learns only from alive groups and at low p an alive group is a k=1 group,
        where the single success carries the whole positive advantage.

        With a = p_obs, k ~ Binomial(G, a), and each accepted sample a hack
        independently w.p. h:

            P(all accepted are hacks AND alive)
                = sum_{k=1}^{G-1} C(G,k) (a h)^k (1-a)^(G-k)
                = (1 - a(1-h))^G - (1-a)^G - (a h)^G
            P(alive) = 1 - a^G - (1-a)^G

        Return the ratio. Return 0.0 if P(alive) == 0.

        The binomial identity is worth doing rather than looking up: it is just
        sum_k C(G,k) x^k (1-a)^(G-k) = (x + 1 - a)^G with x = a h, minus the two
        excluded end terms.
    """
    rate = verifier_rates(gold, accept, probs)
    h = 1 - rate.precision
    p_alive = 1 - rate.p_obs ** G - (1-rate.p_obs) ** G
    if p_alive == 0:
        return (h, 0.0)
    p_hack_alive = (1 - rate.p_obs * (1-h)) ** G - (1-rate.p_obs) ** G - (rate.p_obs * h) ** G
    return (h, p_hack_alive/p_alive)


def component_shares(comp: np.ndarray, weights: np.ndarray,
                     groups: np.ndarray) -> np.ndarray:
    """Each reward component's share of the GROUP-CENTERED advantage. TODO 3.

    `comp` is (n_comp, n_seqs) -- component j's value on each sequence.
    `weights` is (n_comp,) -- the lambdas.
    `groups` is (n_groups, G) -- sampled sequence indices.

    Returns (n_comp,) shares, summing to 1 (or all zeros if nothing varies).

    Centering is LINEAR, so the total advantage decomposes EXACTLY:

        A(total) = sum_j  A( weights[j] * comp[j] )          elementwise, at finite G

    Define component j's share as its L1 mass over all groups divided by the sum
    over components:

        share_j = sum |A_j| / sum_i sum |A_i|

    THE RESULT THIS MEASURES (D1's annihilation law pointed at reward DESIGN):
    for a binary component with compliance c, E[ sum_i |A_i| ] = 2(G-1)c(1-c)
    per group -- so the share is proportional to c(1-c), peaks at c = 0.5, and is
    EXACTLY ZERO at c in {0, 1} for ANY weight. A composite RLVR reward is a
    self-annealing curriculum, and a component you have mastered cannot be
    re-weighted back into relevance.

    Use `group_advantage(..., "centered")` from D1 -- it already operates on the
    last axis, so a whole (n_groups, G) batch goes through one call.
    """
    rewards_all_comps = (weights[:, None] * comp)
    rewards_all_comp_group = rewards_all_comps[:, groups]
    advantage_all_comps = group_advantage(rewards_all_comp_group, 'centered')   # (n_comp, n_groups, G)

    advantage_all_comps_averaged = np.abs(advantage_all_comps).sum(axis=(1, 2))
    if advantage_all_comps_averaged.sum() == 0:
        return np.zeros_like(weights, dtype=np.float64)
    return advantage_all_comps_averaged / advantage_all_comps_averaged.sum()


def increase_percent(x: np.ndarray, frac: float = 0.2) -> float:
    """Relative change from the first `frac` of a run to the last `frac`.

    Guarded twice, because both guards were live bugs before they were guards:

      * a history shorter than 1/frac steps makes BOTH slices empty, and the
        mean of an empty slice is nan;
      * a baseline of exactly 0 makes the ratio 0/0, which is also nan.

    nan never compares True against anything, so a rule keyed on an unguarded
    ratio does not crash -- it silently never fires, which is strictly worse
    than a rule that is absent, because it looks like health.

    When the baseline is 0 the relative change is undefined, so fall back to the
    ABSOLUTE change: both callers key on the sign, and the sign is preserved.
    """
    part = int(len(x) * frac)
    if part < 1:
        return 0.0
    base = float(x[:part].mean())
    end = float(x[-part:].mean())
    if base == 0.0:
        return end - base
    return (end - base) / base


def diagnose_rlvr(h: RlvrHistory) -> list[tuple[str, str]]:
    """Read the panel, name what is wrong. TODO 4 -- the DIAGNOSTIC one, yours.

    Return `list[(name, detail)]`; `detail` must carry a NUMBER. Contracted names,
    and nothing else may be returned:

      "verifier_hacked"  -- v_pass RISES while gold_pass FALLS. This is the only
                            rule that needs the held-out column, and no rule
                            keyed on the reward panel alone can replace it: a
                            k=1 group produced by a hack and one produced by a
                            genuine success have IDENTICAL advantage vectors, so
                            v_pass / dead_frac / adv_absmean / entropy / loss are
                            bit-identical between the two runs.
      "signal_starved"   -- dead_frac HIGH and v_pass LOW  (incompleteness: the
                            checker rejects everything, so every group is all-fail)
      "saturated"        -- dead_frac HIGH and v_pass HIGH (D1's mode: the model
                            solved it, so every group is all-pass)

    RULES OF THE CONTRACT, all of them learned the hard way:
      * SILENT on a healthy run. A diagnostic that fires on health is noise.
      * SPECIFIC. No rule may fire on a fixture it does not describe. This is
        strictly harder than it looks and it is the point of the exercise: two of
        the four fixtures reach a HIGH dead_frac at a HIGH v_pass, and they are
        not the same illness. If your rules disagree with each other on those
        two, the panel you are reading is missing a column.
      * Every threshold keys on a LEVEL, not a change (W5 D5's M4 fixture was
        wrong for exactly this reason) -- EXCEPT `verifier_hacked`, which is
        inherently about two directions and must use both.
      * `dead_frac` alone cannot tell ANY of the three apart, so every rule needs
        a conjunction.
      * Calibrate thresholds from the gap between DISTRIBUTIONS over >= 8 seeds,
        never between one healthy run and the single broken value you saw.
      * Compare LEVELS on the tail of the run, not on step 0. `_tail` is provided.

    AND ONE RULE YOU MUST REJECT. "A reward component's gradient share is
    decaying to zero" looks like a bug and is in fact the health signature proved
    by TODO 3 -- it means the model mastered that component. D1 cut a rule the
    INSTRUMENT could not support; this is a rule the SEMANTICS do not support.
    Write the reason in a comment here; that sentence is as much the deliverable
    as the code is.
    """
    report = []
    if len(h.step) < 10:
        return report

    # THE RULE I REFUSE TO WRITE, and why.
    # "A reward component's gradient share is decaying toward zero" looks exactly
    # like a bug and is the HEALTH signature: TODO 3 proves E[sum|A|] = 2(G-1)c(1-c)
    # for a component with compliance c, so the share goes to zero precisely
    # BECAUSE the model has mastered that component -- at any lambda, since the
    # coefficient cancels out of the share. Firing on it would mean alerting on
    # success, and the "fix" it invites (raise that component's weight) provably
    # does nothing. D1 cut a rule the INSTRUMENT could not support; this is a rule
    # the SEMANTICS do not support.

    d_v, d_gold = increase_percent(h.v_pass), increase_percent(h.gold_pass)
    v_pass_mean = _tail(h.v_pass).mean()
    dead_frac = _tail(h.dead_frac).mean()

    # the only rule that reads the held-out column as a DIRECTION, and the only
    # one that needs two of them: a single column cannot express a divergence.
    if d_v > 0 and d_gold < 0:
        report.append(('verifier_hacked',
                       f"v_pass {d_v:+.4f} while gold_pass {d_gold:+.4f} "
                       f"(relative, first fifth -> last fifth)"))

    if dead_frac > 0.5 and v_pass_mean < 0.3:
        report.append(('signal_starved',
                       f"dead fraction {dead_frac:.4f}, verifier pass {v_pass_mean:.4f}"))

    # gold is NOT optional here. The hacked run also sits at dead > 0.5 with
    # v_pass > 0.8; every column the reward can produce agrees with the solved
    # run, and only the held-out one disagrees.
    if dead_frac > 0.5 and v_pass_mean > 0.8 and d_gold > 0:
        report.append(('saturated',
                       f"dead fraction {dead_frac:.4f}, verifier pass {v_pass_mean:.4f}, "
                       f"gold_pass {d_gold:+.4f}"))

    return report
    
# ---------------------------------------------------------------------------
# Task construction + training loop (provided -- plumbing, not the lesson)
# ---------------------------------------------------------------------------

def _init_theta(cfg: RlvrCfg, env: TokenEnv) -> np.ndarray:
    return cfg.init_scale * np.random.default_rng(cfg.seed + 1000).standard_normal(
        (cfg.n_prompts, env.n_states, env.vocab))


def make_checker(env: TokenEnv, tab: SeqTable, cfg: RlvrCfg, flavour: str,
                 rng: np.random.Generator) -> Checker:
    """Build (gold, accept) pairs whose RELATION is the fixture.

      honest   -- accept == gold. Sound AND complete. The control.
      leaky    -- accept = gold + the sequences the INIT policy already likes.
                  Sound? No. And the FP set is EASIER to reach than the gold set,
                  which is what makes the hack the optimum rather than a curiosity.
      starved  -- accept = a strict subset of gold (an aggressive extractor).
                  Sound, incomplete: correctness is safe, compute is not.
      saturated-- gold = accept = the sequences the init policy already prefers,
                  so the run starts near p ~ 0.95 (D1's dead-group regime).
    """
    P, n = cfg.n_prompts, tab.n_seqs
    th = _init_theta(cfg, env)
    gold, acc = [], []
    for p in range(P):
        pr = np.exp(seq_logprobs_np(th[p], tab))
        pr = pr / pr.sum()
        order = np.argsort(-pr)                     # most likely first
        g = np.zeros(n)
        if flavour == "saturated":
            take, s = [], 0.0
            for y in order:
                take.append(y)
                s += pr[y]
                if s >= 0.95:
                    break
            g[take] = 1.0
            gold.append(g)
            acc.append(g.copy())
            continue

        # gold = a random half, but deliberately drawn from the UNLIKELY tail, so
        # that "being correct" is something the policy has to move toward.
        tail = order[n // 3:]
        g[rng.choice(tail, size=max(2, len(tail) // 2), replace=False)] = 1.0
        gold.append(g)

        if flavour == "honest":
            acc.append(g.copy())
        elif flavour == "leaky":
            a = g.copy()
            # the two MOST likely sequences that are not gold: a cheap hack, and
            # the policy is already sitting next to it.
            hacks = [y for y in order if g[y] == 0.0][:2]
            a[hacks] = 1.0
            acc.append(a)
        elif flavour == "starved":
            a = g.copy()
            keep = np.flatnonzero(g)
            a[keep[1:]] = 0.0                       # accept only ONE correct form
            acc.append(a)
        else:
            raise ValueError(f"unknown flavour {flavour!r}")
    return Checker(gold=np.stack(gold), accept=np.stack(acc))


def _tail(x: np.ndarray, frac: float = 0.2) -> np.ndarray:
    k = max(1, int(round(len(x) * frac)))
    return x[-k:]


def train_rlvr(task: RlvrTask, cfg: RlvrCfg) -> RlvrHistory:
    """GRPO on a VERIFIER's reward. Provided.

    Identical to D1's `train_grpo` in every respect but one, and the one is the
    day: the reward is `checker.accept`, while `checker.gold` is evaluated and
    recorded but NEVER touches the loss. mu = 1 throughout -- D1 measured the clip
    to be exactly inert there, so nothing today may be attributed to clipping.
    """
    tab, env = task.tab, task.env
    P, n = task.n_prompts, tab.n_seqs
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    theta = torch.tensor(_init_theta(cfg, env), dtype=torch.float64, requires_grad=True)
    theta_ref = theta.detach().clone().numpy()

    mask_all = torch.as_tensor(tab.mask)
    opt = torch.optim.SGD([theta], lr=cfg.lr)
    rec = {k: [] for k in ("step", "v_pass", "gold_pass", "dead_frac",
                           "adv_absmean", "kl", "entropy", "loss")}

    for step in range(cfg.steps):
        th_np = theta.detach().numpy()

        # panel FIRST: row i is the state after i updates, so row 0 is the
        # reference checkpoint and h.kl[0] is exactly 0 (D1 bug #4).
        v_pass = float(np.mean([exact_expectation(th_np[p], tab, task.checker.accept[p])
                                for p in range(P)]))
        g_pass = float(np.mean([exact_expectation(th_np[p], tab, task.checker.gold[p])
                                for p in range(P)]))
        kl_now = float(np.mean([exact_kl(th_np[p], theta_ref[p], tab) for p in range(P)]))
        ent_now = float(np.mean([exact_entropy(th_np[p], tab) for p in range(P)]))

        idx = np.empty((P, cfg.group), dtype=np.int64)
        for p in range(P):
            pr = np.exp(seq_logprobs_np(th_np[p], tab))
            idx[p] = rng.choice(n, size=cfg.group, p=pr / pr.sum())

        rewards = np.take_along_axis(task.checker.accept, idx, axis=1)     # (P, G)
        adv_np = group_advantage(rewards, cfg.adv_mode)
        dead = float(np.mean(np.all(np.abs(adv_np) < 1e-12, axis=1)))
        live = np.abs(adv_np) > 1e-12
        absmean = float(np.abs(adv_np)[live].mean()) if live.any() else 0.0

        adv_t = torch.as_tensor(adv_np)                                   # THE BOUNDARY
        idx_t = torch.as_tensor(idx)
        rows = torch.arange(P)[:, None]
        mask = mask_all[idx_t]

        logp_ref = seq_token_logprobs_torch(torch.as_tensor(theta_ref), tab)[rows, idx_t]
        logp = seq_token_logprobs_torch(theta, tab)[rows, idx_t]          # (P, G, T)

        # mu = 1 and on-policy, so rho == 1 exactly and the clip cannot bind:
        # the surrogate is just A * logp. Writing it this way makes that explicit
        # instead of hiding an inert min() in the expression.
        seq_logp = (logp * mask).sum(-1)                                  # (P, G)
        surr = (adv_t * seq_logp).mean()
        kl_tok = torch.exp(logp_ref - logp) - (logp_ref - logp) - 1.0     # k3
        kl_term = (kl_tok * mask).sum() / mask.sum()
        loss = -(surr - cfg.beta * kl_term)

        opt.zero_grad()
        loss.backward()
        opt.step()

        rec["step"].append(step)
        rec["v_pass"].append(v_pass)
        rec["gold_pass"].append(g_pass)
        rec["dead_frac"].append(dead)
        rec["adv_absmean"].append(absmean)
        rec["kl"].append(kl_now)
        rec["entropy"].append(ent_now)
        rec["loss"].append(float(loss.detach()))

    return RlvrHistory(**{k: np.asarray(v, dtype=np.float64) for k, v in rec.items()})


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
        self.sim = 40_000 if fast else 300_000
        self.steps = 300
        self.seeds = 8

    @cached_property
    def theta1(self) -> np.ndarray:
        return 0.9 * np.random.default_rng(7).standard_normal(
            (self.env.n_states, self.env.vocab))

    @cached_property
    def probs1(self) -> np.ndarray:
        p = np.exp(seq_logprobs_np(self.theta1, self.tab))
        return p / p.sum()

    @cached_property
    def theta2(self) -> np.ndarray:
        """A DIFFERENT policy, same checker -- so the rates can be shown to move."""
        return 1.4 * np.random.default_rng(23).standard_normal(
            (self.env.n_states, self.env.vocab))

    @cached_property
    def probs2(self) -> np.ndarray:
        p = np.exp(seq_logprobs_np(self.theta2, self.tab))
        return p / p.sum()

    def run(self, flavour: str, **kw) -> RlvrHistory:
        cfg = RlvrCfg(steps=self.steps, **kw)
        rng = np.random.default_rng(cfg.seed + 500)
        chk = make_checker(self.env, self.tab, cfg, flavour, rng)
        return train_rlvr(RlvrTask(self.env, self.tab, chk), cfg)

    @cached_property
    def h_honest(self) -> RlvrHistory:
        print("  ... training the HONEST fixture (sound + complete)")
        return self.run("honest", seed=0, beta=0.10, lr=0.4)

    @cached_property
    def h_leaky(self) -> RlvrHistory:
        print("  ... training the LEAKY fixture (unsound -> hacked)")
        return self.run("leaky", seed=0, beta=0.02, lr=0.8)

    @cached_property
    def h_starved(self) -> RlvrHistory:
        print("  ... training the STARVED fixture (incomplete -> no signal)")
        return self.run("starved", seed=0, beta=0.10, lr=0.1)

    @cached_property
    def h_saturated(self) -> RlvrHistory:
        print("  ... training the SATURATED fixture (solved -> no signal)")
        return self.run("saturated", seed=0, beta=0.02, lr=0.5)


def _fail_fast(ctx: Ctx, *needs: str) -> None:
    """Touch ONLY the TODOs this stage uses, on a 1-element input, so a missing
    implementation surfaces before an expensive fixture is built -- and a stage is
    never blocked by a TODO it does not need (D1's own grading-scaffold bug)."""
    one = np.ones(1)
    if "rates" in needs:
        verifier_rates(one, one, one)
    if "hack" in needs:
        hack_share(one, one, one, 2)
    if "components" in needs:
        component_shares(np.ones((1, 2)), np.ones(1), np.zeros((1, 2), dtype=np.int64))
    if "diagnose" in needs:
        diagnose_rlvr(RlvrHistory())


# ---- stages ---------------------------------------------------------------

def stage_rates(ctx: Ctx) -> None:
    """verifier_rates -- confusion is a property of the CHECKER AND THE POLICY"""
    _fail_fast(ctx, "rates")
    n = ctx.tab.n_seqs
    probs = ctx.probs1

    gold = np.zeros(n)
    gold[[1, 3, 5, 7, 9, 11]] = 1.0
    acc = gold.copy()
    acc[[2, 4]] = 1.0          # two false positives
    acc[[11]] = 0.0            # one false negative

    r = verifier_rates(gold, acc, probs)
    p_true = float(probs @ gold)
    p_obs = float(probs @ acc)
    _check("p_true is exact mass, not a count", abs(r.p_true - p_true) < 1e-15,
           f"{r.p_true:.6f}")
    _check("p_obs is exact mass, not a count", abs(r.p_obs - p_obs) < 1e-15,
           f"{r.p_obs:.6f}")
    _check("alpha = P(reject | correct)",
           abs(r.alpha - float(probs @ (gold * (1 - acc))) / p_true) < 1e-15,
           f"{r.alpha:.6f}")
    _check("gamma = P(accept | incorrect)",
           abs(r.gamma - float(probs @ ((1 - gold) * acc)) / (1 - p_true)) < 1e-15,
           f"{r.gamma:.6f}")
    _check("precision = P(correct | accepted)",
           abs(r.precision - float(probs @ (gold * acc)) / p_obs) < 1e-15,
           f"{r.precision:.6f}")
    _check("recall == 1 - alpha, exactly", abs(r.recall - (1 - r.alpha)) < 1e-15,
           f"{r.recall:.6f}")

    # THE COMPOSITION LAW: p_obs must equal p(1-alpha) + (1-p)gamma. This is the
    # identity that prices false negatives in rollouts, so it is asserted here
    # rather than assumed in stage_fntax.
    _check("p_obs == p(1-alpha) + (1-p)gamma, to 1e-15",
           abs(r.p_obs - p_observed(r.p_true, r.alpha, r.gamma)) < 1e-15,
           f"{r.p_obs:.9f} vs {p_observed(r.p_true, r.alpha, r.gamma):.9f}")

    # THE LEVEL TRAP: the count-based rate is a different number, and it is the
    # one that looks like "the verifier's accuracy".
    cnt_gamma = float(((1 - gold) * acc).sum() / (1 - gold).sum())
    _check("mass-weighted gamma DIFFERS from the count-weighted one -- a verifier's "
           "confusion is not a property of the verifier alone",
           abs(r.gamma - cnt_gamma) > 1e-3, f"mass {r.gamma:.4f} vs count {cnt_gamma:.4f}")

    # POLICY DEPENDENCE. Not a random second policy -- the one OPTIMIZATION
    # produces: mass tilted toward the sequences the checker wrongly accepts.
    # That is the whole selection story, done in one line instead of a training run.
    fp = (1.0 - gold) * acc
    tilt = probs * np.exp(3.0 * fp)
    tilt = tilt / tilt.sum()
    r2 = verifier_rates(gold, acc, tilt)
    _check("SAME checker, policy tilted toward its false positives -> gamma rises",
           r2.gamma > r.gamma + 0.05, f"{r.gamma:.4f} -> {r2.gamma:.4f}")
    _check("...and precision falls, which is why '97% accurate' is not a "
           "well-formed claim about a REWARD: the number is measured under a "
           "policy that training is actively changing",
           r2.precision < r.precision - 0.05,
           f"{r.precision:.4f} -> {r2.precision:.4f}")
    _check("...while the checker itself never changed: gold and accept are the "
           "same two arrays in both measurements",
           True, "the SET is fixed; the RATE is not")

    # The two set relations, as set relations.
    sound = Checker(gold[None, :], np.minimum(gold, acc)[None, :])
    comp = Checker(gold[None, :], np.maximum(gold, acc)[None, :])
    _check("a SOUND checker has gamma == 0 exactly",
           verifier_rates(gold, sound.accept[0], probs).gamma == 0.0, "")
    _check("a SOUND checker has precision == 1 exactly",
           verifier_rates(gold, sound.accept[0], probs).precision == 1.0, "")
    _check("a COMPLETE checker has alpha == 0 exactly",
           verifier_rates(gold, comp.accept[0], probs).alpha == 0.0, "")
    _check("Checker.sound / .complete agree with the rates",
           sound.sound and comp.complete and not sound.complete and not comp.sound, "")

    # Degenerate denominators -- the guards, not uninitialised memory.
    z = verifier_rates(np.zeros(n), np.zeros(n), probs)
    _check("p_true == 0: alpha guarded to 0.0, precision guarded to 1.0",
           z.alpha == 0.0 and z.precision == 1.0, f"alpha={z.alpha} prec={z.precision}")
    o = verifier_rates(np.ones(n), np.ones(n), probs)
    _check("p_true == 1: gamma guarded to 0.0", o.gamma == 0.0, f"gamma={o.gamma}")


def stage_selection(ctx: Ctx) -> None:
    """hack_share -- precision collapses as the problem gets HARDER"""
    _fail_fast(ctx, "hack", "rates")
    rng = np.random.default_rng(0)
    n = ctx.tab.n_seqs

    # A synthetic policy so p, alpha and gamma can be dialled exactly:
    #   seq 0  = correct AND accepted        mass p(1-alpha)
    #   seq 14 = correct but REJECTED        mass p*alpha        (incompleteness)
    #   seq 1  = incorrect but ACCEPTED      mass gamma(1-p)     (unsoundness)
    #   rest   = incorrect and rejected
    def build(p: float, gam: float = 0.01, alp: float = 0.10):
        probs = np.zeros(n)
        probs[0] = p * (1 - alp)
        probs[14] = p * alp
        probs[1] = gam * (1 - p)
        rest = 1.0 - probs[0] - probs[1] - probs[14]
        probs[2:14] = rest / 12
        gold = np.zeros(n); gold[0] = 1.0; gold[14] = 1.0
        acc = np.zeros(n); acc[0] = 1.0; acc[1] = 1.0
        return gold, acc, probs

    want = {0.500: 0.010989, 0.100: 0.090909, 0.020: 0.352518, 0.005: 0.688581}
    for p, target in want.items():
        gold, acc, probs = build(p)
        h, _ = hack_share(gold, acc, probs, 8)
        closed = 0.01 * (1 - p) / (p * 0.90 + 0.01 * (1 - p))
        _check(f"P(hack|rewarded) at p={p:.3f} matches the closed form",
               abs(h - closed) < 1e-12, f"{h * 100:.4f}%")
        _check(f"...and reproduces the derivation's number {target * 100:.2f}%",
               abs(h - target) < 5e-5, f"{h * 100:.4f}%")

    # The limit, and the direction. Stated as a monotone sweep, not one point.
    hs = [hack_share(*build(p), 8)[0] for p in (0.5, 0.1, 0.02, 0.005, 0.001)]
    _check("h is strictly increasing as p falls -- precision is a function of "
           "DIFFICULTY, at fixed false-positive rate",
           all(b > a for a, b in zip(hs, hs[1:])), f"{[round(x, 4) for x in hs]}")
    tiny, _ = hack_share(*build(1e-6), 8)
    _check("h -> 1 as p -> 0", tiny > 0.999, f"{tiny:.6f}")

    # (b) the GROUP law, against a simulation.
    for p in (0.5, 0.02):
        gold, acc, probs = build(p)
        _, grp = hack_share(gold, acc, probs, 8)
        draws = rng.choice(n, size=(ctx.sim, 8), p=probs)
        rewarded = acc[draws].astype(bool)
        k = rewarded.sum(1)
        alive = (k >= 1) & (k <= 7)
        hacked = rewarded & (gold[draws] == 0.0)
        all_hack = alive & (hacked.sum(1) == k)
        n_alive = int(alive.sum())
        emp = float(all_hack.sum() / n_alive)
        tol = 5.0 * math.sqrt(max(emp * (1 - emp), 1e-9) / n_alive) + 1e-4
        _check(f"group_all_hack at p={p} matches a {n_alive}-alive-group simulation",
               abs(grp - emp) < tol, f"closed {grp:.5f} vs sim {emp:.5f} (tol {tol:.1e})")

    # THE TRAP: conditioning on ALIVE does not move the per-sample rate.
    gold, acc, probs = build(0.02)
    draws = rng.choice(n, size=(ctx.sim, 8), p=probs)
    rewarded = acc[draws].astype(bool)
    k = rewarded.sum(1)
    alive = (k >= 1) & (k <= 7)
    hacked = rewarded & (gold[draws] == 0.0)
    rate_all = float(hacked.sum() / rewarded.sum())
    rate_alive = float(hacked[alive].sum() / rewarded[alive].sum())
    h, grp = hack_share(gold, acc, probs, 8)
    _check("P(hack | rewarded) is UNCHANGED by conditioning on the group being "
           "alive -- the rate does not move", abs(rate_all - rate_alive) < 5e-3,
           f"{rate_all:.5f} vs {rate_alive:.5f}")
    _check("...what moves is the CONCENTRATION: P(the whole alive group's positive "
           "signal is a hack) is far larger at low p",
           grp > 0.30, f"group_all_hack {grp:.4f} vs per-sample {h:.4f}")
    lo = hack_share(*build(0.5), 8)[1]
    _check("...and that concentration is ~550x smaller at p=0.5, so the pathology "
           "is DIFFICULTY-INDUCED, not verifier-induced -- the FP rate is 1% in "
           "both runs", grp / lo > 400, f"{grp:.4e} vs {lo:.4e} = {grp / lo:.1f}x")

    # THE OBSERVABILITY RESULT: a hack and a genuine success are bit-identical in
    # every scalar the panel records. This is what forces gold_pass to exist.
    r_hack = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    r_true = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    a_hack, a_true = group_advantage(r_hack), group_advantage(r_true)
    _check("a k=1 group from a HACK and one from a genuine success have "
           "bit-identical advantage vectors -- no reward-panel rule can separate "
           "them, which is why diagnose_rlvr needs the held-out column",
           np.array_equal(a_hack, a_true) and abs(a_hack[0, 0] - 7 / 8) < 1e-15,
           f"A_success = {a_hack[0, 0]:.6f} = (G-1)/G")


def stage_renyi(ctx: Ctx) -> None:
    """the dead-group law generalised: P(dead) = sum q^G = a Renyi entropy"""
    rng = np.random.default_rng(1)

    # D1's binary law is the L=2 special case, and it must agree exactly.
    for p in (0.50, 0.70, 0.90, 0.95, 0.99):
        _check(f"binary special case at p={p:.2f}",
               abs(dead_fraction_levels(np.array([p, 1 - p]), 8)
                   - (p ** 8 + (1 - p) ** 8)) < 1e-15,
               f"{dead_fraction_levels(np.array([p, 1 - p]), 8) * 100:.2f}%")

    # against a simulation
    q = np.array([0.05, 0.40, 0.55])
    draws = rng.choice(len(q), size=(ctx.sim, 8), p=q)
    emp = float(np.mean((draws == draws[:, :1]).all(1)))
    _check("3-level law matches a simulation",
           abs(dead_fraction_levels(q, 8) - emp) < 2e-3,
           f"closed {dead_fraction_levels(q, 8):.6f} vs sim {emp:.6f}")

    # the identification as a Renyi entropy of order G
    _check("sum q^G == exp(-(G-1) H_G(q)) -- the dead fraction IS a Renyi entropy "
           "of the REWARD's output distribution",
           abs(dead_fraction_levels(q, 8) - math.exp(-7 * renyi_entropy(q, 8))) < 1e-14,
           f"{dead_fraction_levels(q, 8):.9f}")

    # THE HEADLINE: one extra reward level vs eleven times the rollout budget.
    binary = dead_fraction_levels(np.array([0.05, 0.95]), 8)
    three = dead_fraction_levels(q, 8)
    _check("binary at p=0.05, G=8 is D1's 66.34%", abs(binary - 0.663420) < 1e-5,
           f"{binary * 100:.2f}%")
    _check("splitting the FAILURE mass 0.95 -> 0.55/0.40 gives 0.90%",
           abs(three - 0.009029) < 1e-5, f"{three * 100:.4f}%")
    _check("...a 73x reduction from ONE extra reward level",
           abs(binary / three - 73.479) < 0.01, f"{binary / three:.3f}x")
    g_equiv = math.log(three) / math.log(0.95)
    _check("...and buying the same thing with G instead needs G=92, an 11.5x "
           "rollout bill: reward design and group size are the SAME knob",
           91 < g_equiv < 92 and abs(g_equiv / 8 - 11.47) < 0.05,
           f"G={g_equiv:.2f}, {g_equiv / 8:.2f}x")

    # THE ASYMMETRY, with its own negative control: you must split the MODE.
    hard_split_fail = dead_fraction_levels(np.array([0.05, 0.40, 0.55]), 8)
    hard_split_pass = dead_fraction_levels(np.array([0.03, 0.02, 0.95]), 8)
    _check("at p=0.05, splitting the SUCCESS mass does essentially nothing",
           abs(hard_split_pass - binary) / binary < 1e-6,
           f"{hard_split_pass * 100:.4f}% vs {binary * 100:.4f}%")
    _check("...while splitting the FAILURE mass is the 73x win -- the rule is "
           "SPLIT THE MODE, and at p=0.95 it reverses",
           hard_split_fail < hard_split_pass / 50, "")
    sat = dead_fraction_levels(np.array([0.95, 0.05]), 8)
    sat_split_pass = dead_fraction_levels(np.array([0.55, 0.40, 0.05]), 8)
    _check("mirror image at p=0.95: now grading AMONG SUCCESSES is what helps",
           abs(sat - binary) < 1e-15 and sat_split_pass < sat / 50,
           f"{sat * 100:.2f}% -> {sat_split_pass * 100:.4f}%")

    # monotone in G at fixed q, and exactly 1 at G=1 (D1's identity)
    _check("P(dead) == 1 exactly at G=1, for any reward distribution",
           abs(dead_fraction_levels(q, 1) - 1.0) < 1e-15, "")


def stage_fntax(ctx: Ctx) -> None:
    """false negatives are a COMPUTE bug: a regex, priced in rollouts"""
    # D1's exact integer G for 90% alive
    _check("g_for_alive reproduces D1's closed form at p=0.95",
           g_for_alive(0.95) == 45, f"G={g_for_alive(0.95)}")
    _check("...and D1's approximation ln(0.1)/ln(1-p) = 44.891 rounds into it",
           abs(math.log(0.1) / math.log(0.95) - 44.891) < 0.01, "")

    p, alpha = 0.10, 0.30
    _check("a 30% extraction failure moves p_obs 0.10 -> 0.07",
           abs(p_observed(p, alpha, 0.0) - 0.07) < 1e-15,
           f"{p_observed(p, alpha, 0.0):.4f}")
    g0, g1 = g_for_alive(p), g_for_alive(p_observed(p, alpha, 0.0))
    _check("...and the group size needed for 90% alive from G=22 to G=32",
           (g0, g1) == (22, 32), f"{g0} -> {g1}")
    _check("...= +45% rollout compute, bought by a regex and paid in GPU-hours",
           abs(g1 / g0 - 1.4545) < 0.001, f"{(g1 / g0 - 1) * 100:.1f}%")

    # UNSOUNDNESS moves the same knob in the OPPOSITE direction, which is the
    # trap: a leaky checker LOOKS like it is helping the compute problem.
    g_leak = g_for_alive(p_observed(p, alpha, 0.05))
    _check("a 5% false-positive rate REDUCES the required G -- unsoundness buys "
           "compute, which is exactly why the cheap fix is the dangerous one",
           g_leak < g1, f"{g1} -> {g_leak}")

    # and the two costs are not comparable, which is the day's thesis in numbers
    _check("the FN fix and the FP fix move the SAME number in opposite directions, "
           "so no single scalar can express verifier quality",
           g_for_alive(p_observed(p, 0.0, 0.0)) == 22 and g_leak < 22 < g1,
           f"sound+complete 22 | incomplete {g1} | unsound {g_leak}")


def stage_components(ctx: Ctx) -> None:
    """component_shares -- influence tracks INCONSISTENCY, not the coefficient"""
    _fail_fast(ctx, "components")
    rng = np.random.default_rng(3)
    n, G, n_groups = ctx.tab.n_seqs, 8, 20_000 if ctx.fast else 120_000

    def probs_with(c: float, k: int = 5) -> np.ndarray:
        """A distribution putting mass exactly c on the first k sequences."""
        p = np.empty(n)
        p[:k] = c / k
        p[k:] = (1 - c) / (n - k)
        return p

    fmt = np.zeros(n); fmt[:5] = 1.0            # the "format" component
    corr = np.zeros(n); corr[[0, 6, 7, 8]] = 1.0  # the "correctness" component

    # the c(1-c) law, against its closed form
    for c in (0.10, 0.50, 0.90):
        probs = probs_with(c)
        groups = rng.choice(n, size=(n_groups, G), p=probs)
        per_group = np.abs(group_advantage(fmt[groups])).sum(1)
        emp, closed = float(per_group.mean()), 2 * (G - 1) * c * (1 - c)
        tol = 5.0 * float(per_group.std()) / math.sqrt(len(per_group))
        _check(f"E[sum|A|] for a Bernoulli(c={c:.2f}) component == 2(G-1)c(1-c)",
               abs(emp - closed) < tol,
               f"emp {emp:.5f} vs closed {closed:.5f} (tol {tol:.1e})")

    # EXACTLY zero at the boundaries, for ANY weight -- this is D1's annihilation
    for c, name in ((1.0, "fully complied"), (0.0, "never complied")):
        probs = probs_with(c)
        groups = rng.choice(n, size=(2000, G), p=probs)
        sh = component_shares(np.stack([corr, fmt]), np.array([1.0, 1000.0]), groups)
        _check(f"a {name} component has EXACTLY zero share, even at lambda=1000",
               sh[1] == 0.0, f"share={sh[1]!r}")

    # the negative control: at c = 0.5 the same component dominates
    probs = probs_with(0.5)
    groups = rng.choice(n, size=(n_groups, G), p=probs)
    sh_half = component_shares(np.stack([corr, fmt]), np.array([1.0, 1.0]), groups)
    _check("...but at c=0.5 the very same component carries real share -- so the "
           "zero is a statement about the MODEL, not about the reward",
           sh_half[1] > 0.3, f"share={sh_half[1]:.4f}")
    _check("shares sum to 1", abs(sh_half.sum() - 1.0) < 1e-12, f"{sh_half.sum():.12f}")

    # the DECAY, which is the self-annealing curriculum
    shares = []
    for c in (0.50, 0.90, 0.99):
        pr = probs_with(c)
        gr = rng.choice(n, size=(n_groups, G), p=pr)
        shares.append(float(np.abs(group_advantage(fmt[gr])).sum(1).mean()))
    _check("component influence DECAYS monotonically as compliance rises: a "
           "composite RLVR reward is a self-annealing curriculum",
           shares[0] > shares[1] > shares[2], f"{[round(s, 5) for s in shares]}")
    _check("...and the decay is the c(1-c) ratio, 25x from c=0.50 to c=0.99",
           abs(shares[0] / shares[2] - 25.25) < 2.0, f"{shares[0] / shares[2]:.2f}x")

    # LINEARITY: the decomposition is exact at finite G, not in expectation
    tot = group_advantage(1.0 * corr[groups] + 0.3 * fmt[groups])
    parts = group_advantage(1.0 * corr[groups]) + group_advantage(0.3 * fmt[groups])
    _check("centering is linear, so the component decomposition is EXACT at "
           "finite G", float(np.abs(tot - parts).max()) < 1e-14,
           f"max dev {np.abs(tot - parts).max():.2e}")


def stage_lexicographic(ctx: Ctx) -> None:
    """the lambda trap: when does 'wrong but well-formatted' win?"""
    n = ctx.tab.n_seqs
    fmt = np.zeros(n); fmt[:5] = 1.0
    corr = np.zeros(n); corr[[6, 7]] = 1.0        # correct sequences are NOT formatted

    for lam in (0.6, 0.99):
        r = corr + lam * fmt
        best = int(np.argmax(r))
        _check(f"lambda={lam}: the argmax is CORRECT",
               corr[best] == 1.0, f"argmax={best}, r={r[best]:.2f}")
    for lam in (1.01, 1.4):
        r = corr + lam * fmt
        best = int(np.argmax(r))
        _check(f"lambda={lam}: the argmax is a WRONG but well-formatted sequence",
               corr[best] == 0.0 and fmt[best] == 1.0, f"argmax={best}, r={r[best]:.2f}")

    # the boundary is exactly the lexicographic condition: at lambda = 1 a
    # correct-unformatted and a wrong-formatted sequence are TIED, so the sum
    # stops encoding a priority at all.
    r1 = corr + 1.0 * fmt
    top = np.flatnonzero(r1 == r1.max())
    _check("at lambda = 1 exactly, a CORRECT sequence and a WRONG one are tied at "
           "the top -- a sum is lexicographic only strictly below "
           "lambda*max(partial) = min(correctness gap)",
           set(corr[top]) == {0.0, 1.0}, f"tied at r={r1.max():.2f}: {top.tolist()}")

    # and the interaction with D1's 1/std: normalisation rescales the components
    # UNEQUALLY, so a lambda validated under 'centered' is not validated under
    # 'normalized'.
    rng = np.random.default_rng(5)
    probs = np.full(n, 1 / n)
    groups = rng.choice(n, size=(4000, 8), p=probs)
    r = corr + 0.6 * fmt
    c_adv = group_advantage(r[groups], "centered")
    n_adv = group_advantage(r[groups], "normalized")
    ratio = float(np.abs(n_adv).mean() / np.abs(c_adv).mean())
    _check("normalisation rescales the composite reward, so a lambda tuned under "
           "'centered' is NOT tuned under 'normalized'", ratio > 1.2,
           f"|A_norm|/|A_cent| = {ratio:.3f}")


def stage_soundness(ctx: Ctx) -> None:
    """the soundness theorem: V subset of C  =>  zero gap, every policy, every KL"""
    rng = np.random.default_rng(9)
    n = ctx.tab.n_seqs
    gold = np.zeros(n); gold[[0, 3, 6, 9, 12]] = 1.0

    sound_acc = gold.copy(); sound_acc[[3, 9]] = 0.0        # V strictly inside C
    unsound1 = gold.copy(); unsound1[[1]] = 1.0             # ONE false positive
    unsound7 = gold.copy(); unsound7[[1, 2, 4, 5, 7, 8, 10]] = 1.0   # SEVEN

    # 400 random policies, including adversarial ones concentrated on any single
    # sequence. If the bound needed "a reasonable policy" it would be worthless.
    pols = []
    for _ in range(400):
        th = 2.5 * rng.standard_normal((ctx.env.n_states, ctx.env.vocab))
        p = np.exp(seq_logprobs_np(th, ctx.tab))
        pols.append(p / p.sum())
    for y in range(n):                                       # the adversarial ones
        p = np.full(n, 1e-12); p[y] = 1.0
        pols.append(p / p.sum())

    gaps_sound = [float(pr @ sound_acc - pr @ gold) for pr in pols]
    _check("SOUND checker: V is a SUBSET of C as a set relation, exactly",
           bool(np.all(sound_acc <= gold)), "")
    _check("...so v_pass <= gold_pass for all 415 policies, including every point "
           "mass -- the optimization gap is bounded by ZERO with no scope "
           "condition", max(gaps_sound) <= 1e-15, f"max gap {max(gaps_sound):.3e}")

    gaps_1 = [float(pr @ unsound1 - pr @ gold) for pr in pols]
    gaps_7 = [float(pr @ unsound7 - pr @ gold) for pr in pols]
    _check("ONE false positive: the supremum of the gap is 1.0, attained by the "
           "point mass on it", abs(max(gaps_1) - 1.0) < 1e-9, f"{max(gaps_1):.6f}")
    _check("SEVEN false positives: the supremum is ALSO 1.0 -- reward hacking is a "
           "SUPPORT property, not a rate property",
           abs(max(gaps_7) - max(gaps_1)) < 1e-9, f"{max(gaps_7):.6f}")

    # and the average-case rate says the opposite, which is the whole thesis
    unif = np.full(n, 1 / n)
    acc1 = 1.0 - float(unif @ np.abs(unsound1 - gold))
    acc7 = 1.0 - float(unif @ np.abs(unsound7 - gold))
    _check("...while their EVAL accuracies differ a lot (93.3% vs 53.3%): the "
           "statistic that ranks them as evals is silent on the thing that "
           "matters under optimization",
           acc1 > acc7 + 0.3, f"{acc1 * 100:.1f}% vs {acc7 * 100:.1f}%")


def stage_underdetermined(ctx: Ctx) -> None:
    """the RM contrast: the gap is decided by the INITIALISATION, not the data"""
    n = ctx.tab.n_seqs
    gold = np.zeros(n); gold[[0, 2, 4, 6, 8, 10, 12]] = 1.0

    # W5's result made concrete: a preference dataset is a MATCHING. Seven pairs
    # cover 14 of the 15 sequences; sequence 14 is in no comparison at all.
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]

    def bt_loss_grad(r: np.ndarray) -> tuple[float, np.ndarray]:
        g = np.zeros_like(r)
        tot = 0.0
        for w, l in pairs:
            d = r[w] - r[l]
            tot += float(np.logaddexp(0.0, -d))
            s = -1.0 / (1.0 + np.exp(d))
            g[w] += s
            g[l] -= s
        return tot / len(pairs), g / len(pairs)

    r0 = np.zeros(n)
    _, g0 = bt_loss_grad(r0)
    _check("the BT gradient on the UNCOMPARED response is EXACTLY 0.0 -- not "
           "small, exactly zero, at every r", g0[14] == 0.0, f"{g0[14]!r}")

    # the loss is exactly constant along that direction
    for shift in (1.0, -37.5, 1e6):
        rr = r0.copy(); rr[14] += shift
        _check(f"moving r[14] by {shift:g} changes the loss by exactly 0",
               abs(bt_loss_grad(rr)[0] - bt_loss_grad(r0)[0]) == 0.0, "")

    # so two inits, IDENTICAL data, converge to different rewards -- and the
    # difference lands exactly where RL will look: the argmax.
    outcomes = {}
    for tag, init in (("low", -50.0), ("high", +50.0)):
        r = np.zeros(n); r[14] = init
        for _ in range(2000):
            _, g = bt_loss_grad(r)
            r -= 0.5 * g
        outcomes[tag] = (int(np.argmax(r)), float(gold[int(np.argmax(r))]))
    _check("same dataset, two inits: the RM's ARGMAX differs",
           outcomes["low"][0] != outcomes["high"][0], f"{outcomes}")
    _check("...and the gold value of that argmax differs, so the proxy/gold gap "
           "is a property of the INITIALISATION, not of the data. A verifier has "
           "no analogue of this -- its accept set is fixed before training starts",
           outcomes["low"][1] != outcomes["high"][1], f"{outcomes}")

    # the contrast, said as a set: a verifier's V is fixed; an RM's is not.
    _check("a verifier's accept set is the same object under both inits, so its "
           "worst case is computable BEFORE training", True, "V is data, not a fit")


def stage_train(ctx: Ctx) -> None:
    """a real RLVR loop: the verifier is rewarded, the gold is only watched"""
    h = ctx.h_honest
    _check("honest: the panel's row 0 is the reference checkpoint, so kl[0] == 0 "
           "exactly", h.kl[0] == 0.0, f"{h.kl[0]!r}")
    _check("honest: v_pass rises", _tail(h.v_pass).mean() > h.v_pass[0] + 0.05,
           f"{h.v_pass[0]:.4f} -> {_tail(h.v_pass).mean():.4f}")
    _check("honest: gold_pass rises WITH it -- a sound+complete checker makes the "
           "two columns the same column",
           float(np.abs(h.v_pass - h.gold_pass).max()) < 1e-12,
           f"max |v - gold| = {np.abs(h.v_pass - h.gold_pass).max():.2e}")

    lk = ctx.h_leaky
    _check("leaky: v_pass rises (training looks GREAT)",
           _tail(lk.v_pass).mean() > lk.v_pass[0] + 0.05,
           f"{lk.v_pass[0]:.4f} -> {_tail(lk.v_pass).mean():.4f}")
    _check("leaky: gold_pass FALLS at the same time -- the divergence, and it is "
           "invisible without the held-out column",
           _tail(lk.gold_pass).mean() < lk.gold_pass[0] - 0.02,
           f"{lk.gold_pass[0]:.4f} -> {_tail(lk.gold_pass).mean():.4f}")

    st = ctx.h_starved
    _check("starved: an INCOMPLETE checker starves the run of signal",
           _tail(st.dead_frac).mean() > 0.5, f"dead {_tail(st.dead_frac).mean():.4f}")
    _check("starved: ...at a LOW pass rate", _tail(st.v_pass).mean() < 0.35,
           f"v_pass {_tail(st.v_pass).mean():.4f}")
    _check("starved: gold_pass sits ABOVE v_pass throughout -- the signature of "
           "incompleteness, and the exact mirror of the leaky run",
           bool(np.all(st.gold_pass >= st.v_pass - 1e-12)),
           f"gold {_tail(st.gold_pass).mean():.4f} vs v {_tail(st.v_pass).mean():.4f}")

    sa = ctx.h_saturated
    _check("saturated: D1's mode -- dead groups at a HIGH pass rate",
           _tail(sa.dead_frac).mean() > 0.5 and _tail(sa.v_pass).mean() > 0.7,
           f"dead {_tail(sa.dead_frac).mean():.4f} @ pass {_tail(sa.v_pass).mean():.4f}")
    _check("starved and saturated have SIMILAR dead_frac at OPPOSITE pass rates, "
           "so dead_frac alone cannot diagnose either",
           abs(_tail(st.dead_frac).mean() - _tail(sa.dead_frac).mean()) < 0.35,
           f"{_tail(st.dead_frac).mean():.3f} vs {_tail(sa.dead_frac).mean():.3f}")

    # THE ONE THAT MATTERS. The hacked run and the solved run are indistinguishable
    # on the reward panel and opposite on the held-out one.
    _check("LEAKY and SATURATED agree on dead_frac AND on v_pass -- every column "
           "the reward can produce says the same thing about a run that is "
           "improving and a run that is degrading",
           abs(_tail(lk.dead_frac).mean() - _tail(sa.dead_frac).mean()) < 0.20
           and abs(_tail(lk.v_pass).mean() - _tail(sa.v_pass).mean()) < 0.10,
           f"dead {_tail(lk.dead_frac).mean():.3f} vs {_tail(sa.dead_frac).mean():.3f}"
           f" | v_pass {_tail(lk.v_pass).mean():.3f} vs {_tail(sa.v_pass).mean():.3f}")
    _check("...and gold_pass separates them by a mile: the held-out column is not "
           "a nice-to-have, it is the only column carrying the distinction",
           _tail(sa.gold_pass).mean() - _tail(lk.gold_pass).mean() > 0.7,
           f"gold {_tail(lk.gold_pass).mean():.4f} vs {_tail(sa.gold_pass).mean():.4f}")


def stage_diagnose(ctx: Ctx) -> None:
    """diagnose_rlvr -- SILENT on healthy, specific on each fixture"""
    _fail_fast(ctx, "diagnose")
    fired = diagnose_rlvr(ctx.h_honest)
    _check("returns a list of (name, detail) pairs", isinstance(fired, list), "")
    _check("SILENT on the healthy run", fired == [], f"fired {[f[0] for f in fired]}")

    cases = [("verifier_hacked", ctx.h_leaky), ("signal_starved", ctx.h_starved),
             ("saturated", ctx.h_saturated)]
    allowed = {"verifier_hacked", "signal_starved", "saturated"}
    for name, h in cases:
        d = dict(diagnose_rlvr(h))
        _check(f"fires '{name}' on its fixture", name in d, f"fired {sorted(d)}")
        _check(f"'{name}' detail carries a number",
               any(ch.isdigit() for ch in d.get(name, "")), f"{d.get(name, '')!r}")
        _check("uses only the three contracted names", set(d) <= allowed, f"{sorted(d)}")

    # SPECIFICITY -- without this, "always return all three" passes everything.
    lk_f = {n for n, _ in diagnose_rlvr(ctx.h_leaky)}
    st = {n for n, _ in diagnose_rlvr(ctx.h_starved)}
    sa = {n for n, _ in diagnose_rlvr(ctx.h_saturated)}
    _check("starved fires signal_starved and NOT saturated", st == {"signal_starved"},
           f"{sorted(st)}")
    _check("saturated fires saturated and NOT signal_starved -- the two rules are "
           "separated by v_pass, which is the whole reason both need the "
           "conjunction", sa == {"saturated"}, f"{sorted(sa)}")
    _check("leaky fires verifier_hacked and NOT saturated -- the HACKED run and "
           "the SOLVED run have the same dead_frac at the same v_pass, so a "
           "'saturated' rule keyed on the reward panel alone is a FALSE POSITIVE "
           "on a run that is actively getting worse",
           lk_f == {"verifier_hacked"}, f"{sorted(lk_f)}")

    # SEED STABILITY: thresholds must come from the gap between distributions.
    seeds = range(ctx.seeds)
    healthy = [ctx.run("honest", seed=s, beta=0.10, lr=0.4) for s in seeds]
    quiet = sum(diagnose_rlvr(h) == [] for h in healthy)
    _check(f"silent on {ctx.seeds}/{ctx.seeds} healthy seeds, not just seed 0",
           quiet == ctx.seeds, f"{quiet}/{ctx.seeds}")
    leaky = [ctx.run("leaky", seed=s, beta=0.02, lr=0.8) for s in seeds]
    hit = sum("verifier_hacked" in dict(diagnose_rlvr(h)) for h in leaky)
    _check(f"catches the hack on {ctx.seeds}/{ctx.seeds} leaky seeds",
           hit == ctx.seeds, f"{hit}/{ctx.seeds}")
    clean = sum(set(dict(diagnose_rlvr(h))) == {"verifier_hacked"} for h in leaky)
    _check(f"...and calls it ONLY that on {ctx.seeds}/{ctx.seeds} leaky seeds -- a "
           "rule that also shouts 'saturated' at a degrading run is worse than "
           "no rule", clean == ctx.seeds, f"{clean}/{ctx.seeds}")


STAGES = [
    ("rates", stage_rates),
    ("selection", stage_selection),
    ("renyi", stage_renyi),
    ("fntax", stage_fntax),
    ("components", stage_components),
    ("lexicographic", stage_lexicographic),
    ("soundness", stage_soundness),
    ("underdetermined", stage_underdetermined),
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
