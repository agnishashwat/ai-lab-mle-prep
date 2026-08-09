"""W6 D1 -- GRPO vs PPO: the group IS the critic.

THE THESIS
    GRPO does not delete the value function. It replaces a LEARNED estimate of
    E[r|x] with a MONTE-CARLO estimate of the same quantity -- and unlike a
    critic, that estimate VANISHES exactly where the policy succeeds.

Everything here is checked against exact enumeration. The environment is a
tabular `TokenEnv(vocab=3, t_max=3)` -- 7 states, 15 terminating sequences --
ported from `week-05-.../code/dpo.py`, with P independent prompts, each holding
its own parameter block and its own VERIFIABLE checker (a boolean over the 15
sequences). So:

  * every expectation has a closed form (sum over 15 sequences),
  * every gradient has a closed form (sum over 15 score functions),
  * the pass rate `p` is EXACT, not estimated -- which is what makes the
    dead-group law checkable rather than merely plausible.

Sampling appears in exactly two places, and in both the point of the exercise
IS that something is sampled: the `shrinkage` stage (a Monte-Carlo gradient
against an exact one) and the `deadgroup` stage.

THE NUMPY/TORCH BOUNDARY -- one line, and it is not arbitrary:
    rewards and advantages are NUMPY   (they are constants w.r.t. theta)
    log-probabilities are TORCH        (they carry the graph)
The advantage is detached BY CONSTRUCTION, not by remembering to call
`.detach()`. Convert once, at the edge, in `train_grpo`.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-06-grpo-rlvr-reasoning/code/grpo.py
  one stage           ...  grpo.py <stage>
  a few stages        ...  grpo.py advantage shrinkage
  small/fast fixtures ...  grpo.py --fast
  list the stages     ...  grpo.py --list
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Numerics (provided)
# ---------------------------------------------------------------------------

def log_softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax_np(z, axis=axis))


# ---------------------------------------------------------------------------
# Environment (provided -- ported from week-05/code/dpo.py)
# ---------------------------------------------------------------------------

class TokenEnv:
    """Token-level LM environment, small enough to enumerate exhaustively.

    vocab=3 (tokens 0, 1 + eos=2), t_max=3:
      n_states  = 1 + 2 + 4 = 7    (prefixes with no eos and length < t_max)
      sequences = 1 + 2 + 4 + 8 = 15
    """

    def __init__(self, vocab: int = 3, t_max: int = 3):
        assert vocab >= 2 and t_max >= 1
        self.vocab, self.t_max, self.eos = vocab, t_max, vocab - 1
        states, frontier = [()], [()]
        for _ in range(t_max - 1):
            nxt = [p + (a,) for p in frontier for a in range(vocab - 1)]
            states.extend(nxt)
            frontier = nxt
        self.states = states
        self._index = {s: i for i, s in enumerate(states)}
        self.n_states = len(states)

    def state_id(self, prefix: tuple[int, ...]) -> int:
        return self._index[prefix]

    def __repr__(self) -> str:
        return f"TokenEnv(vocab={self.vocab}, t_max={self.t_max}, n_states={self.n_states})"


def enumerate_sequences(env: TokenEnv) -> list[tuple[int, ...]]:
    """Every terminating sequence, in a DETERMINISTIC order (rewards index into it)."""
    out, frontier = [], [()]
    for _ in range(env.t_max):
        nxt = []
        for p in frontier:
            for a in range(env.vocab - 1):
                nxt.append(p + (a,))
            out.append(p + (env.eos,))
        frontier = nxt
    return out + frontier


def grad_sequence_logprob(env: TokenEnv, theta: np.ndarray,
                          seq: tuple[int, ...]) -> np.ndarray:
    """The score function, shape (n_states, vocab). W3 D2 material, provided.

        d/d theta[s, a]  log p_theta(y)  =  sum_t  ( 1[s = s_t, a = y_t] - pi(a|s_t) )

    Every ROW sums to exactly zero: a softmax cannot change a state's total mass.
    """
    pi = softmax_np(theta)
    g = np.zeros_like(theta)
    for t in range(len(seq)):
        s = env.state_id(seq[:t])
        g[s, seq[t]] += 1.0
        g[s, :] -= pi[s, :]
    return g


@dataclass(frozen=True)
class SeqTable:
    """Everything about the 15 sequences, precomputed once, in array form.

    `states`/`actions` are padded to t_max; `mask` says which positions are real.
    This is what turns "walk the tree" into a gather, so both numpy and torch can
    score a whole batch of sequences with one indexing operation.
    """
    seqs: list[tuple[int, ...]]
    states: np.ndarray       # (n_seqs, t_max) int64
    actions: np.ndarray      # (n_seqs, t_max) int64
    mask: np.ndarray         # (n_seqs, t_max) bool
    lengths: np.ndarray      # (n_seqs,) int64

    @property
    def n_seqs(self) -> int:
        return len(self.seqs)


def build_seq_table(env: TokenEnv) -> SeqTable:
    seqs = enumerate_sequences(env)
    n, T = len(seqs), env.t_max
    states = np.zeros((n, T), dtype=np.int64)
    actions = np.zeros((n, T), dtype=np.int64)
    mask = np.zeros((n, T), dtype=bool)
    for i, y in enumerate(seqs):
        for t, a in enumerate(y):
            states[i, t] = env.state_id(y[:t])
            actions[i, t] = a
            mask[i, t] = True
    return SeqTable(seqs=seqs, states=states, actions=actions, mask=mask,
                    lengths=mask.sum(1))


def seq_logprobs_np(theta: np.ndarray, tab: SeqTable) -> np.ndarray:
    """log p_theta(y) for all 15 sequences. theta is (n_states, vocab)."""
    lp = log_softmax_np(theta)
    per_token = lp[tab.states, tab.actions] * tab.mask
    return per_token.sum(1)


def seq_token_logprobs_torch(theta: torch.Tensor, tab: SeqTable) -> torch.Tensor:
    """Per-token log-probs, differentiable. Masked positions are exactly 0.

        theta (n_states, vocab)     -> (n_seqs, t_max)
        theta (P, n_states, vocab)  -> (P, n_seqs, t_max)
    """
    lp = torch.log_softmax(theta, dim=-1)
    st = torch.as_tensor(tab.states)
    ac = torch.as_tensor(tab.actions)
    mk = torch.as_tensor(tab.mask)
    if lp.dim() == 2:
        return lp[st, ac] * mk
    return lp[:, st, ac] * mk


def all_score_functions(env: TokenEnv, theta: np.ndarray, tab: SeqTable) -> np.ndarray:
    """(n_seqs, n_states, vocab) -- grad log p_theta(y) for every y. Provided."""
    return np.stack([grad_sequence_logprob(env, theta, y) for y in tab.seqs])


# ---------------------------------------------------------------------------
# Exact quantities by enumeration (provided -- these are the ground truth)
# ---------------------------------------------------------------------------

def exact_expectation(theta: np.ndarray, tab: SeqTable, values: np.ndarray) -> float:
    """E_{y ~ pi_theta}[ values[y] ], summed over all 15 sequences."""
    return float(np.sum(np.exp(seq_logprobs_np(theta, tab)) * values))


def exact_policy_gradient(env: TokenEnv, theta: np.ndarray, tab: SeqTable,
                          rewards: np.ndarray) -> np.ndarray:
    """grad_theta E_{y ~ pi_theta}[ r(y) ] = sum_y p(y) r(y) grad log p(y). EXACT."""
    p = np.exp(seq_logprobs_np(theta, tab))
    G = all_score_functions(env, theta, tab)
    return np.einsum("y,y,ysv->sv", p, rewards, G)


def exact_entropy(theta: np.ndarray, tab: SeqTable) -> float:
    """H(pi_theta) over whole sequences, in nats."""
    lp = seq_logprobs_np(theta, tab)
    p = np.exp(lp)
    return float(-np.sum(p * lp))


def exact_kl(theta: np.ndarray, theta_ref: np.ndarray, tab: SeqTable) -> float:
    """KL(pi_theta || pi_ref) over whole sequences. No estimator."""
    lp = seq_logprobs_np(theta, tab)
    lpr = seq_logprobs_np(theta_ref, tab)
    return float(np.sum(np.exp(lp) * (lp - lpr)))


# ---------------------------------------------------------------------------
# Task: P prompts, each an independent distribution with a VERIFIABLE checker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """P prompts. `correct[p, y]` is the verifier's answer for sequence y of prompt p.

    Each prompt owns its own parameter block theta[p], exactly as an LLM's
    conditional distribution pi(.|x) differs per prompt. There is deliberately NO
    parameter sharing across prompts: today's subject is the ESTIMATOR, and
    sharing would add representation-learning effects that confound it.
    """
    env: TokenEnv
    tab: SeqTable
    correct: np.ndarray      # (P, n_seqs) float64 in {0.0, 1.0}

    @property
    def n_prompts(self) -> int:
        return self.correct.shape[0]


def exact_pass_rate(theta_p: np.ndarray, tab: SeqTable, correct_p: np.ndarray) -> float:
    """p = P(the verifier accepts) under pi_theta, EXACTLY. This is the knob that
    the dead-group law is a function of, so it must not be estimated."""
    return exact_expectation(theta_p, tab, correct_p)


def make_task(env: TokenEnv, tab: SeqTable, n_prompts: int, rng: np.random.Generator,
              flavour: str = "mixed") -> Task:
    """Verifier masks. `flavour` decides WHICH sequences count as correct:

      mixed  -- a random ~half of the 15 sequences (pass rates land near 0.5)
      long   -- only sequences of maximal length  (reward correlates with |y|)
      sharp  -- exactly ONE correct sequence      (a unique optimum to collapse onto)
      easy   -- the sequences the initial policy already prefers (saturated)
    """
    n = tab.n_seqs
    if flavour == "long":
        base = (tab.lengths == tab.lengths.max()).astype(np.float64)
        return Task(env, tab, np.repeat(base[None, :], n_prompts, axis=0))
    if flavour == "sharp":
        rows = []
        for _ in range(n_prompts):
            m = np.zeros(n)
            m[rng.integers(n)] = 1.0
            rows.append(m)
        return Task(env, tab, np.stack(rows))
    rows = []
    for _ in range(n_prompts):
        m = np.zeros(n)
        m[rng.choice(n, size=n // 2, replace=False)] = 1.0
        rows.append(m)
    return Task(env, tab, np.stack(rows))


def sample_group_indices(probs: np.ndarray, G: int, n_groups: int,
                         rng: np.random.Generator) -> np.ndarray:
    """(n_groups, G) sequence indices drawn i.i.d. from `probs`. Provided.

    Sampling sequence INDICES from the exact 15-way categorical is identical in
    distribution to walking the token tree, and it makes the whole experiment
    vectorisable.
    """
    return rng.choice(len(probs), size=(n_groups, G), p=probs)


# ---------------------------------------------------------------------------
# Config / history
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunCfg:
    steps: int = 400
    group: int = 8               # G -- completions per prompt
    lr: float = 0.5
    beta: float = 0.02           # KL coefficient, applied IN THE LOSS (see kl_penalty)
    eps: float = 0.2             # PPO clip
    mu: int = 1                  # inner epochs per rollout batch
    adv_mode: str = "centered"
    length_norm: str = "token"
    kl_kind: str = "k3"
    seed: int = 0
    flavour: str = "mixed"
    init_scale: float = 0.5
    n_prompts: int = 6


@dataclass
class History:
    """The instrument panel. Every field is a LEVEL, recorded per step.

    `diagnose_grpo` reads only this -- it never sees the task, the config, or a
    gold reward. A rule you cannot state in terms of these columns is a rule you
    cannot ship.
    """
    step: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pass_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))   # EXACT, mean over prompts
    dead_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))   # groups with A == 0
    adv_absmean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kl: np.ndarray = field(default_factory=lambda: np.zeros(0))          # EXACT, mean over prompts
    entropy: np.ndarray = field(default_factory=lambda: np.zeros(0))     # EXACT, nats
    elen: np.ndarray = field(default_factory=lambda: np.zeros(0))        # E[|y|], EXACT
    clip_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# Exercises -- the four TODO bodies
# ---------------------------------------------------------------------------

def group_advantage(rewards: np.ndarray, mode: str = "centered") -> np.ndarray:
    """GRPO's advantage: a WITHIN-GROUP CONTRAST. Operates on the LAST axis.

    `rewards` is (..., G) so this works on one group (G,) and on a whole batch of
    groups (n_groups, G) with the same code. Use `axis=-1, keepdims=True` -- this
    is the numpy form of W1's `keepdim=True` fact, and forgetting it broadcasts
    the wrong way without erroring when the leading dim happens to equal G.

    Three modes, and the differences between them are the whole derivation:

      "centered"    A_i = r_i - mean(r)
                    E[(1/G) sum_i A_i grad log pi(o_i)] = ((G-1)/G) * grad J,
                    EXACTLY -- a deterministic scalar shrinkage, not a bias in
                    direction.

      "normalized"  A_i = (r_i - mean(r)) / std(r)          [population std, ddof=0]
                    std is a NONLINEAR function of the same samples, so this is
                    genuinely biased. It also re-weights each prompt by
                    sqrt(p(1-p)) instead of p(1-p) -- i.e. by the standard
                    deviation instead of the variance, which is the information.

      "loo"         A_i = r_i - mean_{j != i}(r_j)
                    Removes the shrinkage exactly: E[...] = grad J.

    THE EDGE CASES ARE THE POINT -- decide each deliberately, do not let numpy
    decide for you:
      * G == 1: "centered" must give exactly 0.0 (there is no contrast). "loo"
        divides by G-1 == 0; it must ALSO return 0.0 rather than nan, because a
        group of one carries no signal under any baseline and a nan would poison
        an otherwise healthy batch.
      * std == 0 (every reward in the group equal -- the DEAD GROUP, and for a
        binary verifier this is "all right" or "all wrong"): the numerator is
        exactly 0, so "normalized" must return exactly 0.0, NOT 0/0. Guard the
        denominator, do not clamp the numerator.

    Returns an array with the same shape as `rewards`, float64.
    """
    # TODO
    raise NotImplementedError


def grpo_surrogate(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                   mask: torch.Tensor, eps: float = 0.2,
                   length_norm: str = "token") -> torch.Tensor:
    """PPO's clipped surrogate with a GROUP advantage. Returns a scalar to MAXIMISE.

    Shapes -- and the level trap is right here, so read them twice:
        logp, logp_old, mask   (G, T)     PER-TOKEN
        adv                    (G,)       PER-SEQUENCE
    The advantage must be BROADCAST across tokens (`adv[:, None]`), because GRPO
    assigns one number to a whole rollout. That single fact is what "GRPO has no
    intra-sequence credit assignment" means in code. Eight sightings of the
    wrong-level family so far; this is where number nine is waiting.

        rho_t = exp(logp_t - logp_old_t)
        per_token = min( rho_t * A , clip(rho_t, 1-eps, 1+eps) * A )

    The `min` is not decoration: it takes the WORSE of the two branches, which is
    what makes the surrogate a pessimistic lower bound. A comma is not a min.

    `length_norm` decides how per-token values become a scalar:
        "token"  GRPO as published:  (1/G) sum_i [ (1/|o_i|) sum_t ... ]
                 -- each SEQUENCE gets equal weight, so a token in a long
                 sequence counts for less. For a NEGATIVE advantage that means a
                 long wrong answer is penalised less per token: this is Dr. GRPO's
                 response-level length bias.
        "batch"  Dr. GRPO:  ( sum_{i,t} ... ) / ( total live tokens )
                 -- each TOKEN gets equal weight.
    Masked positions must contribute exactly nothing under both. Never divide by
    `numel()`; the denominator is a masked count (W1's running fact).
    """
    # TODO
    raise NotImplementedError


def kl_penalty(logp: torch.Tensor, logp_ref: torch.Tensor,
               kind: str = "k3") -> torch.Tensor:
    """Per-token KL estimator, to be added to the LOSS (never to the reward).

    Why never to the reward: the group mean ANNIHILATES anything common to a
    group, and drift from pi_ref is almost entirely common-mode, so a KL folded
    into the reward survives only as its deviation from the group mean. The
    `annihilate` stage measures how little is left.

    With samples from pi_theta and the ratio  rho = pi_ref / pi_theta  (so that
    E_{pi_theta}[rho] = 1), write  l = log rho = logp_ref - logp:

        k1 = -l                   unbiased; NEGATIVE on a large fraction of samples
        k3 = rho - 1 - l          unbiased AND >= 0 pointwise  (log rho <= rho - 1)

    k3 is DeepSeekMath Eq. (4). The reason to prefer it here is not variance --
    var(k1)/var(k3) REVERSES with KL (423x at KL 0.004, 0.23x at ~0.65). It is
    that a term added straight into a minimised loss must never pay the policy to
    leave pi_ref, and k1 does exactly that whenever it is negative.

    Returns (G, T), same shape as the inputs, unmasked (the caller masks).
    Build k3 from `l`; do not exponentiate a raw log-prob difference twice.
    """
    # TODO
    raise NotImplementedError


def diagnose_grpo(h: History) -> list[tuple[str, str]]:
    """Read the panel, name what is wrong. Return [(name, detail), ...]; [] = healthy.

    THIS ONE IS YOURS ON PURPOSE. Five sessions of evidence say you write the
    mathematical core and delegate the scaffolding, so the DESIGN skills --
    thresholds with air on both sides, negative controls, rules keyed on levels --
    are the ones going unpractised. This is that practice.

    Three rules, these exact names:

      "dead_groups"       too much of the batch carries no signal at all.
                          dead_frac is the fraction of groups whose advantage is
                          identically zero. This is the failure that LOOKS like
                          success: it arrives BECAUSE pass_rate went up.
      "kl_runaway"        the policy has left the region pi_ref vouches for.
      "entropy_collapse"  the policy has gone (near) deterministic.

    (A fourth rule, "length_drift", was cut during calibration. E[|y|] can only
    travel from 1.985 to 3.0 in a t_max=3 world, and EVERY run that learns
    lengthens, so a healthy run at ratio 1.14 and a broken one at 1.35 have no
    gap worth a threshold. Cutting a rule the instrument cannot support is the
    correct move, and noticing it is the skill this TODO is really about.)

    THE CONTRACT, and each clause is a real constraint:

      1. SILENT ON A HEALTHY RUN. The `diagnose` stage asserts `== []` on a
         healthy fixture. A rule that fires on everything is not a rule.
      2. KEY ON LEVELS, NOT CHANGES. W5 D5's M4 fixture was wrong for exactly
         this reason: off-data mass FELL during training while the level stayed
         catastrophic, so a delta rule never fired. Ask "where is it now?"
         before "which way did it move?".
      3. AIR ON BOTH SIDES. Every threshold must sit strictly between the
         healthy fixture's value and the broken one's, with room. Print the two
         numbers, pick the gap, and do not tune until the tests pass.
      4. READ THE TAIL, NOT ONE POINT. A single step is noise. Use the last
         ~20% of the run (`_tail` is provided), never `h.x[-1]` alone.

    The three symptoms are NOT independent, and the stage asserts the asymmetry:
    a run can leave pi_ref while staying stochastic, but it cannot collapse to a
    point without leaving pi_ref. Expect the entropy fixture to fire two rules.

    `detail` is a short string with the NUMBER that fired the rule in it -- an
    alert without its measurement is not actionable.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Training loop (provided -- plumbing, not the lesson)
# ---------------------------------------------------------------------------

def _tail(x: np.ndarray, frac: float = 0.2) -> np.ndarray:
    k = max(1, int(round(len(x) * frac)))
    return x[-k:]


def train_grpo(task: Task, cfg: RunCfg) -> History:
    """A real GRPO loop on the tabular task. Provided.

    Order of operations, and every one of them is load-bearing:
      sample G completions per prompt from pi_theta  (this is pi_old)
        -> verify  -> group advantage  -> mu inner epochs of the clipped
        surrogate  -> minus beta * KL, IN THE LOSS  -> step.
    """
    tab = task.tab
    P, n = task.n_prompts, tab.n_seqs
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    theta = torch.tensor(
        cfg.init_scale * np.asarray(
            np.random.default_rng(cfg.seed + 1000).standard_normal(
                (P, task.env.n_states, task.env.vocab))),
        dtype=torch.float64, requires_grad=True)
    theta_ref = theta.detach().clone().numpy()

    mask_all = torch.as_tensor(tab.mask)
    lengths = torch.as_tensor(tab.lengths, dtype=torch.float64)
    opt = torch.optim.SGD([theta], lr=cfg.lr)

    rec = {k: [] for k in ("step", "pass_rate", "dead_frac", "adv_absmean",
                           "kl", "entropy", "elen", "clip_frac", "loss")}

    for step in range(cfg.steps):
        th_np = theta.detach().numpy()
        idx = np.empty((P, cfg.group), dtype=np.int64)
        for p in range(P):
            pr = np.exp(seq_logprobs_np(th_np[p], tab))
            pr = pr / pr.sum()
            idx[p] = rng.choice(n, size=cfg.group, p=pr)

        rewards = np.take_along_axis(task.correct, idx, axis=1)         # (P, G)
        adv_np = group_advantage(rewards, cfg.adv_mode)                 # (P, G)
        dead = float(np.mean(np.all(np.abs(adv_np) < 1e-12, axis=1)))
        live = np.abs(adv_np) > 1e-12
        absmean = float(np.abs(adv_np)[live].mean()) if live.any() else 0.0

        adv_t = torch.as_tensor(adv_np)                                 # THE BOUNDARY
        idx_t = torch.as_tensor(idx)
        rows = torch.arange(P)[:, None]                                 # (P, 1)

        with torch.no_grad():
            logp_old = seq_token_logprobs_torch(theta, tab)[rows, idx_t]      # (P,G,T)
        logp_ref = seq_token_logprobs_torch(
            torch.as_tensor(theta_ref), tab)[rows, idx_t]

        mask = mask_all[idx_t]                                          # (P, G, T)
        last_clip, last_loss = 0.0, 0.0

        # The panel describes the policy that PRODUCED this batch, so read it here,
        # before any update. Row 0 is therefore the reference checkpoint itself and
        # h.kl[0] is exactly 0 -- which is what makes h.<metric>[0] a usable baseline.
        pass_rate = float(np.mean(
            [exact_pass_rate(th_np[p], tab, task.correct[p]) for p in range(P)]))
        kl_now = float(np.mean([exact_kl(th_np[p], theta_ref[p], tab) for p in range(P)]))
        ent_now = float(np.mean([exact_entropy(th_np[p], tab) for p in range(P)]))
        elen_now = float(np.mean(
            [exact_expectation(th_np[p], tab, tab.lengths.astype(np.float64))
             for p in range(P)]))

        for _ in range(max(1, cfg.mu)):
            logp = seq_token_logprobs_torch(theta, tab)[rows, idx_t]     # (P, G, T)
            surr = torch.stack([
                grpo_surrogate(logp[p], logp_old[p], adv_t[p], mask[p],
                               cfg.eps, cfg.length_norm) for p in range(P)]).mean()
            if cfg.beta > 0.0:
                kl_tok = kl_penalty(logp, logp_ref, cfg.kl_kind)
                kl_term = (kl_tok * mask).sum() / mask.sum()
            else:
                kl_term = torch.zeros((), dtype=torch.float64)
            loss = -(surr - cfg.beta * kl_term)

            # measured AT LOSS TIME, not after the step: the question "did the clip
            # change the objective" is about the rho the surrogate actually saw. Note
            # the condition is the SIGN condition -- rho out of band is not enough.
            with torch.no_grad():
                rho = torch.exp(logp - logp_old)
                bound = (((rho > 1 + cfg.eps) & (adv_t[..., None] > 0)) |
                         ((rho < 1 - cfg.eps) & (adv_t[..., None] < 0))) & mask
                last_clip = max(last_clip, float(bound.sum() / mask.sum()))
                last_loss = float(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

        rec["step"].append(step)
        rec["pass_rate"].append(pass_rate)
        rec["kl"].append(kl_now)
        rec["entropy"].append(ent_now)
        rec["elen"].append(elen_now)
        rec["dead_frac"].append(dead)
        rec["adv_absmean"].append(absmean)
        rec["clip_frac"].append(last_clip)
        rec["loss"].append(last_loss)

    return History(**{k: np.asarray(v, dtype=np.float64) for k, v in rec.items()})


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
        # --fast shrinks the MONTE-CARLO counts only. Step counts are NOT reduced:
        # a fixture that behaves differently under --fast is a fixture you cannot
        # trust, and the training runs cost about a second anyway.
        self.n_groups = 40_000 if fast else 400_000
        self.sim_groups = 20_000 if fast else 200_000
        self.steps = 400

    @cached_property
    def theta1(self) -> np.ndarray:
        """One prompt's parameters -- deliberately NOT near-uniform, so the exact
        gradient has structure the Monte-Carlo estimate has to reproduce."""
        return 0.9 * np.random.default_rng(7).standard_normal(
            (self.env.n_states, self.env.vocab))

    @cached_property
    def rewards1(self) -> np.ndarray:
        """A graded (non-binary) reward for the shrinkage test: binary rewards make
        too many groups dead, which would confound the estimator check."""
        return np.random.default_rng(11).uniform(-1.0, 1.0, size=self.tab.n_seqs)

    @cached_property
    def probs1(self) -> np.ndarray:
        p = np.exp(seq_logprobs_np(self.theta1, self.tab))
        return p / p.sum()

    @cached_property
    def scores1(self) -> np.ndarray:
        return all_score_functions(self.env, self.theta1, self.tab)

    def run(self, **kw) -> History:
        cfg = RunCfg(steps=self.steps, **kw)
        rng = np.random.default_rng(cfg.seed + 500)
        task = make_task(self.env, self.tab, cfg.n_prompts, rng, cfg.flavour)
        if cfg.flavour == "easy":
            task = self._saturate(task, cfg)
        return train_grpo(task, cfg)

    def _saturate(self, task: Task, cfg: RunCfg) -> Task:
        """Verifier = the sequences the INIT policy already prefers, so the run
        starts near p ~ 0.95 and the dead-group law bites immediately."""
        th = cfg.init_scale * np.random.default_rng(cfg.seed + 1000).standard_normal(
            (cfg.n_prompts, self.env.n_states, self.env.vocab))
        rows = []
        for p in range(cfg.n_prompts):
            pr = np.exp(seq_logprobs_np(th[p], self.tab))
            order = np.argsort(-pr)
            m = np.zeros(self.tab.n_seqs)
            take, acc = [], 0.0
            for y in order:
                take.append(y)
                acc += pr[y]
                if acc >= 0.95:
                    break
            m[take] = 1.0
            rows.append(m)
        return Task(task.env, task.tab, np.stack(rows))

    @cached_property
    def h_healthy(self) -> History:
        print("  ... training the HEALTHY fixture")
        return self.run(seed=0, flavour="mixed", beta=0.20, lr=0.3)

    @cached_property
    def h_saturated(self) -> History:
        print("  ... training the SATURATED fixture (dead groups)")
        return self.run(seed=0, flavour="easy", beta=0.02, lr=0.5)

    @cached_property
    def h_nokl(self) -> History:
        print("  ... training the NO-KL fixture (kl runaway)")
        return self.run(seed=0, flavour="sharp", beta=0.0, lr=1.5)

    @cached_property
    def h_greedy(self) -> History:
        print("  ... training the GREEDY fixture (entropy collapse)")
        return self.run(seed=0, flavour="sharp", beta=0.0, lr=8.0)


def _fail_fast(ctx: Ctx) -> None:
    """Touch every TODO a heavy stage needs, on a 1-element input, so a missing
    implementation surfaces BEFORE the expensive fixture is built."""
    group_advantage(np.zeros((1, 2)), "centered")
    z = torch.zeros((1, 1), dtype=torch.float64)
    grpo_surrogate(z, z, torch.zeros(1, dtype=torch.float64),
                   torch.ones((1, 1), dtype=torch.bool))
    kl_penalty(z, z, "k3")


# ---- stages ---------------------------------------------------------------

def stage_advantage(ctx: Ctx) -> None:
    """group_advantage -- closed forms, zero-sum, and the two edge cases"""
    r = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])       # G=8, k=2
    c = group_advantage(r, "centered")
    _check("centered: shape and dtype preserved",
           c.shape == r.shape and np.issubdtype(c.dtype, np.floating), f"{c.shape} {c.dtype}")
    _check("centered: r - mean, exactly",
           np.allclose(c, r - r.mean(), atol=0, rtol=0), f"{np.round(c, 4)}")
    _check("centered: sums to exactly 0", abs(float(c.sum())) < 1e-15, f"{c.sum():.2e}")

    nrm = group_advantage(r, "normalized")
    G, k = 8, 2
    _check("normalized: success == +sqrt((G-k)/k)",
           abs(nrm[0] - math.sqrt((G - k) / k)) < 1e-12,
           f"{nrm[0]:.6f} vs {math.sqrt((G-k)/k):.6f}")
    _check("normalized: failure == -sqrt(k/(G-k))",
           abs(nrm[-1] + math.sqrt(k / (G - k))) < 1e-12,
           f"{nrm[-1]:.6f} vs {-math.sqrt(k/(G-k)):.6f}")
    _check("normalized: still sums to 0", abs(float(nrm.sum())) < 1e-13, f"{nrm.sum():.2e}")

    loo = group_advantage(r, "loo")
    _check("loo == centered * G/(G-1), elementwise",
           np.allclose(loo, c * G / (G - 1)), f"max dev {np.abs(loo - c*G/(G-1)).max():.2e}")

    batch = np.stack([r, 1.0 - r, np.full(8, 0.5)])
    cb = group_advantage(batch, "centered")
    _check("operates on the LAST axis (batch of groups)",
           cb.shape == batch.shape and np.allclose(cb[0], c),
           f"{cb.shape}")
    _check("a constant group is exactly 0 under centered",
           np.all(cb[2] == 0.0), f"{cb[2]}")
    _check("a constant group is exactly 0 under normalized (no 0/0)",
           np.all(group_advantage(batch, "normalized")[2] == 0.0)
           and np.isfinite(group_advantage(batch, "normalized")).all(), "")

    one = group_advantage(np.array([[3.7]]), "centered")
    _check("G=1 centered is exactly 0 -- THE defining property",
           one.shape == (1, 1) and float(one[0, 0]) == 0.0, f"{one}")
    _check("G=1 loo is 0 and finite (not nan)",
           np.isfinite(group_advantage(np.array([[3.7]]), "loo")).all()
           and float(group_advantage(np.array([[3.7]]), "loo")[0, 0]) == 0.0, "")


def stage_shrinkage(ctx: Ctx) -> None:
    """the EXACT (G-1)/G factor, Monte-Carlo vs enumeration"""
    _fail_fast(ctx)
    exact = exact_policy_gradient(ctx.env, ctx.theta1, ctx.tab, ctx.rewards1)
    nrm_exact = float(np.linalg.norm(exact))
    _check("the exact gradient is non-trivial", nrm_exact > 0.05, f"||grad J|| = {nrm_exact:.4f}")

    rng = np.random.default_rng(3)
    n = ctx.n_groups
    for G in (2, 4, 8):
        idx = sample_group_indices(ctx.probs1, G, n, rng)
        radv = group_advantage(ctx.rewards1[idx], "centered")
        w = np.bincount(idx.ravel(), weights=radv.ravel(), minlength=ctx.tab.n_seqs)
        ghat = np.tensordot(w, ctx.scores1, axes=1) / (n * G)
        ratio = float(np.sum(ghat * exact) / np.sum(exact * exact))
        want = (G - 1) / G
        _check(f"G={G}: E[ghat] / gradJ == (G-1)/G = {want:.4f}",
               abs(ratio - want) < 0.02, f"measured {ratio:.4f}")
        cos = float(np.sum(ghat * exact) /
                    (np.linalg.norm(ghat) * nrm_exact))
        _check(f"G={G}: direction is UNCHANGED (cos > 0.99)", cos > 0.99, f"cos {cos:.6f}")

        radv = group_advantage(ctx.rewards1[idx], "loo")
        w = np.bincount(idx.ravel(), weights=radv.ravel(), minlength=ctx.tab.n_seqs)
        gloo = np.tensordot(w, ctx.scores1, axes=1) / (n * G)
        rloo = float(np.sum(gloo * exact) / np.sum(exact * exact))
        _check(f"G={G}: LOO removes the shrinkage (ratio -> 1)",
               abs(rloo - 1.0) < 0.03, f"measured {rloo:.4f}")

    _check("G=1 gives an IDENTICALLY zero gradient, not a small one",
           float(np.abs(group_advantage(ctx.rewards1[
               sample_group_indices(ctx.probs1, 1, 1000, rng)], "centered")).max()) == 0.0, "")


def stage_annihilate(ctx: Ctx) -> None:
    """the group mean is a HIGH-PASS FILTER: c(x) dies exactly; KL-in-reward dies with it"""
    rng = np.random.default_rng(5)
    idx = sample_group_indices(ctx.probs1, 8, 2000, rng)
    r = ctx.rewards1[idx]
    a0 = group_advantage(r, "centered")
    for c in (0.0, 1.0, -37.5, 1e6):
        a = group_advantage(r + c, "centered")
        _check(f"c(x) = {c:g} is annihilated EXACTLY (not in expectation)",
               float(np.abs(a - a0).max()) < 1e-9 * max(1.0, abs(c)),
               f"max dev {np.abs(a - a0).max():.3e}")

    kl_seq = np.abs(ctx.rewards1) * 0.5 + 2.0          # per-sequence "KL", strongly common-mode
    beta = 0.3
    shaped = group_advantage(r - beta * kl_seq[idx], "centered")
    unshaped = group_advantage(r, "centered")
    survives = float(np.abs(shaped - unshaped).mean())
    intended = float(beta * np.abs(kl_seq[idx] - kl_seq[idx].mean(-1, keepdims=True)).mean())
    full = float(beta * np.abs(kl_seq[idx]).mean())
    _check("what survives of a KL-in-reward is ONLY its within-group deviation",
           abs(survives - intended) < 1e-9, f"survives {survives:.6f} vs deviation {intended:.6f}")
    _check("the common-mode part is deleted (survives << full penalty)",
           survives < 0.35 * full, f"{survives:.4f} vs {full:.4f} = {full/max(survives,1e-12):.2f}x")

    const_kl = np.full(ctx.tab.n_seqs, 4.0)
    same = group_advantage(r - beta * const_kl[idx], "centered")
    _check("a UNIFORM drift penalty is invisible to the gradient (exactly)",
           float(np.abs(same - unshaped).max()) < 1e-12,
           f"max dev {np.abs(same - unshaped).max():.2e}")


def stage_stdbias(ctx: Ctx) -> None:
    """1/std swaps VARIANCE weighting for STD weighting -- the difficulty bias"""
    G = 8
    for k in (1, 2, 4, 6, 7):
        r = np.zeros(G)
        r[:k] = 1.0
        cen = group_advantage(r, "centered")
        nrm = group_advantage(r, "normalized")
        p = k / G
        _check(f"k={k}: centered group weight == 2G*p(1-p) (VARIANCE)",
               abs(float(np.abs(cen).sum()) - 2 * G * p * (1 - p)) < 1e-12,
               f"{np.abs(cen).sum():.4f}")
        _check(f"k={k}: normalized group weight == 2G*sqrt(p(1-p)) (STD)",
               abs(float(np.abs(nrm).sum()) - 2 * G * math.sqrt(p * (1 - p))) < 1e-10,
               f"{np.abs(nrm).sum():.4f}")

    def amp(k):
        p = k / G
        return 1.0 / math.sqrt(p * (1 - p))
    _check("the amplification is exactly 1/std and is MINIMAL at the balanced prompt",
           amp(4) < amp(2) < amp(1) and abs(amp(4) - 2.0) < 1e-12,
           f"k=4 {amp(4):.4f} < k=2 {amp(2):.4f} < k=1 {amp(1):.4f}")
    _check("k=1 is amplified 1.51x relative to k=4 -- the least informative prompt wins",
           abs(amp(1) / amp(4) - 1.5119) < 1e-3, f"{amp(1)/amp(4):.4f}x")

    # negative control: the ORDERING of total weight is not reversed, only compressed.
    cen_w = [2 * G * (k / G) * (1 - k / G) for k in (1, 4)]
    nrm_w = [2 * G * math.sqrt((k / G) * (1 - k / G)) for k in (1, 4)]
    _check("CONTROL: balanced still outweighs extreme under BOTH -- the bias compresses, "
           "it does not invert",
           cen_w[1] > cen_w[0] and nrm_w[1] > nrm_w[0],
           f"centered {cen_w[0]:.2f}<{cen_w[1]:.2f}, normalized {nrm_w[0]:.2f}<{nrm_w[1]:.2f}")
    _check("but the RATIO collapses 2.29x -> 1.51x, which is the whole effect",
           abs(cen_w[1] / cen_w[0] - 2.2857) < 1e-3 and abs(nrm_w[1] / nrm_w[0] - 1.5119) < 1e-3,
           f"{cen_w[1]/cen_w[0]:.4f} -> {nrm_w[1]/nrm_w[0]:.4f}")


def stage_deadgroup(ctx: Ctx) -> None:
    """P(dead) = p^G + (1-p)^G, simulated at 5 seeds against the closed form"""
    G = 8
    closed = {p: p ** G + (1 - p) ** G for p in (0.5, 0.7, 0.9, 0.95, 0.99)}
    for p, want in closed.items():
        got = []
        for s in range(5):                       # >= 5 seeds: W5 D5's floor
            rng = np.random.default_rng(100 + s)
            r = (rng.random((ctx.sim_groups, G)) < p).astype(np.float64)
            a = group_advantage(r, "centered")
            got.append(float(np.mean(np.all(a == 0.0, axis=1))))
        m, sd = float(np.mean(got)), float(np.std(got))
        _check(f"p={p}: simulated dead fraction == p^G+(1-p)^G = {want:.4f}",
               abs(m - want) < 4 * sd + 0.004, f"{m:.4f} +/- {sd:.4f}")

    _check("N_eff/N at p=0.95, G=8 is barely a third of the batch",
           abs((1 - closed[0.95]) - 0.33658) < 1e-4, f"{1 - closed[0.95]:.5f}")
    _check("raising pass rate 0.90 -> 0.95 KILLS 23% more of the batch",
           abs((closed[0.95] - closed[0.9]) - 0.23295) < 1e-4,
           f"{closed[0.9]:.4f} -> {closed[0.95]:.4f}")

    need = next(g for g in range(2, 400) if 1 - (0.95 ** g + 0.05 ** g) >= 0.9)
    _check("keeping 90% of prompts alive at p=0.95 needs G = 45",
           need == 45, f"G={need}; ln(0.1)/ln(0.95) = {math.log(0.1)/math.log(0.95):.3f}")
    _check("...and G=44 is NOT enough (the threshold is sharp, not a rounding)",
           1 - (0.95 ** 44 + 0.05 ** 44) < 0.9,
           f"G=44 gives {100*(1-(0.95**44+0.05**44)):.3f}%")


def stage_surrogate(ctx: Ctx) -> None:
    """grpo_surrogate -- levels, masking, and whether the clip does ANYTHING at mu=1"""
    _fail_fast(ctx)
    torch.manual_seed(0)
    G, T = 6, 3
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0],
                         [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    logp_old = torch.log(torch.rand((G, T), dtype=torch.float64) * 0.5 + 0.25)
    adv = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5], dtype=torch.float64)

    same = grpo_surrogate(logp_old.clone(), logp_old, adv, mask, 0.2, "token")
    want = float((adv * torch.ones(G, dtype=torch.float64)).mean())
    _check("mu=1 / on-policy: rho == 1 exactly, so the surrogate == mean(A)",
           abs(float(same) - want) < 1e-12, f"{float(same):.12f} vs {want:.12f}")

    hi = logp_old + math.log(1.6)                     # rho = 1.6, far outside the band
    out = grpo_surrogate(hi, logp_old, adv, mask, 0.2, "token")
    pos = adv.clamp(min=0)
    neg = adv.clamp(max=0)
    want2 = float(((1.2 * pos) + (1.6 * neg)).mean())
    _check("A>0 is CLIPPED at 1+eps; A<0 is NOT -- the sign condition, in code",
           abs(float(out) - want2) < 1e-12, f"{float(out):.8f} vs {want2:.8f}")

    lo = logp_old + math.log(0.5)
    out2 = grpo_surrogate(lo, logp_old, adv, mask, 0.2, "token")
    want3 = float(((0.5 * pos) + (0.8 * neg)).mean())
    _check("A<0 is CLIPPED at 1-eps; A>0 is NOT (the mirror case)",
           abs(float(out2) - want3) < 1e-12, f"{float(out2):.8f} vs {want3:.8f}")

    poison = logp_old.clone()
    poison[~mask] = -99.0
    _check("masked positions contribute exactly nothing",
           abs(float(grpo_surrogate(poison, logp_old, adv, mask, 0.2, "token"))
               - float(same)) < 1e-12, "")

    a_shift = adv + 10.0
    s1 = grpo_surrogate(logp_old.clone(), logp_old, adv, mask, 0.2, "token")
    s2 = grpo_surrogate(logp_old.clone(), logp_old, a_shift, mask, 0.2, "token")
    _check("the advantage enters PER SEQUENCE, broadcast over tokens (level check)",
           abs((float(s2) - float(s1)) - 10.0) < 1e-12,
           f"delta {float(s2)-float(s1):.10f}")

    tok = grpo_surrogate(hi, logp_old, adv, mask, 0.2, "token")
    bat = grpo_surrogate(hi, logp_old, adv, mask, 0.2, "batch")
    _check("length_norm changes the answer (the two are NOT the same reduction)",
           abs(float(tok) - float(bat)) > 1e-3, f"token {float(tok):.6f} vs batch {float(bat):.6f}")

    # The length bias is a statement about the PER-TOKEN GRADIENT, not about the
    # scalar. At rho == 1 with a CONSTANT advantage the two reductions are equal by
    # construction ((1/G)sum_i A_i == sum_i L_i A_i / sum_i L_i), so asserting on the
    # objective would be a fixture on which the naive answer is right. Differentiate.
    lp = logp_old.clone().requires_grad_(True)
    neg = torch.full((G,), -1.0, dtype=torch.float64)
    ratios = {}
    for norm in ("token", "batch"):
        if lp.grad is not None:
            lp.grad = None
        grpo_surrogate(lp, logp_old, neg, mask, 0.2, norm).backward()
        g = lp.grad.abs()
        ratios[norm] = float(g[2, 0] / g[0, 0])          # |o|=1 token vs |o|=3 token
    _check("'token': a token in a 3-long sequence is pushed 3x LESS hard than one in a "
           "1-long sequence -- Dr. GRPO's length bias, exactly",
           abs(ratios["token"] - 3.0) < 1e-9, f"ratio {ratios['token']:.6f}")
    _check("CONTROL 'batch': every token weighs the same regardless of length (ratio 1)",
           abs(ratios["batch"] - 1.0) < 1e-9, f"ratio {ratios['batch']:.6f}")
    _check("=> with A < 0, 'token' penalises LONG wrong answers less PER TOKEN, "
           "which is the pressure toward length", ratios["token"] > ratios["batch"],
           f"{ratios['token']:.2f} vs {ratios['batch']:.2f}")


def stage_kl(ctx: Ctx) -> None:
    """kl_penalty -- k1 unbiased but signed, k3 unbiased AND non-negative"""
    _fail_fast(ctx)
    tab = ctx.tab
    th = ctx.theta1
    th_ref = th + 0.35 * np.random.default_rng(21).standard_normal(th.shape)
    lp = seq_logprobs_np(th, tab)
    lpr = seq_logprobs_np(th_ref, tab)
    p = np.exp(lp)
    true_kl = float(np.sum(p * (lp - lpr)))
    _check("the fixture's exact sequence KL is non-trivial", true_kl > 0.05, f"KL {true_kl:.4f}")

    tok = torch.as_tensor(seq_token_logprobs_torch(torch.as_tensor(th), tab))
    tok_ref = torch.as_tensor(seq_token_logprobs_torch(torch.as_tensor(th_ref), tab))
    m = torch.as_tensor(tab.mask)
    w = torch.as_tensor(p)

    for kind in ("k1", "k3"):
        est = kl_penalty(tok, tok_ref, kind)
        _check(f"{kind}: returns the input shape", est.shape == tok.shape, f"{tuple(est.shape)}")
        per_seq = (est * m).sum(1)
        got = float((w * per_seq).sum())
        _check(f"{kind} is UNBIASED: E[{kind}] == exact KL",
               abs(got - true_kl) < 1e-9, f"{got:.10f} vs {true_kl:.10f}")

    k1 = (kl_penalty(tok, tok_ref, "k1") * m).sum(1).detach().numpy()
    k3 = (kl_penalty(tok, tok_ref, "k3") * m).sum(1).detach().numpy()
    _check("k3 >= 0 POINTWISE, on every one of the 15 sequences",
           bool((k3 >= -1e-12).all()), f"min {k3.min():.3e}")
    frac = float(np.sum(p[k1 < 0]))
    _check("k1 is NEGATIVE on a large fraction of the mass -- it would PAY the policy "
           "to leave pi_ref", frac > 0.15, f"{100*frac:.1f}% of probability mass")
    _check("...and k3 is never negative on that same mass (the discriminator)",
           float(np.sum(p[k3 < 0])) == 0.0, "0.0%")


def stage_clipinert(ctx: Ctx) -> None:
    """is the clip doing ANYTHING? mu=1 says no; mu=8 at the same lr is the control"""
    _fail_fast(ctx)
    h1 = ctx.run(seed=0, flavour="mixed", mu=1, lr=4.0, beta=0.02)
    _check("mu=1: the clip binds on EXACTLY 0 tokens, at every step",
           float(h1.clip_frac.max()) == 0.0, f"max {h1.clip_frac.max():.3e}")
    # single-variable control: same lr, same seed, same task -- only mu changes
    h8 = ctx.run(seed=0, flavour="mixed", mu=8, lr=4.0, beta=0.02)
    _check("CONTROL mu=8 at the SAME lr: it binds on a real fraction, so the check "
           "above is a property of mu=1 and not of this task",
           float(h8.clip_frac.max()) > 0.05,
           f"max {h8.clip_frac.max():.4f}, mean {h8.clip_frac.mean():.4f}")
    _check("=> GRPO at mu=1 is REINFORCE + a group baseline + a KL term, "
           "wearing PPO's notation", True, "the clip cannot bind when rho == 1")


def stage_train(ctx: Ctx) -> None:
    """the loop learns, and the panel records LEVELS"""
    _fail_fast(ctx)
    h = ctx.h_healthy
    _check("pass rate rises", h.pass_rate[-1] > h.pass_rate[0] + 0.05,
           f"{h.pass_rate[0]:.4f} -> {h.pass_rate[-1]:.4f}")
    _check("step 0 KL is exactly 0 (policy and reference are the same checkpoint)",
           float(h.kl[0]) < 1e-9, f"{h.kl[0]:.3e}")
    _check("the healthy run stays inside its KL budget", float(_tail(h.kl).mean()) < 1.0,
           f"tail KL {_tail(h.kl).mean():.4f}")
    _check("the healthy run keeps most of its batch alive",
           float(_tail(h.dead_frac).mean()) < 0.35,
           f"tail dead_frac {_tail(h.dead_frac).mean():.4f}")
    _check("entropy stays well off the floor", float(_tail(h.entropy).mean()) > 0.5,
           f"tail H {_tail(h.entropy).mean():.4f} nats")

    hs = ctx.h_saturated
    _check("the SATURATED run starts already-solved (that is the point)",
           hs.pass_rate[0] > 0.85, f"p0 = {hs.pass_rate[0]:.4f}")
    _check("=> its batch is mostly DEAD while its reward looks excellent",
           float(_tail(hs.dead_frac).mean()) > 0.55 and float(_tail(hs.pass_rate).mean()) > 0.9,
           f"dead {_tail(hs.dead_frac).mean():.4f} at pass {_tail(hs.pass_rate).mean():.4f}")
    _check("THE HEADLINE: dead_frac separates the two runs by a wide margin",
           _tail(hs.dead_frac).mean() > 2.0 * _tail(h.dead_frac).mean() + 0.2,
           f"healthy {_tail(h.dead_frac).mean():.4f} vs saturated "
           f"{_tail(hs.dead_frac).mean():.4f}")


def stage_diagnose(ctx: Ctx) -> None:
    """diagnose_grpo -- SILENT on healthy, and each fixture fires its own rule"""
    _fail_fast(ctx)
    fired = diagnose_grpo(ctx.h_healthy)
    _check("SILENT on the healthy run", fired == [], f"fired {[f[0] for f in fired]}")
    _check("returns a list of (name, detail) pairs", isinstance(fired, list), "")

    cases = [("dead_groups", ctx.h_saturated), ("kl_runaway", ctx.h_nokl),
             ("entropy_collapse", ctx.h_greedy)]
    for name, h in cases:
        details = dict(diagnose_grpo(h))
        _check(f"fires '{name}' on its fixture", name in details,
               f"fired {sorted(details)}")
        _check(f"'{name}' detail carries a number",
               any(ch.isdigit() for ch in details.get(name, "")),
               f"{details.get(name, '')!r}")

    allowed = {"dead_groups", "kl_runaway", "entropy_collapse"}
    for _, h in cases:
        _check("uses only the three contracted names",
               set(n for n, _ in diagnose_grpo(h)) <= allowed, "")

    # SPECIFICITY: the dead-group fixture must not set the other two alarms off.
    # Without this, "return all three names always" would pass every check above.
    sat = {n for n, _ in diagnose_grpo(ctx.h_saturated)}
    _check("the saturated fixture fires dead_groups and NOTHING ELSE -- so the rules "
           "are specific, not a blanket alarm", sat == {"dead_groups"}, f"{sorted(sat)}")

    # THE ASYMMETRY: KL runaway without entropy collapse is possible; the reverse
    # is not. A policy cannot concentrate onto a point and still sit near pi_ref.
    nok = {n for n, _ in diagnose_grpo(ctx.h_nokl)}
    gre = {n for n, _ in diagnose_grpo(ctx.h_greedy)}
    _check("a run can leave pi_ref while staying stochastic (kl fires, entropy does not)",
           nok == {"kl_runaway"}, f"{sorted(nok)}")
    _check("...but it cannot collapse to a point WITHOUT leaving pi_ref, so the entropy "
           "fixture necessarily fires kl_runaway too -- the symptoms are not independent",
           {"kl_runaway", "entropy_collapse"} <= gre, f"{sorted(gre)}")


STAGES = [
    ("advantage", stage_advantage),
    ("shrinkage", stage_shrinkage),
    ("annihilate", stage_annihilate),
    ("stdbias", stage_stdbias),
    ("deadgroup", stage_deadgroup),
    ("surrogate", stage_surrogate),
    ("kl", stage_kl),
    ("clipinert", stage_clipinert),
    ("train", stage_train),
    ("diagnose", stage_diagnose),
]


# ---- runner (copy verbatim; nothing below needs editing) -------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, fn in STAGES:
            print(f"  {name:<12} {fn.__doc__}")
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
