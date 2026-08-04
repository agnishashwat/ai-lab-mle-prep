"""
Week 5 Day 2 -- deriving DPO, and then past where the paper stops.

The derivation is the easy half and it is already yours. This file is the other half:
it turns four claims that are usually asserted into numbers.

  (a) THE EQUIVALENCE, VERIFIED. On a fixture whose comparison graph is strongly
      connected (so the BT MLE exists), `train_dpo` and `bt_then_tilt` -- two
      algorithms that share no code path -- must agree ELEMENTWISE on the resulting
      sequence distribution. Not "approximately". Machine precision. And the check is
      made non-vacuous by tilting at the WRONG beta, which differs by ~14 orders of
      magnitude more.

  (b) THE GAUGE. The implicit reward r_hat = beta*log(pi_theta/pi_ref) is not "the"
      reward -- it is the unique member of the Bradley-Terry equivalence class whose
      normalizer is 1. It equals r - beta*log Z(x), and `sum_y pi_ref(y) e^{r_hat/beta}`
      is identically 1 BECAUSE pi_theta is normalized. That identity is a CONSTRAINT:
      you cannot lower r_hat(y_l) without raising r_hat somewhere else.

  (c) THE DIVERGENCE. A DPO dataset is a perfect MATCHING, which D1 proved is the
      most disconnected non-trivial graph there is. So the BT MLE is at infinity, and
      under the reparameterization "r_l -> -inf" reads literally as "pi(y_l) -> 0".
      The training metrics look perfect the entire time.

  (d) BOTH LOGPROBS FALL. The gradient contains no term that constrains the marginal
      likelihood -- only the difference is penalized. Measured on a CONVERGED,
      well-posed fixture, so it is not a divergence artifact.

Two places where reality is more interesting than the session plan, and the file
asserts the true version:

  * "the KL grows without bound" is FALSE in a finite space. KL(pi_theta||pi_ref) is
    bounded by max_y -log pi_ref(y) (the point-mass bound), because pi_theta collapsing
    to a point mass is the worst it can do. Stage `divergence` asserts the KL rises
    monotonically AND stays under that bound, and prints both. In a real LM the bound
    is astronomically large, which is why "unbounded" is the practical statement.
  * "under mu != pi_ref the chosen logprob rises when mu == pi_ref" is not what a
    tabular model does -- the chosen logprob falls either way. The mismatch tax is real
    but it is paid in KL, not in the logprob's sign, and stage `mismatch` measures it.

  run everything   ./.venv/bin/python week-05-dpo-preference-optimization/code/dpo.py
  one stage        ...  dpo.py equivalence
  a few stages     ...  dpo.py grad implicit
  fast fixtures    ...  dpo.py --fast
  list the stages  ...  dpo.py --list

Build order (each group is runnable the moment it is written):

  1. dpo_loss_and_grad   -> stages `grad`, `weight`, `invariance`
  2. implicit_reward     -> stage `implicit`
  3. train_dpo           -> stages `divergence`, `logprobs`, `mismatch`
  4. bt_then_tilt        -> stage `equivalence`   (the headline)

Full run ~23 s, `--fast` ~12 s (only the two long training runs are shortened; the
equivalence check is never weakened). Run it under `-W error::RuntimeWarning`.
"""

from __future__ import annotations

import sys
import traceback
from functools import cached_property

import numpy as np

# D1's verified code, reused rather than rewritten (same directory).
from preference_models import fit_bt, sigmoid, logsigmoid, tilt


# ---------------------------------------------------------------------------
# Numerics + the enumerable environment (provided -- W4 D1, 73/73)
# ---------------------------------------------------------------------------


def log_softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=axis, keepdims=True)
    s = z - m
    return s - np.log(np.sum(np.exp(s), axis=axis, keepdims=True))


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax_np(z, axis=axis))


class TokenEnv:
    """W4 D1's token-level LM environment, small enough to enumerate exhaustively.

    Default here is vocab=3 (tokens 0, 1 + eos=2), t_max=3:
      n_states = 1 + 2 + 4 = 7   (prefixes with no eos and length < t_max)
      sequences = 1 + 2 + 4 + 8 = 15

    Small on purpose. Every claim today is an equality between two distributions over
    those 15 sequences, so the ground truth is exact and there is no sampling anywhere
    in this file.
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

    def random_theta(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        return scale * rng.standard_normal((self.n_states, self.vocab))

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


def sequence_logprob(env: TokenEnv, theta: np.ndarray, seq: tuple[int, ...]) -> float:
    lp = log_softmax_np(theta)
    return float(sum(lp[env.state_id(seq[:t]), seq[t]] for t in range(len(seq))))


def all_sequence_logprobs(env: TokenEnv, theta: np.ndarray,
                          seqs: list | None = None) -> np.ndarray:
    """log p_theta(y) for every enumerable y, aligned with `enumerate_sequences`."""
    seqs = enumerate_sequences(env) if seqs is None else seqs
    return np.array([sequence_logprob(env, theta, s) for s in seqs])


def grad_sequence_logprob(env: TokenEnv, theta: np.ndarray,
                          seq: tuple[int, ...]) -> np.ndarray:
    """The score function, shape (n_states, vocab). Provided -- this is W3 D2 material.

        d/d theta[s, a]  log p_theta(y)  =  sum_t  ( 1[s = s_t, a = y_t] - pi(a|s_t) )

    Two properties worth knowing before you use it, because stage `grad` asserts both:
      * every ROW sums to exactly zero (a softmax cannot change a state's total mass);
      * for two sequences sharing a prefix AND the same action on it, the per-state
        terms are IDENTICAL, so they cancel exactly in a difference of scores. In a
        tabular policy the shared prefix of a preference pair contributes NOTHING to
        the DPO gradient. The "pushing y_l down drags y_w down with it" story needs
        shared PARAMETERS, which a table does not have.
    """
    pi = softmax_np(theta)
    g = np.zeros_like(theta)
    for t in range(len(seq)):
        s = env.state_id(seq[:t])
        g[s, seq[t]] += 1.0
        g[s, :] -= pi[s, :]
    return g


def exact_kl(env: TokenEnv, theta: np.ndarray, theta_ref: np.ndarray,
             seqs: list | None = None) -> float:
    """KL(p_theta || p_ref) over whole sequences, by enumeration. No estimator."""
    seqs = enumerate_sequences(env) if seqs is None else seqs
    lp = all_sequence_logprobs(env, theta, seqs)
    lpr = all_sequence_logprobs(env, theta_ref, seqs)
    return float(np.sum(np.exp(lp) * (lp - lpr)))


# ---------------------------------------------------------------------------
# Exercises (TODO bodies)
# ---------------------------------------------------------------------------

def dpo_loss_and_grad(env: TokenEnv, theta: np.ndarray, pairs: np.ndarray,
                      theta_ref: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """The DPO loss and its ANALYTIC gradient. `grad.shape == theta.shape`.

    `pairs` is an int array of shape (m, 2); row k is (winner_index, loser_index) into
    `enumerate_sequences(env)`.

        M_k  = beta * [ (log pi_theta(y_w) - log pi_ref(y_w))
                      - (log pi_theta(y_l) - log pi_ref(y_l)) ]        the implicit margin
        L    = -mean_k log sigma(M_k)
        grad = -beta * mean_k sigma(-M_k) * ( grad_logp(y_w) - grad_logp(y_l) )

    Four things this must get right, each of which has its own check:

      * Use `logsigmoid`, never `np.log(sigmoid(...))`. The naive form returns exactly
        0.0 for M <~ 37, and stage `divergence` runs the margin to 10+ and beyond --
        i.e. straight into the region where the naive form silently reports zero loss
        on the pairs that are hurting you.
      * The weight `sigma(-M)` is the SAME function of the SAME object as the
        Bradley-Terry gradient weight you fitted on W4 D1 and again yesterday. That is
        the day's thesis in one line of code.
      * theta_ref is NOT differentiable in theta. It survives only inside M, i.e. only
        inside the scalar weight. Stage `grad` asserts the gradient DIRECTION is
        independent of theta_ref, which is a strong statement and an easy check to
        fail if you accidentally differentiate the reference term.
      * MEAN over pairs, not sum -- otherwise the equivalence check picks up a
        pair-count factor and lands nowhere near 1e-12.
    """
    sequences = enumerate_sequences(env)
    all_seq_logprobs = all_sequence_logprobs(env, theta, sequences)
    all_seq_logprobs_ref = all_sequence_logprobs(env, theta_ref, sequences)

    winner_seq_logprobs = all_seq_logprobs[pairs[:, 0]]
    loser_seq_logprobs = all_seq_logprobs[pairs[:, 1]]

    winner_seq_logprobs_ref = all_seq_logprobs_ref[pairs[:, 0]]
    loser_seq_logprobs_ref = all_seq_logprobs_ref[pairs[:, 1]]

    margin = beta * (winner_seq_logprobs - winner_seq_logprobs_ref - loser_seq_logprobs + loser_seq_logprobs_ref)   
    L = -logsigmoid(margin).mean()

    grad = np.zeros_like(theta, dtype=np.float64)

    for k in range(pairs.shape[0]):
        grad_seq_lopprob_winner = grad_sequence_logprob(env, theta, sequences[pairs[k, 0]])
        grad_seq_lopprob_loser = grad_sequence_logprob(env, theta, sequences[pairs[k, 1]])
        grad += -beta * sigmoid(-margin[k]) * (grad_seq_lopprob_winner - grad_seq_lopprob_loser)

    return (L, grad/pairs.shape[0])


def implicit_reward(env: TokenEnv, theta: np.ndarray, theta_ref: np.ndarray,
                    beta: float) -> np.ndarray:
    """r_hat(y) = beta * log( pi_theta(y) / pi_ref(y) ), over ALL enumerable sequences.

    Returns a vector aligned with `enumerate_sequences(env)`.

    This is the object that makes DPO a reward model. Three facts stage `implicit`
    checks, and the third is the one that answers "isn't that identically 1?":

      * exactly 0.0 everywhere when theta IS theta_ref (so training starts with every
        pair at weight sigma(0) = 0.5 -- the ratio does NOT start large);
      * it equals the BT MLE up to a per-sequence-independent constant -- AFTER
        centering, and NOT before;
      * `sum_y pi_ref(y) exp(r_hat(y)/beta) == 1` identically, because that sum IS
        `sum_y pi_theta(y)`. Nothing was assumed. It is a CONSTRAINT: r_hat is confined
        to a codimension-1 surface, so lowering r_hat(y_l) forces r_hat up elsewhere.

    Work in log space. Do not exponentiate and divide.
    """
    seqs = enumerate_sequences(env)
    all_sequence_logprob_theta = all_sequence_logprobs(env, theta, seqs)
    all_sequence_logprob_theta_ref = all_sequence_logprobs(env, theta_ref, seqs)    
    return beta * (all_sequence_logprob_theta - all_sequence_logprob_theta_ref)




def train_dpo(env: TokenEnv, theta0: np.ndarray, pairs: np.ndarray,
              theta_ref: np.ndarray, beta: float, lr: float = 5.0,
              n_steps: int = 2000, record: tuple[int, ...] = ()) -> tuple:
    """Full-batch gradient descent on the DPO loss, with an exact instrument panel.

    Returns `(theta_final, pi_final, log, history)` where

      * `pi_final` is the sequence pmf under theta_final (aligned with
        `enumerate_sequences`), which is what stage `equivalence` compares;
      * `log` is a dict of 1-D arrays, one entry per step, each measured at the theta
        BEFORE that step's update:
            "loss"    the DPO loss
            "logp_w"  mean log pi_theta(y_w) over pairs      <- NOT the margin
            "logp_l"  mean log pi_theta(y_l) over pairs      <- NOT the margin
            "margin"  mean implicit margin M
            "acc"     fraction of pairs with M > 0
            "kl"      KL(pi_theta || pi_ref) computed EXACTLY by enumeration
      * `history[t]` is a COPY of theta after t in `record` (1-indexed).

    Why the panel is specified this way, and it is the whole point of the day:
    `loss`, `margin` and `acc` can all look perfect while `logp_w`, `logp_l` and `kl`
    say the model is being destroyed. If you log only the first three -- which is what
    a default training script does -- stage `divergence` is invisible.

    `kl` must come from `exact_kl` (enumeration over all 15 sequences), not from a k1
    or k3 estimator on the pairs. The pairs are not samples from pi_theta.

    Record COPIES (`theta.copy()`). W3 D3's `history.append(tensor)`, fourth sighting.
    Plain GD, fixed lr, full batch: the implicit bias of plain GD on separable data is
    the phenomenon being measured, so an optimizer with its own regularization hides it.
    """
    theta = theta0.copy()
    log = {'loss': np.zeros(n_steps, dtype = np.float64), 'logp_w': np.zeros(n_steps, dtype = np.float64), \
        'logp_l': np.zeros(n_steps, dtype = np.float64), 'margin': np.zeros(n_steps, dtype = np.float64), \
        'acc': np.zeros(n_steps, dtype = np.float64), 'kl': np.zeros(n_steps, dtype = np.float64)}
    sequences = enumerate_sequences(env)
    all_seq_logprobs_ref = all_sequence_logprobs(env, theta_ref, sequences)
    history = {}

    for step in range(n_steps):
        l, grad = dpo_loss_and_grad(env, theta, pairs, theta_ref, beta)

        all_seq_logprobs = all_sequence_logprobs(env, theta, sequences)

        winner_seq_logprobs = all_seq_logprobs[pairs[:, 0]]
        loser_seq_logprobs = all_seq_logprobs[pairs[:, 1]]

        winner_seq_logprobs_ref = all_seq_logprobs_ref[pairs[:, 0]]
        loser_seq_logprobs_ref = all_seq_logprobs_ref[pairs[:, 1]]

        margin = beta * (winner_seq_logprobs - winner_seq_logprobs_ref - loser_seq_logprobs + loser_seq_logprobs_ref) 

        log['loss'][step] = l
        log['logp_w'][step] = winner_seq_logprobs.mean()
        log['logp_l'][step] = loser_seq_logprobs.mean()
        log['margin'][step] = margin.mean()
        log['acc'][step] = (np.where(margin>0, 1, 0)).mean()
        log['kl'][step] = exact_kl(env, theta, theta_ref, sequences)

        theta -= lr * grad

        if (step+1) in record:
            history[step+1] = theta.copy()
    
    theta_final = theta.copy()
    pi_final = np.exp(all_sequence_logprobs(env, theta_final, sequences))
    return (theta_final, pi_final, log, history)


def bt_then_tilt(env: TokenEnv, pairs: np.ndarray, theta_ref: np.ndarray, beta: float,
                 n_steps: int = 60_000, lr: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """The PPO-side reference path, collapsed to two lines. Returns `(pi_star, r_bt)`.

      1. Fit a tabular Bradley-Terry reward over SEQUENCES with D1's `fit_bt`
         (the sequences are the items; `n_items = len(enumerate_sequences(env))`).
      2. Tilt pi_ref by it exactly, with D1's `tilt`: pi* prop pi_ref * exp(r_bt/beta).

    That is "fit a reward model, then run KL-regularized RL to convergence" with every
    source of optimization noise deleted -- no rollouts, no clipping, no critic. It is
    the IDEAL that InstructGPT approximates, and stage `equivalence` asserts DPO lands
    on it elementwise.

    Note both provided functions are DIRECTLY reusable: `fit_bt(pairs, n_items,
    n_steps, lr)` returns `(r, nll, history)` and re-centers every step; `tilt(r,
    pi_ref, beta)` returns a pmf. You need pi_ref as a pmf over sequences, which is
    `exp(all_sequence_logprobs(env, theta_ref))`.

    This function must share NO code path with `train_dpo`. That independence is what
    makes the agreement evidence rather than a tautology (W3 D4's standing rule: never
    let the slow reference be able to agree with the fast one by construction).
    """
    sequences = enumerate_sequences(env)
    r_final, nll,  _ = fit_bt(pairs, len(sequences), n_steps, lr)
    pi_ref = np.exp(all_sequence_logprobs(env, theta_ref, sequences))
    pi = tilt(r_final, pi_ref, beta)
    return pi, r_final



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


def _noisy_labels(rng, gold, base, eps):
    p = sigmoid(gold[base[:, 0]] - gold[base[:, 1]])
    p = (1.0 - eps) * p + eps * (1.0 - p)
    win_first = rng.random(len(base)) < p
    return np.where(win_first[:, None], base, base[:, ::-1])


def _strongly_connected(pairs, n):
    adj = np.zeros((n, n), dtype=bool)
    adj[pairs[:, 0], pairs[:, 1]] = True
    np.fill_diagonal(adj, True)
    for _ in range(int(np.ceil(np.log2(n))) + 1):
        adj = adj @ adj
    return bool(adj.all())


class Ctx:
    """Fixtures. ONE environment, ONE reference policy, ONE gold reward -- and four
    comparison graphs over the identical 15 sequences. As on D1, the only thing that
    ever changes between fixtures is which comparisons exist and how they are labelled."""

    BETA = 0.5

    def __init__(self, fast: bool = False):
        self.fast = fast
        self.dpo_steps = 2000      # not shortened either: --fast must not weaken the
                                   # headline equivalence (800 steps only reaches 1e-8)
        self.long_steps = 4_000 if fast else 20_000
        self.match_steps = 2_000 if fast else 6_000
        self.bt_steps = 60_000     # cheap (15 tabular items); never shortened -- the
                                   # equivalence check is the point of the file

    @cached_property
    def env(self):
        return TokenEnv(vocab=3, t_max=3)

    @cached_property
    def seqs(self):
        return enumerate_sequences(self.env)

    @cached_property
    def n(self):
        return len(self.seqs)

    @cached_property
    def theta_ref(self):
        return self.env.random_theta(np.random.default_rng(0), 0.7)

    @cached_property
    def gold(self):
        rng = np.random.default_rng(0)
        rng.standard_normal((self.env.n_states, self.env.vocab))   # burn theta_ref's draw
        return rng.standard_normal(self.n) * 1.5

    @cached_property
    def pi_ref(self):
        return np.exp(all_sequence_logprobs(self.env, self.theta_ref, self.seqs))

    @cached_property
    def all_base(self):
        return np.array([(i, j) for i in range(self.n) for j in range(i + 1, self.n)])

    @cached_property
    def connected(self):
        """All 105 comparisons, labelled at eps=0.15 -> strongly connected -> the BT
        MLE EXISTS, so this is the fixture on which the equivalence is a statement
        about two finite points."""
        return _noisy_labels(np.random.default_rng(7), self.gold, self.all_base, 0.15)

    @cached_property
    def matching(self):
        """Each sequence appears at most once: the shape of a real DPO dataset, and
        D1's divergent case."""
        order = np.argsort(-self.gold)
        out = []
        for k in range(0, 14, 2):
            a, b = int(order[k]), int(order[k + 1])
            out.append((a, b) if self.gold[a] > self.gold[b] else (b, a))
        return np.array(out)

    @cached_property
    def _rank(self):
        return np.argsort(-self.pi_ref)

    @cached_property
    def bulk(self):
        """4 pairs whose WINNERS are pi_ref's four favourite sequences: the data looks
        like something pi_ref would have generated (mu ~ pi_ref)."""
        r = self._rank
        return np.array([(int(r[i]), int(r[self.n - 1 - i])) for i in range(4)])

    @cached_property
    def tail(self):
        """The SAME eight sequences, columns swapped: the winners are now pi_ref's four
        LEAST likely sequences -- the data came from a stronger model (mu != pi_ref).
        One line of the fixture changes."""
        return self.bulk[:, ::-1].copy()

    @cached_property
    def fit_connected(self):
        print("  ... running DPO on the strongly-connected fixture")
        return train_dpo(self.env, self.theta_ref.copy(), self.connected, self.theta_ref,
                         self.BETA, lr=5.0, n_steps=self.dpo_steps)

    @cached_property
    def fit_bt_tilt(self):
        print("  ... fitting BT over sequences and tilting exactly (the PPO path)")
        return bt_then_tilt(self.env, self.connected, self.theta_ref, self.BETA,
                            n_steps=self.bt_steps, lr=2.0)

    @cached_property
    def fit_matching(self):
        print("  ... running DPO on the perfect matching (the divergent case)")
        return train_dpo(self.env, self.theta_ref.copy(), self.matching, self.theta_ref,
                         self.BETA, lr=5.0, n_steps=self.long_steps)


def _fail_fast(ctx: Ctx) -> None:
    """Touch the cheap TODOs a heavy stage needs, so a missing one surfaces before a
    60k-step fit is paid for. Touch ONLY what this stage needs -- a guard that emits a
    FALSE failure is worse than no guard (W4 D3)."""
    dpo_loss_and_grad(ctx.env, ctx.theta_ref, ctx.matching[:1], ctx.theta_ref, ctx.BETA)


# ---- individual stages ----------------------------------------------------


def stage_grad(ctx: Ctx) -> None:
    """dpo_loss_and_grad -- finite differences, log 2, and pi_ref's true role"""
    env, beta = ctx.env, ctx.BETA
    theta = env.random_theta(np.random.default_rng(3), 0.5)
    loss, g = dpo_loss_and_grad(env, theta, ctx.connected, ctx.theta_ref, beta)

    _check("gradient has the shape of theta", g.shape == theta.shape,
           f"{g.shape} vs {theta.shape}")

    fd = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            e = np.zeros_like(theta)
            e[i, j] = 1e-6
            fd[i, j] = (dpo_loss_and_grad(env, theta + e, ctx.connected, ctx.theta_ref, beta)[0]
                        - dpo_loss_and_grad(env, theta - e, ctx.connected, ctx.theta_ref, beta)[0]) / 2e-6
    err = float(np.abs(g - fd).max())
    _check("analytic gradient == central finite differences", err < 1e-8,
           f"max|d| = {err:.3e}")

    l0 = dpo_loss_and_grad(env, ctx.theta_ref, ctx.connected, ctx.theta_ref, beta)[0]
    _check("theta == theta_ref gives EXACTLY log 2 (every margin is 0)",
           abs(l0 - np.log(2)) < 1e-15, f"{l0:.15f} vs {np.log(2):.15f}")

    rows = float(np.abs(g.sum(axis=1)).max())
    _check("every gradient ROW sums to zero (softmax conserves each state's mass)",
           rows < 1e-12, f"max|row sum| = {rows:.3e}")

    # pi_ref enters ONLY through the scalar weight -- the day's sharpest structural claim
    pair = ctx.matching[:1]
    ref_a = ctx.theta_ref
    ref_b = env.random_theta(np.random.default_rng(11), 0.9)
    _, ga = dpo_loss_and_grad(env, theta, pair, ref_a, beta)
    _, gb = dpo_loss_and_grad(env, theta, pair, ref_b, beta)
    wa = float(sigmoid(-beta * ((sequence_logprob(env, theta, ctx.seqs[pair[0, 0]])
                                 - sequence_logprob(env, ref_a, ctx.seqs[pair[0, 0]]))
                                - (sequence_logprob(env, theta, ctx.seqs[pair[0, 1]])
                                   - sequence_logprob(env, ref_a, ctx.seqs[pair[0, 1]])))))
    wb = float(sigmoid(-beta * ((sequence_logprob(env, theta, ctx.seqs[pair[0, 0]])
                                 - sequence_logprob(env, ref_b, ctx.seqs[pair[0, 0]]))
                                - (sequence_logprob(env, theta, ctx.seqs[pair[0, 1]])
                                   - sequence_logprob(env, ref_b, ctx.seqs[pair[0, 1]])))))
    _check("the two references really do give different weights (non-vacuous)",
           abs(wa - wb) > 1e-3, f"sigma(-M) = {wa:.6f} vs {wb:.6f}")
    d = float(np.abs(ga / wa - gb / wb).max())
    _check("pi_ref changes the gradient's SCALE but not its DIRECTION",
           d < 1e-12, f"max|g_a/w_a - g_b/w_b| = {d:.3e}")

    # the shared prefix of a pair contributes exactly nothing in a tabular policy
    shared = [s for s in ctx.seqs if len(s) == 3 and s[0] == 0]
    ya, yb = shared[0], shared[1]
    ga2 = grad_sequence_logprob(env, theta, ya) - grad_sequence_logprob(env, theta, yb)
    root = float(np.abs(ga2[env.state_id(())]).max())
    _check("two sequences sharing a prefix cancel EXACTLY on the shared states",
           root < 1e-15, f"|d| at the root state = {root:.3e}")
    print("  ... => in a TABLE, 'pushing y_l down drags y_w down' cannot happen through a\n"
          "      shared prefix. That failure mode needs shared PARAMETERS, not shared tokens.")


def stage_implicit(ctx: Ctx) -> None:
    """implicit_reward -- zero at the reference, and the gauge is fixed by the simplex"""
    env, beta = ctx.env, ctx.BETA
    r0 = implicit_reward(env, ctx.theta_ref, ctx.theta_ref, beta)
    _check("shape is one reward per enumerable sequence", r0.shape == (ctx.n,), f"{r0.shape}")
    _check("EXACTLY zero when theta is theta_ref", float(np.abs(r0).max()) == 0.0,
           f"max|r_hat| = {float(np.abs(r0).max()):.3e}")
    print("  ... => training starts with every pair at weight sigma(0) = 0.5. The log-ratio\n"
          "      does NOT start large -- what is large is the ratio the LABELS require.")

    theta = env.random_theta(np.random.default_rng(5), 0.8)
    rhat = implicit_reward(env, theta, ctx.theta_ref, beta)
    gauge = float(np.sum(ctx.pi_ref * np.exp(rhat / beta)))
    _check("sum_y pi_ref(y) exp(r_hat(y)/beta) == 1 -- the gauge is FIXED by the simplex",
           abs(gauge - 1.0) < 1e-12, f"{gauge:.15f}")

    # r_hat is the Z==1 representative: build a policy from a KNOWN r, recover r - beta*log Z
    r = np.random.default_rng(6).standard_normal(ctx.n) * 1.2
    logp = np.log(ctx.pi_ref) + r / beta
    logZ = float(np.log(np.sum(np.exp(logp))))
    pi_star = np.exp(logp - logZ)
    # invert directly: r_hat = beta*log(pi_star/pi_ref) must equal r - beta*logZ
    rhat_star = beta * (np.log(pi_star) - np.log(ctx.pi_ref))
    off = float(np.abs(rhat_star - (r - beta * logZ)).max())
    _check("r_hat == r - beta*log Z(x): the parameterization CHOOSES the gauge",
           off < 1e-12, f"max|d| = {off:.3e}")
    _check("...and it is NOT r itself", float(np.abs(rhat_star - r).max()) > 1e-2,
           f"offset = beta*log Z = {beta * logZ:+.6f}")


def stage_equivalence(ctx: Ctx) -> None:
    """THE HEADLINE -- DPO's optimum == tilt of the BT MLE, elementwise"""
    _fail_fast(ctx)
    _check("the fixture's comparison graph IS strongly connected (so the BT MLE exists)",
           _strongly_connected(ctx.connected, ctx.n), f"{len(ctx.connected)} comparisons")

    _, pi_dpo, log, _ = ctx.fit_connected
    pi_bt, r_bt = ctx.fit_bt_tilt

    _check("DPO's distribution is a pmf", abs(float(pi_dpo.sum()) - 1.0) < 1e-12)
    err = float(np.abs(pi_dpo - pi_bt).max())
    _check("train_dpo == bt_then_tilt ELEMENTWISE", err < 1e-12,
           f"max|d| = {err:.3e} over {ctx.n} sequences")

    worst = 0.0
    for b2 in (0.25, 1.0, 2.0):
        pw, _ = bt_then_tilt(ctx.env, ctx.connected, ctx.theta_ref, b2,
                             n_steps=ctx.bt_steps, lr=2.0)
        d = float(np.abs(pi_dpo - pw).max())
        worst = max(worst, d)
        _check(f"...and tilting at the WRONG beta={b2} does NOT agree", d > 1e-3,
               f"max|d| = {d:.3e}")
    print(f"  ... => the check is non-vacuous by a factor of {worst / max(err, 1e-300):.2e}")

    rhat = implicit_reward(ctx.env, ctx.fit_connected[0], ctx.theta_ref, ctx.BETA)
    raw = float(np.abs(rhat - r_bt).max())
    cen = float(np.abs((rhat - rhat.mean()) - (r_bt - r_bt.mean())).max())
    _check("the implicit reward recovers the BT MLE AFTER centering", cen < 1e-10,
           f"max|d| = {cen:.3e}")
    _check("...and NOT before centering", raw > 1e-2,
           f"max|d| = {raw:.4f}, offset = {float(rhat.mean() - r_bt.mean()):+.6f}")
    print("  ... => two algorithms sharing no code path land on the same distribution.\n"
          "      'DPO is equivalent to RLHF' is now a number, not a citation.")


def stage_divergence(ctx: Ctx) -> None:
    """train_dpo on a perfect MATCHING -- the metrics look perfect while it diverges"""
    _fail_fast(ctx)
    _check("a DPO-shaped dataset is NOT strongly connected (D1's theorem)",
           not _strongly_connected(ctx.matching, ctx.n), f"{len(ctx.matching)} pairs")

    _, pi, log, _ = ctx.fit_matching
    acc, kl, lw, ll = log["acc"], log["kl"], log["logp_w"], log["logp_l"]
    first = int(np.argmax(acc >= 1.0)) + 1
    ten = min(10 * first, len(acc))

    _check("log pi(y_l) decreases MONOTONICALLY", bool(np.all(np.diff(ll) < 0)),
           f"{ll[0]:+.4f} -> {ll[-1]:+.4f}")
    _check("the loss goes to zero", log["loss"][-1] < 1e-3, f"{log['loss'][-1]:.3e}")
    _check("pairwise accuracy is 1.0 and stays there", acc[-1] == 1.0)
    _check("the exact KL rises monotonically", bool(np.all(np.diff(kl) > 0)),
           f"{kl[0]:.4f} -> {kl[-1]:.4f}")

    print(f"  ... accuracy first hits 1.0 at step {first}: KL = {kl[first - 1]:.4f}")
    print(f"  ... at 10x that many steps ({ten}):        KL = {kl[ten - 1]:.4f}")
    print(f"  ... at the end ({len(acc)}):                 KL = {kl[-1]:.4f}, "
          f"log pi(y_l) = {ll[-1]:+.4f}, loss = {log['loss'][-1]:.3e}")

    bound = float((-np.log(ctx.pi_ref)).max())
    _check("the KL stays under the point-mass bound max_y -log pi_ref(y)", kl[-1] < bound,
           f"KL {kl[-1]:.4f} < {bound:.4f}")
    losers = float(pi[ctx.matching[:, 1]].sum())
    _check("the losers' total probability is driven to ~0", losers < 1e-6,
           f"{float(ctx.pi_ref[ctx.matching[:, 1]].sum()):.6f} -> {losers:.3e}")
    print("  ... => 'the KL grows without bound' is FALSE in a finite space: pi_theta\n"
          "      collapsing to a point mass is the worst it can do. In a real LM that\n"
          "      bound is -log pi_ref of the single likeliest string, so 'unbounded' is\n"
          "      the practical statement, not the mathematical one.")


def stage_logprobs(ctx: Ctx) -> None:
    """the marginal likelihood is uncontrolled -- margin UP, both terms DOWN"""
    _, _, log, _ = ctx.fit_connected
    lw, ll, m = log["logp_w"], log["logp_l"], log["margin"]
    _check("the margin rises", m[-1] > m[0] + 0.1, f"{m[0]:+.4f} -> {m[-1]:+.4f}")
    _check("mean log pi(y_w) FALLS -- the CHOSEN completion loses probability",
           lw[-1] < lw[0], f"{lw[0]:+.4f} -> {lw[-1]:+.4f} ({lw[-1] - lw[0]:+.4f})")
    _check("mean log pi(y_l) falls further", ll[-1] - ll[0] < lw[-1] - lw[0],
           f"{ll[0]:+.4f} -> {ll[-1]:+.4f} ({ll[-1] - ll[0]:+.4f})")
    print("  ... => and this is the STRONGLY CONNECTED fixture, where the BT MLE exists\n"
          "      and DPO converges to a finite optimum. Both logprobs falling is not a\n"
          "      divergence artifact -- it is what the converged optimum looks like,\n"
          "      because no term in the gradient mentions the marginal likelihood.")


def stage_mismatch(ctx: Ctx) -> None:
    """mu != pi_ref -- the tax is paid in KL, and the loss never reports it"""
    out = {}
    for tag, prs in (("bulk", ctx.bulk), ("tail", ctx.tail)):
        _, _, log, _ = train_dpo(ctx.env, ctx.theta_ref.copy(), prs, ctx.theta_ref,
                                 ctx.BETA, lr=5.0, n_steps=ctx.match_steps)
        out[tag] = log
        print(f"  ... {tag}: mean log pi_ref(y_w) = {np.log(ctx.pi_ref[prs[:, 0]]).mean():+.4f}"
              f"  margin -> {log['margin'][-1]:+.4f}  loss -> {log['loss'][-1]:.3e}"
              f"  KL -> {log['kl'][-1]:.4f}")

    _check("the two fixtures use the SAME eight sequences (only the columns swap)",
           set(ctx.bulk.ravel().tolist()) == set(ctx.tail.ravel().tolist()))
    _check("both reach a comparable margin", abs(out['bulk']['margin'][-1]
                                                 - out['tail']['margin'][-1]) < 2.0,
           f"{out['bulk']['margin'][-1]:.4f} vs {out['tail']['margin'][-1]:.4f}")
    for tag in ("bulk", "tail"):
        dw = out[tag]['logp_w'][-1] - out[tag]['logp_w'][0]
        dl = out[tag]['logp_l'][-1] - out[tag]['logp_l'][0]
        _check(f"[{tag}] the margin is bought almost entirely by pushing y_l DOWN",
               abs(dw) < 0.15 * abs(dl),
               f"d logp_w = {dw:+.4f} vs d logp_l = {dl:+.4f} ({abs(dw) / abs(dl):.4f})")
    ratio = out['tail']['kl'][-1] / out['bulk']['kl'][-1]
    _check("but winners in pi_ref's TAIL cost far more KL for the same margin",
           ratio > 2.0, f"KL {out['tail']['kl'][-1]:.4f} vs {out['bulk']['kl'][-1]:.4f} "
                        f"= {ratio:.2f}x")
    print("  ... the SIGN of d logp_w is a transient and not the discriminator: on `tail`\n"
          "      it is positive early and only crosses zero around step 5000. Do not build\n"
          "      a diagnostic on it. The KL gap is present from the first step and stable.")
    print("  ... => the reference-mismatch tax is real and it is paid in KL, invisibly:\n"
          "      identical pairs, identical margins, comparable loss, and the run that\n"
          "      had to reach into pi_ref's tail spent several times the KL budget you\n"
          "      thought you were buying with beta. THIS is why you SFT on the chosen\n"
          "      completions first -- it moves pi_ref onto the data.")


def stage_weight(ctx: Ctx) -> None:
    """the per-pair gradient weight IS sigma(-M), the Bradley-Terry weight"""
    env, beta = ctx.env, ctx.BETA
    theta = env.random_theta(np.random.default_rng(13), 0.9)
    lp = all_sequence_logprobs(env, theta, ctx.seqs)
    lpr = all_sequence_logprobs(env, ctx.theta_ref, ctx.seqs)
    M = beta * ((lp[ctx.connected[:, 0]] - lpr[ctx.connected[:, 0]])
                - (lp[ctx.connected[:, 1]] - lpr[ctx.connected[:, 1]]))
    norms, ws, base = [], [], []
    for k in range(len(ctx.connected)):
        _, g = dpo_loss_and_grad(env, theta, ctx.connected[k:k + 1], ctx.theta_ref, beta)
        ds = (grad_sequence_logprob(env, theta, ctx.seqs[ctx.connected[k, 0]])
              - grad_sequence_logprob(env, theta, ctx.seqs[ctx.connected[k, 1]]))
        norms.append(float(np.linalg.norm(g)))
        ws.append(float(sigmoid(-M[k])))
        base.append(float(np.linalg.norm(ds)))
    norms, ws, base = np.array(norms), np.array(ws), np.array(base)
    # ||g_k|| = beta * sigma(-M_k) * ||score_w - score_l||: divide out BOTH factors and
    # every pair must return exactly beta. Dividing by sigma(-M) alone does NOT work --
    # the score-difference norm varies per pair, which is itself worth seeing.
    rel = float(np.abs(norms / (ws * base) - beta).max())
    _check("per-pair gradient is EXACTLY beta * sigma(-M) * (score_w - score_l)",
           rel < 1e-12, f"max|  ||g|| / (sigma(-M)*||ds||)  -  beta | = {rel:.3e}")
    spread = float(base.max() / base.min())
    _check("...and the check needed both factors: the score norm varies across pairs",
           spread > 1.5, f"||score_w - score_l|| spans {spread:.2f}x")

    o = np.argsort(M)
    d = max(1, len(o) // 10)
    hard, easy = float(ws[o[:d]].mean()), float(ws[o[-d:]].mean())
    _check("the best-ranked decile contributes far less than the worst", easy < hard / 5,
           f"weight {easy:.6f} (best) vs {hard:.6f} (worst) = {hard / easy:.1f}x")
    print("  ... => same sigma(-Delta) as W4 D1's Bradley-Terry gradient, same object.\n"
          "      DPO is a reward model whose parameterization happens to be a policy --\n"
          "      so a MISLABELLED pair has large negative M and weight -> 1, i.e. label\n"
          "      noise receives the MAXIMUM gradient weight in the batch.")


def stage_invariance(ctx: Ctx) -> None:
    """regression -- shifting the gold reward by c(x) changes nothing, EXACTLY"""
    env, beta = ctx.env, ctx.BETA
    theta = env.random_theta(np.random.default_rng(17), 0.6)
    a = _noisy_labels(np.random.default_rng(7), ctx.gold, ctx.all_base, 0.15)
    b = _noisy_labels(np.random.default_rng(7), ctx.gold + 3.7, ctx.all_base, 0.15)
    _check("r -> r + c(x) produces the IDENTICAL labels", bool(np.array_equal(a, b)))
    la, ga = dpo_loss_and_grad(env, theta, a, ctx.theta_ref, beta)
    lb, gb = dpo_loss_and_grad(env, theta, b, ctx.theta_ref, beta)
    _check("...so the DPO loss changes by exactly 0.0", (lb - la) == 0.0, f"{lb - la:.3e}")
    _check("...and the gradient is bit-identical", bool(np.array_equal(ga, gb)))
    lc, _ = dpo_loss_and_grad(env, theta, a[:, ::-1].copy(), ctx.theta_ref, beta)
    _check("the check is non-vacuous: swapping winner/loser DOES change the loss",
           abs(lc - la) > 1e-3, f"{la:.6f} vs {lc:.6f}")


STAGES = [
    ("grad", stage_grad),
    ("implicit", stage_implicit),
    ("equivalence", stage_equivalence),
    ("divergence", stage_divergence),
    ("logprobs", stage_logprobs),
    ("mismatch", stage_mismatch),
    ("weight", stage_weight),
    ("invariance", stage_invariance),
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
