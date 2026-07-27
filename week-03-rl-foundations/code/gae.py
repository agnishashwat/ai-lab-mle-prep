"""
Week 3 Day 4 -- GAE, bootstrapping, and the bias/variance dial.

Day 3 built the machinery that makes today measurable: `batch_gradient_statistics`
takes a `weight_fn` closure, so several estimators can be compared on IDENTICAL
rollouts, and `exact_gradient` (central finite differences on T Bellman backups)
is a ground-truth gradient. GAE is just two more weight functions -- which means
"how biased is lambda=0?" stops being a claim and becomes a printed number.

The four claims the harness pins down, in order:

  (1) TELESCOPING. sum_{l} gamma^l delta_{t+l} = G_t - V_t(s_t) exactly, for ANY
      value table, provided V_T = 0. So GAE(lambda=1) is not "approximately
      Monte-Carlo" -- it IS the reward-to-go-minus-baseline estimator of Day 3,
      elementwise. Two regression tests chain back through Day 3 to Day 2.

  (2) lambda=1 IS UNBIASED FOR ANY CRITIC, however wrong -- it is a baseline, and
      baselines only need h_t-measurability. lambda<1 BOOTSTRAPS, and is unbiased
      only if the critic is right. With the EXACT finite-horizon V_t, every lambda
      is unbiased: the l>=1 terms are zero-mean noise, because E[A_{t+l}|s_{t+l}]=0
      and grad log pi_t is h_{t+l}-measurable. That is the whole bias story in one
      sentence: **the bias in GAE is the critic's error, nothing else.**

  (3) Therefore variance RISES with lambda (more sampled reward, less bootstrap)
      while bias FALLS with lambda. With a perfect critic lambda=0 strictly wins;
      lambda exists only to hedge against a critic you do not trust. The sweep
      prints bias, variance and single-sample MSE for both a correct and a
      mismatched critic.

  (4) The finite-horizon fix Day 3 left open: a TIME-INDEXED V_t(s) is the value
      function this objective actually has. Day 3's infinite-horizon V^pi was
      unbiased but INCREASED variance 27%; time-indexing turns that cautionary
      result into a constructive one and should beat the pooled b(s) too.

Implement the bodies marked TODO. The harness is STAGED: each stage runs
independently, and a stage whose function is still a TODO prints [TODO] and is
skipped rather than killing the run.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/gae.py
  one stage           ...  gae.py scan
  a few stages        ...  gae.py td kstep scan
  small/fast batches  ...  gae.py --fast
  list the stages     ...  gae.py --list
"""

from __future__ import annotations

import sys
import traceback
from functools import cached_property

import numpy as np
import torch

from mdp import MDP, bellman_expectation_backup, two_state_chain
from reinforce import (
    discount_cumsum,
    episode_gradient,
    exact_J,
    exact_gradient,
    policy_probs,
    sample_episode,
)
from actor_critic import (
    Batch,
    batch_gradient_statistics,
    fit_state_baseline,
    oracle_state_baseline,
    weights_with_baseline,
)


# ---------------------------------------------------------------------------
# Value-table convention (provided -- read this before you start)
# ---------------------------------------------------------------------------
#
# Every function below takes `values` as a TIME-INDEXED table of shape (T+1, S):
#
#     values[t, s]  ~  V_t(s) = E[ sum_{k=t}^{T-1} gamma^{k-t} r_k | s_t = s ]
#
# with the hard convention  values[T, :] = 0.
#
# Two reasons this shape, and not Day 3's pooled (S,):
#   * The objective is the FINITE-horizon J_T, whose value function genuinely
#     depends on how many steps are left. Day 3's V^pi was the infinite-horizon
#     one -- unbiased, but mis-centred, which is why it made variance worse.
#   * V_T = 0 is what makes the telescoping identity exact, which is what makes
#     lambda=1 reproduce Day 3 to the last digit instead of approximately.
# `Batch` never records s_T, and with V_T = 0 it never has to.


def pooled_to_time_indexed(b: np.ndarray, T: int) -> np.ndarray:
    """Lift a Day-3 pooled baseline b(s), shape (S,), to a (T+1, S) table.

    Rows 0..T-1 are all b; row T is zeros. Provided so Day 3's baselines can be
    dropped straight into today's estimators for comparison.
    """
    V = np.zeros((T + 1, len(b)), dtype=np.float64)
    V[:T] = np.asarray(b, dtype=np.float64)
    return V


# ---------------------------------------------------------------------------
# TD errors and k-step advantages
# ---------------------------------------------------------------------------

def td_errors(
    states: np.ndarray, rewards: np.ndarray, values: np.ndarray, gamma: float
) -> np.ndarray:
    """One-step TD errors for a single episode. Returns shape (T,).

        delta_t = r_t + gamma * V_{t+1}(s_{t+1}) - V_t(s_t)

    states, rewards : (T,)          values : (T+1, S) with values[T, :] = 0

    The last step is the one to get right: s_T was never recorded, and does not
    need to be, because V_T(.) = 0 -- so delta_{T-1} = r_{T-1} - V_{T-1}(s_{T-1}).

    Vectorize it. `values[np.arange(T), states]` is the "two aligned index
    arrays -> one element each" idiom from Day 3; the next-state values are the
    same lookup shifted by one, with a 0.0 appended for the terminal step.
    """
    T=states.shape[0]
    val_shift = np.concatenate((values[np.arange(1,T), states[1:]], np.array([0])))
    val = values[np.arange(T), states]
    return rewards + gamma * val_shift - val


def k_step_advantage(deltas: np.ndarray, gamma: float, k: int) -> np.ndarray:
    """The k-step advantage estimator, shape (T,).

        A_t^(k) = sum_{l=0}^{k-1} gamma^l delta_{t+l}

    truncated at the end of the episode (terms with t+l >= T contribute 0, since
    the horizon is over and every later delta is defined to be zero).

    This is the rung ladder GAE averages over: k=1 is the pure one-step
    bootstrap, and k >= T-t is the full Monte-Carlo advantage G_t - V_t(s_t) by
    telescoping. A plain double loop is fine here -- this function exists to be
    the honest, obviously-correct reference that the fast scan is checked
    against, not to be fast.
    """
    T = deltas.shape[0]
    delta_new = np.concatenate((deltas, np.zeros(k)))
    
    A_k = np.zeros(T)

    for t in range(T):
        A_k[t] = (gamma**np.arange(k) * delta_new[t:t+k]).sum()
    return A_k


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def gae_naive(deltas: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """GAE by explicit O(T^2) double loop, shape (T,).

        A_t^{GAE(gamma, lam)} = sum_{l=0}^{T-1-t} (gamma * lam)^l delta_{t+l}

    Deliberately the dumbest possible implementation. It is the reference the
    reverse scan must match; writing the definition twice, once transparently and
    once efficiently, is how you find out whether the efficient one is right.
    """
    T = deltas.shape[0]
    A_gae = np.zeros(T)

    for t in range(T):
        for l in range(T-t):
            A_gae[t] += (gamma*lam)**l * deltas[t+l] if t+l < T else 0
    return A_gae


def gae_reverse(
    deltas: np.ndarray, gamma: float, lam: float, dones: np.ndarray | None = None
) -> np.ndarray:
    """GAE by the O(T) reverse scan, shape (T,).

        A_t = delta_t + gamma * lam * A_{t+1},      A_T = 0

    Same structure as `discount_cumsum` from Day 2, with gamma replaced by
    gamma*lam. This recursion (not the sum) is what every PPO implementation
    actually runs, and it is the reason GAE is free relative to computing the
    deltas.

    `dones`, if given, is a (T,) 0/1 array where dones[t] == 1 means the episode
    ENDED after step t. Credit must not flow across an episode boundary, so the
    recursion resets there:

        A_t = delta_t + gamma * lam * (1 - dones[t]) * A_{t+1}

    Getting this wrong is one of the classic silent RL bugs -- it leaks the next
    episode's advantage backwards into this one, and nothing crashes.
    """
    T = deltas.shape[0]
    A = np.zeros(T+1)
    if dones is None:
        for t in range(T-1, -1, -1):
            A[t] = deltas[t] + gamma * lam * A[t+1]
    else:
        for t in range(T-1, -1, -1):
            A[t] = deltas[t] + gamma * lam * A[t+1] * (1-dones[t])
    return A[:T]


# ---------------------------------------------------------------------------
# Critics
# ---------------------------------------------------------------------------

def exact_time_indexed_values(mdp: MDP, theta: torch.Tensor, T: int) -> np.ndarray:
    """The EXACT finite-horizon value table V_t(s), shape (T+1, S).

    This is the value function the objective J_T actually has, and it is the
    thing Day 3 was missing:

        V_T = 0;        V_t = T^pi V_{t+1}       for t = T-1, ..., 0

    i.e. T-t Bellman expectation backups, so row t is the exact expected
    discounted return of the T-t steps that remain. `bellman_expectation_backup`
    from Day 1 does one backup; this is a backward loop over rows.

    Sanity anchor you get for free: V_0(s0) must equal `exact_J(mdp, theta, T, s0)`,
    since exact_J is the same recursion read at one entry.
    """
    V = np.zeros((T+1, mdp.S))
    pi = torch.softmax(theta, dim=-1).detach().numpy()

    for t in range(T-1, -1, -1):
        V[t, :] = bellman_expectation_backup(mdp, pi, V[t+1, :])

    return V


def fit_time_indexed_values(batch: Batch, n_states: int) -> np.ndarray:
    """Monte-Carlo critic for V_t(s), shape (T+1, S), from observed reward-to-go.

        values[t, s] = mean of G_t over the episodes i with states[i, t] == s
        values[T, :] = 0

    This is Day 3's `fit_state_baseline` with the pooling over t REMOVED: instead
    of one bucket per state, one bucket per (t, s). Unvisited buckets get exactly
    0.0 -- no nan (a baseline may be wrong without being biased, so 0 is safe).

    A per-t np.bincount is fine and clear; the T-step loop is over 8 items, not
    over data. Watch the empty-bucket guard -- it bit you on Day 3.
    """

    T = batch.T
    V = np.zeros((T+1, n_states))

    for t in range(T):
        reward_to_go = np.bincount(batch.states[:, t], batch.returns_to_go[:, t], n_states)
        count = np.bincount(batch.states[:, t], minlength=n_states)
        count[count==0] = 1
        V[t, :] = reward_to_go / count
    return V

    


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def weights_gae(
    batch: Batch, i: int, gamma: float, values: np.ndarray, lam: float
) -> np.ndarray:
    """Psi_t for episode i under GAE(gamma, lam), shape (T,):

        Psi_t = gamma^t * A_t^{GAE(gamma, lam)}

    Compose `td_errors` then `gae_reverse`. Keep the gamma^t for exactly the
    Day-2/Day-3 reason: it is what makes the estimator unbiased for grad J_T, so
    the ground-truth comparison stays tight enough to measure bootstrap bias
    against. (PPO drops it. PPO also has no ground truth to check.)

    At lam=1 this must reproduce Day 3's `weights_with_baseline` EXACTLY -- the
    harness asserts it elementwise, which is the strongest kind of regression
    test: a new, more general estimator that provably contains the old one.
    """
    ep = i
    deltas = td_errors(batch.states[ep], batch.rewards[ep], values, gamma)
    A = gae_reverse(deltas, gamma, lam)
    return gamma ** np.arange(batch.T) * A

# ---------------------------------------------------------------------------
# Training (STRETCH)
# ---------------------------------------------------------------------------

def train_gae(
    mdp: MDP,
    T: int,
    s0: int,
    n_iters: int = 200,
    batch_size: int = 32,
    lr: float = 0.1,
    lam: float = 0.95,
    critic: str = "fit",           # "fit" | "exact"
    seed: int = 0,
) -> tuple[torch.Tensor, list[float]]:
    """Actor-critic with GAE. Returns (theta, history_of_mean_undiscounted_return).

    Per iteration, mirroring Day 3's `train`:
      * critic == "exact" -> V = exact_time_indexed_values for the CURRENT theta
      * critic == "fit"   -> roll a SEPARATE batch of `batch_size` episodes and
                             fit V on it, THEN roll the gradient batch (keeping
                             the critic independent of the episodes it weights)
      * ascent step on the mean gradient from `batch_gradient_statistics` with
        `weights_gae`.

    Same loop invariants as Week 1: seed first, resample every iteration
    (on-policy), update under no_grad. And the Day-3 trap: `theta += numpy_array`
    is NOT in-place -- wrap with torch.as_tensor.
    """
    rng = np.random.default_rng(seed)
    theta = torch.zeros(mdp.S, mdp.A, dtype=torch.float64, requires_grad=True)
    total_return = []

    for iter in range(n_iters):
        if critic == 'exact':
            V = exact_time_indexed_values(mdp, theta, T)
        elif critic == 'fit':
            batch_V = Batch(mdp, theta, T, s0, batch_size, rng)
            V = fit_time_indexed_values(batch_V, mdp.S)
        else:
            raise ValueError(critic)
        batch = Batch(mdp, theta, T, s0, batch_size, rng)
        mean_grad = batch_gradient_statistics(theta, batch, mdp.gamma, lambda batch, iter, gamma: weights_gae(batch, iter, gamma, V, lam))[0]
        total_return.append(batch.total_returns.mean())
        with torch.no_grad():
            theta += lr * torch.as_tensor(mean_grad, dtype=torch.float64)
        
    return (theta, total_return)
            



# ---------------------------------------------------------------------------
# Self-check harness (staged -- see the module docstring for usage)
# ---------------------------------------------------------------------------

_TALLY = {"pass": 0, "fail": 0}


def _check(name: str, cond: bool, detail: str = "") -> None:
    cond = bool(cond)
    _TALLY["pass" if cond else "fail"] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(name)


class Ctx:
    """Shared, lazily-built fixtures.

    The rollout batches use the SAME seeds and sizes as Day 3, so the variance
    numbers are directly comparable and Day 3's results must reproduce exactly.
    """

    def __init__(self, fast: bool = False):
        torch.manual_seed(0)
        self.fast = fast
        self.mdp = two_state_chain(gamma=0.9)
        self.T, self.s0 = 8, 0
        self.gamma = self.mdp.gamma
        self.theta = torch.tensor(
            [[0.4, -0.2], [-0.3, 0.5]], dtype=torch.float64, requires_grad=True
        )
        self.n_big = 4_000 if fast else 20_000
        self.n_fit = 1_000 if fast else 4_000
        self.n_iters = 60 if fast else 200
        self.lams = [0.0, 0.5, 0.9, 0.95, 1.0]
        self._stats: dict[str, tuple[np.ndarray, float]] = {}

    # -- ground truth and data ------------------------------------------------

    @cached_property
    def g_true(self) -> np.ndarray:
        return exact_gradient(self.mdp, self.theta, self.T, self.s0)

    @cached_property
    def big(self) -> Batch:
        print(f"  ... rolling out {self.n_big} episodes for gradient statistics")
        return Batch(self.mdp, self.theta, self.T, self.s0, self.n_big,
                     np.random.default_rng(0))

    @cached_property
    def fit_batch(self) -> Batch:
        """An INDEPENDENT batch, used only to fit critics/baselines."""
        return Batch(self.mdp, self.theta, self.T, self.s0, self.n_fit,
                     np.random.default_rng(1))

    # -- critics -------------------------------------------------------------

    @cached_property
    def V_exact(self) -> np.ndarray:
        return np.asarray(
            exact_time_indexed_values(self.mdp, self.theta, self.T), dtype=np.float64
        )

    @cached_property
    def V_fit(self) -> np.ndarray:
        return np.asarray(
            fit_time_indexed_values(self.fit_batch, self.mdp.S), dtype=np.float64
        )

    @cached_property
    def V_zero(self) -> np.ndarray:
        return np.zeros((self.T + 1, self.mdp.S), dtype=np.float64)

    @cached_property
    def b_pooled(self) -> np.ndarray:
        """Day 3's pooled MC baseline b(s)."""
        return np.asarray(fit_state_baseline(self.fit_batch, self.mdp.S), dtype=np.float64)

    @cached_property
    def V_pooled(self) -> np.ndarray:
        return pooled_to_time_indexed(self.b_pooled, self.T)

    @cached_property
    def V_mismatch(self) -> np.ndarray:
        """Day 3's INFINITE-horizon oracle V^pi, tiled over t.

        Wrong LEVEL, right RANKING: it answers "how much reward from here
        forever" instead of "how much in the T-t steps that remain", so it is
        ~1.46 too high in every row -- but by nearly the SAME amount in both
        states. Watch what that does (and does not) do to the bias.
        """
        return pooled_to_time_indexed(oracle_state_baseline(self.mdp, self.theta), self.T)

    @cached_property
    def V_skew(self) -> np.ndarray:
        """A critic with the right level but the WRONG RANKING of the states.

        The exact table, plus +1 on state 0 and -1 on state 1. Its average error
        is zero, so as a Day-3-style baseline (lambda=1) it is barely mis-centred
        -- but it gets the *relative* value of the two states backwards, which is
        precisely the error a bootstrap propagates.
        """
        V = self.V_exact.copy()
        V[:self.T] += np.array([1.0, -1.0])
        return V

    # -- measurement ---------------------------------------------------------

    def stats(self, key: str, weight_fn) -> tuple[np.ndarray, float]:
        if key not in self._stats:
            self._stats[key] = batch_gradient_statistics(
                self.theta, self.big, self.gamma, weight_fn
            )
        return self._stats[key]

    def gae_stats(self, tag: str, values: np.ndarray, lam: float):
        return self.stats(
            f"{tag}|lam={lam}",
            lambda B, i, gm: weights_gae(B, i, gm, values, lam),
        )

    def err(self, g: np.ndarray) -> float:
        """max|g - g_true| -- the bias measure used all week."""
        return float(np.max(np.abs(np.asarray(g) - self.g_true)))

    def l2(self, g: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(g) - self.g_true))


# ---- individual stages ----------------------------------------------------

def stage_td(ctx: Ctx) -> None:
    """td_errors"""
    # A hand-built episode with a hand-built value table: no sampling, no MDP.
    states = np.array([0, 1, 1], dtype=np.int64)
    rewards = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    V = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [0.0, 0.0]])
    d = np.asarray(td_errors(states, rewards, V, gamma=0.5), dtype=np.float64)
    _check("returns shape (T,)", d.shape == (3,), f"got {d.shape}")
    want = np.array([
        1.0 + 0.5 * 40.0 - 10.0,     # s0=0 -> s1=1 : r + g*V_1(1) - V_0(0)
        2.0 + 0.5 * 60.0 - 40.0,     # s1=1 -> s2=1 : r + g*V_2(1) - V_1(1)
        4.0 + 0.5 * 0.0 - 60.0,      # terminal     : V_3 = 0, s_3 never existed
    ])
    _check("delta_t matches hand computation", np.allclose(d, want),
           f"got {np.round(d, 4)} want {np.round(want, 4)}")

    # Zero critic => the TD error degenerates to the immediate reward.
    d0 = np.asarray(td_errors(states, rewards, np.zeros((4, 2)), 0.5), dtype=np.float64)
    _check("V == 0 makes delta_t == r_t", np.allclose(d0, rewards),
           f"got {np.round(d0, 4)}")

    # THE identity everything today rests on. It holds for ANY value table,
    # because V_T = 0 kills the surviving boundary term:
    #   sum_l gamma^l delta_{t+l} = G_t - V_t(s_t)
    b = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=3, rng=np.random.default_rng(11))
    Vr = np.random.default_rng(4).normal(size=(ctx.T + 1, ctx.mdp.S))
    Vr[ctx.T] = 0.0
    d = np.asarray(td_errors(b.states[0], b.rewards[0], Vr, ctx.gamma), dtype=np.float64)
    G = np.asarray(b.returns_to_go[0], dtype=np.float64)
    tele = np.array([
        float(np.sum(ctx.gamma ** np.arange(ctx.T - t) * d[t:])) for t in range(ctx.T)
    ])
    want = G - Vr[np.arange(ctx.T), b.states[0]]
    _check("TELESCOPING: sum_l gamma^l delta_{t+l} == G_t - V_t(s_t)",
           np.allclose(tele, want),
           f"max|diff| {np.max(np.abs(tele - want)):.2e} (holds for a RANDOM V)")


def stage_kstep(ctx: Ctx) -> None:
    """k_step_advantage"""
    d = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float64)
    g = 0.5
    a1 = np.asarray(k_step_advantage(d, g, k=1), dtype=np.float64)
    _check("returns shape (T,)", a1.shape == d.shape, f"got {a1.shape}")
    _check("k=1 is exactly the one-step bootstrap delta_t", np.allclose(a1, d),
           f"got {np.round(a1, 4)}")

    a2 = np.asarray(k_step_advantage(d, g, k=2), dtype=np.float64)
    want2 = np.array([1.0 + g * -2.0, -2.0 + g * 3.0, 3.0 + g * 0.5, 0.5])
    _check("k=2 matches hand computation (truncating at the horizon)",
           np.allclose(a2, want2), f"got {np.round(a2, 4)} want {np.round(want2, 4)}")

    # k >= T saturates: nothing left to add.
    aT = np.asarray(k_step_advantage(d, g, k=len(d)), dtype=np.float64)
    abig = np.asarray(k_step_advantage(d, g, k=len(d) + 5), dtype=np.float64)
    _check("k >= T-t saturates (k=T and k=T+5 agree)", np.allclose(aT, abig))
    # ... and equals the full Monte-Carlo advantage by telescoping.
    b = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=2, rng=np.random.default_rng(11))
    V = ctx.V_zero
    dd = np.asarray(td_errors(b.states[0], b.rewards[0], V, ctx.gamma), dtype=np.float64)
    full = np.asarray(k_step_advantage(dd, ctx.gamma, k=ctx.T), dtype=np.float64)
    _check("k=T with V=0 recovers the raw reward-to-go G_t",
           np.allclose(full, np.asarray(b.returns_to_go[0])),
           f"max|diff| {np.max(np.abs(full - np.asarray(b.returns_to_go[0]))):.2e}")


def stage_scan(ctx: Ctx) -> None:
    """gae_naive + gae_reverse (and the dones reset)"""
    rng = np.random.default_rng(2)
    d = rng.normal(size=9)
    for lam in [0.0, 0.3, 0.95, 1.0]:
        a_slow = np.asarray(gae_naive(d, 0.9, lam), dtype=np.float64)
        a_fast = np.asarray(gae_reverse(d, 0.9, lam), dtype=np.float64)
        _check(f"reverse scan == naive double loop (lam={lam})",
               a_slow.shape == d.shape and np.allclose(a_slow, a_fast),
               f"max|diff| {np.max(np.abs(a_slow - a_fast)):.2e}")

    _check("lam=0 collapses to delta_t (pure one-step bootstrap)",
           np.allclose(np.asarray(gae_reverse(d, 0.9, 0.0), dtype=np.float64), d))
    _check("lam=1 collapses to the k=T advantage (pure Monte-Carlo)",
           np.allclose(np.asarray(gae_reverse(d, 0.9, 1.0), dtype=np.float64),
                       np.asarray(k_step_advantage(d, 0.9, k=len(d)), dtype=np.float64)))
    # GAE is the (gamma*lam)-geometric average of the k-step rungs, so a constant
    # delta sequence gives the closed form directly -- a hand-checkable case that
    # does not just re-run the same recursion.
    ones = np.ones(6)
    a = np.asarray(gae_reverse(ones, 0.9, 0.5), dtype=np.float64)
    gl = 0.9 * 0.5
    want = np.array([sum(gl ** l for l in range(6 - t)) for t in range(6)])
    _check("delta==1 gives the geometric partial sums sum_l (gamma*lam)^l",
           np.allclose(a, want), f"got {np.round(a, 4)} want {np.round(want, 4)}")

    # dones: credit must NOT flow across an episode boundary. Two length-4
    # episodes concatenated must give the same answer as scanning them apart.
    d8 = rng.normal(size=8)
    dones = np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)
    joint = np.asarray(gae_reverse(d8, 0.9, 0.95, dones=dones), dtype=np.float64)
    apart = np.concatenate([
        np.asarray(gae_reverse(d8[:4], 0.9, 0.95), dtype=np.float64),
        np.asarray(gae_reverse(d8[4:], 0.9, 0.95), dtype=np.float64),
    ])
    _check("dones reset the scan (concatenated == scanned separately)",
           np.allclose(joint, apart), f"max|diff| {np.max(np.abs(joint - apart)):.2e}")
    _check("without dones the boundary LEAKS (so the test above has teeth)",
           not np.allclose(np.asarray(gae_reverse(d8, 0.9, 0.95), dtype=np.float64), apart))


def stage_exactvalues(ctx: Ctx) -> None:
    """exact_time_indexed_values"""
    V = ctx.V_exact
    _check("returns shape (T+1, S)", V.shape == (ctx.T + 1, ctx.mdp.S), f"got {V.shape}")
    _check("V_T is exactly zero (the horizon ends)", np.all(V[ctx.T] == 0.0))
    _check("V_0(s0) == exact_J (same recursion, one entry)",
           np.isclose(V[0, ctx.s0], exact_J(ctx.mdp, ctx.theta, ctx.T, ctx.s0)),
           f"V_0(s0)={V[0, ctx.s0]:.6f} vs exact_J={exact_J(ctx.mdp, ctx.theta, ctx.T, ctx.s0):.6f}")
    pi = policy_probs(ctx.theta).detach().numpy()
    _check("each row is one Bellman backup of the next",
           all(np.allclose(V[t], bellman_expectation_backup(ctx.mdp, pi, V[t + 1]))
               for t in range(ctx.T)))
    _check("V_t decreases with t (fewer steps left, rewards >= 0)",
           np.all(np.diff(V, axis=0) <= 1e-12))
    # And the contrast with Day 3: the infinite-horizon V^pi dominates every row.
    v_inf = oracle_state_baseline(ctx.mdp, ctx.theta)
    _check("infinite-horizon V^pi dominates every finite-horizon row",
           np.all(V <= v_inf[None, :] + 1e-9),
           f"V^pi_inf = {np.round(v_inf, 4)}   V_0 = {np.round(V[0], 4)}")
    print("\n  exact finite-horizon value table V_t(s):")
    for t in range(ctx.T + 1):
        print(f"    t={t}  {np.round(V[t], 4)}")


def stage_fitvalues(ctx: Ctx) -> None:
    """fit_time_indexed_values"""
    tiny = Batch(ctx.mdp, ctx.theta, T=3, s0=ctx.s0, n=6, rng=np.random.default_rng(7))
    V = np.asarray(fit_time_indexed_values(tiny, ctx.mdp.S), dtype=np.float64)
    _check("returns shape (T+1, S)", V.shape == (4, ctx.mdp.S), f"got {V.shape}")
    _check("V_T row is exactly zero", np.all(V[3] == 0.0))
    expected = np.zeros((4, ctx.mdp.S))
    for t in range(3):
        for s in range(ctx.mdp.S):
            m = tiny.states[:, t] == s
            expected[t, s] = tiny.returns_to_go[m, t].mean() if m.any() else 0.0
    _check("V[t,s] == mean of G_t over episodes with s_t == s", np.allclose(V, expected),
           f"max|diff| {np.max(np.abs(V - expected)):.2e}")
    _check("unvisited (t,s) buckets are exactly 0.0, not nan", np.isfinite(V).all())

    # Every episode starts at s0, so the (t=0, s != s0) bucket can NEVER be
    # visited. It must read exactly 0.0 -- and that is fine: a baseline is
    # allowed to be arbitrarily wrong without being biased. (This is also why the
    # comparison below has to be restricted to visited buckets: the unvisited
    # ones are 0 by design, not by error.)
    other = 1 - ctx.s0
    _check("the never-visitable (t=0, s != s0) bucket is exactly 0.0",
           ctx.V_fit[0, other] == 0.0, f"got {ctx.V_fit[0, other]}")

    # On VISITED buckets the MC critic must approach the exact table.
    visited = np.zeros((ctx.T + 1, ctx.mdp.S), dtype=bool)
    for t in range(ctx.T):
        for s in range(ctx.mdp.S):
            visited[t, s] = bool((ctx.fit_batch.states[:, t] == s).any())
    gap = float(np.max(np.abs((ctx.V_fit - ctx.V_exact)[visited])))
    _check("fitted critic approximates the exact table on visited buckets",
           gap < 0.5, f"max|V_fit - V_exact| {gap:.4f} over {visited.sum()} buckets"
                      f" (n_fit={ctx.n_fit})")


def _fail_fast(ctx: Ctx) -> None:
    """Touch every function the heavy stages need, on a 2-episode batch."""
    mini = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=2, rng=np.random.default_rng(99))
    exact_time_indexed_values(ctx.mdp, ctx.theta, ctx.T)
    fit_time_indexed_values(mini, ctx.mdp.S)
    td_errors(mini.states[0], mini.rewards[0], ctx.V_zero, ctx.gamma)
    gae_reverse(np.ones(ctx.T), ctx.gamma, 0.95)
    weights_gae(mini, 0, ctx.gamma, ctx.V_zero, 0.95)
    batch_gradient_statistics(
        ctx.theta, mini, ctx.gamma, lambda B, i, gm: weights_gae(B, i, gm, ctx.V_zero, 1.0)
    )


def stage_regression(ctx: Ctx) -> None:
    """lambda=1 must REPRODUCE Day 3 and Day 2 exactly, not approximately"""
    _fail_fast(ctx)
    b = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=4, rng=np.random.default_rng(11))
    for i in range(b.n):
        w_new = np.asarray(weights_gae(b, i, ctx.gamma, ctx.V_zero, 1.0), dtype=np.float64)
        w_old = np.asarray(weights_with_baseline(b, i, ctx.gamma, None), dtype=np.float64)
        _check(f"GAE(lam=1, V=0) == Day-2 reward_to_go weights [episode {i}]",
               np.allclose(w_new, w_old), f"max|diff| {np.max(np.abs(w_new - w_old)):.2e}")
    for i in range(b.n):
        w_new = np.asarray(weights_gae(b, i, ctx.gamma, ctx.V_pooled, 1.0), dtype=np.float64)
        w_old = np.asarray(weights_with_baseline(b, i, ctx.gamma, ctx.b_pooled),
                           dtype=np.float64)
        _check(f"GAE(lam=1, pooled b) == Day-3 weights_with_baseline [episode {i}]",
               np.allclose(w_new, w_old), f"max|diff| {np.max(np.abs(w_new - w_old)):.2e}")


def stage_unbiased(ctx: Ctx) -> None:
    """with an EXACT critic, every lambda is unbiased (the l>=1 terms are noise)"""
    _fail_fast(ctx)
    tol = 0.20 if ctx.fast else 0.10
    for lam in [0.0, 0.5, 0.95, 1.0]:
        g, _ = ctx.gae_stats("exact", ctx.V_exact, lam)
        _check(f"GAE(lam={lam}) with the exact V_t is unbiased", ctx.err(g) < tol,
               f"max|err| {ctx.err(g):.4f}")
    # lambda=1 is unbiased even with a WRONG critic, and it does not matter HOW
    # wrong: at lam=1 the critic appears only as -V(s_t), a state-only function,
    # and a baseline needs h_t-measurability, never correctness.
    for tag, V in (("mismatch", ctx.V_mismatch), ("skew", ctx.V_skew)):
        g, _ = ctx.gae_stats(tag, V, 1.0)
        _check(f"GAE(lam=1) is unbiased even with the {tag} critic",
               ctx.err(g) < tol, f"max|err| {ctx.err(g):.4f}")

    # ... and lambda=0 with a wrong-RANKING critic is not. The bug must be
    # detectably a bug, or the test has no teeth.
    g_ref, _ = ctx.gae_stats("exact", ctx.V_exact, 0.0)
    g_skew, _ = ctx.gae_stats("skew", ctx.V_skew, 0.0)
    _check("GAE(lam=0) with the wrong-RANKING critic IS biased",
           ctx.err(g_skew) > max(5 * ctx.err(g_ref), 0.20),
           f"max|err| {ctx.err(g_skew):.4f} vs {ctx.err(g_ref):.4f} with the right critic")

    # The one that is worth more than everything above: the WRONG-LEVEL critic
    # bootstraps almost without bias. Its error V^pi_inf(s) - V_t(s) is ~1.46 in
    # BOTH states, and delta_t only ever sees the error through
    #     gamma * e(s_{t+1}) - e(s_t),
    # so a state-INDEPENDENT error is a constant, a constant is a baseline, and a
    # baseline is free. Bootstrap bias is created by the part of the critic's
    # error that VARIES with the state you bootstrap into -- i.e. by getting the
    # relative values wrong, not the absolute ones. (An unconverged PPO critic
    # with a uniform offset is therefore much less dangerous than one that ranks
    # states incorrectly, even at the same explained variance.)
    g_mis, _ = ctx.gae_stats("mismatch", ctx.V_mismatch, 0.0)
    _check("a wrong-LEVEL, right-RANKING critic bootstraps ~without bias",
           ctx.err(g_mis) < tol,
           f"max|err| {ctx.err(g_mis):.4f} despite being off by "
           f"{np.mean(ctx.V_mismatch[:ctx.T] - ctx.V_exact[:ctx.T]):.3f} everywhere")


def stage_sweep(ctx: Ctx) -> None:
    """THE headline: the lambda dial, measured for a right and a wrong critic"""
    _fail_fast(ctx)
    critics = (
        ("exact", ctx.V_exact, "EXACT finite-horizon V_t -- the right critic"),
        ("skew", ctx.V_skew, "WRONG RANKING (+1/-1 per state), right level"),
        ("mismatch", ctx.V_mismatch, "WRONG LEVEL (infinite-horizon V^pi), right ranking"),
    )
    rows = {}
    for tag, V, _ in critics:
        rows[tag] = []
        for lam in ctx.lams:
            g, var = ctx.gae_stats(tag, V, lam)
            rows[tag].append((lam, ctx.err(g), ctx.l2(g), float(var)))

    for tag, _, _ in critics:
        var_seq = [r[3] for r in rows[tag]]
        _check(f"variance RISES monotonically with lambda [{tag} critic]",
               all(np.diff(var_seq) > 0),
               "  ".join(f"{v:.3f}" for v in var_seq))
    bias_seq = [r[2] for r in rows["skew"]]
    _check("bias FALLS monotonically with lambda [wrong-ranking critic]",
           all(np.diff(bias_seq) < 0), "  ".join(f"{b:.4f}" for b in bias_seq))
    bias_exact = [r[2] for r in rows["exact"]]
    _check("bias is ~flat and ~zero for every lambda [exact critic]",
           max(bias_exact) < 0.15,
           "  ".join(f"{b:.4f}" for b in bias_exact) + "  <- no bias to trade away")
    # The tradeoff is only a real tradeoff when both terms move: with the
    # wrong-ranking critic the MSE-optimal lambda must be strictly INTERIOR.
    mse = [r[2] ** 2 + r[3] for r in rows["skew"]]
    _check("MSE-optimal lambda is strictly interior for the wrong-ranking critic",
           0 < int(np.argmin(mse)) < len(mse) - 1,
           f"argmin at lambda={ctx.lams[int(np.argmin(mse))]}  ("
           + "  ".join(f"{m:.3f}" for m in mse) + ")")

    for tag, _, title in critics:
        print(f"\n  {title}:")
        print(f"    {'lambda':>7} {'max|bias|':>10} {'||bias||':>9} {'variance':>10} {'MSE':>10}")
        for lam, e, l2, var in rows[tag]:
            print(f"    {lam:7.2f} {e:10.4f} {l2:9.4f} {var:10.3f} {l2 ** 2 + var:10.3f}")
        best = min(rows[tag], key=lambda r: r[2] ** 2 + r[3])
        print(f"    -> single-sample MSE is minimized at lambda = {best[0]}")
    print("\n  Read it as: lambda=1 pays full variance for zero bias no matter how bad"
          "\n  the critic is; lambda<1 pays the critic's error as BIAS in exchange for"
          "\n  lower variance. A correct critic has no bias to trade, so lambda=0 wins"
          "\n  outright -- lambda>0 exists ONLY as a hedge against a critic you do not"
          "\n  trust. And note which kind of wrongness costs you: the wrong-LEVEL critic"
          "\n  bootstraps nearly bias-free (its error cancels in gamma*e(s') - e(s)) but"
          "\n  is a variance disaster; the wrong-RANKING critic is the one that biases.")


def stage_ladder(ctx: Ctx) -> None:
    """time-indexing V_t is the constructive fix for Day 3's mis-centred baseline"""
    _fail_fast(ctx)
    _, var_zero = ctx.gae_stats("zero", ctx.V_zero, 1.0)
    _, var_pool = ctx.gae_stats("pooled", ctx.V_pooled, 1.0)
    _, var_time = ctx.gae_stats("fit_t", ctx.V_fit, 1.0)
    _, var_exact = ctx.gae_stats("exact", ctx.V_exact, 1.0)
    _, var_mis = ctx.gae_stats("mismatch", ctx.V_mismatch, 1.0)
    g_time, _ = ctx.gae_stats("fit_t", ctx.V_fit, 1.0)

    _check("Day-3 numbers reproduce (V=0 at lam=1 is Day-2 reward_to_go)",
           abs(var_zero - 6.317) < (1.0 if ctx.fast else 0.05),
           f"var {var_zero:.3f} vs Day 3's 6.317")
    _check("pooled b(s) beats no baseline", var_pool < var_zero,
           f"{var_pool:.3f} < {var_zero:.3f}  ({var_zero / var_pool:.2f}x)")
    _check("TIME-INDEXED V_t(s) beats the pooled b(s)", var_time < var_pool,
           f"{var_time:.3f} < {var_pool:.3f}  ({var_pool / var_time:.2f}x on top)")
    _check("the exact V_t is the best baseline of all", var_exact <= var_time * 1.02,
           f"exact {var_exact:.3f} vs fitted {var_time:.3f}")
    _check("Day 3's mismatched V^pi is still WORSE than no baseline", var_mis > var_zero,
           f"{var_mis:.3f} > {var_zero:.3f}  -- unbiased, and worse than nothing")
    _check("the time-indexed baseline is still unbiased",
           ctx.err(g_time) < (0.20 if ctx.fast else 0.10),
           f"max|err| {ctx.err(g_time):.4f}")

    print("\n  baseline ladder at lam=1 (identical rollouts, only V differs):")
    print(f"    no baseline (V=0)          {var_zero:10.3f}   1.00x")
    print(f"    pooled b(s)      [Day 3]   {var_pool:10.3f}   {var_zero / var_pool:.2f}x")
    print(f"    fitted V_t(s)    [today]   {var_time:10.3f}   {var_zero / var_time:.2f}x")
    print(f"    exact V_t(s)     [oracle]  {var_exact:10.3f}   {var_zero / var_exact:.2f}x")
    print(f"    infinite-horizon V^pi      {var_mis:10.3f}   {var_zero / var_mis:.2f}x"
          "  <- Day 3's cautionary result, now explained: wrong HORIZON")


def stage_train(ctx: Ctx) -> None:
    """STRETCH: actor-critic with GAE actually learns, for every lambda"""
    print(f"\n  {'lambda':>7} {'critic':>7} {'return':>8}   pi(.|s0)")
    results = []
    for lam in [0.0, 0.95, 1.0]:
        th, hist = train_gae(ctx.mdp, ctx.T, ctx.s0, n_iters=ctx.n_iters,
                             lam=lam, critic="fit")
        ret = float(np.mean(hist[-10:]))
        pi = policy_probs(th).detach().numpy()
        results.append((lam, ret, pi))
        print(f"    {lam:7.2f} {'fit':>7} {ret:8.3f}   {np.round(pi[0], 3)}")
    for lam, ret, pi in results:
        _check(f"lam={lam} learns the task", ret > 6.5, f"return {ret:.3f} / {ctx.T}")
        _check(f"lam={lam} reaches the Day-1 optimum ('reach & stay')",
               pi[0, 1] > 0.9 and pi[1, 0] > 0.9, f"pi = {np.round(pi, 3).tolist()}")


STAGES = [
    ("td", stage_td),
    ("kstep", stage_kstep),
    ("scan", stage_scan),
    ("exactvalues", stage_exactvalues),
    ("fitvalues", stage_fitvalues),
    ("regression", stage_regression),
    ("unbiased", stage_unbiased),
    ("sweep", stage_sweep),
    ("ladder", stage_ladder),
    ("train", stage_train),
]


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
