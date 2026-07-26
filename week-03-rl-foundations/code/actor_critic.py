"""
Week 3 Day 3 -- Baselines, the advantage form, and variance reduction.

Day 2 established the two claims that make this file possible:

  * `exact_J` (T Bellman backups) + central finite differences give a GROUND-TRUTH
    gradient for the tabular MDP, so "is my estimator unbiased?" is an ASSERTION,
    not a hope.
  * reward-to-go cut the gradient variance 3.15x at identical mean.

Today's three claims, in the order the harness checks them:

  (1) A baseline b(s) that is h_t-MEASURABLE (known before a_t is drawn, and
      independent of the episode it is applied to) leaves the mean gradient
      UNCHANGED. Both an oracle V^pi baseline and one fit on a SEPARATE batch of
      rollouts still match the finite-difference ground truth.

  (2) A baseline that peeks at a_t is BIASED -- and not subtly. The
      within-episode "G_t - mean(G)" baseline (the exact bug from Day 2) is
      asserted to FAIL the ground-truth check. A test that proves your bug is a
      bug is worth more than one that proves your fix is a fix.

  (3) Baselines STACK with reward-to-go. total_return > reward_to_go >
      reward_to_go + baseline in variance, at the same mean.

  (4) Stretch: the theoretically optimal baseline from the derivation,
      b* = E[||grad log pi||^2 G] / E[||grad log pi||^2], beats V^pi -- but only
      barely, which is why nobody uses it.

Implement the bodies marked TODO. Everything else is plumbing or reuse.

The harness is STAGED: every stage runs independently, and a stage whose function
is still a TODO reports [TODO] and is skipped instead of killing the run. So you
can implement one function, run the file, and watch the PASS count climb.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/actor_critic.py
  one stage           ...  actor_critic.py weights
  a few stages        ...  actor_critic.py fit oracle weights
  small/fast batches  ...  actor_critic.py --fast
  list the stages     ...  actor_critic.py --list
"""

from __future__ import annotations

import sys
import traceback
from functools import cached_property

import numpy as np
import torch

from mdp import (
    MDP,
    bellman_expectation_backup,
    policy_evaluation_linear,
    two_state_chain,
)
from reinforce import (
    discount_cumsum,
    episode_gradient,
    exact_gradient,
    policy_probs,
    sample_episode,
)


# ---------------------------------------------------------------------------
# Rollout storage (provided -- plumbing)
# ---------------------------------------------------------------------------

class Batch:
    """A batch of rollouts, stored so several estimators can reuse the SAME data.

    states, actions : (n, T) int arrays
    rewards         : (n, T) float array
    returns_to_go   : (n, T) float array -- G_t, undiscounted-at-t reward-to-go
                      i.e. G_t = sum_{k>=t} gamma^{k-t} r_k
    """

    def __init__(self, mdp: MDP, theta: torch.Tensor, T: int, s0: int,
                 n: int, rng: np.random.Generator):
        self.n, self.T = n, T
        self.states = np.zeros((n, T), dtype=np.int64)
        self.actions = np.zeros((n, T), dtype=np.int64)
        self.rewards = np.zeros((n, T), dtype=np.float64)
        for i in range(n):
            s, a, r = sample_episode(mdp, theta, T, s0, rng)
            self.states[i], self.actions[i], self.rewards[i] = s, a, r
        self.returns_to_go = np.stack(
            [np.asarray(discount_cumsum(self.rewards[i], mdp.gamma), dtype=np.float64)
             for i in range(n)]
        )
        self.total_returns = self.rewards.sum(axis=1)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def fit_state_baseline(batch: Batch, n_states: int) -> np.ndarray:
    """Monte-Carlo value baseline: b[s] = mean of G_t over every (i, t) with s_t == s.

    This is the crudest possible critic -- a table regressed on observed
    reward-to-go, which is exactly what a critic network does with a neural
    function approximator instead of a table.

    Returns a (n_states,) float array. Unvisited states must get 0.0 (a baseline
    is allowed to be WRONG without being biased -- only its variance reduction
    suffers -- so a default of 0 is safe, which is itself worth internalizing).

    Vectorize with np.bincount (weights= for the sum, then divide by counts);
    do NOT loop over states with boolean masks unless you want to.
    """
    state_baseline = np.bincount(batch.states.ravel(), batch.returns_to_go.ravel(), minlength=n_states)
    state_count = np.bincount(batch.states.ravel(), minlength=n_states)
    state_count[state_count==0] = 1
    return state_baseline/state_count



def oracle_state_baseline(mdp: MDP, theta: torch.Tensor) -> np.ndarray:
    """The exact V^pi(s) for the current policy, via the Day-1 linear solve.

    Note this is the INFINITE-horizon V^pi while the objective is the T-step
    J_T -- so it is not even the "right" value function for this objective. It is
    still a perfectly unbiased baseline, because unbiasedness only needs
    h_t-measurability, never correctness. Two lines.
    """
    pi = policy_probs(theta)
    return policy_evaluation_linear(mdp, pi.detach().numpy())


def optimal_state_baseline(
    mdp: MDP, theta: torch.Tensor, batch: Batch
) -> np.ndarray:
    """STRETCH -- the variance-minimizing baseline you derived.

    For vector theta, minimizing the TOTAL variance sum_i Var(g_i) gives

        b*(s) = E[ ||grad_theta log pi(a|s)||^2 * G ] / E[ ||grad_theta log pi(a|s)||^2 ]

    i.e. the score-magnitude-WEIGHTED average of returns, versus V^pi which is the
    UNWEIGHTED average. Estimate both expectations from `batch`, per state.

    For a tabular softmax policy the score has a closed form -- no autograd needed:
        grad_theta log pi(a|s) is nonzero only in row s, where it equals
        (e_a - pi(.|s)).
    So ||grad log pi(a|s)||^2 = ||e_a - pi(.|s)||^2, computable per (i, t) by
    lookup. Returns (S,), 0.0 for unvisited states.
    """
    actions, states = mdp.A, mdp.S
    pi = policy_probs(theta).detach().numpy()
    episodes, T = batch.n, batch.T
    pi_rows = pi[batch.states]
    one_hot = np.eye(actions)[batch.actions]
    w = ((one_hot - pi_rows)**2).sum(axis=-1)

    state_baseline = np.bincount(batch.states.ravel(), (batch.returns_to_go * w).ravel(), minlength=states)
    state_count = np.bincount(batch.states.ravel(), w.ravel(), minlength=states)
    state_count[state_count==0] = 1
    return state_baseline/state_count
    
    


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def weights_with_baseline(
    batch: Batch, i: int, gamma: float, baseline: np.ndarray | None
) -> np.ndarray:
    """Psi_t for episode i, shape (T,):

        Psi_t = gamma^t * ( G_t - b(s_t) )

    With baseline=None this must reproduce Day 2's "reward_to_go" mode exactly
    (the harness asserts that as a regression test). Keep the gamma^t -- it is
    what makes the estimator exactly unbiased for grad J_T, which is the only
    reason the ground-truth check can be tight.
    """
    T = batch.T
    if baseline is None:
        return gamma**np.arange(T) * batch.returns_to_go[i]
    else:
        return gamma**np.arange(T) * (batch.returns_to_go[i] - baseline[batch.states[i]])
    

def weights_within_episode_baseline(
    batch: Batch, i: int, gamma: float
) -> np.ndarray:
    """The BUG, implemented on purpose (you wrote this on Day 2).

        Psi_t = gamma^t * ( G_t - mean_t(G_t) )     <-- mean over THIS episode

    The subtracted quantity depends on a_t and on everything after it, so it is
    not h_t-measurable and the estimator is biased. The harness asserts that this
    FAILS to match the ground-truth gradient.
    """
    return gamma**np.arange(batch.T) * (batch.returns_to_go[i] - batch.returns_to_go[i].mean())


# ---------------------------------------------------------------------------
# Estimator statistics
# ---------------------------------------------------------------------------

def batch_gradient_statistics(
    theta: torch.Tensor,
    batch: Batch,
    gamma: float,
    weight_fn,
) -> tuple[np.ndarray, float]:
    """(mean_gradient, total_variance) over the episodes in `batch`.

    weight_fn(batch, i, gamma) -> (T,) array of Psi_t, so the caller decides which
    estimator is under test while the DATA stays fixed. Holding the rollouts fixed
    across estimators is the whole point: any variance difference you measure is
    then attributable to Psi_t and not to sampling luck.

    mean_gradient : (S, A)
    total_variance: sum over (s, a) coordinates of the per-episode variance
                    (ddof=1), matching Day 2's definition so the numbers compose.
    """
    # TODO -- loop episodes, call weight_fn, then reinforce.episode_gradient
    grads = np.zeros((batch.n, theta.shape[0], theta.shape[1]), dtype=np.float64)
    for ep in range(batch.n):
        Psi = weight_fn(batch, ep, gamma)
        grads[ep] = episode_gradient(theta, batch.states[ep], batch.actions[ep], Psi)
    mean_grad = grads.mean(axis=0)
    var = grads.var(axis=0, ddof=1).sum()
    return (mean_grad, var)




# ---------------------------------------------------------------------------
# Entropy regularization
# ---------------------------------------------------------------------------

def policy_entropy(theta: torch.Tensor) -> torch.Tensor:
    """Per-state entropy H(pi(.|s)) = -sum_a pi log pi. Returns a (S,) tensor,
    differentiable w.r.t. theta. Use log_softmax, never softmax(...).log().
    """
    log_pi = torch.log_softmax(theta, dim=-1)
    return -(log_pi.exp()*log_pi).sum(axis=-1)


def entropy_gradient(theta: torch.Tensor) -> torch.Tensor:
    """grad_theta of the MEAN per-state entropy. Returns a detached (S, A) tensor.

    Added to the ascent direction with coefficient beta, this is the entropy
    BONUS: it pushes the policy back toward uniform, fighting the premature
    determinism that a policy gradient drives it to (once pi(a|s) ~ 1 the
    gradient vanishes and no other action is ever sampled again -- collapse is
    self-locking).
    """
    entropy_mean = policy_entropy(theta).mean()
    return torch.autograd.grad(entropy_mean, theta)[0].detach()
    


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    mdp: MDP,
    T: int,
    s0: int,
    n_iters: int = 200,
    batch_size: int = 32,
    lr: float = 0.1,
    baseline: str = "none",       # "none" | "fit" | "oracle"
    beta_entropy: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """REINFORCE with an optional baseline and entropy bonus.

    Returns (theta, return_history, entropy_history) where entropy_history[k] is
    the mean per-state entropy after iteration k.

    Per iteration:
      * baseline == "none"   -> b = None
      * baseline == "oracle" -> b = V^pi for the CURRENT theta (exact)
      * baseline == "fit"    -> roll out a SEPARATE batch of `batch_size` episodes
                                and fit b on it, THEN roll out the gradient batch.
                                Two independent batches is what keeps b
                                independent of the episodes it weights. (Costs 2x
                                rollouts; a real critic amortizes this by
                                carrying b across iterations instead.)
      * ascent step on the mean gradient, plus beta_entropy * entropy_gradient.

    Same loop invariants as Week 1: seed first, resample every iteration
    (on-policy), update under no_grad.
    """
    rng = np.random.default_rng(seed)
    theta = torch.zeros(mdp.S, mdp.A, dtype=torch.float64, requires_grad=True)
    return_history, entropy_history = [], []
    for iter in range(n_iters):
        if baseline == 'none':
            b = None
        elif baseline == 'oracle':
            b = oracle_state_baseline(mdp, theta)
        elif baseline == 'fit':
            batch_baseline = Batch(mdp, theta, T, s0, batch_size, rng)
            b = fit_state_baseline(batch_baseline, mdp.S)
        else:
            raise ValueError(baseline)
        batch = Batch(mdp, theta, T, s0, batch_size, rng)
        mean_grad, _ = batch_gradient_statistics(theta, batch, mdp.gamma, lambda B, i, gm: weights_with_baseline(B, i, gm, b))
        return_history.append(batch.total_returns.mean().item())
        entropy_history.append(policy_entropy(theta).mean().item())
        entropy_grad = entropy_gradient(theta)
        with torch.no_grad():
            theta += lr * (torch.as_tensor(mean_grad, dtype=entropy_grad.dtype) + beta_entropy * entropy_grad)
    
    return (theta, return_history, entropy_history)






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

    Everything expensive is a cached_property with its OWN seeded generator, so a
    stage costs nothing unless it is actually run and the data does not depend on
    which other stages ran first.
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
        self._stats: dict[str, tuple[np.ndarray, float]] = {}

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
        """An INDEPENDENT batch, used only to fit the baseline."""
        return Batch(self.mdp, self.theta, self.T, self.s0, self.n_fit,
                     np.random.default_rng(1))

    @cached_property
    def b_fit(self) -> np.ndarray:
        return fit_state_baseline(self.fit_batch, self.mdp.S)

    @cached_property
    def b_oracle(self) -> np.ndarray:
        return oracle_state_baseline(self.mdp, self.theta)

    def stats(self, key: str, weight_fn) -> tuple[np.ndarray, float]:
        if key not in self._stats:
            self._stats[key] = batch_gradient_statistics(
                self.theta, self.big, self.gamma, weight_fn
            )
        return self._stats[key]

    def err(self, g: np.ndarray) -> float:
        return float(np.max(np.abs(g - self.g_true)))


# ---- individual stages ----------------------------------------------------

def stage_fit(ctx: Ctx) -> None:
    """fit_state_baseline"""
    tiny = Batch(ctx.mdp, ctx.theta, T=3, s0=ctx.s0, n=2, rng=np.random.default_rng(7))
    b = np.asarray(fit_state_baseline(tiny, ctx.mdp.S), dtype=np.float64)
    _check("returns shape (S,)", b.shape == (ctx.mdp.S,), f"got {b.shape}")

    expected = np.zeros(ctx.mdp.S)
    for s in range(ctx.mdp.S):
        m = tiny.states == s
        expected[s] = tiny.returns_to_go[m].mean() if m.any() else 0.0
    _check("b[s] == per-state mean of G_t", np.allclose(b, expected),
           f"got {np.round(b, 4)} want {np.round(expected, 4)}")

    # A 1-step episode from s0=0 can never visit state 1 -> exercises the guard.
    lonely = Batch(ctx.mdp, ctx.theta, T=1, s0=0, n=1, rng=np.random.default_rng(5))
    b2 = np.asarray(fit_state_baseline(lonely, ctx.mdp.S), dtype=np.float64)
    _check("unvisited state gets exactly 0.0 (no nan)", np.isfinite(b2).all() and b2[1] == 0.0,
           f"got {np.round(b2, 4)}")


def stage_oracle(ctx: Ctx) -> None:
    """oracle_state_baseline"""
    b = np.asarray(oracle_state_baseline(ctx.mdp, ctx.theta), dtype=np.float64)
    pi = policy_probs(ctx.theta).detach().numpy()
    _check("returns shape (S,)", b.shape == (ctx.mdp.S,), f"got {b.shape}")
    # Independent check: V^pi must be a fixed point of the Bellman expectation
    # operator (not merely equal to whatever policy_evaluation_linear returned).
    _check("V^pi is a fixed point of T^pi",
           np.allclose(bellman_expectation_backup(ctx.mdp, pi, b), b, atol=1e-8),
           f"V^pi = {np.round(b, 4)}")
    _check("0 <= V^pi <= 1/(1-gamma)", np.all(b >= 0) and np.all(b <= 1 / (1 - ctx.gamma)))


def stage_weights(ctx: Ctx) -> None:
    """weights_with_baseline"""
    from reinforce import episode_weights

    b0 = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=3, rng=np.random.default_rng(11))
    w = np.asarray(weights_with_baseline(b0, 0, ctx.gamma, None), dtype=np.float64)
    _check("returns shape (T,)", w.shape == (ctx.T,), f"got {w.shape}")
    _check(
        "baseline=None reproduces Day-2 reward_to_go exactly",
        all(
            np.allclose(
                np.asarray(weights_with_baseline(b0, i, ctx.gamma, None), dtype=np.float64),
                np.asarray(episode_weights(b0.rewards[i], ctx.gamma, "reward_to_go"),
                           dtype=np.float64),
            )
            for i in range(b0.n)
        ),
    )
    # The baseline must be discounted by the SAME gamma^t as the return.
    b = np.array([1.0, -2.0])
    wb = np.asarray(weights_with_baseline(b0, 0, ctx.gamma, b), dtype=np.float64)
    shift = ctx.gamma ** np.arange(ctx.T) * b[b0.states[0]]
    _check("subtracting b(s_t) shifts Psi_t by exactly gamma^t * b(s_t)",
           np.allclose(wb, w - shift), f"max|diff| {np.max(np.abs(wb - (w - shift))):.2e}")


def stage_badweights(ctx: Ctx) -> None:
    """weights_within_episode_baseline"""
    b0 = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=3, rng=np.random.default_rng(11))
    w = np.asarray(weights_within_episode_baseline(b0, 0, ctx.gamma), dtype=np.float64)
    G = b0.returns_to_go[0]
    expected = ctx.gamma ** np.arange(ctx.T) * (G - G.mean())
    _check("Psi_t == gamma^t (G_t - mean_t G_t)", np.allclose(w, expected),
           f"max|diff| {np.max(np.abs(w - expected)):.2e}")
    _check("undiscounted centred weights sum to 0 within the episode",
           np.isclose((G - G.mean()).sum(), 0.0))


def stage_entropy(ctx: Ctx) -> None:
    """policy_entropy + entropy_gradient"""
    H = policy_entropy(ctx.theta)
    _check("policy_entropy returns shape (S,)", tuple(H.shape) == (ctx.mdp.S,),
           f"got {tuple(H.shape)}")
    _check("entropy is differentiable w.r.t. theta", H.requires_grad)

    uniform = torch.zeros(ctx.mdp.S, ctx.mdp.A, dtype=torch.float64, requires_grad=True)
    _check("uniform policy has H = ln A",
           torch.allclose(policy_entropy(uniform),
                          torch.full((ctx.mdp.S,), float(np.log(ctx.mdp.A)), dtype=torch.float64)),
           f"got {policy_entropy(uniform).detach().numpy().round(4)}")

    peaked = torch.tensor([[50.0, -50.0], [-50.0, 50.0]], dtype=torch.float64, requires_grad=True)
    _check("near-deterministic policy has H ~ 0",
           float(policy_entropy(peaked).max().detach()) < 1e-6,
           f"got {policy_entropy(peaked).detach().numpy().round(8)}")

    g = np.asarray(entropy_gradient(ctx.theta), dtype=np.float64)
    _check("entropy_gradient returns shape (S, A)", g.shape == (ctx.mdp.S, ctx.mdp.A),
           f"got {g.shape}")
    # Softmax over-parameterization: adding a constant to a logit row is a no-op,
    # so EVERY gradient row must sum to zero. Same invariant as the Day-2 PG.
    _check("every gradient row sums to zero (softmax null space)",
           np.allclose(g.sum(axis=1), 0.0, atol=1e-10), f"row sums {g.sum(axis=1)}")
    with torch.no_grad():
        stepped = (ctx.theta + 0.5 * torch.as_tensor(g)).clone().requires_grad_(True)
    _check("an ascent step on the entropy gradient RAISES entropy",
           float(policy_entropy(stepped).mean()) > float(policy_entropy(ctx.theta).mean()),
           f"{float(policy_entropy(ctx.theta).mean()):.5f} -> "
           f"{float(policy_entropy(stepped).mean()):.5f}")


def stage_gradstats(ctx: Ctx) -> None:
    """batch_gradient_statistics (cheap smoke test on 300 episodes)"""
    small = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=300, rng=np.random.default_rng(3))
    g, var = batch_gradient_statistics(
        ctx.theta, small, ctx.gamma,
        lambda B, i, gm: weights_with_baseline(B, i, gm, None),
    )
    g = np.asarray(g)
    _check("mean gradient has shape (S, A)", g.shape == (ctx.mdp.S, ctx.mdp.A), f"got {g.shape}")
    # The Day-2 `dtype=np.long` trap: an int accumulator truncates every float to
    # 0 and every downstream check then "passes" on a matrix of zeros.
    _check("mean gradient is floating point and not all zeros",
           np.issubdtype(g.dtype, np.floating) and not np.allclose(g, 0.0),
           f"dtype {g.dtype}")
    _check("every gradient row sums to zero", np.allclose(g.sum(axis=1), 0.0, atol=1e-10),
           f"row sums {g.sum(axis=1)}")
    _check("total variance is a positive scalar", np.isscalar(var) or np.ndim(var) == 0)
    _check("300-episode estimate is in the right ballpark", ctx.err(g) < 0.6,
           f"max|err| {ctx.err(g):.4f} vs truth (noisy at n=300, just a smoke test)")


def _fail_fast(ctx: Ctx) -> None:
    """Touch every function the heavy stages need, on a 1-episode batch.

    Without this, an unimplemented TODO is only discovered AFTER paying for
    thousands of rollouts.
    """
    # n=2, not 1: a single episode makes the ddof=1 variance undefined and numpy
    # warns about it.
    mini = Batch(ctx.mdp, ctx.theta, ctx.T, ctx.s0, n=2, rng=np.random.default_rng(99))
    fit_state_baseline(mini, ctx.mdp.S)
    oracle_state_baseline(ctx.mdp, ctx.theta)
    weights_with_baseline(mini, 0, ctx.gamma, None)
    weights_within_episode_baseline(mini, 0, ctx.gamma)
    batch_gradient_statistics(
        ctx.theta, mini, ctx.gamma, lambda B, i, gm: weights_with_baseline(B, i, gm, None)
    )


def stage_unbiased(ctx: Ctx) -> None:
    """unbiasedness of every baseline, against the finite-difference ground truth"""
    _fail_fast(ctx)
    g_none, _ = ctx.stats("none", lambda B, i, gm: weights_with_baseline(B, i, gm, None))
    g_fit, _ = ctx.stats("fit", lambda B, i, gm: weights_with_baseline(B, i, gm, ctx.b_fit))
    g_orc, _ = ctx.stats("oracle", lambda B, i, gm: weights_with_baseline(B, i, gm, ctx.b_oracle))
    g_bad, _ = ctx.stats("within", weights_within_episode_baseline)

    tol = 0.20 if ctx.fast else 0.10
    _check("reward_to_go (no baseline) is unbiased", ctx.err(g_none) < tol,
           f"max|err| {ctx.err(g_none):.4f}")
    _check("independent-batch baseline is unbiased", ctx.err(g_fit) < tol,
           f"max|err| {ctx.err(g_fit):.4f}")
    _check("oracle V^pi baseline is unbiased", ctx.err(g_orc) < tol,
           f"max|err| {ctx.err(g_orc):.4f}")
    # The bug must be DETECTABLY a bug: a test that only proves the fix works
    # tells you nothing about whether the test has teeth.
    _check("within-episode baseline is BIASED (must miss the ground truth)",
           ctx.err(g_bad) > max(5 * ctx.err(g_none), 0.20),
           f"max|err| {ctx.err(g_bad):.4f} vs unbiased {ctx.err(g_none):.4f}")

    print("\n  true grad        =\n", np.round(ctx.g_true, 4))
    print("  MC (rtg, no b)   =\n", np.round(g_none, 4))
    print("  MC (rtg + b_fit) =\n", np.round(g_fit, 4))
    print("  MC (WITHIN-ep b) =\n", np.round(g_bad, 4), "  <-- biased: the MEAN is wrong")
    print(f"\n  b_fit = {np.round(ctx.b_fit, 4)}    V^pi = {np.round(ctx.b_oracle, 4)}")


def stage_variance(ctx: Ctx) -> None:
    """the punchline: baselines stack on top of reward-to-go"""
    _fail_fast(ctx)
    _, var_total = ctx.stats(
        "total",
        lambda B, i, gm: np.full(B.T, float(np.sum(gm ** np.arange(B.T) * B.rewards[i]))),
    )
    _, var_none = ctx.stats("none", lambda B, i, gm: weights_with_baseline(B, i, gm, None))
    _, var_fit = ctx.stats("fit", lambda B, i, gm: weights_with_baseline(B, i, gm, ctx.b_fit))
    _, var_orc = ctx.stats("oracle", lambda B, i, gm: weights_with_baseline(B, i, gm, ctx.b_oracle))
    _, var_bad = ctx.stats("within", weights_within_episode_baseline)

    _check("reward_to_go beats total_return", var_none < var_total,
           f"{var_none:.3f} < {var_total:.3f}  ({var_total / var_none:.2f}x)")
    _check("fitted baseline stacks on top of reward_to_go", var_fit < var_none,
           f"{var_fit:.3f} < {var_none:.3f}  ({var_none / var_fit:.2f}x)")
    # A MISMATCHED baseline is still unbiased (asserted in `unbiased`) but makes
    # things WORSE: V^pi is the infinite-horizon value ~1/(1-gamma), while Psi_t
    # uses the T-step reward-to-go pooled over t, which averages far lower. So
    # subtracting V^pi decentres Psi_t instead of centring it. Unbiasedness and
    # variance reduction are independent properties -- this is the assertion that
    # pins that down, and it is the failure mode of an unconverged PPO critic.
    _check("MISMATCHED oracle V^pi is unbiased but INCREASES variance", var_orc > var_none,
           f"{var_orc:.3f} > {var_none:.3f}  ({var_none / var_orc:.2f}x, i.e. worse)")
    _check("a well-matched baseline beats a mismatched one", var_fit < var_orc,
           f"fitted {var_fit:.3f} vs V^pi {var_orc:.3f}")

    print("\n  variance ladder (identical rollouts -- only Psi_t differs):")
    print(f"    total_return           {var_total:10.3f}   1.00x")
    print(f"    reward_to_go           {var_none:10.3f}   {var_total / var_none:.2f}x")
    print(f"    rtg + fitted baseline  {var_fit:10.3f}   {var_total / var_fit:.2f}x")
    print(f"    rtg + oracle V^pi      {var_orc:10.3f}   {var_total / var_orc:.2f}x"
          "  <-- horizon-mismatched: unbiased but WORSE")
    print(f"    rtg + WITHIN-ep (bad)  {var_bad:10.3f}   {var_total / var_bad:.2f}x"
          "  <-- cheap variance, bought with bias")


def stage_train(ctx: Ctx) -> None:
    """train + the entropy-bonus ablation (beta sweep)"""
    betas = [0.0, 0.5, 2.0, 5.0]
    rets, ents, pis = [], [], []
    for beta in betas:
        th, hist, ent = train(ctx.mdp, ctx.T, ctx.s0, n_iters=ctx.n_iters,
                              baseline="fit", beta_entropy=beta)
        rets.append(float(np.mean(hist[-10:])))
        ents.append(float(np.mean(ent[-10:])))
        pis.append(policy_probs(th).detach().numpy())

    _check("baseline run learns the task", rets[0] > 6.5,
           f"final return {rets[0]:.3f} / {ctx.T}")
    _check("policy converges to the Day-1 optimum ('reach & stay')",
           pis[0][0, 1] > 0.9 and pis[0][1, 0] > 0.9,
           f"pi = {np.round(pis[0], 3).tolist()}")
    # beta=0.05 is far too weak to see: the entropy gradient
    # dH/dz_k = -p_k(log p_k + H) SHRINKS as the policy sharpens, so the bonus is
    # weakest exactly where you want it most. It only bites at beta ~ O(1), the
    # scale of the policy gradient itself.
    _check("entropy rises monotonically with beta", all(np.diff(ents) > 0),
           f"H {np.round(ents, 4).tolist()}")
    _check("return falls monotonically with beta (the exploration price)",
           all(np.diff(rets) < 0), f"return {np.round(rets, 3).tolist()}")
    _check("a large entropy bonus measurably costs return", rets[-1] < rets[0] - 0.5,
           f"beta={betas[-1]}: {rets[-1]:.3f} vs beta=0: {rets[0]:.3f}")

    print(f"\n  entropy-bonus sweep (baseline='fit', {ctx.n_iters} iters, "
          f"uniform H = {np.log(ctx.mdp.A):.4f}):")
    print(f"    {'beta':>6} {'return':>8} {'entropy':>8}   pi(.|s0)")
    for beta, r, h, pi in zip(betas, rets, ents, pis):
        print(f"    {beta:6.2f} {r:8.3f} {h:8.4f}   {np.round(pi[0], 3)}")


def stage_optimal(ctx: Ctx) -> None:
    """STRETCH: the variance-minimizing baseline b*"""
    _fail_fast(ctx)
    b_star = np.asarray(optimal_state_baseline(ctx.mdp, ctx.theta, ctx.fit_batch),
                        dtype=np.float64)
    _check("returns shape (S,)", b_star.shape == (ctx.mdp.S,), f"got {b_star.shape}")
    g_star, var_star = ctx.stats(
        "bstar", lambda B, i, gm: weights_with_baseline(B, i, gm, b_star)
    )
    _, var_orc = ctx.stats("oracle", lambda B, i, gm: weights_with_baseline(B, i, gm, ctx.b_oracle))
    tol = 0.20 if ctx.fast else 0.10
    _check("b* is still unbiased", ctx.err(g_star) < tol, f"max|err| {ctx.err(g_star):.4f}")
    _check("b* is at least as good as V^pi", var_star <= var_orc * 1.02,
           f"var {var_star:.4f} vs V^pi {var_orc:.4f}")
    print(f"\n  b*   = {np.round(b_star, 4)}   (score-magnitude-weighted mean return)")
    print(f"  V^pi = {np.round(ctx.b_oracle, 4)}   (unweighted mean return)")


STAGES = [
    ("fit", stage_fit),
    ("oracle", stage_oracle),
    ("weights", stage_weights),
    ("badweights", stage_badweights),
    ("entropy", stage_entropy),
    ("gradstats", stage_gradstats),
    ("unbiased", stage_unbiased),
    ("variance", stage_variance),
    ("train", stage_train),
    ("optimal", stage_optimal),
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
            print(f"  [TODO] not implemented yet -- skipped")
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
