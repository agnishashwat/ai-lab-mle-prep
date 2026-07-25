"""
Week 3 Day 2 -- REINFORCE / the score-function policy gradient.

The point of this file is not "train a toy agent" -- it is to make the two claims
from the derivation *measurable*:

  (1) UNBIASEDNESS. The Monte-Carlo score-function estimator, with either the
      total-return weight R(tau) or the reward-to-go weight, converges to the TRUE
      gradient of J. We can check this exactly, because the environment is the
      tabular MDP from Day 1: J_T(theta) is computable in closed form by T Bellman
      backups, so a finite-difference gradient of that closed form is ground truth.

  (2) VARIANCE. Reward-to-go has strictly lower variance than the total return,
      because it deletes the provably zero-mean cross terms (t' < t).

Implement the bodies marked TODO. The harness at the bottom asserts the invariants.

Policy parameterization: tabular logits `theta` of shape (S, A); the policy is
softmax over the action axis. Deliberately the same log_softmax + gather idiom you
use for an LM -- a policy gradient is cross-entropy on your OWN samples, reweighted.

Run:  ./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/reinforce.py
"""

from __future__ import annotations

import numpy as np
import torch

from mdp import MDP, bellman_expectation_backup, two_state_chain


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def policy_probs(theta: torch.Tensor) -> torch.Tensor:
    """Softmax over the action axis. theta is (S, A) logits -> returns (S, A) probs."""
    return torch.softmax(theta, dim=-1)


def episode_logprobs(
    theta: torch.Tensor, states: np.ndarray, actions: np.ndarray
) -> torch.Tensor:
    """Per-step log pi_theta(a_t | s_t) for one episode.

    states, actions: int arrays of shape (T,). Returns a (T,) tensor that is
    differentiable w.r.t. theta.

    Use the LM idiom: log_softmax over the action axis, then gather the taken action.
    Do NOT build this with a Python loop over t -- index theta[states] to get a
    (T, A) block of logits first.
    """
    theta_seq = theta[states]
    logprob_seq_all = torch.log_softmax(theta_seq, dim=-1)
    actions_torch = torch.as_tensor(actions, dtype=torch.long).unsqueeze(-1)
    logprob_seq_action = torch.gather(logprob_seq_all, dim=-1, index=actions_torch).squeeze(-1)
    return logprob_seq_action



# ---------------------------------------------------------------------------
# Rollouts (provided -- plumbing, not the lesson)
# ---------------------------------------------------------------------------

def sample_episode(
    mdp: MDP, theta: torch.Tensor, T: int, s0: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out exactly T steps from state s0 under pi_theta.

    Returns (states, actions, rewards), each of shape (T,). rewards[t] = r(s_t, a_t)
    is the EXPECTED immediate reward of the taken action (the Day-1 MDP stores r(s,a),
    not a sampled reward), which keeps the ground-truth comparison exact.
    """
    pi = policy_probs(theta).detach().numpy()
    states = np.zeros(T, dtype=np.int64)
    actions = np.zeros(T, dtype=np.int64)
    rewards = np.zeros(T, dtype=np.float64)
    s = s0
    for t in range(T):
        a = rng.choice(mdp.A, p=pi[s])
        states[t], actions[t], rewards[t] = s, a, mdp.R[s, a]
        s = rng.choice(mdp.S, p=mdp.P[s, a])
    return states, actions, rewards


# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------

def discount_cumsum(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Reward-to-go: out[t] = sum_{k >= t} gamma^{k-t} * rewards[k]. Shape (T,).

    Write the O(T) backward recursion, not the O(T^2) double loop:
        out[T-1] = r[T-1];  out[t] = r[t] + gamma * out[t+1]
    """
    T = rewards.shape[0]
    out = torch.zeros(T, dtype=torch.float64)
    out[-1] = rewards[-1]
    for i in range(T-2, -1, -1):
        out[i] = rewards[i] + gamma * out[i+1]
    return out


def episode_weights(rewards: np.ndarray, gamma: float, mode: str) -> np.ndarray:
    """The scalar weight Psi_t multiplying grad log pi(a_t|s_t), shape (T,).

    mode == "total_return":
        Psi_t = R(tau) = sum_t gamma^t r_t   (the SAME scalar for every t)
        This is the naive estimator straight out of the log-derivative trick.

    mode == "reward_to_go":
        Psi_t = gamma^t * G_t  where G_t = sum_{k>=t} gamma^{k-t} r_k
        i.e. Psi_t = sum_{k >= t} gamma^k r_k.

        NOTE the gamma^t factor. It comes from differentiating the DISCOUNTED
        objective and is what makes this estimator exactly unbiased for grad J_T.
        Practical implementations (PPO, GRPO) drop it and use G_t alone -- formally
        biased, empirically better-conditioned. Keep it here so the ground-truth
        check below passes to Monte-Carlo precision.

    mode == "reward_to_go_baseline":   (STRETCH -- previews Day 3)
        Psi_t = gamma^t * (G_t - b) with b the mean of G_t across this batch.
        Handled by the caller, not here; raise for now if you want to skip it.
    """
    T = rewards.shape[0]
    psi = torch.ones(T, dtype=torch.float64)
    
    if mode == 'total_return':
        psi_val = 0
        for i in range(T):
            psi_val += (gamma**i) * rewards[i]
        psi = psi * psi_val
    elif mode == 'reward_to_go':
        G_t = discount_cumsum(rewards, gamma)
        psi = G_t * gamma**torch.arange(T)
    elif mode == "reward_to_go_baseline":
        raise NotImplementedError("baseline is a batch-level quantity; see Day 3")
    else:
        raise ValueError(mode)
    return psi




# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def episode_gradient(
    theta: torch.Tensor, states: np.ndarray, actions: np.ndarray, weights: np.ndarray
) -> torch.Tensor:
    """Single-episode score-function gradient estimate, shape (S, A).

        g = sum_t Psi_t * grad_theta log pi_theta(a_t | s_t)

    Build the surrogate objective  sum_t (Psi_t * logp_t)  and take
    torch.autograd.grad of it w.r.t. theta. The weights are DATA, not part of the
    graph -- make sure nothing differentiates through them.

    Return a detached (S, A) tensor.
    """
    log_pi_theta = episode_logprobs(theta, states, actions)
    w = torch.as_tensor(weights, dtype=log_pi_theta.dtype)
    surrogate = (log_pi_theta * w).sum(dim=-1)
    g = torch.autograd.grad(surrogate, theta)[0]
    return g.detach()


def collect_batch(
    mdp: MDP,
    theta: torch.Tensor,
    T: int,
    s0: int,
    n_episodes: int,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out n_episodes under the CURRENT theta (provided -- plumbing).

    Returns
    -------
    grads   : (n_episodes, S, A) -- one score-function gradient estimate per episode.
    returns : (n_episodes,)      -- per-episode UNDISCOUNTED total reward, the
                                    human-readable score (the discounted one is what
                                    you optimize; this one is what you plot).

    Both callers below need the same rollouts, so collection lives here and is done
    exactly once per batch.
    """
    grads = np.zeros((n_episodes, mdp.S, mdp.A), dtype=np.float64)
    returns = np.zeros(n_episodes, dtype=np.float64)

    for i in range(n_episodes):
        states, actions, rewards = sample_episode(mdp, theta, T, s0, rng)
        weights = episode_weights(rewards, mdp.gamma, mode)
        grads[i] = episode_gradient(theta, states, actions, weights).numpy()
        returns[i] = rewards.sum()

    return grads, returns


def gradient_statistics(
    mdp: MDP,
    theta: torch.Tensor,
    T: int,
    s0: int,
    n_episodes: int,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Roll out n_episodes and return (mean_gradient, total_variance).

    mean_gradient : (S, A) numpy array, the average of the per-episode estimates.
    total_variance: scalar = sum over all (s,a) coordinates of the per-episode
                    variance, i.e. an estimate of E||g_hat - E[g_hat]||^2.
                    This is the number that must DROP when you switch from
                    "total_return" to "reward_to_go".
    """
    grads, _ = collect_batch(mdp, theta, T, s0, n_episodes, mode, rng)

    mean_gradient = grads.mean(axis=0)
    variance = grads.var(axis=0, ddof=1).sum()
    return (mean_gradient, variance)


# ---------------------------------------------------------------------------
# Ground truth (provided): exact finite-horizon J and its finite-difference gradient
# ---------------------------------------------------------------------------

def exact_J(mdp: MDP, theta: torch.Tensor, T: int, s0: int) -> float:
    """J_T(theta) = E[sum_{t<T} gamma^t r_t] computed exactly by T Bellman backups.

    V_0 = 0;  V_{k+1} = T^pi V_k.  After T backups, V_T(s0) is the exact expected
    discounted return of a T-step rollout from s0. No sampling involved.
    """
    pi = policy_probs(theta).detach().numpy()
    V = np.zeros(mdp.S)
    for _ in range(T):
        V = bellman_expectation_backup(mdp, pi, V)
    return float(V[s0])


def exact_gradient(
    mdp: MDP, theta: torch.Tensor, T: int, s0: int, eps: float = 1e-5
) -> np.ndarray:
    """Central finite-difference gradient of exact_J w.r.t. theta. Shape (S, A)."""
    base = theta.detach().clone()
    g = np.zeros((mdp.S, mdp.A))
    for s in range(mdp.S):
        for a in range(mdp.A):
            plus, minus = base.clone(), base.clone()
            plus[s, a] += eps
            minus[s, a] -= eps
            g[s, a] = (exact_J(mdp, plus, T, s0) - exact_J(mdp, minus, T, s0)) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reinforce(
    mdp: MDP,
    T: int,
    s0: int,
    n_iters: int = 200,
    batch: int = 32,
    lr: float = 0.1,
    mode: str = "reward_to_go",
    seed: int = 0,
) -> tuple[torch.Tensor, list[float]]:
    """Vanilla REINFORCE. Returns (theta, history_of_mean_undiscounted_return).

    Structure (mirror your Week-1 loop invariants):
      seed -> init theta as ZEROS (uniform policy) -> for each iter:
        `collect_batch` gives you (grads, returns) from ONE set of rollouts,
        take an ASCENT step on J with the mean gradient (the update is not part of
        any graph -- scope it with no_grad), and append the batch's mean return.

    Note this is on-policy: every iteration must resample with the CURRENT theta.
    Reusing last iteration's episodes is the bug PPO's importance ratio exists to fix.
    """
    rng = np.random.default_rng(seed=seed)
    theta = torch.zeros(mdp.S, mdp.A, dtype=torch.float64, requires_grad=True)
    history: list[float] = []

    for _ in range(n_iters):
        grads, returns = collect_batch(mdp, theta, T, s0, batch, mode, rng)
        with torch.no_grad():
            theta += lr * torch.as_tensor(grads.mean(axis=0), dtype=theta.dtype)
        history.append(returns.mean(axis=0))
        
    return theta, history


# ---------------------------------------------------------------------------
# Self-check harness
# ---------------------------------------------------------------------------

def _check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(name)


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    mdp = two_state_chain(gamma=0.9)
    T, s0 = 8, 0

    # A non-uniform policy, so the gradient is not accidentally symmetric.
    theta = torch.tensor([[0.4, -0.2], [-0.3, 0.5]], dtype=torch.float64, requires_grad=True)

    # 1. reward-to-go arithmetic on a hand-checkable case.
    rtg = discount_cumsum(np.array([1.0, 2.0, 3.0]), gamma=0.5)
    _check(
        "discount_cumsum matches hand computation",
        np.allclose(rtg, [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3.0]),
        f"got {np.round(rtg, 4)}",
    )

    # 2. The two weightings agree on the t=0 term (both start from the full return).
    r = np.array([1.0, 0.0, 2.0, 0.0])
    w_tot = episode_weights(r, 0.9, "total_return")
    w_rtg = episode_weights(r, 0.9, "reward_to_go")
    _check("weights: total_return is constant in t", np.allclose(w_tot, w_tot[0]))
    _check("weights: reward_to_go[0] == R(tau)", np.isclose(w_rtg[0], w_tot[0]))
    _check("weights: reward_to_go is non-increasing in magnitude", w_rtg[-1] <= w_rtg[0])

    # 3. Monte-Carlo return matches the exact finite-horizon J (the env is right).
    J_star = exact_J(mdp, theta, T, s0)
    rets = []
    for _ in range(20_000):
        _, _, rw = sample_episode(mdp, theta, T, s0, rng)
        rets.append(np.sum(mdp.gamma ** np.arange(T) * rw))
    _check(
        "MC discounted return == exact J_T",
        abs(np.mean(rets) - J_star) < 0.02,
        f"MC {np.mean(rets):.4f} vs exact {J_star:.4f}",
    )

    # 4. UNBIASEDNESS: both estimators converge to the true gradient.
    g_true = exact_gradient(mdp, theta, T, s0)
    g_tot, var_tot = gradient_statistics(mdp, theta, T, s0, 40_000, "total_return", rng)
    g_rtg, var_rtg = gradient_statistics(mdp, theta, T, s0, 40_000, "reward_to_go", rng)
    _check(
        "total_return estimator is unbiased",
        np.max(np.abs(g_tot - g_true)) < 0.10,
        f"max|err| {np.max(np.abs(g_tot - g_true)):.4f}",
    )
    _check(
        "reward_to_go estimator is unbiased",
        np.max(np.abs(g_rtg - g_true)) < 0.10,
        f"max|err| {np.max(np.abs(g_rtg - g_true)):.4f}",
    )

    # 5. THE PUNCHLINE: same mean, strictly less variance.
    _check(
        "reward_to_go has lower variance than total_return",
        var_rtg < var_tot,
        f"var {var_rtg:.3f} vs {var_tot:.3f}  ({var_tot / max(var_rtg, 1e-12):.2f}x)",
    )

    # 6. Learning: REINFORCE beats the uniform-random policy.
    theta_star, hist = train_reinforce(mdp, T, s0, mode="reward_to_go")
    theta_unif = torch.zeros((mdp.S, mdp.A), dtype=torch.float64, requires_grad=True)
    base_ret = np.mean(
        [np.sum(sample_episode(mdp, theta_unif, T, s0, rng)[2]) for _ in range(2000)]
    )
    _check(
        "REINFORCE improves over the random policy",
        np.mean(hist[-10:]) > base_ret + 1.0,
        f"final {np.mean(hist[-10:]):.3f} vs random {base_ret:.3f} (max {float(T)})",
    )

    print("\ntrue grad  =\n", np.round(g_true, 4))
    print("MC (R_tau) =\n", np.round(g_tot, 4))
    print("MC (rtg)   =\n", np.round(g_rtg, 4))
    print(f"\nvariance:  total_return {var_tot:.3f}   reward_to_go {var_rtg:.3f}"
          f"   -> {var_tot / max(var_rtg, 1e-12):.2f}x reduction")
    print("learned pi =\n", np.round(policy_probs(theta_star).detach().numpy(), 3))
    print("\nAll invariants passed.")


if __name__ == "__main__":
    main()
