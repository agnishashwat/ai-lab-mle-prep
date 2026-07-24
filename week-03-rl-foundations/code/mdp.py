"""
Week 3 Day 1 -- Finite MDPs, value functions, Bellman operators.

A finite MDP is the one setting where every quantity you derived on the whiteboard
has an exact closed-form ground truth, so your iterative methods can be checked to
machine precision. Implement the bodies marked TODO; the self-check harness at the
bottom asserts the invariants (contraction convergence, iterative==linear-solve,
advantage-sums-to-zero, value-iteration fixed point).

Run:  ./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/mdp.py
"""

from __future__ import annotations

import numpy as np


class MDP:
    """A finite, tabular MDP.

    Attributes
    ----------
    P : np.ndarray, shape (S, A, S)
        Transition probabilities. P[s, a, s2] = Pr(s' = s2 | s, a).
        Each P[s, a, :] sums to 1.
    R : np.ndarray, shape (S, A)
        Expected immediate reward r(s, a).
    gamma : float
        Discount factor in [0, 1).
    """

    def __init__(self, P: np.ndarray, R: np.ndarray, gamma: float):
        S, A = R.shape
        assert P.shape == (S, A, S), f"P must be (S,A,S)={(S,A,S)}, got {P.shape}"
        assert np.allclose(P.sum(axis=2), 1.0), "each P[s,a,:] must sum to 1"
        assert 0.0 <= gamma < 1.0, "gamma must be in [0,1)"
        self.P = P
        self.R = R
        self.gamma = gamma
        self.S = S
        self.A = A


# ---------------------------------------------------------------------------
# Policy-conditioned quantities
# ---------------------------------------------------------------------------

def policy_matrices(mdp: MDP, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a stochastic policy pi of shape (S, A), return (P_pi, r_pi) where
        P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a)      -- shape (S, S)
        r_pi[s]     = sum_a pi(a|s) r(s,a)         -- shape (S,)
    These are the ingredients of the linear Bellman system.
    """
    assert pi.shape == (mdp.S, mdp.A)
    assert np.allclose(pi.sum(axis=1), 1.0), "each pi[s,:] must sum to 1"
    
    P = np.einsum('sa,sat->st', pi, mdp.P)
    r = np.einsum('sa,sa->s', pi, mdp.R)

    return (P, r)

def bellman_expectation_backup(mdp: MDP, pi: np.ndarray, V: np.ndarray) -> np.ndarray:
    """One application of the Bellman expectation operator T^pi to V.
        (T^pi V)(s) = sum_a pi(a|s) [ r(s,a) + gamma sum_s' P(s'|s,a) V(s') ]
    Returns the new value vector, shape (S,).
    """
    Q = mdp.R + mdp.gamma * np.einsum('sat,t->sa', mdp.P, V)
    return np.einsum('sa,sa->s', pi, Q)


def policy_evaluation_iterative(
    mdp: MDP, pi: np.ndarray, tol: float = 1e-10, max_iters: int = 100_000
) -> np.ndarray:
    """Evaluate V^pi by iterating the Bellman backup to convergence.
    Stop when ||V_new - V||_inf < tol. Return V^pi, shape (S,).
    """
    V = np.random.rand(mdp.S)
    error = 100 

    curr_iter = 0
    while error > tol and curr_iter < max_iters:
        V_new = bellman_expectation_backup(mdp, pi, V)
        error = np.linalg.norm(V_new-V, ord=np.inf)
        V = V_new
        curr_iter += 1
    return V



def policy_evaluation_linear(mdp: MDP, pi: np.ndarray) -> np.ndarray:
    """Evaluate V^pi exactly via the linear solve V = (I - gamma P_pi)^{-1} r_pi.
    Return V^pi, shape (S,).
    """
    P, r = policy_matrices(mdp, pi)
    A = np.eye(mdp.S) - mdp.gamma * P
    return np.linalg.solve(A, r)


def q_from_v(mdp: MDP, V: np.ndarray) -> np.ndarray:
    """Compute Q(s,a) = r(s,a) + gamma sum_s' P(s'|s,a) V(s') for all (s,a).
    Return Q, shape (S, A).
    """
    return mdp.R + mdp.gamma * np.einsum('sat,t->sa', mdp.P, V)    


def advantage(mdp: MDP, pi: np.ndarray, V: np.ndarray) -> np.ndarray:
    """A(s,a) = Q(s,a) - V(s). Return shape (S, A)."""
    Q = q_from_v(mdp, V)
    return Q - V[:, None]



# ---------------------------------------------------------------------------
# Optimality
# ---------------------------------------------------------------------------

def value_iteration(
    mdp: MDP, tol: float = 1e-10, max_iters: int = 100_000
) -> tuple[np.ndarray, np.ndarray]:
    """Compute V^* via the Bellman optimality backup
        (T* V)(s) = max_a [ r(s,a) + gamma sum_s' P(s'|s,a) V(s') ]
    and extract a greedy deterministic policy pi^* (as a one-hot (S, A) matrix).
    Return (V_star, pi_star).
    """
    V = np.random.rand(mdp.S)
    error = 100 

    curr_iter = 0
    while error > tol and curr_iter < max_iters:
        Q = q_from_v(mdp, V)
        V_new = np.max(Q, axis=-1)
        error = np.linalg.norm(V_new-V, ord=np.inf)
        V = V_new
        curr_iter += 1
    pi_star = np.argmax(Q, axis=-1)
    pi_star_one_hot = np.eye(mdp.A)[pi_star]
    return (V, pi_star_one_hot)


# ---------------------------------------------------------------------------
# Toy environments
# ---------------------------------------------------------------------------

def two_state_chain(gamma: float = 0.9) -> MDP:
    """Minimal 2-state, 2-action MDP for hand-checkable numbers.

    States {0, 1}. Actions {0='stay', 1='switch'}.
    Deterministic dynamics: 'stay' keeps you, 'switch' flips you.
    Reward: being in state 1 pays +1 (on the action that lands you there / keeps you).
    """
    S, A = 2, 2
    P = np.zeros((S, A, S))
    # action 0 = stay
    P[0, 0, 0] = 1.0
    P[1, 0, 1] = 1.0
    # action 1 = switch
    P[0, 1, 1] = 1.0
    P[1, 1, 0] = 1.0
    R = np.zeros((S, A))
    # reward = 1 if the resulting state is 1
    R[0, 1] = 1.0  # 0 --switch--> 1
    R[1, 0] = 1.0  # 1 --stay----> 1
    return MDP(P, R, gamma)


# ---------------------------------------------------------------------------
# Self-check harness
# ---------------------------------------------------------------------------

def _check(name: str, cond: bool) -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def main() -> None:
    rng = np.random.default_rng(0)
    mdp = two_state_chain(gamma=0.9)

    # A random stochastic policy to evaluate.
    pi = rng.random((mdp.S, mdp.A))
    pi /= pi.sum(axis=1, keepdims=True)

    V_iter = policy_evaluation_iterative(mdp, pi)
    V_lin = policy_evaluation_linear(mdp, pi)
    _check("iterative policy eval == linear solve", np.allclose(V_iter, V_lin, atol=1e-8))

    # V^pi is a fixed point of T^pi.
    _check(
        "V^pi is a fixed point of T^pi",
        np.allclose(bellman_expectation_backup(mdp, pi, V_lin), V_lin, atol=1e-8),
    )

    # V(s) = sum_a pi(a|s) Q(s,a).
    Q = q_from_v(mdp, V_lin)
    _check(
        "V(s) == E_{a~pi}[Q(s,a)]",
        np.allclose((pi * Q).sum(axis=1), V_lin, atol=1e-8),
    )

    # E_{a~pi}[A(s,a)] == 0 for every state (the reason V is the natural baseline).
    A = advantage(mdp, pi, V_lin)
    _check("E_{a~pi}[A(s,a)] == 0 per state", np.allclose((pi * A).sum(axis=1), 0.0, atol=1e-8))

    # Value iteration: V^* is a fixed point of the optimality backup; greedy pi^*
    # evaluated gives back V^*; and V^* >= V^pi elementwise.
    V_star, pi_star = value_iteration(mdp)
    Q_star = q_from_v(mdp, V_star)
    _check(
        "V^* is a fixed point of Bellman optimality",
        np.allclose(Q_star.max(axis=1), V_star, atol=1e-8),
    )
    _check(
        "greedy pi^* evaluates to V^*",
        np.allclose(policy_evaluation_linear(mdp, pi_star), V_star, atol=1e-8),
    )
    _check("V^* dominates V^pi", np.all(V_star + 1e-8 >= V_lin))

    print("\nV^pi   =", np.round(V_lin, 4))
    print("V^*    =", np.round(V_star, 4))
    print("pi^*   =\n", np.round(pi_star, 3))
    print("\nAll invariants passed.")


if __name__ == "__main__":
    main()
