"""
Week 3 Day 5 -- KL-constrained RL and the reference policy (Week 3 capstone).

Days 1-4 built an estimator of grad J. Today is about the OBJECTIVE: nobody
maximizes reward in post-training. Everyone maximizes

    J(pi) = E_{a~pi}[ r(s,a) ] - beta * KL( pi(.|s) || pi_ref(.|s) )

and this file is about what that one extra term does. Unlike Days 2-4, most of
today has a CLOSED FORM, so "ground truth" stops being a finite-difference
approximation and becomes algebra:

  (1) THE OPTIMUM IS AN EXPONENTIAL TILT of the reference,
          pi*(a|s) = pi_ref(a|s) exp(r(s,a)/beta) / Z(s),
      with optimal value  J* = beta * log Z(s)  (a soft-max / log-sum-exp value).
      The harness checks the closed form against a BLACK-BOX numerical ascent on
      your own objective -- same trust structure as Day 2's finite-difference
      gradient, one level up: there the ground truth was a gradient, here it is
      an argmax.

  (2) THE FRONTIER IS PARAMETERIZED BY beta, and beta is not an arbitrary knob:
      it is the EXCHANGE RATE between reward and KL. Two exact identities you
      will measure, both consequences of the envelope theorem:
          dJ*/dbeta  = -KL(pi*||pi_ref)
          dE[r]/dKL  = beta          (so the frontier is concave in KL)

  (3) THE ASYMMETRY IS THE WHOLE DESIGN. RLHF penalizes REVERSE KL,
      KL(pi_theta||pi_ref): mode-seeking, and support(pi) subset support(pi_ref),
      i.e. RL can never discover a token the reference gives probability zero.
      Forward KL, KL(pi_ref||pi_theta), is mass-covering and support-expanding.
      Both are measured here on a deliberately bimodal reference.

  (4) YOU CANNOT SUM OVER SEQUENCES, so in practice KL is ESTIMATED from
      samples: k1 (unbiased, high variance, goes NEGATIVE), k2 (biased, always
      >= 0), k3 (unbiased AND always >= 0, via the control variate E[rho] = 1).
      The harness measures all three against the exact tabular KL.

  (5) OVER-OPTIMIZATION IS NOT AN ARTEFACT OF RL, IT IS GOODHART. With a
      mis-specified proxy reward, the proxy rises monotonically as beta falls
      while the TRUE reward peaks at an interior beta and then falls BELOW the
      reference's own true reward. The negative control -- tilting by the true
      reward instead -- is monotone, which is what makes the hump attributable
      to the proxy and not to the tilt.

  (6) THE MECHANISM SHIPPED IN InstructGPT is per-token reward shaping:
          r_t <- r_t - beta ( log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t) )
      whose expected sum is exactly beta * sum_t E[KL_t]. The harness verifies
      that against the exact state distribution, so "KL in the reward implements
      the KL penalty" is a measured claim, not a hand-wave.

Implement the bodies marked TODO. The harness is STAGED: each stage runs
independently, and a stage whose function is still a TODO prints [TODO] and is
skipped rather than killing the run.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-03-rl-foundations/code/kl_rl.py
  one stage           ...  kl_rl.py tilt
  a few stages        ...  kl_rl.py kl tilt objective
  small/fast fixtures ...  kl_rl.py --fast
  list the stages     ...  kl_rl.py --list
"""

from __future__ import annotations

import sys
import traceback
from functools import cached_property

import numpy as np
import torch

from mdp import MDP, policy_matrices, two_state_chain
from reinforce import discount_cumsum, policy_probs
from actor_critic import Batch, batch_gradient_statistics


# ---------------------------------------------------------------------------
# Provided helpers (read these; they are the plumbing, not the exercise)
# ---------------------------------------------------------------------------

def log_softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Week-1's stable log_softmax, in numpy: z - logsumexp(z), max-subtracted.

    Handles -inf entries correctly (an action the reference forbids), provided at
    least one entry per row is finite.
    """
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    zc = z - m
    return zc - np.log(np.exp(zc).sum(axis=axis, keepdims=True))


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax_np(z, axis=axis))


def entropy_np(p: np.ndarray) -> np.ndarray:
    """H(p) = -sum p log p over the last axis, with 0 log 0 = 0."""
    p = np.asarray(p, dtype=np.float64)
    return -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=-1)


def solve_by_ascent(objective, A: int, n_steps: int = 1500, lr: float = 0.05,
                    eps: float = 1e-6, seed: int = 0) -> np.ndarray:
    """Maximize `objective(pi)` over the simplex by BLACK-BOX ascent, returning pi.

    Parameterizes pi = softmax(z) and estimates d objective / dz by central
    finite differences -- deliberately knowing NOTHING about the objective's
    form. This is the Day-2 `exact_gradient` trick reused: the reference must be
    structurally independent of the thing it checks, or agreement proves nothing.

    `objective` takes a (A,) probability vector and returns a float.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 0.01, size=A)
    m = np.zeros(A)
    v = np.zeros(A)
    b1, b2, tiny = 0.9, 0.999, 1e-8
    for step in range(1, n_steps + 1):
        g = np.zeros(A)
        for j in range(A):
            zp, zm = z.copy(), z.copy()
            zp[j] += eps
            zm[j] -= eps
            g[j] = (float(objective(softmax_np(zp)))
                    - float(objective(softmax_np(zm)))) / (2.0 * eps)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        z = z + lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + tiny)
    return softmax_np(z)


def state_distributions(mdp: MDP, pi: np.ndarray, T: int, s0: int) -> np.ndarray:
    """Exact per-step state distribution d_t(s), shape (T, S).

    d_0 = one-hot(s0);  d_{t+1} = d_t P_pi. This is the ground truth that turns
    "the expected sum of per-token KL penalties equals the sequence KL" into a
    number instead of an argument.
    """
    P_pi, _ = policy_matrices(mdp, pi)
    d = np.zeros((T, mdp.S), dtype=np.float64)
    d[0, s0] = 1.0
    for t in range(1, T):
        d[t] = d[t - 1] @ P_pi
    return d


def token_log_ratios(batch: Batch, i: int, theta: torch.Tensor,
                     theta_ref: torch.Tensor) -> np.ndarray:
    """log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t) for episode i, shape (T,).

    The per-token log-ratio: the raw material of every KL estimator, and the
    only thing an LLM RLHF loop ever actually has (you get the logprob of the
    token you sampled, not the whole vocabulary distribution).
    """
    log_pi = torch.log_softmax(theta, dim=-1).detach().numpy()
    log_ref = torch.log_softmax(theta_ref, dim=-1).detach().numpy()
    s, a = batch.states[i], batch.actions[i]
    return log_pi[s, a] - log_ref[s, a]


def fit_shaped_baseline(batch: Batch, n_states: int, theta: torch.Tensor,
                        theta_ref: torch.Tensor, beta: float, gamma: float,
                        estimator: str = "k1") -> np.ndarray:
    """Day-3's tabular critic, but regressed on SHAPED reward-to-go.

    The baseline must target the reward the estimator actually uses; a critic
    fitted to the unshaped return is a Day-3/Day-4 level-mismatch (unbiased,
    but it decentres Psi_t).
    """
    G = np.stack([
        np.asarray(discount_cumsum(shaped_rewards(batch, i, theta, theta_ref,
                                                  beta, estimator), gamma))
        for i in range(batch.n)
    ])
    sums = np.bincount(batch.states.ravel(), G.ravel(), minlength=n_states)
    counts = np.bincount(batch.states.ravel(), minlength=n_states)
    return sums / np.maximum(counts, 1)


def weights_kl_shaped(batch: Batch, i: int, gamma: float, theta: torch.Tensor,
                      theta_ref: torch.Tensor, beta: float,
                      baseline: np.ndarray | None,
                      estimator: str = "k1") -> np.ndarray:
    """Psi_t = gamma^t * ( G_t(shaped) - b(s_t) ), shape (T,).

    Day 3's weight function with the shaped reward substituted in. Note what is
    NOT here: no separate KL term in the loss. The penalty entered through the
    reward, which is exactly what InstructGPT does.
    """
    r_shaped = shaped_rewards(batch, i, theta, theta_ref, beta, estimator)
    G = np.asarray(discount_cumsum(r_shaped, gamma), dtype=np.float64)
    b = 0.0 if baseline is None else np.asarray(baseline)[batch.states[i]]
    return gamma ** np.arange(batch.T) * (G - b)


# ---------------------------------------------------------------------------
# 1. KL divergence, exactly
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) = sum_a p(a) log( p(a) / q(a) ), over the LAST axis.

    p, q : (..., A) rows that each sum to 1.   Returns shape (...).

    Two conventions you must implement explicitly, because they carry the whole
    support story of this day:
      * 0 log 0 = 0. Where p(a) == 0 the term contributes NOTHING, however small
        q(a) is. (p can be zero where q is not -- that is mode-dropping.)
      * p(a) > 0 with q(a) == 0 is +inf. There is no finite-KL way for p to put
        mass where q has none. Return np.inf for that row.

    The trap: `p * np.log(p / q)` gives nan at p == 0 (0 * -inf), and
    `np.where(p > 0, p * np.log(p / q), 0.0)` still evaluates the nan arm and
    still warns -- np.where is not lazy (the Day-3 `beta * f(x)` lesson). Make
    the LOG's argument safe, not the result.
    """
    log_p = np.log(np.where(p > 0, p, 1))
    log_q = np.log(np.where(q > 0, q, 1))
    res = np.where(p == 0, 0, p * (log_p - log_q))
    res = np.where((p > 0) & (q == 0), np.inf, res)
    return res.sum(axis=-1)


# ---------------------------------------------------------------------------
# 2. The closed-form optimum: an exponential tilt of the reference
# ---------------------------------------------------------------------------

def tilted_policy(pi_ref: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """The maximizer of  E_{a~pi}[r] - beta KL(pi||pi_ref), in closed form:

        pi*(a) = pi_ref(a) exp( r(a) / beta ) / Z,     Z = sum_a pi_ref(a) e^{r(a)/beta}

    pi_ref, r : (..., A).   beta > 0.   Returns (..., A) rows summing to 1.

    Compute it in LOG SPACE:  log pi* = log pi_ref + r/beta - logsumexp(...).
    Writing `pi_ref * np.exp(r / beta)` is the bug this stage is built to catch:
    at beta = 1e-3 with rewards of order 1, exp(1000) = inf and the normalized
    result is nan. `log_softmax_np` above already tolerates -inf entries, which
    is how forbidden actions (pi_ref == 0) stay forbidden.

    Three properties worth predicting before you run it:
      * beta -> inf recovers pi_ref (no reward signal survives the division).
      * beta -> 0 concentrates on argmax r WITHIN the support of pi_ref.
      * r -> r + c changes nothing: the constant is absorbed by Z. A reward model
        is only identified up to a per-prompt constant -- which is precisely why
        a per-prompt baseline (GRPO's group mean) is free.
    """
    z = np.where(pi_ref > 0, np.log(pi_ref), -np.inf) + r/beta 
    return softmax_np(z)


def regularized_objective(pi: np.ndarray, pi_ref: np.ndarray, r: np.ndarray,
                          beta: float) -> np.ndarray:
    """J(pi) = E_{a~pi}[ r(a) ] - beta * KL( pi || pi_ref ), over the last axis.

    Returns shape (...). Two lines, given `kl_divergence`. This is the function
    `solve_by_ascent` will maximize numerically, so a bug here shows up as the
    closed form and the numeric solve disagreeing -- and one of them being
    wrong is more likely than both.
    """
    return (pi * r).sum(axis=-1) - beta * kl_divergence(pi, pi_ref)


def forward_kl_objective(pi: np.ndarray, pi_ref: np.ndarray, r: np.ndarray,
                         beta: float) -> np.ndarray:
    """The same objective with the KL ARGUMENTS SWAPPED:

        J_fwd(pi) = E_{a~pi}[ r(a) ] - beta * KL( pi_ref || pi )

    One character of difference, completely different solution -- and no closed
    form, which is half of why RLHF uses the reverse direction. The other half is
    behavioural, and this stage measures it: forward KL is MASS-COVERING (it
    blows up as pi -> 0 anywhere pi_ref has mass, so no mode can be dropped),
    reverse KL is MODE-SEEKING (dropping a mode costs only the mass you moved).
    """
    return (pi * r).sum(axis=-1) - beta * kl_divergence(pi_ref, pi)


# ---------------------------------------------------------------------------
# 3. The reward-KL frontier
# ---------------------------------------------------------------------------

def frontier(pi_ref: np.ndarray, r: np.ndarray,
             betas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep beta and return (expected_reward, kl_to_ref, optimal_value), each (B,).

    pi_ref, r : (A,).  betas : (B,) ASCENDING.  For each beta, build pi* with
    `tilted_policy` and report
        expected_reward[j] = E_{pi*}[r]
        kl_to_ref[j]       = KL(pi* || pi_ref)
        optimal_value[j]   = J(pi*) = E[r] - beta KL
    This is the Pareto curve every RLHF paper plots as "reward vs KL", and every
    point on it is an OPTIMAL policy -- for its own beta. A run that lands
    strictly inside this curve is leaving something on the table; a point above
    it is impossible.
    """
    B = betas.shape[0]
    expected_reward, kl_to_ref, optimal_value = np.zeros(B), np.zeros(B), np.zeros(B)
    for b in range(B):
        beta = betas[b]
        pi_star = tilted_policy(pi_ref, r, beta)
        expected_reward[b] = (pi_star * r).sum(axis=-1)
        kl_to_ref[b] = kl_divergence(pi_star, pi_ref)
        optimal_value[b] = regularized_objective(pi_star, pi_ref, r, beta)

    return (expected_reward, kl_to_ref, optimal_value)


    


def overoptimization(pi_ref: np.ndarray, r_proxy: np.ndarray, r_true: np.ndarray,
                     betas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reward hacking, measured. Returns (proxy_reward, true_reward, kl), each (B,).

    The policy is optimized against `r_proxy` (what the reward model says) and
    scored under BOTH rewards. Concretely: pi*_beta = tilted_policy(pi_ref,
    r_proxy, beta), then report E_{pi*}[r_proxy], E_{pi*}[r_true], and
    KL(pi*||pi_ref).

    You are reproducing the shape of Gao et al.'s over-optimization curves: the
    proxy is monotone in 1/beta by construction (that is what optimization
    means), while the true reward rises, peaks, and then falls -- eventually
    BELOW the reference's own true reward. "Our reward went up" is therefore not
    evidence of anything.
    """
    
    B = betas.shape[0]
    proxy_reward, true_reward, kl = np.zeros(B), np.zeros(B), np.zeros(B)
    for b in range(B):
        beta = betas[b]
        pi_star_proxy = tilted_policy(pi_ref, r_proxy, beta)
        proxy_reward[b] = (pi_star_proxy * r_proxy).sum(axis=-1)
        true_reward[b] = (pi_star_proxy * r_true).sum(axis=-1)
        kl[b] = kl_divergence(pi_star_proxy, pi_ref)

    return (proxy_reward, true_reward, kl)


# ---------------------------------------------------------------------------
# 4. Estimating KL from samples (you cannot sum over sequences)
# ---------------------------------------------------------------------------

def kl_estimators(log_ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schulman's k1, k2, k3 -- per-sample estimators of KL(pi || pi_ref).

    `log_ratio` = log pi(a) - log pi_ref(a) for actions a SAMPLED FROM pi.
    Returns (k1, k2, k3), each the same shape as the input.

    Write rho = pi_ref(a)/pi(a) = exp(-log_ratio), so E_{a~pi}[rho] = 1 exactly.

        k1 = log_ratio                 unbiased; the definition of KL as a mean.
                                       Individual samples go NEGATIVE, so a
                                       logged "KL" can be negative, which is
                                       alarming and correct.
        k2 = 0.5 * log_ratio^2         always >= 0, low variance, BIASED.
        k3 = rho - 1 - log(rho)        = expm1(-log_ratio) + log_ratio.
                                       Unbiased (E[rho - 1] = 0 is a control
                                       variate) AND >= 0 pointwise, since
                                       x - 1 - log x >= 0 with equality at x = 1.
                                       This is what TRL/GRPO use.

    Use np.expm1 for k3: at log_ratio ~ 1e-8, `np.exp(-lr) - 1` loses every
    significant digit to cancellation and can go negative, breaking the one
    property k3 exists for.
    """
    k1 = log_ratio
    k2 = 0.5 * log_ratio ** 2
    k3 = np.expm1(-log_ratio) + log_ratio
    return (k1, k2, k3)


# ---------------------------------------------------------------------------
# 5. Per-token KL as reward shaping (the InstructGPT mechanism)
# ---------------------------------------------------------------------------

def shaped_rewards(batch: Batch, i: int, theta: torch.Tensor,
                   theta_ref: torch.Tensor, beta: float,
                   estimator: str = "k1") -> np.ndarray:
    """The KL penalty folded into the REWARD, per token. Returns (T,).

        estimator == "k1":  r_t - beta * ( log pi_theta(a_t|s_t) - log pi_ref(...) )
        estimator == "k3":  r_t - beta * k3_t

    Use `token_log_ratios` and `kl_estimators`; `batch.rewards[i]` is r_t.

    Why this is the shipped mechanism rather than a loss term: the environment
    reward is a single scalar at EOS, but the KL penalty is available at EVERY
    token, so shaping this way hands the critic a dense signal (Day 4's credit
    assignment problem, attacked from the reward side instead of the estimator
    side). And in expectation the sum of the per-token penalties is exactly
    beta * sum_t E[ KL_t ] -- the sequence-level penalty. The harness checks that
    against `state_distributions`.

    Note the sign asymmetry that k3 fixes. A token the policy makes LESS likely
    than the reference has a NEGATIVE log-ratio, so the k1 "penalty" is a positive
    BONUS on that token -- it pays the policy for deviating, on roughly half of
    them. k3 is >= 0 pointwise with the same expectation, so it is a penalty
    everywhere and strictly lower variance.
    """
    log_ratio = token_log_ratios(batch, i, theta, theta_ref)
    k1, _, k3 = kl_estimators(log_ratio)
    if estimator=='k1':
        return batch.rewards[i] - beta * k1
    elif estimator=='k3':
        return batch.rewards[i] - beta * k3
    else:
        raise ValueError(estimator)


    
    


def train_kl(mdp: MDP, T: int, s0: int, theta_ref: torch.Tensor, beta: float,
             n_iters: int = 200, batch_size: int = 32, lr: float = 0.1,
             estimator: str = "k1", seed: int = 0
             ) -> tuple[torch.Tensor, list[float], list[float]]:
    """REINFORCE with a fitted baseline, on KL-SHAPED rewards. Third rep of the
    loop; this time the only new thing is which reward you feed it.

    Returns (theta, return_history, kl_history) where
      return_history[k] = mean TRUE (unshaped) total return at iteration k
      kl_history[k]     = sum_t E_{d_t}[ KL(pi_theta(.|s) || pi_ref(.|s)) ], the
                          EXACT sequence KL (use state_distributions + your
                          kl_divergence, not an estimator -- this is tabular, so
                          measure the thing itself and keep the estimators for
                          the stage that studies them).

    Structure, deliberately spelled out because this is the invariant list you
    have now built three times:
      * theta = theta_ref.clone() -- start AT the reference, as post-training
        does. requires_grad_(True).
      * per iteration: Batch(mdp, theta, T, s0, n=batch_size, rng=<seeded>)
        -- pass batch_size BY KEYWORD (Day 4's silent positional bug),
        baseline = fit_shaped_baseline(...),
        weight_fn = lambda B, i, gm: weights_kl_shaped(B, i, gm, theta,
                                     theta_ref, beta, baseline, estimator),
        g, _ = batch_gradient_statistics(theta, batch, mdp.gamma, weight_fn),
        ASCEND: with torch.no_grad(): theta += lr * torch.as_tensor(g, dtype=...)
        -- `theta += numpy_array` silently rebinds theta to a non-leaf-free
        fresh tensor with requires_grad=False (Day 3).
      * log FLOATS, never tensors or arrays (Day 3 / Day 4).

    Predict before running: beta = 0 must reproduce Day 3/Day 4's optimum
    ("reach & stay", return ~7.9); large beta must leave theta ~ theta_ref.
    """
    theta = theta_ref.clone().requires_grad_(True)
    rng = np.random.default_rng(seed)
    return_history, kl_history = [], []
    for iter in range(n_iters):
        batch = Batch(mdp, theta, T, s0, n=batch_size, rng=rng)
        baseline = fit_shaped_baseline(batch, mdp.S, theta, theta_ref, beta, mdp.gamma, estimator)
        weight_fn = lambda B, i, gm: weights_kl_shaped(B, i, gm, theta, theta_ref, beta, baseline, estimator)
        g, _ = batch_gradient_statistics(theta, batch, mdp.gamma, weight_fn)
        state_distribution = state_distributions(mdp, torch.softmax(theta, dim=-1).detach().numpy(), T, s0)
        return_history.append(batch.total_returns.mean())
        kl_div = kl_divergence(torch.softmax(theta, dim=-1).detach().numpy(), torch.softmax(theta_ref, dim=-1).numpy())
        kl_div_state_weighted = (state_distribution * kl_div).sum()
        kl_history.append(kl_div_state_weighted)
        with torch.no_grad():
            theta += lr * torch.as_tensor(g, dtype=theta.dtype)
    return (theta, return_history, kl_history)





# ---------------------------------------------------------------------------
# Fixtures: a "next-token" bandit
# ---------------------------------------------------------------------------

def token_bandit(A: int = 12, seed: int = 0, gap: float = 0.4,
                 damp: float = 6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """A one-step bandit standing in for "one next-token decision".

    Returns (pi_ref, r_true, r_proxy, hack). `r_proxy` agrees with `r_true`
    everywhere except on the action `hack`, where it scores `gap` ABOVE the true
    best action while its true reward is the WORST of all -- a reward model that
    has learned to love bullet points. The reference gives `hack` almost no mass
    (`damp` nats of logit suppression), so the tilt only finds it once beta is
    small: the two effects live on different scales, which is what makes the
    over-optimization hump appear at an interior beta instead of immediately.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, size=A)
    r_true = rng.normal(0.0, 1.0, size=A)
    hack = int(np.argmin(r_true))
    logits[hack] = logits.min() - damp
    pi_ref = softmax_np(logits)
    r_proxy = r_true.copy()
    r_proxy[hack] = float(r_true.max()) + gap
    return pi_ref, r_true, r_proxy, hack


def bimodal_bandit() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A FULL-SUPPORT reference with two modes and a reward that likes one of them.

    Returns (pi_ref, r, mode_hi, mode_lo) with index arrays for the two modes.
    Full support on purpose: this fixture isolates mode-seeking vs mass-covering
    from the separate question of support, which `forbidden_bandit` handles.
    """
    mode_hi = np.array([0, 1, 2])
    mode_lo = np.array([3, 4, 5])
    pi_ref = np.array([0.18, 0.20, 0.12, 0.14, 0.22, 0.14])
    r = np.array([1.6, 2.0, 1.4, -0.6, -0.4, -0.5])
    return pi_ref, r, mode_hi, mode_lo


def forbidden_bandit() -> tuple[np.ndarray, np.ndarray, int]:
    """A reference with a hole: action 3 has pi_ref == 0 and the highest reward.

    Returns (pi_ref, r, forbidden). This is the "token the base model never
    emits": reverse KL can never reach it at any beta, forward KL can and does.
    """
    pi_ref = np.array([0.5, 0.3, 0.2, 0.0])
    r = np.array([0.5, 0.0, 0.0, 2.5])
    return pi_ref, r, 3


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
    """Shared, lazily-built fixtures, each from its own seeded generator."""

    def __init__(self, fast: bool = False):
        torch.manual_seed(0)
        self.fast = fast
        self.mdp = two_state_chain(gamma=0.9)
        self.gamma = self.mdp.gamma
        self.T, self.s0 = 8, 0
        self.pi_ref, self.r_true, self.r_proxy, self.hack = token_bandit()
        self.A = len(self.pi_ref)
        # Bandit sampling is cheap enough to keep exact in --fast mode; the
        # statistical checks are calibrated to it (a 10x smaller sample would
        # make "k2 is biased" undetectable, which is a real lesson but not this
        # stage's).
        self.n_mc = 200_000
        self.n_roll = 400 if fast else 2_000
        self.n_iters = 60 if fast else 200
        self.ascent_steps = 400 if fast else 1500

        # A deliberately mediocre reference policy for the MDP: it prefers the
        # WRONG action in state 0, so post-training has something to fix.
        self.theta_ref = torch.tensor([[0.8, -0.8], [0.2, -0.2]], dtype=torch.float64)

    @cached_property
    def theta_pol(self) -> torch.Tensor:
        """A policy distinct from the reference, for the shaping checks."""
        return torch.tensor([[-0.3, 0.9], [0.6, -0.4]], dtype=torch.float64)

    @cached_property
    def mc_batch(self) -> Batch:
        print(f"  ... rolling out {self.n_roll} episodes")
        return Batch(self.mdp, self.theta_pol, self.T, self.s0,
                     n=self.n_roll, rng=np.random.default_rng(0))



# ---- individual stages ----------------------------------------------------

def stage_kl(ctx: Ctx) -> None:
    """kl_divergence"""
    p = np.array([0.5, 0.25, 0.25])
    q = np.array([0.1, 0.3, 0.6])
    kl = float(kl_divergence(p, q))
    hand = (0.5 * np.log(0.5 / 0.1) + 0.25 * np.log(0.25 / 0.3)
            + 0.25 * np.log(0.25 / 0.6))
    _check("matches the hand computation", np.isclose(kl, hand), f"{kl:.6f} vs {hand:.6f}")
    _check("KL(p||p) == 0 exactly", float(kl_divergence(p, p)) == 0.0)
    _check("KL >= 0 (Gibbs)", kl > 0 and float(kl_divergence(q, p)) > 0)
    _check("KL is NOT symmetric", not np.isclose(kl, float(kl_divergence(q, p))),
           f"fwd {kl:.4f} rev {float(kl_divergence(q, p)):.4f}")

    # 0 log 0 = 0: p may drop an action q likes, for a FINITE price.
    p0 = np.array([0.7, 0.3, 0.0])
    _check("p(a) == 0 contributes 0, not nan", np.isfinite(float(kl_divergence(p0, q))),
           f"got {float(kl_divergence(p0, q)):.4f}")
    # ...but q may not drop an action p likes.
    q0 = np.array([0.6, 0.4, 0.0])
    _check("q(a) == 0 with p(a) > 0 is +inf", np.isinf(float(kl_divergence(p, q0))))
    _check("...and is finite in the other direction",
           np.isfinite(float(kl_divergence(q0, p))))

    # Batched rows, and the identity that makes Day 3 a special case of Day 5.
    P = np.stack([p, q, p0])
    _check("vectorizes over leading axes", kl_divergence(P, q[None, :]).shape == (3,),
           f"got {kl_divergence(P, q[None, :]).shape}")
    A = 5
    unif = np.full(A, 1.0 / A)
    rng = np.random.default_rng(0)
    pr = softmax_np(rng.normal(size=(4, A)))
    _check("KL(p||uniform) == log A - H(p)  [entropy bonus IS KL-to-uniform]",
           np.allclose(kl_divergence(pr, unif[None, :]), np.log(A) - entropy_np(pr)),
           f"max|diff| {np.max(np.abs(kl_divergence(pr, unif[None, :]) - (np.log(A) - entropy_np(pr)))):.2e}")


def stage_tilt(ctx: Ctx) -> None:
    """tilted_policy"""
    ref = np.array([0.5, 0.5])
    r = np.array([1.0, 0.0])
    pi = np.asarray(tilted_policy(ref, r, beta=1.0), dtype=np.float64)
    _check("returns the input shape", pi.shape == (2,), f"got {pi.shape}")
    _check("rows sum to 1", np.isclose(pi.sum(), 1.0), f"sum {pi.sum():.12f}")
    sig = 1.0 / (1.0 + np.exp(-1.0))
    _check("hand case: uniform ref, r=[1,0], beta=1 -> [sigmoid(1), 1-sigmoid(1)]",
           np.allclose(pi, [sig, 1 - sig]), f"got {np.round(pi, 6)}")

    ref2, r2 = ctx.pi_ref, ctx.r_true
    _check("beta -> inf recovers pi_ref",
           np.allclose(tilted_policy(ref2, r2, beta=1e8), ref2, atol=1e-6),
           f"max|diff| {np.max(np.abs(tilted_policy(ref2, r2, 1e8) - ref2)):.2e}")
    greedy = np.zeros_like(r2)
    greedy[np.argmax(r2)] = 1.0
    _check("beta -> 0 concentrates on argmax r",
           np.allclose(tilted_policy(ref2, r2, beta=1e-3), greedy, atol=1e-6),
           f"p(argmax) = {tilted_policy(ref2, r2, 1e-3)[np.argmax(r2)]:.8f}")

    # NUMERICAL: the naive pi_ref * exp(r/beta) is inf/inf = nan here.
    small = tilted_policy(ref2, r2, beta=1e-3)
    _check("survives beta = 1e-3 without overflow (log-space required)",
           np.all(np.isfinite(small)) and np.isclose(small.sum(), 1.0),
           f"finite={np.all(np.isfinite(small))} sum={small.sum():.6f}")
    big = tilted_policy(ref2, r2 * 500.0, beta=1.0)
    _check("survives rewards of magnitude 500 without overflow",
           np.all(np.isfinite(big)) and np.isclose(big.sum(), 1.0))

    _check("invariant to r -> r + c (Z absorbs it; RM identified up to a constant)",
           np.allclose(tilted_policy(ref2, r2 + 7.3, beta=0.7),
                       tilted_policy(ref2, r2, beta=0.7)))

    # Support: reverse KL cannot reach what the reference forbids, at ANY beta.
    ref3, r3, forb = forbidden_bandit()
    for b in (2.0, 0.5, 0.01):
        pi3 = tilted_policy(ref3, r3, beta=b)
        _check(f"support(pi*) subset support(pi_ref) at beta={b} "
               f"(the best action, r={r3[forb]}, stays unreachable)",
               pi3[forb] == 0.0 and np.isclose(pi3.sum(), 1.0),
               f"pi*[{forb}] = {pi3[forb]:.3e}")


def stage_objective(ctx: Ctx) -> None:
    """regularized_objective + the value identity J* = beta log Z"""
    ref, r = ctx.pi_ref, ctx.r_true
    beta = 0.8
    _check("J(pi_ref) == E_ref[r] (KL term vanishes)",
           np.isclose(float(regularized_objective(ref, ref, r, beta)), float(ref @ r)),
           f"{float(regularized_objective(ref, ref, r, beta)):.6f} vs {float(ref @ r):.6f}")

    pi_star = tilted_policy(ref, r, beta)
    J_star = float(regularized_objective(pi_star, ref, r, beta))
    logZ = float(np.log(np.sum(ref * np.exp(r / beta))))
    _check("J(pi*) == beta * log Z  (the soft-max value function)",
           np.isclose(J_star, beta * logZ), f"{J_star:.8f} vs {beta * logZ:.8f}")

    # pi* is the ARGMAX: no random policy beats it.
    rng = np.random.default_rng(5)
    cand = softmax_np(rng.normal(0.0, 1.5, size=(2000, ctx.A)))
    Js = np.asarray([float(regularized_objective(c, ref, r, beta)) for c in cand])
    _check("no random policy beats pi*", np.all(Js <= J_star + 1e-12),
           f"best random {Js.max():.6f} vs J* {J_star:.6f}")
    # ...and neither does a small perturbation of pi* in any direction.
    pert = softmax_np(np.log(pi_star) + rng.normal(0.0, 0.02, size=(500, ctx.A)))
    Jp = np.asarray([float(regularized_objective(c, ref, r, beta)) for c in pert])
    _check("pi* is a local max (500 small perturbations, all worse)",
           np.all(Jp <= J_star + 1e-12), f"best perturbation {Jp.max() - J_star:.2e} above")


def stage_numeric(ctx: Ctx) -> None:
    """the closed form vs a black-box numerical solve of YOUR objective"""
    ref, r = ctx.pi_ref, ctx.r_true
    for beta in (0.4, 1.0, 3.0):
        pi_hat = solve_by_ascent(
            lambda p: float(regularized_objective(p, ref, r, beta)),
            ctx.A, n_steps=ctx.ascent_steps, seed=0)
        pi_star = tilted_policy(ref, r, beta)
        tol = 5e-3 if ctx.fast else 5e-4
        _check(f"closed form == numeric argmax at beta={beta}",
               np.max(np.abs(pi_hat - pi_star)) < tol,
               f"max|diff| {np.max(np.abs(pi_hat - pi_star)):.2e}")


def stage_frontier(ctx: Ctx) -> None:
    """frontier: monotonicity, the beta = dE[r]/dKL identity, and the envelope theorem"""
    ref, r = ctx.pi_ref, ctx.r_true
    betas = 0.05 * 2.0 ** np.arange(13)
    Er, kl, J = (np.asarray(x, dtype=np.float64) for x in frontier(ref, r, betas))
    _check("returns three arrays of shape (B,)",
           Er.shape == kl.shape == J.shape == betas.shape,
           f"{Er.shape} {kl.shape} {J.shape}")

    _check("E[r] DEcreases with beta (less optimization pressure)",
           np.all(np.diff(Er) < 0), f"diffs {np.round(np.diff(Er), 4)}")
    _check("KL DEcreases with beta (tighter leash)", np.all(np.diff(kl) < 0))
    _check("largest beta ~ the reference", np.isclose(Er[-1], ref @ r, atol=2e-2)
           and kl[-1] < 5e-2, f"E[r]={Er[-1]:.4f} vs ref {float(ref @ r):.4f}, KL={kl[-1]:.4f}")
    _check("smallest beta ~ greedy", Er[0] > r.max() - 1e-2, f"E[r]={Er[0]:.4f} vs max r {r.max():.4f}")

    logZ = np.array([np.log(np.sum(ref * np.exp(r / b))) for b in betas])
    _check("J == beta log Z at every beta", np.allclose(J, betas * logZ),
           f"max|diff| {np.max(np.abs(J - betas * logZ)):.2e}")

    # dE[r]/dKL == beta: beta IS the exchange rate, exactly.
    for b in (0.3, 1.0, 4.0):
        h = 1e-5 * b
        Ep, klp, _ = frontier(ref, r, np.array([b + h]))
        Em, klm, _ = frontier(ref, r, np.array([b - h]))
        slope = float((Ep[0] - Em[0]) / (klp[0] - klm[0]))
        _check(f"dE[r]/dKL == beta at beta={b} (the multiplier IS the exchange rate)",
               np.isclose(slope, b, rtol=1e-4), f"slope {slope:.6f} vs beta {b}")
    slope_grid = np.diff(Er) / np.diff(kl)
    _check("frontier is concave in KL (slope shrinks as KL grows)",
           np.all(np.diff(slope_grid[::-1]) < 0),
           f"slopes {np.round(slope_grid[::-1], 3)}")

    # Envelope theorem: dJ*/dbeta = -KL(pi*||pi_ref).
    for b in (0.3, 1.0, 4.0):
        h = 1e-5 * b
        Jp = float(regularized_objective(tilted_policy(ref, r, b + h), ref, r, b + h))
        Jm = float(regularized_objective(tilted_policy(ref, r, b - h), ref, r, b - h))
        dJ = (Jp - Jm) / (2 * h)
        kl_b = float(kl_divergence(tilted_policy(ref, r, b), ref))
        _check(f"dJ*/dbeta == -KL at beta={b} (envelope theorem)",
               np.isclose(dJ, -kl_b, rtol=1e-4), f"{dJ:.6f} vs {-kl_b:.6f}")


def stage_asymmetry(ctx: Ctx) -> None:
    """forward vs reverse KL: mode-seeking, mass-covering, and support"""
    ref, r, mode_hi, mode_lo = bimodal_bandit()
    beta = 0.5
    A = len(ref)

    pi_rev = np.asarray(tilted_policy(ref, r, beta), dtype=np.float64)
    pi_fwd = solve_by_ascent(lambda p: float(forward_kl_objective(p, ref, r, beta)),
                             A, n_steps=ctx.ascent_steps, seed=1)
    _check("J_fwd(pi_ref) == E_ref[r] (both objectives agree at the reference)",
           np.isclose(float(forward_kl_objective(ref, ref, r, beta)), float(ref @ r)))

    lo_ref = float(ref[mode_lo].sum())
    lo_rev, lo_fwd = float(pi_rev[mode_lo].sum()), float(pi_fwd[mode_lo].sum())
    print(f"  mass on the LOW-reward mode:  reference {lo_ref:.4f}"
          f" | forward-KL optimum {lo_fwd:.4f} | reverse-KL optimum {lo_rev:.4f}")
    _check("reverse KL DROPS the low-reward mode (mode-seeking)",
           lo_rev < 0.25 * lo_fwd, f"reverse {lo_rev:.4f} vs forward {lo_fwd:.4f}")
    _check("forward KL keeps a real share of it (mass-covering)",
           lo_fwd > 0.06, f"forward mass on the bad mode {lo_fwd:.4f}")
    _check("forward KL puts mass on EVERY action pi_ref uses",
           np.all(pi_fwd > 5e-3), f"min {pi_fwd.min():.2e}")
    _check("the forward optimum is worse under the REVERSE objective",
           float(regularized_objective(pi_fwd, ref, r, beta))
           < float(regularized_objective(pi_rev, ref, r, beta)))
    _check("the reverse optimum is worse under the FORWARD objective",
           float(forward_kl_objective(pi_rev, ref, r, beta))
           < float(forward_kl_objective(pi_fwd, ref, r, beta)))

    # Support, the consequence that matters for RLHF.
    ref2, r2, forb = forbidden_bandit()
    q_rev = np.asarray(tilted_policy(ref2, r2, beta), dtype=np.float64)
    q_fwd = solve_by_ascent(lambda p: float(forward_kl_objective(p, ref2, r2, beta)),
                            len(ref2), n_steps=ctx.ascent_steps, seed=2)
    print(f"  mass on the action the reference NEVER emits (its best-rewarded one):"
          f" reverse {q_rev[forb]:.4f} | forward {q_fwd[forb]:.4f}")
    _check("reverse KL can never leave the reference's support",
           q_rev[forb] == 0.0)
    _check("forward KL happily does (and takes most of the mass)",
           q_fwd[forb] > 0.5, f"pi_fwd[forbidden] = {q_fwd[forb]:.4f}")


def stage_estimators(ctx: Ctx) -> None:
    """k1 / k2 / k3 against the exact KL"""
    lr0 = np.array([0.0, 0.5, -0.5, 3.0, -3.0])
    k1, k2, k3 = (np.asarray(x, dtype=np.float64) for x in kl_estimators(lr0))
    _check("shapes preserved", k1.shape == k2.shape == k3.shape == lr0.shape)
    _check("all three vanish at log_ratio = 0",
           k1[0] == 0.0 and k2[0] == 0.0 and np.isclose(k3[0], 0.0, atol=1e-15))
    _check("k1 == log_ratio (so it takes both signs)",
           np.allclose(k1, lr0) and k1.min() < 0)
    _check("k2 == 0.5 log_ratio^2 >= 0", np.allclose(k2, 0.5 * lr0 ** 2))
    hand = np.exp(-lr0) - 1.0 + lr0
    _check("k3 == rho - 1 - log rho, matched by hand", np.allclose(k3, hand),
           f"max|diff| {np.max(np.abs(k3 - hand)):.2e}")
    rng = np.random.default_rng(9)
    wide = rng.normal(0.0, 2.0, size=100_000)
    _check("k3 >= 0 pointwise over 100k random log-ratios",
           np.all(np.asarray(kl_estimators(wide)[2]) >= 0.0),
           f"min {np.min(np.asarray(kl_estimators(wide)[2])):.2e}")
    tiny = np.array([1e-9, -1e-9, 1e-12])
    _check("k3 stays >= 0 at |log_ratio| ~ 1e-12 (expm1, not exp - 1)",
           np.all(np.asarray(kl_estimators(tiny)[2]) >= 0.0),
           f"got {np.asarray(kl_estimators(tiny)[2])}")

    # Monte Carlo against the exact tabular KL, across FOUR KL regimes. The
    # variance ordering is regime-dependent, which is the point of the sweep.
    ref = ctx.pi_ref
    rng = np.random.default_rng(3)
    stats = {}
    print("   beta   exact KL   bias k1   bias k2   bias k3    var k1    var k3   ratio  neg(k1)")
    for beta in (8.0, 2.0, 1.0, 0.5):
        pi = np.asarray(tilted_policy(ref, ctx.r_true, beta), dtype=np.float64)
        exact = float(kl_divergence(pi, ref))
        a = rng.choice(ctx.A, size=ctx.n_mc, p=pi)
        log_ratio = np.log(pi[a]) - np.log(ref[a])
        K1, K2, K3 = (np.asarray(x, dtype=np.float64) for x in kl_estimators(log_ratio))
        st = dict(exact=exact,
                  b1=K1.mean() - exact, b2=K2.mean() - exact, b3=K3.mean() - exact,
                  v1=K1.var(), v3=K3.var(),
                  se1=K1.std(ddof=1) / np.sqrt(len(K1)),
                  se3=K3.std(ddof=1) / np.sqrt(len(K3)),
                  neg=(K1 < 0).mean())
        stats[beta] = st
        print(f"  {beta:5.1f}   {exact:8.5f}  {st['b1']:+8.5f}  {st['b2']:+8.5f}  "
              f"{st['b3']:+8.5f}  {st['v1']:8.4f}  {st['v3']:8.4f}  "
              f"{st['v1'] / st['v3']:6.2f}   {st['neg']:6.1%}")

    for beta, st in stats.items():
        _check(f"k1 unbiased at beta={beta}", abs(st["b1"]) < 4 * st["se1"],
               f"{st['b1']:+.5f} = {abs(st['b1']) / st['se1']:.1f} se")
        _check(f"k3 unbiased at beta={beta}", abs(st["b3"]) < 4 * st["se3"],
               f"{st['b3']:+.5f} = {abs(st['b3']) / st['se3']:.1f} se")
        _check(f"k1 samples go negative at beta={beta} (a logged KL reads below 0)",
               st["neg"] > 0.1, f"{st['neg']:.1%}")

    _check("k2 is DETECTABLY biased once the KL is real (beta<=1)",
           all(abs(stats[b]["b2"]) > 5 * stats[b]["se1"] for b in (1.0, 0.5)),
           f"{abs(stats[1.0]['b2']) / stats[1.0]['se1']:.1f} se, "
           f"{abs(stats[0.5]['b2']) / stats[0.5]['se1']:.1f} se")
    _check("...and INVISIBLE at small KL, which is why it survives in the wild",
           abs(stats[8.0]["b2"]) < 4 * stats[8.0]["se1"],
           f"{abs(stats[8.0]['b2']) / stats[8.0]['se1']:.1f} se at KL={stats[8.0]['exact']:.5f}")
    _check("var(k3) << var(k1) in the small-KL regime RLHF actually runs in",
           stats[8.0]["v1"] / stats[8.0]["v3"] > 10
           and stats[2.0]["v1"] / stats[2.0]["v3"] > 10,
           f"{stats[8.0]['v1'] / stats[8.0]['v3']:.0f}x at KL={stats[8.0]['exact']:.4f}, "
           f"{stats[2.0]['v1'] / stats[2.0]['v3']:.0f}x at KL={stats[2.0]['exact']:.4f}")
    _check("but the ordering REVERSES at large KL (k3's rho is a ratio)",
           stats[0.5]["v3"] > stats[0.5]["v1"],
           f"var k3 {stats[0.5]['v3']:.3f} > var k1 {stats[0.5]['v1']:.3f} "
           f"at KL={stats[0.5]['exact']:.3f}")


def _fail_fast(ctx: Ctx) -> None:
    """Touch every function the heavy stages need, on a 2-episode batch."""
    tiny = Batch(ctx.mdp, ctx.theta_pol, T=2, s0=ctx.s0, n=2,
                 rng=np.random.default_rng(0))
    shaped_rewards(tiny, 0, ctx.theta_pol, ctx.theta_ref, 0.1)
    kl_divergence(np.array([0.5, 0.5]), np.array([0.5, 0.5]))


def stage_shaping(ctx: Ctx) -> None:
    """shaped_rewards: per-token KL in the reward == the sequence KL, in expectation"""
    _fail_fast(ctx)
    batch = ctx.mc_batch
    beta = 0.3

    raw = batch.rewards[0]
    _check("beta = 0 leaves the reward untouched",
           np.allclose(shaped_rewards(batch, 0, ctx.theta_pol, ctx.theta_ref, 0.0), raw))
    _check("theta_ref == theta leaves the reward untouched",
           np.allclose(shaped_rewards(batch, 0, ctx.theta_pol, ctx.theta_pol, beta), raw))
    sh = np.asarray(shaped_rewards(batch, 0, ctx.theta_pol, ctx.theta_ref, beta))
    _check("returns shape (T,)", sh.shape == (ctx.T,), f"got {sh.shape}")
    expected = raw - beta * token_log_ratios(batch, 0, ctx.theta_pol, ctx.theta_ref)
    _check("k1 shaping matches r_t - beta * log_ratio_t", np.allclose(sh, expected))

    # The claim: E[ sum_t (r_t - shaped_t) ] == beta * sum_t E_{d_t}[KL_t].
    pi = policy_probs(ctx.theta_pol).detach().numpy()
    ref = policy_probs(ctx.theta_ref).detach().numpy()
    d = state_distributions(ctx.mdp, pi, ctx.T, ctx.s0)
    kl_per_state = np.asarray(kl_divergence(pi, ref), dtype=np.float64)
    exact_seq_kl = float((d @ kl_per_state).sum())

    var = {}
    for est in ("k1", "k3"):
        pen = np.array([
            float(np.sum(batch.rewards[i]
                         - np.asarray(shaped_rewards(batch, i, ctx.theta_pol,
                                                     ctx.theta_ref, beta, est))))
            for i in range(batch.n)
        ])
        se = pen.std(ddof=1) / np.sqrt(len(pen))
        var[est] = float(pen.var())
        print(f"  {est}: mean penalty {pen.mean():.5f} (se {se:.5f}, var {pen.var():.5f})"
              f" | beta * exact seq KL = {beta * exact_seq_kl:.5f}")
        _check(f"{est} penalty sum is unbiased for beta * sequence KL",
               abs(pen.mean() - beta * exact_seq_kl) < 4 * se,
               f"bias {pen.mean() - beta * exact_seq_kl:+.5f}")
    _check("k3 shaping has strictly lower variance at the same mean",
           var["k3"] < var["k1"],
           f"{var['k3']:.5f} vs {var['k1']:.5f} ({var['k1'] / var['k3']:.2f}x)")

    p1 = np.concatenate([batch.rewards[i] - np.asarray(
        shaped_rewards(batch, i, ctx.theta_pol, ctx.theta_ref, beta, "k1"))
        for i in range(50)])
    p3 = np.concatenate([batch.rewards[i] - np.asarray(
        shaped_rewards(batch, i, ctx.theta_pol, ctx.theta_ref, beta, "k3"))
        for i in range(50)])
    _check("k1 PAYS a bonus on some tokens (penalty < 0)", (p1 < 0).mean() > 0.1,
           f"{(p1 < 0).mean():.1%} of tokens rewarded for deviating")
    _check("k3 never does (>= 0 on every token, same expectation)", np.all(p3 >= 0.0))


def stage_train(ctx: Ctx) -> None:
    """train_kl: the beta sweep, in an actual RL loop"""
    _fail_fast(ctx)
    betas = [0.0, 0.2, 1.0, 5.0]
    rets, kls, thetas = [], [], []
    for beta in betas:
        th, rh, kh = train_kl(ctx.mdp, ctx.T, ctx.s0, ctx.theta_ref, beta,
                              n_iters=ctx.n_iters, seed=0)
        rets.append(float(np.mean(rh[-10:])))
        kls.append(float(np.mean(kh[-10:])))
        thetas.append(th)
        print(f"  beta={beta:<5} true return {rets[-1]:.3f}   sequence KL to ref {kls[-1]:.4f}")

    _check("return DEcreases with beta", all(np.diff(rets) < 0),
           f"returns {np.round(rets, 3)}")
    _check("KL to the reference DEcreases with beta", all(np.diff(kls) < 0),
           f"KLs {np.round(kls, 4)}")

    pi0 = policy_probs(thetas[0]).detach().numpy()
    thresh = 0.90 if ctx.fast else 0.95
    _check("beta = 0 reproduces Day 3/Day 4's optimum ('reach & stay')",
           pi0[0, 1] > thresh and pi0[1, 0] > thresh, f"pi*=\n{np.round(pi0, 4)}")
    _check("beta = 0 hits the Day-3/Day-4 return of ~7.9",
           rets[0] > (7.4 if ctx.fast else 7.7), f"got {rets[0]:.3f}")

    pi_big = policy_probs(thetas[-1]).detach().numpy()
    ref = policy_probs(ctx.theta_ref).detach().numpy()
    _check("large beta leaves the policy near the reference",
           np.max(np.abs(pi_big - ref)) < 0.15,
           f"max|diff| {np.max(np.abs(pi_big - ref)):.4f}")

    ref_return = float(Batch(ctx.mdp, ctx.theta_ref, ctx.T, ctx.s0, n=2000,
                             rng=np.random.default_rng(17)).total_returns.mean())
    print(f"  the reference policy's own true return: {ref_return:.3f}")
    _check("every beta improves on the reference (a leash is not a muzzle)",
           all(x > ref_return - 0.3 for x in rets), f"returns {np.round(rets, 3)}")
    _check("beta = 0 improves on it by a wide margin",
           rets[0] > ref_return + 1.0, f"{rets[0]:.3f} vs {ref_return:.3f}")


def stage_hacking(ctx: Ctx) -> None:
    """over-optimization: the proxy rises monotonically, the truth peaks and falls"""
    ref, r_true, r_proxy, hack = ctx.pi_ref, ctx.r_true, ctx.r_proxy, ctx.hack
    betas = 0.02 * 2.0 ** np.arange(13)
    proxy, true, kl = (np.asarray(x, dtype=np.float64)
                       for x in overoptimization(ref, r_proxy, r_true, betas))
    ref_true = float(ref @ r_true)
    print("   beta      KL     proxy      true")
    for b, k, p, t in zip(betas, kl, proxy, true):
        print(f"  {b:6.2f}  {k:6.3f}  {p:8.4f}  {t:8.4f}"
              + ("   <- best true" if t == true.max() else ""))
    print(f"  reference: KL 0.000, proxy {float(ref @ r_proxy):8.4f}, true {ref_true:8.4f}")

    _check("proxy reward is monotone in 1/beta (that is what optimizing means)",
           np.all(np.diff(proxy) < 0), f"diffs {np.round(np.diff(proxy), 4)}")
    _check("KL is monotone in 1/beta", np.all(np.diff(kl) < 0))
    j = int(np.argmax(true))
    _check("true reward peaks at an INTERIOR beta (Goodhart)",
           0 < j < len(betas) - 1, f"argmax at index {j}, beta={betas[j]}")
    _check("over-optimizing drives true reward BELOW the reference's own",
           true[0] < ref_true, f"true {true[0]:.4f} < reference {ref_true:.4f}")
    _check("the peak still beats the reference (RL is worth doing at the right beta)",
           true[j] > ref_true, f"{true[j]:.4f} > {ref_true:.4f}")
    pi_small = tilted_policy(ref, r_proxy, float(betas[0]))
    _check("small beta puts ~all mass on the hacked action",
           pi_small[hack] > 0.99, f"pi[hack] = {pi_small[hack]:.6f}")

    # NEGATIVE CONTROL: the hump is caused by the PROXY, not by the tilt.
    _, true_honest, _ = (np.asarray(x, dtype=np.float64)
                         for x in overoptimization(ref, r_true, r_true, betas))
    _check("tilting by the TRUE reward is monotone -- no hump (negative control)",
           np.all(np.diff(true_honest) < 0), f"diffs {np.round(np.diff(true_honest), 4)}")


STAGES = [
    ("kl", stage_kl),
    ("tilt", stage_tilt),
    ("objective", stage_objective),
    ("numeric", stage_numeric),
    ("frontier", stage_frontier),
    ("asymmetry", stage_asymmetry),
    ("estimators", stage_estimators),
    ("hacking", stage_hacking),
    ("shaping", stage_shaping),
    ("train", stage_train),
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
