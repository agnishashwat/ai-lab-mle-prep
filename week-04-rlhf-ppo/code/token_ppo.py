"""
Week 4 Day 3 -- token-level PPO: value heads, sequence rewards, per-token KL.

Days 1-2 gave you an enumerable token-level LM environment with a fitted Bradley-Terry
RM, and the PPO objective itself. Today adds ONE structural piece -- a value head
trained by REGRESSION -- and then assembles InstructGPT Equation (2) with every term
measured rather than asserted.

The whole day is one mismatch: the reward model emits ONE NUMBER at EOS, and you have
T DECISIONS to credit. Two attacks, and you own both:

    estimator side   GAE spreads the terminal signal backwards
    reward side      per-token KL densifies the reward, which densifies the critic's
                     regression target, which densifies every delta_t

On W3 D4 you INJECTED critic error by hand along interpretable axes and proved only the
state-VARYING part biases the gradient. Today the critic's error is EARNED by a real
regression, and the question is whether it looks like the free kind or the expensive kind.

Four results to come out of here as ASSERTIONS, not claims:

  1. At lambda = 1 the critic enters ONLY as a baseline, so the exact gradient is
     independent of it -- for ANY critic, including a catastrophically bad one. Equal to
     the finite-difference truth to ~1e-9. At lambda < 1 it bootstraps and a wrong critic
     biases you. That pair IS the reconciliation asked for in checkpoint question 3.
  2. Gradient bias tracks the state-VARYING part of the critic error, ||e~||, and not the
     total error ||e|| and not the mean e_bar. W3 D4's theorem, re-derived from a real
     regression instead of a hand-built critic.
  3. Explained variance is invariant to exactly the component that is free, which is why
     it is the right instrument and MSE is the wrong one. Negative control: V + c has
     unbounded MSE, IDENTICAL EV, and identically zero bias at lambda = 1.
  4. The per-token KL shaping is not an approximation to W3 D5's sequence-KL objective --
     it IS that objective, sampled. E[sum_t penalty_t] = beta * exact_sequence_kl, now
     over an honest terminal state and variable length.

Ground truth is enumeration, inherited from Day 1: `exact_J`, `exact_sequence_kl`,
`finite_difference_gradient`, plus today's `exact_values`. Nothing here is established
by a plot.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-04-rlhf-ppo/code/token_ppo.py
  one stage           ...  token_ppo.py decomposition
  a few stages        ...  token_ppo.py gae shaping
  small/fast fixtures ...  token_ppo.py --fast
  list the stages     ...  token_ppo.py --list

Build order (each group is runnable the moment it is written):

  1. value_table_init, value_predict        -> stage `value_table`
  2. gae_advantages                         -> stage `gae`            (regression vs W3 D4)
  3. shape_rewards                          -> stage `shaping`
  4. value_targets, value_loss              -> stage `targets`
  5. fit_value_head                         -> stage `fit`
  6. whiten                                 -> stage `whiten`
  7. (nothing new) -> stages `baseline_vs_bootstrap`, `decomposition`,
     `negative_control`                                              <- the headline
  8. zero_baseline_probe                    -> stage `zero_baseline`  <- the standing D4 gap
  9. instructgpt_loss                       -> stages `loss`, `c1_sweep`

SHAPE TRAP, DELIBERATE: the fixture env is vocab=5, t_max=4, so V=5, T=4, S=85 and B=341
are four different numbers, and the value arrays are (B, T+1) while everything else is
(B, T). A transposed or off-by-one array cannot pass silently. The (B, T+1) vs (B, T)
split is the single likeliest bug in this file and it is deliberate: it is the same
off-by-one that `V[T, :] = 0` made structurally impossible on W3 D4.

BOUNDARY AS DATA: the value table has n_states + 1 rows and row `env.n_states` is
TERMINAL, pinned to 0 forever. `batch.value_ids` is (B, T+1) and already points at that
row for every padded position and for the bootstrap column. So `value_predict` reads 0
at the boundary without a single `if`, and `dones` is then a redundant second guard --
which is the point. Stage `gae` asserts the two agree.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "week-03-rl-foundations" / "code"))

from toy_rlhf import (  # noqa: E402
    TokenEnv,
    enumerate_sequences,
    exact_J,
    exact_sequence_kl,
    log_softmax_np,
    sample_sequences,
    sequence_logprob,
    softmax_np,
)
from ppo_loss import (  # noqa: E402
    Batch,
    _gold_reward,
    clipped_surrogate,
    finite_difference_gradient,
    score_function_gradient,
    surrogate_gradient,
    token_logratios,
)
from gae import gae_naive, gae_reverse  # noqa: E402  (W3 D4 -- today's regression target)


# ---------------------------------------------------------------------------
# The token-level rollout batch (provided -- READ IT, you build against this contract)
# ---------------------------------------------------------------------------


@dataclass
class TokenBatch:
    """One frozen rollout batch, at TOKEN granularity.

    Day 2's `Batch` carried a per-sequence return broadcast along t. Today every
    quantity is genuinely per-token, and three new fields exist purely so that the
    episode boundary is DATA rather than control flow:

    states     (B, T)   int64  state id of the prefix y[:t]; 0 where padded
    actions    (B, T)   int64  token emitted at step t;      0 where padded
    mask       (B, T)   bool   True where the step is real
    dones      (B, T)   float  1.0 at the LAST REAL token of each row, else 0.0.
                               Exactly one per row -- `dones.sum(1) == 1`.
    value_ids  (B, T+1) int64  <- NEW, and the reason this file has few `if`s.
                               value_ids[b, t] is the state whose value is V(s_t), and
                               it is TERMINAL (= env.n_states) for every t >= len(y_b)
                               INCLUDING the bootstrap column t = T. So
                               `value_predict(phi, batch.value_ids)` is already 0 at the
                               boundary, for any phi with a pinned terminal row.
    logp_old   (B, T)   float  log pi_old(a_t|s_t), cached at rollout time
    logp_ref   (B, T)   float  log pi_ref(a_t|s_t), ALSO cached at rollout time -- this
                               is why a production stack queries the reference model
                               during the rollout and can then drop it from memory
    weights    (B,)     float  per-sequence weight, sums to 1. 1/B for a sampled batch,
                               p_theta_old(y) for an ENUMERATED batch -- which is what
                               turns every estimator in this file into an exact
                               expectation instead of an estimate.
    lengths    (B,)     int64  len(y_b)
    """

    states: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    dones: np.ndarray
    value_ids: np.ndarray
    logp_old: np.ndarray
    logp_ref: np.ndarray
    weights: np.ndarray
    lengths: np.ndarray
    seqs: list
    terminal: int

    @property
    def B(self) -> int:
        return self.mask.shape[0]

    @property
    def T(self) -> int:
        return self.mask.shape[1]

    @property
    def n_tokens(self) -> int:
        return int(self.mask.sum())

    @cached_property
    def tok_w(self) -> np.ndarray:
        """(B, T) token weights, 0 where padded, summing to 1 over REAL tokens.

        The weighting under which every mean/variance/EV in this file is taken. On an
        enumerated batch this is the exact occupancy of (state, step) pairs, so a
        "weighted mean" here is a true expectation.
        """
        w = self.weights[:, None] * self.mask
        return w / w.sum()

    def as_ppo_batch(self, advantages: np.ndarray) -> Batch:
        """Adapt to Day 2's `Batch` so `clipped_surrogate`, `surrogate_gradient` and
        `score_function_gradient` can be reused UNCHANGED with an arbitrary Psi.

        That reuse is the point: today's Psi_t is a per-token GAE advantage instead of a
        broadcast sequence return, and NOTHING in Day 2's PPO machinery has to change.
        The Psi ladder again -- every algorithm differs only in Psi_t.
        """
        return Batch(self.states, self.actions, self.mask, np.asarray(advantages, float),
                     self.logp_old, self.weights, np.zeros(self.B), list(self.seqs))


def pack_tokens(env: TokenEnv, seqs, theta_old, theta_ref, weights) -> TokenBatch:
    """Pack sequences into the (B, T) / (B, T+1) contract above (provided)."""
    B, T = len(seqs), env.t_max
    states = np.zeros((B, T), dtype=np.int64)
    actions = np.zeros((B, T), dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    dones = np.zeros((B, T))
    value_ids = np.full((B, T + 1), env.n_states, dtype=np.int64)
    logp_old = np.zeros((B, T))
    logp_ref = np.zeros((B, T))
    lengths = np.zeros(B, dtype=np.int64)

    theta_ref = theta_old if theta_ref is None else theta_ref
    lp, lp_ref = log_softmax_np(theta_old), log_softmax_np(theta_ref)
    for b, s in enumerate(seqs):
        lengths[b] = len(s)
        for t, a in enumerate(s):
            sid = env.state_id(s[:t])
            states[b, t] = sid
            actions[b, t] = a
            mask[b, t] = True
            value_ids[b, t] = sid
            logp_old[b, t] = lp[sid, a]
            logp_ref[b, t] = lp_ref[sid, a]
        dones[b, len(s) - 1] = 1.0

    w = np.asarray(weights, dtype=float)
    return TokenBatch(states, actions, mask, dones, value_ids, logp_old, logp_ref,
                      w / w.sum(), lengths, list(seqs), env.n_states)


def exact_token_batch(env: TokenEnv, theta_old, theta_ref=None) -> TokenBatch:
    """EVERY terminating sequence, weighted by its exact probability under theta_old.

    `theta_ref=None` means "the reference IS theta_old", i.e. zero KL by construction --
    a null control, not a missing value.
    """
    seqs = enumerate_sequences(env)
    p = np.exp([sequence_logprob(env, theta_old, s) for s in seqs])
    return pack_tokens(env, seqs, theta_old, theta_ref, p)


def sample_token_batch(env: TokenEnv, theta_old, n, rng, theta_ref=None) -> TokenBatch:
    """Monte-Carlo rollouts, weights = 1/n."""
    seqs = sample_sequences(env, theta_old, n, rng)
    return pack_tokens(env, seqs, theta_old, theta_ref, np.full(n, 1.0 / n))


# ---------------------------------------------------------------------------
# Ground truth and instruments (provided)
# ---------------------------------------------------------------------------


def exact_values(env: TokenEnv, theta, reward_fn, gamma: float = 1.0,
                 theta_ref=None, beta: float = 0.0) -> np.ndarray:
    """The EXACT V^pi table for this env, shape (n_states + 1,) with V[TERMINAL] = 0.

    TODAY'S GROUND TRUTH, the analogue of W3 D4's `exact_time_indexed_values`. Backward
    induction over the prefix tree:

        V(s) = sum_a pi(a|s) [ r(s, a) + gamma * V(s') ]

    where emitting `a` from `s` terminates (eos, or truncation at t_max) -- in which case
    r(s, a) is the sequence reward and s' is TERMINAL -- and otherwise pays 0 and moves to
    the prefix s + (a,).

    Note there is NO time index on V here, unlike W3 D4. In this env every prefix state
    carries its own depth, so V is a function of the state alone and a horizon mismatch is
    structurally impossible. That is why stage `zero_baseline` asserts an ORDERING of
    variances rather than reproducing W3 D3's +27% -- see its docstring.

    With `beta > 0` the per-token KL penalty is folded in as part of r(s, a). Worth
    noticing: the expected penalty at s is beta * KL(pi(.|s) || pi_ref(.|s)) regardless of
    whether the SAMPLED estimator is k1 or k3, so V* is estimator-independent even though
    the realised rewards are not.
    """
    pi = softmax_np(theta)
    V = np.zeros(env.n_states + 1)
    pen = np.zeros((env.n_states, env.vocab))
    if beta and theta_ref is not None:
        pen = beta * (log_softmax_np(theta) - log_softmax_np(theta_ref))
    order = sorted(range(env.n_states), key=lambda i: -len(env.states[i]))
    for sid in order:
        prefix = env.states[sid]
        total = 0.0
        for a in range(env.vocab):
            terminal = (a == env.eos) or (len(prefix) + 1 >= env.t_max)
            nxt = prefix + (a,)
            r = reward_fn(nxt) if terminal else 0.0
            r = r - pen[sid, a]
            v_next = 0.0 if terminal else V[env.state_id(nxt)]
            total += pi[sid, a] * (r + gamma * v_next)
        V[sid] = total
    return V


def explained_variance(pred, target, w) -> float:
    """EV = 1 - Var(target - pred) / Var(target), under the weights `w` (provided).

    VARIANCE of the residual, not its mean square. That single choice is the entire
    argument of the day: Var is invariant to an additive constant, which W3 D4 proved is
    exactly the free component of the critic error, so EV measures precisely the part
    that can bias you and MSE conflates the two. Choose a diagnostic whose invariances
    match the invariances of the theorem you are relying on.

    Degenerate denominator returns 0.0, not nan (house rule). EV can legitimately be
    NEGATIVE -- that means the critic is worse than predicting the mean.
    """
    w = np.asarray(w, float).ravel()
    w = w / w.sum()
    t = np.asarray(target, float).ravel()
    r = t - np.asarray(pred, float).ravel()
    var_t = float(w @ (t - w @ t) ** 2)
    var_r = float(w @ (r - w @ r) ** 2)
    return 0.0 if var_t <= 1e-300 else 1.0 - var_r / var_t


def token_occupancy(env: TokenEnv, batch: TokenBatch) -> np.ndarray:
    """(n_states + 1,) total token weight landing on each state (provided).

    The weighting for the error decomposition: a critic's error at a state nobody visits
    cannot bias anything, so `e_bar` and `||e~||` must be occupancy-weighted.
    """
    occ = np.zeros(env.n_states + 1)
    np.add.at(occ, batch.states[batch.mask], batch.tok_w[batch.mask])
    return occ


def error_decomposition(phi, phi_star, occ) -> dict:
    """Split the critic error e = V_phi - V* into its free and expensive parts (provided).

        e_bar    occupancy-weighted MEAN error         -- a constant, hence a baseline,
                                                          hence FREE (W3 D4)
        e_tilde  e - e_bar, the state-VARYING part     -- action-dependent once you
                                                          bootstrap, hence EXPENSIVE
    Returns norms under the occupancy weights, plus |e_bar|. `e_norm**2 == e_bar**2 +
    e_tilde_norm**2` is Pythagoras and stage `decomposition` asserts it.
    """
    occ = np.asarray(occ, float)
    tot = occ.sum()
    if tot <= 1e-300:
        return dict(e_bar=0.0, e_norm=0.0, e_tilde_norm=0.0)
    w = occ / tot
    e = np.asarray(phi, float) - np.asarray(phi_star, float)
    e_bar = float(w @ e)
    return dict(e_bar=e_bar,
                e_norm=float(np.sqrt(w @ e ** 2)),
                e_tilde_norm=float(np.sqrt(w @ (e - e_bar) ** 2)))


def policy_entropy(env: TokenEnv, theta, batch: TokenBatch) -> float:
    """Token-weighted mean of H(pi_theta(.|s_t)) over the real tokens (provided).

    H is per-POSITION entropy averaged over positions, not the entropy of the sequence
    distribution. Recall W3 D3: dH/dz_k = -p_k (log p_k + H), so the restoring force
    vanishes at BOTH ends of a collapse -- entropy bonuses are a preventative, not a cure.
    """
    p = softmax_np(theta)
    H = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)
    return float((batch.tok_w[batch.mask] * H[batch.states[batch.mask]]).sum())


def grad_error(g, g_true) -> float:
    """Relative Frobenius distance between a gradient estimate and the truth (provided)."""
    return float(np.linalg.norm(g - g_true) / np.linalg.norm(g_true))


def spearman(a, b) -> float:
    """Rank correlation (provided). Used because the bias-vs-||e~|| relationship is
    monotone but not linear, and asserting monotonicity is the honest claim."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return 0.0 if den == 0 else float(ra @ rb / den)


def shared_trunk_sweep(env: TokenEnv, theta0, reward_fn, c1_list, *, n_iters=20, lr=0.15,
                       eps=0.2, gamma=1.0, lam=0.95, head_seed=11) -> dict:
    """Run the joint loss with a genuinely SHARED trunk, once per c1 (provided).

    Everywhere else in this file the value table is a separate parameter, so c1 is a
    formality: the two gradients never meet. That is NOT the production situation. In a
    real stack the value head is a scalar readout on the policy's transformer body, so
    `c1 * L_VF` puts gradient into the SAME weights the policy is using, and a large
    value loss can drag the trunk somewhere the policy did not ask to go.

    Reproduced here at the smallest honest scale: the trunk is the logit table, and

        V(s) = head . theta[s] + b

    so d L_VF / d theta is nonzero and lands on the policy's own parameters. `head` is
    initialised RANDOM and not constant on purpose -- a constant readout would push every
    logit row by a multiple of the all-ones vector, which softmax is invariant to, and the
    experiment would silently measure nothing.

    Returns {c1: {'J': [...], 'J_final': float}} where J is the exact unshaped return.
    """
    rng = np.random.default_rng(head_seed)
    out = {}
    for c1 in c1_list:
        theta = theta0.copy()
        head = 0.3 * rng.standard_normal(env.vocab)
        b = 0.0
        traj = []
        for _ in range(n_iters):
            batch = exact_token_batch(env, theta)
            rewards = np.zeros((batch.B, batch.T))
            for i, s in enumerate(batch.seqs):
                rewards[i, len(s) - 1] = reward_fn(s)

            vals_all = np.concatenate([theta @ head + b, [0.0]])
            values = vals_all[batch.value_ids]
            adv = gae_advantages(rewards, values, batch.dones, gamma, lam)
            targets = value_targets(adv, values)

            g_pi = surrogate_gradient(env, theta, batch.as_ppo_batch(adv), eps=eps)

            m = batch.mask
            resid = (values[:, :-1] - targets)[m]
            coef = 2.0 * batch.tok_w[m] * resid
            g_theta_v = np.zeros_like(theta)
            np.add.at(g_theta_v, batch.states[m], coef[:, None] * head[None, :])
            g_head = (coef[:, None] * theta[batch.states[m]]).sum(axis=0)
            g_b = float(coef.sum())

            theta = theta + lr * (g_pi - c1 * g_theta_v)
            head = head - lr * c1 * g_head
            b = b - lr * c1 * g_b
            traj.append(float(exact_J(env, theta, reward_fn)))
        out[c1] = dict(J=traj, J_final=traj[-1])
    return out


# ---------------------------------------------------------------------------
# Exercises (TODO bodies)
# ---------------------------------------------------------------------------


def value_table_init(env: TokenEnv) -> np.ndarray:
    """The tabular stand-in for a value head: shape (env.n_states + 1,), all zeros.

    Row `env.n_states` is TERMINAL and must be 0 forever -- nothing after the end of a
    sequence is worth anything. Pinning it as DATA rather than testing for it in the GAE
    recursion is the same move as W3 D4's `(T+1, S)` table with `V[T, :] = 0`, which made
    the off-by-one structurally impossible instead of merely absent.

    Float dtype. An integer accumulator here reproduces W3 D2's `np.long` gradient buffer,
    which truncated every update to zero and reported a beautifully clean result.

    Real value heads are initialised from the SFT model's trunk (or the RM's weights,
    which already score sequences) precisely because a zero critic makes every early
    delta_t pure noise with a state-VARYING error -- by your own W3 D4 theorem, the
    bias-inducing kind. Stage `decomposition` measures exactly that decay.
    """
    return np.zeros(env.n_states+1,)


def value_predict(phi: np.ndarray, state_ids: np.ndarray) -> np.ndarray:
    """Gather V_phi at `state_ids`. Returns an array with the SHAPE OF state_ids.

    One line, and it is a gather, not a loop. The reason it is worth naming as its own
    function: you will call it on `batch.value_ids`, which is (B, T+1) and already points
    at TERMINAL wherever a value must be 0, so the (B, T+1) values array that GAE wants
    comes out with its boundary correct and no branching.

    Do not clamp, mask or special-case anything here. If a 0 is needed at the boundary it
    is because the caller pointed you at the pinned terminal row; that is the contract.
    """
    return phi[state_ids]


def gae_advantages(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                   gamma: float, lam: float) -> np.ndarray:
    """Batched GAE. rewards (B, T), values (B, T+1), dones (B, T) -> advantages (B, T).

        delta_t = r_t + gamma (1 - done_t) V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma lam (1 - done_t) A_{t+1},     A_T = 0

    This is W3 D4's reverse scan, batched. Stage `gae` is a PURE REGRESSION TEST against
    `gae.py` -- your rows must agree with `gae_naive` and `gae_reverse` to ~1e-16, because
    this is algebra and not statistics.

    Two things to get right:

      * `values` is (B, T+1). `values[:, :-1]` is V(s_t) and `values[:, 1:]` is V(s_{t+1}).
        Slicing one of them wrong is the classic silent off-by-one; the terminal column is
        already 0 so it will not crash, it will just be wrong.
      * the `(1 - done_t)` factor appears TWICE, in the delta and in the recursion, and it
        is doing different jobs each time: no bootstrapping past the end, and no credit
        flowing backwards across an episode boundary. In THIS env every row holds exactly
        one sequence and the padding is zeros, so dropping the reset changes nothing and
        the bug is invisible -- which is why stage `gae` also feeds you a hand-built row
        containing TWO episodes, where it is not.

    Vectorise the scan over B (loop over t, not over sequences). Padded positions must come
    out exactly 0.
    """
    delta = rewards + gamma * (1-dones) * values[:, 1:] - values[:, :-1]
    B, T = rewards.shape
    advantages = np.zeros((B, T+1), dtype=np.float64)
    for t in range(T-1, -1, -1):
        advantages[:, t] = delta[:, t] + gamma * lam * (1-dones[:, t]) * advantages[:, t+1]
    return advantages[:, :-1]



def shape_rewards(env: TokenEnv, batch: TokenBatch, theta: np.ndarray, reward_fn,
                  beta: float = 0.0, estimator: str = "k1") -> np.ndarray:
    """InstructGPT Equation (2)'s reward, per token. Returns (B, T).

        r_t = r_psi(y) 1[t = L-1]  -  beta * khat_t

    SPARSE from the RM (one scalar, at the last real token -- `batch.dones` marks it),
    DENSE from the KL (every real token). That asymmetry is the whole of LLM RL.

    `estimator` selects W3 D5's per-token KL estimator, with `lr = logp_ref - logp_theta`:

        k1 = -lr                        unbiased, SIGNED -- it pays a BONUS on any token
                                        the reference likes more than you do
        k3 = exp(lr) - 1 - lr           unbiased, and >= 0 POINTWISE, always

    Both have expectation KL(pi_theta(.|s) || pi_ref(.|s)) under a_t ~ pi_theta, which is
    what makes stage `shaping` able to assert `E[sum_t penalty_t] = beta * sequence KL`
    EXACTLY on an enumerated batch, rather than "within Monte-Carlo error". That identity
    is the reason to believe the per-token form is not an approximation of the D5
    objective but literally is it.

    `theta` is the policy the rollout came from, so recompute log pi_theta(a_t|s_t) from
    it; `batch.logp_ref` is already cached. When theta is theta_old your recomputation must
    equal `batch.logp_old` exactly, and stage `shaping` checks that -- because in the real
    loop this reward is computed ONCE, at rollout time, and then frozen exactly like the
    advantages. Recomputing it inside the epoch loop is a real bug that nothing detects.

    beta = 0 must reproduce the unshaped rewards ELEMENTWISE, not approximately, and
    padded positions must be exactly 0.
    """
    log_pi_theta_all = log_softmax_np(theta, axis=-1)
    logp_theta = log_pi_theta_all[batch.states, batch.actions]
    lr = batch.logp_ref - logp_theta
    if estimator == 'k1':
        khat_t = -lr
    elif estimator == 'k3':
        khat_t = -lr + np.exp(lr) - 1
    khat_t = np.where(batch.mask, khat_t, 0)
    B = batch.B
    terminal_reward = np.zeros_like(khat_t)
    
    for b in range(B):
        terminal_reward[b, batch.lengths[b]-1] = reward_fn(batch.seqs[b])

    reward = terminal_reward - beta * khat_t
    return reward



def value_targets(advantages: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The critic's regression target. advantages (B, T), values (B, T+1) -> (B, T).

        R_hat_t = A_t + V_phi_old(s_t)

    NOT the Monte-Carlo return, and the difference matters. This is the same telescoping
    identity you proved on W3 D4: at lam = 1, sum_l (gamma lam)^l delta_{t+l} = G_t - V(s_t),
    so the target IS the empirical return; at lam < 1 it is a bootstrapped, lower-variance,
    biased target. The critic is trained on the same bias/variance dial as the actor, which
    is a fact most people who have "used TRL" cannot state.

    `values` is the same (B, T+1) array you handed to `gae_advantages`; you want its first
    T columns. Stage `targets` asserts the lam = 1 identity ELEMENTWISE to 1e-15 -- it is
    algebra, so anything looser is hiding a bug.
    """
    return advantages + values[:, :-1]


def value_loss(values_new: np.ndarray, targets: np.ndarray, values_old: np.ndarray,
               tok_w: np.ndarray, clip_eps: float | None = None) -> tuple[float, dict]:
    """L^VF, plain or clipped. All arrays (B, T); `tok_w` is the token weight (0 padded).

        plain    L = sum_bt w_bt (V_new - R_hat)^2
        clipped  L = sum_bt w_bt MAX( (V_new - R_hat)^2 ,
                                      (clip(V_new, V_old +- eps) - R_hat)^2 )

    Return `(loss, diagnostics)` with diagnostics carrying at least

        clip_frac  fraction of real tokens where the MAX selected the clipped branch
        plain      the unclipped loss, so the caller can see what the clip cost

    It is a MAX, not a min -- pessimism applied to a regression. Same principle as
    L^CLIP's `min`, flipped because this objective is MINIMISED: in both cases you take
    the WORSE of the two branches. Getting the direction backwards still trains happily
    and just silently disables the clip, which is why stage `targets` asserts
    `clipped >= plain` pointwise on a hand-computed case rather than only checking that
    the number changed.

    Be honest about this one in an interview: the value clip has no theoretical
    justification whatsoever, it is not in the PPO paper's theory, and it is in every
    implementation. Know it because you will be asked to read code containing it.
    """
    B, T = values_new.shape
    plain_loss = (tok_w * (values_new-targets)**2).sum()
    if clip_eps is not None:
        error_unclipped = (values_new-targets)**2
        error_clipped = (np.clip(values_new, values_old-clip_eps, values_old+clip_eps) - targets)**2
        total_count = B * T - np.where(tok_w==0, 1, 0).sum()
        clip_frac = np.where(error_unclipped<error_clipped, 1, 0).sum()/total_count
        loss = (tok_w * np.maximum(error_unclipped, error_clipped)).sum()
    else:
        loss = plain_loss
        clip_frac = 0
        
    return (loss, {'clip_frac': clip_frac, 'plain': plain_loss})





def fit_value_head(env: TokenEnv, batch: TokenBatch, targets: np.ndarray, n_steps: int,
                   lr: float, phi0: np.ndarray | None = None,
                   snapshot_every: int = 0) -> tuple[np.ndarray, dict]:
    """Regress the value table onto `targets` by gradient descent. Returns (phi, log).

        L(phi) = sum_bt w_bt (phi[s_bt] - R_hat_bt)^2
        dL/dphi[s] = 2 sum_{bt : s_bt = s} w_bt (phi[s] - R_hat_bt)
        phi <- phi - lr * dL/dphi

    Plain gradient descent, not a closed-form solve, because the POINT is the trajectory:
    stage `decomposition` reads the gradient bias off at checkpoints along it and watches
    it die as explained variance rises. A one-shot solve would answer the question but
    would not let you see the answer.

    Three things the checks will find:

      * It is a SCATTER-ADD. Many (b, t) pairs land on the same state, and
        `g[states] += c` with fancy indexing gathers-then-scatters and keeps only the LAST
        write per duplicated index. `np.add.at`. Day 1's bug, in its most dangerous form:
        the result looks plausible.
      * The TERMINAL row must never move. It receives no gradient because no token is
        indexed to it -- but assert it, because a stray `phi -= lr * g` over a wrongly
        shaped g will move it and every downstream boundary silently becomes nonzero.
      * `phi0=None` means start from `value_table_init`, i.e. an UNTRAINED critic, which is
        the realistic step-0 condition and the one that produces the early bias.

    `log` is a dict of LISTS OF PYTHON FLOATS (W3 D3: `history.append(tensor)` leaked the
    graph, and the habit is enforced here rather than remembered):

        step   the step index at each snapshot
        ev     explained_variance(V_phi at the batch's tokens, targets, batch.tok_w)
        mse    the token-weighted mean squared error
        phi    the snapshot itself (an np.ndarray copy -- this one is not a float)

    Snapshot at step 0 (before any update) and after the final step, plus every
    `snapshot_every` steps if that is nonzero. The step-0 entry is the one that carries the
    day's headline, so do not skip it.

    This converges to the token-weighted MEAN target at each state, which on an enumerated
    batch with lam = 1 targets is E[G_t | s_t = s] = V*(s) exactly. Stage `fit` asserts
    that limit against `exact_values`, so the regression has a right answer and not merely
    a decreasing loss.
    """
    if phi0 is None:
        phi = value_table_init(env)
    else:
        phi = phi0.copy()
    
    log = {'step': [], 'ev': [], 'mse': [], 'phi': []}
    def update_dict(step, phi):
        log['step'].append(step)
        log['ev'].append(explained_variance(phi[batch.states], targets, batch.tok_w))
        log['mse'].append(float((batch.tok_w * (phi[batch.states]-targets)**2).sum()))
        log['phi'].append(phi.copy())

    update_dict(0, phi)

    for step in range(n_steps):
        grad = np.zeros_like(phi, dtype=np.float64)
        np.add.at(grad, batch.states, 2 * batch.tok_w * (phi[batch.states]-targets))
        phi -= grad * lr
        if snapshot_every>0 and (step+1)%snapshot_every==0:
            update_dict(step+1, phi)
            
    if log['step'][-1] != n_steps:
        update_dict(n_steps, phi)
    return (phi, log)

    


def whiten(adv: np.ndarray, tok_w: np.ndarray, mask: np.ndarray,
           center_only: bool = True) -> np.ndarray:
    """Advantage whitening over the real tokens of a batch. Returns (B, T), padded 0.

        center_only=True    A - mean_w(A)
        center_only=False   (A - mean_w(A)) / std_w(A)

    The two halves are NOT the same kind of operation and the distinction is the point:

      * CENTERING is a batch-level constant baseline. It is exactly the thing W3 D3 proved
        is free, and stage `whiten` asserts the exact gradient is UNCHANGED to 1e-12 --
        not "approximately", not "in direction", identical.
      * DIVIDING BY THE STD is not. Within one batch it is a scalar and factors out of the
        gradient, so it is only a per-batch learning rate. ACROSS batches -- or across
        prompt groups within a batch -- the std differs, so it silently reweights groups
        against each other and the low-variance group gets amplified. You already own this
        object under another name: it is exactly the Dr. GRPO critique of GRPO's std
        division. Same bug, different letterhead. Stage `whiten` constructs two groups with
        different spreads and measures the reweighting.

    Weighted mean and weighted std under `tok_w`, over REAL tokens only -- padded entries
    are not tokens and must not enter either moment, and must come out 0.

    DEGENERATE BATCH: when every advantage is equal the std is 0. Return the centered
    array (all zeros), FINITE. Not nan, not inf. Standing house rule since W1: an empty or
    degenerate denominator produces 0.0, and a check enforces it because this is the shape
    of failure that survives all the way to a training run that quietly does nothing.
    """
    adv_masked = np.where(mask, adv, 0)
    tok_w_masked = np.where(mask, tok_w, 0)
    if center_only:
        return np.where(mask, adv_masked - (tok_w_masked * adv_masked).sum(), 0)
    else:
        sigma_sq = (tok_w_masked * (adv_masked - (tok_w_masked * adv_masked).sum())**2).sum()
        sigma = np.sqrt(sigma_sq)
        if sigma < 1e-10:
            return np.zeros_like(adv, dtype=np.float64)
        return np.where(mask, (adv_masked - (tok_w_masked * adv_masked).sum())/sigma, 0)


def instructgpt_loss(env: TokenEnv, theta: np.ndarray, phi: np.ndarray,
                     batch: TokenBatch, rewards: np.ndarray, values_old: np.ndarray, *,
                     gamma: float = 1.0, lam: float = 0.95, eps: float = 0.2,
                     c1: float = 0.5, c2: float = 0.0,
                     value_clip_eps: float | None = None,
                     whiten_mode: str | None = None,
                     reduce: str = "sum") -> tuple[float, dict]:
    """THE WHOLE THING. Returns `(total, parts)` where total is a LOSS to MINIMISE.

        L(theta, phi) = - L^CLIP(theta) + c1 L^VF(phi) - c2 E[H(pi_theta)]

    with `rewards` already shaped by `shape_rewards` and `values_old` the (B, T+1) values
    from the FROZEN phi_old. Assemble, in this order, and the order is the exercise:

      1. advantages = gae_advantages(rewards, values_old, batch.dones, gamma, lam)
      2. targets    = value_targets(advantages, values_old)      <- BEFORE whitening
      3. whiten the advantages if asked ("center" | "scale" | None)
      4. L^CLIP via Day 2's `clipped_surrogate` on `batch.as_ppo_batch(adv)`
      5. L^VF   via `value_loss`, with values_new = value_predict(phi, batch.value_ids)[:, :-1]
      6. H      via `policy_entropy`
      7. combine, minding the signs

    Step 2 before step 3 is not stylistic. The critic regresses onto RETURNS, which live in
    reward units; whitened advantages do not. Whitening first gives you a critic fitted to
    a per-batch-rescaled quantity, which trains, produces no nan, and is wrong.

    `reduce` is the sum-vs-mean choice over tokens, and it applies to the POLICY term only
    (L^VF and H are always token-weighted means, as in every real implementation):

        "sum"   sum_t inside each sequence     -- what the policy gradient theorem says
        "mean"  (1/L_b) sum_t                  -- what most implementations ship

    Implement "mean" by scaling each sequence's advantages by 1/len(y_b) before the
    surrogate. Doing it that way makes the consequence unmissable and it is the point of
    including the switch: dividing by length is IDENTICAL to multiplying each sequence's
    weight by 1/L, i.e. it silently upweights short completions. You are not choosing a
    normalisation, you are choosing a preference over lengths. Derive the consequence
    properly tomorrow; today just make the choice explicit and RECORD it.

    `parts` must carry at least: 'clip', 'vf', 'entropy', 'total', 'clip_frac', 'oob_frac',
    'vf_clip_frac', 'ev', 'adv_mean', 'adv_std', 'reduce', where 'ev' is
    `explained_variance(values_old[:, :-1], targets, tok_w)` over the real tokens -- the
    critic you ROLLED OUT with, scored against the targets it is about to be fitted to --
    and 'adv_mean'/'adv_std' are the token-weighted moments of the advantages actually
    handed to the surrogate (i.e. after whitening, if any). Sign discipline: raising L^CLIP
    must LOWER `total`, and raising L^VF must RAISE it. Stage `loss` asserts both
    directions, because a sign error here trains without complaint and you will spend a
    day blaming the learning rate.

    Note c2 = 0.0 is the DEFAULT and that is not laziness: InstructGPT Equation (2) has no
    entropy term. A uniform-entropy target is actively wrong for text -- you want high
    entropy where genuinely uncertain, not after "the capital of France is" -- and the KL
    to a reference LANGUAGE MODEL expresses a position-dependent target that a scalar
    entropy bonus cannot.
    """
    advantages = gae_advantages(rewards, values_old, batch.dones, gamma, lam)
    targets = value_targets(advantages, values_old)

    if whiten_mode is not None:
        if whiten_mode not in ("center", "scale"):
            raise ValueError(f"whiten_mode must be 'center', 'scale' or None, got {whiten_mode!r}")
        advantages = whiten(advantages, batch.tok_w, batch.mask,
                            center_only=(whiten_mode == "center"))

    m, w = batch.mask, batch.tok_w
    adv_mean = float((w * advantages).sum())
    adv_std = float(np.sqrt((w * (advantages - adv_mean) ** 2).sum()))

    if reduce == "mean":
        adv_eff = np.where(m, advantages / batch.lengths[:, None], 0.0)
    elif reduce == "sum":
        adv_eff = advantages
    else:
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")

    pb = batch.as_ppo_batch(adv_eff)
    l_clip, diag_clip = clipped_surrogate(token_logratios(env, theta, pb), pb, eps)

    values_new = value_predict(phi, batch.value_ids)[:, :-1]
    l_vf, diag_vf = value_loss(values_new, targets, values_old[:, :-1], w, value_clip_eps)

    H = policy_entropy(env, theta, batch)
    total = -l_clip + c1 * l_vf - c2 * H

    parts = dict(
        clip=float(l_clip), vf=float(l_vf), entropy=float(H), total=float(total),
        clip_frac=float(diag_clip["clip_frac"]), oob_frac=float(diag_clip["oob_frac"]),
        vf_clip_frac=float(diag_vf["clip_frac"]),
        ev=float(explained_variance(values_old[:, :-1][m], targets[m], w[m])),
        adv_mean=adv_mean, adv_std=adv_std, reduce=reduce,
    )
    return float(total), parts



def zero_baseline_probe(env: TokenEnv, theta: np.ndarray, batch: TokenBatch,
                        psi_variants: dict) -> dict:
    """THE DECISIVE EXPERIMENT. Returns {name: gradient variance} for each Psi given.

    Per-sequence gradient estimate and its variance under the batch weights:

        g_b  = sum_t Psi_bt grad_theta log pi_theta(a_bt | s_bt)
        gbar = sum_b w_b g_b
        Var  = sum_b w_b || g_b - gbar ||_F^2

    the trace of the estimator's covariance -- one number, comparable across variants
    because they are computed on the SAME rollouts.

    This is the answer you have now failed to produce twice, so learn it as a sentence:
    to find out whether a critic is helping, ZERO THE BASELINE AND RECOMPUTE THE GRADIENT
    VARIANCE ON THE SAME ROLLOUTS. A healthy critic reduces it; a mis-specified one can
    increase it. It is decisive precisely because it needs no ground-truth gradient --
    which a real stack does not have, and which is why "compare against the true gradient"
    is an unusable answer in an interview.

    Reuse `grad_theta log pi(a|s) = e_a - p(.|s)` (W1's dCE/dz with the sign flipped). The
    cheap route is one (B, n_states, vocab) buffer and a single `np.add.at` with the
    sequence index carried along; the float-dtype and `np.add.at`-not-`+=` traps are
    exactly as they were in `surrogate_gradient`.

    Padded positions contribute nothing (Psi is 0 there). Return plain Python floats.
    """
    p = softmax_np(theta)
    m = batch.mask
    seq_idx = np.repeat(np.arange(batch.B)[:, None], batch.T, axis=1)[m]
    s, a = batch.states[m], batch.actions[m]

    out = {}
    for name, psi in psi_variants.items():
        psi = np.asarray(psi, float)[m]
        # Per-sequence gradient: (B, n_states, vocab), scattered by (sequence, state).
        g = np.zeros((batch.B, env.n_states, env.vocab))
        np.add.at(g, (seq_idx, s, a), psi)                    # the e_a term
        np.add.at(g, (seq_idx, s), -psi[:, None] * p[s])      # the -p(.|s) term
        g_bar = (batch.weights[:, None, None] * g).sum(axis=0)
        out[name] = float((batch.weights * ((g - g_bar) ** 2).sum(axis=(1, 2))).sum())
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
    """Lazily-built fixtures, each from its OWN seeded generator so a stage's data does
    not depend on which other stages ran first."""

    def __init__(self, fast: bool = False):
        self.fast = fast
        # V=5, T=4, S=85, B=341 -- four different numbers, and the value arrays are
        # (B, T+1) while everything else is (B, T). See the module docstring.
        self.env = TokenEnv(vocab=5, t_max=4)
        self.gamma = 1.0
        self.lam = 0.95          # the "shipping" lambda, used by the loss assembly
        self.lam_boot = 0.5      # bootstrapping hard ON, used by every bias measurement
        self.eps = 0.2
        self.beta = 0.3
        self.c_offset = 25.0     # the negative control's offset -- deliberately absurd
        # --fast shrinks DATA and loosens TOLERANCES. It never cuts gradient STEPS: on
        # D1 that turned a true assertion false for reasons unrelated to the assertion.
        self.fit_steps = 4000
        self.fit_lr = 1.5
        self.snapshot_every = 200 if not fast else 400
        self.n_mc = 20_000 if not fast else 4_000
        self.trunk_iters = 30 if not fast else 30
        self.noise_scales = (0.05, 0.15, 0.4, 1.0, 2.5) if not fast else (0.1, 0.5, 2.0)

    # -- policies and reward -------------------------------------------------
    @cached_property
    def theta(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(0), scale=1.0)

    @cached_property
    def theta_ref(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(1), scale=0.8)

    @cached_property
    def reward(self):
        return _gold_reward(self.env)

    # -- batches -------------------------------------------------------------
    @cached_property
    def eb(self) -> TokenBatch:
        """Enumerated batch, reference == policy (zero KL by construction)."""
        return exact_token_batch(self.env, self.theta)

    @cached_property
    def ebr(self) -> TokenBatch:
        """Enumerated batch carrying a genuine reference policy."""
        return exact_token_batch(self.env, self.theta, self.theta_ref)

    @cached_property
    def mb(self) -> TokenBatch:
        return sample_token_batch(self.env, self.theta, self.n_mc,
                                  np.random.default_rng(3), self.theta_ref)

    # -- ground truth --------------------------------------------------------
    @cached_property
    def r_term(self) -> np.ndarray:
        """(B, T) terminal-only rewards, built WITHOUT `shape_rewards` so the stage that
        tests `shape_rewards` has something independent to be compared against."""
        b = self.eb
        r = np.zeros((b.B, b.T))
        for i, s in enumerate(b.seqs):
            r[i, len(s) - 1] = self.reward(s)
        return r

    @cached_property
    def g_true(self) -> np.ndarray:
        print("  ... building the finite-difference gradient (enumerated, ~2.5s)")
        return finite_difference_gradient(self.env, self.theta, self.reward)

    @cached_property
    def v_star(self) -> np.ndarray:
        return exact_values(self.env, self.theta, self.reward, self.gamma)

    @cached_property
    def returns_to_go(self) -> np.ndarray:
        """(B, T) G_t by W3's reverse scan on the rewards -- the lam = 1 target, reached
        without touching today's code."""
        b = self.eb
        return np.stack([gae_reverse(self.r_term[i], self.gamma, 1.0, b.dones[i])
                         for i in range(b.B)]) * b.mask

    @cached_property
    def occ(self) -> np.ndarray:
        return token_occupancy(self.env, self.eb)

    # -- the FIXED-LENGTH world -------------------------------------------------
    # Same env, same reward, same everything -- except eos is driven to probability
    # ~4e-18, so every rollout runs to t_max and the episode LENGTH stops being an
    # action. That single change is the control for stage `decomposition`'s headline.
    @cached_property
    def theta_fixed(self) -> np.ndarray:
        th = self.theta.copy()
        th[:, self.env.eos] = -40.0
        return th

    @cached_property
    def eb_fixed(self) -> TokenBatch:
        return exact_token_batch(self.env, self.theta_fixed)

    @cached_property
    def r_term_fixed(self) -> np.ndarray:
        b = self.eb_fixed
        r = np.zeros((b.B, b.T))
        for i, seq in enumerate(b.seqs):
            r[i, len(seq) - 1] = self.reward(seq)
        return r

    @cached_property
    def g_true_fixed(self) -> np.ndarray:
        print("  ... building the fixed-length finite-difference gradient (~2.5s)")
        return finite_difference_gradient(self.env, self.theta_fixed, self.reward)

    @cached_property
    def v_star_fixed(self) -> np.ndarray:
        return exact_values(self.env, self.theta_fixed, self.reward, self.gamma)

    def bias_fixed(self, phi, lam) -> float:
        b = self.eb_fixed
        adv = gae_advantages(self.r_term_fixed, value_predict(phi, b.value_ids),
                             b.dones, self.gamma, lam)
        g = score_function_gradient(self.env, self.theta_fixed, b.as_ppo_batch(adv))
        return grad_error(g, self.g_true_fixed)

    # -- shared measurement routines ----------------------------------------
    def values_of(self, phi, batch=None) -> np.ndarray:
        b = self.eb if batch is None else batch
        return value_predict(phi, b.value_ids)

    def advantages_of(self, phi, lam, batch=None, rewards=None) -> np.ndarray:
        b = self.eb if batch is None else batch
        r = self.r_term if rewards is None else rewards
        return gae_advantages(r, self.values_of(phi, b), b.dones, self.gamma, lam)

    def bias_of(self, phi, lam) -> float:
        """Relative gradient error of the exact estimator built on this critic."""
        adv = self.advantages_of(phi, lam)
        g = score_function_gradient(self.env, self.theta, self.eb.as_ppo_batch(adv))
        return grad_error(g, self.g_true)

    def ev_of(self, phi) -> float:
        v = self.values_of(phi)[:, :-1]
        return explained_variance(v[self.eb.mask], self.returns_to_go[self.eb.mask],
                                  self.eb.tok_w[self.eb.mask])

    def mse_of(self, phi) -> float:
        v = self.values_of(phi)[:, :-1]
        m = self.eb.mask
        return float((self.eb.tok_w[m] * (v[m] - self.returns_to_go[m]) ** 2).sum())

    def phi_star(self) -> np.ndarray:
        """V* as a value TABLE, terminal row already pinned by `exact_values`."""
        return self.v_star.copy()


def _fail_fast(ctx: Ctx, *needed: str) -> None:
    """Touch ONLY the TODOs this stage actually needs, on a 1-element input, so a missing
    implementation surfaces before we pay for a finite-difference gradient or a 4000-step
    regression -- and so a stage is never blocked by a TODO it does not use.

    Each probe is built from raw numpy, not from the output of another TODO, so probing
    `gae_advantages` fails if and only if `gae_advantages` is missing. The exception is
    `instructgpt_loss`, which genuinely calls the others.
    """
    tiny = TokenEnv(vocab=3, t_max=2)
    th = np.zeros((tiny.n_states, tiny.vocab))
    b = exact_token_batch(tiny, th, th)
    z_t = np.zeros((b.B, b.T))
    z_t1 = np.zeros((b.B, b.T + 1))
    z_phi = np.zeros(tiny.n_states + 1)

    probes = {
        "value_table_init": lambda: value_table_init(tiny),
        "value_predict": lambda: value_predict(z_phi, b.value_ids),
        "gae_advantages": lambda: gae_advantages(z_t, z_t1, b.dones, 1.0, 0.5),
        "shape_rewards": lambda: shape_rewards(tiny, b, th, lambda s: 1.0, beta=0.0),
        "value_targets": lambda: value_targets(z_t, z_t1),
        "value_loss": lambda: value_loss(z_t, z_t, z_t, b.tok_w, None),
        "fit_value_head": lambda: fit_value_head(tiny, b, z_t, 1, 0.1, phi0=z_phi),
        "whiten": lambda: whiten(z_t, b.tok_w, b.mask, center_only=True),
        "zero_baseline_probe": lambda: zero_baseline_probe(tiny, th, b, {"a": z_t}),
        "instructgpt_loss": lambda: instructgpt_loss(tiny, th, z_phi, b, z_t, z_t1, lam=0.5),
    }
    for name in needed:
        probes[name]()


# ---- individual stages ----------------------------------------------------


def stage_value_table(ctx: Ctx) -> None:
    """value_table_init / value_predict -- shapes, the pinned terminal row, the gather"""
    env, b = ctx.env, ctx.eb
    phi = value_table_init(env)

    _check("shape is (n_states + 1,), not (n_states,)", phi.shape == (env.n_states + 1,),
           f"got {phi.shape}, want {(env.n_states + 1,)}")
    _check("dtype is floating (an int table truncates every update to 0)",
           np.issubdtype(phi.dtype, np.floating), str(phi.dtype))
    _check("initialised to exactly zero", np.all(phi == 0.0))
    _check("TERMINAL row exists at index env.n_states", phi[env.n_states] == 0.0)

    rng = np.random.default_rng(21)
    phi_r = rng.standard_normal(env.n_states + 1)
    phi_r[env.n_states] = 0.0

    v = value_predict(phi_r, b.value_ids)
    _check("value_predict returns the shape of state_ids, i.e. (B, T+1)",
           v.shape == (b.B, b.T + 1), f"got {v.shape}, want {(b.B, b.T + 1)}")
    _check("it is a gather: one hand-checked entry matches",
           v[7, 1] == phi_r[b.value_ids[7, 1]],
           f"{v[7, 1]:.6f} vs {phi_r[b.value_ids[7, 1]]:.6f}")

    # The whole point of `value_ids`: the boundary comes out 0 with no branching.
    beyond = np.arange(b.T + 1)[None, :] >= b.lengths[:, None]
    _check("every position at or past the end of a sequence reads EXACTLY 0",
           np.all(v[beyond] == 0.0), f"{int((v[beyond] != 0).sum())} nonzero")
    _check("the bootstrap column t = T is entirely 0", np.all(v[:, -1] == 0.0))
    _check("real positions are generally nonzero", np.any(v[~beyond] != 0.0))

    v1 = value_predict(phi_r, b.states[b.mask])
    _check("works on a 1-D index array too (shape follows the input)",
           v1.shape == (b.n_tokens,), f"got {v1.shape}")


def stage_gae(ctx: Ctx) -> None:
    """gae_advantages -- regression against W3 D4's gae.py, and the done-reset"""
    env, b = ctx.env, ctx.eb
    rng = np.random.default_rng(22)
    phi = rng.standard_normal(env.n_states + 1)
    phi[env.n_states] = 0.0
    values = value_predict(phi, b.value_ids)

    for lam in (1.0, 0.7, 0.0):
        adv = gae_advantages(ctx.r_term, values, b.dones, ctx.gamma, lam)
        _check(f"lam={lam}: shape is (B, T)", adv.shape == (b.B, b.T), f"got {adv.shape}")
        _check(f"lam={lam}: padded positions are exactly 0",
               np.all(adv[~b.mask] == 0.0), f"max|.| = {np.abs(adv[~b.mask]).max():.3e}")

        # Row-by-row regression against BOTH of W3 D4's implementations.
        worst_naive = worst_rev = 0.0
        for i in range(0, b.B, 7):
            L = int(b.lengths[i])
            d = (ctx.r_term[i] + ctx.gamma * (1.0 - b.dones[i]) * values[i, 1:]
                 - values[i, :-1])
            worst_naive = max(worst_naive,
                              float(np.abs(gae_naive(d[:L], ctx.gamma, lam) - adv[i, :L]).max()))
            worst_rev = max(worst_rev,
                            float(np.abs(gae_reverse(d, ctx.gamma, lam, b.dones[i])
                                         - adv[i]).max()))
        _check(f"lam={lam}: matches gae_naive elementwise (algebra, not statistics)",
               worst_naive < 1e-14, f"max|diff| = {worst_naive:.3e}")
        _check(f"lam={lam}: matches gae_reverse elementwise",
               worst_rev < 1e-14, f"max|diff| = {worst_rev:.3e}")

    # lam = 0 must collapse to the one-step TD error, by definition.
    a0 = gae_advantages(ctx.r_term, values, b.dones, ctx.gamma, 0.0)
    d_all = ctx.r_term + ctx.gamma * (1.0 - b.dones) * values[:, 1:] - values[:, :-1]
    _check("lam = 0 is exactly delta_t", np.max(np.abs(a0 - d_all * b.mask)) < 1e-15)

    # BOUNDARY AS DATA, twice over: because value_ids already points at TERMINAL, the
    # (1 - done) factor in the DELTA is redundant here. Not in the recursion.
    d_nodone = ctx.r_term + ctx.gamma * values[:, 1:] - values[:, :-1]
    _check("value_ids makes the (1-done) factor redundant in the delta",
           np.max(np.abs((d_all - d_nodone)[b.mask])) < 1e-15)

    # ... and here is the row where it is NOT redundant in the recursion: two episodes
    # packed into one row, which is what every real flat rollout buffer looks like.
    r2 = np.array([[1.0, 2.0, 3.0, 4.0]])
    v2 = np.array([[0.5, -1.0, 2.0, 0.25, 7.0]])
    d2 = np.array([[0.0, 1.0, 0.0, 1.0]])
    got = gae_advantages(r2, v2, d2, 1.0, 0.5)
    want = np.array([[1.0, 3.0, 3.125, 3.75]])
    _check("two episodes in one row: matches the hand computation",
           np.allclose(got, want, atol=1e-15), f"got {np.round(got, 6)}, want {want}")
    leak = np.array([[3.25, 4.5, 3.125, 3.75]])
    _check("...and is NOT the value you get by leaking credit across the boundary",
           not np.allclose(got, leak, atol=1e-9), "the (1-done) reset is missing")
    _check("the value beyond a done is never bootstrapped (v[4]=7 is ignored)",
           abs(float(got[0, 3]) - 3.75) < 1e-15, f"got {got[0, 3]:.6f}")


def stage_shaping(ctx: Ctx) -> None:
    """shape_rewards -- beta=0 exactness, the sequence-KL identity, k1 vs k3"""
    env, b = ctx.env, ctx.ebr

    r0 = shape_rewards(env, b, ctx.theta, ctx.reward, beta=0.0)
    r_ref = np.zeros((b.B, b.T))
    for i, s in enumerate(b.seqs):
        r_ref[i, len(s) - 1] = ctx.reward(s)
    _check("shape is (B, T)", r0.shape == (b.B, b.T), f"got {r0.shape}")
    _check("beta = 0 reproduces the unshaped rewards ELEMENTWISE",
           np.max(np.abs(r0 - r_ref)) < 1e-15, f"max|diff| = {np.max(np.abs(r0 - r_ref)):.3e}")
    _check("the RM reward lands only on the last real token",
           np.all(r0[b.mask & (b.dones == 0)] == 0.0))
    _check("padded positions are exactly 0", np.all(r0[~b.mask] == 0.0))

    kl_exact = exact_sequence_kl(env, ctx.theta, ctx.theta_ref)
    for est in ("k1", "k3"):
        r = shape_rewards(env, b, ctx.theta, ctx.reward, beta=ctx.beta, estimator=est)
        pen = r0 - r                                     # = beta * khat_t
        got = float((b.weights * pen.sum(axis=1)).sum())
        want = ctx.beta * kl_exact
        _check(f"{est}: E[sum_t penalty_t] == beta * exact_sequence_kl EXACTLY",
               abs(got - want) < 1e-11,
               f"{got:.12f} vs {want:.12f}  (seq KL = {kl_exact:.6f} nats)")
        _check(f"{est}: padded positions stay 0", np.all(pen[~b.mask] == 0.0))

    k1 = (r0 - shape_rewards(env, b, ctx.theta, ctx.reward, ctx.beta, "k1"))[b.mask]
    k3 = (r0 - shape_rewards(env, b, ctx.theta, ctx.reward, ctx.beta, "k3"))[b.mask]
    w = b.tok_w[b.mask]
    m1, m3 = float(w @ k1), float(w @ k3)
    v1 = float(w @ (k1 - m1) ** 2)
    v3 = float(w @ (k3 - m3) ** 2)
    _check("k1 and k3 have the SAME mean, exactly (not within MC error)",
           abs(m1 - m3) < 1e-12, f"{m1:.12f} vs {m3:.12f}")
    _check("k3 is >= 0 POINTWISE", k3.min() >= -1e-15, f"min = {k3.min():.3e}")
    frac_bonus = float((w * (k1 < 0)).sum())
    _check("k1 pays a BONUS on a nonzero fraction of tokens (that is the whole point)",
           frac_bonus > 0.05, f"{100 * frac_bonus:.1f}% of token weight")
    _check("k3 never pays a bonus", float((w * (k3 < -1e-15)).sum()) == 0.0)
    print(f"  ... variance ratio k3/k1 = {v3 / v1:.4f}   "
          f"(k1 var {v1:.5f}, k3 var {v3:.5f}, k1 bonus on {100 * frac_bonus:.1f}%)")

    # Monte Carlo must converge to the same number by a structurally different route.
    mb = ctx.mb
    r0m = shape_rewards(env, mb, ctx.theta, ctx.reward, beta=0.0)
    pm = (r0m - shape_rewards(env, mb, ctx.theta, ctx.reward, ctx.beta, "k1"))
    got_mc = float((mb.weights * pm.sum(axis=1)).sum())
    tol = 0.03 if not ctx.fast else 0.08
    _check("sampled rollouts converge to the same shaping mass",
           abs(got_mc - ctx.beta * kl_exact) < tol,
           f"MC {got_mc:.5f} vs exact {ctx.beta * kl_exact:.5f} (n={mb.B})")


def stage_targets(ctx: Ctx) -> None:
    """value_targets / value_loss -- the lam=1 identity, and the clip's DIRECTION"""
    env, b = ctx.env, ctx.eb
    rng = np.random.default_rng(24)
    phi = rng.standard_normal(env.n_states + 1)
    phi[env.n_states] = 0.0
    values = value_predict(phi, b.value_ids)

    a1 = gae_advantages(ctx.r_term, values, b.dones, ctx.gamma, 1.0)
    t1 = value_targets(a1, values)
    _check("shape is (B, T)", t1.shape == (b.B, b.T), f"got {t1.shape}")
    _check("at lam = 1 the target IS the empirical return G_t, elementwise",
           np.max(np.abs((t1 - ctx.returns_to_go)[b.mask])) < 1e-13,
           f"max|diff| = {np.max(np.abs((t1 - ctx.returns_to_go)[b.mask])):.3e}")

    a2 = gae_advantages(ctx.r_term, values, b.dones, ctx.gamma, 0.6)
    t2 = value_targets(a2, values)
    m, w = b.mask, b.tok_w[b.mask]
    _check("at lam < 1 the target is NOT the empirical return",
           np.max(np.abs((t2 - ctx.returns_to_go)[m])) > 1e-6)
    var1 = float(w @ (t1[m] - w @ t1[m]) ** 2)
    var2 = float(w @ (t2[m] - w @ t2[m]) ** 2)
    _check("...and it has LOWER variance (the bias/variance dial, on the critic)",
           var2 < var1, f"var(lam=0.6) {var2:.4f} < var(lam=1) {var1:.4f}")

    # Hand-computed value loss, both branches.
    vn = np.array([[3.0]]); tg = np.array([[1.0]]); vo = np.array([[2.0]])
    tw = np.array([[1.0]])
    lp, _ = value_loss(vn, tg, vo, tw, None)
    lc, dc = value_loss(vn, tg, vo, tw, 0.5)
    _check("plain loss is the hand value (3-1)^2 = 4", abs(lp - 4.0) < 1e-13, f"{lp}")
    _check("clipped picks the WORSE branch: max(4, 2.25) = 4", abs(lc - 4.0) < 1e-13, f"{lc}")
    _check("...and reports that the clip did not bind", dc["clip_frac"] == 0.0)

    # The case that exposes the value clip for what it is: a critic that jumped straight
    # to the correct answer is PUNISHED, because the clip anchors it to a stale V_old.
    vn2 = np.array([[5.0]]); tg2 = np.array([[5.0]]); vo2 = np.array([[0.0]])
    lp2, _ = value_loss(vn2, tg2, vo2, tw, None)
    lc2, dc2 = value_loss(vn2, tg2, vo2, tw, 1.0)
    _check("plain loss of an exactly-correct critic is 0", abs(lp2) < 1e-13, f"{lp2}")
    _check("the value clip turns that 0 into 16 -- it is a MAX, not a min",
           abs(lc2 - 16.0) < 1e-13, f"{lc2}")
    _check("...and reports clip_frac = 1", dc2["clip_frac"] == 1.0)

    vfull = value_predict(phi, b.value_ids)[:, :-1]
    vold = value_predict(phi * 0.9, b.value_ids)[:, :-1]
    lp3, d3 = value_loss(vfull, t1, vold, b.tok_w, None)
    lc3, d4 = value_loss(vfull, t1, vold, b.tok_w, 0.1)
    _check("on a real batch the clipped loss is >= the plain loss", lc3 >= lp3 - 1e-15,
           f"clipped {lc3:.5f} >= plain {lp3:.5f}")
    _check("diagnostics carry the plain loss for comparison",
           abs(d4.get("plain", np.nan) - lp3) < 1e-12, f"{d4.get('plain')}")
    _check("clip_frac is a fraction of REAL tokens", 0.0 <= d4["clip_frac"] <= 1.0,
           f"{d4['clip_frac']:.4f}")


def stage_fit(ctx: Ctx) -> None:
    """fit_value_head -- EV rises, the terminal row never moves, V* is the limit"""
    _fail_fast(ctx, "value_table_init", "fit_value_head")
    env, b = ctx.env, ctx.eb
    targets = ctx.returns_to_go            # lam = 1 targets, so the limit is exactly V*
    phi, log = fit_value_head(env, b, targets, ctx.fit_steps, ctx.fit_lr,
                              snapshot_every=ctx.snapshot_every)

    _check("returns a (n_states + 1,) table", phi.shape == (env.n_states + 1,), f"{phi.shape}")
    _check("the TERMINAL row never moved", phi[env.n_states] == 0.0, f"{phi[env.n_states]}")
    for key in ("step", "ev", "mse", "phi"):
        _check(f"log carries '{key}'", key in log)
    _check("log['ev'] entries are python floats, not numpy scalars",
           all(type(x) is float for x in log["ev"]), str({type(x) for x in log["ev"]}))
    _check("the first snapshot is step 0, BEFORE any update", log["step"][0] == 0,
           f"{log['step'][:3]}")
    _check("the step-0 critic is the untrained one", np.all(log["phi"][0] == 0.0))

    ev, mse = np.asarray(log["ev"]), np.asarray(log["mse"])
    _check("MSE decreases monotonically", np.all(np.diff(mse) <= 1e-12),
           f"worst increase {np.max(np.diff(mse)):.3e}")
    _check("EV increases monotonically", np.all(np.diff(ev) >= -1e-12),
           f"worst decrease {np.min(np.diff(ev)):.3e}")
    _check("EV starts at ~0 (a zero critic explains nothing)", abs(ev[0]) < 1e-9,
           f"{ev[0]:.3e}")

    # The regression has a RIGHT ANSWER, not merely a decreasing loss: the token-weighted
    # mean target at each state IS E[G_t | s_t = s] = V*(s).
    err = np.abs(phi - ctx.v_star)
    w = ctx.occ / ctx.occ.sum()
    rms = float(np.sqrt(w @ (phi - ctx.v_star) ** 2))
    _check("converges to the exact V* in occupancy-weighted RMS", rms < 2e-2,
           f"RMS|V_phi - V*| = {rms:.3e}")
    _check("...and pointwise wherever there is real data (occupancy > 1e-3)",
           err[ctx.occ > 1e-3].max() < 1e-3,
           f"max|V_phi - V*| = {err[ctx.occ > 1e-3].max():.3e}")
    # The tail does NOT converge, and that is the honest picture: a state seen with
    # weight 1e-4 gets a gradient 1e-4 as large, so it moves ~3000x slower. A critic can
    # therefore post a healthy EV and still be badly wrong exactly where you rarely go --
    # which is where an over-optimising policy is about to take you. Day 4's problem.
    tail = err[(ctx.occ > 0) & (ctx.occ < 1e-4)]
    if tail.size:
        print(f"  ... rare-state tail (occupancy < 1e-4, n={tail.size}) is still off by "
              f"up to {tail.max():.3f} after {ctx.fit_steps} steps")

    # ...and EV converges to a computable CEILING below 1: the residual is the
    # irreducible within-state variance of the return, which no critic can explain.
    m, w = b.mask, b.tok_w[b.mask]
    g = ctx.returns_to_go[m]
    vs = ctx.v_star[b.states[m]]
    ceiling = 1.0 - float(w @ (g - vs) ** 2) / float(w @ (g - w @ g) ** 2)
    _check("EV converges to the analytic ceiling, not to 1",
           abs(ev[-1] - ceiling) < 2e-3 and ceiling < 0.999,
           f"EV {ev[-1]:.5f} vs ceiling {ceiling:.5f}")
    print(f"  ... EV {ev[0]:.4f} -> {ev[-1]:.4f} (ceiling {ceiling:.4f}), "
          f"MSE {mse[0]:.4f} -> {mse[-1]:.4f} over {ctx.fit_steps} steps")


def stage_whiten(ctx: Ctx) -> None:
    """whiten -- centering is free, the std division is Dr. GRPO's bug, and the guard"""
    _fail_fast(ctx, "value_predict", "gae_advantages", "whiten")
    env, b = ctx.env, ctx.eb
    adv = ctx.advantages_of(ctx.phi_star(), ctx.lam)

    c = whiten(adv, b.tok_w, b.mask, center_only=True)
    s = whiten(adv, b.tok_w, b.mask, center_only=False)
    m, w = b.mask, b.tok_w[b.mask]
    _check("shape is preserved", c.shape == adv.shape and s.shape == adv.shape)
    _check("padded positions are exactly 0",
           np.all(c[~m] == 0.0) and np.all(s[~m] == 0.0))
    _check("centering makes the weighted mean 0", abs(float(w @ c[m])) < 1e-12,
           f"{float(w @ c[m]):.3e}")
    _check("scaling makes the weighted std 1",
           abs(float(np.sqrt(w @ (s[m] - w @ s[m]) ** 2)) - 1.0) < 1e-12)

    # Centering is a batch-level CONSTANT baseline: W3 D3 says it is free, and on an
    # enumerated batch "free" means the exact gradient is IDENTICAL, not similar.
    g_raw = score_function_gradient(env, ctx.theta, b.as_ppo_batch(adv))
    g_c = score_function_gradient(env, ctx.theta, b.as_ppo_batch(c))
    g_s = score_function_gradient(env, ctx.theta, b.as_ppo_batch(s))
    _check("centering leaves the exact gradient IDENTICAL",
           np.max(np.abs(g_raw - g_c)) < 1e-12,
           f"max|diff| = {np.max(np.abs(g_raw - g_c)):.3e}")
    sd = float(np.sqrt(w @ (adv[m] - w @ adv[m]) ** 2))
    _check("within ONE batch the std division is exactly a learning-rate rescale",
           np.max(np.abs(g_s * sd - g_c)) < 1e-11,
           f"std = {sd:.5f}, max|diff| = {np.max(np.abs(g_s * sd - g_c)):.3e}")

    # ACROSS groups it is not. Two groups, same means, spreads differing 5x.
    ga = np.array([[-1.0, 1.0]]); gb = np.array([[-5.0, 5.0]])
    ones = np.ones((1, 2)); mk = np.ones((1, 2), dtype=bool)
    pooled = whiten(np.concatenate([ga, gb], axis=1), np.ones((1, 4)) / 4,
                    np.ones((1, 4), dtype=bool), center_only=False)
    wa = whiten(ga, ones / 2, mk, center_only=False)
    wb = whiten(gb, ones / 2, mk, center_only=False)
    ratio_pooled = float(np.abs(pooled[0, 1]) / np.abs(pooled[0, 3]))
    ratio_group = float(np.abs(wa[0, 1]) / np.abs(wb[0, 1]))
    _check("pooled scaling PRESERVES the 5x spread between groups",
           abs(ratio_pooled - 0.2) < 1e-12, f"ratio {ratio_pooled:.6f}")
    _check("per-group scaling DESTROYS it -- the low-variance group is amplified 5x "
           "(this is Dr. GRPO's critique, in two lines)",
           abs(ratio_group - 1.0) < 1e-12, f"ratio {ratio_group:.6f}")

    # Degenerate batch: every advantage equal. std = 0.
    flat = np.where(b.mask, 3.0, 0.0)
    d = whiten(flat, b.tok_w, b.mask, center_only=False)
    _check("a degenerate batch returns FINITE values, not nan", np.all(np.isfinite(d)),
           f"{int((~np.isfinite(d)).sum())} non-finite")
    _check("...and they are exactly 0", np.max(np.abs(d)) == 0.0)


def stage_baseline_vs_bootstrap(ctx: Ctx) -> None:
    """THE RECONCILIATION: a critic cannot bias you as a baseline, only as a bootstrap"""
    _fail_fast(ctx, "value_table_init", "value_predict", "gae_advantages")
    env, b = ctx.env, ctx.eb
    rng = np.random.default_rng(27)

    def table(x):
        p = np.asarray(x, float).copy()
        p[env.n_states] = 0.0
        return p

    critics = {
        "zero": value_table_init(env),
        "exact V*": ctx.phi_star(),
        "offset V* + 25": table(ctx.phi_star() + ctx.c_offset),
        "pure noise": table(6.0 * rng.standard_normal(env.n_states + 1)),
        "anti -V*": table(-ctx.phi_star()),
    }

    # lam = 1: the critic appears ONLY as V(s_t) subtracted, i.e. as a baseline. It is
    # h_t-measurable by construction, so it cannot bias -- for ANY critic, however bad.
    for name, phi in critics.items():
        err = ctx.bias_of(phi, 1.0)
        _check(f"lam=1, critic '{name}': gradient == finite-difference truth",
               err < 1e-7, f"relative error {err:.3e}")

    # lam < 1: it bootstraps, and now a WRONG critic biases while a RIGHT one still does not.
    err_star = ctx.bias_of(ctx.phi_star(), ctx.lam_boot)
    _check(f"lam={ctx.lam_boot}, the EXACT critic is still unbiased "
           "(GAE with V^pi is unbiased at every lambda)",
           err_star < 1e-7, f"relative error {err_star:.3e}")

    for name in ("zero", "pure noise", "anti -V*"):
        err = ctx.bias_of(critics[name], ctx.lam_boot)
        _check(f"lam={ctx.lam_boot}, critic '{name}': now genuinely BIASED",
               err > 1e-3, f"relative error {err:.4f}")

    e0 = ctx.bias_of(critics["zero"], ctx.lam_boot)
    print(f"  ... zero critic: bias 0.000 at lam=1, {e0:.4f} at lam={ctx.lam_boot}. "
          "Same critic, same rollouts; only the estimator's use of it changed.")


def stage_decomposition(ctx: Ctx) -> None:
    """THE HEADLINE: gradient bias tracks the state-VARYING error, not the total error"""
    _fail_fast(ctx, "value_table_init", "value_predict", "gae_advantages",
                "fit_value_head")
    env, b = ctx.env, ctx.eb
    rng = np.random.default_rng(28)

    _, log = fit_value_head(env, b, ctx.returns_to_go, ctx.fit_steps, ctx.fit_lr,
                            snapshot_every=ctx.snapshot_every)

    rows = []
    for step, phi in zip(log["step"], log["phi"]):
        d = error_decomposition(phi, ctx.v_star, ctx.occ)
        rows.append(("fit@%d" % step, d, ctx.ev_of(phi), ctx.bias_of(phi, ctx.lam_boot)))
    for sig in ctx.noise_scales:
        phi = ctx.phi_star() + sig * rng.standard_normal(env.n_states + 1)
        phi[env.n_states] = 0.0
        d = error_decomposition(phi, ctx.v_star, ctx.occ)
        rows.append(("noise %.2f" % sig, d, ctx.ev_of(phi), ctx.bias_of(phi, ctx.lam_boot)))

    d0 = rows[0][1]
    _check("Pythagoras: ||e||^2 == e_bar^2 + ||e~||^2",
           abs(d0["e_norm"] ** 2 - d0["e_bar"] ** 2 - d0["e_tilde_norm"] ** 2) < 1e-10,
           f"{d0['e_norm']**2:.8f} vs {d0['e_bar']**2 + d0['e_tilde_norm']**2:.8f}")

    bias = [r[3] for r in rows]
    ev = [r[2] for r in rows]
    et = [r[1]["e_tilde_norm"] for r in rows]
    _check("bias is monotone in ||e~||, the state-varying error",
           spearman(bias, et) > 0.9, f"spearman = {spearman(bias, et):.4f}")
    _check("bias falls monotonically as explained variance rises",
           spearman(bias, ev) < -0.85, f"spearman = {spearman(bias, ev):.4f}")

    print("  ... critic                 |e_bar|    ||e||   ||e~||       EV     bias")
    for name, d, e, bi in rows:
        print(f"  ... {name:<18} {abs(d['e_bar']):8.4f} {d['e_norm']:8.4f} "
              f"{d['e_tilde_norm']:8.4f} {e:8.4f} {bi:8.4f}")

    # THE DISCRIMINATING PAIR. Two critics with EXACTLY the same ||e||: one whose error
    # is a pure level shift (e~ = 0 over emitted tokens), one whose error is pure ranking.
    # If bias tracked ||e||, these would be equally bad.
    c = 1.5
    sgn = rng.choice([-1.0, 1.0], size=env.n_states + 1)

    def pair(vstar):
        lvl = vstar + c
        lvl[env.n_states] = 0.0
        rk = vstar + c * sgn
        rk[env.n_states] = 0.0
        return lvl, rk

    lvl, rank = pair(ctx.v_star)
    dl = error_decomposition(lvl, ctx.v_star, ctx.occ)
    dr = error_decomposition(rank, ctx.v_star, ctx.occ)
    _check("the pair has IDENTICAL ||e||",
           abs(dl["e_norm"] - dr["e_norm"]) < 1e-12,
           f"{dl['e_norm']:.10f} vs {dr['e_norm']:.10f}")
    _check("wrong LEVEL has ~zero state-varying part",
           dl["e_tilde_norm"] < 1e-12, f"{dl['e_tilde_norm']:.3e}")
    _check("wrong RANKING has all of it in the state-varying part",
           abs(dr["e_tilde_norm"] - c) < 0.25 * c, f"{dr['e_tilde_norm']:.4f} vs c={c}")

    # Now the control that says what "free" actually depends on. Suppress eos, so every
    # rollout runs to t_max: the remaining horizon L-1-t becomes a function of t ALONE,
    # the level error's advantage shift -c lam^(L-1-t) becomes h_t-measurable, and it is
    # therefore a baseline -- EXACTLY free, even though the estimator is bootstrapping.
    fl, fr = pair(ctx.v_star_fixed)
    b_ref = ctx.bias_fixed(ctx.v_star_fixed, ctx.lam_boot)
    b_lvl = ctx.bias_fixed(fl, ctx.lam_boot)
    b_rank = ctx.bias_fixed(fr, ctx.lam_boot)
    _check("FIXED length, lam < 1: a wrong LEVEL costs literally nothing "
           "(identical to the exact critic, at the finite-difference floor)",
           abs(b_lvl - b_ref) < 1e-9, f"level {b_lvl:.3e} vs exact {b_ref:.3e}")
    _check("FIXED length, lam < 1: a wrong RANKING of the SAME ||e|| is catastrophic",
           b_rank > 1e6 * b_lvl, f"ranking {b_rank:.4f} vs level {b_lvl:.3e}")

    # Put eos back and the same level error starts costing you, because now the LENGTH
    # is an action -- which is the one thing an LLM environment always has.
    ranks = []
    for sd in range(5):
        r_ = ctx.v_star + c * np.random.default_rng(500 + sd).choice(
            [-1.0, 1.0], size=env.n_states + 1)
        r_[env.n_states] = 0.0
        ranks.append(ctx.bias_of(r_, ctx.lam_boot))
    b_lvl_v, b_rank_v = ctx.bias_of(lvl, ctx.lam_boot), float(np.mean(ranks))
    _check("VARIABLE length: the SAME level error now DOES bias you -- eos made the "
           "episode length an action",
           b_lvl_v > 0.05, f"bias {b_lvl_v:.4f} (was {b_lvl:.3e} at fixed length)")
    _check("...and a ranking error of equal size still costs more",
           b_rank_v > 1.4 * b_lvl_v, f"ranking {b_rank_v:.4f} vs level {b_lvl_v:.4f}")
    print(f"  ... equal ||e|| = {c}, lam = {ctx.lam_boot}:")
    print(f"  ...   fixed length     level {b_lvl:.2e}   ranking {b_rank:.4f}")
    print(f"  ...   variable length  level {b_lvl_v:.4f}      ranking {b_rank_v:.4f}")


def stage_negative_control(ctx: Ctx) -> None:
    """EV vs MSE, and where 'a constant is free' stops being true"""
    _fail_fast(ctx, "value_predict", "gae_advantages")
    env, b = ctx.env, ctx.eb
    c = ctx.c_offset
    phi0 = ctx.phi_star()
    phic = phi0 + c
    phic[env.n_states] = 0.0

    # 1) The instrument comparison, exactly as claimed.
    _check("MSE grows by EXACTLY c^2 -- unbounded, and it says 'catastrophic'",
           abs((ctx.mse_of(phic) - ctx.mse_of(phi0)) - c ** 2) < 1e-8,
           f"delta {ctx.mse_of(phic) - ctx.mse_of(phi0):.6f} vs c^2 = {c ** 2:.6f}")
    _check("EV is EXACTLY unchanged -- variance is shift-invariant",
           abs(ctx.ev_of(phic) - ctx.ev_of(phi0)) < 1e-12,
           f"{ctx.ev_of(phic):.12f} vs {ctx.ev_of(phi0):.12f}")

    g0 = score_function_gradient(env, ctx.theta, b.as_ppo_batch(ctx.advantages_of(phi0, 1.0)))
    gc = score_function_gradient(env, ctx.theta, b.as_ppo_batch(ctx.advantages_of(phic, 1.0)))
    _check("at lam = 1 the gradient is bit-for-bit independent of c",
           np.max(np.abs(g0 - gc)) < 1e-10, f"max|diff| = {np.max(np.abs(g0 - gc)):.3e}")
    _check("...and both equal the finite-difference truth",
           ctx.bias_of(phic, 1.0) < 1e-7, f"{ctx.bias_of(phic, 1.0):.3e}")

    # 2) Now bootstrap, and watch the claim break in a way the theorem PREDICTS.
    lam = ctx.lam_boot
    a0 = ctx.advantages_of(phi0, lam)
    ac = ctx.advantages_of(phic, lam)
    t = np.arange(b.T)[None, :]
    want = -c * lam ** (b.lengths[:, None] - 1 - t) * b.mask
    _check("at lam < 1 the offset shifts A_t by EXACTLY -c * lam^(L-1-t)",
           np.max(np.abs((ac - a0 - want)[b.mask])) < 1e-9,
           f"max|diff| = {np.max(np.abs((ac - a0 - want)[b.mask])):.3e}")
    _check("that shift depends on the LENGTH, hence on the actions, hence it biases",
           ctx.bias_of(phic, lam) > 10 * ctx.bias_of(phi0, lam),
           f"bias {ctx.bias_of(phic, lam):.5f} vs exact critic {ctx.bias_of(phi0, lam):.3e}")

    g_half = score_function_gradient(
        env, ctx.theta, b.as_ppo_batch(ctx.advantages_of(phi0 + c / 2, lam)))
    g_full = score_function_gradient(env, ctx.theta, b.as_ppo_batch(ac))
    g_base = score_function_gradient(env, ctx.theta, b.as_ppo_batch(a0))
    _check("...and the induced gradient error is exactly LINEAR in c",
           np.max(np.abs((g_full - g_base) - 2.0 * (g_half - g_base))) < 1e-9)

    # 3) The reconciliation. Measured over the states the BOOTSTRAP actually reads --
    # which includes TERMINAL, whose value is pinned at 0 and cannot be shifted -- the
    # "constant" offset is not constant at all. W3 D4's theorem never said otherwise.
    occ_full = np.zeros(env.n_states + 1)
    valid = np.arange(b.T + 1)[None, :] <= b.lengths[:, None]
    wfull = np.broadcast_to(b.weights[:, None], (b.B, b.T + 1))
    np.add.at(occ_full, b.value_ids[valid], wfull[valid])
    d_tok = error_decomposition(phic, phi0, ctx.occ)
    d_all = error_decomposition(phic, phi0, occ_full)
    _check("over emitted tokens only, the offset looks like a pure LEVEL error",
           d_tok["e_tilde_norm"] < 1e-12, f"||e~|| = {d_tok['e_tilde_norm']:.3e}")
    _check("over the states the bootstrap reads, it is a state-VARYING error",
           d_all["e_tilde_norm"] > 0.1 * c, f"||e~|| = {d_all['e_tilde_norm']:.4f}")
    print("  ... 'wrong level is free' is a statement about the BASELINE role. Under "
          "bootstrapping\n  ... the pinned terminal row means a uniform offset is not "
          "uniform, and it costs you.")


def stage_zero_baseline(ctx: Ctx) -> None:
    """zero_baseline_probe -- the decisive experiment, run four ways"""
    _fail_fast(ctx, "value_table_init", "value_predict", "gae_advantages",
                "zero_baseline_probe")
    env, b = ctx.env, ctx.eb

    depths = np.array([len(s) for s in env.states] + [0])
    phi_depth = np.zeros(env.n_states + 1)
    for d in range(env.t_max):
        sel = (depths == d)
        sel[env.n_states] = False
        w = ctx.occ * sel
        if w.sum() > 0:
            phi_depth[sel] = float(w @ ctx.v_star) / w.sum()
    phi_depth[env.n_states] = 0.0

    anti = -ctx.phi_star()
    anti[env.n_states] = 0.0
    variants = {
        "no baseline": ctx.advantages_of(value_table_init(env), 1.0),
        "depth-only":  ctx.advantages_of(phi_depth, 1.0),
        "exact V*":    ctx.advantages_of(ctx.phi_star(), 1.0),
        "anti-critic": ctx.advantages_of(anti, 1.0),
    }
    var = zero_baseline_probe(env, ctx.theta, b, variants)

    _check("returns one number per variant", set(var) == set(variants), str(sorted(var)))
    _check("all variances are plain positive floats",
           all(type(v) is float and v > 0 for v in var.values()), str(var))

    # Every variant here is a pure baseline at lam = 1, so they all have the SAME
    # expectation. That is what makes the variance comparison meaningful.
    for name, adv in variants.items():
        g = score_function_gradient(env, ctx.theta, b.as_ppo_batch(adv))
        _check(f"'{name}' is unbiased (same gradient, different variance)",
               grad_error(g, ctx.g_true) < 1e-7, f"{grad_error(g, ctx.g_true):.3e}")

    _check("a healthy critic REDUCES gradient variance",
           var["exact V*"] < var["no baseline"],
           f"{var['exact V*']:.4f} < {var['no baseline']:.4f}")
    _check("a partially-specified (depth-only) critic helps, but less",
           var["exact V*"] < var["depth-only"] < var["no baseline"],
           f"{var['exact V*']:.4f} < {var['depth-only']:.4f} < {var['no baseline']:.4f}")
    _check("a mis-specified (anti-correlated) critic INCREASES it -- which is what the "
           "probe is for",
           var["anti-critic"] > var["no baseline"],
           f"{var['anti-critic']:.4f} > {var['no baseline']:.4f}")

    base = var["no baseline"]
    print("  ... " + "  ".join(f"{k}: {v:.3f} ({100 * (v / base - 1):+.1f}%)"
                               for k, v in var.items()))


def stage_loss(ctx: Ctx) -> None:
    """instructgpt_loss -- term signs, the ordering trap, and sum-vs-mean"""
    _fail_fast(ctx, "value_table_init", "value_predict", "gae_advantages",
                "shape_rewards", "value_targets", "value_loss", "fit_value_head",
                "whiten", "instructgpt_loss")
    env, b = ctx.env, ctx.ebr
    rewards = shape_rewards(env, b, ctx.theta, ctx.reward, ctx.beta, "k1")
    phi, _ = fit_value_head(env, b, ctx.returns_to_go, 60, ctx.fit_lr)
    values_old = value_predict(phi, b.value_ids)

    kw = dict(gamma=ctx.gamma, lam=ctx.lam, eps=ctx.eps, value_clip_eps=0.2)
    tot0, p0 = instructgpt_loss(env, ctx.theta, phi, b, rewards, values_old,
                                c1=0.0, c2=0.0, **kw)
    tot1, p1 = instructgpt_loss(env, ctx.theta, phi, b, rewards, values_old,
                                c1=1.0, c2=0.0, **kw)
    tot2, p2 = instructgpt_loss(env, ctx.theta, phi, b, rewards, values_old,
                                c1=0.0, c2=1.0, **kw)

    for key in ("clip", "vf", "entropy", "total", "clip_frac", "oob_frac",
                "vf_clip_frac", "ev", "adv_mean", "adv_std", "reduce"):
        _check(f"parts carries '{key}'", key in p0)
    _check("with c1 = c2 = 0 the total is exactly -L_CLIP",
           abs(tot0 + p0["clip"]) < 1e-12, f"{tot0:.10f} vs {-p0['clip']:.10f}")
    _check("the value term ADDS to the loss (it is minimised)",
           abs((tot1 - tot0) - p0["vf"]) < 1e-10 and p0["vf"] > 0,
           f"delta {tot1 - tot0:.8f} vs vf {p0['vf']:.8f}")
    _check("the entropy term SUBTRACTS from the loss (it is a bonus)",
           abs((tot2 - tot0) + p0["entropy"]) < 1e-10 and p0["entropy"] > 0,
           f"delta {tot2 - tot0:.8f} vs H {p0['entropy']:.8f}")
    _check("entropy is per-token and below log V",
           0 < p0["entropy"] < np.log(env.vocab), f"H = {p0['entropy']:.5f}")
    _check("clip_frac <= oob_frac (only half the out-of-band cells lose their gradient)",
           p0["clip_frac"] <= p0["oob_frac"] + 1e-15,
           f"{p0['clip_frac']:.4f} vs {p0['oob_frac']:.4f}")

    # THE ORDERING TRAP: targets are built from the raw advantages, BEFORE whitening.
    # If you whiten first, the critic regresses onto a per-batch-rescaled quantity --
    # it trains, it never nans, and it is wrong.
    _, pw = instructgpt_loss(env, ctx.theta, phi, b, rewards, values_old,
                             c1=1.0, whiten_mode="scale", **kw)
    _, pc = instructgpt_loss(env, ctx.theta, phi, b, rewards, values_old,
                             c1=1.0, whiten_mode="center", **kw)
    _check("whitening does NOT change the value loss (targets were built first)",
           abs(pw["vf"] - p1["vf"]) < 1e-12 and abs(pc["vf"] - p1["vf"]) < 1e-12,
           f"none {p1['vf']:.8f} / center {pc['vf']:.8f} / scale {pw['vf']:.8f}")
    _check("whitening leaves the advantages centered", abs(pw["adv_mean"]) < 1e-10,
           f"{pw['adv_mean']:.3e}")
    _check("scale-whitening leaves them at unit std", abs(pw["adv_std"] - 1.0) < 1e-10,
           f"{pw['adv_std']:.8f}")
    _check("every whitening mode returns finite numbers",
           all(np.isfinite([pw["total"], pc["total"], p0["total"]])))

    ev_ref = explained_variance(values_old[:, :-1][b.mask],
                               value_targets(ctx.advantages_of(phi, ctx.lam, b, rewards),
                                             values_old)[b.mask], b.tok_w[b.mask])
    _check("parts['ev'] is the critic's explained variance against its own targets",
           abs(p0["ev"] - ev_ref) < 1e-10, f"{p0['ev']:.8f} vs {ev_ref:.8f}")

    # SUM vs MEAN. At theta == theta_old every rho is 1, so L_CLIP is just E[sum_t A_t]
    # and the two conventions can be compared in closed form.
    kws = dict(gamma=ctx.gamma, lam=ctx.lam, eps=None, c1=0.0, c2=0.0)
    _, ps = instructgpt_loss(env, ctx.theta, phi, b, rewards, values_old,
                             reduce="sum", **kws)
    _, pm = instructgpt_loss(env, ctx.theta, phi, b, rewards, values_old,
                             reduce="mean", **kws)
    adv = ctx.advantages_of(phi, ctx.lam, b, rewards)
    want_sum = float((b.weights * adv.sum(axis=1)).sum())
    want_mean = float((b.weights / b.lengths * adv.sum(axis=1)).sum())
    _check("reduce='sum' is sum_b w_b sum_t A_bt", abs(ps["clip"] - want_sum) < 1e-10,
           f"{ps['clip']:.10f} vs {want_sum:.10f}")
    _check("reduce='mean' is sum_b (w_b / L_b) sum_t A_bt",
           abs(pm["clip"] - want_mean) < 1e-10, f"{pm['clip']:.10f} vs {want_mean:.10f}")
    _check("the two conventions genuinely differ", abs(ps["clip"] - pm["clip"]) > 1e-6,
           f"sum {ps['clip']:.6f} vs mean {pm['clip']:.6f}")
    _check("the choice is recorded, not implicit",
           ps["reduce"] == "sum" and pm["reduce"] == "mean")

    eff = b.weights / b.lengths
    eff = eff / eff.sum()
    short = b.lengths <= 2
    _check("dividing by length silently UPWEIGHTS short completions "
           "(tomorrow's length exploitation, one day early)",
           eff[short].sum() > b.weights[short].sum() + 1e-6,
           f"weight on len<=2: {b.weights[short].sum():.4f} -> {eff[short].sum():.4f}")


def stage_c1_sweep(ctx: Ctx) -> None:
    """shared trunk: the value loss competes with the policy for the same weights"""
    _fail_fast(ctx, "gae_advantages", "value_targets")
    c1_list = (0.0, 0.1, 0.5, 5.0)
    out = shared_trunk_sweep(ctx.env, ctx.theta, ctx.reward, c1_list,
                             n_iters=ctx.trunk_iters, lam=ctx.lam, eps=ctx.eps)
    j0 = float(exact_J(ctx.env, ctx.theta, ctx.reward))
    finals = {c: out[c]["J_final"] for c in c1_list}
    print("  ... J(theta_0) = %.4f   " % j0
          + "  ".join(f"c1={c}: {v:.4f}" for c, v in finals.items()))

    gain = {c: finals[c] - j0 for c in c1_list}
    _check("with c1 = 0 (decoupled) the policy improves", gain[0.0] > 0.05,
           f"{j0:.4f} -> {finals[0.0]:.4f}")
    _check("c1 across the usual range costs essentially nothing",
           all(abs(gain[c] - gain[0.0]) < 0.15 * gain[0.0] for c in (0.1, 0.5)),
           f"gain {gain[0.0]:.4f} -> {gain[0.1]:.4f} (c1=0.1), {gain[0.5]:.4f} (c1=0.5)")
    _check("a large c1 DESTROYS it -- the value loss competes for the trunk",
           gain[5.0] < 0.1 * gain[0.0],
           f"c1=5 keeps {100 * gain[5.0] / gain[0.0]:.1f}% of the gain")
    # Note the shape of the damage: it is NOT monotone-and-gradual. That is why c1 reads
    # as a formality right up until it doesn't, and why "we tuned c1 and it made no
    # difference" is not evidence that the coupling is harmless.
    print("  ... gain over J(theta_0): " + "  ".join(f"c1={c}: {gain[c]:+.4f}"
                                                     for c in c1_list))


STAGES = [
    ("value_table", stage_value_table),
    ("gae", stage_gae),
    ("shaping", stage_shaping),
    ("targets", stage_targets),
    ("fit", stage_fit),
    ("whiten", stage_whiten),
    ("baseline_vs_bootstrap", stage_baseline_vs_bootstrap),
    ("decomposition", stage_decomposition),
    ("negative_control", stage_negative_control),
    ("zero_baseline", stage_zero_baseline),
    ("loss", stage_loss),
    ("c1_sweep", stage_c1_sweep),
]


# ---- runner (copy verbatim; nothing below needs editing) -------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, fn in STAGES:
            print(f"  {name:<22} {fn.__doc__}")
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
