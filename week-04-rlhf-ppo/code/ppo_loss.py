"""
Week 4 Day 2 -- the PPO objective: importance ratio, clipping, epochs.

One question generates this whole file: HOW MANY GRADIENT STEPS CAN I TAKE ON DATA
I ALREADY COLLECTED? Rollouts cost 10-100x a gradient step, so taking one step per
batch wastes the expensive thing. Reuse the batch -> the instant you take step 2 you
are off-policy -> you need a ratio -> the ratio's variance explodes -> you clip it.

Three results to come out of here as ASSERTIONS, not claims:

  1. grad L_IS at theta == theta_old IS Week 3's policy gradient, elementwise, and
     both match the enumerated finite-difference gradient of `exact_J`. PPO is
     REINFORCE-with-advantages plus permission to keep going.
  2. L_CLIP <= L_IS pointwise, with gradient EXACTLY 0.0 in two of the four cells of
     the (sign of A) x (side of the band) table.
  3. Clipping bounds the STEP, never the WALK. You will measure a run whose clip
     fraction stays ~5% while KL(pi_theta || pi_ref) walks to ~4.7 nats, next to a
     beta-penalised run that plateaus below 1 nat.

Ground truth is enumeration, imported from Day 1: `exact_J`, `exact_sequence_kl`, and
`enumerate_sequences`. Nothing here is established by a plot.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-04-rlhf-ppo/code/ppo_loss.py
  one stage           ...  ppo_loss.py grad_regression
  a few stages        ...  ppo_loss.py logratios surrogate
  small/fast fixtures ...  ppo_loss.py --fast
  list the stages     ...  ppo_loss.py --list

Build order (each group is runnable the moment it is written):

  1. token_logratios                      -> stage `logratios`
  2. is_surrogate                         -> stage `surrogate`
  3. surrogate_gradient                   -> stage `grad_regression`   <- the big one
  4. clipped_surrogate                    -> stages `clip_table`, `clip_bounds`
  5. sequence_ratio_stats                 -> stage `ratio_blowup`
  6. ppo_update                           -> stage `epoch_sweep`
  7. drift_experiment                     -> stages `drift`, `frontier`

SHAPE TRAP, DELIBERATE: the fixture env is vocab=5, t_max=3, so V=5, T=3 and B=85 are
all different numbers. Every W1/W3/W4 session so far has had at least one bug hidden by
a square shape (T==V in the CE reshape, S==A in the MDP, the (n,2) index on a square
psi). Here a transposed (B,T) vs (B,V) cannot pass silently.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from toy_rlhf import (  # noqa: E402
    TokenEnv,
    enumerate_sequences,
    exact_J,
    exact_sequence_kl,
    log_softmax_np,
    sample_sequences,
    sequence_logprob,
    softmax_np,
    state_occupancy,
)


# ---------------------------------------------------------------------------
# The rollout batch (provided -- READ IT, you build against this contract)
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """One frozen rollout batch. Everything in here is computed ONCE, under
    `theta_old`, and never recomputed during the reuse loop -- which is not a
    performance choice, it is the definition of the algorithm. `logp_old` and
    `advantages` being frozen fields of an immutable-by-convention object is how
    this file makes "theta_old and the advantages are frozen" impossible to get
    wrong.

    states     (B, T) int64  -- state id of the prefix seq[:t]; 0 where padded
    actions    (B, T) int64  -- token emitted at step t;         0 where padded
    mask       (B, T) bool   -- True where the step is real (sequences vary in length)
    advantages (B, T) float  -- Psi_t. Terminal reward only, so this is constant
                                along t within a sequence (the value head is Day 3).
    logp_old   (B, T) float  -- log pi_old(a_t | s_t), cached at rollout time. In a
                                real stack this is the ONLY thing you keep from the
                                rollout worker, and caching it is why you do not need
                                pi_old resident during the optimisation phase.
    weights    (B,)   float  -- per-sequence weight, sums to 1. This is 1/B for a
                                sampled batch and p_theta_old(y) for an ENUMERATED
                                batch, which is the trick that turns every estimator
                                in this file into an exact expectation.
    returns    (B,)   float  -- the (possibly KL-shaped) sequence reward
    """

    states: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    advantages: np.ndarray
    logp_old: np.ndarray
    weights: np.ndarray
    returns: np.ndarray
    seqs: list

    @property
    def B(self) -> int:
        return self.mask.shape[0]

    @property
    def n_tokens(self) -> int:
        return int(self.mask.sum())

    def subset(self, idx) -> "Batch":
        """A minibatch. Weights are RENORMALISED to sum to 1, so a minibatch is an
        estimate of the same expectation rather than a fraction of it."""
        idx = np.asarray(idx, dtype=np.int64)
        w = self.weights[idx]
        return Batch(self.states[idx], self.actions[idx], self.mask[idx],
                     self.advantages[idx], self.logp_old[idx], w / w.sum(),
                     self.returns[idx], [self.seqs[i] for i in idx])


def _pack(env, seqs, theta_old, returns, weights, center):
    B, T = len(seqs), env.t_max
    states = np.zeros((B, T), dtype=np.int64)
    actions = np.zeros((B, T), dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    logp_old = np.zeros((B, T))
    lp = log_softmax_np(theta_old)
    for b, s in enumerate(seqs):
        for t, a in enumerate(s):
            sid = env.state_id(s[:t])
            states[b, t] = sid
            actions[b, t] = a
            mask[b, t] = True
            logp_old[b, t] = lp[sid, a]
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    base = float(weights @ returns) if center else 0.0
    adv = np.where(mask, (returns - base)[:, None], 0.0)
    return Batch(states, actions, mask, adv, logp_old, weights, returns, list(seqs))


def _shaped_returns(env, seqs, theta_old, reward_fn, theta_ref, beta):
    """r(y) - beta * [log p_theta_old(y) - log p_ref(y)].

    The KL penalty as a SEQUENCE-level reward shaping term, which is what
    InstructGPT Eq. (2) literally is. Note it uses only quantities available at
    rollout time -- `log p_ref` is why the reference model is queried during the
    rollout and can then be dropped. Day 3 moves this to a per-TOKEN shaping term
    and proves the two agree, using the occupancy identity from today's quiz.
    """
    r = np.array([reward_fn(s) for s in seqs], dtype=float)
    if beta and theta_ref is not None:
        lp = np.array([sequence_logprob(env, theta_old, s) for s in seqs])
        lp_ref = np.array([sequence_logprob(env, theta_ref, s) for s in seqs])
        r = r - beta * (lp - lp_ref)
    return r


def collect_batch(env, theta_old, rng, n_seqs, reward_fn, theta_ref=None,
                  beta=0.0, center=True) -> Batch:
    """Sampled rollouts. `weights = 1/B`, so estimators built on it are Monte Carlo."""
    seqs = sample_sequences(env, theta_old, n_seqs, rng)
    returns = _shaped_returns(env, seqs, theta_old, reward_fn, theta_ref, beta)
    return _pack(env, seqs, theta_old, returns, np.full(len(seqs), 1.0 / len(seqs)), center)


def exact_batch(env, theta_old, reward_fn, theta_ref=None, beta=0.0, center=False) -> Batch:
    """EVERY terminating sequence, weighted by its exact probability under theta_old.

    This is the week's ground-truth pattern applied to the estimator itself: an
    estimator evaluated on this batch is not an estimate, it is the exact
    expectation. That is what lets `grad_regression` assert agreement with a finite
    difference of `exact_J` at 1e-11 instead of "within two standard errors".
    """
    seqs = enumerate_sequences(env)
    returns = _shaped_returns(env, seqs, theta_old, reward_fn, theta_ref, beta)
    p = np.exp([sequence_logprob(env, theta_old, s) for s in seqs])
    return _pack(env, seqs, theta_old, returns, p, center)


def score_function_gradient(env, theta, batch) -> np.ndarray:
    """WEEK 3, TRANSPLANTED (provided): sum_t Psi_t grad_theta log pi(a_t|s_t), using
    grad_z log pi(a|s) = e_a - p(.|s) -- W1's `dCE/dz = p - e_y` with the sign flipped.

    This is the REFERENCE for stage `grad_regression`. It knows nothing about ratios,
    so if your `surrogate_gradient` agrees with it at theta == theta_old, that
    agreement means something.
    """
    p = softmax_np(theta)
    grad = np.zeros_like(theta)
    m = batch.mask
    coef = (batch.advantages * batch.weights[:, None])[m]
    s, a = batch.states[m], batch.actions[m]
    np.add.at(grad, (s, a), coef)
    np.add.at(grad, s, -coef[:, None] * p[s])
    return grad


def finite_difference_gradient(env, theta, reward_fn, h=1e-5) -> np.ndarray:
    """Central finite differences of the ENUMERATED J. The structurally independent
    ground truth: it never touches a batch, a ratio, or a log-prob of a sampled token."""
    g = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            tp, tm = theta.copy(), theta.copy()
            tp[i, j] += h
            tm[i, j] -= h
            g[i, j] = (exact_J(env, tp, reward_fn) - exact_J(env, tm, reward_fn)) / (2 * h)
    return g


def weighted_quantile(x, w, q) -> float:
    """Quantile of `x` under the probability weights `w` (provided)."""
    o = np.argsort(x)
    return float(np.interp(q, np.cumsum(np.asarray(w)[o]), np.asarray(x)[o]))


def token_ratio_stats(env, theta, theta_old) -> dict:
    """Occupancy-weighted PER-TOKEN ratio statistics (provided).

    Uses Day 1's `state_occupancy`, i.e. the same weighting that made the sequence KL
    come out right. `log_std` here is the sigma_1 that stage `ratio_blowup` multiplies
    by sqrt(T), and `frac_in_band` says what fraction of individual tokens satisfy
    PPO's |rho - 1| <= eps. The punchline of that stage needs BOTH numbers: almost
    every token inside the band, and the sequence ratio still degenerate.
    """
    d = state_occupancy(env, theta_old)
    pi_o, pi_n = softmax_np(theta_old), softmax_np(theta)
    w = d[:, None] * pi_o
    w = w / w.sum()
    lr = np.log(pi_n) - np.log(pi_o)
    mean = float((w * lr).sum())
    return dict(log_std=float(np.sqrt((w * (lr - mean) ** 2).sum())),
                frac_in_band=float((w * (np.abs(pi_n / pi_o - 1.0) <= 0.2)).sum()))


# ---------------------------------------------------------------------------
# Exercises (TODO bodies)
# ---------------------------------------------------------------------------


def token_logratios(env, theta, batch) -> np.ndarray:
    """Per-token log rho_t = log pi_theta(a_t|s_t) - log pi_old(a_t|s_t), shape (B, T).

    STAY IN LOG SPACE. Never form pi_theta / pi_old as a quotient of probabilities:
    both are exp() of numbers that can be very negative, and the quotient of two
    underflowed zeros is nan while the difference of their logs is exact. This is the
    same rule as D1's "never exponentiate an unnormalised log-weight", one level down.

    Return 0.0 at masked-out (padded) positions. Not because 0 is meaningful there --
    rho = 1 at a position that does not exist is nonsense -- but because every consumer
    multiplies by `batch.advantages`, which is also 0 there, and a nan would poison the
    sum. Assert-worthy consequence: your answer must be EXACTLY the zero matrix when
    theta is theta_old, at every position, padded or not.
    """
    states = batch.states
    actions = batch.actions
    log_pi_old = batch.logp_old
    log_pi_theta_all = log_softmax_np(theta[states], axis=-1)
    log_pi_theta_actions = np.squeeze(np.take_along_axis(log_pi_theta_all, np.expand_dims(actions, axis=-1), axis=-1), axis=-1)
    return np.where(batch.mask, log_pi_theta_actions - log_pi_old, 0)

def is_surrogate(logratios, batch) -> float:
    """L_IS = sum_b w_b sum_t rho_{b,t} A_{b,t}.

    Note the two different reductions: a WEIGHTED sum over sequences (w_b is a
    probability) and a PLAIN sum over t (the policy-gradient theorem has sum_t, not
    mean_t). Getting this pair backwards is the single most common normalisation bug
    in PPO implementations, and it is silent -- it only rescales the gradient, so it
    looks like a learning-rate problem forever.

    Consequence worth predicting before you check it: because A_{b,t} is constant along
    t, this equals sum_b w_b * len(y_b) * A_b. So even with perfectly centred
    advantages the value is NOT zero -- it is zero only if LENGTH is uncorrelated with
    advantage. Stage `surrogate` measures the residual, and that residual is the
    length-bias of RLHF showing up two days before Day 4 names it.
    """
    return float(np.sum(batch.weights * np.sum(np.exp(logratios) * batch.advantages, axis=-1)))


def clipped_surrogate(logratios, batch, eps):
    """L_CLIP = sum_b w_b sum_t min( rho A , clip(rho, 1-eps, 1+eps) A ), plus
    instruments. Returns `(value, diagnostics)` where diagnostics is a dict with:

      clip_frac  fraction of real tokens where the `min` selected the CLIPPED branch,
                 i.e. where the gradient has been killed.
      oob_frac   fraction of real tokens with |rho - 1| > eps.
      frac_high  killed AND rho > 1+eps   (these are the A > 0 cells)
      frac_low   killed AND rho < 1-eps   (these are the A < 0 cells)

    `clip_frac` and `oob_frac` are DIFFERENT numbers and both get logged in real
    stacks. Every out-of-band token is a candidate but only half of them (by sign of A)
    actually lose their gradient, so `clip_frac <= oob_frac` always -- an invariant with
    teeth, asserted in `clip_bounds`. If you find they are equal you have clipped both
    sides, which is the classic wrong implementation and the one the 2x2 table exists
    to rule out.

    All fractions are over `batch.n_tokens`, not over B*T -- padding is not a token.
    """
    if eps is None:
        diagnostics = {'oob_frac': 0, 'frac_high': 0, 'frac_low': 0, 'clip_frac': 0}
        return (is_surrogate(logratios, batch), diagnostics)
    else:
        rho = np.exp(logratios)
        oob_frac = np.where(abs(rho-1)>eps, 1, 0).sum()/batch.n_tokens
        
        frac_low = np.where((rho < 1-eps) & (batch.advantages < 0), 1, 0).sum()/batch.n_tokens
        frac_high = np.where((rho > 1+eps) & (batch.advantages > 0), 1, 0).sum()/batch.n_tokens
        clip_frac = frac_high+frac_low
        diagnostics = {'oob_frac': float(oob_frac), 'frac_high': float(frac_high), 'frac_low': float(frac_low), 'clip_frac': float(clip_frac)}

        unclipped = rho * batch.advantages
        clipped = np.clip(rho, 1-eps, 1+eps) * batch.advantages 

        inner = np.sum(np.minimum(clipped, unclipped), axis=-1)
        l_clip = float(np.sum(batch.weights * inner))
        return (l_clip, diagnostics)   


def surrogate_gradient(env, theta, batch, eps=None) -> np.ndarray:
    """Analytic gradient of L_IS (eps=None) or L_CLIP (eps=float), shape == theta.shape.

    Two facts do all the work:
        grad_theta rho_t = rho_t grad_theta log pi_theta(a_t|s_t)
        grad_{z(s)} log pi(a|s) = e_a - p(.|s)
    so the contribution of token (b,t) is  w_b * gate * rho * A * (e_a - p(.|s))
    scattered into row s.

    `gate` is the entire content of clipping. It is 1 everywhere when eps is None. When
    eps is a float it is 0 exactly on the two cells where the update would push
    FURTHER out of the band:
        A > 0 and rho > 1+eps      (already raised this good token enough)
        A < 0 and rho < 1-eps      (already suppressed this bad token enough)
    and 1 on the other two, where the gradient flows at FULL strength -- including the
    A < 0, rho > 1+eps cell, which is unbounded and is where PPO takes its largest
    steps. At rho exactly 1+eps the two branches are equal; let the gradient flow
    (use strict inequalities), and know that you have made that choice.

    Two implementation traps, both of which the checks will find:
      * this is a SCATTER-ADD over repeated (state, token) pairs. `grad[s, a] += c`
        with fancy indexing gathers-then-scatters and keeps only the last write for
        duplicated indices. Use `np.add.at`, and pass the index as a TUPLE `(s, a)`.
        (D1's bug, in its most dangerous form: the result looks plausible.)
      * the accumulator must be FLOAT. `np.zeros(shape, dtype=np.int64)` truncates
        every contribution to 0 and reports a perfectly clean all-zero gradient --
        W3 D2's `np.long` bug, which cost an hour.
    """
    rho = np.exp(token_logratios(env, theta, batch))
    pi = softmax_np(theta, axis=-1)
    grad = np.zeros_like(theta, dtype=np.float64)
    if eps is None:
        gate = 1
    else:
        gate = np.where(((batch.advantages > 0) & (rho > 1+eps))|((batch.advantages < 0) & (rho < 1-eps)), 0, 1)
    mult_factor = batch.weights[:, None] * gate * rho * batch.advantages
    np.add.at(grad, batch.states, np.expand_dims(mult_factor, axis=-1) * (np.eye(grad.shape[-1])[batch.actions] - pi[batch.states]))
    return grad

def ppo_update(env, theta_old, batch, n_epochs=1, n_minibatches=1, eps=0.2,
               lr=0.1, rng=None):
    """The reuse loop. Returns `(theta_new, log)`.

    for each of n_epochs:            shuffle the sequences
      for each of n_minibatches:     ascend  theta += lr * surrogate_gradient(...)

    ASCENT, not descent -- this is an objective. (W3 D3: `theta += numpy_array` is not
    in-place and silently rebinds; here theta is plain numpy so it is fine, but write
    `theta = theta + lr * g` and mean it.)

    `theta_old` and `batch` are never modified. Do not recompute the advantages or
    `logp_old` inside the loop, and note that you CANNOT accidentally do so, because
    they live in the frozen batch -- that is the design.

    `log` is a dict of LISTS OF PYTHON FLOATS, one entry per minibatch step:
    'surrogate', 'clip_frac', 'oob_frac'. Floats, not numpy scalars and definitely not
    tensors -- W3 D3's `history.append(tensor)` leaked the graph, and the check here
    asserts `type(x) is float` so the habit is enforced rather than remembered.

    Note what the two loop bounds mean: n_epochs * n_minibatches is how many gradient
    steps you take away from theta_old, so "we only use 4 epochs" is not a safety
    argument on its own.
    """
    theta = theta_old.copy()
    log = {'surrogate': [], 'clip_frac': [], 'oob_frac':[]}
    for _ in range(n_epochs):
        
        split = np.array_split(rng.permutation(batch.B), n_minibatches)
        for i in range(n_minibatches):
            minibatch = batch.subset(split[i])
            logratio = token_logratios(env, theta, minibatch)
            l_clip, diag = clipped_surrogate(logratio, minibatch, eps)
            log["surrogate"].append(l_clip)
            log["clip_frac"].append(diag['clip_frac'])
            log["oob_frac"].append(diag['oob_frac'])
            grad = surrogate_gradient(env, theta, minibatch, eps)
            theta = theta + lr * grad
    return (theta, log)


def sequence_ratio_stats(env, theta, theta_old) -> dict:
    """The distribution of the WHOLE-SEQUENCE ratio rho(y) = prod_t rho_t, EXACTLY.

    No sampling: enumerate every terminating sequence, weight by p_theta_old(y), and
    read the distribution off directly. Return a dict with

      mean      E_old[rho]                      -- must be 1, and it is not an
                                                   approximation: sum_y p_old (p_new/p_old)
                                                   = sum_y p_new = 1. A likelihood ratio
                                                   has mean exactly 1 under the sampling
                                                   distribution, always.
      m2        E_old[rho^2]
      chi2      m2 - 1 = chi^2(p_new || p_old)  -- the divergence IS divergence controlling
      ess_frac  1 / m2                             the variance of every IS estimator;
                                                   ESS/n = 1/(1+chi^2) because E[rho] = 1.
      median    the 50th percentile of rho under p_old
      p99       the 99th percentile
      log_std   std of log rho under p_old      -- the sigma of the log-normal
      mean_len  E_old[len(y)]

    The point of the stage: `mean` stays pinned at 1 while `median` sinks and `p99`
    climbs. That gap is the whole argument against a sequence-level ratio, and it is
    not a variance-is-large hand-wave -- it is a statement about a log-normal, whose
    mean is held up by a vanishing fraction of the samples.
    """
    sequences = enumerate_sequences(env)
    logprob_old = np.array([sequence_logprob(env, theta_old, seq) for seq in sequences])
    logprob_new = np.array([sequence_logprob(env, theta, seq) for seq in sequences])
    p_old = np.exp(logprob_old)
    rho = np.exp(logprob_new - logprob_old)
    stats = {}
    stats['mean'] = float((p_old * rho).sum())
    stats['m2'] = float((p_old * rho ** 2).sum())
    stats['chi2'] = stats['m2']-1
    stats['ess_frac'] = 1/stats['m2']
    lengths = [len(seq) for seq in sequences]
    stats['mean_len'] = (p_old * lengths).sum()
    log_rho = logprob_new - logprob_old
    log_mean = float((p_old * log_rho).sum())
    log_m2 = float((p_old * log_rho ** 2).sum())
    stats['log_std'] = float(np.sqrt(log_m2 - log_mean ** 2))
    stats['median'] = weighted_quantile(rho, p_old, 0.5)
    stats['p99'] = weighted_quantile(rho, p_old, 0.99)

    return stats





def drift_experiment(env, theta_init, theta_ref, reward_fn, mode="both", n_iters=20,
                     beta=1.0, eps=0.2, n_epochs=4, n_minibatches=4, lr=0.1,
                     n_seqs=256, rng=None):
    """THE HEADLINE. N iterations of the real loop, under four regimes:

        mode = "clip"     clip only, no KL penalty       (eps=eps,  beta=0)
        mode = "penalty"  KL penalty only, no clip       (eps=None, beta=beta)
        mode = "both"     what InstructGPT actually ships
        mode = "none"     neither -- the negative control

    Each iteration: collect a FRESH batch under the current theta (so theta_old is
    reset every iteration -- that reset is the reason clipping cannot bound the walk),
    run `ppo_update`, then measure. Returns a dict of lists, one entry per iteration:

      kl_ref    exact_sequence_kl(env, theta_new, theta_ref)   <- the WALK
      kl_step   exact_sequence_kl(env, theta_new, theta_old)   <- the STEP
      clip_frac mean clip fraction over the iteration's minibatch steps
      J         exact_J(env, theta_new, reward_fn)             <- unshaped reward

    plus 'theta' = the final parameters.

    Log `kl_ref` against `theta_ref`, NOT against the previous iteration's theta. The
    entire result is that the sum of small steps is not a small walk, and you cannot
    see that if your only instrument is the step.
    """
    theta = theta_init.copy()
    report = {'kl_ref': [], 'kl_step': [], 'clip_frac': [], 'J': []}
    for _ in range(n_iters):
        theta_old = theta.copy()
        
        if mode == 'clip':
            batch = collect_batch(env, theta_old, rng, n_seqs, reward_fn, theta_ref, 0)
            theta, log = ppo_update(env, theta_old, batch, n_epochs, n_minibatches, eps, lr, rng)
        elif mode == 'penalty':
            batch = collect_batch(env, theta_old, rng, n_seqs, reward_fn, theta_ref, beta)
            theta, log = ppo_update(env, theta_old, batch, n_epochs, n_minibatches, None, lr, rng)
        elif mode == 'both':
            batch = collect_batch(env, theta_old, rng, n_seqs, reward_fn, theta_ref, beta)
            theta, log = ppo_update(env, theta_old, batch, n_epochs, n_minibatches, eps, lr, rng)
        elif mode == 'none':
            batch = collect_batch(env, theta_old, rng, n_seqs, reward_fn, theta_ref, 0)
            theta, log = ppo_update(env, theta_old, batch, n_epochs, n_minibatches, None, lr, rng)
        
        report['clip_frac'].append(sum(log['clip_frac'])/len(log['clip_frac']))
        report['kl_ref'].append(float(exact_sequence_kl(env, theta, theta_ref)))
        report['kl_step'].append(float(exact_sequence_kl(env, theta, theta_old)))
        report['J'].append(float(exact_J(env, theta, reward_fn)))
    
    report['theta'] = theta
    return report








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


def _gold_reward(env: TokenEnv):
    """Day 1's structured sequence reward, unchanged: token 0 pays, length costs,
    clean termination earns a bonus."""
    def r(seq: tuple[int, ...]) -> float:
        n_good = sum(1 for a in seq if a == 0)
        ends_clean = 1.0 if seq and seq[-1] == env.eos else 0.0
        return 2.0 * n_good - 0.4 * len(seq) + 0.75 * ends_clean
    return r


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ca = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(ca, -1.0, 1.0))))


class Ctx:
    """Lazily-built fixtures, each from its OWN seeded generator so a stage's data does
    not depend on which other stages ran first."""

    def __init__(self, fast: bool = False):
        self.fast = fast
        # V=5, T=3, B=85 -- deliberately three different numbers. See the module docstring.
        self.env = TokenEnv(vocab=5, t_max=3)
        self.eps = 0.2
        self.lr_sweep = 0.2
        self.beta = 1.0
        # --fast shrinks DATA and loosens TOLERANCES. It must never cut gradient STEPS:
        # on D1 that turned a true assertion false for reasons unrelated to the
        # assertion, which is the worst kind of test failure.
        self.n_iters = 25 if not fast else 14
        self.n_seqs = 256 if not fast else 96
        self.frontier_iters = 45 if not fast else 30
        self.betas = (0.7, 1.0, 1.5, 2.5)
        self.t_list = (1, 2, 4, 8, 12) if not fast else (1, 2, 4, 8)

    @cached_property
    def theta_old(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(0), scale=1.0)

    @cached_property
    def theta_ref(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(1), scale=0.8)

    @cached_property
    def reward(self):
        return _gold_reward(self.env)

    @cached_property
    def eb(self) -> Batch:
        """Enumerated batch, UNCENTERED advantages -- Psi_t = R(y), so the estimator on
        it is exactly grad J and can be compared to a finite difference."""
        return exact_batch(self.env, self.theta_old, self.reward, center=False)

    @cached_property
    def ebc(self) -> Batch:
        """Enumerated batch, CENTERED advantages (a constant baseline)."""
        return exact_batch(self.env, self.theta_old, self.reward, center=True)

    @cached_property
    def g_true(self) -> np.ndarray:
        return finite_difference_gradient(self.env, self.theta_old, self.reward)

    @cached_property
    def theta_far(self) -> np.ndarray:
        """A theta far enough from theta_old that the clip actually binds."""
        return self.theta_old + 0.35 * np.random.default_rng(5).standard_normal(
            self.theta_old.shape)

    def spread_fixture(self, t_max: int):
        """A vocab=3 chain with eos suppressed, so every sequence runs to t_max and the
        per-token perturbation is held FIXED as T grows. Both details matter: with eos
        live the lengths shrink and the T axis is confounded, and with vocab=2 the one
        real token already has probability ~1 so perturbing its logit changes nothing
        (that fixture reads sigma = 0.000 at every T -- a silent null experiment)."""
        e = TokenEnv(vocab=3, t_max=t_max)
        th_o = np.zeros((e.n_states, e.vocab))
        th_o[:, e.eos] = -14.0
        pert = np.random.default_rng(7).standard_normal((e.n_states, e.vocab)) * 0.2
        pert[:, e.eos] = 0.0
        return e, th_o, th_o + pert


def _fail_fast(ctx: Ctx) -> None:
    """Touch the cheap functions a heavy stage needs on a 1-element input, so a missing
    TODO surfaces before we pay for a finite-difference gradient or a 25-iteration run."""
    tiny = TokenEnv(vocab=3, t_max=1)
    th = np.zeros((tiny.n_states, tiny.vocab))
    b = exact_batch(tiny, th, lambda s: 1.0)
    lr = token_logratios(tiny, th, b)
    is_surrogate(lr, b)
    clipped_surrogate(lr, b, 0.2)
    surrogate_gradient(tiny, th, b, eps=0.2)


# ---- individual stages ----------------------------------------------------


def stage_logratios(ctx: Ctx) -> None:
    """token_logratios -- exact zeros at theta_old, shapes, log-space stability"""
    env, b = ctx.env, ctx.eb

    lr = token_logratios(env, ctx.theta_old, b)
    _check("shape is (B, T) and not (B, V)", lr.shape == (b.B, env.t_max),
           f"got {lr.shape}, want {(b.B, env.t_max)} with V={env.vocab}")
    _check("dtype is floating", np.issubdtype(lr.dtype, np.floating), str(lr.dtype))
    _check("theta == theta_old gives EXACTLY the zero matrix", np.all(lr == 0.0),
           f"max|.| = {np.abs(lr).max():.3e}")

    lr2 = token_logratios(env, ctx.theta_far, b)
    _check("padded positions stay exactly 0", np.all(lr2[~b.mask] == 0.0))
    _check("real positions are generally nonzero", np.any(lr2[b.mask] != 0.0))

    # A hand-computable single token.
    lp_new = log_softmax_np(ctx.theta_far)
    s0, a0 = b.states[0, 0], b.actions[0, 0]
    want = lp_new[s0, a0] - b.logp_old[0, 0]
    _check("matches an explicit log pi_new - log pi_old at one token",
           np.isclose(lr2[0, 0], want, atol=1e-15), f"{lr2[0, 0]:.12f} vs {want:.12f}")

    # Log space: a token that is astronomically unlikely under one policy must give a
    # finite log-ratio, where a probability quotient would give 0/0 = nan.
    hard = ctx.theta_old.copy()
    hard[:, 0] -= 700.0
    lr3 = token_logratios(env, hard, b)
    _check("survives a 700-nat logit gap (log space, not a quotient)",
           np.all(np.isfinite(lr3)), f"{np.sum(~np.isfinite(lr3))} non-finite entries")


def stage_surrogate(ctx: Ctx) -> None:
    """is_surrogate -- the sum/weighted-sum contract, and the length-bias residual"""
    env = ctx.env
    b, bc = ctx.eb, ctx.ebc

    lr0 = token_logratios(env, ctx.theta_old, b)
    v = is_surrogate(lr0, b)
    _check("returns a python float", type(v) is float, str(type(v)))

    lens = np.array([len(s) for s in b.seqs], dtype=float)
    want = float(b.weights @ (lens * b.advantages[:, 0]))
    _check("at theta_old equals sum_b w_b len(y_b) A_b (rho == 1)",
           abs(v - want) < 1e-14, f"{v:.12f} vs {want:.12f}")

    vc = is_surrogate(token_logratios(env, ctx.theta_old, bc), bc)
    mean_adv = float(bc.weights @ bc.advantages[:, 0])
    _check("centring makes the ADVANTAGE mean exactly 0", abs(mean_adv) < 1e-14,
           f"{mean_adv:.3e}")
    _check("...but the surrogate VALUE is still nonzero", abs(vc) > 1e-6,
           f"L_IS(theta_old) = {vc:.6f}")
    print(f"  ... length-advantage covariance leaks {vc:+.6f} into a 'zero' loss: the "
          f"surrogate value\n      is uninformative BY CONSTRUCTION, which is why no "
          f"PPO run is debugged from its loss curve.")

    # Equal-length limit, which proves the leak above is a LENGTH effect and not a bug
    # in your reduction: in a one-step bandit every sequence has length exactly 1, so
    # sum_b w_b len(y_b) A_b collapses to sum_b w_b A_b = 0 for centred advantages.
    e2 = TokenEnv(vocab=6, t_max=1)
    th2 = e2.random_theta(np.random.default_rng(4), scale=1.0)
    b2 = exact_batch(e2, th2, _gold_reward(e2), center=True)
    v2 = is_surrogate(token_logratios(e2, th2, b2), b2)
    _check("one-step bandit (all lengths equal) => centred advantages give EXACTLY 0",
           abs(v2) < 1e-14, f"{v2:.3e}")

    # Linearity in the advantages, which pins down the reduction over t.
    b3 = Batch(b.states, b.actions, b.mask, 3.0 * b.advantages, b.logp_old,
               b.weights, b.returns, b.seqs)
    _check("linear in the advantages (scale by 3)",
           abs(is_surrogate(lr0, b3) - 3.0 * v) < 1e-13)


def stage_grad_regression(ctx: Ctx) -> None:
    """surrogate_gradient at rho=1 IS Week 3's policy gradient, and both ARE grad J"""
    _fail_fast(ctx)
    env, b = ctx.env, ctx.eb

    g = surrogate_gradient(env, ctx.theta_old, b, eps=None)
    _check("shape == theta.shape", g.shape == ctx.theta_old.shape, str(g.shape))
    _check("dtype is floating (an int accumulator silently returns all zeros)",
           np.issubdtype(g.dtype, np.floating), str(g.dtype))
    _check("gradient is not identically zero", np.abs(g).max() > 1e-6)

    g_sf = score_function_gradient(env, ctx.theta_old, b)
    d = float(np.abs(g - g_sf).max())
    _check("== W3's score-function gradient ELEMENTWISE at theta_old", d < 1e-12,
           f"max|delta| = {d:.3e}")

    g_fd = ctx.g_true
    d2 = float(np.abs(g - g_fd).max())
    _check("== central finite difference of the ENUMERATED J", d2 < 1e-7,
           f"max|delta| = {d2:.3e}, scale {np.abs(g_fd).max():.4f}")

    rs = float(np.abs(g.sum(axis=1)).max())
    _check("every row sums to zero (softmax over-parameterisation)", rs < 1e-12,
           f"max|row sum| = {rs:.3e}")

    # A baseline is free: centring the advantages cannot move the gradient at all.
    gc = surrogate_gradient(env, ctx.theta_old, ctx.ebc, eps=None)
    d3 = float(np.abs(gc - g).max())
    _check("a constant baseline changes the gradient by 0 (EGLP, exactly)", d3 < 1e-12,
           f"max|delta| = {d3:.3e}")

    # Teeth: the clipped gradient at theta_old must EQUAL the unclipped one, because
    # rho == 1 is inside every band. If eps is being applied to the wrong side this
    # fails immediately.
    gclip = surrogate_gradient(env, ctx.theta_old, b, eps=ctx.eps)
    _check("at theta_old the CLIPPED gradient is identical (rho=1 is inside the band)",
           float(np.abs(gclip - g).max()) < 1e-14)

    # And teeth the other way: an eps so small the band is a point must NOT reproduce
    # it away from theta_old.
    gf = surrogate_gradient(env, ctx.theta_far, b, eps=None)
    gf0 = surrogate_gradient(env, ctx.theta_far, b, eps=0.0)
    _check("away from theta_old, eps=0 does change the gradient",
           float(np.abs(gf - gf0).max()) > 1e-6,
           f"max|delta| = {float(np.abs(gf - gf0).max()):.4f}")


def stage_clip_table(ctx: Ctx) -> None:
    """the 2x2 table by hand: value AND gradient, with EXACT zeros in two cells"""
    eps = 0.2

    def one_token(adv: float, rho: float, vocab: int = 3):
        """A single-token batch whose ratio is exactly `rho` by construction."""
        env = TokenEnv(vocab=vocab, t_max=1)   # one state, one decision: pure algebra
        sid = env.state_id(())
        theta_old = np.zeros((env.n_states, env.vocab))
        lp_old = log_softmax_np(theta_old)[sid, 0]
        b = Batch(np.array([[sid]]), np.array([[0]]), np.array([[True]]),
                  np.array([[adv]]), np.array([[lp_old]]), np.array([1.0]),
                  np.array([adv]), [(0,)])
        # solve for the theta whose log pi(0|s) is log pi_old + log rho
        theta = theta_old.copy()
        target = lp_old + np.log(rho)
        lo, hi = -50.0, 50.0
        for _ in range(200):                   # bisection: exact to machine precision
            mid = 0.5 * (lo + hi)
            theta[sid, 0] = mid
            if log_softmax_np(theta)[sid, 0] < target:
                lo = mid
            else:
                hi = mid
        theta[sid, 0] = 0.5 * (lo + hi)
        lr = token_logratios(env, theta, b)
        val, diag = clipped_surrogate(lr, b, eps)
        grad = surrogate_gradient(env, theta, b, eps=eps)
        return float(np.exp(lr[0, 0])), val, float(np.abs(grad).max()), diag

    # `side` pins which of frac_high / frac_low the killed token must be counted in.
    # Without it, SWAPPING the two labels passes every other check in this file --
    # `frac_high + frac_low == clip_frac` is symmetric in them, and so is clip_frac.
    cases = [
        ("A>0, rho>1+eps  CLIPPED, grad 0", +2.0, 1.6, (1 + eps) * 2.0, True, "high"),
        ("A>0, rho<1-eps  free,    grad!=0", +2.0, 0.5, 0.5 * 2.0, False, None),
        ("A<0, rho<1-eps  CLIPPED, grad 0", -2.0, 0.5, (1 - eps) * -2.0, True, "low"),
        ("A<0, rho>1+eps  free,    grad!=0", -2.0, 1.6, 1.6 * -2.0, False, None),
    ]
    for name, adv, rho, want_val, want_killed, side in cases:
        got_rho, val, gmax, diag = one_token(adv, rho)
        _check(f"{name}: rho built exactly", abs(got_rho - rho) < 1e-12,
               f"{got_rho:.12f}")
        _check(f"{name}: value", abs(val - want_val) < 1e-10,
               f"got {val:.10f} want {want_val:.10f}")
        if want_killed:
            _check(f"{name}: gradient is EXACTLY 0.0", gmax == 0.0, f"{gmax:.3e}")
            _check(f"{name}: clip_frac == 1", diag["clip_frac"] == 1.0)
            other = "low" if side == "high" else "high"
            _check(f"{name}: counted in frac_{side}, and frac_{other} == 0",
                   diag[f"frac_{side}"] == 1.0 and diag[f"frac_{other}"] == 0.0,
                   f"high={diag['frac_high']}, low={diag['frac_low']}")
            _check(f"{name}: oob_frac == 1 (it IS out of band)",
                   diag["oob_frac"] == 1.0, f"{diag['oob_frac']}")
        else:
            _check(f"{name}: gradient is nonzero", gmax > 1e-6, f"{gmax:.6f}")
            _check(f"{name}: clip_frac == 0", diag["clip_frac"] == 0.0)
            _check(f"{name}: out of band but NOT clipped (clip_frac < oob_frac)",
                   diag["oob_frac"] == 1.0 and diag["clip_frac"] == 0.0,
                   f"oob={diag['oob_frac']}, clip={diag['clip_frac']}")

    # The wrong implementation this table exists to catch: clipping BOTH sides kills
    # all four gradients, so a test that only looked at values would pass it.
    n_killed = sum(1 for _, adv, rho, *_ in cases if one_token(adv, rho)[2] == 0.0)
    _check("exactly 2 of the 4 cells have zero gradient (not 4: both-sides clipping)",
           n_killed == 2, f"{n_killed} cells killed")

    # THE UNBOUNDED CELL. In the A<0, rho>1+eps cell the objective is rho*A with
    # nothing capping rho, so it runs to -infinity LINEARLY in rho. The other three
    # cells are bounded: two are pinned at (1 +/- eps)A and the third has rho < 1-eps.
    # vocab=8 so that pi_old = 1/8 and a ratio of 6 is actually reachable.
    v2 = one_token(-2.0, 2.0, vocab=8)[1]
    v6 = one_token(-2.0, 6.0, vocab=8)[1]
    _check("A<0, rho>1+eps: the objective is unbounded, exactly linear in rho",
           abs(v6 / v2 - 3.0) < 1e-9, f"L(rho=2) = {v2:.4f} -> L(rho=6) = {v6:.4f}")
    for name, adv, r1, r2 in [("A>0 above band", +2.0, 2.0, 6.0),
                              ("A<0 below band", -2.0, 0.5, 0.1),
                              ("A>0 below band", +2.0, 0.5, 0.1)]:
        a = one_token(adv, r1, vocab=8)[1]
        bb = one_token(adv, r2, vocab=8)[1]
        _check(f"{name}: bounded -- |L| does not grow with the ratio",
               abs(bb) <= abs(a) + 1e-9, f"{a:.4f} -> {bb:.4f}")
    print("  ... only ONE of the four cells is unbounded, and it is the cell dual-clip "
          "PPO patches\n      with an outer max(., c*A) for c ~ 3. Note also that in a "
          "TABULAR softmax the\n      theta-gradient stays bounded even there, because "
          "the softmax saturates: the\n      unboundedness lives in the objective and "
          "in rho, not in this parameterisation.")


def stage_clip_bounds(ctx: Ctx) -> None:
    """L_CLIP <= L_IS pointwise; the eps limits; clip_frac <= oob_frac"""
    env, b = ctx.env, ctx.ebc
    rng = np.random.default_rng(21)

    # Randomised sweep over (theta, eps), including A = 0 and rho exactly at the edge.
    worst = 0.0
    for _ in range(40 if not ctx.fast else 12):
        th = ctx.theta_old + rng.uniform(0.05, 0.8) * rng.standard_normal(ctx.theta_old.shape)
        eps = float(rng.choice([0.0, 0.01, 0.1, 0.2, 0.5, 2.0]))
        lr = token_logratios(env, th, b)
        vc, diag = clipped_surrogate(lr, b, eps)
        vi = is_surrogate(lr, b)
        worst = max(worst, vc - vi)
        assert diag["clip_frac"] <= diag["oob_frac"] + 1e-15
    _check("L_CLIP <= L_IS over a randomised (theta, eps) sweep", worst < 1e-12,
           f"worst violation {worst:.3e}")
    _check("clip_frac <= oob_frac throughout the sweep", True)

    # A == 0: clipping must be a no-op, whatever rho is.
    b0 = Batch(b.states, b.actions, b.mask, np.zeros_like(b.advantages), b.logp_old,
               b.weights, b.returns, b.seqs)
    lrf = token_logratios(env, ctx.theta_far, b0)
    v0, d0 = clipped_surrogate(lrf, b0, 0.2)
    _check("A == 0 => L_CLIP == L_IS == 0", abs(v0) < 1e-15 and
           abs(is_surrogate(lrf, b0)) < 1e-15, f"{v0:.3e}")
    _check("A == 0 => nothing is counted as clipped", d0["clip_frac"] == 0.0)

    # eps -> infinity reproduces the IS surrogate EXACTLY, not approximately.
    lrf = token_logratios(env, ctx.theta_far, b)
    vi = is_surrogate(lrf, b)
    vbig, dbig = clipped_surrogate(lrf, b, 1e9)
    _check("eps -> inf reproduces is_surrogate exactly", abs(vbig - vi) < 1e-15,
           f"delta {vbig - vi:.3e}")
    _check("eps -> inf clips nothing", dbig["clip_frac"] == 0.0)

    # Monotone in eps: a wider band can only let more of the surrogate through.
    vals = [clipped_surrogate(lrf, b, e)[0] for e in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)]
    _check("L_CLIP is nondecreasing in eps",
           all(v2 >= v1 - 1e-14 for v1, v2 in zip(vals, vals[1:])),
           " -> ".join(f"{v:.5f}" for v in vals))
    fracs = [clipped_surrogate(lrf, b, e)[1]["clip_frac"] for e in (0.0, 0.05, 0.2, 1.0)]
    _check("clip_frac is nonincreasing in eps",
           all(f2 <= f1 + 1e-14 for f1, f2 in zip(fracs, fracs[1:])),
           " -> ".join(f"{f:.3f}" for f in fracs))

    # frac_high + frac_low must account for every killed token.
    v, d = clipped_surrogate(lrf, b, 0.2)
    _check("frac_high + frac_low == clip_frac",
           abs(d["frac_high"] + d["frac_low"] - d["clip_frac"]) < 1e-15,
           f"{d['frac_high']:.4f} + {d['frac_low']:.4f} vs {d['clip_frac']:.4f}")
    print(f"  ... at |dtheta|~0.35: clip_frac={d['clip_frac']:.3f} "
          f"oob_frac={d['oob_frac']:.3f} (high {d['frac_high']:.3f} / "
          f"low {d['frac_low']:.3f})")


def stage_ratio_blowup(ctx: Ctx) -> None:
    """the sequence ratio prod_t rho_t: mean pinned at 1, median sinking, ESS collapsing"""
    print("  per-token perturbation held FIXED; only the length T grows.")
    print(f"  {'T':>4} {'mean':>16} {'median':>8} {'p99':>8} {'p99/med':>8} "
          f"{'ESS/n':>7} {'chi2':>8} {'sigma':>7} {'sig1':>7} {'in band':>8}")
    rows = {}
    for tm in ctx.t_list:
        e, th_o, th_n = ctx.spread_fixture(tm)
        s = sequence_ratio_stats(e, th_n, th_o)
        tk = token_ratio_stats(e, th_n, th_o)
        rows[tm] = (s, tk)
        print(f"  {tm:>4} {s['mean']:>16.13f} {s['median']:>8.4f} {s['p99']:>8.3f} "
              f"{s['p99'] / s['median']:>8.2f} {s['ess_frac']:>7.4f} {s['chi2']:>8.4f} "
              f"{s['log_std']:>7.4f} {tk['log_std']:>7.4f} {tk['frac_in_band']:>8.3f}")

    for tm, (s, tk) in rows.items():
        _check(f"T={tm}: E_old[rho] == 1 EXACTLY (it is a likelihood ratio)",
               abs(s["mean"] - 1.0) < 1e-12, f"{s['mean']:.15f}")
        _check(f"T={tm}: ESS/n == 1/(1+chi^2)",
               abs(s["ess_frac"] - 1.0 / (1.0 + s["chi2"])) < 1e-12)
        # eos is suppressed to -14, not to -inf, so a ~4e-7 slice of the mass still
        # terminates early. Close enough that T is the length; not close enough for 1e-9.
        _check(f"T={tm}: mean_len == T", abs(s["mean_len"] - tm) < 1e-3,
               f"{s['mean_len']:.6f}")

    ts = sorted(rows)
    spreads = [rows[t][0]["p99"] / rows[t][0]["median"] for t in ts]
    esss = [rows[t][0]["ess_frac"] for t in ts]
    _check("p99/median grows strictly with T",
           all(b > a for a, b in zip(spreads, spreads[1:])),
           " -> ".join(f"{x:.2f}" for x in spreads))
    _check("ESS/n falls strictly with T", all(b < a for a, b in zip(esss, esss[1:])),
           " -> ".join(f"{x:.4f}" for x in esss))

    # THE SCALING LAW: sigma(T) = sigma_1 * sqrt(T). This is the log-normal signature,
    # and it is what licenses extrapolating to T=500 from a T=12 measurement.
    worst = 0.0
    for tm in ts:
        s, tk = rows[tm]
        pred = tk["log_std"] * np.sqrt(tm)
        worst = max(worst, abs(s["log_std"] - pred) / max(pred, 1e-12))
    _check("sigma(T) == sigma_1 * sqrt(T) (the sqrt-T law)", worst < 0.05,
           f"worst relative error {worst:.4f}")

    # And the log-normal's own prediction for the ESS.
    worst2 = max(abs(rows[t][0]["ess_frac"] - np.exp(-rows[t][0]["log_std"] ** 2))
                 / rows[t][0]["ess_frac"] for t in ts)
    _check("ESS/n == exp(-sigma^2), the log-normal prediction", worst2 < 0.05,
           f"worst relative error {worst2:.4f}")

    big = max(ts)
    sig1 = rows[big][1]["log_std"]
    print(f"\n  extrapolating the VERIFIED law (sigma_1 = {sig1:.4f}, measured with "
          f"{rows[big][1]['frac_in_band'] * 100:.1f}% of\n  individual tokens inside "
          f"PPO's |rho-1| <= 0.2 band):")
    for T in (100, 500, 2000):
        sd = sig1 * np.sqrt(T)
        print(f"    T={T:<5} sigma={sd:6.3f}  p99/median={np.exp(2.326 * sd):9.3e}  "
              f"ESS/n={np.exp(-sd ** 2):9.3e}")
    # Stated as an effective sample COUNT rather than a fraction, so the threshold does
    # not move when --fast truncates the T ladder at a slightly smaller sigma_1.
    ess500 = float(np.exp(-(sig1 ** 2) * 500))
    _check("at T=500 a batch of 1024 sequences has an effective size below 5",
           1024 * ess500 < 5.0,
           f"ESS/n = {ess500:.3e} => {1024 * ess500:.2f} effective sequences out of 1024")
    print("  => the argument for a PER-TOKEN ratio is not 'variance is large'. The mean "
          "is exactly\n     1 while the median sinks: a vanishing fraction of samples "
          "carries all the weight.")


def stage_epoch_sweep(ctx: Ctx) -> None:
    """ppo_update reuse: how far the update drifts from the true gradient as K grows"""
    _fail_fast(ctx)
    env, b = ctx.env, ctx.ebc
    g_true = ctx.g_true
    lr = ctx.lr_sweep

    # NEGATIVE CONTROL: one epoch, one minibatch. Nothing is off-policy yet, so
    # clipping cannot possibly do anything and the direction must be exact.
    t_clip, log = ppo_update(env, ctx.theta_old, b, 1, 1, eps=ctx.eps, lr=lr,
                             rng=np.random.default_rng(0))
    t_none, _ = ppo_update(env, ctx.theta_old, b, 1, 1, eps=None, lr=lr,
                           rng=np.random.default_rng(0))
    _check("K=1, 1 minibatch: clip and no-clip agree to machine precision",
           float(np.abs(t_clip - t_none).max()) < 1e-15,
           f"max|delta| = {float(np.abs(t_clip - t_none).max()):.3e}")
    _check("K=1: update direction is EXACTLY the true grad J (0 degrees)",
           _angle_deg(t_clip - ctx.theta_old, g_true) < 1e-4,
           f"{_angle_deg(t_clip - ctx.theta_old, g_true):.2e} deg")
    _check("log entries are python floats, not numpy scalars or tensors",
           all(type(x) is float for k in ("surrogate", "clip_frac", "oob_frac")
               for x in log[k]),
           str({k: type(v[0]).__name__ for k, v in log.items()}))
    _check("one log entry per minibatch step", len(log["surrogate"]) == 1)

    t2, log2 = ppo_update(env, ctx.theta_old, b, 3, 4, eps=ctx.eps, lr=lr,
                          rng=np.random.default_rng(0))
    _check("n_epochs * n_minibatches log entries", len(log2["surrogate"]) == 12,
           f"got {len(log2['surrogate'])}")
    _check("ppo_update does not mutate theta_old",
           np.array_equal(ctx.theta_old,
                          ctx.env.random_theta(np.random.default_rng(0), scale=1.0)))

    print(f"\n  K epochs, 1 minibatch, lr={lr}: angle to the TRUE grad J, and the KL "
          f"actually travelled")
    print(f"  {'K':>3} {'angle_clip':>11} {'angle_noclip':>13} {'clip_frac':>10} "
          f"{'KL_clip':>9} {'KL_noclip':>10} {'J_clip':>8} {'J_noclip':>9}")
    ks = (1, 2, 4, 8, 16, 32, 64) if not ctx.fast else (1, 2, 8, 32)
    rec = {}
    for K in ks:
        th_c, lg_c = ppo_update(env, ctx.theta_old, b, K, 1, eps=ctx.eps, lr=lr,
                                rng=np.random.default_rng(0))
        th_n, _ = ppo_update(env, ctx.theta_old, b, K, 1, eps=None, lr=lr,
                             rng=np.random.default_rng(0))
        a_c = _angle_deg(th_c - ctx.theta_old, g_true)
        a_n = _angle_deg(th_n - ctx.theta_old, g_true)
        kl_c = exact_sequence_kl(env, ctx.theta_old, th_c)
        kl_n = exact_sequence_kl(env, ctx.theta_old, th_n)
        rec[K] = (a_c, a_n, kl_c, kl_n, float(np.mean(lg_c["clip_frac"])))
        print(f"  {K:>3} {a_c:>11.3f} {a_n:>13.3f} {np.mean(lg_c['clip_frac']):>10.3f} "
              f"{kl_c:>9.5f} {kl_n:>10.5f} {exact_J(env, th_c, ctx.reward):>8.4f} "
              f"{exact_J(env, th_n, ctx.reward):>9.4f}")

    ks_sorted = sorted(rec)
    angles = [rec[k][0] for k in ks_sorted]
    _check("the update direction degrades monotonically with K",
           all(b >= a - 1e-9 for a, b in zip(angles, angles[1:])),
           " -> ".join(f"{a:.1f}" for a in angles))
    _check("at K=1 the angle is 0 and at the largest K it is substantial",
           angles[0] < 1e-4 and angles[-1] > 20.0,
           f"{angles[0]:.2e} -> {angles[-1]:.1f} deg")

    kbig = ks_sorted[-1]
    a_c, a_n, kl_c, kl_n, cf = rec[kbig]
    _check(f"K={kbig}: clipping cuts the KL travelled by >5x", kl_n > 5.0 * kl_c,
           f"{kl_n:.4f} vs {kl_c:.4f} ({kl_n / kl_c:.1f}x)")
    _check(f"K={kbig}: clipping makes the ANGLE strictly WORSE, not better",
           a_c > a_n + 5.0, f"clip {a_c:.1f} deg vs no-clip {a_n:.1f} deg")
    print(f"  => clipping does NOT protect the update DIRECTION -- it makes it "
          f"{a_c - a_n:.1f} deg worse.\n     What it protects is the SIZE of the step: "
          f"{kl_n / kl_c:.1f}x less KL travelled. Those {a_c:.0f} degrees\n     ARE the "
          f"bias from Part 4 of the derivation, measured. And note the no-clip run "
          f"wins\n     on J for the same batch, which is exactly what a Goodhart "
          f"failure looks like from inside.")


def stage_drift(ctx: Ctx) -> None:
    """THE HEADLINE: clipping bounds the step, never the walk"""
    _fail_fast(ctx)
    env, R = ctx.env, ctx.reward
    out = {}
    for mode in ("clip", "penalty", "both", "none"):
        h = drift_experiment(env, ctx.theta_ref.copy(), ctx.theta_ref, R, mode=mode,
                             n_iters=ctx.n_iters, beta=ctx.beta, eps=ctx.eps,
                             n_epochs=4, n_minibatches=4, lr=0.1, n_seqs=ctx.n_seqs,
                             rng=np.random.default_rng(3))
        out[mode] = h
        print(f"  {mode:<8} KL(theta||ref): {h['kl_ref'][0]:6.4f} -> "
              f"{h['kl_ref'][-1]:7.4f}   J: {h['J'][0]:6.4f} -> {h['J'][-1]:6.4f}   "
              f"clip_frac mean {np.mean(h['clip_frac']):.4f} max "
              f"{np.max(h['clip_frac']):.4f}")
        print(f"           trajectory: {np.round(h['kl_ref'][::max(1, ctx.n_iters // 5)], 3)}")

    for mode, h in out.items():
        _check(f"{mode}: all four histories have n_iters entries",
               all(len(h[k]) == ctx.n_iters for k in ("kl_ref", "kl_step", "clip_frac", "J")))
        _check(f"{mode}: every step's KL is far smaller than the walk",
               np.mean(h["kl_step"]) < 0.2 * h["kl_ref"][-1],
               f"mean step {np.mean(h['kl_step']):.4f} vs final walk {h['kl_ref'][-1]:.4f}")

    c, p = out["clip"], out["penalty"]
    _check("clip-only drifts strictly FURTHER from pi_ref than the beta-penalised run",
           c["kl_ref"][-1] > 2.0 * p["kl_ref"][-1],
           f"{c['kl_ref'][-1]:.4f} vs {p['kl_ref'][-1]:.4f} "
           f"({c['kl_ref'][-1] / p['kl_ref'][-1]:.1f}x)")
    _check("...while the clip-only run's clip fraction stays LOW the whole time",
           np.max(c["clip_frac"]) < 0.25,
           f"mean {np.mean(c['clip_frac']):.4f}, max {np.max(c['clip_frac']):.4f}")

    tail = max(4, ctx.n_iters // 3)
    pen_spread = float(np.max(p["kl_ref"][-tail:]) - np.min(p["kl_ref"][-tail:]))
    clip_rise = float(c["kl_ref"][-1] - c["kl_ref"][-tail])
    _check("the beta-penalised run PLATEAUS", pen_spread < 0.2,
           f"last {tail} iterations span {pen_spread:.4f} nats")
    _check("the clip-only run is still CLIMBING at the end", clip_rise > 0.3,
           f"rose {clip_rise:.4f} nats over the last {tail} iterations")

    _check("'none' drifts at least as far as clip-only (the clip does something)",
           out["none"]["kl_ref"][-1] > c["kl_ref"][-1])
    _check("'both' is the tightest of the four",
           out["both"]["kl_ref"][-1] <= min(out[m]["kl_ref"][-1]
                                            for m in ("clip", "penalty", "none")) + 1e-9,
           f"both {out['both']['kl_ref'][-1]:.4f}")

    print(f"  => clip fraction {np.mean(c['clip_frac']) * 100:.1f}% for "
          f"{ctx.n_iters} iterations, every single step inside the\n     trust region, "
          f"and the policy walked {c['kl_ref'][-1]:.2f} nats from where it started. The "
          f"beta run\n     stopped at {p['kl_ref'][-1]:.2f}. This is why InstructGPT "
          f"ships the clip AND the penalty:\n     they bound different objects and "
          f"neither implies the other.")


def stage_frontier(ctx: Ctx) -> None:
    """where the penalty plateaus: dE[r]/dKL == beta, and the frontier is CONCAVE"""
    _fail_fast(ctx)
    env, R = ctx.env, ctx.reward
    pts = {}
    print(f"  {'beta':>5} {'KL(theta||ref)':>15} {'J':>9} {'tail spread':>12}")
    for beta in ctx.betas:
        h = drift_experiment(env, ctx.theta_ref.copy(), ctx.theta_ref, R, mode="penalty",
                             n_iters=ctx.frontier_iters, beta=beta, eps=ctx.eps,
                             n_epochs=4, n_minibatches=4, lr=0.1, n_seqs=ctx.n_seqs,
                             rng=np.random.default_rng(3))
        tail = h["kl_ref"][-8:]
        pts[beta] = (h["kl_ref"][-1], h["J"][-1])
        print(f"  {beta:>5.2f} {h['kl_ref'][-1]:>15.4f} {h['J'][-1]:>9.4f} "
              f"{max(tail) - min(tail):>12.4f}")
        _check(f"beta={beta}: the run has plateaued", max(tail) - min(tail) < 0.15,
               f"{max(tail) - min(tail):.4f}")

    bs = sorted(pts)
    kls = [pts[b][0] for b in bs]
    js = [pts[b][1] for b in bs]
    _check("a bigger beta buys a smaller KL", all(a > b for a, b in zip(kls, kls[1:])),
           " -> ".join(f"{k:.3f}" for k in kls))
    _check("a bigger beta also buys less reward",
           all(a > b for a, b in zip(js, js[1:])), " -> ".join(f"{j:.3f}" for j in js))

    print("\n  secant slope dJ/dKL between adjacent plateaus vs the two betas:")
    slopes = []                                  # walked from the LARGEST KL downwards
    for b1, b2 in zip(bs[:-1], bs[1:]):          # b1 < b2, so KL(b1) > KL(b2)
        (k1, j1), (k2, j2) = pts[b1], pts[b2]
        slope = (j1 - j2) / (k1 - k2)
        slopes.append((0.5 * (k1 + k2), slope))
        print(f"    beta {b2:.2f} -> {b1:.2f}  (KL {k2:.3f} -> {k1:.3f}): "
              f"dJ/dKL = {slope:.4f}")
        _check(f"slope between beta={b2} and beta={b1} lies BETWEEN them",
               b1 - 1e-9 <= slope <= b2 + 1e-9,
               f"{slope:.4f} vs the bracket [{b1}, {b2}]")
    slopes.sort()                                # ascending KL
    seq = [s for _, s in slopes]
    _check("dJ/dKL DECREASES as KL grows: the frontier is concave",
           all(a > b for a, b in zip(seq, seq[1:])),
           "  ".join(f"KL~{k:.2f}: {s:.3f}" for k, s in slopes))
    print("  => dE[r]/dKL = +beta at the optimum, measured, with the sign that cold "
          "recall got\n     wrong yesterday: beta is a PRICE. And the price falls as "
          "you buy more nats, which is\n     the concavity that makes a single beta "
          "pick out an operating point at all.")


STAGES = [
    ("logratios", stage_logratios),
    ("surrogate", stage_surrogate),
    ("grad_regression", stage_grad_regression),
    ("clip_table", stage_clip_table),
    ("clip_bounds", stage_clip_bounds),
    ("ratio_blowup", stage_ratio_blowup),
    ("epoch_sweep", stage_epoch_sweep),
    ("drift", stage_drift),
    ("frontier", stage_frontier),
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
