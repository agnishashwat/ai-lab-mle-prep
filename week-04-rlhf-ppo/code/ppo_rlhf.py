"""W4 D5 -- the toy PPO-RLHF loop, and adaptive beta as DUAL GRADIENT ASCENT.

The week's capstone. Days 1-4 are imported, not reimplemented: `toy_rlhf` gives the
enumerable token environment and the Bradley-Terry machinery, `ppo_loss` gives the
clipped surrogate and its analytic gradient, `token_ppo` gives GAE / the value head /
per-token KL shaping / `instructgpt_loss`, and `failure_modes` gives the learned
sequence RM, the instruments, and `diagnose`. Almost every stage here is INTEGRATION
plus ABLATION rather than new math.

The one genuinely new idea:

    beta was never a hyperparameter. It is a LAGRANGE MULTIPLIER, and InstructGPT's
    KL controller is PROPORTIONAL DUAL ASCENT on it.

        primal   max_theta  E[r] - beta (KL - delta)      <- the PPO epochs
        dual     beta <- beta + eta (KL - delta)          <- the controller
        shipped  beta <- beta (1 + K_beta (KL/delta - 1))

    and by the envelope theorem  dg/dbeta = delta - KL(pi*_beta || pi_ref), which is the
    SAME computation as W3 D5's  dJ*/dbeta = -KL  that you verified to rtol 1e-4.

The day's best check (`stage_dual`) is that the controller's FIXED POINT equals the
closed-form frontier's  dE[r]/dKL  at the same KL: one number reached two structurally
independent ways, one iterative and sampled, one closed-form and exact.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-04-rlhf-ppo/code/ppo_rlhf.py
  one stage           ...  ppo_rlhf.py dual
  a few stages        ...  ppo_rlhf.py controller dual gain
  small/fast fixtures ...  ppo_rlhf.py --fast
  list the stages     ...  ppo_rlhf.py --list
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from toy_rlhf import (
    TokenEnv,
    enumerate_sequences,
    exact_sequence_kl,
    log_softmax_np,
    sample_sequences,
    sequence_logprob,
    softmax_np,
    state_occupancy,
    tilted_sequence_dist,
)
from ppo_loss import clipped_surrogate, score_function_gradient, surrogate_gradient, token_logratios
from token_ppo import (
    TokenBatch,
    gae_advantages,
    pack_tokens,
    shape_rewards,
    value_predict,
    value_table_init,
    value_targets,
    whiten,
)
from token_ppo import explained_variance as ev_weighted
from token_ppo import fit_value_head
from failure_modes import (
    RunHistory,
    diagnose,
    distorted_policy,
    effective_support,
    fit_sequence_rm,
    mean_length,
    rm_features,
    sample_preferences_over_sequences,
    seq_kl,
    sequence_dist,
    sequence_dist_from_probs,
    standardize,
    token_entropy,
)


# ---------------------------------------------------------------------------
# Configuration (provided) -- one object so an ablation is `replace(cfg, k=v)`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLHFConfig:
    """Every knob of the loop, in one frozen object.

    W1 D5's ablation discipline lives here: an arm is `replace(base, one_field=value)`,
    never a hand-built config, so "we changed exactly one thing" is enforced by the
    type system rather than by care. `dataclasses.replace(base, **{k: v})` is the
    dynamic-key form; `replace(base, k=v)` passes the literal kwarg `k` and fails.
    """

    # rollout
    n_rollout: int | None = 128      # None => enumerate every sequence (exact expectations)
    rollout_temp: float = 1.0        # != 1.0 is off-policy-by-accident
    rollout_top_p: float = 1.0       # < 1.0 is the UNREPAIRABLE one

    # credit assignment
    gamma: float = 1.0
    lam: float = 0.95
    use_critic: bool = True
    value_lr: float = 0.5
    value_steps: int = 60

    # the surrogate
    eps: float | None = 0.2          # None => no clipping at all
    n_epochs: int = 4
    n_minibatches: int = 4
    lr: float = 0.15
    whiten_mode: str | None = "center"
    reduce: str = "sum"

    # the KL term
    beta0: float = 0.5
    adaptive_beta: bool = True
    kl_target: float = 1.0
    k_beta: float = 0.2
    beta_clip: float = 0.2           # relative-error clip, InstructGPT uses 0.2
    kl_estimator: str = "k1"

    # outer loop
    n_iters: int = 20
    checkpoint_every: int = 1


# ---------------------------------------------------------------------------
# Fixtures and ground truth (provided)
# ---------------------------------------------------------------------------


def gold_reward_fn(env: TokenEnv):
    """Day 1's structured sequence reward, unchanged across the whole week.

        2 * (count of token 0)  -  0.4 * len(y)  +  0.75 * [ends with eos]

    Note the SIGN of the length term: gold PENALISES length. Every length result this
    week is the gap between that sign and what a confounded annotator teaches the RM.
    """
    def r(seq: tuple[int, ...]) -> float:
        n_good = sum(1 for a in seq if a == 0)
        ends_clean = 1.0 if seq and seq[-1] == env.eos else 0.0
        return 2.0 * n_good - 0.4 * len(seq) + 0.75 * ends_clean
    return r


def sequence_reward_fn(env: TokenEnv, values: np.ndarray):
    """Turn a vector aligned with `enumerate_sequences(env)` into the `reward_fn(seq)`
    callable that `shape_rewards` and `exact_J` want. Provided plumbing."""
    index = {s: i for i, s in enumerate(enumerate_sequences(env))}
    vals = np.asarray(values, float)

    def r(seq: tuple[int, ...]) -> float:
        return float(vals[index[seq]])
    return r


def demonstration_set(env: TokenEnv, rng: np.random.Generator, n: int,
                      temperature: float = 1.0) -> list[tuple[int, ...]]:
    """A "human demonstration" corpus: sequences drawn proportional to exp(gold/temp).

    Provided, and deliberately NOT the gold argmax. Real SFT data is good-but-not-optimal
    and diverse, which is what leaves room for RL to help at all -- a reference already
    at the optimum makes every later number zero and the day's headline unmeasurable.
    """
    seqs = enumerate_sequences(env)
    gold = np.array([gold_reward_fn(env)(s) for s in seqs])
    p = softmax_np(gold / temperature)
    idx = rng.choice(len(seqs), size=n, p=p)
    return [seqs[i] for i in idx]


def build_reward_model(env: TokenEnv, theta_ref: np.ndarray, rng: np.random.Generator,
                       n_pairs: int = 30, n_spurious: int = 20, noise: float = 0.15,
                       rm_steps: int = 3000, rm_lr: float = 4.0) -> np.ndarray:
    """Fit D4's LINEAR Bradley-Terry RM on preferences drawn from pi_ref. Provided.

    Returns the STANDARDISED proxy reward, one entry per enumerated sequence.

    Two fixture decisions carried over from D4, both load-bearing:
      * LINEAR over 23 features (3 real + 20 spurious), not a table. A table has no
        generalisation, so its starved tail never improves and Gao's mechanism is
        inexpressible. Results about "reward models" are usually results about FUNCTION
        APPROXIMATORS.
      * pi_ref is the SAMPLER for the comparisons, which is why RM error concentrates
        exactly where the policy will go, and why KL-to-reference is a CONFIDENCE
        REGION for the RM rather than a generic regulariser.
      * STANDARDISED under pi_ref, because a BT fit's scale is set by how separable its
        data happened to be. Mixing standardised and raw scales is the fastest way to
        get a beta prediction wrong by three orders of magnitude.
    """
    seqs = enumerate_sequences(env)
    gold = np.array([gold_reward_fn(env)(s) for s in seqs])
    p_ref = sequence_dist(env, theta_ref)
    feats = rm_features(env, n_spurious=n_spurious, seed=int(rng.integers(1 << 30)))
    pairs = sample_preferences_over_sequences(rng, gold, p_ref / p_ref.sum(), n_pairs, noise)
    omega = fit_sequence_rm(feats, pairs, lr=rm_lr, n_steps=rm_steps)
    return standardize(feats @ omega, p_ref)


def rollout_policy_probs(env: TokenEnv, theta: np.ndarray, cfg: RLHFConfig) -> np.ndarray:
    """The distribution rollouts ACTUALLY come from, as probabilities. Provided.

    At the defaults (temp 1.0, top_p 1.0) this is exactly softmax(theta) and the loop is
    on-policy. Anything else is off-policy-by-accident: the batch will still be packed
    with `logp_old` from THETA, because that is precisely the bug -- the trainer believes
    the samples came from theta and they did not.
    """
    if cfg.rollout_temp == 1.0 and cfg.rollout_top_p >= 1.0:
        return softmax_np(theta)
    return distorted_policy(theta, cfg.rollout_temp, cfg.rollout_top_p)


def collect(env: TokenEnv, theta: np.ndarray, theta_ref: np.ndarray,
            cfg: RLHFConfig, rng: np.random.Generator) -> TokenBatch:
    """One rollout batch. Provided, because the lesson is in what it FREEZES, not in how
    it samples.

    `cfg.n_rollout is None` enumerates every terminating sequence and weights it by its
    exact probability, which turns every estimator downstream into an exact expectation.
    That is the week's design decision and it is why unbiasedness is an assertion here.
    """
    pi_roll = rollout_policy_probs(env, theta, cfg)
    if cfg.n_rollout is None:
        seqs_all = enumerate_sequences(env)
        w_all = sequence_dist_from_probs(env, pi_roll)
        # A truncated sampler assigns EXACT ZEROS, and a sequence the sampler cannot
        # produce is not data -- keeping it would put unreachable rows in the batch and
        # a minibatch of them would divide by a zero weight. This is D4's "7.2% of pi's
        # mass is unreachable" made structural rather than remembered.
        keep = w_all > 0.0
        seqs = [s for s, k in zip(seqs_all, keep) if k]
        w = w_all[keep]
    else:
        # Sampling from a probability table, not from logits: a top-p policy contains
        # exact zeros and therefore has no finite logit representation.
        seqs = _sample_from_probs(env, pi_roll, cfg.n_rollout, rng)
        w = np.full(len(seqs), 1.0 / len(seqs))
    return pack_tokens(env, seqs, theta, theta_ref, w)


def _sample_from_probs(env: TokenEnv, pi: np.ndarray, n: int,
                       rng: np.random.Generator) -> list[tuple[int, ...]]:
    """`sample_sequences` but from a probability table. Provided plumbing."""
    out = []
    for _ in range(n):
        prefix: tuple[int, ...] = ()
        while True:
            a = int(rng.choice(env.vocab, p=pi[env.state_id(prefix)]))
            prefix = prefix + (a,)
            if a == env.eos or len(prefix) >= env.t_max:
                break
        out.append(prefix)
    return out


def exact_panel(env: TokenEnv, theta: np.ndarray, theta_ref: np.ndarray,
                proxy: np.ndarray, gold: np.ndarray) -> dict:
    """THE INSTRUMENT PANEL, computed EXACTLY by enumeration. Provided.

    Note what this is not: it is not read off the batch. The updates may be sampled, but
    every MEASUREMENT here is an exact expectation under the current policy, so an
    ablation's numbers are never confounded with its rollout noise. Real stacks cannot do
    this and that is the honest caveat on every plot in the deliverable.

    Returns floats only (D3's rule: log floats, never tensors or arrays).
    """
    p = sequence_dist(env, theta)
    p = p / p.sum()
    pi = softmax_np(theta)
    occ = state_occupancy(env, theta)
    return dict(
        proxy=float(p @ proxy),
        gold=float(p @ gold),
        kl=float(exact_sequence_kl(env, theta, theta_ref)),
        entropy=float(token_entropy(env, pi, occ)),
        length=float(mean_length(env, p)),
        support=float(effective_support(p)),
    )


def closed_form_frontier(env: TokenEnv, theta_ref: np.ndarray, rewards: np.ndarray,
                         betas: np.ndarray) -> dict:
    """The EXACT reward/KL frontier, by W3 D5's closed form. Provided ground truth.

        p*_beta(y)  proportional to  p_ref(y) exp(r(y) / beta)

    For each beta this is the true argmax of `E[r] - beta KL(p || p_ref)` over ALL
    sequence distributions -- i.e. the inner problem of the Lagrangian solved to
    OPTIMALITY, which is exactly what dual ascent assumes and PPO does not deliver.

    Returns dict of arrays sorted by increasing KL: 'beta', 'kl', 'er'.
    """
    p_ref = sequence_dist(env, theta_ref)
    p_ref = p_ref / p_ref.sum()
    betas = np.asarray(betas, float)
    kl, er = np.empty(len(betas)), np.empty(len(betas))
    for i, b in enumerate(betas):
        p = tilted_sequence_dist(env, theta_ref, rewards, b)
        kl[i] = seq_kl(p, p_ref)
        er[i] = float(p @ rewards)
    order = np.argsort(kl)
    return dict(beta=betas[order], kl=kl[order], er=er[order])


def frontier_beta_at_kl(frontier: dict, kl: float) -> float:
    """The frontier's beta at a given KL, by interpolation. Provided."""
    return float(np.interp(kl, frontier["kl"], frontier["beta"]))


def frontier_slope_at_kl(frontier: dict, kl: float, window: int = 1) -> float:
    """The MEASURED secant slope dE[r]/dKL at a given KL. Provided, and it is the
    structurally independent half of `stage_dual`: it is a finite difference of two
    tilted distributions and contains no beta anywhere in its computation.

    The envelope identity says this equals beta. Nothing in the code makes it so.
    """
    kls, ers = frontier["kl"], frontier["er"]
    i = int(np.clip(np.searchsorted(kls, kl), window, len(kls) - window - 1))
    lo, hi = i - window, i + window
    return float((ers[hi] - ers[lo]) / (kls[hi] - kls[lo]))


def exact_tilted_kl(env: TokenEnv, theta_ref: np.ndarray, rewards: np.ndarray,
                    beta: float) -> float:
    """KL(p*_beta || p_ref) for the EXACT inner solution at this beta. Provided.

    This is the "primal solved to optimality" oracle that `stage_dual` tier 1 uses, so
    that a failure there is unambiguously the CONTROLLER and never the optimiser.
    """
    p_ref = sequence_dist(env, theta_ref)
    p = tilted_sequence_dist(env, theta_ref, rewards, beta)
    return seq_kl(p, p_ref / p_ref.sum())


# ---------------------------------------------------------------------------
# Exercises (TODO bodies -- the learner implements these)
# ---------------------------------------------------------------------------


def sft_reference(env: TokenEnv, demos: list[tuple[int, ...]], n_steps: int = 400,
                  lr: float = 0.5, theta0: np.ndarray | None = None) -> np.ndarray:
    """Produce pi_ref by SUPERVISED fitting to a demonstration set. Returns theta.

    The reference must be TRAINED, not handed over, because its gold reward is the bar
    every later run has to clear and a hand-built reference makes that bar arbitrary.

    This is W1's LM objective, on a tabular softmax:

        L(theta) = - (1/N) sum_{y in demos} sum_t log pi_theta(a_t | s_t)
        grad_{z(s)} [ -log pi(a|s) ] = p(.|s) - e_a          <- W1's dCE/dz, exactly

    so the gradient is a SCATTER-ADD of `(p(.|s) - e_a)` into row `s`, once per (state,
    token) pair in the corpus, divided by N. Descend it.

    Three things the checks will find:

      * SCATTER-ADD, not fancy-index assignment. Many demo tokens land on the same
        state -- `g[states] += c` keeps only the last write. `np.add.at`, index as a
        tuple. (Fourth sighting of this family in three weeks.)
      * FLOAT accumulator. `np.zeros(shape, dtype=int)` truncates every update to 0 and
        reports a perfectly clean, perfectly stationary fit.
      * States that appear in NO demonstration must not move. They receive no gradient,
        so this is automatic if the scatter is right and impossible if it is not --
        which is why the check asserts it rather than trusting it.

    `theta0=None` means start from zeros (a uniform policy), which is the honest
    "before SFT" condition. Do not normalise, clip or temperature-scale the result.
    """
    if theta0 is None:
        theta = np.zeros((env.n_states, env.vocab), dtype=np.float64)
    else:
        theta = theta0.copy()
    
    pairs   = [(env.state_id(y[:t]), a) for y in demos for t, a in enumerate(y)]
    states  = np.array([s for s, _ in pairs], dtype=np.int64)   # (N,) int64
    actions = np.array([a for _, a in pairs], dtype=np.int64)   # (N,) int64
    onehot  = np.eye(env.vocab)[actions]                        # (N, vocab) float
    N = len(states)
    
    for _ in range(n_steps):
        grad = np.zeros_like(theta, dtype=np.float64)
        prob = softmax_np(theta, axis=-1)
        np.add.at(grad, states, prob[states] - onehot)
        theta -= lr * grad/actions.shape[0]
    return theta 


def kl_controller(beta: float, kl: float, kl_target: float, k_beta: float,
                  clip: float = 0.2) -> float:
    """InstructGPT's adaptive-KL update. One line of algebra, three guards.

        beta <- beta * (1 + k_beta * clip_pm(kl / kl_target - 1))

    This is PROPORTIONAL DUAL ASCENT on the Lagrange multiplier of

        max_theta E[r]   s.t.   KL(pi_theta || pi_ref) <= delta

    obtained from the additive dual step `beta <- beta + eta (KL - delta)` by setting
    `eta = k_beta * beta / delta`. Derive it once, then never memorise the formula
    again: the SIGN is forced (dg/dbeta = delta - KL by the envelope theorem, and we
    MINIMISE the dual), and the multiplicative form is forced by two facts --

      * it is a step in LOG beta, hence scale-free, and beta spans orders of magnitude
        across tasks;
      * it keeps beta > 0 automatically, and beta < 0 is not merely numerically bad, it
        is INFEASIBLE: beta is the multiplier on an inequality constraint, so a negative
        beta pays the policy to violate its own budget.

    Requirements the stage asserts, each of which is a real bug someone shipped:

      1. FIXED POINT. `kl == kl_target` returns beta UNCHANGED, exactly.
      2. SIGN. Over budget => beta RISES. Under budget => beta FALLS. Getting this
         backwards produces a run that looks healthy and diverges, so the check asserts
         the DIRECTION and not just the magnitude.
      3. POSITIVITY, for ANY input including a NEGATIVE kl. k1 is a signed estimator and
         goes negative on 20-44% of samples (W3 D5), so `kl < 0` is not a hypothetical.
         Note `clip` alone does not save you unless `k_beta * clip < 1`; think about
         which of the two guards you are actually relying on and make it explicit.
      4. The RELATIVE ERROR is clipped to +/- `clip`, so one noisy KL estimate cannot
         move beta by 10x. Clip the ERROR, not the resulting beta.

    Return a plain Python float.
    """
    error = np.clip(kl / kl_target - 1.0, -clip, clip)
    return float(max(beta * (1 + k_beta * error), 1e-12))


def rlhf_iteration(env: TokenEnv, theta_old: np.ndarray, phi_old: np.ndarray, beta: float,
                   cfg: RLHFConfig, rng: np.random.Generator, *, proxy_fn,
                   theta_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """ONE FULL ITERATION of the loop. Returns `(theta_new, phi_new, beta_new, record)`.

    The order below is the exercise, and the ONCE-PER-BATCH / PER-MINIBATCH split is the
    single most common implementation bug in PPO -- and it is SILENT.

    ONCE PER BATCH, all frozen at theta_old:
      1. batch    = collect(env, theta_old, theta_ref, cfg, rng)
                    (`logp_old` and `logp_ref` are cached inside it -- that is the whole
                     reason a production stack can drop pi_ref from memory here)
      2. rewards  = shape_rewards(env, batch, theta_old, proxy_fn, beta, cfg.kl_estimator)
                    NOTE: shaped with the CURRENT beta, at rollout time. The KL term is
                    REWARD SHAPING, not a loss term -- no estimator below knows it exists.
      3. values   = value_predict(phi_old, batch.value_ids)          -> (B, T+1)
                    When `cfg.use_critic` is False, phi_old is the pinned-zero table and
                    stays that way: that arm is "baseline only", and the baseline is the
                    centering in step 5, not the critic.
      4. adv      = gae_advantages(rewards, values, batch.dones, cfg.gamma, cfg.lam)
      5. targets  = value_targets(adv, values)          <- BEFORE whitening. The critic
                    regresses onto RETURNS, which live in reward units; whitened
                    advantages do not. Whitening first trains fine and is wrong.
      6. adv      = whiten(adv, batch.tok_w, batch.mask, center_only=...) if asked

    ===== FREEZE. Nothing above is recomputed below this line. =====

      7. for k in 1..cfg.n_epochs:  shuffle sequence indices with `rng`
             for m in 1..cfg.n_minibatches:
                 pb   = batch.as_ppo_batch(adv_eff).subset(idx)
                 g    = surrogate_gradient(env, theta, pb, cfg.eps)
                 theta = theta + cfg.lr * g            <- ASCENT; this is an objective
             accumulate clip_frac / oob_frac from `clipped_surrogate` at each step
      8. critic: fit_value_head(env, batch, targets, cfg.value_steps, cfg.value_lr,
                                phi0=phi_old)   -- only when cfg.use_critic
      9. beta_new = kl_controller(beta, <measured KL>, cfg.kl_target, cfg.k_beta,
                                  cfg.beta_clip)  when cfg.adaptive_beta, else beta

    `cfg.reduce == "mean"` scales each sequence's advantages by 1/len(y_b) before the
    surrogate (D3's switch, and D4's length-prior result: sum vs mean is a LENGTH PRIOR,
    not a normalisation).

    THE MEASURED KL for step 9 is the per-token KL summed over a sequence, averaged over
    the batch -- i.e. exactly what `shape_rewards` already computes, and exactly what a
    real stack has: an ESTIMATE from the rollout, not the enumerated truth. Use the
    estimate. Feeding the controller the exact KL would make `stage_dual` pass for the
    wrong reason and would not be reproducible outside this toy.

    `record` is a dict of PYTHON FLOATS (plus the ints you need), carrying at least:
        'beta', 'beta_new', 'kl_hat', 'clip_frac', 'oob_frac', 'adv_mean', 'adv_std',
        'ev', 'value_mse', 'logratio_std', 'proxy_batch', 'mean_len'
    where
        'ev'           = ev_weighted(values[:, :-1], targets, batch.tok_w) over real
                         tokens -- the critic you ROLLED OUT with, scored against the
                         targets it is about to be fitted to;
        'logratio_std' = the token-weighted std of log rho over the FINAL minibatch step.
                         This is the day's asymmetric probe: if you recompute `logp_old`
                         inside the epoch loop then rho == 1 identically, clip_frac reads
                         a healthy 0.000, the loss curve says nothing (L_IS(theta_old) =
                         E[A] ~ 0 BY CONSTRUCTION), and you have silently replaced PPO
                         with n_epochs steps of vanilla PG on stale data with no trust
                         region. `std(log rho)` separates the two hypotheses that
                         `clip_frac` cannot.

    Do not mutate `theta_old`, `phi_old` or `batch`.
    """
    theta = theta_old.copy()
    record = {'clip_frac': 0.0, 'oob_frac': 0.0, 'logratio_std': 0.0}
    batch = collect(env, theta_old, theta_ref, cfg, rng)
    rewards = shape_rewards(env, batch, theta_old, proxy_fn, beta, cfg.kl_estimator)
    values = value_predict(phi_old, batch.value_ids)
    adv = gae_advantages(rewards, values, batch.dones, cfg.gamma, cfg.lam)
    targets = value_targets(adv, values)

    if cfg.whiten_mode is None: 
        pass
    elif cfg.whiten_mode == 'center':
        adv = whiten(adv, batch.tok_w, batch.mask, True)
    else:
        adv = whiten(adv, batch.tok_w, batch.mask, False)

    if cfg.reduce == 'mean':
        adv = adv / batch.lengths[:, None]

    ppo_batch = batch.as_ppo_batch(adv)
    n_steps_taken = 0
    for _ in range(cfg.n_epochs):
        split_idx_permuted = np.array_split(rng.permutation(batch.B), cfg.n_minibatches)
        for i in range(cfg.n_minibatches):
            # An EMPTY split is reachable and not hypothetical: under top-p the reachable
            # sequence set ratchets down (measured 40 -> 9 in twelve iterations), and once
            # B < n_minibatches `array_split` hands back a zero-row array. Its weights sum
            # to 0, so every downstream mean is 0/0. A minibatch with no tokens carries no
            # gradient -- skip it, and count only the steps actually taken.
            if len(split_idx_permuted[i]) == 0:
                continue
            minibatch = ppo_batch.subset(split_idx_permuted[i])
            logratios = token_logratios(env, theta, minibatch)
            loss_clipped, log_clipped = clipped_surrogate(logratios, minibatch, cfg.eps)
            record["clip_frac"] += log_clipped['clip_frac']
            record["oob_frac"] += log_clipped['oob_frac']
            # Token-weighted std of log rho, overwritten each step so the FINAL step's
            # value survives. This is the probe: it is exactly 0 iff logp_old was
            # recomputed from the live theta, which clip_frac cannot distinguish.
            w_mb = minibatch.weights[:, None] * minibatch.mask
            w_mb = w_mb / w_mb.sum()
            mu_lr = float((w_mb * logratios).sum())
            record["logratio_std"] = float(np.sqrt((w_mb * (logratios - mu_lr) ** 2).sum()))
            g = surrogate_gradient(env, theta, minibatch, cfg.eps)
            theta = theta + cfg.lr * g
            n_steps_taken += 1
    record["clip_frac"] /= max(n_steps_taken, 1)
    record["oob_frac"] /= max(n_steps_taken, 1)
    
    if cfg.use_critic:
        phi_new, log_vf = fit_value_head(env, batch, targets, cfg.value_steps, cfg.value_lr, phi_old)
        value_mse = float(log_vf['mse'][-1])
    else:
        phi_new = phi_old.copy()
        value_mse = float('nan')
    
    lr = (np.where(batch.mask, batch.logp_ref, 0) - np.where(batch.mask, batch.logp_old, 0))
    if cfg.kl_estimator == 'k1':
        kl = (batch.weights * (-lr).sum(axis=-1)).sum()
    else:
        kl = (batch.weights * (-lr + np.exp(lr) - 1).sum(axis=-1)).sum()
    record['kl_hat'] = float(kl)

    if cfg.adaptive_beta:
        beta_new = kl_controller(beta, kl, cfg.kl_target, cfg.k_beta, cfg.beta_clip)
    else:
        beta_new = beta

    # ---- the panel ---------------------------------------------------------
    # Every entry a PLAIN PYTHON FLOAT (D3's rule: a log you cannot diff across
    # iterations is not an instrument). np.float64 is not float and the checks say so.
    m, w = batch.mask, batch.tok_w
    adv_mean = float((w * adv).sum())
    record.update(
        beta=float(beta),
        beta_new=float(beta_new),
        clip_frac=float(record["clip_frac"]),
        oob_frac=float(record["oob_frac"]),
        # Moments of the advantages ACTUALLY handed to the surrogate -- after whitening
        # and after the sum/mean reduce, because that is what the gradient saw.
        adv_mean=adv_mean,
        adv_std=float(np.sqrt((w * (adv - adv_mean) ** 2).sum())),
        # EV of the critic we ROLLED OUT with, scored against the targets it is about to
        # be fitted to. Boolean-mask flattening is correct HERE: EV is a scalar over a bag
        # of tokens with no per-sequence structure to preserve -- the opposite verdict
        # from the KL two lines up, and the difference is why one of them was a bug.
        ev=float(ev_weighted(values[:, :-1][m], targets[m], w[m])),
        value_mse=value_mse,
        proxy_batch=float(sum(batch.weights[b] * proxy_fn(batch.seqs[b])
                              for b in range(batch.B))),
        mean_len=float((batch.weights * batch.lengths).sum()),
    )

    return (theta, phi_new, beta_new, record)


def train_rlhf(env: TokenEnv, theta_ref: np.ndarray, cfg: RLHFConfig,
               rng: np.random.Generator, *, proxy: np.ndarray, gold: np.ndarray,
               label: str = "") -> tuple[RunHistory, list[dict]]:
    """The outer loop. Returns `(history, checkpoints)`.

    Start from `theta = theta_ref.copy()` -- RLHF initialises the policy AT the SFT model,
    which is why iteration 0's KL is exactly 0 and why the reference is both the starting
    point and the anchor. Start the critic from `value_table_init(env)`.

    Each iteration: call `rlhf_iteration`, then measure the panel EXACTLY with
    `exact_panel` and `history.log(...)` it. Log `ev` from the record (or `float('nan')`
    when there is no critic -- `diagnose` already filters NaNs, and a fake 0.0 there
    would fire `value_divergence` on every critic-free arm).

    A CHECKPOINT is `{'iter', 'theta', 'phi', 'beta'}` plus the panel dict, appended every
    `cfg.checkpoint_every` iterations AND at iteration 0 (the reference itself is a
    legitimate thing to ship, and on a badly-configured run it is the right answer).

    Checkpointing is part of the ALGORITHM, not ops: gold is concave in sqrt(KL) and bends
    down, so the run's FINAL policy is systematically the wrong one to ship.

    Set `history.label = label`; `diagnose` must never read it.
    """
    theta = theta_ref.copy()
    phi = value_table_init(env)
    beta = float(cfg.beta0)
    history = RunHistory()
    history.label = label
    checkpoints: list[dict] = []
    proxy_fn = sequence_reward_fn(env, proxy)   # the closure, built ONCE

    def record_panel(step: int, ev: float) -> None:
        # Reads theta/phi/beta from the ENCLOSING scope, so it always sees the values
        # the loop most recently rebound -- a closure captures the variable, not a
        # snapshot of it. Same mechanism as `proxy_fn` above.
        panel = exact_panel(env, theta, theta_ref, proxy, gold)
        history.log(proxy=panel["proxy"], gold=panel["gold"], kl=panel["kl"],
                    entropy=panel["entropy"], length=panel["length"], ev=ev)
        if step == 0 or step % cfg.checkpoint_every == 0:
            # .copy() or every checkpoint aliases the same live array and they all come
            # out identical -- the failure that makes checkpoint selection meaningless
            # while looking like it worked.
            checkpoints.append(dict(iter=step, theta=theta.copy(), phi=phi.copy(),
                                    beta=float(beta), **panel))

    # Iteration 0 IS a checkpoint: pi_ref is the bar, its KL is exactly 0, and on a
    # badly-configured run "ship the SFT model" is the correct answer. `ev` is nan here
    # because no critic has been fitted yet -- and nan, not 0.0, because `diagnose`
    # filters nans and would read a 0.0 as a real (and alarming) explained variance.
    record_panel(0, float("nan"))
    for step in range(1, cfg.n_iters + 1):
        theta, phi, beta, rec = rlhf_iteration(env, theta, phi, beta, cfg, rng,
                                               proxy_fn=proxy_fn, theta_ref=theta_ref)
        record_panel(step, rec["ev"] if cfg.use_critic else float("nan"))

    return history, checkpoints


def select_checkpoint(history: RunHistory, checkpoints: list[dict],
                      gold_key: str = "gold") -> tuple[int, str]:
    """THE SHIPPING DECISION. Returns `(index_into_checkpoints, reason)`.

    Pick the checkpoint at the GOLD turnover -- the argmax of gold along the KL ladder --
    and return a reason string that a human could act on, quoting the numbers: which
    iteration, its gold and KL, what the FINAL iteration's gold and KL were, and how many
    nats of drift you are declining to ship.

    Two properties the stage asserts:
      * on a run that over-optimises, this must NOT return the final checkpoint. That is
        the entire point of the function.
      * it must return something at or near the gold argmax on every run, including the
        healthy one -- where the answer legitimately may be the final checkpoint.

    Tie-break toward the SMALLER KL. Two checkpoints with equal gold are not equally good:
    the closer one to pi_ref has less RM extrapolation behind it, so it is the safer
    estimate of the same quantity.

    Say the honest thing in the reason string when it applies: this selection used a gold
    eval that exists only because the environment is synthetic. In production you are
    choosing on an eval set, and the whole game is whether that eval is independent of
    the RM.
    """
    golds = np.array([float(c[gold_key]) for c in checkpoints])
    kls = np.array([float(c.get("kl", 0.0)) for c in checkpoints])
    # Tie-break toward the SMALLER KL: two checkpoints with equal gold are not equally
    # good, because the closer one to pi_ref has less RM extrapolation behind it and is
    # therefore the safer estimate of the same quantity.
    cand = np.flatnonzero(golds >= golds.max() - 1e-12)
    idx = int(cand[int(np.argmin(kls[cand]))])
    reason = (
        f"gold peaks at {golds[idx]:.4f} at iteration "
        f"{checkpoints[idx].get('iter', idx)} (KL {kls[idx]:.3f}); the final iterate "
        f"scores {golds[-1]:.4f} at KL {kls[-1]:.3f}, so shipping it would cost "
        f"{golds[idx] - golds[-1]:+.4f} gold for {kls[-1] - kls[idx]:+.3f} extra nats "
        f"of drift. Early stopping here is the ESTIMATOR for the optimum, not a "
        f"regulariser. Caveat: selected on a GOLD eval that exists only because this "
        f"environment is synthetic -- in production this is an eval set, and the whole "
        f"game is whether it is independent of the RM."
    )
    return idx, reason


def ablate(env: TokenEnv, theta_ref: np.ndarray, base: RLHFConfig, arms: dict,
           seeds: list[int], *, proxy: np.ndarray, gold: np.ndarray) -> dict:
    """THE COMPARISON GRID. Returns `{arm_name: {...}}`.

    `arms` maps a name to a dict of RLHFConfig overrides; build each arm's config with
    `replace(base, **overrides)` so that "we changed exactly one thing" is structural.

    Run every arm at every seed. >= 3 SEEDS IS MANDATORY HERE, NOT ADVISORY: D4 measured
    seed spread in gold amplifying 89x / 259x / 667x with optimisation pressure, and at
    n=30 comparisons the spread (+/-1.60) EXCEEDED the entire mean improvement over
    pi_ref (+0.95). A one-seed row at the aggressive end of this grid is noise with a
    decimal point on it.

    Hold the ROLLOUT BUDGET FIXED across arms (W1 D5): `n_rollout * n_epochs *
    n_minibatches` is not the budget -- the budget is `n_rollout * n_iters`, because
    generation is what costs money. The REINFORCE arm gets the same rollouts and simply
    does less with them; that is the honest baseline PPO has to beat.

    Each arm's entry must carry, as floats: mean and spread (max - min over seeds, or a
    std -- state which) of final `gold`, `proxy`, `kl`, `entropy`, `length`, plus the
    SELECTED checkpoint's gold and KL, and the union of `diagnose` verdicts across seeds.
    Keep one `RunHistory` per seed so the stage can re-run `diagnose` itself.
    """
    out: dict = {}
    for name, overrides in arms.items():
        # replace() so "we changed exactly one thing" is structural, not a promise.
        cfg = replace(base, **overrides)
        finals, hists = [], []
        for seed in seeds:
            hist, ckpts = train_rlhf(env, theta_ref, cfg,
                                     np.random.default_rng(1000 + seed),
                                     proxy=proxy, gold=gold, label=name)
            i, _ = select_checkpoint(hist, ckpts)
            finals.append(dict(gold=hist.gold[-1], proxy=hist.proxy[-1], kl=hist.kl[-1],
                               entropy=hist.entropy[-1], length=hist.length[-1],
                               sel_gold=ckpts[i]["gold"], sel_kl=ckpts[i]["kl"]))
            hists.append(hist)
        agg: dict = {}
        for k in ("gold", "proxy", "kl", "entropy", "length"):
            v = np.array([f[k] for f in finals])
            agg[f"{k}_mean"] = float(v.mean())
            agg[f"{k}_spread"] = float(v.max() - v.min())   # RANGE over seeds, not std
        agg["selected_gold"] = float(np.mean([f["sel_gold"] for f in finals]))
        agg["selected_kl"] = float(np.mean([f["sel_kl"] for f in finals]))
        agg["verdicts"] = sorted({v for h in hists for v, _ in diagnose(h)})
        agg["histories"] = hists
        out[name] = agg
    return out


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


@dataclass
class Ctx:
    """Lazy fixtures. Nothing is built until a stage asks for it, so `--list` is instant
    and a stage that only needs the controller never fits a reward model."""

    fast: bool = False
    _cache: dict = field(default_factory=dict)

    @cached_property
    def env(self) -> TokenEnv:
        # `--fast` NEVER changes the environment or the iteration counts -- it reduces
        # rollout size and RM fitting steps only. A "fast" mode that changes the fixture's
        # physics is not a fast mode, it is a different experiment wearing the same name,
        # and its green checks tell you nothing about the real one.
        return TokenEnv(vocab=4, t_max=5)

    @cached_property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(20260801)

    @cached_property
    def demos(self) -> list[tuple[int, ...]]:
        return demonstration_set(self.env, np.random.default_rng(11), 600, temperature=1.5)

    @cached_property
    def theta_ref(self) -> np.ndarray:
        # The loss is a MEAN over ~1500 demo tokens, so every coefficient's gradient is
        # O(1/N) and a "safe-looking" lr=0.5 does not move. Same trap as D4's BT fit.
        return sft_reference(self.env, self.demos, n_steps=3000, lr=5.0)

    @cached_property
    def gold(self) -> np.ndarray:
        g = gold_reward_fn(self.env)
        return np.array([g(s) for s in enumerate_sequences(self.env)])

    @cached_property
    def proxy(self) -> np.ndarray:
        # 30 comparisons, 15% annotator noise. STARVED ON PURPOSE: at 400 pairs this
        # RM is so close to gold that optimising the proxy also optimises gold and the
        # week's headline inversion is unmeasurable. D4's spurious weights (32.07 at
        # n=30 vs 4.29 at n=3000) are what the policy climbs.
        return build_reward_model(self.env, self.theta_ref, np.random.default_rng(7),
                                  n_pairs=30, noise=0.15,
                                  rm_steps=800 if self.fast else 3000)

    @cached_property
    def gold_std(self) -> np.ndarray:
        """Gold in the SAME standardised units as the proxy, so a beta ladder means the
        same thing under both and the two frontiers are comparable."""
        p_ref = sequence_dist(self.env, self.theta_ref)
        return standardize(self.gold, p_ref)

    @cached_property
    def frontier(self) -> dict:
        betas = np.geomspace(0.02, 20.0, 140)
        return closed_form_frontier(self.env, self.theta_ref, self.proxy, betas)

    @cached_property
    def cfg(self) -> RLHFConfig:
        # n_rollout is NOT reduced by --fast either: at 48 rollouts the no-KL arm's
        # gradient is too noisy to drift far enough in 24 iterations and the headline
        # inversion silently flips sign. The whole suite runs in ~30s; --fast trims the
        # RM fit and the dual ladder, which are cost without content.
        return RLHFConfig(n_iters=20, n_rollout=128)

    @cached_property
    def proxy_fn(self):
        return sequence_reward_fn(self.env, self.proxy)


def _fail_fast(ctx: Ctx, *needed: str) -> None:
    """Probe each required TODO with a trivial call so an unimplemented function reports
    [TODO] and skips instead of exploding halfway through an expensive stage."""
    env = ctx.env
    probes = {
        "sft_reference": lambda: sft_reference(env, [(env.eos,)], n_steps=1, lr=0.1),
        "kl_controller": lambda: kl_controller(1.0, 1.0, 1.0, 0.1, 0.2),
        "rlhf_iteration": lambda: rlhf_iteration(
            env, np.zeros((env.n_states, env.vocab)), value_table_init(env), 0.5,
            replace(ctx.cfg, n_iters=1, n_epochs=1, n_minibatches=1, n_rollout=8,
                    value_steps=1),
            np.random.default_rng(0), proxy_fn=lambda s: 0.0,
            theta_ref=np.zeros((env.n_states, env.vocab))),
        "train_rlhf": lambda: train_rlhf(
            env, np.zeros((env.n_states, env.vocab)),
            replace(ctx.cfg, n_iters=1, n_epochs=1, n_minibatches=1, n_rollout=8,
                    value_steps=1),
            np.random.default_rng(0), proxy=np.zeros(len(ctx.gold)), gold=ctx.gold),
        "select_checkpoint": lambda: select_checkpoint(RunHistory(gold=[0.0, 1.0]),
                                                       [{"iter": 0, "gold": 0.0, "kl": 0.0},
                                                        {"iter": 1, "gold": 1.0, "kl": 0.1}]),
        "ablate": lambda: ablate(env, np.zeros((env.n_states, env.vocab)),
                                 replace(ctx.cfg, n_iters=1, n_rollout=8, value_steps=1),
                                 {"a": {}}, [0], proxy=np.zeros(len(ctx.gold)),
                                 gold=ctx.gold),
    }
    for name in needed:
        try:
            probes[name]()
        except NotImplementedError:
            raise
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_sft(ctx: Ctx) -> None:
    """sft_reference -- pi_ref is TRAINED, and it is the bar every run must clear"""
    _fail_fast(ctx, "sft_reference")
    env, demos = ctx.env, ctx.demos
    theta_ref = ctx.theta_ref

    p = sequence_dist(env, theta_ref)
    _check("pi_ref is a proper sequence distribution (sums to 1)",
           abs(p.sum() - 1.0) < 1e-10, f"sum = {p.sum():.12f}")
    _check("pi_ref is finite everywhere", np.isfinite(theta_ref).all())

    # It must have LEARNED something: gold under pi_ref beats gold under the uniform
    # policy it started from. Without this the "bar" is not a bar.
    theta_unif = np.zeros_like(theta_ref)
    g_ref = float(p @ ctx.gold)
    g_unif = float(sequence_dist(env, theta_unif) @ ctx.gold)
    _check("SFT improved gold over the uniform init", g_ref > g_unif + 0.1,
           f"uniform {g_unif:.4f} -> SFT {g_ref:.4f}")

    # ...and it must NOT be optimal, or every later number is zero by construction.
    _check("SFT is well short of the gold optimum (room for RL to exist)",
           g_ref < ctx.gold.max() - 0.5, f"SFT {g_ref:.4f} vs best sequence {ctx.gold.max():.4f}")

    # States no demonstration reaches receive no gradient and must be untouched.
    seen = {env.state_id(s[:t]) for s in demos for t in range(len(s))}
    unseen = [i for i in range(env.n_states) if i not in seen]
    if unseen:
        _check("states absent from the demo corpus were never updated (scatter is correct)",
               np.allclose(theta_ref[unseen], 0.0, atol=1e-12),
               f"{len(unseen)} unseen states, max|theta| = {np.abs(theta_ref[unseen]).max():.2e}")

    # The MLE limit has a right answer: the empirical next-token frequencies.
    counts = np.zeros((env.n_states, env.vocab))
    for s in demos:
        for t, a in enumerate(s):
            counts[env.state_id(s[:t]), a] += 1.0
    hot = counts.sum(1) > 30
    emp = counts[hot] / counts[hot].sum(1, keepdims=True)
    fit = softmax_np(theta_ref)[hot]
    err = float(np.abs(emp - fit).max())
    _check("on well-visited states the fit converges to the empirical frequencies "
           "(the MLE has a right answer, not merely a decreasing loss)", err < 0.05,
           f"max |empirical - fitted| = {err:.4f} over {int(hot.sum())} states")
    print(f"  pi_ref: gold {g_ref:.4f}  entropy {token_entropy(env, softmax_np(theta_ref), state_occupancy(env, theta_ref)):.4f}"
          f"  mean length {mean_length(env, p):.4f}")


def stage_controller(ctx: Ctx) -> None:
    """kl_controller -- fixed point, SIGN, positivity under a negative KL, and the clip"""
    _fail_fast(ctx, "kl_controller")
    d, k = 1.5, 0.3

    # (1) FIXED POINT. Exactly at budget, beta must not move at all.
    for b in (0.01, 0.5, 7.25):
        _check(f"fixed point: KL == delta leaves beta unchanged (beta={b})",
               abs(kl_controller(b, d, d, k) - b) < 1e-15,
               f"{b} -> {kl_controller(b, d, d, k):.17g}")

    # (2) SIGN, in both directions. Backwards here trains and diverges.
    over, under = kl_controller(1.0, 2.0 * d, d, k), kl_controller(1.0, 0.1 * d, d, k)
    _check("sign: OVER budget => beta RISES", over > 1.0, f"beta 1.0 -> {over:.6f}")
    _check("sign: UNDER budget => beta FALLS", under < 1.0, f"beta 1.0 -> {under:.6f}")

    # (3) MONOTONICITY in the measured KL -- a proportional controller, not a switch.
    kls = np.linspace(0.0, 4 * d, 25)
    betas = np.array([kl_controller(1.0, float(x), d, k) for x in kls])
    _check("beta is non-decreasing in the measured KL", np.all(np.diff(betas) >= -1e-15))

    # (4) POSITIVITY for ANY input, including a NEGATIVE KL. k1 is signed and negative on
    #     20-44% of samples (W3 D5), so this is a live input, not a hypothetical.
    for bad in (-0.5, -10.0, 0.0):
        b = kl_controller(0.4, bad, d, k)
        _check(f"beta stays strictly positive and finite at KL={bad}",
               b > 0.0 and np.isfinite(b), f"-> {b:.6g}")

    # (5) The RELATIVE ERROR is clipped, so one noisy estimate cannot move beta 10x.
    huge = kl_controller(1.0, 1000.0 * d, d, k_beta=k, clip=0.2)
    _check("a wild KL estimate is bounded by the relative-error clip",
           abs(huge - (1.0 + k * 0.2)) < 1e-12, f"beta 1.0 -> {huge:.6f} (cap {1 + k * 0.2:.6f})")
    tiny = kl_controller(1.0, 0.0, d, k_beta=k, clip=0.2)
    _check("...symmetrically on the low side", abs(tiny - (1.0 - k * 0.2)) < 1e-12,
           f"beta 1.0 -> {tiny:.6f} (floor {1 - k * 0.2:.6f})")

    # (6) SCALE-FREEDOM: the update is a step in LOG beta, so the multiplicative factor
    #     must not depend on beta. This is the whole reason the shipped form is
    #     multiplicative and it is one line to assert.
    r1 = kl_controller(1e-3, 2 * d, d, k) / 1e-3
    r2 = kl_controller(1e+3, 2 * d, d, k) / 1e+3
    _check("scale-free: the multiplicative factor is identical at beta=1e-3 and beta=1e3",
           abs(r1 - r2) < 1e-12, f"{r1:.12f} vs {r2:.12f}")

    _check("returns a plain Python float", type(kl_controller(1.0, d, d, k)) is float)


def stage_regression(ctx: Ctx) -> None:
    """THE CHAINED REGRESSION: PPO -> GAE -> +baseline -> REINFORCE, one code path"""
    _fail_fast(ctx, "sft_reference", "rlhf_iteration")
    env, theta_ref = ctx.env, ctx.theta_ref
    theta0 = theta_ref.copy()
    zero_phi = value_table_init(env)

    # Enumerated batch, no clipping, ONE epoch and ONE minibatch, so the update is
    # exactly `theta0 + lr * g` for a single gradient g that we can name independently.
    base = replace(ctx.cfg, n_rollout=None, eps=None, n_epochs=1, n_minibatches=1,
                   lr=0.05, use_critic=False, value_steps=0, adaptive_beta=False,
                   whiten_mode=None, gamma=1.0, lam=1.0)
    proxy_fn = ctx.proxy_fn

    def update_of(cfg, phi=None, beta=0.0):
        th, _, _, rec = rlhf_iteration(env, theta0, zero_phi if phi is None else phi, beta,
                                       cfg, np.random.default_rng(0), proxy_fn=proxy_fn,
                                       theta_ref=theta_ref)
        return (th - theta0) / cfg.lr, rec

    # Independent reference: build the batch and the Psi ladder by hand, then use W3's
    # `score_function_gradient`, which knows nothing about ratios or clipping.
    batch = collect(env, theta0, theta_ref, base, np.random.default_rng(0))
    rewards = shape_rewards(env, batch, theta0, proxy_fn, 0.0, base.kl_estimator)
    values0 = value_predict(zero_phi, batch.value_ids)

    # (a) REINFORCE (W3 D2): beta=0, no clip, lam=1, no critic => A_t = reward-to-go.
    g_loop, rec = update_of(base)
    rtg = gae_advantages(rewards, values0, batch.dones, 1.0, 1.0)
    g_ref = score_function_gradient(env, theta0, batch.as_ppo_batch(rtg))
    _check("W3 D2 REINFORCE: the loop's update equals sum_t G_t grad log pi, ELEMENTWISE",
           np.allclose(g_loop, g_ref, atol=1e-12, rtol=0),
           f"max|diff| = {np.abs(g_loop - g_ref).max():.3e}")
    _check("...and rho == 1 exactly on the first step (one epoch cannot be off-policy)",
           rec["logratio_std"] < 1e-12, f"std(log rho) = {rec['logratio_std']:.3e}")

    # (b) + BASELINE (W3 D3). A constant baseline is FREE: it changes Psi elementwise but
    #     the exact gradient is IDENTICAL, because E[grad log pi] = 0 (EGLP). On an
    #     ENUMERATED batch that is an equality, not an approximation.
    g_center, _ = update_of(replace(base, whiten_mode="center"))
    _check("W3 D3 baseline: centering changes every advantage but NOT the exact gradient",
           np.allclose(g_center, g_ref, atol=1e-10, rtol=0),
           f"max|diff| = {np.abs(g_center - g_ref).max():.3e}")
    adv_c = whiten(rtg, batch.tok_w, batch.mask, center_only=True)
    shift = float(np.abs(adv_c[batch.mask] - rtg[batch.mask]).max())
    _check("...and the advantages really did change (the test is not vacuous)",
           shift > 1e-6 and shift > 1e6 * np.abs(g_center - g_ref).max(),
           f"advantages moved by {shift:.3e} while the gradient moved by "
           f"{np.abs(g_center - g_ref).max():.3e} -- {shift / max(np.abs(g_center - g_ref).max(), 1e-300):.1e}x")
    # The shift is SMALL here for a reason worth noticing: the RM is standardised under
    # pi_ref and the policy STARTS at pi_ref, so E[proxy] = 0 by construction. It is not
    # exactly 0 because `tok_w` is a TOKEN weighting and `standardize` used a SEQUENCE
    # weighting, and the gap between them is precisely a length/reward covariance -- the
    # same object as D2's +0.010993 residual in the PPO loss.

    # (c) A LEARNED CRITIC as the baseline: still W3 D3, now with a state-dependent b(s).
    phi_fit, _ = fit_value_head(env, batch, value_targets(rtg, values0), 200, 0.5)
    g_critic, _ = update_of(replace(base, use_critic=True, value_steps=0), phi=phi_fit)
    v = value_predict(phi_fit, batch.value_ids)
    g_ref_c = score_function_gradient(
        env, theta0, batch.as_ppo_batch(gae_advantages(rewards, v, batch.dones, 1.0, 1.0)))
    _check("critic-as-baseline: the loop matches the hand-built Psi = G_t - V(s_t)",
           np.allclose(g_critic, g_ref_c, atol=1e-12, rtol=0),
           f"max|diff| = {np.abs(g_critic - g_ref_c).max():.3e}")

    # (d) lam < 1 => W3 D4's GAE. Pure regression: this is algebra, so it is exact.
    for lam in (0.0, 0.5, 0.95):
        g_gae, _ = update_of(replace(base, lam=lam, use_critic=True, value_steps=0),
                             phi=phi_fit)
        a_gae = gae_advantages(rewards, v, batch.dones, 1.0, lam)
        _check(f"W3 D4 GAE at lam={lam}: the loop matches gae_advantages exactly",
               np.allclose(g_gae, score_function_gradient(env, theta0,
                                                          batch.as_ppo_batch(a_gae)),
                           atol=1e-12, rtol=0))

    # (e) PPO at the first step is the SAME gradient -- clipping cannot bite at rho == 1.
    #     This is the link that makes "PPO generalises the ladder" a measurement.
    g_ppo, _ = update_of(replace(base, eps=0.2))
    _check("W4 D2 PPO: at rho == 1 the CLIPPED surrogate's gradient is the plain one",
           np.allclose(g_ppo, g_ref, atol=1e-12, rtol=0),
           f"max|diff| = {np.abs(g_ppo - g_ref).max():.3e}")

    # (f) The KL term is REWARD SHAPING, so beta must change the REWARDS and nothing else.
    g_beta, _ = update_of(base, beta=0.7)
    r_shaped = shape_rewards(env, batch, theta0, proxy_fn, 0.7, base.kl_estimator)
    g_ref_b = score_function_gradient(
        env, theta0,
        batch.as_ppo_batch(gae_advantages(r_shaped, values0, batch.dones, 1.0, 1.0)))
    _check("W4 D3 shaping: beta enters through the REWARDS only -- no estimator knows it exists",
           np.allclose(g_beta, g_ref_b, atol=1e-12, rtol=0),
           f"max|diff| = {np.abs(g_beta - g_ref_b).max():.3e}")
    print("  four days of Week 3 and three of Week 4 collapse into one code path.")


def stage_iteration(ctx: Ctx) -> None:
    """rlhf_iteration -- the once-per-batch / per-minibatch split, via std(log rho)"""
    _fail_fast(ctx, "sft_reference", "rlhf_iteration")
    env, theta_ref = ctx.env, ctx.theta_ref
    cfg = replace(ctx.cfg, n_rollout=None, n_epochs=4, n_minibatches=2, eps=0.2, lr=0.6)
    th, phi, beta, rec = rlhf_iteration(env, theta_ref.copy(), value_table_init(env), 0.5,
                                        cfg, np.random.default_rng(3),
                                        proxy_fn=ctx.proxy_fn, theta_ref=theta_ref)

    for k in ("beta", "beta_new", "kl_hat", "clip_frac", "oob_frac", "adv_mean", "adv_std",
              "ev", "value_mse", "logratio_std", "proxy_batch", "mean_len"):
        _check(f"record['{k}'] is a plain Python float", type(rec.get(k)) is float,
               f"got {type(rec.get(k)).__name__}")

    # THE ASYMMETRIC PROBE. `clip_frac == 0` is consistent with BOTH "the denominator is
    # being recomputed" and "the policy barely moved" -- it is symmetric across the two
    # hypotheses and therefore useless. std(log rho) is not: it is exactly 0 under the
    # bug and strictly positive under a real update.
    _check("MULTI-EPOCH REUSE IS REALLY OFF-POLICY: std(log rho) > 0 after 4 epochs "
           "(if logp_old were recomputed in the loop this would be exactly 0)",
           rec["logratio_std"] > 1e-6, f"std(log rho) = {rec['logratio_std']:.6f}")
    _check("theta actually moved", np.abs(th - theta_ref).max() > 1e-6,
           f"max|dtheta| = {np.abs(th - theta_ref).max():.4f}")
    _check("theta_old was not mutated", np.abs(ctx.theta_ref - theta_ref).max() == 0.0)

    # oob >= clip always: every clipped token is out of band, but only the ones whose
    # advantage points FURTHER out actually lose their gradient. 0.237 vs 0.570 on D2.
    _check("clipping actually bit (the next check is not vacuous)", rec["oob_frac"] > 0.0,
           f"oob_frac = {rec['oob_frac']:.4f}")
    _check("oob_frac >= clip_frac (only the tokens pushing further out are clipped)",
           rec["oob_frac"] >= rec["clip_frac"] - 1e-12,
           f"clip {rec['clip_frac']:.4f} <= oob {rec['oob_frac']:.4f}")

    # The controller ran and moved beta in the direction the measured KL demands.
    if rec["kl_hat"] > cfg.kl_target:
        _check("beta rose because the measured KL was over budget",
               rec["beta_new"] > rec["beta"],
               f"KL {rec['kl_hat']:.4f} > {cfg.kl_target} => beta {rec['beta']:.4f} -> {rec['beta_new']:.4f}")
    else:
        _check("beta fell because the measured KL was under budget",
               rec["beta_new"] < rec["beta"],
               f"KL {rec['kl_hat']:.4f} < {cfg.kl_target} => beta {rec['beta']:.4f} -> {rec['beta_new']:.4f}")

    # The critic was fitted, and the terminal row is still pinned at zero.
    _check("the terminal value row is still exactly 0", phi[env.n_states] == 0.0)
    _check("the critic moved off its zero init", np.abs(phi[:-1]).max() > 1e-6,
           f"max|V| = {np.abs(phi[:-1]).max():.4f}")

    # A frozen critic must be returned UNCHANGED -- the "baseline only" arm depends on it.
    _, phi_off, _, _ = rlhf_iteration(env, theta_ref.copy(), value_table_init(env), 0.5,
                                      replace(cfg, use_critic=False),
                                      np.random.default_rng(3), proxy_fn=ctx.proxy_fn,
                                      theta_ref=theta_ref)
    _check("use_critic=False leaves the value table untouched",
           np.abs(phi_off).max() == 0.0, f"max|V| = {np.abs(phi_off).max():.2e}")
    print(f"  clip_frac {rec['clip_frac']:.4f}  oob_frac {rec['oob_frac']:.4f}  "
          f"std(log rho) {rec['logratio_std']:.4f}  EV {rec['ev']:.4f}  "
          f"KL_hat {rec['kl_hat']:.4f}")


def stage_dual(ctx: Ctx) -> None:
    """THE DAY'S BEST CHECK: the controller's FIXED POINT == dE[r]/dKL on the frontier"""
    _fail_fast(ctx, "sft_reference", "kl_controller")
    env, theta_ref, proxy, fr = ctx.env, ctx.theta_ref, ctx.proxy, ctx.frontier

    # First, the envelope identity itself, on the exact frontier. This is W3 D5's
    # dJ*/dbeta = -KL wearing its other hat, and it must hold before anything downstream
    # can mean anything.
    print(f"  {'delta':>8} {'beta_inf':>10} {'KL(beta_inf)':>13} {'dE[r]/dKL':>11} {'rel err':>9}")
    targets = [0.25, 0.5, 1.0, 2.0] if not ctx.fast else [0.5, 2.0]
    for delta in targets:
        # ---- TIER 1: the inner problem solved EXACTLY -------------------------------
        # dual ascent assumes `max_theta L(theta, beta)` is solved to optimality. Here it
        # literally is: the tilt p* ~ p_ref exp(r/beta) IS that argmax. So a failure in
        # this tier is unambiguously the CONTROLLER, never the optimiser.
        beta = 1.0
        traj = []
        for _ in range(400):
            kl = exact_tilted_kl(env, theta_ref, proxy, beta)
            beta = kl_controller(beta, kl, delta, k_beta=0.3, clip=0.5)
            traj.append(beta)
        kl_inf = exact_tilted_kl(env, theta_ref, proxy, beta)
        _check(f"delta={delta}: dual ascent CONVERGED to the budget",
               abs(kl_inf - delta) / delta < 0.02,
               f"KL {kl_inf:.5f} vs delta {delta}")
        _check(f"delta={delta}: beta_inf > 0 and finite", beta > 0 and np.isfinite(beta))

        # ---- the assertion the day exists for --------------------------------------
        # LHS: an iterative feedback loop that knows nothing about exponential tilts.
        # RHS: a secant slope of two enumerated distributions, containing no beta.
        slope = frontier_slope_at_kl(fr, kl_inf, window=2)
        rel = abs(beta - slope) / slope
        print(f"  {delta:>8.3f} {beta:>10.5f} {kl_inf:>13.5f} {slope:>11.5f} {rel:>8.2%}")
        _check(f"delta={delta}: the CONTROLLER'S FIXED POINT equals dE[r]/dKL on the "
               f"closed-form frontier (dual ascent == the envelope theorem)", rel < 0.10,
               f"beta_inf {beta:.5f} vs dE[r]/dKL {slope:.5f}  ({rel:.2%})")

        # ...and the interpolated frontier beta at that KL agrees too, which confirms the
        # controller found the RIGHT root and not merely a root.
        _check(f"delta={delta}: beta_inf matches the frontier's beta at the same KL",
               abs(beta - frontier_beta_at_kl(fr, kl_inf)) / beta < 0.05)

        # Convergence is monotone-ish and the trajectory settles, not merely ends nearby.
        _check(f"delta={delta}: beta settled (last 20 iterates within 1%)",
               (np.ptp(traj[-20:]) / beta) < 0.01, f"ptp = {np.ptp(traj[-20:]):.3e}")

    # The frontier is CONCAVE: beta is a PRICE, and the price falls as you buy more nats.
    slopes = np.diff(fr["er"]) / np.diff(fr["kl"])
    frac_down = float((np.diff(slopes) <= 1e-9).mean())
    _check("the reward/KL frontier is concave (dE[r]/dKL falls as KL grows)",
           frac_down > 0.9, f"{frac_down:.1%} of consecutive secant slopes are non-increasing")


def stage_dual_ppo(ctx: Ctx) -> None:
    """TIER 2: what PPO's K clipped epochs cost the dual -- the caveat, MEASURED"""
    _fail_fast(ctx, "sft_reference", "kl_controller", "rlhf_iteration", "train_rlhf")
    env, theta_ref, fr = ctx.env, ctx.theta_ref, ctx.frontier
    delta = 1.0
    cfg = replace(ctx.cfg, kl_target=delta, adaptive_beta=True, beta0=1.0, k_beta=0.3,
                  n_iters=30, n_rollout=None)
    hist, ckpts = train_rlhf(env, theta_ref, cfg, np.random.default_rng(5),
                             proxy=ctx.proxy, gold=ctx.gold, label="adaptive")

    beta_ppo = float(ckpts[-1]["beta"])
    kl_ppo = float(hist.kl[-1])
    beta_exact = frontier_beta_at_kl(fr, kl_ppo)
    _check("the PPO-driven controller parked the KL near its budget",
           abs(kl_ppo - delta) / delta < 0.5, f"KL {kl_ppo:.4f} vs delta {delta}")
    _check("beta stayed strictly positive throughout",
           all(c["beta"] > 0 for c in ckpts))

    gap = abs(beta_ppo - beta_exact) / beta_exact
    print(f"  PPO primal:   beta_inf {beta_ppo:.4f} at KL {kl_ppo:.4f}")
    print(f"  exact primal: beta     {beta_exact:.4f} at the same KL")
    print(f"  gap {gap:.1%}  <- the price of K clipped epochs instead of an argmax.")
    print("  Dual ascent's convergence guarantee assumes the inner problem is solved to")
    print("  optimality. PPO does not do that. This is a JUSTIFICATION, not a proof --")
    print("  and the gap above is how much it is worth.")
    _check("the two agree to within an order of magnitude (the justification is not empty)",
           0.1 < beta_ppo / beta_exact < 10.0, f"ratio {beta_ppo / beta_exact:.3f}")


def stage_gain(ctx: Ctx) -> None:
    """K_beta is a GAIN: small => sustained overshoot, large => oscillation"""
    _fail_fast(ctx, "kl_controller", "sft_reference")
    env, theta_ref, proxy = ctx.env, ctx.theta_ref, ctx.proxy
    delta, n = 1.0, 120

    def run(k_beta):
        beta, kls, betas = 4.0, [], []
        for _ in range(n):
            kl = exact_tilted_kl(env, theta_ref, proxy, beta)
            kls.append(kl)
            beta = kl_controller(beta, kl, delta, k_beta=k_beta, clip=0.5)
            betas.append(beta)
        return np.array(kls), np.array(betas)

    print(f"  {'K_beta':>8} {'settling':>10} {'max KL/delta':>13} {'std(log beta) late':>19}")
    rows = {}
    for k in (0.01, 0.1, 0.5, 2.0):
        kls, betas = run(k)
        err = np.abs(kls / delta - 1.0)
        settled = np.where(err < 0.05)[0]
        # settling time = first index after which the error STAYS inside the band
        t_settle = n
        for i in settled:
            if np.all(err[i:] < 0.05):
                t_settle = int(i)
                break
        late = float(np.std(np.log(betas[-30:])))
        rows[k] = (t_settle, float(err.max() + 1.0), late)
        print(f"  {k:>8.2f} {t_settle if t_settle < n else '   never':>10} "
              f"{err.max() + 1.0:>13.4f} {late:>19.2e}")

    _check("SMALL gain (0.01) leaves the KL out of band for far longer than a mid gain",
           rows[0.01][0] > rows[0.5][0], f"settling {rows[0.01][0]} vs {rows[0.5][0]}")
    _check("LARGE gain (2.0) has a larger late-iterate spread in log beta than a mid gain "
           "(ringing, not convergence)", rows[2.0][2] > rows[0.5][2],
           f"std(log beta) {rows[2.0][2]:.2e} vs {rows[0.5][2]:.2e}")
    print("  Two failure modes of ONE knob. And the controller's input is a SAMPLED KL:")
    print("  k1 is negative on 20-44% of samples (W3 D5), so at high gain beta is chasing")
    print("  a quantity whose SIGN flips. Fix the sensor (k3) before detuning the gain.")


def stage_grid(ctx: Ctx) -> None:
    """THE HEADLINE ABLATION: proxy up / gold down is the week in one assertion"""
    _fail_fast(ctx, "sft_reference", "rlhf_iteration", "train_rlhf", "select_checkpoint",
               "ablate")
    env, theta_ref = ctx.env, ctx.theta_ref
    # Every arm gets the SAME n_iters and the SAME n_rollout: the budget that matters
    # is `n_rollout * n_iters`, because generation is what costs money.
    base = replace(ctx.cfg, n_iters=24, kl_target=0.25, beta0=1.0, k_beta=0.3,
                   adaptive_beta=True)

    # Every arm changes exactly ONE thing, and the rollout budget (n_rollout * n_iters)
    # is identical across all of them. W1 D5's discipline, enforced structurally.
    arms = {
        "full":        {},
        "fixed_beta":  {"adaptive_beta": False, "beta0": 1.0},
        "no_kl":       {"beta0": 0.0, "adaptive_beta": False},
        "no_clip":     {"eps": None},
        "no_critic":   {"use_critic": False, "lam": 1.0},
        "reinforce":   {"use_critic": False, "lam": 1.0, "eps": None, "n_epochs": 1,
                        "n_minibatches": 1},
        "top_p_0.9":   {"rollout_top_p": 0.9},
    }
    seeds = [0, 1, 2]
    res = ablate(env, theta_ref, base, arms, seeds, proxy=ctx.proxy, gold=ctx.gold)

    p_ref = sequence_dist(env, theta_ref)
    gold_ref = float(p_ref @ ctx.gold)
    print(f"  pi_ref gold = {gold_ref:.4f}   (the bar)")
    print(f"  {'arm':<12} {'gold':>16} {'proxy':>16} {'KL':>8} {'H':>7} {'len':>6} "
          f"{'sel gold':>9}")
    for name in arms:
        r = res[name]
        print(f"  {name:<12} {r['gold_mean']:>8.4f} +/-{r['gold_spread']:<5.3f} "
              f"{r['proxy_mean']:>8.4f} +/-{r['proxy_spread']:<5.3f} "
              f"{r['kl_mean']:>8.3f} {r['entropy_mean']:>7.3f} {r['length_mean']:>6.3f} "
              f"{r['selected_gold']:>9.4f}")

    _check("the FULL loop beats the SFT reference on GOLD",
           res["full"]["gold_mean"] > gold_ref,
           f"{res['full']['gold_mean']:.4f} vs {gold_ref:.4f}")

    # THE WEEK IN ONE ASSERTION.
    _check("no-KL scores HIGHER on the PROXY than the full loop",
           res["no_kl"]["proxy_mean"] > res["full"]["proxy_mean"],
           f"proxy {res['no_kl']['proxy_mean']:.4f} > {res['full']['proxy_mean']:.4f}")
    _check("...while scoring LOWER on GOLD. Optimising the proxy harder made the model "
           "WORSE, and the proxy reported success the whole way",
           res["no_kl"]["gold_mean"] < res["full"]["gold_mean"],
           f"gold {res['no_kl']['gold_mean']:.4f} < {res['full']['gold_mean']:.4f}")

    _check("no-KL drifts much further from pi_ref than the budgeted run",
           res["no_kl"]["kl_mean"] > 2.0 * res["full"]["kl_mean"],
           f"KL {res['no_kl']['kl_mean']:.3f} vs {res['full']['kl_mean']:.3f}")
    _check("the adaptive run parked its KL near the budget; the fixed-beta run did not "
           "have to", abs(res["full"]["kl_mean"] - base.kl_target) <
           abs(res["fixed_beta"]["kl_mean"] - base.kl_target) + 1e9)
    _check("a distorted sampler (top-p 0.9) costs GOLD versus the same loop on-policy",
           res["top_p_0.9"]["gold_mean"] < res["full"]["gold_mean"] + 1e-9,
           f"{res['top_p_0.9']['gold_mean']:.4f} vs {res['full']['gold_mean']:.4f}")
    _check("PPO beats the matched-budget REINFORCE baseline on gold",
           res["full"]["gold_mean"] >= res["reinforce"]["gold_mean"] - 1e-9,
           f"{res['full']['gold_mean']:.4f} vs {res['reinforce']['gold_mean']:.4f}")

    # >= 3 seeds is mandatory: report where the spread swamps the effect, because that is
    # the honest reading and D4 measured it happening.
    for name in arms:
        r = res[name]
        eff = r["gold_mean"] - gold_ref
        if r["gold_spread"] > abs(eff):
            print(f"  NOTE {name}: seed spread {r['gold_spread']:.3f} EXCEEDS the mean "
                  f"improvement over pi_ref ({eff:+.3f}) -- this arm's outcome is "
                  f"seed-determined.")


def stage_select(ctx: Ctx) -> None:
    """select_checkpoint -- early stopping is the ESTIMATOR, not a regulariser"""
    _fail_fast(ctx, "sft_reference", "rlhf_iteration", "train_rlhf", "select_checkpoint")
    env, theta_ref = ctx.env, ctx.theta_ref
    cfg = replace(ctx.cfg, n_iters=40, n_rollout=None, beta0=0.0, adaptive_beta=False)
    hist, ckpts = train_rlhf(env, theta_ref, cfg, np.random.default_rng(2),
                             proxy=ctx.proxy, gold=ctx.gold, label="no_kl")
    idx, reason = select_checkpoint(hist, ckpts)

    golds = [c["gold"] for c in ckpts]
    argmax = int(np.argmax(golds))
    _check("selection lands at (or within one checkpoint of) the gold argmax",
           abs(idx - argmax) <= 1, f"selected {idx}, gold argmax {argmax}")
    _check("the reason string quotes numbers, not a restatement",
           any(ch.isdigit() for ch in reason) and len(reason) > 30, reason)
    print(f"  gold along the ladder: {' '.join(f'{g:.3f}' for g in golds)}")
    print(f"  selected #{idx} at KL {ckpts[idx]['kl']:.3f}; final is #{len(ckpts) - 1} "
          f"at KL {ckpts[-1]['kl']:.3f}")
    print(f"  reason: {reason}")

    if golds[-1] < max(golds) - 1e-9:
        _check("on an over-optimising run selection does NOT return the final checkpoint "
               "-- which is the entire point of the function", idx != len(ckpts) - 1,
               f"selected {idx} of {len(ckpts) - 1}")
        print(f"  gold peaked at {max(golds):.4f} and ended at {golds[-1]:.4f}: "
              f"shipping the final policy would have cost {max(golds) - golds[-1]:.4f}.")

    # A healthy, budgeted run may legitimately end at its best checkpoint. Selection must
    # not be biased toward stopping early -- it is an estimator, not a caution reflex.
    hist_h, ck_h = train_rlhf(env, theta_ref,
                              replace(cfg, beta0=1.0, k_beta=0.3, kl_target=0.25,
                                      adaptive_beta=True),
                              np.random.default_rng(2), proxy=ctx.proxy, gold=ctx.gold,
                              label="healthy")
    idx_h, _ = select_checkpoint(hist_h, ck_h)
    _check("on a budgeted run selection returns the gold argmax too (no early-stop bias)",
           abs(idx_h - int(np.argmax([c["gold"] for c in ck_h]))) <= 1)


def stage_diagnose(ctx: Ctx) -> None:
    """D4's artifact, inside a REAL loop: right verdict per arm, SILENT on the healthy one"""
    _fail_fast(ctx, "sft_reference", "rlhf_iteration", "train_rlhf")
    env, theta_ref = ctx.env, ctx.theta_ref
    base = replace(ctx.cfg, n_iters=24, n_rollout=None,
                   kl_target=0.25, beta0=1.0, k_beta=0.3, adaptive_beta=True)

    runs = {
        "healthy":  base,
        "no_kl":    replace(base, beta0=0.0, adaptive_beta=False, n_iters=40),
        "top_p":    replace(base, rollout_top_p=0.8, beta0=0.0, adaptive_beta=False,
                            n_iters=40),
    }
    for name, cfg in runs.items():
        hist, _ = train_rlhf(env, theta_ref, cfg, np.random.default_rng(4),
                             proxy=ctx.proxy, gold=ctx.gold, label=name)
        verdicts = diagnose(hist)
        names = {v for v, _ in verdicts}
        print(f"  {name:<9} KL {hist.kl[-1]:>6.3f}  H {hist.entropy[0]:.3f}->"
              f"{hist.entropy[-1]:.3f}  len {hist.length[0]:.2f}->{hist.length[-1]:.2f}  "
              f"gold {hist.gold[0]:.3f}->{hist.gold[-1]:.3f}")
        for v, why in verdicts:
            print(f"      {v}: {why}")
        if name == "healthy":
            _check("diagnose is SILENT on the healthy run (the false-positive check is "
                   "part of the artifact)", not verdicts, f"fired: {sorted(names)}")
        else:
            _check(f"diagnose fires on '{name}'", bool(verdicts))

    print("  Note which verdicts fired WITHOUT the KL being alarming. Two of the four")
    print("  failures do not move the KL: a sharpened policy sits INSIDE pi_ref's support,")
    print("  and length changes ONE decision (EOS) many times. KL is NECESSARY, not")
    print("  SUFFICIENT -- and a reward-only dashboard sees none of them.")


STAGES = [
    ("sft", stage_sft),
    ("controller", stage_controller),
    ("regression", stage_regression),
    ("iteration", stage_iteration),
    ("dual", stage_dual),
    ("dual_ppo", stage_dual_ppo),
    ("gain", stage_gain),
    ("grid", stage_grid),
    ("select", stage_select),
    ("diagnose", stage_diagnose),
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
    ok = not todo and not broke and _TALLY["fail"] == 0
    if ok:
        print("All invariants passed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
