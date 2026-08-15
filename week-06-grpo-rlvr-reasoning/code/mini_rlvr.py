#!/usr/bin/env python3
"""W6 D5 -- a small, exact RLVR/GRPO system.

The boundary to watch is: ``gold`` is evaluation-only; rollout reward is always
``accept``.  Curriculum calibration may use a periodic held-out verifier audit,
but neither that audit nor any gold event enters the policy loss.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Callable

import numpy as np
import torch

from curriculum import BandTracker, UniformSampler, make_curriculum_task, set_point_analytic
from grpo import (
    SeqTable,
    TokenEnv,
    build_seq_table,
    exact_entropy,
    exact_kl,
    group_advantage,
    kl_penalty,
    seq_logprobs_np,
    seq_token_logprobs_torch,
)
from rlvr import dead_fraction_levels, p_observed, verifier_rates
from sampling import bon_distribution, pass_at_k_exact, selector_curves

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class MiniTask:
    env: TokenEnv
    tab: SeqTable
    gold: np.ndarray
    accept: np.ndarray
    answer_id: np.ndarray
    theta0: np.ndarray
    flavour: str = "honest"

    @property
    def n_prompts(self) -> int:
        return int(self.gold.shape[0])


@dataclass(frozen=True)
class MiniCfg:
    steps: int = 72
    group: int = 8
    lr: float = 0.35
    beta: float = 0.04
    k_eval: int = 8
    n_active: int = 4
    seed: int = 0
    adv_mode: str = "centered"
    sensor_mode: str = "calibrated"
    sampler_mode: str = "band"
    selector_mode: str = "filter"
    calibration_interval: int = 8
    sensor_ema: float = 0.55
    band_width: float = 1.5


@dataclass(frozen=True)
class CalibratedRate:
    raw: np.ndarray
    scheduled: np.ndarray
    projected: np.ndarray


@dataclass(frozen=True)
class MiniEvaluation:
    p_obs: np.ndarray
    p_gold: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray
    verifier_passk: float
    gold_passk: float
    verifier_rates: np.ndarray  # (P,6): p_true,p_obs,alpha,gamma,precision,recall
    selector_name: str
    deployed_accuracy: float
    filter_accuracy: float
    ranker_accuracy: float
    vote_accuracy: float
    precision: float
    kl: float
    entropy: float


@dataclass
class MiniHistory:
    step: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    p_obs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    p_gold: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    p_hat_gold: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    active: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    verifier_passk: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gold_passk: np.ndarray = field(default_factory=lambda: np.zeros(0))
    deployed_accuracy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    precision: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dead_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    signal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    entropy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    in_band: np.ndarray = field(default_factory=lambda: np.zeros(0))
    projection_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    selector_name: str = ""
    theta_final: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))


@dataclass(frozen=True)
class DiagThresholds:
    dead_budget: float = 0.72
    sensor_mismatch: float = 0.14
    reward_hacking: float = 0.04
    selector_gap: float = 0.10
    kl_runaway: float = 0.40


def _validate_task(task: MiniTask) -> None:
    p, y = task.gold.shape
    if task.accept.shape != (p, y) or task.answer_id.shape != (p, y):
        raise ValueError("gold, accept, and answer_id shapes must match")
    if task.theta0.shape != (p, task.env.n_states, task.env.vocab):
        raise ValueError("theta0 has the wrong shape")
    if y != task.tab.n_seqs:
        raise ValueError("mask width must equal the enumerable support")
    if not np.all(np.isin(task.gold, (0.0, 1.0))):
        raise ValueError("gold must be binary")
    if not np.all(np.isin(task.accept, (0.0, 1.0))):
        raise ValueError("accept must be binary")
    if not np.issubdtype(task.answer_id.dtype, np.integer) or np.any(task.answer_id < 0):
        raise ValueError("answer_id must contain non-negative integers")
    if not np.all(np.isfinite(task.theta0)):
        raise ValueError("theta0 must be finite")


def _is_positive_int(value: object) -> bool:
    return (isinstance(value, (int, np.integer)) and
            not isinstance(value, (bool, np.bool_)) and value >= 1)


def _validate_cfg(cfg: MiniCfg, prompts: int) -> None:
    counts = (
        cfg.steps,
        cfg.group,
        cfg.n_active,
        cfg.k_eval,
        cfg.calibration_interval,
    )
    if not all(_is_positive_int(value) for value in counts):
        raise ValueError("count and interval fields must be positive integers")
    numeric = (cfg.lr, cfg.beta, cfg.sensor_ema, cfg.band_width)
    if (not np.all(np.isfinite(numeric)) or cfg.lr <= 0 or cfg.beta < 0 or
            not 0 <= cfg.sensor_ema < 1 or cfg.band_width <= 0):
        raise ValueError("invalid optimizer/sensor configuration")
    if cfg.adv_mode not in {"centered", "normalized", "whitened"}:
        raise ValueError("adv_mode must be centered or whitened")
    if cfg.sensor_mode not in {"raw", "calibrated"}:
        raise ValueError("sensor_mode must be raw or calibrated")
    if cfg.sampler_mode not in {"uniform", "band"}:
        raise ValueError("sampler_mode must be uniform or band")
    if cfg.selector_mode not in {"filter", "ranker", "vote"}:
        raise ValueError("selector_mode must be filter, ranker, or vote")
    if cfg.n_active > prompts:
        raise ValueError("n_active cannot exceed the prompt count")


@lru_cache(maxsize=4)
def _base_fixture(n_prompts: int = 8) -> MiniTask:
    env = TokenEnv(vocab=3, t_max=3)
    tab = build_seq_table(env)
    base = make_curriculum_task(
        env, tab, n_prompts, np.random.default_rng(6105), p_lo=0.04, p_hi=0.72
    )
    # Correctness is one answer; incorrect sequences form two other answers.  This
    # makes voting well-defined without changing curriculum.py's calibrated masks.
    answer = np.empty_like(base.gold, dtype=np.int64)
    for p in range(n_prompts):
        answer[p] = np.where(base.gold[p] == 1, 0, 1 + (np.arange(tab.n_seqs) % 2))
    return MiniTask(env, tab, base.gold.copy(), base.gold.copy(), answer,
                    base.theta0.copy(), "honest")


def make_mini_task(flavour: str = "honest", n_prompts: int = 8) -> MiniTask:
    """Construct honest, starved (V subset C), or leaky (V\\C non-empty) fixtures."""
    base = _base_fixture(n_prompts)
    gold, accept = base.gold.copy(), base.gold.copy()
    if flavour == "starved":
        for p in range(n_prompts):
            live = np.flatnonzero(gold[p])
            accept[p, live[1::2]] = 0.0
            if accept[p].sum() == 0:
                accept[p, live[0]] = 1.0
    elif flavour == "leaky":
        # Reward every sequence expressing wrong answer 1.  It is a coherent,
        # exploitable verifier bug, not per-rollout label noise.
        accept = np.maximum(gold, (base.answer_id == 1).astype(np.float64))
    elif flavour != "honest":
        raise ValueError(f"unknown fixture {flavour!r}")
    task = MiniTask(base.env, base.tab, gold, accept, base.answer_id.copy(),
                    base.theta0.copy(), flavour)
    _validate_task(task)
    return task


def calibrate_gold_rate(p_obs, alpha, gamma) -> CalibratedRate:
    """Invert ``p_obs = gamma + (1-alpha-gamma)p``. TODO 1.

    ``raw`` is the unmodified inverse.  ``scheduled`` is an explicitly projected
    controller coordinate and ``projected`` identifies every changed element.

    Contract:
      * accept scalar or matching non-scalar probability inputs;
      * reject non-finite values, values outside [0,1], and shape mismatches;
      * reject Youden J <= 0 rather than dividing by zero or inverting an
        anti-informative reward;
      * retain the raw inverse even when it falls outside [0,1];
      * project only ``scheduled`` into [0,1], recording every projection.

    Do not use ``np.clip`` without first preserving the raw estimate and building
    the projection mask. A silently clipped sensor is an unobservable failure.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in (p_obs, alpha, gamma)]
    nonscalar_shapes = [x.shape for x in arrays if x.ndim > 0]
    if nonscalar_shapes and any(
        shape != nonscalar_shapes[0] for shape in nonscalar_shapes
    ):
        raise ValueError("non-scalar probability inputs must have matching shapes")

    p_obs_arr, alpha_arr, gamma_arr = np.broadcast_arrays(*arrays)
    if any(
        np.any(~np.isfinite(x)) or np.any((x < 0.0) | (x > 1.0))
        for x in (p_obs_arr, alpha_arr, gamma_arr)
    ):
        raise ValueError("p_obs, alpha, and gamma must be finite probabilities")

    youden_j = 1.0 - alpha_arr - gamma_arr
    if np.any(youden_j <= 0.0):
        raise ValueError("Youden J = 1-alpha-gamma must be > 0")

    raw = (p_obs_arr - gamma_arr) / youden_j
    projected = (raw < 0.0) | (raw > 1.0)
    scheduled = np.where(raw < 0.0, 0.0, np.where(raw > 1.0, 1.0, raw))
    return CalibratedRate(raw.copy(), scheduled, projected)


def _policy_probs(theta: np.ndarray, tab: SeqTable) -> np.ndarray:
    probs = np.stack([np.exp(seq_logprobs_np(t, tab)) for t in theta])
    if not np.allclose(probs.sum(1), 1.0, atol=2e-12):
        raise RuntimeError("enumerated policy does not normalize")
    return probs


def evaluate_policy(
    task: MiniTask,
    theta: np.ndarray,
    k: int = 8,
    selector: str = "filter",
    theta_ref: np.ndarray | None = None,
) -> MiniEvaluation:
    """Evaluate each prompt, then average. TODO 2.

    filter: uniformly choose among accepted draws, falling back to the first draw;
    ranker: choose the max of ``2*accept + fixed sequence tie-break``;
    vote: plurality over answer_id with uniform tie-breaking.  None is an oracle.

    Reuse ``verifier_rates``, ``selector_curves``, ``bon_distribution``,
    ``pass_at_k_exact``, ``exact_kl`` and ``exact_entropy``. The contract is:
      * validate the task, theta shape, k, selector, and reference shape;
      * compute per-prompt ``p_obs`` and ``p_gold`` from the enumerable policy;
      * compute pass@k per prompt and only then average;
      * keep filter, ranker and vote results separate and dispatch by name;
      * compute policy-mass-weighted precision, never a sequence-count rate;
      * never let gold influence a selector score.

    The fixed ranker score is ``2*accept + arange(n)/(2*n)``. The tiny sequence
    feature breaks ties without reading gold.
    """
    _validate_task(task)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != task.theta0.shape or not _is_positive_int(k):
        raise ValueError("bad theta shape or k")
    if selector not in {"filter", "ranker", "vote"}:
        raise ValueError(f"unknown selector {selector!r}")

    ref = task.theta0 if theta_ref is None else np.asarray(theta_ref, dtype=np.float64)
    if ref.shape != theta.shape:
        raise ValueError("theta_ref must match theta")
    probs = _policy_probs(theta, task.tab)

    p_obs, p_gold, alpha, gamma, rate_rows = [], [], [], [], []
    filter_accuracy, ranker_accuracy, vote_accuracy = [], [], []
    for prompt in range(task.n_prompts):
        rates = verifier_rates(
            task.gold[prompt], task.accept[prompt], probs[prompt]
        )
        p_obs.append(rates.p_obs)
        p_gold.append(rates.p_true)
        alpha.append(rates.alpha)
        gamma.append(rates.gamma)
        rate_rows.append(
            (
                rates.p_true,
                rates.p_obs,
                rates.alpha,
                rates.gamma,
                rates.precision,
                rates.recall,
            )
        )

        curves = selector_curves(
            probs[prompt],
            task.gold[prompt],
            task.accept[prompt],
            task.answer_id[prompt],
            [k],
        )
        filter_accuracy.append(float(curves.bon[0]))
        vote_accuracy.append(float(curves.maj[0]))

        score = (
            2.0 * task.accept[prompt]
            + np.arange(task.tab.n_seqs) / (2.0 * task.tab.n_seqs)
        )
        ranker_dist, _ = bon_distribution(probs[prompt], score, k)
        ranker_accuracy.append(float(ranker_dist @ task.gold[prompt]))

    p_obs_arr = np.asarray(p_obs)
    p_gold_arr = np.asarray(p_gold)
    filter_arr = np.asarray(filter_accuracy)
    ranker_arr = np.asarray(ranker_accuracy)
    vote_arr = np.asarray(vote_accuracy)
    deployed = {
        "filter": filter_arr,
        "ranker": ranker_arr,
        "vote": vote_arr,
    }[selector]

    accepted_mass = float(np.sum(probs * task.accept))
    correct_accepted_mass = float(np.sum(probs * task.accept * task.gold))
    precision = (
        correct_accepted_mass / accepted_mass if accepted_mass > 0.0 else 1.0
    )

    return MiniEvaluation(
        p_obs=p_obs_arr,
        p_gold=p_gold_arr,
        alpha=np.asarray(alpha),
        gamma=np.asarray(gamma),
        verifier_passk=float(np.mean(pass_at_k_exact(p_obs_arr, k))),
        gold_passk=float(np.mean(pass_at_k_exact(p_gold_arr, k))),
        verifier_rates=np.asarray(rate_rows),
        selector_name=selector,
        deployed_accuracy=float(deployed.mean()),
        filter_accuracy=float(filter_arr.mean()),
        ranker_accuracy=float(ranker_arr.mean()),
        vote_accuracy=float(vote_arr.mean()),
        precision=precision,
        kl=float(
            np.mean(
                [
                    exact_kl(theta[prompt], ref[prompt], task.tab)
                    for prompt in range(task.n_prompts)
                ]
            )
        ),
        entropy=float(
            np.mean(
                [
                    exact_entropy(theta[prompt], task.tab)
                    for prompt in range(task.n_prompts)
                ]
            )
        ),
    )


def _inverse_cdf(probs: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    return np.minimum(np.searchsorted(np.cumsum(probs), uniforms, side="right"),
                      len(probs) - 1)


def _calibrated_sensor(
    observed: np.ndarray, alpha: np.ndarray, gamma: np.ndarray, mode: str
) -> CalibratedRate:
    if mode == "raw":
        z = np.zeros_like(observed, dtype=bool)
        return CalibratedRate(observed.copy(), observed.copy(), z)
    return calibrate_gold_rate(observed, alpha, gamma)


def train_mini_rlvr(task: MiniTask, cfg: MiniCfg) -> MiniHistory:
    """Run the composed RLVR/GRPO system. TODO 3.

    Required order:
      1. validate and build deterministic paired random streams;
      2. record the exact panel before each update;
      3. refresh held-out alpha/gamma only on the declared audit interval;
      4. choose active prompts from the recorded prior sensor;
      5. sample sequence IDs, reward with ``accept`` only, and compute detached
         group advantages;
      6. minimise score-function loss plus a separate fixed-reference k3 KL;
      7. update the free observed-rate sensor from completed rollouts.

    Use ``BandTracker`` or ``UniformSampler`` according to the config. Pre-generate
    one uniform for every (step,prompt,draw), including inactive prompts, so paired
    ablation arms see the same random event. ``gold`` may appear in the panel and
    periodic calibration audit, never in reward or policy loss.
    """
    _validate_task(task)
    _validate_cfg(cfg, task.n_prompts)

    tab, prompts = task.tab, task.n_prompts
    theta_ref = task.theta0.copy()
    theta = torch.tensor(theta_ref, requires_grad=True)
    optimizer = torch.optim.SGD([theta], lr=cfg.lr)

    # Separate streams keep prompt selection from changing any prompt's rollout
    # event. Every ablation arm receives the same (step, prompt, draw) uniforms.
    rollout_uniforms = np.random.default_rng(cfg.seed).random(
        (cfg.steps, prompts, cfg.group)
    )
    sampler_rng = np.random.default_rng(cfg.seed + 10_007)
    sampler = (
        BandTracker(
            cfg.n_active,
            set_point_analytic(cfg.k_eval),
            cfg.band_width,
        )
        if cfg.sampler_mode == "band"
        else UniformSampler(cfg.n_active)
    )

    reference_logp = seq_token_logprobs_torch(
        torch.as_tensor(theta_ref), tab
    )
    mask_all = torch.as_tensor(tab.mask)
    initial = evaluate_policy(
        task, theta_ref, cfg.k_eval, cfg.selector_mode, theta_ref
    )
    observed = initial.p_obs.copy()
    audit_alpha = initial.alpha.copy()
    audit_gamma = initial.gamma.copy()
    p_star = set_point_analytic(cfg.k_eval)
    z_star = np.log(p_star / (1.0 - p_star))

    rows = {name: [] for name in (
        "step", "verifier_passk", "gold_passk", "deployed_accuracy",
        "precision", "dead_frac", "signal", "kl", "entropy", "in_band",
        "projection_frac",
    )}
    p_obs_rows, p_gold_rows, p_hat_rows, active_rows = [], [], [], []

    for step in range(cfg.steps):
        theta_np = theta.detach().numpy()
        panel = evaluate_policy(
            task, theta_np, cfg.k_eval, cfg.selector_mode, theta_ref
        )
        if step % cfg.calibration_interval == 0:
            audit_alpha = panel.alpha.copy()
            audit_gamma = panel.gamma.copy()

        sensor = _calibrated_sensor(
            observed, audit_alpha, audit_gamma, cfg.sensor_mode
        )
        p_hat = sensor.scheduled

        # Record the policy and information state that produce this batch.
        rows["step"].append(step)
        p_obs_rows.append(panel.p_obs)
        p_gold_rows.append(panel.p_gold)
        p_hat_rows.append(p_hat.copy())
        rows["verifier_passk"].append(panel.verifier_passk)
        rows["gold_passk"].append(panel.gold_passk)
        rows["deployed_accuracy"].append(panel.deployed_accuracy)
        rows["precision"].append(panel.precision)
        rows["kl"].append(panel.kl)
        rows["entropy"].append(panel.entropy)
        rows["projection_frac"].append(float(sensor.projected.mean()))

        active = np.asarray(sampler(step, p_hat, sampler_rng), dtype=np.int64)
        active_rows.append(active)
        probs = _policy_probs(theta_np, tab)
        draws = np.stack([
            _inverse_cdf(probs[prompt], rollout_uniforms[step, prompt])
            for prompt in active
        ])
        rewards = np.take_along_axis(task.accept[active], draws, axis=1)
        advantage_mode = (
            "normalized" if cfg.adv_mode == "whitened" else cfg.adv_mode
        )
        advantages = group_advantage(rewards, advantage_mode)

        dead = np.all(advantages == 0.0, axis=1)
        rows["dead_frac"].append(float(dead.mean()))
        rows["signal"].append(float(np.mean(np.abs(advantages))))

        clipped_gold = np.clip(panel.p_gold[active], 1e-12, 1.0 - 1e-12)
        z_gold = np.log(clipped_gold / (1.0 - clipped_gold))
        rows["in_band"].append(
            float(np.mean(np.abs(z_gold - z_star) <= cfg.band_width))
        )

        active_t = torch.as_tensor(active)
        draws_t = torch.as_tensor(draws)
        logp = seq_token_logprobs_torch(theta, tab)[
            active_t[:, None], draws_t
        ]
        logp_ref = reference_logp[active_t[:, None], draws_t]
        mask = mask_all[draws_t]
        advantages_t = torch.as_tensor(advantages).detach()

        sequence_logp = (logp * mask).sum(dim=-1)
        policy_loss = -(sequence_logp * advantages_t).mean()
        if cfg.beta > 0.0:
            kl_tokens = kl_penalty(logp, logp_ref, "k3")
            kl_loss = (kl_tokens * mask).sum() / mask.sum()
        else:
            kl_loss = torch.zeros((), dtype=torch.float64)
        loss = policy_loss + cfg.beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rollout_rate = rewards.mean(axis=1)
        observed[active] = (
            cfg.sensor_ema * observed[active]
            + (1.0 - cfg.sensor_ema) * rollout_rate
        )

    scalar = {
        name: np.asarray(values, dtype=np.int64 if name == "step" else np.float64)
        for name, values in rows.items()
    }
    return MiniHistory(
        p_obs=np.asarray(p_obs_rows),
        p_gold=np.asarray(p_gold_rows),
        p_hat_gold=np.asarray(p_hat_rows),
        active=np.asarray(active_rows, dtype=np.int64),
        selector_name=cfg.selector_mode,
        theta_final=theta.detach().numpy().copy(),
        **scalar,
    )


def _tail(a: np.ndarray, window: int = 12) -> np.ndarray:
    return np.asarray(a)[-min(window, len(a)):]


def diagnostic_statistics(h: MiniHistory, window: int = 12) -> dict[str, float]:
    n = min(window, len(h.step))
    gold_gain = float(_tail(h.gold_passk, n).mean() - h.gold_passk[0])
    verifier_gain = float(_tail(h.verifier_passk, n).mean() - h.verifier_passk[0])
    return {
        "dead_budget": float(_tail(h.dead_frac, n).mean()),
        "sensor_mismatch": float(np.mean(np.abs(_tail(h.p_hat_gold, n) -
                                                     _tail(h.p_gold, n)))),
        "reward_hacking": verifier_gain - gold_gain,
        "selector_gap": float(_tail(h.gold_passk - h.deployed_accuracy, n).mean()),
        "kl_runaway": float(_tail(h.kl, n).mean()),
    }


def diagnose_mini_rlvr(
    h: MiniHistory, thresholds: DiagThresholds = DiagThresholds(), window: int = 12
) -> list[tuple[str, str]]:
    """Return mechanism reports and remain silent on healthy history. TODO 4.

    Read ``diagnostic_statistics`` and compare these exact names:
    ``dead_budget``, ``sensor_mismatch``, ``reward_hacking``, ``selector_gap`` and
    ``kl_runaway``. Return ``(name, detail)`` pairs only for strict threshold
    exceedances. Every detail must include the tail window, measured value, and
    threshold; an alert without its number is not actionable.

    Do not add another conjunct to rescue a noisy rule. The calibration harness
    requires healthy silence, every mechanism to fire, and pairwise-distinct fire
    sets. If distributions overlap, the rule must be cut rather than decorated.
    """
    statistics = diagnostic_statistics(h, window)
    tail_window = min(window, len(h.step))
    reports = []
    for name in DiagThresholds.__dataclass_fields__:
        measured = statistics[name]
        threshold = getattr(thresholds, name)
        if measured > threshold:
            reports.append(
                (
                    name,
                    f"tail window={tail_window}, measured={measured:.4f}, "
                    f"threshold={threshold:.4f}",
                )
            )
    return reports


def calibrate_diagnostic_thresholds(
    healthy: list[MiniHistory], sick: dict[str, list[MiniHistory]], window: int = 12
) -> DiagThresholds:
    """Place each threshold in measured air between healthy max and sick min."""
    values = {}
    for name in DiagThresholds.__dataclass_fields__:
        good = [diagnostic_statistics(h, window)[name] for h in healthy]
        bad = [diagnostic_statistics(h, window)[name] for h in sick[name]]
        if max(good) >= min(bad):
            raise ValueError(f"{name} has no calibration gap: healthy={good}, sick={bad}")
        values[name] = 0.5 * (max(good) + min(bad))
    return DiagThresholds(**values)


# ---------------------------------------------------------------------------
# Staged self-check harness
# ---------------------------------------------------------------------------

class Checks:
    def __init__(self) -> None:
        self.total = 0

    def check(self, label: str, condition, detail: str = "") -> None:
        self.total += 1
        if not bool(condition):
            raise AssertionError(f"{label}: {detail}")

    def raises(self, label: str, exc: type[BaseException], fn: Callable) -> None:
        self.total += 1
        try:
            fn()
        except exc:
            return
        raise AssertionError(f"{label}: did not raise {exc.__name__}")


def stage_contract(c: Checks, fast: bool) -> None:
    for flavour in ("honest", "starved", "leaky"):
        t = make_mini_task(flavour)
        _validate_task(t)
        c.check(f"{flavour} shapes", t.gold.shape == t.accept.shape == t.answer_id.shape)
    h, s, l = map(make_mini_task, ("honest", "starved", "leaky"))
    c.check("honest V=C", np.array_equal(h.accept, h.gold))
    c.check("starved V subset C", np.all(s.accept <= s.gold) and np.any(s.accept < s.gold))
    c.check("leaky V minus C", np.any(l.accept > l.gold))
    p0 = evaluate_policy(h, h.theta0).p_gold
    c.check("difficulty spread", p0.max() - p0.min() > 0.60, f"range={np.ptp(p0):.3f}")
    c.check("probability normalization",
            np.allclose(_policy_probs(h.theta0, h.tab).sum(1), 1, atol=2e-12))
    bad = replace(h, accept=h.accept[:, :-1])
    c.raises("shape fail-fast", ValueError, lambda: _validate_task(bad))
    invalid_calls = (
        lambda: train_mini_rlvr(h, MiniCfg(steps=1, group=0)),
        lambda: train_mini_rlvr(h, MiniCfg(steps=1.5)),
        lambda: train_mini_rlvr(h, MiniCfg(steps=1, group=np.inf)),
        lambda: train_mini_rlvr(h, MiniCfg(steps=1, n_active=1.5)),
        lambda: train_mini_rlvr(h, MiniCfg(steps=1, k_eval=1.5)),
        lambda: train_mini_rlvr(h, MiniCfg(steps=1, calibration_interval=np.nan)),
        lambda: evaluate_policy(h, h.theta0, k=1.5),
    )
    rejected = 0
    for fn in invalid_calls:
        try:
            fn()
        except ValueError:
            rejected += 1
    c.check("config/evaluation fail-fast", rejected == len(invalid_calls))


def stage_sensor(c: Checks, fast: bool) -> None:
    p = np.linspace(0.02, 0.98, 17)
    alpha, gamma = np.full_like(p, 0.31), np.full_like(p, 0.07)
    observed = p_observed(p, alpha, gamma)
    got = calibrate_gold_rate(observed, alpha, gamma)
    c.check("inverse round-trip", np.max(np.abs(got.raw - p)) < 1e-12)
    c.check("honest identity",
            np.array_equal(calibrate_gold_rate(p, 0.0, 0.0).scheduled, p))
    c.check("no noiseless projections", not got.projected.any())
    projected = calibrate_gold_rate(np.array([0.0, 1.0]), 0.2, 0.1)
    c.check("explicit projection mask", projected.projected.tolist() == [True, True])
    c.check("raw inverse retained", projected.raw[0] < 0 and projected.raw[1] > 1)
    c.check("scheduled estimate bounded",
            np.array_equal(projected.scheduled, np.array([0.0, 1.0])))
    c.raises("J<=0 rejected", ValueError, lambda: calibrate_gold_rate(.5, .6, .4))
    c.raises("invalid probability rejected", ValueError,
             lambda: calibrate_gold_rate(1.1, 0, 0))
    c.raises("shape mismatch rejected", ValueError,
             lambda: calibrate_gold_rate(np.ones(2), np.ones(3) / 4, 0))


def stage_rollout(c: Checks, fast: bool) -> None:
    t = make_mini_task("starved")
    p = 3
    probs = _policy_probs(t.theta0, t.tab)[p]
    rng = np.random.default_rng(31)
    n = 12_000 if fast else 80_000
    idx = rng.choice(t.tab.n_seqs, n, p=probs)
    reward = t.accept[p, idx]
    c.check("sampled reward is accept", np.array_equal(reward, t.accept[p, idx]))
    po, pg = probs @ t.accept[p], probs @ t.gold[p]
    c.check("fixture separates observed and gold", abs(po - pg) > 0.02,
            f"obs={po:.3f}, gold={pg:.3f}")
    groups = reward[: (n // 8) * 8].reshape(-1, 8)
    empirical = np.mean(np.all(groups == groups[:, :1], axis=1))
    right = dead_fraction_levels(np.array([po, 1 - po]), 8)
    wrong = dead_fraction_levels(np.array([pg, 1 - pg]), 8)
    c.check("dead rate follows p_obs law", abs(empirical - right) < 0.025,
            f"emp={empirical:.3f}, obs-law={right:.3f}")
    c.check("dead rate rejects p_gold law", abs(empirical - right) < abs(empirical - wrong))
    zero = group_advantage(np.zeros(8), "centered")
    c.check("dead rollout is exact zero", np.array_equal(zero, np.zeros(8)))


def _short_cfg(**kw) -> MiniCfg:
    return replace(MiniCfg(steps=4, group=32, n_active=4, calibration_interval=2,
                           sensor_ema=0.0), **kw)


def stage_truth_boundary(c: Checks, fast: bool) -> None:
    t = make_mini_task("starved")
    gold2 = 1.0 - t.gold
    same_reward = replace(t, gold=gold2)
    # Hold the data policy fixed so this is specifically a policy-loss boundary
    # test; calibrated curriculum audits are intentionally tested separately.
    cfg = _short_cfg(seed=17, sensor_mode="raw", sampler_mode="uniform")
    a = train_mini_rlvr(t, cfg)
    b = train_mini_rlvr(same_reward, cfg)
    c.check("gold mutation leaves update unchanged",
            np.array_equal(a.theta_final, b.theta_final))
    no_accept = replace(t, accept=np.zeros_like(t.accept))
    d = train_mini_rlvr(no_accept, cfg)
    c.check("accept mutation changes update",
            np.max(np.abs(a.theta_final - d.theta_final)) > 1e-6)
    c.check("no-op accept path is rejected",
            np.array_equal(d.theta_final, t.theta0) and
            not np.array_equal(a.theta_final, t.theta0))


def stage_estimator(c: Checks, fast: bool) -> None:
    r = np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.float64)
    a = group_advantage(r, "centered")
    c.check("imported advantage shape", a.shape == r.shape)
    c.check("advantages zero-sum", np.allclose(a.sum(1), 0))
    c.check("dead group zero advantage", np.array_equal(a[1], np.zeros(4)))
    white = group_advantage(r, "normalized")
    c.check("whitened dead group stays finite and zero",
            np.all(np.isfinite(white)) and np.array_equal(white[1], np.zeros(4)))
    detached = torch.as_tensor(a).detach()
    c.check("advantage detached", not detached.requires_grad)
    t = make_mini_task("honest")
    h = train_mini_rlvr(t, _short_cfg(steps=2, seed=9))
    initial = evaluate_policy(t, t.theta0, selector="filter")
    c.check("row zero before update", np.array_equal(h.p_gold[0], initial.p_gold))
    c.check("policy moves under non-dead reward",
            np.max(np.abs(h.theta_final - t.theta0)) > 1e-8)


def stage_controller(c: Checks, fast: bool) -> None:
    honest, starved = make_mini_task("honest"), make_mini_task("starved")
    raw_h = train_mini_rlvr(honest, _short_cfg(steps=2, sensor_mode="raw"))
    cal_h = train_mini_rlvr(honest, _short_cfg(steps=2, sensor_mode="calibrated"))
    c.check("honest raw/calibrated identity",
            np.array_equal(raw_h.p_hat_gold, cal_h.p_hat_gold))
    raw_s = train_mini_rlvr(starved, _short_cfg(steps=2, sensor_mode="raw"))
    cal_s = train_mini_rlvr(starved, _short_cfg(steps=2, sensor_mode="calibrated"))
    c.check("incomplete checker sensor separation",
            np.max(np.abs(raw_s.p_hat_gold[0] - cal_s.p_hat_gold[0])) > .02)
    c.check("raw sensor misuse rejected",
            np.mean(np.abs(cal_s.p_hat_gold[0] - cal_s.p_gold[0])) <
            np.mean(np.abs(raw_s.p_hat_gold[0] - raw_s.p_gold[0])))
    c.check("active set uses recorded prior estimate",
            np.all(cal_s.active[0] == BandTracker(4, set_point_analytic(8), 1.5)(
                0, cal_s.p_hat_gold[0], np.random.default_rng(10_007))))
    c.check("active IDs valid", np.all((cal_s.active >= 0) &
                                       (cal_s.active < starved.n_prompts)))
    c.check("in-band and fallback accounted",
            np.all((cal_s.in_band >= 0) & (cal_s.in_band <= 1)))


def stage_selectors(c: Checks, fast: bool) -> None:
    t = make_mini_task("leaky")
    e = evaluate_policy(t, t.theta0, k=8, selector="filter")
    right = np.mean(pass_at_k_exact(e.p_gold, 8))
    wrong = pass_at_k_exact(np.mean(e.p_gold), 8)
    c.check("per-prompt then mean pass@k", abs(e.gold_passk - right) < 1e-13)
    c.check("mean-then-pass rejected non-vacuously", abs(right - wrong) > 0.03,
            f"gap={abs(right-wrong):.3f}")
    c.check("filter/ranker/vote are distinct",
            np.ptp([e.filter_accuracy, e.ranker_accuracy, e.vote_accuracy]) > .01)
    c.check("selector is named", e.selector_name == "filter")
    c.check("oracle not copied into selector",
            abs(e.gold_passk - e.deployed_accuracy) > .01)
    vote = evaluate_policy(t, t.theta0, k=8, selector="vote")
    c.check("selector dispatch", vote.deployed_accuracy == vote.vote_accuracy)
    c.check("policy-weighted precision bounded", 0 <= e.precision <= 1)


def stage_train(c: Checks, fast: bool) -> None:
    steps = 28 if fast else 60
    t = make_mini_task("honest")
    cfg = replace(MiniCfg(), steps=steps, seed=23)
    h1, h2 = train_mini_rlvr(t, cfg), train_mini_rlvr(t, cfg)
    c.check("deterministic seed", np.array_equal(h1.theta_final, h2.theta_final))
    c.check("all tracked arrays finite",
            all(np.all(np.isfinite(getattr(h1, n))) for n in
                ("p_obs", "p_gold", "p_hat_gold", "verifier_passk", "gold_passk",
                 "deployed_accuracy", "precision", "dead_frac", "signal", "kl",
                 "entropy", "in_band", "projection_frac")))
    c.check("history length", len(h1.step) == steps and h1.active.shape == (steps, 4))
    c.check("honest verifier/gold trajectories agree",
            np.array_equal(h1.verifier_passk, h1.gold_passk))
    c.check("training improves verifier", h1.verifier_passk[-1] > h1.verifier_passk[0])
    c.check("KL explicit and nonzero", h1.kl[-1] > 0)
    c.check("entropy tracked and changes", abs(h1.entropy[-1] - h1.entropy[0]) > 1e-5)


def stage_ablation(c: Checks, fast: bool) -> None:
    steps = 20 if fast else 42
    seeds = range(5)
    honest = make_mini_task("honest")
    starved = make_mini_task("starved")
    leaky = make_mini_task("leaky")
    honest_diffs, sensor_gaps = [], []
    band_gains_per_1k, uniform_gains_per_1k = [], []
    leaky_sensor_repairs, leaky_gold_gains, leaky_reward_gaps = [], [], []
    base = replace(MiniCfg(), steps=steps)
    gain_scale = 1_000.0 / (base.steps * base.n_active * base.group)

    def final_panel(task: MiniTask, history: MiniHistory, cfg: MiniCfg) -> MiniEvaluation:
        return evaluate_policy(
           task, history.theta_final, cfg.k_eval, cfg.selector_mode, task.theta0
        )

    for seed in seeds:
        common = replace(base, seed=seed)
        hr = train_mini_rlvr(honest, replace(common, sensor_mode="raw"))
        hc = train_mini_rlvr(honest, replace(common, sensor_mode="calibrated"))
        sr = train_mini_rlvr(starved, replace(common, sensor_mode="raw"))
        sc = train_mini_rlvr(starved, replace(common, sensor_mode="calibrated"))
        lr = train_mini_rlvr(leaky, replace(common, sensor_mode="raw"))
        lc = train_mini_rlvr(leaky, replace(common, sensor_mode="calibrated"))
        un = train_mini_rlvr(starved, replace(common, sensor_mode="calibrated",
                                              sampler_mode="uniform"))
        sc_final = final_panel(starved, sc, common)
        un_final = final_panel(starved, un, common)
        lc_final = final_panel(leaky, lc, common)
        honest_diffs.append(np.max(np.abs(hr.theta_final - hc.theta_final)))
        sensor_gaps.append(np.mean(np.abs(sr.p_hat_gold - sr.p_gold)) -
                          np.mean(np.abs(sc.p_hat_gold - sc.p_gold)))
        leaky_sensor_repairs.append(
           np.mean(np.abs(lr.p_hat_gold - lr.p_gold)) -
           np.mean(np.abs(lc.p_hat_gold - lc.p_gold)))
        leaky_gold_gains.append(lc_final.gold_passk - lc.gold_passk[0])
        leaky_reward_gaps.append(lc_final.verifier_passk - lc_final.gold_passk)
        band_gains_per_1k.append(
           (sc_final.gold_passk - sc.gold_passk[0]) * gain_scale
        )
        uniform_gains_per_1k.append(
           (un_final.gold_passk - un.gold_passk[0]) * gain_scale
        )
    c.check("five paired seeds", len(honest_diffs) >= 5)
    c.check("honest negative control exact", max(honest_diffs) == 0.0)
    c.check("calibration repairs starved coordinate", min(sensor_gaps) > 0)
    c.check("calibration repairs leaky coordinate", min(leaky_sensor_repairs) > 0.25)
    c.check("calibration cannot repair leaky objective", max(leaky_gold_gains) < 0.0)
    c.check("leaky verifier/gold split remains", min(leaky_reward_gaps) > 0.20)
    effect = float(np.mean(band_gains_per_1k) - np.mean(uniform_gains_per_1k))
    spread = float(np.std(np.subtract(band_gains_per_1k, uniform_gains_per_1k)))
    c.check("paired effect/spread reported", np.isfinite(effect + spread),
           f"effect/1k={effect:+.4f}, paired spread/1k={spread:.4f}")
    # A null is a valid pre-registered outcome; this check rejects significance
    # theater rather than demanding a direction from five toy seeds.
    verdict = "null" if abs(effect) <= 2 * spread else "resolved"
    c.check("null accepted", verdict in {"null", "resolved"},
           f"effect/1k={effect:+.4f}, spread/1k={spread:.4f}")
    print(f"  final-policy gains/1k: band={np.mean(band_gains_per_1k):+.4f}, "
          f"uniform={np.mean(uniform_gains_per_1k):+.4f}, "
          f"effect={effect:+.4f}, spread={spread:.4f}: {verdict}")
    print(f"  leaky calibrated gold gain={np.mean(leaky_gold_gains):+.4f}, "
          f"verifier-gold gap={np.mean(leaky_reward_gaps):.4f}")


def _diagnostic_bench(fast: bool):
    steps = 28 if fast else 52
    healthy, sick = [], {n: [] for n in DiagThresholds.__dataclass_fields__}
    for seed in range(3):
        base = replace(MiniCfg(), steps=steps, seed=100 + seed)
        hh = train_mini_rlvr(make_mini_task("honest"), base)
        healthy.append(hh)
        # Start with independent measured histories, then inject exactly one
        # mechanism at the panel boundary.  This calibrates diagnostic semantics
        # without relying on accidental correlations among training fixtures.
        for name in sick:
            x = copy.deepcopy(hh)
            if name == "dead_budget":
                x.dead_frac[-12:] = 0.94
            elif name == "sensor_mismatch":
                x.p_hat_gold[-12:] = np.clip(x.p_gold[-12:] + .28, 0, 1)
            elif name == "reward_hacking":
                x.verifier_passk[-12:] = np.clip(x.verifier_passk[0] + .22, 0, 1)
                x.gold_passk[-12:] = x.gold_passk[0]
            elif name == "selector_gap":
                x.deployed_accuracy[-12:] = np.maximum(0, x.gold_passk[-12:] - .30)
            elif name == "kl_runaway":
                x.kl[-12:] = .85
            sick[name].append(x)
    return healthy, sick


def stage_diagnose(c: Checks, fast: bool) -> None:
    healthy, sick = _diagnostic_bench(fast)
    thresholds = calibrate_diagnostic_thresholds(healthy, sick)
    c.check("healthy silence",
            all(diagnose_mini_rlvr(h, thresholds) == [] for h in healthy))
    fire_sets = {}
    for mechanism, histories in sick.items():
        fired = [frozenset(n for n, _ in diagnose_mini_rlvr(h, thresholds))
                 for h in histories]
        fire_sets[mechanism] = fired[0]
        c.check(f"{mechanism} fires", all(mechanism in f for f in fired))
        details = dict(diagnose_mini_rlvr(histories[0], thresholds))
        c.check(f"{mechanism} reason has number",
                any(ch.isdigit() for ch in details[mechanism]))
    c.check("pairwise fire sets differ",
            len(set(fire_sets.values())) == len(fire_sets), str(fire_sets))
    c.check("thresholds measured in calibration air",
            all(np.isfinite(getattr(thresholds, n))
                for n in DiagThresholds.__dataclass_fields__))


STAGES = [
    ("contract", stage_contract),
    ("sensor", stage_sensor),
    ("rollout", stage_rollout),
    ("truth_boundary", stage_truth_boundary),
    ("estimator", stage_estimator),
    ("controller", stage_controller),
    ("selectors", stage_selectors),
    ("train", stage_train),
    ("ablation", stage_ablation),
    ("diagnose", stage_diagnose),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="*", help="stage names (default: all)")
    parser.add_argument("--fast", action="store_true", help="smaller validation fixtures")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(name for name, _ in STAGES))
        return 0
    known = {name: fn for name, fn in STAGES}
    requested = args.stages or list(known)
    unknown = sorted(set(requested) - set(known))
    if unknown:
        parser.error(f"unknown stages: {', '.join(unknown)}")
    checks = Checks()
    started = time.perf_counter()
    failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for name in requested:
            before, t0 = checks.total, time.perf_counter()
            try:
                known[name](checks, args.fast)
            except Exception as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
                traceback.print_exc()
            else:
                print(f"PASS {name:14s} {checks.total-before:2d} checks "
                      f"{time.perf_counter()-t0:6.2f}s")
    elapsed = time.perf_counter() - started
    status = "PASS" if failed == 0 else "FAIL"
    print(f"{status}: {checks.total} checks, {failed} failed, {elapsed:.2f}s")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
