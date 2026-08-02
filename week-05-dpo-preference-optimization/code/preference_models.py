"""
Week 5 Day 1 -- what preference data can and cannot identify.

W4 D1 treated the reward model as a FUNCTION APPROXIMATION problem: fit r_psi, worry
about generalization off-distribution. Today treats it as an IDENTIFICATION problem.
Delete approximation error entirely -- infinite data, a tabular r with one free
parameter per completion, a perfectly specified model -- and ask what is STILL unknown.

The answer is much larger than "an additive constant per prompt", and the extra part is
the whole of Week 5's failure surface. One thesis:

    The scale of a Bradley-Terry reward is not a property of quality.
    It is a measurement of annotator disagreement.

Four consequences, one stage each:

  (a) FORD/ZERMELO. The BT MLE is finite IFF the wins digraph is strongly connected.
      Corollary: a noiseless annotator on a strictly ordered item set produces a
      transitive tournament -- a DAG -- so the MLE is at INFINITY with infinite data.
      More data adds edges but never a cycle.
  (b) THE DPO SHAPE. A preference dataset is (x, y_w, y_l) triples in which each
      completion appears exactly ONCE: a perfect matching, the most disconnected
      non-trivial graph there is. Tomorrow you show DPO's implicit reward is
      beta*log(pi/pi_ref), so "r_l -> -inf" reads literally as "pi(y_l) -> 0".
  (c) NOISE IS THE REGULARIZER. Flip noise eps caps the saturated per-pair margin at
      logit(1-eps) -- finite. IPO and cDPO (Day 4, Day 3) put that cap back by hand.
  (d) SCALE AND BETA ARE ONE KNOB. tilt(c*r, c*beta) == tilt(r, beta) exactly, so only
      r/beta is identified by the policy. Chain (c) into (d): the agreement rate sets
      the scale, the scale sets what beta means. beta is a property of your ANNOTATION
      PROCESS, which is why it does not transfer and why "just use 0.1" is a confession.

  run everything   ./.venv/bin/python ai-lab-mle-prep/week-05-dpo-preference-optimization/code/preference_models.py
  one stage        ...  preference_models.py existence
  a few stages     ...  preference_models.py bt_grad graph
  fast fixtures    ...  preference_models.py --fast
  list the stages  ...  preference_models.py --list

Build order (each group is runnable the moment it is written):

  1. bt_nll_and_grad                      -> stage `bt_grad`
  2. wins_digraph + is_strongly_connected -> stage `graph`
  3. noisy_labels                         -> stage `graph` (needs both)
  4. fit_bt                               -> stages `existence`, `direction`, `matching`
  5. (nothing new)                        -> stage `cap`
  6. tilt                                 -> stage `scale`
  7. plackett_luce_nll                    -> stage `plackett`
  8. bayes_accuracy                       -> stage `ceiling`

Full run ~35 s, `--fast` ~8 s. Run it under `-W error::RuntimeWarning`.
"""

from __future__ import annotations

import sys
import traceback
from functools import cached_property

import numpy as np


# ---------------------------------------------------------------------------
# Numerics helpers (provided)
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free sigmoid: exponentiate only the negative branch."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def logsigmoid(z: np.ndarray) -> np.ndarray:
    """log sigma(z) = -softplus(-z), computed stably.

    Use THIS, never `np.log(sigmoid(z))`. The naive form returns exactly 0.0 for
    z <~ -37 -- and a diverging fit lives precisely in that region, so the naive form
    silently reports zero loss on the pairs that are hurting you. Day 3 turns this into
    a measured check.
    """
    return -np.logaddexp(0.0, -np.asarray(z, dtype=float))


def pairwise_accuracy(r: np.ndarray, pairs: np.ndarray) -> float:
    """Fraction of pairs whose first column (the recorded winner) has the higher r."""
    return float(np.mean(r[pairs[:, 0]] > r[pairs[:, 1]]))


# ---------------------------------------------------------------------------
# Exercises (TODO bodies)
# ---------------------------------------------------------------------------


def bt_nll_and_grad(r: np.ndarray, pairs: np.ndarray) -> tuple[float, np.ndarray]:
    """Bradley-Terry negative log-likelihood and its analytic gradient.

    `pairs` is an int array of shape (m, 2); row k is (winner_index, loser_index).
    `r` is a TABULAR reward, shape (n_items,) -- one free parameter per completion.

        P(w beats l) = sigma(r_w - r_l)
        NLL          = -mean_k log sigma(r[w_k] - r[l_k])

    Returns `(nll, grad)` with `grad.shape == r.shape`.

    You derived the gradient on W4 D1: the per-pair weight is sigma(-Delta), so pairs
    the model already ranks correctly contribute almost nothing and the fit is driven
    by near-ties and mislabels. That is the whole reason annotator disagreement is
    load-bearing rather than a nuisance.

    Idiom: an item appears in MANY pairs, so the per-pair contributions COLLIDE when
    scattered back to item indices -> `np.add.at`. (W4 D4's rule: collisions ->
    `add.at`; a permutation -> `put_along_axis`.) And `np.add.at` MUTATES and returns
    None, so `g = np.add.at(...)` gives you None -- third sighting of that family.

    Use `logsigmoid`, not `np.log(sigmoid(...))`.
    """
    # TODO
    raise NotImplementedError


def wins_digraph(pairs: np.ndarray, n_items: int) -> np.ndarray:
    """Boolean adjacency of the "beats" digraph: adj[i, j] is True iff i beat j at least once.

    Shape (n_items, n_items). Self-loops stay False. Repeated comparisons collapse to a
    single edge -- existence is a QUESTION ABOUT STRUCTURE, not about counts, which is
    exactly why collecting more of the same comparisons cannot fix it.
    """
    # TODO
    raise NotImplementedError


def is_strongly_connected(adj: np.ndarray) -> bool:
    """True iff every item is reachable from every other along "beats" edges.

    Ford (1957) / Zermelo (1929): the BT MLE is FINITE iff this holds. Equivalently,
    for every partition of the items into non-empty S and S^c, someone in S beat
    someone in S^c AND vice versa -- i.e. the data is NOT linearly separable in the
    difference features e_w - e_l.

    Idiom: boolean matrix multiplication IS reachability composition. Start from the
    REFLEXIVE closure (`adj | I`) and square it about log2(n) times; strong connectivity
    is then "the closure is all True". n is small here, so clarity beats Tarjan.
    """
    # TODO
    raise NotImplementedError


def fit_bt(pairs: np.ndarray, n_items: int, n_steps: int = 20_000,
           lr: float = 1.0, record: tuple[int, ...] = ()) -> tuple[np.ndarray, float, dict]:
    """Fit a tabular Bradley-Terry reward by plain gradient descent, recording the path.

    Returns `(r_final, nll_final, history)` where `history[t]` is a COPY of `r` after
    step t, for each t in `record`. Steps are 1-indexed.

    Two things that are not optional:

      * RE-CENTER `r` after every step (`r -= r.mean()`). This gauge-fixes the location
        invariance you already know about. It matters because today's result is that
        the MLE can fail to exist for a reason that is NOT the additive constant -- and
        the only way to demonstrate that unambiguously is to remove the constant first
        and watch the norm run away anyway.
      * Record COPIES. `history[t] = r` stores a reference; the next step mutates it and
        every recorded entry becomes the same final array.

    Plain GD, fixed lr, full batch. No momentum, no decay -- the implicit bias of plain
    gradient descent on separable logistic data is precisely the phenomenon being
    measured, and an optimizer with its own regularization would hide it.
    """
    # TODO
    raise NotImplementedError


def noisy_labels(rng: np.random.Generator, gold_r: np.ndarray, pairs: np.ndarray,
                 eps: float, stochastic: bool = True) -> np.ndarray:
    """Label unordered comparisons, with the two noise sources kept SEPARATE.

    `pairs` shape (m, 2), unordered candidates. Returns shape (m, 2) ORDERED as
    (winner, loser).

    Two independent sources, and the distinction is the day's precision point:

      * `stochastic=True`  -> the annotator is a Bradley-Terry sampler at `gold_r`:
        item 0 wins with probability sigma(gold_r[i] - gold_r[j]). This ALREADY produces
        upsets whenever the true margin is small. A BT annotator at finite gold_r is
        itself a noisy annotator.
      * `stochastic=False` -> a DETERMINISTIC annotator: the truly-better item always
        wins. This is the noiseless case, and it is the one that diverges.

    Then, independently, flip the recorded label with probability `eps`.

    Idiom: `first_wins ^ flipped` -- XOR is the flip. And to reorder rows use
    `np.where(first_wins[:, None], pairs, pairs[:, ::-1])`. Do NOT reach for boolean
    mask indexing: `x[mask]` FLATTENS and drops the pair axis (fourth family sighting,
    after W4 D5's 2.16x KL error). `np.where` preserves shape.
    """
    # TODO
    raise NotImplementedError


def plackett_luce_nll(r: np.ndarray, rankings: list) -> float:
    """Plackett-Luce negative log-likelihood, averaged over rankings.

    A ranking is a sequence of item indices, best first. PL is a sequence of softmax
    choices over the REMAINING items:

        P(ranking) = prod_k  exp(r[y_k]) / sum_{j >= k} exp(r[y_j])

    K = 2 is exactly Bradley-Terry, and the harness asserts that to ~1e-15. It is a
    cheap check with real teeth: the single likeliest bug is summing the denominator
    over ALL items instead of the remaining ones, and at K=2 that reduces to the wrong
    thing in a way BT catches immediately.

    Note it inherits the same location invariance -- the constant cancels in every
    factor. Use a stable log-sum-exp for the denominator.
    """
    # TODO
    raise NotImplementedError


def tilt(r: np.ndarray, pi_ref: np.ndarray, beta: float) -> np.ndarray:
    """The KL-regularized optimum: pi*(a) prop pi_ref(a) * exp(r(a) / beta). Returns a pmf.

    You derived this on W3 D5; it is imported here rather than re-derived because today
    needs one specific property of it. Compute in LOG space.

    Guard: pi_ref may contain exact zeros (support confinement is a real part of this
    object). `np.where` is NOT short-circuit -- both branches are fully evaluated, so a
    guard downstream of `np.log` only LAUNDERS a warning that already happened. Sanitize
    the INPUT: `np.log(np.where(pi_ref > 0, pi_ref, 1.0))`, then force the forbidden
    entries to -inf explicitly. Sixth sighting of this family in three weeks.
    """
    # TODO
    raise NotImplementedError


def bayes_accuracy(gold_r: np.ndarray, pairs: np.ndarray) -> float:
    """The Bayes-optimal pairwise accuracy achievable against a BT annotator at `gold_r`.

    For each pair the label is a coin with p = sigma(gold_r[i] - gold_r[j]); the best any
    predictor can do is call the more likely side, so the ceiling is E[max(p, 1-p)].

    This is the number that makes "our RM gets 96% pairwise accuracy" answerable. The
    ceiling is set by the ANNOTATOR, not by the model, so a reported accuracy above it
    means your pair distribution is easy or your labels are near-deterministic -- and
    near-deterministic labels are exactly the divergent regime.
    """
    # TODO
    raise NotImplementedError


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


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


class Ctx:
    """Shared fixtures. Eight completions with a STRICT quality ordering, and three
    comparison graphs over the identical items -- the only thing that changes between
    them is which comparisons exist and how they were labelled."""

    N_ITEMS = 8

    def __init__(self, fast: bool = False):
        self.fast = fast
        self.long_steps = 20_000 if fast else 200_000
        self.rec = ((100, 1_000, 2_000, 10_000, 20_000)
                    if fast else (100, 1_000, 2_000, 10_000, 20_000, 100_000, 200_000))
        self.dense_rep = 30 if fast else 100

    @cached_property
    def gold(self):
        return np.arange(self.N_ITEMS, dtype=float)

    @cached_property
    def all_pairs(self):
        n = self.N_ITEMS
        return np.array([(i, j) for i in range(n) for j in range(n) if i > j])

    @cached_property
    def matching(self):
        """Each completion appears EXACTLY ONCE. This is the shape of a DPO dataset."""
        return np.array([(0, 1), (2, 3), (4, 5), (6, 7)])

    @cached_property
    def tournament(self):
        """All 28 comparisons, deterministic annotator, no upsets."""
        return noisy_labels(np.random.default_rng(1), self.gold, self.all_pairs,
                            0.0, stochastic=False)

    @cached_property
    def noised(self):
        """The SAME comparisons, labelled by the same deterministic annotator, then
        flipped at eps=0.10. Nothing about the items or the reward has changed."""
        rng = np.random.default_rng(2)
        rep = np.repeat(self.all_pairs, 60, axis=0)
        return noisy_labels(rng, self.gold, rep, 0.10, stochastic=False)

    @cached_property
    def fit_tournament(self):
        print("  ... fitting the transitive tournament (long run)")
        return fit_bt(self.tournament, self.N_ITEMS, n_steps=self.long_steps,
                      lr=1.0, record=self.rec)

    @cached_property
    def fit_noised(self):
        print("  ... fitting the eps=0.10 relabelling of the same comparisons")
        return fit_bt(self.noised, self.N_ITEMS, n_steps=self.long_steps,
                      lr=1.0, record=self.rec)


def _fail_fast(ctx: Ctx) -> None:
    """Touch the cheap functions a heavy stage needs, so a missing TODO surfaces before
    a 200k-step fit is paid for."""
    bt_nll_and_grad(np.zeros(2), np.array([[0, 1]]))
    fit_bt(np.array([[0, 1]]), 2, n_steps=2, lr=1.0)


# ---- individual stages ----------------------------------------------------


def stage_bt_grad(ctx: Ctx) -> None:
    """bt_nll_and_grad -- finite differences, log 2, and the sigma(-Delta) weight"""
    rng = np.random.default_rng(0)
    r0 = rng.standard_normal(ctx.N_ITEMS)
    nll, g = bt_nll_and_grad(r0, ctx.all_pairs)

    _check("gradient has the shape of r", g.shape == r0.shape, f"{g.shape} vs {r0.shape}")

    fd = np.zeros_like(r0)
    for i in range(len(r0)):
        e = np.zeros_like(r0)
        e[i] = 1e-6
        fd[i] = (bt_nll_and_grad(r0 + e, ctx.all_pairs)[0]
                 - bt_nll_and_grad(r0 - e, ctx.all_pairs)[0]) / 2e-6
    err = float(np.abs(g - fd).max())
    _check("analytic gradient == central finite differences", err < 1e-8, f"max|d| = {err:.3e}")

    flat = bt_nll_and_grad(np.zeros(ctx.N_ITEMS), ctx.all_pairs)[0]
    _check("equal rewards give exactly log 2", abs(flat - np.log(2)) < 1e-15,
           f"{flat:.15f} vs {np.log(2):.15f}")

    _check("gradient sums to zero (the location direction is FLAT)",
           abs(float(g.sum())) < 1e-12, f"{float(g.sum()):.3e}")

    probe = np.array([[0, 1]])
    for delta in (0.0, 2.0, 5.0):
        _, gg = bt_nll_and_grad(np.array([delta, 0.0]), probe)
        _check(f"per-pair gradient magnitude at margin {delta} is sigma(-Delta)",
               abs(abs(float(gg[0])) - float(sigmoid(-delta))) < 1e-12,
               f"|g| = {abs(float(gg[0])):.6f}, sigma(-d) = {float(sigmoid(-delta)):.6f}")
    print("  ... => correctly-ranked pairs contribute ~nothing; the fit is driven by\n"
          "      near-ties and mislabels. That is why disagreement is load-bearing.")


def stage_graph(ctx: Ctx) -> None:
    """wins_digraph + is_strongly_connected -- Ford's condition on three fixtures"""
    adj = wins_digraph(ctx.tournament, ctx.N_ITEMS)
    _check("adjacency is (n, n) boolean", adj.shape == (ctx.N_ITEMS, ctx.N_ITEMS)
           and adj.dtype == bool, f"{adj.shape}, {adj.dtype}")
    _check("no self-loops", not bool(np.diag(adj).any()))
    _check("a deterministic annotator gives a TRANSITIVE TOURNAMENT (upper triangular)",
           bool((adj == np.triu(np.ones_like(adj), 1)[::-1, ::-1].T[::-1, ::-1]).all())
           or bool((np.tril(adj, -1) == adj).all()) or bool((np.triu(adj, 1) == adj).all()),
           "every edge points the same way through the ordering")

    _check("transitive tournament is NOT strongly connected -> MLE at infinity",
           not is_strongly_connected(adj))
    _check("perfect matching is NOT strongly connected -> MLE at infinity",
           not is_strongly_connected(wins_digraph(ctx.matching, ctx.N_ITEMS)))
    _check("eps=0.10 relabelling of the SAME comparisons IS strongly connected",
           is_strongly_connected(wins_digraph(ctx.noised, ctx.N_ITEMS)))

    single = wins_digraph(np.array([[0, 1], [1, 0]]), 2)
    _check("two items with one upset each way: strongly connected",
           is_strongly_connected(single))
    _check("two items, one-way only: not strongly connected",
           not is_strongly_connected(wins_digraph(np.array([[0, 1]]), 2)))

    flipped = float(np.mean(ctx.noised[:, 0] < ctx.noised[:, 1]))
    print(f"  ... upset rate in the noised fixture: {flipped:.3f} (eps was 0.10)")
    print("  ... => the ONLY difference between fixture 1 and fixture 3 is whether the\n"
          "      raters ever disagreed. Same items, same reward, same comparisons.")


def stage_existence(ctx: Ctx) -> None:
    """fit_bt -- the MLE diverges on clean data and converges on noisy data"""
    _fail_fast(ctx)
    r_t, nll_t, hist_t = ctx.fit_tournament
    norms = [float(np.linalg.norm(hist_t[t])) for t in ctx.rec]
    for t, nm in zip(ctx.rec, norms):
        print(f"      t={t:>7}   ||r|| = {nm:8.4f}")

    _check("the fit was re-centred (location gauge is fixed)", abs(float(r_t.mean())) < 1e-9,
           f"mean(r) = {float(r_t.mean()):.3e}")
    _check("||r|| is monotone increasing across every decade",
           all(b > a for a, b in zip(norms, norms[1:])), f"{[round(x, 2) for x in norms]}")
    _check("pairwise accuracy is EXACTLY 1.0 while the parameter diverges",
           pairwise_accuracy(r_t, ctx.tournament) == 1.0)

    r_n, nll_n, hist_n = ctx.fit_noised
    # Relative, not an arbitrary constant: the separable fit drives the loss toward 0,
    # the non-separable one cannot go below the entropy its own upsets imply.
    _check("clean-label loss is driven toward 0", nll_t < 0.01 * nll_n,
           f"nll = {nll_t:.3e} vs the noisy fit's {nll_n:.3e} ({nll_n/nll_t:.0f}x)")
    norms_n = [float(np.linalg.norm(hist_n[t])) for t in ctx.rec]
    for t, nm in zip(ctx.rec, norms_n):
        print(f"      t={t:>7}   ||r|| = {nm:8.4f}   (eps = 0.10)")

    tail_t = abs(norms[-1] - norms[-2]) / norms[-2]
    tail_n = abs(norms_n[-1] - norms_n[-2]) / norms_n[-2]
    _check("clean-label fit is still MOVING over the last decade", tail_t > 1e-2,
           f"relative change = {tail_t:.3e}")
    _check("noisy-label fit has CONVERGED over the last decade", tail_n < 1e-6,
           f"relative change = {tail_n:.3e}")
    _check("the noisy fit's norm is far smaller", norms_n[-1] < 0.5 * norms[-1],
           f"{norms_n[-1]:.3f} vs {norms[-1]:.3f}")
    print("  ... => a noiseless annotator on a strictly ordered set gives an MLE at\n"
          "      INFINITY, with infinite data. More data adds edges but never a cycle.")


def stage_direction(ctx: Ctx) -> None:
    """the divergence is invisible: direction converges while the norm runs away"""
    _, _, hist = ctx.fit_tournament
    decades = [(a, b) for a, b in zip(ctx.rec, ctx.rec[1:]) if b >= 2 * a]
    _check("have at least two decades to compare", len(decades) >= 2, f"{decades}")
    for a, b in decades:
        u = hist[a] / np.linalg.norm(hist[a])
        v = hist[b] / np.linalg.norm(hist[b])
        cos = float(u @ v)
        ratio = float(np.linalg.norm(hist[b]) / np.linalg.norm(hist[a]))
        _check(f"direction has converged between t={a} and t={b}", cos > 0.999,
               f"cos = {cos:.10f}")
        _check(f"but the norm has NOT, between t={a} and t={b}", ratio > 1.05,
               f"||r_b||/||r_a|| = {ratio:.4f}")
    print("  ... => the direction converges to the max-margin separator while ||r|| grows\n"
          "      logarithmically without bound (Soudry et al., implicit bias of GD).\n"
          "      Nothing errors, nothing NaNs, and accuracy reads 100%.")


def stage_matching(ctx: Ctx) -> None:
    """the DPO shape: each completion appears exactly once"""
    rec = (1_000, 10_000) if ctx.fast else (1_000, 10_000, 100_000)
    r, nll, hist = fit_bt(ctx.matching, ctx.N_ITEMS, n_steps=max(rec), lr=1.0, record=rec)

    _check("not strongly connected", not is_strongly_connected(wins_digraph(ctx.matching, ctx.N_ITEMS)))
    mins = [float(hist[t].min()) for t in rec]
    for t, m in zip(rec, mins):
        margin = float((hist[t][ctx.matching[:, 0]] - hist[t][ctx.matching[:, 1]]).min())
        print(f"      t={t:>7}   ||r|| = {float(np.linalg.norm(hist[t])):8.4f}   "
              f"min r_i = {m:8.4f}   min margin = {margin:8.4f}")
    _check("min r_i falls monotonically without bound",
           all(b < a for a, b in zip(mins, mins[1:])), f"{[round(x, 3) for x in mins]}")
    _check("every pair's margin exceeds 8 nats", 
           float((r[ctx.matching[:, 0]] - r[ctx.matching[:, 1]]).min()) > 8.0)
    _check("loss ~0 and accuracy 1.0 throughout", nll < 1e-3
           and pairwise_accuracy(r, ctx.matching) == 1.0, f"nll = {nll:.3e}")

    flipped = ctx.matching[:, ::-1]
    _check("flipping every label does NOT create a cycle -- noise has nowhere to act",
           not is_strongly_connected(wins_digraph(flipped, ctx.N_ITEMS)))
    print("  ... => tomorrow: DPO's implicit reward is beta*log(pi/pi_ref), so this\n"
          f"      min r_i = {float(r.min()):.4f} IS the rejected completion's logprob.\n"
          "      'DPO drives log pi(y_l) to -inf' is Ford's theorem, not a bug.")


def stage_cap(ctx: Ctx) -> None:
    """flip noise caps the achievable margin at logit(1 - eps)"""
    print("  saturated case (2 items, one pair, EXACT win counts -- no sampling noise):")
    m_total = 2_000
    for eps in (0.02, 0.05, 0.10, 0.20, 0.35):
        n_lose = int(round(eps * m_total))
        pp = np.array([[0, 1]] * (m_total - n_lose) + [[1, 0]] * n_lose)
        r, _, _ = fit_bt(pp, 2, n_steps=6_000, lr=4.0)
        got = float(r[0] - r[1])
        want = float(np.log((1 - eps) / eps))
        _check(f"eps={eps}: fitted margin == logit(1-eps)", abs(got / want - 1) < 1e-6,
               f"{got:.4f} vs {want:.4f}")

    print("  parametric case (8 items, 28 pairs -- BT is MISSPECIFIED under flip noise,")
    print("  so the fit compromises; finite and monotone is the claim, not the cap):")
    prev = np.inf
    for eps in (0.02, 0.05, 0.10, 0.20, 0.35):
        rng = np.random.default_rng(12)
        pp = noisy_labels(rng, ctx.gold, np.repeat(ctx.all_pairs, ctx.dense_rep, axis=0),
                          eps, stochastic=False)
        r, _, _ = fit_bt(pp, ctx.N_ITEMS, n_steps=20_000, lr=1.0)
        spread = float(r.max() - r.min())
        _check(f"eps={eps}: graph is strongly connected",
               is_strongly_connected(wins_digraph(pp, ctx.N_ITEMS)))
        _check(f"eps={eps}: spread is smaller than at the previous eps", spread < prev,
               f"spread = {spread:.4f}")
        prev = spread
    print("  ... => eps -> 0 is the transitive tournament, i.e. the existence stage.\n"
          "      The two results are ONE result. Noise is the regularizer, and Day 3's\n"
          "      cDPO / Day 4's IPO are two ways of putting the cap back by hand.")


def stage_scale(ctx: Ctx) -> None:
    """tilt -- scale and beta are ONE knob; location is free"""
    rng = np.random.default_rng(3)
    pi_ref = rng.random(ctx.N_ITEMS)
    pi_ref /= pi_ref.sum()
    base = tilt(ctx.gold, pi_ref, 0.5)

    _check("tilt returns a pmf", abs(float(base.sum()) - 1.0) < 1e-12 and bool((base >= 0).all()))
    for c in (0.1, 1.0, 3.0, 17.0):
        err = float(np.abs(tilt(c * ctx.gold, pi_ref, c * 0.5) - base).max())
        _check(f"tilt(c*r, c*beta) == tilt(r, beta) at c={c}", err < 1e-15, f"max|d| = {err:.3e}")

    shifted = ctx.gold + 4.2
    _check("location is free in the tilt",
           float(np.abs(tilt(shifted, pi_ref, 0.5) - base).max()) < 1e-15)
    _check("location is free in the NLL",
           abs(bt_nll_and_grad(shifted, ctx.all_pairs)[0]
               - bt_nll_and_grad(ctx.gold, ctx.all_pairs)[0]) < 1e-15)

    zero_ref = pi_ref.copy()
    zero_ref[3] = 0.0
    zero_ref /= zero_ref.sum()
    forced = tilt(ctx.gold, zero_ref, 0.5)
    _check("support confinement: pi*[a] is EXACTLY 0 where pi_ref is 0",
           float(forced[3]) == 0.0, f"got {float(forced[3]):.3e}")

    kls = [_kl(tilt(s * ctx.gold, pi_ref, 0.5), pi_ref) for s in (0.5, 1.0, 2.0, 4.0)]
    for s, k in zip((0.5, 1.0, 2.0, 4.0), kls):
        print(f"      reward scale x{s:<4}  ->  KL(pi||pi_ref) = {k:8.4f}   (beta fixed at 0.5)")
    _check("KL is strictly increasing in the reward's SCALE at fixed beta",
           all(b > a for a, b in zip(kls, kls[1:])))
    _check("and the change is large, not a rounding effect", kls[-1] > 2.0 * kls[0],
           f"{kls[-1]:.4f} vs {kls[0]:.4f}")
    print("  ... => only r/beta is identified by the policy. Chain it with `cap`:\n"
          "      agreement rate -> scale of r -> meaning of beta -> the KL you land at.\n"
          "      beta is a property of your ANNOTATION PROCESS.")


def stage_plackett(ctx: Ctx) -> None:
    """plackett_luce_nll -- K=2 reduces to Bradley-Terry exactly"""
    rng = np.random.default_rng(4)
    r = rng.standard_normal(ctx.N_ITEMS)
    n = ctx.N_ITEMS
    rank2 = [[a, b] for a in range(n) for b in range(n) if a != b]
    pl = plackett_luce_nll(r, rank2)
    bt = bt_nll_and_grad(r, np.array(rank2))[0]
    _check("PL at K=2 == BT", abs(pl - bt) < 1e-14, f"{pl:.15f} vs {bt:.15f}")

    rank4 = [list(rng.permutation(n)[:4]) for _ in range(50)]
    v4 = plackett_luce_nll(r, rank4)
    _check("PL at K=4 is finite and positive", np.isfinite(v4) and v4 > 0, f"{v4:.6f}")
    _check("PL inherits location invariance",
           abs(plackett_luce_nll(r + 3.7, rank4) - v4) < 1e-12)
    print("  ... => PL is a sequence of softmax choices over the REMAINING items; the\n"
          "      denominator shrinking is the whole model. Same softmax-of-rewards\n"
          "      object as the tilt, and the same constant cancels.")


def stage_ceiling(ctx: Ctx) -> None:
    """bayes_accuracy -- what '96% pairwise accuracy' can and cannot mean"""
    rng = np.random.default_rng(5)
    gold_soft = ctx.gold * 0.45      # finite scale => a genuinely stochastic annotator
    rep = np.repeat(ctx.all_pairs, ctx.dense_rep * 4, axis=0)
    train = noisy_labels(rng, gold_soft, rep, 0.0, stochastic=True)
    held = noisy_labels(rng, gold_soft, rep, 0.0, stochastic=True)

    r, _, _ = fit_bt(train, ctx.N_ITEMS, n_steps=20_000, lr=1.0)
    a_fit = pairwise_accuracy(r, held)
    a_gold = pairwise_accuracy(gold_soft, held)
    ceil = bayes_accuracy(gold_soft, held)
    se = float(np.sqrt(ceil * (1 - ceil) / len(held)))

    print(f"      bayes ceiling     = {ceil:.4f}   (n = {len(held)}, standard error {se:.4f})")
    print(f"      GOLD reward's acc = {a_gold:.4f}")
    print(f"      fitted RM acc     = {a_fit:.4f}")
    _check("the ceiling is in (0.5, 1)", 0.5 < ceil < 1.0)
    _check("the GOLD reward itself sits at the ceiling (it IS the Bayes rule)",
           abs(a_gold - ceil) <= 3 * se, f"|{a_gold:.4f} - {ceil:.4f}| vs 3 SE = {3*se:.4f}")
    _check("the fitted RM cannot beat the gold reward", a_fit <= a_gold + 1e-9,
           f"{a_fit:.4f} vs {a_gold:.4f}")
    _check("a deterministic annotator has ceiling 1.0",
           abs(bayes_accuracy(ctx.gold * 50.0, ctx.all_pairs) - 1.0) < 1e-9)
    print(f"  ... => {ceil:.1%} is the ceiling here NO MATTER the model. Accuracy also depends\n"
          "      only on the SIGN of the margin, so it is invariant to scale and cannot\n"
          "      distinguish a converged fit from one whose norm is running to infinity.\n"
          "      A high reported accuracy is evidence of near-deterministic labels --\n"
          "      which is the divergent regime. It is the alarm, not the reassurance.")


STAGES = [
    ("bt_grad", stage_bt_grad),
    ("graph", stage_graph),
    ("existence", stage_existence),
    ("direction", stage_direction),
    ("matching", stage_matching),
    ("cap", stage_cap),
    ("scale", stage_scale),
    ("plackett", stage_plackett),
    ("ceiling", stage_ceiling),
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
