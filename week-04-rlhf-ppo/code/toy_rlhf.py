"""
Week 4 Day 1 -- the InstructGPT pipeline, the Bradley-Terry reward model, and the
token-level environment that the whole week runs on.

This file is INFRASTRUCTURE. Days 2-5 import `TokenEnv`, `enumerate_sequences`,
`sequence_logprob`, `exact_J` and `exact_sequence_kl` from it, so a bug here is a
bug in four more days of results. Its checks are correspondingly paranoid.

Week 3's discipline was: build a setting with EXACT ground truth (a finite MDP, T
Bellman backups, central finite differences), so that "this estimator is unbiased"
is an ASSERTION and not a claim. Week 4 keeps the discipline and changes the
setting: a vocabulary with a real EOS token, variable-length sequences, sequence
level reward, and token-level log-probabilities -- but small enough that every
terminating sequence can be ENUMERATED. Enumeration is this week's `exact_J`.

  run everything      ./.venv/bin/python ai-lab-mle-prep/week-04-rlhf-ppo/code/toy_rlhf.py
  one stage           ...  toy_rlhf.py exact_j
  a few stages        ...  toy_rlhf.py enumerate logprob
  small/fast fixtures ...  toy_rlhf.py --fast
  list the stages     ...  toy_rlhf.py --list

Build order (each group is runnable the moment it is written):

  1. enumerate_sequences                      -> stage `enumerate`
  2. sequence_logprob                         -> stage `logprob`   (sum-to-one)
  3. exact_J                                  -> stage `exact_j`
  4. exact_sequence_kl                        -> stage `seq_kl`
  5. tilted_sequence_dist                     -> stage `tilt`
  6. bt_loss_and_grad                         -> stage `bt_grad`
  7. fit_reward_model + sample_preferences    -> stages `identifiable`, `noise`
  8. (nothing new)                            -> stages `validity`, `starvation`

Steps 1-5 are the week's infrastructure; 6-8 are today's reward-model result.
Full run is ~11 s, `--fast` ~3 s.
"""

from __future__ import annotations

import sys
import traceback
from functools import cached_property

import numpy as np

# ---------------------------------------------------------------------------
# Numerics helpers (provided -- same idioms as W1/W3)
# ---------------------------------------------------------------------------


def log_softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-softmax that tolerates -inf entries (forbidden actions)."""
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    shifted = z - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax_np(z, axis=axis))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable logistic, branchless. exp(-|x|) never overflows."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


# ---------------------------------------------------------------------------
# The environment (provided -- READ IT, you build against this contract)
# ---------------------------------------------------------------------------


class TokenEnv:
    """A token-level LM environment small enough to enumerate exhaustively.

    Vocabulary is `vocab` tokens, the last of which (`eos = vocab - 1`) is EOS.
    A state is the tuple of tokens emitted so far, containing NO eos and of
    length < t_max -- i.e. exactly the prefixes from which a decision is still
    pending. Dynamics are a deterministic append (W3 D2's "all the randomness
    lives in pi_theta"). An episode terminates when eos is emitted OR when the
    length reaches t_max (truncation).

    The policy is a tabular softmax over logits `theta` of shape
    (n_states, vocab): row `state_id(prefix)` holds the logits at that prefix.

    Default size: vocab=4 (tokens 0,1,2 + eos=3), t_max=4
      -> n_states = 1 + 3 + 9 + 27 = 40
      -> 40 eos-terminated sequences + 3**4 = 81 truncated = 121 total.
    """

    def __init__(self, vocab: int = 4, t_max: int = 4):
        assert vocab >= 2 and t_max >= 1
        self.vocab = vocab
        self.t_max = t_max
        self.eos = vocab - 1

        states: list[tuple[int, ...]] = [()]
        frontier: list[tuple[int, ...]] = [()]
        for _ in range(t_max - 1):
            nxt = [p + (a,) for p in frontier for a in range(vocab - 1)]
            states.extend(nxt)
            frontier = nxt
        self.states = states
        self._index = {s: i for i, s in enumerate(states)}
        self.n_states = len(states)

    def state_id(self, prefix: tuple[int, ...]) -> int:
        return self._index[prefix]

    def random_theta(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        return scale * rng.standard_normal((self.n_states, self.vocab))

    def __repr__(self) -> str:
        return f"TokenEnv(vocab={self.vocab}, t_max={self.t_max}, n_states={self.n_states})"


def sample_sequences(env: TokenEnv, theta: np.ndarray, n: int,
                     rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Monte-Carlo rollouts from the tabular policy (provided).

    Structurally independent of `enumerate_sequences` / `exact_J` -- it walks the
    tree forward and never consults the enumeration. That independence is the
    whole point (W3 D4: keep the slow reference from being able to agree with the
    fast one by construction).
    """
    pi = softmax_np(theta)
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


def state_occupancy(env: TokenEnv, theta: np.ndarray) -> np.ndarray:
    """d(s) = P_theta(the rollout reaches prefix s with a decision still pending).

    Provided, and it is the INDEPENDENT route to the sequence KL: because
    every state in this env has a unique time index (its prefix length),
    W3 D5's `sum_t E_{d_t}[KL_t]` collapses to `sum_s d(s) KL(pi(.|s)||pi_ref(.|s))`.
    Stage `seq_kl` cross-checks your enumeration against this.

    D5's actual bug was logging the UNWEIGHTED per-state mean KL (1.053) instead
    of the occupancy-weighted sequence KL (5.056), and 94/94 passed anyway
    because every assertion only tested monotonicity in beta. The tell was that
    a provided helper was never called. So: this helper exists, and if you finish
    the file without calling it, that is a bug report about your own code.
    """
    pi = softmax_np(theta)
    d = np.zeros(env.n_states)
    d[env.state_id(())] = 1.0
    for s_id, prefix in enumerate(env.states):
        if len(prefix) + 1 >= env.t_max:
            continue
        for a in range(env.vocab - 1):          # eos terminates, no successor state
            d[env.state_id(prefix + (a,))] += d[s_id] * pi[s_id, a]
    return d


# ---------------------------------------------------------------------------
# Exercises (TODO bodies)
# ---------------------------------------------------------------------------


def enumerate_sequences(env: TokenEnv) -> list[tuple[int, ...]]:
    """Every terminating token sequence in `env`, as a list of tuples.

    A sequence terminates when it emits eos, or when it reaches length t_max.
    Order must be DETERMINISTIC: the reward model below is a table indexed by
    position in this list, so a reordering silently reindexes every reward.

    The likeliest bug in this whole file is here: dropping the eos branch at
    depth t_max - 1, or double counting a prefix. Stage `logprob` catches both
    by asserting that the probabilities of everything you return sum to
    EXACTLY 1 under an arbitrary theta -- a normalization identity, which only
    holds if you have enumerated a partition of the sample space.

    For the default env this must return 121 sequences.
    """
    sequence = []
    frontier = [()]

    for _ in range(env.t_max):
        frontier_new = []
        for p in frontier:
            for a in range(env.vocab-1):
                frontier_new.append(p + (a,))
            sequence.append(p+(env.vocab-1,))
            frontier = frontier_new
    return sequence + frontier


def sequence_logprob(env: TokenEnv, theta: np.ndarray, seq: tuple[int, ...]) -> float:
    """log p_theta(seq) = sum_t log pi_theta(a_t | s_t), accumulated over tokens.

    `s_t` is the prefix `seq[:t]`; use `env.state_id` to find its row. This is
    the same `log_softmax` -> pick-the-taken-token idiom as W1's token log-probs
    and W3's `episode_logprobs`, now over a tree instead of a fixed-length batch.

    Note there is NO separate "probability of stopping" factor: emitting eos is
    an ACTION with its own logit, and truncation at t_max is forced (probability
    1 given you got there). If you find yourself adding a stop probability, the
    sum-to-one check will read > 1.
    """
    log_pi = log_softmax_np(theta)
    log_pi_total = 0
    
    for t in range(len(seq)):
        prefix = seq[:t]
        row = env.state_id(prefix)
        log_pi_total += log_pi[row, seq[t]]
    return log_pi_total




def exact_J(env: TokenEnv, theta: np.ndarray, reward_fn) -> float:
    """J(theta) = sum over ALL terminating sequences of p_theta(y) * r(y).

    THE WEEK'S GROUND TRUTH. No sampling, no Monte Carlo, no confidence
    interval -- an exact expectation by enumeration, which is what lets Days 2-5
    assert unbiasedness instead of plotting it.

    `reward_fn(seq) -> float` takes a sequence tuple and returns its scalar
    reward. Terminal reward only, which is the RLHF case: one number per
    completed response.
    """
    sequences = enumerate_sequences(env)
    J = 0
    for seq in sequences:
        seq_logprob = sequence_logprob(env, theta, seq)
        reward = reward_fn(seq)
        J += reward * np.exp(seq_logprob)
    return J


def exact_sequence_kl(env: TokenEnv, theta: np.ndarray, theta_ref: np.ndarray) -> float:
    """KL( p_theta(y) || p_ref(y) ) over the full sequence distribution, exactly.

        sum_y p_theta(y) [ log p_theta(y) - log p_ref(y) ]

    This is the quantity beta multiplies in InstructGPT Eq. (2), in nats per
    COMPLETION. Stage `seq_kl` checks it three ways: exactly 0.0 when the two
    thetas are the same object, non-negative always, and -- the one with teeth --
    equal to the occupancy-weighted per-state sum from `state_occupancy`, which
    reaches the same number by a completely different route.
    """
    sequences = enumerate_sequences(env)
    KL_exact = 0

    for seq in sequences:
        seq_logprob = sequence_logprob(env, theta, seq)
        seq_logprob_ref = sequence_logprob(env, theta_ref, seq)
        KL_exact += np.exp(seq_logprob) * (seq_logprob - seq_logprob_ref)
    return KL_exact



def tilted_sequence_dist(env: TokenEnv, theta_ref: np.ndarray, rewards: np.ndarray,
                         beta: float) -> np.ndarray:
    """W3 D5's closed form, lifted to whole SEQUENCES:

        p*(y) proportional to  p_ref(y) * exp( r(y) / beta )

    `rewards` is a vector aligned with `enumerate_sequences(env)`. Returns a
    probability vector over those sequences.

    Two things this makes concrete that the per-token version could not:
      * the "action" here really is an entire completion, so this normalizer is
        the intractable Z of the DPO derivation -- except that here we CAN sum
        over all |V|^T sequences, which is the only reason the week has ground
        truth at all.
      * compute it in LOG SPACE. `p_ref * exp(r/beta)` overflows to inf for
        small beta and normalizes to nan (the D5 bug, one level up).
    """
    sequences = enumerate_sequences(env)
    logprob_ref = np.array([sequence_logprob(env, theta_ref, seq) for seq in sequences])
    p_opt_log = logprob_ref + rewards/beta
    return softmax_np(p_opt_log)

def bt_loss_and_grad(psi: np.ndarray, pairs: np.ndarray) -> tuple[float, np.ndarray]:
    """Bradley-Terry negative log-likelihood and its ANALYTIC gradient.

        L(psi) = mean over pairs of  -log sigmoid( psi[p, w] - psi[p, l] )

    psi   : (n_prompts, n_items) tabular reward model.
    pairs : (n_pairs, 3) int array of rows (prompt, winner_item, loser_item).
            Every comparison is WITHIN one prompt -- that is the hypothesis the
            identifiability theorem consumes.

    Returns (loss, grad) with grad.shape == psi.shape.

    Two traps, both of which the finite-difference check will find:
      * use `np.logaddexp(0, -delta)` for the loss, not `log(1 + exp(-delta))`,
        which overflows for delta very negative.
      * the gradient is a SCATTER-ADD. Items repeat across pairs, and
        `grad[p, w] -= g` with fancy indexing does NOT accumulate on repeated
        indices -- it writes the last one and silently discards the rest. Use
        `np.add.at`. (W3's indexing rule, in its most dangerous form: this bug
        makes the gradient plausible-looking but wrong.)
    """
    p, w, l = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    delta = psi[p, w] - psi[p, l]
    loss = np.logaddexp(0, -delta).mean()
    grad = np.zeros_like(psi)
    g = sigmoid(-delta)/len(delta)
    np.add.at(grad, (p, w), -g)
    np.add.at(grad, (p, l), g)
    return (loss, grad)


def fit_reward_model(psi0: np.ndarray, pairs: np.ndarray, lr: float = 0.5,
                     n_steps: int = 2000) -> np.ndarray:
    """Fit the tabular RM by plain gradient descent from `psi0`. Returns psi.

    Do not center, clip, or normalize anything. The point of stage `identifiable`
    is to observe what this fit does and does not pin down, so any tidying up
    here would hide the day's headline.
    """
    psi = psi0.copy()
    for step in range(n_steps):
        _, grad = bt_loss_and_grad(psi, pairs)
        psi -= lr * grad

    return psi



def sample_preferences(rng: np.random.Generator, gold_r: np.ndarray,
                       sampler: np.ndarray, n_pairs: int,
                       noise: float = 0.0) -> np.ndarray:
    """Draw labelled comparison pairs -- the stage-2 data collection protocol.

    gold_r  : (n_prompts, n_items) the TRUE reward, which the RM never sees.
    sampler : (n_prompts, n_items) the distribution completions are drawn from.
              This is pi_ref, and it is the reason RM error is concentrated off
              the SFT distribution: an item with tiny `sampler` mass appears in
              almost no comparison and is therefore almost unconstrained.
    noise   : annotator error rate. Label by a Bradley-Terry draw on the gold
              difference, THEN flip the label with probability `noise`.
              Annotator disagreement is a parameter here, not an afterthought:
              real inter-annotator agreement on preference data runs in the 70s%.

    Returns (n_pairs, 3) rows of (prompt, winner_item, loser_item). Draw the
    prompt uniformly and the two items from `sampler` WITHOUT replacement.
    """
    n_prompts, n_items = gold_r.shape[0], gold_r.shape[1]

    prompts = rng.choice(n_prompts, n_pairs)
    a, b = np.zeros(n_pairs, dtype=np.long), np.zeros(n_pairs, dtype=np.long)
    for i, prompt in enumerate(prompts):
        a[i], b[i] = rng.choice(n_items, 2, replace=False, p=sampler[prompt])
    delta = gold_r[prompts, a] - gold_r[prompts, b]
    prob_bt = sigmoid(delta)
    rand = rng.uniform(0, 1, n_pairs)
    labels_bt = np.where(rand < prob_bt, 1, -1)
    rand2 = rng.uniform(0, 1, n_pairs)
    labels_noisy = np.where(rand2>noise, labels_bt, -labels_bt)

    labeled_pairs = np.zeros((n_pairs, 3), dtype=np.long)
    labeled_pairs[:, 0] = prompts
    winner = np.where(labels_noisy>0, a, b)
    loser = np.where(labels_noisy>0, b, a)
    labeled_pairs[:, 1] =winner
    labeled_pairs[:, 2] =loser

    return labeled_pairs


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


def _gold_sequence_reward(env: TokenEnv):
    """A structured sequence-level reward: token 0 is 'good', length costs, and
    terminating cleanly with eos pays a bonus. Deterministic in the sequence,
    which is the RLHF case (the RM scores a finished response)."""
    def r(seq: tuple[int, ...]) -> float:
        n_good = sum(1 for a in seq if a == 0)
        ends_clean = 1.0 if seq and seq[-1] == env.eos else 0.0
        return 2.0 * n_good - 0.4 * len(seq) + 0.75 * ends_clean
    return r


class Ctx:
    """Lazily-built fixtures, each from its OWN seeded generator so a stage's
    data does not depend on which other stages ran first."""

    def __init__(self, fast: bool = False):
        self.fast = fast
        self.env = TokenEnv(vocab=4, t_max=4)
        self.n_mc = 12_000 if not fast else 3_000
        self.n_pairs = 8_000 if not fast else 2_500
        # NOT reduced by --fast: gradient-descent STEPS buy convergence, and an
        # unconverged reward model breaks the stages' claims for a reason that
        # has nothing to do with the claims. --fast shrinks DATA and loosens
        # TOLERANCES; it must not turn a true assertion false.
        self.n_steps = 1_500
        self.n_seeds = 3
        # The BT loss is a MEAN over pairs, so each item's gradient is O(1/n_items)
        # no matter how much data there is -- a "safe-looking" lr of 0.5 simply
        # does not converge here. The Hessian's top eigenvalue is ~0.02, so the
        # stable range is lr < ~100 and 8.0 is still very conservative.
        self.lr = 8.0

    # -- policy-side fixtures ------------------------------------------------

    @cached_property
    def theta(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(0), scale=1.2)

    @cached_property
    def theta_ref(self) -> np.ndarray:
        return self.env.random_theta(np.random.default_rng(1), scale=0.8)

    @cached_property
    def gold_r(self):
        return _gold_sequence_reward(self.env)

    @cached_property
    def seqs(self) -> list[tuple[int, ...]]:
        return enumerate_sequences(self.env)

    @cached_property
    def gold_vec(self) -> np.ndarray:
        return np.array([self.gold_r(s) for s in self.seqs])

    # -- reward-model-side fixtures ------------------------------------------

    @cached_property
    def bt_world(self):
        """A multi-PROMPT world for the identifiability experiment.

        Per-prompt LEVELS are deliberately spread far apart (that is the
        unidentifiable direction) while within-prompt differences are O(1)
        (that is the identifiable part).
        """
        rng = np.random.default_rng(7)
        n_prompts, n_items = 6, 8
        levels = rng.uniform(-8.0, 8.0, size=(n_prompts, 1))
        fine = rng.standard_normal((n_prompts, n_items)) * 1.2
        gold = levels + fine
        sampler = np.full((n_prompts, n_items), 1.0 / n_items)
        psi0 = rng.standard_normal((n_prompts, n_items)) * 3.0
        psi0_b = rng.standard_normal((n_prompts, n_items)) * 3.0
        return dict(gold=gold, sampler=sampler, psi0=psi0, psi0_b=psi0_b,
                    n_prompts=n_prompts, n_items=n_items, levels=levels)


def _fail_fast(ctx: Ctx) -> None:
    """Touch the cheap functions a heavy stage needs, so a missing TODO surfaces
    before we pay for a 20k-rollout fixture."""
    tiny = TokenEnv(vocab=2, t_max=1)
    enumerate_sequences(tiny)
    sequence_logprob(tiny, np.zeros((tiny.n_states, tiny.vocab)), (0,))


# ---- individual stages ----------------------------------------------------


def stage_enumerate(ctx: Ctx) -> None:
    """enumerate_sequences -- counts, structure, determinism"""
    env = ctx.env
    seqs = enumerate_sequences(env)

    _check("returns a list of tuples", isinstance(seqs, list) and all(isinstance(s, tuple) for s in seqs))
    _check("no duplicates", len(set(seqs)) == len(seqs), f"{len(seqs)} returned, {len(set(seqs))} unique")
    _check("121 terminating sequences for vocab=4, t_max=4", len(seqs) == 121, f"got {len(seqs)}")

    n_eos = sum(1 for s in seqs if s and s[-1] == env.eos)
    n_trunc = sum(1 for s in seqs if len(s) == env.t_max and env.eos not in s)
    _check("40 eos-terminated (one per state)", n_eos == env.n_states, f"got {n_eos}")
    _check("81 truncated (3**4)", n_trunc == 81, f"got {n_trunc}")
    _check("every sequence is eos-terminated or exactly t_max long",
           all((s[-1] == env.eos) or len(s) == env.t_max for s in seqs))
    _check("eos appears only as the final token",
           all(env.eos not in s[:-1] for s in seqs))
    _check("every proper prefix is a valid state",
           all(s[:t] in env._index for s in seqs for t in range(len(s))))
    _check("order is deterministic across calls", enumerate_sequences(env) == seqs)

    tiny = TokenEnv(vocab=3, t_max=1)
    _check("t_max=1 reduces to a one-step bandit (V sequences)",
           sorted(enumerate_sequences(tiny)) == [(0,), (1,), (2,)],
           f"got {sorted(enumerate_sequences(tiny))}")


def stage_logprob(ctx: Ctx) -> None:
    """sequence_logprob -- hand case, and the sum-to-one normalization identity"""
    env = ctx.env

    zeros = np.zeros((env.n_states, env.vocab))
    lp = sequence_logprob(env, zeros, (0, 1, env.eos))
    _check("uniform logits: 3 tokens => log(1/4**3)", np.isclose(lp, 3 * np.log(0.25)),
           f"got {lp:.6f} want {3 * np.log(0.25):.6f}")
    _check("returns a float scalar", np.isscalar(lp) or np.asarray(lp).ndim == 0)

    theta = ctx.theta
    seqs = ctx.seqs
    logps = np.array([sequence_logprob(env, theta, s) for s in seqs])
    total = float(np.exp(logps).sum())
    _check("probabilities of all terminating sequences sum to EXACTLY 1",
           abs(total - 1.0) < 1e-12, f"got {total:.15f}")

    theta2 = ctx.theta_ref
    total2 = float(np.exp([sequence_logprob(env, theta2, s) for s in seqs]).sum())
    _check("sum-to-one holds under a second, unrelated theta",
           abs(total2 - 1.0) < 1e-12, f"got {total2:.15f}")

    _check("all log-probs are finite and negative", np.all(np.isfinite(logps)) and np.all(logps < 0))

    # A hand-checkable factorization: p(prefix then eos) = p(prefix) * pi(eos|prefix)
    pi = softmax_np(theta)
    lp_pref = sequence_logprob(env, theta, (2, 0, env.eos))
    lp_manual = (np.log(pi[env.state_id(()), 2]) + np.log(pi[env.state_id((2,)), 0])
                 + np.log(pi[env.state_id((2, 0)), env.eos]))
    _check("matches an explicit three-factor product", np.isclose(lp_pref, lp_manual),
           f"{lp_pref:.8f} vs {lp_manual:.8f}")


def stage_exact_j(ctx: Ctx) -> None:
    """exact_J -- against a large Monte-Carlo rollout, plus a bandit hand case"""
    _fail_fast(ctx)
    env, theta = ctx.env, ctx.theta

    j = exact_J(env, theta, ctx.gold_r)

    # Hand case: t_max=1 is a one-step bandit, J = sum_a pi(a) r(a).
    bandit = TokenEnv(vocab=3, t_max=1)
    bt = np.array([[0.5, -1.0, 2.0]])
    pi_b = softmax_np(bt)[0]
    want = float(sum(pi_b[a] * (1.0 * a) for a in range(3)))
    got = exact_J(bandit, bt, lambda s: 1.0 * s[0])
    _check("t_max=1 bandit matches sum_a pi(a) r(a) by hand", np.isclose(got, want),
           f"{got:.10f} vs {want:.10f}")

    # Monte Carlo, structurally independent of the enumeration.
    rng = np.random.default_rng(11)
    rolls = sample_sequences(env, theta, ctx.n_mc, rng)
    rs = np.array([ctx.gold_r(s) for s in rolls])
    mc, se = float(rs.mean()), float(rs.std(ddof=1) / np.sqrt(len(rs)))
    print(f"  ... exact_J = {j:.6f}   MC({ctx.n_mc}) = {mc:.6f} +/- {se:.6f}"
          f"   z = {(mc - j) / se:+.2f}")
    _check("exact_J agrees with Monte Carlo within 4 standard errors",
           abs(mc - j) < 4 * se, f"|{mc - j:.5f}| vs 4se = {4 * se:.5f}")

    # The sign of the discrepancy must be random over seeds -- a systematic
    # offset means the enumeration is missing sequences, which MC would not be.
    zs = []
    for seed in range(2, 2 + (3 if ctx.fast else 5)):
        r2 = np.array([ctx.gold_r(s) for s in
                       sample_sequences(env, theta, ctx.n_mc // 2, np.random.default_rng(seed))])
        zs.append((r2.mean() - j) / (r2.std(ddof=1) / np.sqrt(len(r2))))
    print(f"  ... z-scores across seeds: {np.round(zs, 2)}")
    _check("MC discrepancy is not systematic (both signs appear, |mean z| < 2)",
           abs(float(np.mean(zs))) < 2.0, f"mean z = {np.mean(zs):+.2f}")

    # J is linear in the reward: a constant reward returns that constant exactly.
    _check("a constant reward gives exactly that constant (mass sums to 1)",
           np.isclose(exact_J(env, theta, lambda s: 3.25), 3.25, atol=1e-12),
           f"got {exact_J(env, theta, lambda s: 3.25):.15f}")


def stage_seq_kl(ctx: Ctx) -> None:
    """exact_sequence_kl -- zero at identity, non-negative, and the occupancy cross-check"""
    _fail_fast(ctx)
    env, theta, theta_ref = ctx.env, ctx.theta, ctx.theta_ref

    same = exact_sequence_kl(env, theta, theta)
    _check("EXACTLY 0.0 when theta is theta_ref", same == 0.0 or abs(same) < 1e-14,
           f"got {same!r}")

    kl = exact_sequence_kl(env, theta, theta_ref)
    _check("non-negative", kl > 0.0, f"got {kl:.6f}")
    _check("asymmetric (reverse != forward)",
           not np.isclose(kl, exact_sequence_kl(env, theta_ref, theta), rtol=1e-3),
           f"fwd {kl:.5f} rev {exact_sequence_kl(env, theta_ref, theta):.5f}")

    # The independent route: sum_s d(s) KL(pi(.|s) || pi_ref(.|s)).
    d = state_occupancy(env, theta)
    lp, lpr = log_softmax_np(theta), log_softmax_np(theta_ref)
    per_state = np.sum(np.exp(lp) * (lp - lpr), axis=-1)
    occ = float(np.sum(d * per_state))
    unweighted = float(np.mean(per_state))
    print(f"  ... enumerated {kl:.8f}   occupancy-weighted {occ:.8f}"
          f"   (unweighted per-state mean would be {unweighted:.8f})")
    _check("matches the occupancy-weighted per-state sum", np.isclose(kl, occ, rtol=1e-10),
           f"{kl:.10f} vs {occ:.10f}")
    _check("and is NOT the unweighted per-state mean (D5's actual bug)",
           not np.isclose(kl, unweighted, rtol=1e-3),
           f"{kl:.5f} vs {unweighted:.5f}")

    _check("occupancy of the root is 1", np.isclose(d[env.state_id(())], 1.0))

    # Scaling theta_ref toward theta must shrink the KL monotonically.
    kls = [exact_sequence_kl(env, theta, theta + t * (theta_ref - theta))
           for t in (0.25, 0.5, 0.75, 1.0)]
    print(f"  ... KL along the path to theta_ref: {np.round(kls, 5)}")
    _check("KL increases monotonically along the path to theta_ref",
           all(b > a for a, b in zip(kls, kls[1:])), f"{np.round(kls, 5)}")


def stage_tilt(ctx: Ctx) -> None:
    """tilted_sequence_dist -- the D5 closed form at sequence level"""
    _fail_fast(ctx)
    env, theta_ref, seqs = ctx.env, ctx.theta_ref, ctx.seqs
    gold = ctx.gold_vec
    p_ref = np.array([np.exp(sequence_logprob(env, theta_ref, s)) for s in seqs])

    p = tilted_sequence_dist(env, theta_ref, gold, beta=1.0)
    _check("returns a probability vector over the enumerated sequences",
           p.shape == (len(seqs),) and np.all(p >= 0) and np.isclose(p.sum(), 1.0),
           f"shape {p.shape}, sum {p.sum():.12f}")

    big = tilted_sequence_dist(env, theta_ref, gold, beta=200.0)
    _check("beta -> inf recovers p_ref", np.allclose(big, p_ref, atol=2e-2),
           f"max|diff| {np.max(np.abs(big - p_ref)):.4f}")

    small = tilted_sequence_dist(env, theta_ref, gold, beta=1e-3)
    _check("beta -> 0 concentrates on the argmax WITHIN the support",
           np.isclose(small[int(np.argmax(gold))], 1.0, atol=1e-6),
           f"mass on argmax = {small[int(np.argmax(gold))]:.8f}")
    _check("no overflow at beta = 1e-3 (log-space required)", np.all(np.isfinite(small)))

    shifted = tilted_sequence_dist(env, theta_ref, gold + 17.0, beta=1.0)
    _check("invariant to r -> r + c (Z absorbs it)", np.allclose(p, shifted, atol=1e-12),
           f"max|diff| {np.max(np.abs(p - shifted)):.2e}")

    e_ref = float(p_ref @ gold)
    es = [float(tilted_sequence_dist(env, theta_ref, gold, b) @ gold) for b in (4, 2, 1, 0.5)]
    print(f"  ... E[r]: ref {e_ref:.4f} -> {np.round(es, 4)} at beta 4,2,1,0.5")
    _check("expected reward rises monotonically as beta falls",
           all(b > a for a, b in zip([e_ref] + es, es)), f"{np.round(es, 4)}")


def stage_bt_grad(ctx: Ctx) -> None:
    """bt_loss_and_grad -- finite differences, log 2 at a tie, scatter-add correctness"""
    psi = np.array([[0.0, 0.0, 1.0], [2.0, -1.0, 0.5]])
    tie = np.array([[0, 0, 1]])
    loss, grad = bt_loss_and_grad(psi, tie)
    _check("r_w == r_l gives loss exactly log 2", np.isclose(loss, np.log(2.0)),
           f"got {loss:.10f}")
    _check("grad has psi's shape", grad.shape == psi.shape, f"got {grad.shape}")
    _check("at a tie the gradient magnitude is exactly 1/2",
           np.isclose(grad[0, 0], -0.5) and np.isclose(grad[0, 1], 0.5),
           f"got {grad[0, 0]:.6f}, {grad[0, 1]:.6f}")
    _check("untouched prompts get zero gradient", np.allclose(grad[1], 0.0))
    _check("gradient sums to zero WITHIN each prompt (the free direction)",
           np.allclose(grad.sum(axis=1), 0.0), f"row sums {grad.sum(axis=1)}")

    # Repeated indices: item 0 wins twice and loses once. Fancy-index assignment
    # would silently keep only one contribution.
    rng = np.random.default_rng(3)
    psi = rng.standard_normal((3, 5))
    pairs = np.array([[0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 3, 4], [1, 3, 0], [2, 2, 2]])
    loss, grad = bt_loss_and_grad(psi, pairs)

    fd = np.zeros_like(psi)
    eps = 1e-6
    for i in range(psi.shape[0]):
        for j in range(psi.shape[1]):
            up, dn = psi.copy(), psi.copy()
            up[i, j] += eps
            dn[i, j] -= eps
            fd[i, j] = (bt_loss_and_grad(up, pairs)[0] - bt_loss_and_grad(dn, pairs)[0]) / (2 * eps)
    err = float(np.max(np.abs(grad - fd)))
    print(f"  ... max|analytic - central FD| = {err:.3e}")
    _check("analytic gradient matches central finite differences", err < 1e-7, f"{err:.3e}")

    big = np.array([[0.0, 40.0]])
    l_big, g_big = bt_loss_and_grad(big, np.array([[0, 0, 1]]))
    _check("no overflow when the model is confidently WRONG (delta = -40)",
           np.isfinite(l_big) and np.isclose(l_big, 40.0, rtol=1e-6), f"loss {l_big}")
    _check("and the gradient saturates at magnitude 1", np.isclose(g_big[0, 0], -1.0, atol=1e-9),
           f"got {g_big[0, 0]:.10f}")

    l_right, g_right = bt_loss_and_grad(np.array([[40.0, 0.0]]), np.array([[0, 0, 1]]))
    _check("a confidently CORRECT pair contributes ~nothing (sigma(-delta) -> 0)",
           abs(g_right[0, 0]) < 1e-15, f"|grad| {abs(g_right[0, 0]):.3e}")


def stage_identifiable(ctx: Ctx) -> None:
    """fit_reward_model -- the per-prompt identifiability headline + negative control"""
    _fail_fast(ctx)
    bt_loss_and_grad(np.zeros((1, 2)), np.array([[0, 0, 1]]))
    w = ctx.bt_world
    gold, sampler, psi0 = w["gold"], w["sampler"], w["psi0"]
    rng = np.random.default_rng(21)
    pairs = sample_preferences(rng, gold, sampler, ctx.n_pairs, noise=0.0)
    psi = fit_reward_model(psi0, pairs, lr=ctx.lr, n_steps=ctx.n_steps)

    _check("psi keeps its shape", psi.shape == gold.shape, f"got {psi.shape}")

    raw_err = float(np.max(np.abs(psi - gold)))
    print(f"  ... max|psi - gold| RAW          = {raw_err:.4f}")
    _check("recovery is NOT exact -- the level is unidentifiable", raw_err > 2.0,
           f"got {raw_err:.4f}")

    ctr = lambda a: a - a.mean(axis=1, keepdims=True)
    cen_err = float(np.max(np.abs(ctr(psi) - ctr(gold))))
    tol = 1.2 if ctx.fast else 0.6
    print(f"  ... max|psi - gold| PER-PROMPT CENTERED = {cen_err:.4f}")
    _check("but per-prompt centered recovery is tight", cen_err < tol,
           f"got {cen_err:.4f}, tol {tol}")
    _check("centering buys at least 5x", raw_err / max(cen_err, 1e-9) > 5.0,
           f"ratio {raw_err / max(cen_err, 1e-9):.1f}x")

    # The sharpest form of the theorem: the per-prompt MEAN is exactly the
    # initialization's, untouched by any amount of data, because every gradient
    # contribution is +g to one item and -g to another within the same prompt.
    drift = float(np.max(np.abs(psi.mean(axis=1) - psi0.mean(axis=1))))
    print(f"  ... max drift of the per-prompt mean from its INIT = {drift:.3e}")
    _check("the per-prompt mean is exactly the init -- data cannot move it", drift < 1e-8,
           f"got {drift:.3e}")

    # Negative control: uncentered cross-prompt comparison is not usable.
    def acc(a, b, cross):
        rg = np.random.default_rng(5)
        n = 4000
        p1 = rg.integers(0, gold.shape[0], n)
        p2 = rg.integers(0, gold.shape[0], n) if cross else p1
        i1 = rg.integers(0, gold.shape[1], n)
        i2 = rg.integers(0, gold.shape[1], n)
        keep = ~((p1 == p2) & (i1 == i2))
        dg = b[p1, i1] - b[p2, i2]
        dp = a[p1, i1] - a[p2, i2]
        return float(np.mean((np.sign(dg) == np.sign(dp))[keep]))

    within = acc(psi, gold, cross=False)
    cross = acc(psi, gold, cross=True)
    cross_c = acc(ctr(psi), ctr(gold), cross=True)
    print(f"  ... pairwise accuracy: within-prompt {within:.3f} | "
          f"cross-prompt RAW {cross:.3f} | cross-prompt after centering both {cross_c:.3f}")
    _check("within-prompt ranking is essentially perfect",
           within > (0.90 if ctx.fast else 0.95), f"{within:.3f}")
    _check("cross-prompt ranking with RAW scores is near chance or worse",
           cross < 0.65, f"{cross:.3f}")
    _check("and it is far worse than the within-prompt number", within - cross > 0.25,
           f"gap {within - cross:.3f}")

    # The theorem as a controlled experiment: refit from a DIFFERENT init on the
    # SAME data. Anything the data identifies must agree; anything it does not
    # is free to differ. This is deterministic, unlike a correlation over 6
    # prompts (which was the first version of this check, and was junk: the
    # sampling variance of a correlation at n=6 is enormous).
    psi_b = fit_reward_model(w["psi0_b"], pairs, lr=ctx.lr, n_steps=ctx.n_steps)
    d_raw = float(np.max(np.abs(psi - psi_b)))
    d_cen = float(np.max(np.abs(ctr(psi) - ctr(psi_b))))
    print(f"  ... refit from another init: max|raw diff| {d_raw:.4f}  "
          f"max|centered diff| {d_cen:.2e}")
    _check("two inits give materially different RAW scores", d_raw > 1.0, f"{d_raw:.4f}")
    _check("but IDENTICAL within-prompt differences (that is what the data pins down)",
           d_cen < 5e-3, f"{d_cen:.3e}")
    print(f"  ... mean RM score per prompt = {np.round(psi.mean(axis=1), 2)}")
    print(f"  ... mean gold      per prompt = {np.round(gold.mean(axis=1), 2)}")
    print("  ... => averaging raw RM score over a mixed prompt set measures the "
          "prompt mix (and your init)")


def stage_noise(ctx: Ctx) -> None:
    """sample_preferences -- annotator-noise sweep, >=3 seeds, mean +/- spread"""
    _fail_fast(ctx)
    bt_loss_and_grad(np.zeros((1, 2)), np.array([[0, 0, 1]]))
    w = ctx.bt_world
    gold, sampler, psi0 = w["gold"], w["sampler"], w["psi0"]
    ctr = lambda a: a - a.mean(axis=1, keepdims=True)

    pairs = sample_preferences(np.random.default_rng(1), gold, sampler, 500, noise=0.0)
    _check("returns (n_pairs, 3) integers", pairs.shape == (500, 3)
           and np.issubdtype(pairs.dtype, np.integer), f"{pairs.shape}, {pairs.dtype}")
    _check("winner and loser are always distinct items",
           np.all(pairs[:, 1] != pairs[:, 2]))
    _check("indices are in range", pairs[:, 0].max() < gold.shape[0]
           and pairs[:, 1:].max() < gold.shape[1])
    _check("every prompt gets sampled", len(np.unique(pairs[:, 0])) == gold.shape[0],
           f"{len(np.unique(pairs[:, 0]))} of {gold.shape[0]}")

    def inverted(noise: float) -> float:
        """Fraction of labels where the winner has the LOWER gold reward."""
        pr = sample_preferences(np.random.default_rng(2), gold, sampler, 4000, noise)
        return float(np.mean(gold[pr[:, 0], pr[:, 1]] < gold[pr[:, 0], pr[:, 2]]))

    lo, hi = inverted(0.0), inverted(0.5)
    _check("at noise=0 the winner usually has the higher gold reward", lo < 0.30,
           f"gold-inverted fraction {lo:.3f}")
    _check("noise=0.5 drives the inverted fraction toward 1/2", hi > lo + 0.10,
           f"{lo:.3f} -> {hi:.3f}")
    _check("and noise=0.5 really is a coin flip", abs(hi - 0.5) < 0.05, f"{hi:.3f}")

    rows = []
    for noise in (0.0, 0.1, 0.25, 0.4):
        accs = []
        for seed in range(ctx.n_seeds):
            rng = np.random.default_rng(100 + seed)
            pr = sample_preferences(rng, gold, sampler, ctx.n_pairs, noise=noise)
            ps = fit_reward_model(psi0, pr, lr=ctx.lr, n_steps=ctx.n_steps)
            a, b = ctr(ps), ctr(gold)
            rg = np.random.default_rng(9)
            n = 4000
            p = rg.integers(0, gold.shape[0], n)
            i1 = rg.integers(0, gold.shape[1], n)
            i2 = rg.integers(0, gold.shape[1], n)
            keep = i1 != i2
            accs.append(float(np.mean((np.sign(b[p, i1] - b[p, i2])
                                       == np.sign(a[p, i1] - a[p, i2]))[keep])))
        rows.append((noise, float(np.mean(accs)), float(np.std(accs))))
        print(f"  ... noise {noise:.2f} -> within-prompt accuracy "
              f"{rows[-1][1]:.4f} +/- {rows[-1][2]:.4f}  (n_seeds={ctx.n_seeds})")

    _check("clean labels recover the within-prompt ranking almost perfectly",
           rows[0][1] > (0.92 if ctx.fast else 0.95), f"{rows[0][1]:.4f}")
    _check("accuracy degrades monotonically with annotator noise",
           all(b[1] <= a[1] + 1e-9 for a, b in zip(rows, rows[1:])),
           f"{[round(r[1], 4) for r in rows]}")
    _check("the degradation beats seed noise (non-overlapping spreads)",
           rows[0][1] - rows[-1][1] > 3 * max(rows[0][2], rows[-1][2], 1e-6),
           f"drop {rows[0][1] - rows[-1][1]:.4f} vs spread {max(rows[0][2], rows[-1][2]):.4f}")


def stage_validity(ctx: Ctx) -> None:
    """RM error grows with sequence KL -- the KL penalty as a confidence region"""
    _fail_fast(ctx)
    bt_loss_and_grad(np.zeros((1, 2)), np.array([[0, 0, 1]]))
    env, theta_ref, seqs = ctx.env, ctx.theta_ref, ctx.seqs
    gold = ctx.gold_vec
    n_seq = len(seqs)

    # Completions for the RM's training set are drawn from pi_ref -- so the RM
    # is constrained where pi_ref has mass, and nowhere else.
    p_ref = np.array([np.exp(sequence_logprob(env, theta_ref, s)) for s in seqs])
    gold_1p = gold[None, :]
    sampler = (p_ref / p_ref.sum())[None, :]
    psi0 = np.zeros((1, n_seq))
    pairs = sample_preferences(np.random.default_rng(33), gold_1p, sampler,
                              ctx.n_pairs * 6, noise=0.05)
    psi = fit_reward_model(psi0, pairs, lr=ctx.lr, n_steps=ctx.n_steps)[0]

    # Calibrate both to the reference distribution -- the level is free, so the
    # only meaningful comparison centres both where the RM's data actually was.
    psi_c = psi - float(p_ref @ psi)
    gold_c = gold - float(p_ref @ gold)

    err = psi_c - gold_c
    betas = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25)

    def ladder(tilt_by: np.ndarray, label: str):
        rows = []
        print(f"  -- ladder tilted by the {label} --")
        for beta in betas:
            p = tilted_sequence_dist(env, theta_ref, tilt_by, beta)
            kl = float(np.sum(p * (np.log(np.maximum(p, 1e-300))
                                   - np.log(np.maximum(p_ref, 1e-300)))))
            rows.append((beta, kl, float(p @ err ** 2), float(p @ np.maximum(err, 0.0))))
            print(f"     beta {beta:>5.2f}   seq KL {kl:7.4f}   "
                  f"E[(r_psi-r_gold)^2] {rows[-1][2]:8.4f}   "
                  f"E[over-estimate] {rows[-1][3]:7.4f}")
        return rows

    # On-distribution the RM is good -- which is exactly why held-out accuracy is
    # a misleading summary of a reward model (drill 2).
    on_dist = float(p_ref @ err ** 2)
    print(f"  ... on-distribution (p_ref) error = {on_dist:.4f}")
    _check("the RM is accurate ON the distribution it was fit on", on_dist < 0.6,
           f"got {on_dist:.4f}")

    gold_rows = ladder(gold, "GOLD reward (negative control -- no real policy does this)")
    prox_rows = ladder(psi, "LEARNED PROXY r_psi (what a real policy maximizes)")

    for rows, name in ((gold_rows, "gold"), (prox_rows, "proxy")):
        kls = [r[1] for r in rows]
        mses = [r[2] for r in rows]
        _check(f"[{name}] the KL ladder increases as beta falls",
               all(b > a for a, b in zip(kls, kls[1:])), f"{np.round(kls, 4)}")
        _check(f"[{name}] RM squared error is MONOTONE INCREASING in the measured "
               "sequence KL", all(b > a for a, b in zip(mses, mses[1:])),
               f"{np.round(mses, 4)}")
        _check(f"[{name}] error at the widest KL dwarfs the on-distribution error",
               mses[-1] / max(on_dist, 1e-12) > 4.0,
               f"{mses[-1]:.4f} / {on_dist:.4f} = {mses[-1] / max(on_dist, 1e-12):.1f}x")

    # Over-estimation is REPORTED here, not asserted, and the reason is worth
    # knowing. `E[max(err, 0)]` degenerates as beta -> 0: the tilted
    # distribution collapses onto a single sequence, so the statistic becomes
    # the error at one point and reads exactly 0.0 whenever that point happens
    # to be under-rated. It is a bad summary of "the optimizer seeks positive
    # error" -- and with 48k clean-ish comparisons over 121 sequences this RM is
    # simply too good for over-optimization to appear at all. Stage `starvation`
    # is where that effect lives, because it needs a DATA-STARVED RM.
    print(f"  ... E[over-estimate] gold  ladder: "
          f"{np.round([r[3] for r in gold_rows], 4)}")
    print(f"  ... E[over-estimate] proxy ladder: "
          f"{np.round([r[3] for r in prox_rows], 4)}")
    print("  ... (both fall -- see stage `starvation`; a good RM has no hump)")


def stage_starvation(ctx: Ctx) -> None:
    """RM error, over-estimation and regret vs PREFERENCE-DATA SIZE"""
    _fail_fast(ctx)
    bt_loss_and_grad(np.zeros((1, 2)), np.array([[0, 0, 1]]))
    env, theta_ref, seqs = ctx.env, ctx.theta_ref, ctx.seqs
    gold, n_seq = ctx.gold_vec, len(ctx.seqs)

    p_ref = np.array([np.exp(sequence_logprob(env, theta_ref, s)) for s in seqs])
    sampler = (p_ref / p_ref.sum())[None, :]
    psi0 = np.zeros((1, n_seq))
    gold_c = gold - float(p_ref @ gold)

    sizes = (300, 1_200, 5_000, 20_000) if not ctx.fast else (300, 1_200, 5_000)
    rows = []
    print(f"  {'n_pairs':>8} {'on_dist':>8} {'err@tilt':>9} {'ratio':>6} "
          f"{'E[over]':>8} {'regret':>7}")
    for n in sizes:
        pairs = sample_preferences(np.random.default_rng(33), gold[None, :], sampler,
                                  n, noise=0.15)
        psi = fit_reward_model(psi0, pairs, lr=ctx.lr, n_steps=ctx.n_steps)[0]
        err = (psi - float(p_ref @ psi)) - gold_c
        on = float(p_ref @ err ** 2)
        p = tilted_sequence_dist(env, theta_ref, psi, 0.5)
        e_kl = float(p @ err ** 2)
        over = float(p @ np.maximum(err, 0.0))
        regret = float(gold.max() - gold[int(np.argmax(psi))])
        rows.append((n, on, e_kl, over, regret))
        print(f"  {n:>8} {on:>8.4f} {e_kl:>9.4f} {e_kl / on:>6.2f} "
              f"{over:>8.4f} {regret:>7.3f}")

    on = [r[1] for r in rows]
    ratio = [r[2] / r[1] for r in rows]
    over = [r[3] for r in rows]
    regret = [r[4] for r in rows]

    _check("more comparisons => lower ON-DISTRIBUTION error", on[0] > 2.5 * on[-1],
           f"{on[0]:.4f} -> {on[-1]:.4f} ({on[0] / on[-1]:.1f}x)")
    # Direction at every size, magnitude only where the fit has converged. The
    # first version of this demanded "> 2x at every size" and that was fitted to
    # one dataset: across four seeds the middle sizes range down to 1.6x, because
    # at 1200 comparisons over 121 items the RM is still noisy everywhere, not
    # just off-distribution.
    _check("off-distribution error is worse at EVERY data size",
           all(r > 1.0 for r in ratio), f"ratios {np.round(ratio, 2)}")
    _check("and once the RM has converged, several times worse",
           ratio[-1] > 3.0, f"{ratio[-1]:.2f}x at n={rows[-1][0]}")
    _check("OVER-estimation -- the error an optimizer seeks -- collapses with data",
           over[0] > 10.0 * over[-1],
           f"{over[0]:.4f} -> {over[-1]:.4f} ({over[0] / max(over[-1], 1e-9):.0f}x)")
    # Regret is REPORTED, not asserted. `gold.max() - gold[argmax(psi)]` hangs on
    # a discrete argmax over 121 items, so one noisy estimate reaching the top
    # moves it by a whole reward gap; across seeds it is non-monotone and
    # sometimes 0 at the smallest data size. A statistic that depends on an
    # argmax has no business in an assertion without averaging over seeds.
    print(f"  ... regret of trusting the proxy's argmax: {np.round(regret, 2)} "
          "(reported, not asserted -- argmax statistics are too noisy at n=1)")
    print("  ... => beta is a property of the RM's DATA, not of the policy: a "
          "better-fit RM\n      earns a wider trust region. Day 4 turns this into "
          "a moving Goodhart turnover.")


STAGES = [
    ("enumerate", stage_enumerate),
    ("logprob", stage_logprob),
    ("exact_j", stage_exact_j),
    ("seq_kl", stage_seq_kl),
    ("tilt", stage_tilt),
    ("bt_grad", stage_bt_grad),
    ("identifiable", stage_identifiable),
    ("noise", stage_noise),
    ("validity", stage_validity),
    ("starvation", stage_starvation),
]


# ---- runner (copy verbatim; nothing below needs editing) -------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv or "-l" in argv:
        print("stages (run a subset by name):")
        for name, fn in STAGES:
            print(f"  {name:<14} {fn.__doc__}")
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
