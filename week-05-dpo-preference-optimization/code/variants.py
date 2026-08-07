"""
Week 5 Day 4 -- the variant grid: DPO / cDPO / IPO / KTO / ORPO / SimPO in ONE harness.

The zoo is normally memorised as four papers. It is actually a 5x5 grid with two axes,
and every named method is one line in it:

    L = link( f(pi_theta, pi_ref, y_w) , f(pi_theta, pi_ref, y_l) )

    axis f -- the IMPLICIT REWARD          axis link -- the LINK FUNCTION
      ref_sum     b*sum_t log(pi/pi_ref)     logsigmoid            -log s(d)
      ref_mean    (b/|y|)*sum_t log(pi/pi_ref)  logsigmoid_smoothed   cDPO's label smoothing
      noref_sum   b*sum_t log pi             squared_target        (d - tau)^2
      noref_mean  (b/|y|)*sum_t log pi       prospect              KTO's asymmetric value
      odds        b*log[P/(1-P)],  P=e^{mean_t log pi}             logsigmoid_margin  -log s(d - g)

                    logsigmoid   smoothed    squared     prospect    margin
      ref_sum         DPO         cDPO        IPO          KTO         .
      ref_mean        len-DPO      .        <- EMPTY ->    .           .
      noref_sum        .           .           .           .           .
      noref_mean       .           .           .           .         SimPO
      odds          ORPO(+SFT)     .           .           .           .

Three problems, measured on D1-D3, and the fix each variant sells:

  P1  THE OPTIMUM IS AT INFINITY.  D1: Ford -- the BT MLE is finite iff the wins-digraph
      is strongly connected, i.e. iff the data are NOT separable in e_w - e_l. D2: on a
      perfect matching the losers' mass went 0.343396 -> 1.649e-09 with accuracy 1.0 the
      whole way.  Fixes: cDPO (cap the LABEL), IPO (cap the LOSS), SimPO's margin.

  P2  THE IMPLICIT REWARD IS EXTENSIVE IN |y|.  D3: constant per-token drift +0.2 gave
      margin-vs-length slope +0.200000 under sum and 0.000000 under mean.
      Sold as: divide by |y| -- SimPO, ORPO's length-normalised P. But see the third
      correction at the bottom: a flat margin-vs-length SLOPE does not mean a flat
      TRAINED length, and at a matched margin the mean reduction moves E[|y|] FURTHER.

  P3  YOU NEED PAIRS AND A REFERENCE MODEL.  D3: 2 preprocessing forwards, 8 bytes/pair,
      a frozen second copy of the model.  Fixes: KTO (no pairs), ORPO/SimPO (no reference).

WHAT THIS FILE ASSERTS THAT THE PAPERS ONLY CLAIM
-------------------------------------------------
  (a) THE GRID REPRODUCES THE SPECIALS.  Every named method equals an independently
      written direct implementation ELEMENTWISE to ~1e-15, and the check is made
      non-vacuous by reporting how far apart the WRONG cells are. Chained regression,
      W4 D5's trick.

  (b) THE LINK AXIS HAS EXACTLY ONE MEMBER THAT IS NOT A FUNCTION OF THE DIFFERENCE.
      Add a constant c to BOTH r_w and r_l: logsigmoid / smoothed / squared / margin are
      unchanged to machine precision. `prospect` is not. That single invariance IS the
      "both logprobs can fall" pathology, and it is why KTO needs no pairs -- a link that
      only sees d cannot be evaluated on a lone example.
      CAREFUL: difference-only in r is NOT the same as blind to a common fall in log pi.
      That extra step needs the REWARD to be affine in log pi with the SAME slope for both
      sequences. `ref_sum`/`noref_sum` qualify (slope 1). `ref_mean`/`noref_mean` do NOT --
      adding c to every logprob moves d by exactly beta*c*(1/|y_w| - 1/|y_l|), which is
      zero only on equal-length pairs. `odds` is not even affine.

  (c) A UNIFORM REFERENCE IS NOT A TRUST REGION.  log(pi(y)/V^-|y|) = log pi(y) + |y| log V.
      Under MEAN reduction that constant cancels in d, so uniform-reference is bitwise
      identical to reference-FREE. Under SUM reduction it does not cancel, so it is a pure
      length bonus of log V nats/token. A reference only constrains you if it is informative.

  (d) THE THREE FINITE OPTIMA, IN CLOSED FORM, AGAINST A NUMERICAL MINIMISER.
      For one pair observed n times with win-fraction f, in LINK-INPUT units (d = r_w - r_l,
      i.e. D3's preference logit M -- `implicit_reward` has already applied beta):
          DPO       M* = logit(f)                             -> +-inf as f -> 0,1
          cDPO(eps) M* = logit(f(1-eps)+(1-f)eps)              finite, EPS-determined
          IPO(tau)  M* = (2f - 1)*tau                          finite, TAU-determined
      Divide by beta for D3's log-ratio units: at eps=0.10, beta=0.3, M* = 2.197225 and
      Dhat* = 7.324082. cDPO stops where the DATA stops being trustworthy; IPO stops where
      you TOLD it to. The FOURTH link, logsigmoid_margin, has NO closed form -- gamma is
      subtracted whichever way the label points, which destroys BT's antisymmetry. The
      natural guess "M* = logit(f) + gamma" is WRONG (0.905465 predicted, 0.533227 true at
      f=0.60, gamma=0.50), and the margin variant still diverges as f -> 1.

  (e) IPO'S COST AS A NUMBER, NOT A CAVEAT.  IPO's optimum is AFFINE in f and bounded by
      tau, so it COMPRESSES the dynamic range of preference strength. Measured on
      f = 0.99 vs f = 0.60: BT's ratio 11.3330 collapses to IPO's 4.9000, a 2.31x squeeze.

  (f) ORPO: BOTH INGREDIENTS MATTER, AND ONLY ONE OF THEM IS ROBUST -- by 2x2 ABLATION
      PLUS A BETA SWEEP.  The tempting claim "the mechanism is the SFT term, because the
      odds link is difference-only like every other one" is TOO GLIB, and the file proves
      it wrong twice over. First, difference-only in r does NOT imply blind to a common
      fall in log pi: the `_sum` rewards have slope 1 for every sequence so a common shift
      cancels exactly, but `odds` is NONLINEAR in log pi (d(log-odds)/d(mean lp) = 1/(1-P)
      >= 1) so the same shift DOES move d. Second, the odds link's effect on P2 CHANGES
      SIGN with beta (-1.899, -0.457, +0.184, +0.385 at beta = 0.25, 0.5, 1, 2) while the
      SFT term raises the rejected logprob at EVERY beta (+0.566, +0.528, +0.522, +0.557).
      Moral: run the ablation at more than one hyperparameter before you name a mechanism.

  (g) EACH EXPERIMENT MOVES EXACTLY ONE AXIS, AT A MATCHED MARGIN.
          length   : ref_sum   vs ref_mean     (same reference, reduction changes)
          noref    : ref_sum   vs noref_sum    (same reduction, reference changes)
      Anything else is a two-variable ablation and cannot attribute an effect. Matching the
      margin is not optional: run the length ladder to convergence instead and BOTH arms
      saturate at |y| = t_max, which hides the effect completely.

  (h) THE EMPTY CELL.  ref_mean + squared_target -- "length-normalised IPO" -- occupies a
      cell no paper does. The two axes act on DIFFERENT things and do not interfere: the
      LINK decides whether the margin is finite (both squared cells stop dead, both
      logsigmoid cells keep climbing), the REWARD decides where the length goes. The
      squared link also hands us a matched margin for free, since both squared cells
      converge to the identical margin 1.619048.

  (i) KTO IS THE PAIRED LOSS WITH THE PAIRING DELETED, AND YOU CAN SEE THE SEAM. The
      prospect link is a SUM OF TWO INDEPENDENT PER-EXAMPLE TERMS, so evaluating it never
      needed the two halves to come from the same pair. `stage_kto` asserts the identity
      to 1e-15: KTO on flattened pair data == 0.5 * mean(prospect link) on those pairs,
      the 2 being 2n examples against n pairs. Dropping pairs is exact, not an
      approximation. It needs a PromptEnv (not TabEnv) to be visible, because z_ref's
      mismatch roll is a permutation on unconditional data and therefore a no-op there.

FOUR PLACES WHERE THE RECEIVED STORY IS WRONG AND THE FILE ASSERTS THE TRUE VERSION
------------------------------------------------------------------------------------
  * "IPO's converged margin on the huge pair is capped at the SAME VALUE as on the small
    ones" is FALSE. IPO's optimum is (2f-1)*tau, which still depends on f. The true and
    more interesting statement is COMPRESSION: the ratio between a strong and a weak
    preference shrinks from logit(.99)/logit(.60) = 11.3330 to (2*.99-1)/(2*.60-1) = 4.9.
  * "assert DPO's exact KL diverges" is FALSE in a finite space -- D2 established that
    KL(pi||pi_ref) <= max_y -log pi_ref(y), the point-mass bound. The file asserts
    monotone growth toward that bound and prints the bound, as D2's `divergence` does.
    What actually diverges is the MARGIN, and its signature is not "% growth" (which decays
    even for a divergent run) but a CONSTANT increment per DOUBLING of the budget:
    +0.7163 then +0.7078. That is margin ~ log t, the separable-logistic-regression rate.
  * "SimPO's length normalisation removes length bias" is FALSE, or at least not for the
    reason usually given. At a MATCHED MARGIN, on both a longer-wins and a shorter-wins
    ladder, the MEAN reduction moves E[|y|] FURTHER than the sum reduction in all four
    cells (+1.834 -> +2.210, +2.520 -> +2.672, -0.876 -> -1.135, -1.091 -> -1.196).
    Dividing by |y| does not oppose length; since d(r)/d(logp) = beta/|y| is largest at
    |y| = 1, it makes SHORT sequences the cheapest lever, so the optimiser buys its margin
    at the short end and pushes E[|y|] further in WHICHEVER direction the labels point.
    Length normalisation AMPLIFIES the labels' length signal. What shortens SimPO's
    outputs on real data is gamma and reference-freeness, not 1/|y|.
  * "KTO's loss is BOUNDED, so unlike DPO it cannot run away on separable data" is FALSE,
    and it is the most seductive of the four because the premise is true. A BOUNDED LOSS
    IS NOT A LOSS WITH A FINITE MINIMISER: inf_u sigma(-u) = 0 is attained only at
    u = +infinity, so KTO's optimum on separable data sits at infinity exactly like DPO's.
    Both gradients decay like e^-u, so both give margin ~ log t, and `stage_kto` measures
    KTO climbing +0.9294 per doubling next to DPO's +0.7207 -- FASTER, not slower.
    Boundedness buys something real but different: BOUNDED INFLUENCE. As a single example
    goes badly wrong, -log sigma's gradient saturates at 1 and never lets go, while
    prospect's decays to 2.1e-9 by d = -20. With 4/40 labels flipped, DPO's clean-pair
    margin falls 1.2411 against KTO's 0.6684, and DPO is dragged +2.8343 toward the wrong
    labels against KTO's +1.5870. KTO answers D3's noise problem, not D1's P1.

  run everything   ./.venv/bin/python week-05-dpo-preference-optimization/code/variants.py
  one stage        ...  variants.py separable
  a few stages     ...  variants.py reward link reduce
  fast fixtures    ...  variants.py --fast
  list the stages  ...  variants.py --list

Build order (each group is runnable the moment it is written):

  1. implicit_reward  -> stages `reward`
  2. link             -> stages `link`, `optima`, `kto`   (`kto` is provided code -- it
                         consumes only these two, so it runs as soon as they are done)
  3. variant_loss     -> stage  `reduce`
  4. train_variant    -> stages `separable`, `ipo_cost`, `orpo`, `simpo`, `empty_cell`

Run it under `-W error::RuntimeWarning`.
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# The enumerable environment (provided -- W4 D1 / W5 D2, in torch so autograd
# supplies every gradient and no variant needs one derived by hand)
# ---------------------------------------------------------------------------


class TabEnv:
    """A tabular token-level LM, small enough to enumerate exhaustively.

    Every sequence terminates in `eos` or hits `t_max`, so the sequence set is finite and
    `seq_logprobs` returns an EXACT pmf. Nothing in this file is sampled.

    Two environments are used:
      TabEnv(vocab=3, t_max=3) -> 7 states, 15 sequences, lengths {1,2,3}   (general work)
      TabEnv(vocab=2, t_max=5) -> 5 states,  5 sequences, lengths {1,2,3,4,5}
          the LENGTH LADDER: exactly one sequence per length, so length is the ONLY thing
          that distinguishes two sequences and a length effect cannot be confounded.
    """

    def __init__(self, vocab: int = 3, t_max: int = 3):
        assert vocab >= 2 and t_max >= 1
        self.vocab, self.t_max, self.eos = vocab, t_max, vocab - 1

        states, frontier = [()], [()]
        for _ in range(t_max - 1):
            nxt = [p + (a,) for p in frontier for a in range(vocab - 1)]
            states.extend(nxt)
            frontier = nxt
        self.states = states
        self._index = {s: i for i, s in enumerate(states)}
        self.n_states = len(states)

        out, frontier = [], [()]
        for _ in range(t_max):
            nxt = []
            for p in frontier:
                for a in range(vocab - 1):
                    nxt.append(p + (a,))
                out.append(p + (self.eos,))
            frontier = nxt
        self.seqs = out + frontier
        self.n_seqs = len(self.seqs)

        idx_state, idx_token, idx_seq = [], [], []
        for i, s in enumerate(self.seqs):
            for t in range(len(s)):
                idx_state.append(self._index[s[:t]])
                idx_token.append(s[t])
                idx_seq.append(i)
        self.idx_state = torch.tensor(idx_state, dtype=torch.long)
        self.idx_token = torch.tensor(idx_token, dtype=torch.long)
        self.idx_seq = torch.tensor(idx_seq, dtype=torch.long)
        self.lengths = torch.tensor([float(len(s)) for s in self.seqs])

    def seq_logprobs(self, theta: torch.Tensor) -> torch.Tensor:
        """log pi_theta(y) for every enumerable y, shape (n_seqs,). Differentiable."""
        lp = F.log_softmax(theta, dim=-1)
        tok = lp[self.idx_state, self.idx_token]
        out = torch.zeros(self.n_seqs, dtype=tok.dtype)
        return out.index_add(0, self.idx_seq, tok)

    def uniform_theta(self) -> torch.Tensor:
        """All-zero logits == uniform over the vocab at every state."""
        return torch.zeros(self.n_states, self.vocab)

    def random_theta(self, seed: int, scale: float = 1.0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return scale * torch.randn(self.n_states, self.vocab, generator=g)

    def __repr__(self) -> str:
        return (f"TabEnv(vocab={self.vocab}, t_max={self.t_max}, "
                f"n_states={self.n_states}, n_seqs={self.n_seqs})")


def exact_kl(lp: torch.Tensor, lp_ref: torch.Tensor) -> float:
    """KL(pi || pi_ref) over whole sequences, by enumeration. No estimator."""
    return float((lp.exp() * (lp - lp_ref)).sum())


def entropy(lp: torch.Tensor) -> float:
    return float(-(lp.exp() * lp).sum())


def expected_length(lp: torch.Tensor, lengths: torch.Tensor) -> float:
    return float((lp.exp() * lengths).sum())


def point_mass_kl_bound(lp_ref: torch.Tensor) -> float:
    """max_y -log pi_ref(y): the largest KL a policy in this space can reach (D2)."""
    return float((-lp_ref).max())


def stable_log_odds(mean_lp: torch.Tensor) -> torch.Tensor:
    """log[P/(1-P)] for P = exp(mean_lp), mean_lp < 0.

    log P - log(1-P) = mean_lp - log(-expm1(mean_lp)). Never form 1-P by subtraction:
    for mean_lp near 0 that is catastrophic cancellation, and for the ORPO odds it is
    exactly the regime the loss drives you into.
    """
    return mean_lp - torch.log(-torch.expm1(mean_lp.clamp(max=-1e-12)))


class PromptEnv:
    """P prompts, each with its OWN table -- the minimum structure that makes KTO's z_ref
    a real quantity instead of a decorative one.

    `TabEnv` is unconditional: there is no x, so pi(y) is one distribution and the reward
    r(y) is one number per sequence. KTO's reference level is estimated on MISMATCHED
    (prompt, completion) combinations:

        z_ref = max(0, mean_i log[ pi(y_{i+1} | x_i) / pi_ref(y_{i+1} | x_i) ])

    With no x, "score y_{i+1} under x_i" degenerates to "look at y_{i+1}", and since the
    reduce is a MEAN -- which is permutation-invariant -- the mismatch does nothing at all
    (stage `reduce` measures the gap: ~1e-16, pure summation order). The roll is only
    load-bearing when the reward is a function of the PAIRING, which needs the model to be
    prompt-conditioned. That is the entire content of this class.

    theta has shape (n_prompts, n_states, vocab): P independent copies of the same tabular
    LM. Every index tensor TabEnv precomputes broadcasts over the leading axis unchanged,
    so `logprobs` is the same index_add with one extra dimension and returns (P, n_seqs).
    """

    def __init__(self, vocab: int = 3, t_max: int = 3, n_prompts: int = 4):
        self.tab = TabEnv(vocab=vocab, t_max=t_max)
        self.n_prompts = n_prompts
        self.n_seqs = self.tab.n_seqs
        self.lengths = self.tab.lengths
        self.vocab, self.n_states = self.tab.vocab, self.tab.n_states

    def logprobs(self, theta: torch.Tensor) -> torch.Tensor:
        """log pi_theta(y | x) for every (prompt, sequence), shape (P, n_seqs)."""
        t = self.tab
        lp = F.log_softmax(theta, dim=-1)
        tok = lp[:, t.idx_state, t.idx_token]
        out = torch.zeros(theta.shape[0], self.n_seqs, dtype=tok.dtype)
        return out.index_add(1, t.idx_seq, tok)

    def random_theta(self, seed: int, scale: float = 1.0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return scale * torch.randn(self.n_prompts, self.n_states, self.vocab, generator=g)

    def __repr__(self) -> str:
        return (f"PromptEnv(n_prompts={self.n_prompts}, n_seqs={self.n_seqs}, "
                f"n_states={self.n_states})")


def mean_kl(lp: torch.Tensor, lp_ref: torch.Tensor) -> float:
    """Mean over prompts of the EXACT per-prompt KL(pi(.|x) || pi_ref(.|x))."""
    return float((lp.exp() * (lp - lp_ref)).sum(dim=-1).mean())


# ---------------------------------------------------------------------------
# The variant specification -- one dataclass, one row of the grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """One cell of the grid, plus its hyperparameters.

    `sft_weight` is NOT an axis -- it is an additive term (ORPO's L_SFT), and it is kept
    separate precisely so stage `orpo` can ablate it against the odds link.
    """
    name: str
    f_kind: str
    l_kind: str
    beta: float = 1.0
    eps: float = 0.1          # logsigmoid_smoothed
    tau: float = 1.0          # squared_target
    gamma: float = 0.0        # logsigmoid_margin
    lam_d: float = 1.0        # prospect, desirable weight
    lam_u: float = 1.0        # prospect, undesirable weight
    sft_weight: float = 0.0   # additive -log pi(y_w)/|y_w|
    or_weight: float = 1.0    # ORPO's lambda on the preference term


F_KINDS = ("ref_sum", "ref_mean", "noref_sum", "noref_mean", "odds")
L_KINDS = ("logsigmoid", "logsigmoid_smoothed", "squared_target", "prospect",
           "logsigmoid_margin")


# ---------------------------------------------------------------------------
# Exercises (TODO bodies)
# ---------------------------------------------------------------------------


def implicit_reward(kind: str, seq_logp: torch.Tensor, seq_logp_ref: torch.Tensor,
                    lengths: torch.Tensor, beta: float) -> torch.Tensor:
    """Axis f. Returns r_hat for EVERY enumerable sequence, shape == seq_logp.shape.

        ref_sum      beta * ( log pi(y) - log pi_ref(y) )
        ref_mean     beta * ( log pi(y) - log pi_ref(y) ) / |y|
        noref_sum    beta *   log pi(y)
        noref_mean   beta *   log pi(y) / |y|
        odds         beta * log[ P/(1-P) ],   P = exp( log pi(y) / |y| )

    Four things this must get right, each with its own check:

      * `ref_*` must be EXACTLY 0 everywhere when pi == pi_ref. D2 measured this: the
        implicit reward does not start large, the ratio the LABELS demand is what is large.
      * `*_mean` divides by that sequence's OWN length. There is no "total length": y_w and
        y_l have different lengths and that asymmetry IS the mechanism. Dividing the LOSS
        by a length instead of the REWARD is a per-example reweighting and changes nothing
        about the length prior -- same family as D3's "correct token values, missing the
        sequence reduction". Ask which object you are transforming.
      * `odds` needs `stable_log_odds`, provided above. Do not write log(P) - log(1-P) with
        P formed by exponentiating; ORPO's loss drives P toward 1 and that is exactly where
        1-P cancels to zero.
      * Everything stays differentiable -- no .item(), no numpy, no in-place writes into
        seq_logp.

    Work in log space throughout.
    """
    if kind == 'ref_sum':
        return beta * (seq_logp - seq_logp_ref)
    elif kind == 'ref_mean':
        return beta * (seq_logp - seq_logp_ref) / lengths
    elif kind == 'noref_sum':
        return beta * seq_logp
    elif kind == 'noref_mean':
        return beta * seq_logp / lengths
    elif kind == 'odds':
        mean_lp = seq_logp / lengths
        return beta * stable_log_odds(mean_lp)
    else:
        raise ValueError(kind)


def link(kind: str, r_w: torch.Tensor, r_l: torch.Tensor, spec: Spec,
         z_ref: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Axis link. Returns the PER-PAIR loss, shape == r_w.shape. A LOSS: smaller is better.

        d = r_w - r_l

        logsigmoid           -log sigma(d)
        logsigmoid_smoothed  -(1-eps) log sigma(d) - eps log sigma(-d)
        squared_target       (d - tau)^2
        logsigmoid_margin    -log sigma(d - gamma)
        prospect             lam_d * sigma( -(r_w - z_ref) ) + lam_u * sigma( -(z_ref - r_l) )

    `prospect` is KTO, and the signature is the point of the whole file: it is the ONLY
    link that reads r_w and r_l SEPARATELY. The other four are functions of d alone, so
        - they are invariant to adding a constant to both  ->  "both logprobs can fall";
        - they cannot be evaluated on a lone example       ->  they REQUIRE pairs.
    KTO buys unpaired data by giving up difference-only-ness, and it pays for it with a
    reference level z_ref that has to be estimated. It is one trade, not two features.

    The prospect form above is KTO's `E[lam_y - v]` after using 1 - sigma(u) = sigma(-u);
    a pair is consumed as two unpaired examples (y_w desirable, y_l undesirable), which is
    exactly how the KTO paper applies it to paired data.

    Traps, all of which have a check:

      * USE `F.logsigmoid`, never `torch.log(torch.sigmoid(...))`. D3 measured the naive
        form saturating to exactly 0 at z=+40 and to -inf at z=-100, and `separable` runs
        the margin well past both.
      * SIGN. Every branch returns a LOSS. The standing probe `d=0 => -log sigma(0) =
        +log 2 = 0.693147` catches a flipped logsigmoid. It does NOT catch a flipped
        squared term, so use the additive probe instead: EVALUATE EACH TERM AT THE POINT
        IT SHOULD BE HAPPIEST AND CONFIRM IT IS A MINIMUM. At d = tau the squared term
        must read 0 and be the smallest value it ever takes. (Missing minus sign on a loss:
        four consecutive days.)
      * `prospect` must stay bounded: 0 < loss < lam_d + lam_u for every finite input.
        That boundedness is why KTO is immune to P1 for free.
      * z_ref is DETACHED by the caller. Do not differentiate through it.
    """
    d = r_w - r_l
    if kind == 'logsigmoid':
        return - F.logsigmoid(d)
    elif kind == 'logsigmoid_smoothed':
        return - (1-spec.eps) * F.logsigmoid(d) - spec.eps * F.logsigmoid(-d)
    elif kind == 'squared_target':
        return (d - spec.tau) ** 2
    elif kind == 'logsigmoid_margin':
        return - F.logsigmoid(d - spec.gamma)
    elif kind == 'prospect':
        return spec.lam_d * F.sigmoid(-(r_w - z_ref)) + spec.lam_u * F.sigmoid(-(z_ref - r_l))
    else:
        raise ValueError(kind)





def variant_loss(spec: Spec, seq_logp: torch.Tensor, seq_logp_ref: torch.Tensor,
                 lengths: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """The grid: a 0-d tensor loss for any (f_kind, l_kind) plus the optional SFT term.

        r      = implicit_reward(spec.f_kind, ...)
        r_w    = r[pairs[:,0]]      r_l = r[pairs[:,1]]
        L_pref = link(spec.l_kind, r_w, r_l, spec, z_ref)          per-pair
        L      = spec.sft_weight * mean_k [ -log pi(y_w) / |y_w| ]
               + spec.or_weight  * mean_k [ L_pref ]

    Three contracts:

      * MEAN over pairs, never sum. A pair-count factor silently rescales every learning
        rate and puts the `reduce` equalities nowhere near 1e-15.
      * The SFT term is the MEAN-TOKEN NLL of the CHOSEN completion, -log pi(y_w)/|y_w|,
        which is what ORPO uses. It is the marginal-likelihood term that the pairwise loss
        structurally cannot contain (see the invariance in `link`). Note it touches y_w
        ONLY -- ORPO's anchor is the chosen completions and nothing else, which is exactly
        why "no pi_ref" means "no trust region".
      * z_ref for `prospect` is KTO's batch KL estimate, computed HERE and passed down:
            mismatch y_l by rolling it one position (y_l' = r[pairs[:,1].roll(1)]),
            z_ref = clamp( mean_k [ r on the mismatched example ], min=0 )
            and DETACH it.
        A fixed z_ref = 0 does not work: r_hat is identically 0 at pi = pi_ref, so a zero
        baseline calls every example neutral at initialisation and supplies no scale.

        HONEST NOTE, AND `reduce` ASSERTS IT: in THIS env the roll is a NO-OP. A roll is a
        permutation and the mean is permutation-invariant, so mean(r[l.roll(1)]) equals
        mean(r[l]) to summation-order noise (~1e-16). It is kept because it marks where the
        real mechanism lives. Real KTO scores completion y_{i+1} UNDER PROMPT x_i:
            z_ref = max(0, mean_i log[ pi(y_{i+1}|x_i) / pi_ref(y_{i+1}|x_i) ])
        which is a different SET of numbers, not a permutation of the matched ones -- the
        model's log-ratio on MISMATCHED prompt/completion combinations. That mismatch is
        the point: z_ref must be what an UNRELATED output is worth. Estimate it on matched
        pairs and you are measuring the log-ratio on exactly the completions the loss is
        raising, so the baseline tracks the signal and cancels it. TabEnv's sequences are
        unconditional -- there is no x -- so the mechanism degenerates. The roll can only
        matter when the reward depends on the PAIRING.

    If you finish this function without calling both `implicit_reward` and `link`, that is
    a bug report about your own code (standing rule, W5 D2).
    """
    r_all = implicit_reward(spec.f_kind, seq_logp, seq_logp_ref, lengths, spec.beta)
    r_w, r_l = r_all[pairs[:, 0]], r_all[pairs[:, 1]]
    z_l_mismatch = r_all[pairs[:, 1].roll(1)]
    z_ref = z_l_mismatch.mean().clamp(0).detach()
    l_pref = link(spec.l_kind, r_w, r_l, spec, z_ref)
    L = - spec.sft_weight * (seq_logp[pairs[:, 0]] / lengths[pairs[:, 0]]).mean() + spec.or_weight * l_pref.mean()
    return L




@torch.no_grad()
def update_record(record, spec, theta, seq_logp_ref, env, pairs):
    seq_logp = env.seq_logprobs(theta)
    record['loss'].append(float(variant_loss(spec, seq_logp, seq_logp_ref, env.lengths, pairs)))
    r_all = implicit_reward(spec.f_kind, seq_logp, seq_logp_ref, env.lengths, spec.beta)
    r_w, r_l = r_all[pairs[:, 0]], r_all[pairs[:, 1]]
    record['margin'].append(float((r_w-r_l).mean()))
    record['acc'].append(float((r_w>r_l).double().mean()))
    record['kl'].append(exact_kl(seq_logp, seq_logp_ref))
    record['logp_w'].append(float(seq_logp[pairs[:, 0]].mean()))
    record['logp_l'].append(float(seq_logp[pairs[:, 1]].mean()))
    record['elen'].append(expected_length(seq_logp, env.lengths))
    record['nll_w'].append(float(-(seq_logp[pairs[:, 0]] / env.lengths[pairs[:, 0]]).mean()))
    record['ent'].append(entropy(seq_logp))



def train_variant(env: TabEnv, theta0: torch.Tensor, theta_ref: torch.Tensor,
                  pairs: torch.Tensor, spec: Spec, n_steps: int = 2000,
                  lr: float = 1.0, stop_margin: float | None = None) -> dict:
    """Full-batch plain gradient descent on `variant_loss`, with an exact instrument panel.

    Returns a dict of numpy arrays, one entry per step, each measured at the theta BEFORE
    that step's update, plus `theta` (the final parameters) and `stopped_at`:

        loss     the variant's own loss
        margin   mean_k ( r_w - r_l ) under THIS spec's f_kind      <- not comparable across f
        acc      fraction of pairs with margin > 0
        kl       KL(pi_theta || pi_ref), EXACT, by enumeration
        logp_w   mean_k log pi_theta(y_w)      <- the marginal the pairwise loss cannot see
        logp_l   mean_k log pi_theta(y_l)
        elen     E_pi[|y|]                     <- the length instrument
        ent      H(pi) over sequences          <- SimPO's price shows up here
        nll_w    mean_k -log pi_theta(y_w)/|y_w|   <- ORPO's SFT objective, always logged

    `stop_margin`: if given, stop at the FIRST step whose mean margin reaches it and record
    the index in `stopped_at` (-1 if never reached). This is D2's matched-margin design and
    it is the only honest way to compare KL across two objectives -- otherwise you are
    measuring which one trained longer.

    Requirements:
      * PLAIN GD, fixed lr, full batch. The implicit bias of plain GD on separable data is
        the phenomenon `separable` measures; an optimiser with its own regularisation
        hides it (D2's standing note).
      * Separate OPTIMISATION from LOGGING: build the graph, step, then recompute the panel
        under `torch.no_grad()`. Never relabel a pre-update record as a post-update metric
        (D3 bug 6). Step 0 is measured before any update.
      * theta_ref is frozen. It must never receive a gradient.
      * Record the FINAL theta after the last update, not a stale copy (D3 bug; W3 D1's
        argmax-lag, fourth sighting).
      * lr is a CONFOUND in every comparison stage, so it is passed in, never chosen per
        arm inside here. Be aware the squared link tolerates far less than the log-sigmoid
        one: lr=2.0 on IPO's loss diverges to NaN outright on `separable`, while DPO is
        happy at the same lr. If a stage compares arms, they must share the lr.
    """
    theta = theta0.clone().requires_grad_(True)
    with torch.no_grad():
        seq_logp_ref = env.seq_logprobs(theta_ref)

    record = {k: [] for k in ("loss", "margin", "acc", "kl",
                              "logp_w", "logp_l", "elen", "ent", "nll_w")}

    # Step 0 is measured BEFORE any update, so the panel has n_steps + 1 entries.
    update_record(record, spec, theta, seq_logp_ref, env, pairs)
    stopped_at = 0 if (stop_margin is not None
                       and record['margin'][-1] >= stop_margin) else -1

    if stopped_at < 0:
        for step in range(n_steps):
            seq_logp = env.seq_logprobs(theta)
            loss = variant_loss(spec, seq_logp, seq_logp_ref, env.lengths, pairs)
            grad, = torch.autograd.grad(loss, theta)
            with torch.no_grad():
                theta -= lr * grad
            update_record(record, spec, theta, seq_logp_ref, env, pairs)
            if stop_margin is not None and record['margin'][-1] >= stop_margin:
                stopped_at = step + 1
                break

    out = {k: np.asarray(v, dtype=float) for k, v in record.items()}
    out["theta"] = theta.detach().clone()
    out["stopped_at"] = stopped_at
    return out


# ---------------------------------------------------------------------------
# KTO on unpaired data (provided -- it consumes `implicit_reward` and `link`)
# ---------------------------------------------------------------------------


def _pairs_from(ex: torch.Tensor, n_prompts: int) -> torch.Tensor:
    """Induce (prompt, desirable, undesirable) pairs from unpaired labels, within prompt.

    This is the only way to hand unpaired data to a BT method at all, and it is lossy:
    a prompt with n_d desirable and n_u undesirable examples yields min(n_d, n_u) pairs,
    so any surplus on either side is DISCARDED. That is the data cost KTO removes.
    """
    rows = []
    for p in range(n_prompts):
        sel = ex[ex[:, 0] == p]
        d = sel[sel[:, 2] > 0][:, 1].tolist()
        u = sel[sel[:, 2] < 0][:, 1].tolist()
        rows.extend([[p, a, b] for a, b in zip(d, u)])
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 3)


def kto_z_ref(r: torch.Tensor, ex: torch.Tensor, roll: bool = True) -> torch.Tensor:
    """KTO's reference level: mean reward on MISMATCHED (prompt, completion) pairs.

    `ex` is (n, 3): prompt index, sequence index, label in {+1 desirable, -1 undesirable}.
    With roll=True the completion of example i+1 is scored under the prompt of example i,
    which on a PromptEnv is a genuinely different number -- not a permutation of the
    matched set, as it is on TabEnv. Clamped at 0 because it estimates a KL, and DETACHED
    because it is a baseline, not a parameter.
    """
    p, y = ex[:, 0], ex[:, 1]
    y_used = y.roll(1) if roll else y
    return r[p, y_used].mean().clamp(min=0.0).detach()


def kto_loss(spec: Spec, lp: torch.Tensor, lp_ref: torch.Tensor,
             lengths: torch.Tensor, ex: torch.Tensor, roll: bool = True) -> torch.Tensor:
    """KTO's objective on UNPAIRED data. There is no `pairs` tensor anywhere in here.

        r(x,y) = implicit_reward(...)                      shape (P, n_seqs)
        z      = kto_z_ref(r, ex)                          scalar, detached
        L      = mean_i  lam_{s_i} * sigma( -s_i * (r_i - z) ),   s_i = +-1

    This is KTO's `E[lam_y - v(x,y)]` after 1 - sigma(u) = sigma(-u), and it is bounded in
    (0, max(lam_d, lam_u)) for every finite input -- which is where KTO's immunity to P1
    comes from, at no tuning cost.

    Stage `kto` proves the structural claim that licenses dropping pairs: the paired
    `prospect` link is a SUM OF TWO INDEPENDENT PER-EXAMPLE TERMS,
        link(r_w, r_l) = lam_d*sigma(-(r_w - z)) + lam_u*sigma(-(z - r_l)),
    so evaluating it never actually required the two halves to come from the same pair.
    """
    r = implicit_reward(spec.f_kind, lp, lp_ref, lengths, spec.beta)
    z = kto_z_ref(r, ex, roll=roll)
    p, y, s = ex[:, 0], ex[:, 1], ex[:, 2].to(lp.dtype)
    lam = torch.where(s > 0, torch.as_tensor(spec.lam_d), torch.as_tensor(spec.lam_u))
    return (lam * torch.sigmoid(-s * (r[p, y] - z))).mean()


def train_prompt(penv: PromptEnv, theta0: torch.Tensor, theta_ref: torch.Tensor,
                 loss_fn, ex: torch.Tensor, spec: Spec, n_steps: int = 2000,
                 lr: float = 1.0) -> dict:
    """Plain full-batch GD on a PromptEnv, for any loss written as loss_fn(lp, lp_ref).

    Same conventions as `train_variant`: plain GD (no optimiser with its own
    regularisation), metrics recorded BEFORE each update, theta_ref frozen.

    `margin` here is the DESIRABLE-minus-UNDESIRABLE mean reward -- the unpaired analogue
    of r_w - r_l, and the quantity that runs away under P1.
    """
    theta = theta0.clone().requires_grad_(True)
    with torch.no_grad():
        lp_ref = penv.logprobs(theta_ref)
    p, y, s = ex[:, 0], ex[:, 1], ex[:, 2]
    d_mask, u_mask = s > 0, s < 0
    log = {k: [] for k in ("loss", "margin", "acc", "kl", "z_ref", "r_d", "r_u")}

    def snapshot():
        with torch.no_grad():
            lp = penv.logprobs(theta)
            r = implicit_reward(spec.f_kind, lp, lp_ref, penv.lengths, spec.beta)
            ri = r[p, y]
            z = kto_z_ref(r, ex)
            log["loss"].append(float(loss_fn(lp, lp_ref)))
            log["r_d"].append(float(ri[d_mask].mean()))
            log["r_u"].append(float(ri[u_mask].mean()))
            log["margin"].append(log["r_d"][-1] - log["r_u"][-1])
            log["acc"].append(float(((ri > z) == d_mask).double().mean()))
            log["kl"].append(mean_kl(lp, lp_ref))
            log["z_ref"].append(float(z))

    snapshot()
    for _ in range(n_steps):
        loss = loss_fn(penv.logprobs(theta), lp_ref)
        g, = torch.autograd.grad(loss, theta)
        with torch.no_grad():
            theta -= lr * g
        snapshot()

    out = {k: np.asarray(v, dtype=float) for k, v in log.items()}
    out["theta"] = theta.detach().clone()
    return out


# ---------------------------------------------------------------------------
# Named methods, as one line each (the grid's payoff)
# ---------------------------------------------------------------------------

def named_specs(beta: float = 0.5, eps: float = 0.1, tau: float = 2.0,
                gamma: float = 0.5, or_lambda: float = 1.0) -> dict[str, Spec]:
    """Every named method is a cell plus hyperparameters. Note IPO takes beta = 1.0:
    its tau ABSORBS beta, which is why IPO's cap is a knob and cDPO's is a measurement."""
    return {
        "DPO":   Spec("DPO",   "ref_sum",    "logsigmoid",          beta=beta),
        "cDPO":  Spec("cDPO",  "ref_sum",    "logsigmoid_smoothed", beta=beta, eps=eps),
        "IPO":   Spec("IPO",   "ref_sum",    "squared_target",      beta=1.0,  tau=tau),
        "KTO":   Spec("KTO",   "ref_sum",    "prospect",            beta=beta,
                      lam_d=1.0, lam_u=1.5),
        "ORPO":  Spec("ORPO",  "odds",       "logsigmoid",          beta=1.0,
                      sft_weight=1.0, or_weight=or_lambda),
        "SimPO": Spec("SimPO", "noref_mean", "logsigmoid_margin",   beta=beta, gamma=gamma),
        # the empty cell: length-normalised IPO. No paper occupies it.
        "IPO_mean": Spec("IPO_mean", "ref_mean", "squared_target",  beta=1.0,  tau=tau),
    }


# ---------------------------------------------------------------------------
# Independent direct implementations (the chained-regression controls).
# These share NO code path with the grid. That independence is what makes the
# agreement evidence rather than a tautology.
# ---------------------------------------------------------------------------

def _direct_dpo(lp, lpr, pairs, beta):
    w, l = pairs[:, 0], pairs[:, 1]
    d = beta * ((lp[w] - lpr[w]) - (lp[l] - lpr[l]))
    return -F.logsigmoid(d)


def _direct_cdpo(lp, lpr, pairs, beta, eps):
    w, l = pairs[:, 0], pairs[:, 1]
    d = beta * ((lp[w] - lpr[w]) - (lp[l] - lpr[l]))
    return -(1.0 - eps) * F.logsigmoid(d) - eps * F.logsigmoid(-d)


def _direct_ipo(lp, lpr, pairs, tau):
    w, l = pairs[:, 0], pairs[:, 1]
    h = (lp[w] - lpr[w]) - (lp[l] - lpr[l])
    return (h - tau) ** 2


def _direct_simpo(lp, lengths, pairs, beta, gamma):
    w, l = pairs[:, 0], pairs[:, 1]
    rw = beta * lp[w] / lengths[w]
    rl = beta * lp[l] / lengths[l]
    return -F.logsigmoid(rw - rl - gamma)


def _direct_orpo(lp, lengths, pairs, or_lambda):
    w, l = pairs[:, 0], pairs[:, 1]
    mlp = lp / lengths
    lo = mlp - torch.log(-torch.expm1(mlp.clamp(max=-1e-12)))
    l_or = -F.logsigmoid(lo[w] - lo[l])
    l_sft = -mlp[w]
    return l_sft.mean() + or_lambda * l_or.mean()


def _direct_kto(lp, lpr, pairs, beta, lam_d, lam_u, z_ref):
    w, l = pairs[:, 0], pairs[:, 1]
    rw = beta * (lp[w] - lpr[w])
    rl = beta * (lp[l] - lpr[l])
    return lam_d * torch.sigmoid(-(rw - z_ref)) + lam_u * torch.sigmoid(-(z_ref - rl))


# ---------------------------------------------------------------------------
# Closed forms for the single-pair optima (stage `optima`, stage `ipo_cost`)
# ---------------------------------------------------------------------------

def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def closed_form_optimum(l_kind: str, f: float, spec: Spec) -> float:
    """Minimiser of the mean loss over n observations of ONE pair with win-fraction f,
    IN LINK-INPUT UNITS -- i.e. the optimal d = r_w - r_l, which is D3's PREFERENCE LOGIT
    M, not the log-ratio margin. `implicit_reward` has already applied beta.

        logsigmoid           M* = logit(f)                        -> +-inf at f in {0,1}
        logsigmoid_smoothed  M* = logit(f(1-eps) + (1-f)eps)       finite, EPS-determined
        squared_target       M* = (2f - 1) * tau                   finite, TAU-determined
        logsigmoid_margin    NO CLOSED FORM -- solved by bisection below

    THE TWO UNITS ARE THE WHOLE POINT AND D3 MEASURED BOTH:
        M*    = logit(1-eps)        = 2.197225   at eps=0.10   (preference logit)
        Dhat* = logit(1-eps)/beta   = 7.324082   at beta=0.3   (log-ratio margin)
    Use `logratio_optimum` to convert. Note IPO's tau lives in M-units already, which is
    why IPO is specified with beta = 1.0: tau ABSORBS beta.

    The cDPO line is D1's epsilon-cap and D3's loss-side cap being the SAME result: label
    noise at rate eps and a smoothed label at rate eps have the identical optimum.

    WHY THE MARGIN LINK HAS NO CLOSED FORM (a fact worth more than the formula):
    the other three links are ANTISYMMETRIC in the sense that flipping the label maps
    l(d) -> l(-d), so the f-weighted stationarity condition collapses to one invertible
    scalar equation. gamma destroys that: it is subtracted whichever way the label points,
    so both terms carry the SAME +gamma and the two sides no longer combine.
    Stationarity is
        f * sigma(gamma - M) = (1 - f) * sigma(M + gamma)
    which has no elementary solution. The naive guess "M* = logit(f) + gamma" is WRONG:
    at f=0.60, gamma=0.50 it predicts 0.905465 and the truth is 0.533227.
    Sanity: at gamma = 0 the condition reduces to f/(1-f) = e^M, i.e. M* = logit(f). And
    as f -> 1 the LHS forces M -> +inf, so the margin variant does NOT fix problem P1.
    """
    if l_kind == "logsigmoid":
        return logit(f)
    if l_kind == "logsigmoid_smoothed":
        return logit(f * (1.0 - spec.eps) + (1.0 - f) * spec.eps)
    if l_kind == "squared_target":
        return (2.0 * f - 1.0) * spec.tau
    if l_kind == "logsigmoid_margin":
        g = spec.gamma

        def stat(m: float) -> float:
            return f * _sigmoid(g - m) - (1.0 - f) * _sigmoid(m + g)

        lo, hi = -60.0, 60.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if stat(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    raise ValueError(l_kind)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def logratio_optimum(l_kind: str, f: float, spec: Spec) -> float:
    """The same optimum expressed as a LOG-RATIO margin Dhat = M/beta (D3's units)."""
    return closed_form_optimum(l_kind, f, spec) / spec.beta


def numeric_optimum(l_kind: str, f: float, spec: Spec, n_steps: int = 20_000,
                    lr: float = 0.05) -> float:
    """Minimise the same scalar objective in d directly, so the closed form is CHECKED."""
    d = torch.zeros((), requires_grad=True)
    for _ in range(n_steps):
        r_w = d.reshape(1)
        r_l = torch.zeros(1)
        fwd = link(l_kind, r_w, r_l, spec)
        bwd = link(l_kind, -r_w, r_l, spec)
        loss = f * fwd.mean() + (1.0 - f) * bwd.mean()
        g, = torch.autograd.grad(loss, d)
        with torch.no_grad():
            d -= lr * g
    return float(d.detach())


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


def _pairs(rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def _matching(env: TabEnv) -> torch.Tensor:
    """A perfect matching: every sequence appears in exactly ONE comparison.

    D1's result -- the most disconnected non-trivial comparison graph there is, and it is
    the shape a real DPO dataset has. Flipping every label cannot create a cycle, so
    annotator noise has nowhere to act and the BT MLE is at infinity no matter what.
    """
    n = env.n_seqs - (env.n_seqs % 2)
    perm = list(range(n))
    return _pairs([[perm[i], perm[i + 1]] for i in range(0, n, 2)])


def _connected(env: TabEnv) -> torch.Tensor:
    """A strongly connected comparison graph: a directed cycle over every sequence, plus
    its reverse on a few edges so no vertex is a source or a sink. The BT MLE is finite,
    so anything that fails to converge here is a property of the LOSS, not of the data."""
    n = env.n_seqs
    rows = [[i, (i + 1) % n] for i in range(n)]
    rows += [[(i + 1) % n, i] for i in range(0, n, 3)]
    return _pairs(rows)


def _strongly_connected(pairs: torch.Tensor, n: int) -> bool:
    p = pairs.numpy()
    adj = np.zeros((n, n), dtype=bool)
    adj[p[:, 0], p[:, 1]] = True
    np.fill_diagonal(adj, True)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        adj = adj @ adj
    return bool(adj.all())


class Ctx:
    """Fixtures. Lazily built, each with its own seed, so a stage costs nothing unless run.

    ONE general environment and ONE length ladder. Every experiment below changes exactly
    one thing about one of them.
    """

    BETA = 0.5
    EPS = 0.1
    TAU = 2.0
    GAMMA = 0.5

    def __init__(self, fast: bool = False):
        self.fast = fast
        self.long_steps = 1_500 if fast else 8_000
        self.mid_steps = 800 if fast else 3_000
        self.opt_steps = 4_000 if fast else 20_000

    @cached_property
    def env(self) -> TabEnv:
        return TabEnv(vocab=3, t_max=3)

    @cached_property
    def ladder(self) -> TabEnv:
        return TabEnv(vocab=2, t_max=5)

    @cached_property
    def theta_ref(self) -> torch.Tensor:
        return self.env.random_theta(seed=0, scale=0.7)

    @cached_property
    def theta_ref_ladder(self) -> torch.Tensor:
        return self.ladder.random_theta(seed=3, scale=0.5)

    @cached_property
    def lp_ref(self) -> torch.Tensor:
        with torch.no_grad():
            return self.env.seq_logprobs(self.theta_ref)

    @cached_property
    def matching(self) -> torch.Tensor:
        return _matching(self.env)

    @cached_property
    def connected(self) -> torch.Tensor:
        return _connected(self.env)

    @cached_property
    def long_beats_short(self) -> torch.Tensor:
        """On the ladder, the longer sequence always wins. Length is the only signal."""
        n = self.ladder.n_seqs
        rows = []
        for i in range(n):
            for j in range(n):
                if self.ladder.lengths[i] > self.ladder.lengths[j]:
                    rows.append([i, j])
        return _pairs(rows)

    @cached_property
    def equal_length(self) -> torch.Tensor:
        """Pairs drawn ONLY from same-length sequences: a fixture with no length confound
        at all, so anything a length-sensitive method does here is a side effect."""
        L = self.env.lengths
        rows = []
        for i in range(self.env.n_seqs):
            for j in range(self.env.n_seqs):
                if i != j and float(L[i]) == float(L[j]) and (i + j) % 3 == 0 and i < j:
                    rows.append([i, j])
        return _pairs(rows)

    def specs(self) -> dict[str, Spec]:
        return named_specs(beta=self.BETA, eps=self.EPS, tau=self.TAU, gamma=self.GAMMA)

    # --- KTO: prompt-conditioned, so z_ref is a real quantity --------------------

    @cached_property
    def penv(self) -> PromptEnv:
        return PromptEnv(vocab=3, t_max=3, n_prompts=4)

    @cached_property
    def theta_ref_prompt(self) -> torch.Tensor:
        return self.penv.random_theta(seed=11, scale=0.7)

    @cached_property
    def lp_ref_prompt(self) -> torch.Tensor:
        with torch.no_grad():
            return self.penv.logprobs(self.theta_ref_prompt)

    @cached_property
    def kto_data(self) -> torch.Tensor:
        """UNPAIRED examples (prompt, sequence, +-1), SEPARABLE by construction.

        For each prompt the desirable and undesirable sets are disjoint, so no finite
        reward assignment is ever 'wrong' -- exactly D1's separability, in unpaired form.
        A BT-style loss on the induced pairs therefore has its optimum at infinity, which
        is what makes this the right fixture for testing KTO's boundedness.
        """
        rows = []
        for p in range(self.penv.n_prompts):
            for i in range(self.penv.n_seqs):
                m = (i + p) % 3
                if m == 0:
                    rows.append([p, i, 1])
                elif m == 1:
                    rows.append([p, i, -1])
        return torch.tensor(rows, dtype=torch.long)

    @cached_property
    def kto_pairs(self) -> torch.Tensor:
        """The SAME labels made paired: (prompt, desirable, undesirable). Only used to give
        DPO something to run on, so the comparison is data-matched."""
        return _pairs_from(self.kto_data, self.penv.n_prompts)

    @cached_property
    def kto_noisy(self) -> torch.Tensor:
        """`kto_data` with a fixed 10% of the labels FLIPPED -- no longer separable."""
        ex = self.kto_data.clone()
        g = torch.Generator().manual_seed(4)
        idx = torch.randperm(ex.shape[0], generator=g)[: max(1, ex.shape[0] // 10)]
        ex[idx, 2] *= -1
        return ex


def _fail_fast(ctx: Ctx) -> None:
    """Touch every TODO on a 1-element input so a missing body surfaces BEFORE the
    expensive training runs."""
    env, lp_ref = ctx.env, ctx.lp_ref
    lp = env.seq_logprobs(ctx.theta_ref)
    implicit_reward("ref_sum", lp, lp_ref, env.lengths, 0.5)
    link("logsigmoid", torch.zeros(1), torch.zeros(1), Spec("t", "ref_sum", "logsigmoid"))
    variant_loss(Spec("t", "ref_sum", "logsigmoid", beta=0.5), lp, lp_ref,
                 env.lengths, ctx.matching)


# ---- individual stages ----------------------------------------------------


def stage_reward(ctx: Ctx) -> None:
    """implicit_reward -- the f axis, incl. the uniform-reference result"""
    env = ctx.env
    theta = env.random_theta(seed=1, scale=1.1)
    with torch.no_grad():
        lp = env.seq_logprobs(theta)
        lp_ref = ctx.lp_ref
        L = env.lengths
        b = 0.5

        rs = implicit_reward("ref_sum", lp, lp_ref, L, b)
        rm = implicit_reward("ref_mean", lp, lp_ref, L, b)
        ns = implicit_reward("noref_sum", lp, lp_ref, L, b)
        nm = implicit_reward("noref_mean", lp, lp_ref, L, b)
        od = implicit_reward("odds", lp, lp_ref, L, b)

        _check("shapes are (n_seqs,) for every kind",
               all(r.shape == (env.n_seqs,) for r in (rs, rm, ns, nm, od)),
               f"{tuple(rs.shape)}")
        _check("ref_sum is EXACTLY 0 when pi == pi_ref",
               float(implicit_reward("ref_sum", lp_ref, lp_ref, L, b).abs().max()) == 0.0)
        _check("ref_mean == ref_sum / |y| elementwise",
               torch.allclose(rm, rs / L, atol=1e-15, rtol=0.0),
               f"max dev {float((rm - rs / L).abs().max()):.3e}")
        _check("noref_mean == noref_sum / |y| elementwise",
               torch.allclose(nm, ns / L, atol=1e-15, rtol=0.0),
               f"max dev {float((nm - ns / L).abs().max()):.3e}")
        _check("noref_sum == beta * log pi", torch.allclose(ns, b * lp, atol=1e-15,
                                                            rtol=0.0))
        _check("ref_sum == noref_sum - beta*log pi_ref",
               torch.allclose(rs, ns - b * lp_ref, atol=1e-14, rtol=0.0),
               f"max dev {float((rs - (ns - b * lp_ref)).abs().max()):.3e}")

        P = torch.exp(lp / L)
        _check("odds is beta*log[P/(1-P)] with P = exp(mean-token logprob)",
               torch.allclose(od, b * torch.log(P / (1 - P)), atol=1e-12, rtol=0.0),
               f"max dev {float((od - b * torch.log(P / (1 - P))).abs().max()):.3e}")
        _check("odds is monotone in P",
               bool(((od.argsort() == P.argsort()).all())))

        # --- the uniform-reference result (Block 2's correction, measured) -----------
        lp_unif = env.seq_logprobs(env.uniform_theta())
        logV = math.log(env.vocab)
        _check("uniform pi_ref is exactly V^-|y|",
               torch.allclose(lp_unif, -L * logV, atol=1e-14, rtol=0.0),
               f"log V = {logV:.6f}")

        rs_u = implicit_reward("ref_sum", lp, lp_unif, L, b)
        rm_u = implicit_reward("ref_mean", lp, lp_unif, L, b)
        _check("SUM + uniform ref == reference-free PLUS a length bonus of beta*log V/token",
               torch.allclose(rs_u, ns + b * L * logV, atol=1e-13, rtol=0.0),
               f"bonus {b * logV:.6f} nats/token")
        _check("MEAN + uniform ref == reference-free PLUS a CONSTANT",
               torch.allclose(rm_u, nm + b * logV, atol=1e-13, rtol=0.0))

        w, l = ctx.matching[:, 0], ctx.matching[:, 1]
        d_mean_u = (rm_u[w] - rm_u[l])
        d_mean_free = (nm[w] - nm[l])
        d_sum_u = (rs_u[w] - rs_u[l])
        d_sum_free = (ns[w] - ns[l])
        _check("=> under MEAN the constant CANCELS: uniform-ref is bitwise reference-FREE",
               float((d_mean_u - d_mean_free).abs().max()) < 1e-13,
               f"max dev {float((d_mean_u - d_mean_free).abs().max()):.3e}")
        gap = float((d_sum_u - d_sum_free).abs().max())
        _check("=> under SUM it does NOT cancel: a pure length reward, not a trust region",
               gap > 0.1, f"margins shift by up to {gap:.6f}")
        print(f"     uniform-reference length bonus = {b * logV:.6f} nats/token "
              f"(beta={b}, V={env.vocab}); at V=50257 that is {b * math.log(50257):.4f}")


def stage_link(ctx: Ctx) -> None:
    """link -- the l axis, incl. the difference-only invariance that isolates KTO"""
    spec = Spec("t", "ref_sum", "logsigmoid", beta=1.0, eps=0.1, tau=2.0, gamma=0.5,
                lam_d=1.0, lam_u=1.5)

    z = torch.zeros(1)
    _check("logsigmoid at d=0 is +log 2 (the standing sign probe)",
           abs(float(link("logsigmoid", z, z, spec)) - math.log(2.0)) < 1e-15,
           f"{float(link('logsigmoid', z, z, spec)):.6f}")
    _check("smoothed at d=0 is also +log 2 (eps only tilts the target, not the value at 0)",
           abs(float(link("logsigmoid_smoothed", z, z, spec)) - math.log(2.0)) < 1e-15)

    # the additive probe: each term must be at a MINIMUM where it should be happiest
    tau = spec.tau
    at_tau = float(link("squared_target", torch.tensor([tau]), z, spec))
    near = [float(link("squared_target", torch.tensor([tau + e]), z, spec))
            for e in (-0.3, -0.05, 0.05, 0.3)]
    _check("squared_target reads exactly 0 at d = tau", abs(at_tau) < 1e-28, f"{at_tau:.3e}")
    _check("squared_target is a MINIMUM at d = tau, not a maximum",
           all(v > at_tau for v in near), f"neighbours {[round(v, 4) for v in near]}")

    big = torch.tensor([200.0])
    _check("logsigmoid stays finite and positive at d = -200 (no naive log(sigmoid))",
           0.0 < float(link("logsigmoid", -big, z, spec)) < 1e9,
           f"{float(link('logsigmoid', -big, z, spec)):.4f}")
    _check("logsigmoid stays strictly positive at d = +200 (no saturation to exactly 0)",
           float(link("logsigmoid", big, z, spec)) >= 0.0)

    # Boundedness is the point, so probe it where an UNBOUNDED link would blow up.
    huge = torch.tensor([-1e6, 0.0, 1e6])
    pro = link("prospect", huge, -huge, spec)
    ls = link("logsigmoid", huge, -huge, spec)
    cap = spec.lam_d + spec.lam_u
    _check("prospect stays in [0, lam_d + lam_u] even at |r| = 1e6",
           bool(((pro >= 0.0) & (pro <= cap)).all()) and bool(torch.isfinite(pro).all()),
           f"range [{float(pro.min()):.4f}, {float(pro.max()):.4f}], cap {cap}")
    _check("...while logsigmoid on the same input is unbounded -- bounded LOSS, not finite optimum",
           float(ls.max()) > 1e5, f"logsigmoid max {float(ls.max()):.4g} vs prospect cap {cap}")
    print("     NB: the bound caps each example's INFLUENCE, not the optimum. `kto` shows the")
    print("     margin still diverging on separable data -- inf sigma(-u) = 0 only at u = +inf.")
    mid = link("prospect", torch.tensor([-6.0, 6.0]), torch.tensor([6.0, -6.0]), spec)
    _check("prospect is ASYMMETRIC: swapping which side is extreme changes the loss",
           abs(float(mid[0]) - float(mid[1])) > 1e-6,
           f"{float(mid[0]):.6f} vs {float(mid[1]):.6f}")

    # --- the invariance that defines the axis -----------------------------------
    g = torch.Generator().manual_seed(7)
    r_w = torch.randn(9, generator=g)
    r_l = torch.randn(9, generator=g)
    c = 3.7
    print("     common-shift invariance (add c=+3.7 to BOTH r_w and r_l):")
    for kind in L_KINDS:
        a = link(kind, r_w, r_l, spec)
        b = link(kind, r_w + c, r_l + c, spec)
        dev = float((a - b).abs().max())
        print(f"       {kind:<20} max|L - L_shifted| = {dev:.3e}")
        if kind == "prospect":
            _check("prospect is NOT difference-only (it sees the marginals)", dev > 0.05,
                   f"{dev:.6f}")
        else:
            _check(f"{kind} is a function of d alone", dev < 1e-14, f"{dev:.3e}")
    print("     => four links cannot see a move that lowers BOTH logprobs. That invariance")
    print("        IS 'both logprobs fall', and it is why they all REQUIRE pairs.")


def stage_optima(ctx: Ctx) -> None:
    """closed-form single-pair optima, checked against a numerical minimiser"""
    spec = Spec("t", "ref_sum", "logsigmoid", beta=ctx.BETA, eps=ctx.EPS, tau=ctx.TAU,
                gamma=ctx.GAMMA)
    print(f"     beta={spec.beta}  eps={spec.eps}  tau={spec.tau}  gamma={spec.gamma}")
    for kind in ("logsigmoid", "logsigmoid_smoothed", "squared_target",
                 "logsigmoid_margin"):
        for f in (0.6, 0.8):
            cf = closed_form_optimum(kind, f, spec)
            num = numeric_optimum(kind, f, spec, n_steps=ctx.opt_steps)
            _check(f"{kind} optimum at f={f}: closed form == numerical",
                   abs(cf - num) < 2e-4, f"closed {cf:.6f} vs numeric {num:.6f}")

    # cDPO's cap IS D1's eps-cap, in BOTH units. D3 measured the log-ratio column.
    d3 = Spec("t", "ref_sum", "logsigmoid_smoothed", beta=0.3, eps=0.10)
    m_star = closed_form_optimum("logsigmoid_smoothed", 1.0, d3)
    _check("cDPO at f=1, eps=0.10 reproduces D3's preference logit M* = 2.197225",
           abs(m_star - 2.197225) < 1e-5, f"{m_star:.6f}")
    print(f"     {'eps':>6}{'M* = logit(1-eps)':>20}{'Dhat* = M*/beta':>18}   (beta=0.3)")
    for e, want in ((0.05, 9.814797), (0.10, 7.324082), (0.20, 4.620981),
                    (0.35, 2.063464)):
        s = replace(d3, eps=e)
        m = closed_form_optimum("logsigmoid_smoothed", 1.0, s)
        dh = logratio_optimum("logsigmoid_smoothed", 1.0, s)
        print(f"     {e:>6.2f}{m:>20.6f}{dh:>18.6f}")
        _check(f"cDPO log-ratio cap at eps={e:.2f} reproduces D3's {want}",
               abs(dh - want) < 1e-5, f"{dh:.6f}")

    _check("DPO's optimum diverges as f -> 1 (Ford, in one scalar)",
           closed_form_optimum("logsigmoid", 1 - 1e-9, spec) > 15.0,
           f"{closed_form_optimum('logsigmoid', 1 - 1e-9, spec):.4f} at f=1-1e-9")
    _check("IPO's optimum stays at tau as f -> 1",
           abs(closed_form_optimum("squared_target", 1.0, spec) - spec.tau) < 1e-15)

    # The margin link is the one with NO closed form -- and the plausible guess is wrong.
    naive = logit(0.6) + spec.gamma
    true = closed_form_optimum("logsigmoid_margin", 0.6, spec)
    print(f"     margin link at f=0.60, gamma={spec.gamma}: "
          f"naive 'logit(f)+gamma' = {naive:.6f}, TRUTH = {true:.6f}")
    _check("the guess 'M* = logit(f) + gamma' is WRONG -- gamma breaks BT's antisymmetry",
           abs(naive - true) > 0.3, f"{naive:.6f} vs {true:.6f}")
    _check("...but at gamma = 0 the margin link collapses back to plain BT",
           abs(closed_form_optimum("logsigmoid_margin", 0.6, replace(spec, gamma=0.0))
               - logit(0.6)) < 1e-9)
    _check("margin-DPO still DIVERGES as f -> 1: gamma shifts P1, it does not fix it",
           closed_form_optimum("logsigmoid_margin", 1 - 1e-9, spec) > 15.0,
           f"{closed_form_optimum('logsigmoid_margin', 1 - 1e-9, spec):.4f}")
    print("     => only two of the five links have a FINITE optimum at f = 1:")
    print("        squared_target (tau) and logsigmoid_smoothed (eps). Adding a margin")
    print("        to BT moves the runaway; it does not stop it.")


def stage_reduce(ctx: Ctx) -> None:
    """the grid REPRODUCES the specials, elementwise, and the check is non-vacuous"""
    _fail_fast(ctx)
    env, lp_ref = ctx.env, ctx.lp_ref
    theta = env.random_theta(seed=2, scale=1.3)
    pairs = ctx.connected
    specs = ctx.specs()
    with torch.no_grad():
        lp = env.seq_logprobs(theta)
        L = env.lengths

        def grid(spec):
            r = implicit_reward(spec.f_kind, lp, lp_ref, L, spec.beta)
            return link(spec.l_kind, r[pairs[:, 0]], r[pairs[:, 1]], spec)

        cases = [
            ("DPO", grid(specs["DPO"]), _direct_dpo(lp, lp_ref, pairs, ctx.BETA)),
            ("cDPO", grid(specs["cDPO"]),
             _direct_cdpo(lp, lp_ref, pairs, ctx.BETA, ctx.EPS)),
            ("IPO", grid(specs["IPO"]), _direct_ipo(lp, lp_ref, pairs, ctx.TAU)),
            ("SimPO", grid(specs["SimPO"]),
             _direct_simpo(lp, L, pairs, ctx.BETA, ctx.GAMMA)),
        ]
        for name, a, b in cases:
            dev = float((a - b).abs().max())
            _check(f"{name}: grid == independent direct implementation, elementwise",
                   dev < 1e-14, f"max dev {dev:.3e}")

        orpo = variant_loss(specs["ORPO"], lp, lp_ref, L, pairs)
        direct = _direct_orpo(lp, L, pairs, specs["ORPO"].or_weight)
        _check("ORPO: grid loss (odds + logsigmoid + SFT) == direct implementation",
               abs(float(orpo) - float(direct)) < 1e-13,
               f"{float(orpo):.12f} vs {float(direct):.12f}")

        # KTO's z_ref: the mismatch roll is a DELIBERATE NO-OP in this env, and saying so
        # out loud is worth more than quietly relying on it.
        r_kto = implicit_reward("ref_sum", lp, lp_ref, L, ctx.BETA)
        l_idx = pairs[:, 1]
        matched, rolled = r_kto[l_idx], r_kto[l_idx.roll(1)]
        _check("z_ref: rolling y_l is a PERMUTATION -- same multiset, so the mean is the "
               "same to summation-order noise",
               bool(torch.equal(torch.sort(matched).values, torch.sort(rolled).values))
               and abs(float(matched.mean()) - float(rolled.mean())) < 1e-14,
               f"mean {float(matched.mean()):.17f} vs {float(rolled.mean()):.17f}")
        print("     => the roll cannot matter without PROMPT CONDITIONING. Real KTO scores")
        print("        y_{i+1} under x_i, a different SET of numbers (the log-ratio on")
        print("        MISMATCHED prompt/completion combinations), not a permutation of the")
        print("        matched ones. TabEnv is unconditional, so the mechanism degenerates.")
        z = rolled.mean().clamp(min=0.0)
        _check("z_ref is clamped at 0 (it estimates a KL, which cannot be negative)",
               float(z) >= 0.0, f"raw mean {float(rolled.mean()):.6f} -> z_ref {float(z):.6f}")
        z0 = implicit_reward("ref_sum", lp_ref, lp_ref, L, ctx.BETA)[l_idx].mean()
        _check("...and at pi == pi_ref it is EXACTLY 0 -- which is why a hardcoded z_ref=0 "
               "is useless: it calls every example neutral at init",
               float(z0) == 0.0, f"{float(z0):.3e}")

        # non-vacuity: how far apart are the WRONG cells?
        dpo_v = cases[0][1]
        gaps = []
        for name, a, _ in cases[1:]:
            gaps.append((name, float((a - dpo_v).abs().mean())))
        worst = min(g for _, g in gaps)
        _check("non-vacuous: every other named cell differs from DPO by a lot",
               worst > 1e-2,
               "  ".join(f"{n} {g:.4f}" for n, g in gaps))

        # every grid cell must at least evaluate -- the empty cells are reachable
        n_ok = 0
        for fk in F_KINDS:
            for lk in L_KINDS:
                s = Spec(f"{fk}+{lk}", fk, lk, beta=ctx.BETA, eps=ctx.EPS, tau=ctx.TAU,
                         gamma=ctx.GAMMA, lam_u=1.5)
                v = variant_loss(s, lp, lp_ref, L, pairs)
                assert torch.isfinite(v), (fk, lk)
                n_ok += 1
        _check("all 25 grid cells evaluate finitely (the empty cells are REACHABLE)",
               n_ok == 25, f"{n_ok}/25")


def stage_separable(ctx: Ctx) -> None:
    """THE HEADLINE: on D1's matching, DPO runs away; cDPO and IPO stop where predicted"""
    _fail_fast(ctx)
    env, pairs = ctx.env, ctx.matching
    _check("the matching fixture is NOT strongly connected (Ford: MLE at infinity)",
           not _strongly_connected(pairs, env.n_seqs))

    theta_ref = ctx.theta_ref
    specs = ctx.specs()
    bound = point_mass_kl_bound(ctx.lp_ref)
    rows = {}
    for name in ("DPO", "cDPO", "IPO"):
        # ONE lr for all three arms: this stage is a comparison, so the optimiser must
        # not be a confound. lr=2.0 makes IPO's squared loss diverge to NaN outright.
        rows[name] = train_variant(env, theta_ref.clone(), theta_ref, pairs,
                                   specs[name], n_steps=ctx.long_steps, lr=0.5)

    def at(r, frac):
        return r["margin"][int((len(r["margin"]) - 1) * frac)]

    print(f"     steps={ctx.long_steps}   point-mass KL bound = {bound:.4f}")
    print(f"     {'variant':<8}{'m(T/4)':>10}{'m(T/2)':>10}{'m(T)':>10}"
          f"{'d1':>9}{'d2':>9}{'KL':>9}{'acc':>7}")
    for name, r in rows.items():
        q, h, e = at(r, 0.25), at(r, 0.5), r["margin"][-1]
        print(f"     {name:<8}{q:>10.4f}{h:>10.4f}{e:>10.4f}"
              f"{h - q:>9.4f}{e - h:>9.4f}{r['kl'][-1]:>9.4f}{r['acc'][-1]:>7.3f}")

    d = rows["DPO"]
    dq, dh, de = at(d, 0.25), at(d, 0.5), d["margin"][-1]
    _check("DPO: accuracy is 1.0 -- the metric that reports nothing",
           d["acc"][-1] == 1.0, f"{d['acc'][-1]:.3f}")
    # The right signature is NOT '% growth' (which decays even for a divergent run) but
    # a roughly CONSTANT increment per DOUBLING of the step budget: margin ~ log t.
    inc1, inc2 = dh - dq, de - dh
    _check("DPO: the margin gains a full nat over each doubling of the budget",
           inc1 > 0.5 and inc2 > 0.5, f"T/4->T/2 {inc1:.4f}, T/2->T {inc2:.4f}")
    _check("DPO: those two increments are EQUAL -- margin ~ log t, the separable signature",
           abs(inc2 - inc1) / inc1 < 0.20, f"{inc1:.4f} vs {inc2:.4f}")
    _check("DPO: KL rises monotonically",
           bool(np.all(np.diff(d["kl"]) > -1e-9)))
    _check("DPO: KL stays under the point-mass bound (D2 -- 'unbounded' is false here)",
           d["kl"][-1] < bound, f"{d['kl'][-1]:.4f} < {bound:.4f}")

    for name in ("cDPO", "IPO"):
        want = closed_form_optimum(specs[name].l_kind, 1.0, specs[name])
        r = rows[name]
        inc_a, inc_b = at(r, 0.5) - at(r, 0.25), r["margin"][-1] - at(r, 0.5)
        # Scale-free: the point is not that the increment is small in absolute terms
        # (that depends on the budget) but that it is ORDERS below DPO's and SHRINKING,
        # while DPO's is CONSTANT.
        _check(f"{name}: the margin increment is a small fraction of DPO's",
               abs(inc_b) < 0.15 * inc2, f"{inc_b:.3e} vs DPO's {inc2:.4f}")
        _check(f"{name}: and it is SHRINKING, not constant (this is convergence)",
               abs(inc_b) <= abs(inc_a) + 1e-12,
               f"{inc_a:.3e} -> {inc_b:.3e}  (DPO: {inc1:.4f} -> {inc2:.4f})")
        tol = 3e-2 if ctx.fast else 1e-4
        _check(f"{name}: it is converging on the CLOSED-FORM optimum {want:.6f}",
               abs(r["margin"][-1] - want) < tol,
               f"got {r['margin'][-1]:.6f}, want {want:.6f} (tol {tol})")
        _check(f"{name}: KL is far below DPO's on identical data",
               r["kl"][-1] < 0.75 * d["kl"][-1],
               f"{r['kl'][-1]:.4f} vs DPO {d['kl'][-1]:.4f} "
               f"({r['kl'][-1] / d['kl'][-1]:.3f}x)")

    print("     => cDPO stops because the DATA stops being trustworthy (eps);")
    print("        IPO stops because you TOLD it to (tau). Same finiteness, different source.")


def stage_ipo_cost(ctx: Ctx) -> None:
    """IPO's symmetric cost, as a compression ratio rather than a caveat"""
    spec_dpo = Spec("d", "ref_sum", "logsigmoid", beta=ctx.BETA)
    spec_ipo = Spec("i", "ref_sum", "squared_target", tau=ctx.TAU)
    strong, weak = 0.99, 0.60

    dpo_s = closed_form_optimum("logsigmoid", strong, spec_dpo)
    dpo_w = closed_form_optimum("logsigmoid", weak, spec_dpo)
    ipo_s = closed_form_optimum("squared_target", strong, spec_ipo)
    ipo_w = closed_form_optimum("squared_target", weak, spec_ipo)

    for kind, sp, s_val, w_val in (("logsigmoid", spec_dpo, dpo_s, dpo_w),
                                   ("squared_target", spec_ipo, ipo_s, ipo_w)):
        for f, want in ((strong, s_val), (weak, w_val)):
            got = numeric_optimum(kind, f, sp, n_steps=ctx.opt_steps)
            tol = 0.25 if ctx.fast else 3e-4
            _check(f"{kind} f={f}: numerical optimum matches closed form",
                   abs(got - want) < tol, f"{got:.6f} vs {want:.6f} (tol {tol})")

    r_dpo = dpo_s / dpo_w
    r_ipo = ipo_s / ipo_w
    print(f"     strong pair f={strong}, weak pair f={weak}, tau={ctx.TAU}, beta={ctx.BETA}")
    print(f"     DPO/BT  margins {dpo_s:.4f} vs {dpo_w:.4f}   ratio {r_dpo:.4f}")
    print(f"     IPO     margins {ipo_s:.4f} vs {ipo_w:.4f}   ratio {r_ipo:.4f}")
    _check("IPO COMPRESSES the strong/weak preference ratio relative to BT",
           r_ipo < 0.6 * r_dpo, f"{r_ipo:.4f} vs {r_dpo:.4f} ({r_dpo / r_ipo:.2f}x)")
    _check("IPO's margin is bounded by tau no matter how strong the preference",
           abs(closed_form_optimum("squared_target", 1.0, spec_ipo)) <= ctx.TAU + 1e-12)
    _check("BT's margin is NOT bounded (it is the same object that diverges in `separable`)",
           closed_form_optimum("logsigmoid", 1 - 1e-12, spec_dpo) > 20.0)
    print("     => the fix has a price, and the price is that a near-certain preference")
    print("        and a 60/40 preference are forced into the same bounded band.")


def stage_orpo(ctx: Ctx) -> None:
    """ORPO: 2x2 ablation + a beta sweep. BOTH ingredients matter; only SFT is robust"""
    _fail_fast(ctx)
    env, pairs = ctx.env, ctx.connected
    _check("the ORPO fixture IS strongly connected (so nothing here is a divergence artifact)",
           _strongly_connected(pairs, env.n_seqs))
    theta_ref = ctx.theta_ref
    specs = ctx.specs()

    arms = {
        "DPO": specs["DPO"],
        "DPO+SFT": replace(specs["DPO"], name="DPO+SFT", sft_weight=1.0),
        "ORPO": specs["ORPO"],
        "ORPO-noSFT": replace(specs["ORPO"], name="ORPO-noSFT", sft_weight=0.0),
    }
    out = {k: train_variant(env, theta_ref.clone(), theta_ref, pairs, s,
                            n_steps=ctx.mid_steps, lr=1.0) for k, s in arms.items()}

    print(f"     steps={ctx.mid_steps}, strongly-connected fixture")
    print(f"     {'arm':<12}{'chosen NLL/tok':>16}{'d(NLL)':>10}"
          f"{'d logp_w':>10}{'d logp_l':>10}{'margin':>9}{'acc':>7}")
    for k, r in out.items():
        print(f"     {k:<12}{r['nll_w'][-1]:>16.6f}{r['nll_w'][-1] - r['nll_w'][0]:>10.6f}"
              f"{r['logp_w'][-1] - r['logp_w'][0]:>10.4f}"
              f"{r['logp_l'][-1] - r['logp_l'][0]:>10.4f}"
              f"{r['margin'][-1]:>9.4f}{r['acc'][-1]:>7.3f}")

    for k in ("DPO+SFT", "ORPO"):
        _check(f"{k}: chosen NLL/token IMPROVES", out[k]["nll_w"][-1] < out[k]["nll_w"][0],
               f"{out[k]['nll_w'][0]:.6f} -> {out[k]['nll_w'][-1]:.6f}")
        _check(f"{k}: it still learns the preference (margin > 0, acc > 0.5)",
               out[k]["margin"][-1] > 0 and out[k]["acc"][-1] > 0.5,
               f"margin {out[k]['margin'][-1]:.4f} acc {out[k]['acc'][-1]:.3f}")

    # The odds link is NOT neutral -- an honest ablation has to say so and say why.
    d_odds = out["ORPO-noSFT"]["nll_w"][-1] - out["DPO"]["nll_w"][-1]
    d_sft = out["ORPO"]["nll_w"][-1] - out["ORPO-noSFT"]["nll_w"][-1]
    print(f"     decomposition of chosen NLL vs DPO:  odds link {d_odds:+.6f},"
          f"  then adding SFT {d_sft:+.6f}")
    _check("BOTH ingredients help -- 'it is only the SFT term' is too glib",
           d_odds < 0 and d_sft < 0, f"odds {d_odds:+.6f}, sft {d_sft:+.6f}")

    # P2 (does the REJECTED logprob fall?) is the sharper question. Only ONE of the two
    # ingredients answers it ROBUSTLY -- and establishing that needs a beta sweep, because
    # the odds arm's answer FLIPS SIGN with beta. This is the stage's real lesson.
    _check("DPO: the rejected logprob falls -- this is P2",
           out["DPO"]["logp_l"][-1] - out["DPO"]["logp_l"][0] < 0,
           f"{out['DPO']['logp_l'][-1] - out['DPO']['logp_l'][0]:+.4f}")
    for k in ("DPO+SFT", "ORPO"):
        rise = out[k]["logp_l"][-1] - out[k]["logp_l"][0]
        _check(f"{k}: the SFT term turns the rejected logprob AROUND", rise > 0,
               f"{rise:+.4f}")

    print(f"     {'beta':>6}{'d logp_l (odds, no SFT)':>26}{'d logp_l (odds + SFT)':>24}")
    signs = []
    for b in (0.25, 0.5, 1.0, 2.0):
        base = replace(specs["ORPO"], beta=b)
        no = train_variant(env, theta_ref.clone(), theta_ref, pairs,
                           replace(base, sft_weight=0.0), n_steps=ctx.mid_steps, lr=1.0)
        yes = train_variant(env, theta_ref.clone(), theta_ref, pairs, base,
                            n_steps=ctx.mid_steps, lr=1.0)
        dn = no["logp_l"][-1] - no["logp_l"][0]
        dy = yes["logp_l"][-1] - yes["logp_l"][0]
        signs.append((dn, dy))
        print(f"     {b:>6.2f}{dn:>26.4f}{dy:>24.4f}")
    _check("the ODDS-ALONE answer to P2 is NOT robust: it changes sign across beta",
           any(a < 0 for a, _ in signs) and any(a > 0 for a, _ in signs),
           " ".join(f"{a:+.3f}" for a, _ in signs))
    _check("the SFT answer to P2 IS robust: rejected logprob rises at EVERY beta",
           all(b > 0 for _, b in signs), " ".join(f"{b:+.3f}" for _, b in signs))

    # WHY neither ingredient is neutral, although the LINK is difference-only in both.
    # This also corrects a natural over-generalisation: "mean" rewards are affine in
    # log pi but with a LENGTH-DEPENDENT SLOPE, so they are only shift-invariant on
    # equal-length pairs.
    lp = ctx.lp_ref.clone()
    L = ctx.env.lengths
    w, l = pairs[:, 0], pairs[:, 1]
    c, b = 0.37, 0.5
    for fk in ("ref_sum", "noref_sum", "ref_mean", "noref_mean", "odds"):
        r0 = implicit_reward(fk, lp, ctx.lp_ref, L, b)
        r1 = implicit_reward(fk, lp + c, ctx.lp_ref, L, b)
        moved = (r1[w] - r1[l]) - (r0[w] - r0[l])
        shift = float(moved.abs().max())
        print(f"     add {c} to EVERY logprob: margin under {fk:<11} moves {shift:.4e}")
        if fk.endswith("_sum"):
            _check(f"{fk}: slope is 1 for every sequence -> a common fall is INVISIBLE",
                   shift < 1e-12, f"{shift:.3e}")
        elif fk.endswith("_mean"):
            want = b * c * (1.0 / L[w].double() - 1.0 / L[l].double())
            _check(f"{fk}: shifts by EXACTLY beta*c*(1/|y_w| - 1/|y_l|), not by 0",
                   float((moved - want).abs().max()) < 1e-12,
                   f"max dev {float((moved - want).abs().max()):.3e}, "
                   f"max shift {shift:.4f}")
        else:
            _check("odds: NONLINEAR in log pi, so a common fall always moves d",
                   shift > 1e-3, f"{shift:.3e}")

    eq = ctx.equal_length
    r0 = implicit_reward("ref_mean", lp, ctx.lp_ref, L, b)
    r1 = implicit_reward("ref_mean", lp + c, ctx.lp_ref, L, b)
    dev = float(((r1[eq[:, 0]] - r1[eq[:, 1]])
                 - (r0[eq[:, 0]] - r0[eq[:, 1]])).abs().max())
    _check("...and on EQUAL-LENGTH pairs ref_mean IS shift-invariant again", dev < 1e-12,
           f"{dev:.3e}")

    print("     => P2 comes from the loss being a function of an AFFINE, EQUAL-SLOPE")
    print("        difference. Both of ORPO's ingredients break that -- L_SFT is not a")
    print("        difference at all, and the odds MAP is not affine (d(log-odds)/d(mean")
    print("        lp) = 1/(1-P) >= 1, so it weights the winner more as P_w -> 1). Length")
    print("        normalisation breaks it too, but only when the lengths DIFFER.")
    print("        Only L_SFT breaks it in a direction that does not depend on beta.")
    print("        Moral: run the ablation at more than one hyperparameter before you")
    print("        name a mechanism.")


def stage_simpo(ctx: Ctx) -> None:
    """SimPO unbundled: 1/|y| AMPLIFIES length bias at matched margin; noref costs entropy"""
    _fail_fast(ctx)

    # --- (1) LENGTH. One axis moves: ref_sum -> ref_mean. Same reference. -----------
    # MATCHED MARGIN (D2's design) and BOTH label directions. Run to convergence instead
    # and the ladder saturates at |y| = t_max, which hides the effect entirely.
    lad = ctx.ladder
    theta_ref = ctx.theta_ref_ladder
    long_wins = ctx.long_beats_short
    short_wins = long_wins.flip(1).contiguous()
    with torch.no_grad():
        base_len = expected_length(lad.seq_logprobs(theta_ref), lad.lengths)

    print(f"     length ladder ({lad.n_seqs} sequences, lengths 1..{lad.t_max}), "
          f"E[|y|] at init = {base_len:.4f}")
    print(f"     {'labels':<12}{'margin':>8}{'sum E|y|':>11}{'d(sum)':>9}"
          f"{'mean E|y|':>11}{'d(mean)':>9}")
    cells = {}
    for label, prs in (("LONG wins", long_wins), ("SHORT wins", short_wins)):
        for target in (1.0, 2.0):
            got = {}
            for nm, fk in (("sum", "ref_sum"), ("mean", "ref_mean")):
                r = train_variant(lad, theta_ref.clone(), theta_ref, prs,
                                  Spec(nm, fk, "logsigmoid", beta=ctx.BETA),
                                  n_steps=ctx.long_steps, lr=1.0, stop_margin=target)
                _check(f"{label} @{target} {nm}: reached the matched margin",
                       r["stopped_at"] >= 0, f"stopped at {r['stopped_at']}")
                got[nm] = r["elen"][-1] - base_len
                cells[(label, target, nm)] = got[nm]
            print(f"     {label:<12}{target:>8.1f}"
                  f"{base_len + got['sum']:>11.4f}{got['sum']:>+9.4f}"
                  f"{base_len + got['mean']:>11.4f}{got['mean']:>+9.4f}")

    _check("when LONG wins, both reductions lengthen",
           all(cells[("LONG wins", t, nm)] > 0 for t in (1.0, 2.0)
               for nm in ("sum", "mean")))
    _check("when SHORT wins, both reductions shorten",
           all(cells[("SHORT wins", t, nm)] < 0 for t in (1.0, 2.0)
               for nm in ("sum", "mean")))
    _check("MEAN moves length FURTHER than SUM in all four cells -- length "
           "normalisation AMPLIFIES the labels' length signal, it does not brake it",
           all(abs(cells[(lb, t, "mean")]) > abs(cells[(lb, t, "sum")])
               for lb in ("LONG wins", "SHORT wins") for t in (1.0, 2.0)),
           "  ".join(f"{lb[:5]}@{t}: {cells[(lb, t, 'sum')]:+.3f} -> "
                     f"{cells[(lb, t, 'mean')]:+.3f}"
                     for lb in ("LONG wins", "SHORT wins") for t in (1.0, 2.0)))
    print("     => THIS CONTRADICTS THE USUAL STORY. Dividing by |y| does not oppose")
    print("        length; it makes SHORT sequences the cheapest lever (r = b*dlogp/|y|,")
    print("        so d(r)/d(logp) = b/|y| is largest at |y| = 1). The optimiser buys")
    print("        margin at the short end, which pushes E[|y|] further in WHICHEVER")
    print("        direction the labels point. SimPO shortens outputs on real data")
    print("        because of gamma and reference-freeness, not because 1/|y| is a brake.")

    # --- (2) THE PRICE. One axis moves: ref_sum -> noref_sum. Same reduction. -------
    env, eq = ctx.env, ctx.equal_length
    L = env.lengths
    _check("the no-confound fixture has |y_w| == |y_l| in EVERY pair",
           bool((L[eq[:, 0]] == L[eq[:, 1]]).all()), f"{eq.shape[0]} pairs")

    theta_ref2 = ctx.theta_ref
    ref_spec = Spec("ref", "ref_sum", "logsigmoid", beta=ctx.BETA)
    free_spec = Spec("free", "noref_sum", "logsigmoid", beta=ctx.BETA)
    target = 3.0
    a = train_variant(env, theta_ref2.clone(), theta_ref2, eq, ref_spec,
                      n_steps=ctx.mid_steps, lr=1.0, stop_margin=target)
    b = train_variant(env, theta_ref2.clone(), theta_ref2, eq, free_spec,
                      n_steps=ctx.mid_steps, lr=1.0, stop_margin=target)
    h0 = entropy(ctx.lp_ref)
    print(f"     no-length-confound fixture, MATCHED MARGIN = {target} "
          f"(D2's matched-margin design)")
    print(f"     {'reward':<12}{'step':>7}{'margin':>10}{'KL':>10}{'H(pi)':>10}{'E[|y|]':>10}")
    for nm, r in (("ref_sum", a), ("noref_sum", b)):
        print(f"     {nm:<12}{r['stopped_at']:>7}{r['margin'][-1]:>10.4f}"
              f"{r['kl'][-1]:>10.4f}{r['ent'][-1]:>10.4f}{r['elen'][-1]:>10.4f}")
    print(f"     H(pi_ref) = {h0:.4f}")
    _check("both arms actually reached the matched margin",
           a["stopped_at"] >= 0 and b["stopped_at"] >= 0,
           f"{a['stopped_at']} / {b['stopped_at']}")
    _check("REFERENCE-FREE collapses entropy further at the SAME margin",
           b["ent"][-1] < a["ent"][-1],
           f"noref {b['ent'][-1]:.4f} vs ref {a['ent'][-1]:.4f}")
    _check("REFERENCE-FREE moves further from pi_ref at the same margin",
           b["kl"][-1] > a["kl"][-1], f"noref {b['kl'][-1]:.4f} vs ref {a['kl'][-1]:.4f}")
    print("     => the reference-free reward IS log pi, so it rewards LIKELY text.")
    print("        Length normalisation and reference-freeness are separate trades and")
    print("        SimPO ships them together; gamma is fighting the second one.")


def stage_empty_cell(ctx: Ctx) -> None:
    """the empty cell: ref_mean + squared_target -- the two axes compose without interfering"""
    _fail_fast(ctx)
    lad, pairs = ctx.ladder, ctx.long_beats_short
    theta_ref = ctx.theta_ref_ladder
    with torch.no_grad():
        base_len = expected_length(lad.seq_logprobs(theta_ref), lad.lengths)

    grid = {
        "DPO      (ref_sum, logsigmoid)": Spec("a", "ref_sum", "logsigmoid", beta=ctx.BETA),
        "IPO      (ref_sum, squared)": Spec("b", "ref_sum", "squared_target", tau=ctx.TAU),
        "len-DPO  (ref_mean, logsigmoid)": Spec("c", "ref_mean", "logsigmoid",
                                                beta=ctx.BETA),
        "IPO_mean (ref_mean, squared) <-": Spec("d", "ref_mean", "squared_target",
                                                tau=ctx.TAU),
    }
    out = {k: train_variant(lad, theta_ref.clone(), theta_ref, pairs, s,
                            n_steps=ctx.mid_steps, lr=1.0) for k, s in grid.items()}

    print(f"     length ladder, longer always wins, E[|y|] at init = {base_len:.4f}, "
          f"steps={ctx.mid_steps}")
    print(f"     {'cell':<34}{'m(T/2)':>10}{'m(T)':>10}{'incr':>9}{'E[|y|]':>10}{'KL':>9}")
    inc, mar, el = {}, {}, {}
    for k, r in out.items():
        half = r["margin"][len(r["margin"]) // 2]
        inc[k] = r["margin"][-1] - half
        mar[k], el[k] = r["margin"][-1], r["elen"][-1]
        print(f"     {k:<34}{half:>10.4f}{mar[k]:>10.4f}{inc[k]:>9.4f}"
              f"{el[k]:>10.4f}{r['kl'][-1]:>9.4f}")

    sq = ["IPO      (ref_sum, squared)", "IPO_mean (ref_mean, squared) <-"]
    ls = ["DPO      (ref_sum, logsigmoid)", "len-DPO  (ref_mean, logsigmoid)"]

    # AXIS L (link) acts on FINITENESS, and it does so for BOTH rewards.
    for k in sq:
        _check(f"squared link stops the margin: {k.split('(')[0].strip()}",
               abs(inc[k]) < 1e-3 * max(inc[x] for x in ls),
               f"increment {inc[k]:.3e} vs logsigmoid's {max(inc[x] for x in ls):.4f}")
    for k in ls:
        _check(f"logsigmoid link does NOT stop: {k.split('(')[0].strip()}",
               inc[k] > 0.3, f"increment {inc[k]:+.4f}")

    # AXIS f (reward) acts on LENGTH, and the squared link hands us a MATCHED MARGIN for
    # free -- both squared cells converge to the same margin, so the length comparison
    # below is controlled without a stop_margin.
    _check("both squared cells converge to the SAME margin (a free matched-margin "
           "comparison)", abs(mar[sq[0]] - mar[sq[1]]) < 1e-6,
           f"{mar[sq[0]]:.6f} vs {mar[sq[1]]:.6f}")
    _check("at that matched margin the MEAN reward lengthens FURTHER (stage `simpo`'s "
           "rule, now inside the empty cell)", el[sq[1]] > el[sq[0]],
           f"mean {el[sq[1]]:.4f} vs sum {el[sq[0]]:.4f}")

    print("     => the two axes act on DIFFERENT things and do not interfere: the link")
    print("        decides whether the margin is finite, the reward decides where the")
    print("        length goes. The empty cell inherits tau's cap from IPO and the")
    print("        mean reduction's length amplification from SimPO, exactly as the grid")
    print("        predicts. The grid is a mechanism, not a taxonomy of four papers.")


def stage_kto(ctx: Ctx) -> None:
    """KTO: the roll becomes real, prospect separates per example, and boundedness != P1"""
    penv, ex, lp_ref = ctx.penv, ctx.kto_data, ctx.lp_ref_prompt
    theta_ref = ctx.theta_ref_prompt
    spec = replace(ctx.specs()["KTO"], lam_d=1.0, lam_u=1.0)
    theta = penv.random_theta(seed=12, scale=1.2)
    with torch.no_grad():
        lp = penv.logprobs(theta)
        r = implicit_reward(spec.f_kind, lp, lp_ref, penv.lengths, spec.beta)

    # (1) THE ROLL IS NOW LOAD-BEARING. On TabEnv it was a permutation (stage `reduce`).
    p, y = ex[:, 0], ex[:, 1]
    matched, rolled = r[p, y], r[p, y.roll(1)]
    gap = abs(float(matched.mean()) - float(rolled.mean()))
    print(f"     PromptEnv: {penv.n_prompts} prompts x {penv.n_seqs} sequences, "
          f"{ex.shape[0]} unpaired examples")
    print(f"     mean r matched  {float(matched.mean()):+.6f}")
    print(f"     mean r rolled   {float(rolled.mean()):+.6f}    gap {gap:.6f}")
    _check("the mismatch roll now CHANGES z_ref -- it is no longer a permutation",
           gap > 1e-3, f"gap {gap:.6f} (on TabEnv this was ~1e-16)")
    _check("...because the rolled values are not even the same MULTISET any more",
           not bool(torch.equal(torch.sort(matched).values, torch.sort(rolled).values)))
    _check("z_ref is still clamped and detached",
           float(kto_z_ref(r, ex)) >= 0.0 and not kto_z_ref(r, ex).requires_grad,
           f"z_ref = {float(kto_z_ref(r, ex)):.6f}")

    # (2) WHY PAIRS WERE NEVER REQUIRED: prospect is a SUM OF PER-EXAMPLE TERMS.
    prs = ctx.kto_pairs
    pp, w, l = prs[:, 0], prs[:, 1], prs[:, 2]
    flat = torch.stack([torch.cat([pp, pp]),
                        torch.cat([w, l]),
                        torch.cat([torch.ones_like(w), -torch.ones_like(l)])], dim=1)
    z = kto_z_ref(r, flat)
    paired = link("prospect", r[pp, w], r[pp, l], spec, z).mean()
    unpaired = kto_loss(spec, lp, lp_ref, penv.lengths, flat)
    _check("KTO(unpaired) == 0.5 * mean(prospect link) on the SAME data, to 1e-15",
           abs(float(unpaired) - 0.5 * float(paired)) < 1e-15,
           f"{float(unpaired):.15f} vs {0.5 * float(paired):.15f}")
    print("     => the factor 2 is just 2n examples vs n pairs. The paired form was")
    print("        ALREADY separable into two independent per-example terms, which is the")
    print("        structural reason KTO can drop pairs -- not an approximation.")

    # (3) P1 IS *NOT* FIXED. This is where the received story ("KTO's loss is bounded,
    #     so it cannot diverge") breaks: a bounded loss is not a loss with a finite
    #     minimiser. inf_u sigma(-u) = 0, attained only as u -> +inf.
    dpo_spec = Spec("dpo", "ref_sum", "logsigmoid", beta=ctx.BETA)

    def dpo_loss(lp_, lpr_):
        rr = implicit_reward(dpo_spec.f_kind, lp_, lpr_, penv.lengths, dpo_spec.beta)
        return link("logsigmoid", rr[pp, w], rr[pp, l], dpo_spec).mean()

    runs = {
        "DPO ": train_prompt(penv, theta_ref.clone(), theta_ref, dpo_loss, flat, dpo_spec,
                             n_steps=ctx.long_steps, lr=0.5),
        "KTO ": train_prompt(penv, theta_ref.clone(), theta_ref,
                             lambda a, b: kto_loss(spec, a, b, penv.lengths, ex),
                             ex, spec, n_steps=ctx.long_steps, lr=0.5),
    }
    print(f"     steps={ctx.long_steps}, lr=0.5, SEPARABLE labels "
          f"(disjoint desirable/undesirable per prompt)")
    print(f"     {'run':<6}{'m(T/4)':>10}{'m(T/2)':>10}{'m(T)':>10}{'d1':>9}{'d2':>9}"
          f"{'KL':>9}{'z_ref':>9}")
    inc = {}
    for k, rr in runs.items():
        q = rr["margin"][int((len(rr["margin"]) - 1) * 0.25)]
        h = rr["margin"][int((len(rr["margin"]) - 1) * 0.5)]
        e = rr["margin"][-1]
        inc[k] = (h - q, e - h)
        print(f"     {k:<6}{q:>10.4f}{h:>10.4f}{e:>10.4f}{h - q:>9.4f}{e - h:>9.4f}"
              f"{rr['kl'][-1]:>9.4f}{rr['z_ref'][-1]:>9.4f}")

    _check("DPO on this data runs away: constant margin gain per DOUBLING (margin ~ log t)",
           inc["DPO "][0] > 0.3 and inc["DPO "][1] > 0.3
           and abs(inc["DPO "][1] - inc["DPO "][0]) / inc["DPO "][0] < 0.35,
           f"{inc['DPO '][0]:.4f} then {inc['DPO '][1]:.4f}")
    _check("!! KTO DOES NOT STOP EITHER -- its margin also gains per doubling",
           inc["KTO "][1] > 0.3,
           f"KTO {inc['KTO '][1]:.4f} vs DPO {inc['DPO '][1]:.4f}")
    print("     => A BOUNDED LOSS IS NOT A LOSS WITH A FINITE MINIMISER.")
    print("        sigma(-u) is strictly decreasing with infimum 0, attained only at u=+inf,")
    print("        so KTO's optimum on separable data is at infinity exactly like DPO's.")
    print("        Both gradients decay like e^-u, so both give margin ~ log t.")
    print("        cDPO and IPO stop because their losses have an INTERIOR minimum")
    print("        (a smoothed target, a squared target). KTO has neither. It is a")
    print("        D1 method, not a D2 method.")

    # (4) WHAT BOUNDEDNESS ACTUALLY BUYS: bounded INFLUENCE. Not finiteness -- robustness.
    print(f"     {'d (a badly wrong example)':<28}{'|grad| logsigmoid':>20}"
          f"{'|grad| prospect':>18}")
    gl, gp = [], []
    for d in (-2.0, -5.0, -10.0, -20.0):
        a = torch.tensor(d, requires_grad=True)
        g1, = torch.autograd.grad(-F.logsigmoid(a), a)
        b = torch.tensor(d, requires_grad=True)
        g2, = torch.autograd.grad(torch.sigmoid(-b), b)
        gl.append(abs(float(g1)))
        gp.append(abs(float(g2)))
        print(f"     {d:<28.1f}{abs(float(g1)):>20.10f}{abs(float(g2)):>18.3e}")
    _check("logsigmoid's per-example gradient SATURATES AT 1 -- an outlier never lets go",
           gl[-1] > 0.999 and gl[-1] >= gl[0])
    _check("prospect's per-example gradient VANISHES -- an outlier is eventually ignored",
           gp[-1] < 1e-8 and gp[-1] < gp[0])

    noisy = ctx.kto_noisy
    flip = noisy[:, 2] != ex[:, 2]
    noisy_pairs = _pairs_from(noisy, penv.n_prompts)

    def _dpo_on(prs):
        q, a, b = prs[:, 0], prs[:, 1], prs[:, 2]

        def f(lp_, lpr_):
            rr = implicit_reward(dpo_spec.f_kind, lp_, lpr_, penv.lengths, dpo_spec.beta)
            return link("logsigmoid", rr[q, a], rr[q, b], dpo_spec).mean()
        return f

    def _kto_on(dat):
        return lambda a, b: kto_loss(spec, a, b, penv.lengths, dat)

    arms = {("DPO ", "clean"): (_dpo_on(prs), flat),
            ("DPO ", "noisy"): (_dpo_on(noisy_pairs), flat),
            ("KTO ", "clean"): (_kto_on(ex), ex),
            ("KTO ", "noisy"): (_kto_on(noisy), noisy)}
    print(f"     {int(flip.sum())}/{ex.shape[0]} labels FLIPPED (data no longer separable); "
          f"steps={ctx.mid_steps}")
    print(f"     {'run':<12}{'clean margin':>14}{'clean acc':>11}{'pull toward noise':>19}")
    res = {}
    for (k, tag), (lf, dat) in arms.items():
        rr = train_prompt(penv, theta_ref.clone(), theta_ref, lf, dat, spec,
                          n_steps=ctx.mid_steps, lr=0.5)
        with torch.no_grad():
            rn = implicit_reward(spec.f_kind, penv.logprobs(rr["theta"]), lp_ref,
                                 penv.lengths, spec.beta)
        m = float((rn[pp, w] - rn[pp, l]).mean())
        a = float(((rn[pp, w] - rn[pp, l]) > 0).double().mean())
        ri, zz = rn[ex[:, 0], ex[:, 1]], kto_z_ref(rn, ex)
        obey = float(((noisy[:, 2].double().sign() * (ri - zz))[flip]).mean())
        res[(k, tag)] = (m, a, obey)
        print(f"     {k + tag:<12}{m:>14.4f}{a:>11.4f}{obey:>19.4f}")

    drop = {k: res[(k, "clean")][0] - res[(k, "noisy")][0] for k in ("DPO ", "KTO ")}
    _check("the flipped labels DISTORT DPO's clean margin ~2x more than KTO's",
           drop["DPO "] > 1.5 * drop["KTO "],
           f"DPO -{drop['DPO ']:.4f} vs KTO -{drop['KTO ']:.4f}")
    _check("DPO is dragged further TOWARD the wrong labels than KTO is",
           res[("DPO ", "noisy")][2] > res[("KTO ", "noisy")][2],
           f"DPO {res[('DPO ', 'noisy')][2]:+.4f} vs KTO {res[('KTO ', 'noisy')][2]:+.4f}")
    _check("...and both are dragged, relative to their own clean runs",
           res[("DPO ", "noisy")][2] > res[("DPO ", "clean")][2]
           and res[("KTO ", "noisy")][2] > res[("KTO ", "clean")][2])
    print("     => THIS is what the bounded loss is for. A mislabelled example contributes")
    print("        at most lam to the objective and its gradient dies, so the fit abandons")
    print("        it. Under -log sigma it contributes ~|d| and pulls at strength 1 forever,")
    print("        so the fit eventually goes and SATISFIES it, at everyone else's expense.")
    print("        Bounded loss buys ROBUSTNESS (D3's noise problem), not FINITENESS (P1).")

    # (5) THE LAMBDAS DO WHAT THEY CLAIM.
    print(f"     {'lam_u':>7}{'r_desirable':>14}{'r_undesirable':>16}{'margin':>10}")
    r_u = []
    for lu in (0.5, 1.0, 2.0):
        rr = train_prompt(penv, theta_ref.clone(), theta_ref,
                          (lambda s_: lambda a, b: kto_loss(s_, a, b, penv.lengths, ex))(
                              replace(spec, lam_u=lu)),
                          ex, spec, n_steps=ctx.mid_steps, lr=0.5)
        r_u.append(rr["r_u"][-1])
        print(f"     {lu:>7.2f}{rr['r_d'][-1]:>14.4f}{rr['r_u'][-1]:>16.4f}"
              f"{rr['margin'][-1]:>10.4f}")
    _check("raising lam_u pushes the UNDESIRABLE reward monotonically further down",
           r_u[0] > r_u[1] > r_u[2], "  ".join(f"{v:.4f}" for v in r_u))
    print("     => lam_d/lam_u is the knob KTO uses for BOTH loss aversion and class")
    print("        imbalance; the paper keeps lam_d*n_d/(lam_u*n_u) in [1, 4/3].")


STAGES = [
    ("reward", stage_reward),
    ("link", stage_link),
    ("optima", stage_optima),
    ("reduce", stage_reduce),
    ("separable", stage_separable),
    ("ipo_cost", stage_ipo_cost),
    ("orpo", stage_orpo),
    ("simpo", stage_simpo),
    ("empty_cell", stage_empty_cell),
    ("kto", stage_kto),
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
