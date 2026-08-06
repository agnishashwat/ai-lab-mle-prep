"""
Week 5 Day 3 -- DPO implementation details on the Week 2 TinyGPT.

Day 2 established the objective. This file turns the implementation contracts around
that objective into staged exercises:

  1. `sequence_logprob_sum` / `sequence_logprob_mean`
  2. `precompute_reference_logprobs`
  3. `dpo_loss`
  4. `sft_warmup`
  5. `train`

The harness deliberately corrects four stale claims from the original session plan:

  * With policy initialized from the reference, the initial log-ratio is exactly zero.
    Under data/reference mismatch, the ratio REQUIRED BY THE LABELS can still be large.
  * Both chosen and rejected log-probs can fall even when data came from the reference.
    The stable mismatch diagnostic is extra KL at matched margin/loss, not that sign.
  * The DPO gradient vanishes as beta -> 0. It does not vanish uniformly as beta -> inf.
  * At label_smoothing=0.5, the loss is minimized at zero margin; it is not constant.

Run:

  everything       python dpo_trainer.py
  one stage        python dpo_trainer.py logprobs
  several stages   python dpo_trainer.py loss length
  faster training  python dpo_trainer.py --fast
  list stages      python dpo_trainer.py --list

Use `python -W error::RuntimeWarning ...` for the final run.
"""

from __future__ import annotations

import copy
import inspect
import math
import sys
import traceback
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
WEEK2_CODE = ROOT / "week-02-transformers-generation" / "code"
if str(WEEK2_CODE) not in sys.path:
    sys.path.insert(0, str(WEEK2_CODE))

from transformer_block import GPTConfig, TinyGPT  # noqa: E402


PAD = 0
BOS = 1
EOS = 2
GOOD = 3
BAD = 4
FILLER = 5
PROMPT_A = 6
PROMPT_B = 7
VOCAB_SIZE = 8


# ---------------------------------------------------------------------------
# Data contracts (provided)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceBatch:
    """Padded prompt+completion sequences and a mask over completion TARGETS.

    `input_ids` has shape (B, T). A causal LM's logits at position t predict
    `input_ids[:, t+1]`, so `completion_mask` has shape (B, T-1) and is aligned with
    those shifted targets. It includes EOS and excludes prompt and padding tokens.
    """

    input_ids: torch.Tensor
    completion_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, T)")
        if self.input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token ids")
        expected = (self.input_ids.shape[0], self.input_ids.shape[1] - 1)
        if tuple(self.completion_mask.shape) != expected:
            raise ValueError(
                f"completion_mask must align with shifted targets: expected {expected}, "
                f"got {tuple(self.completion_mask.shape)}"
            )
        if self.completion_mask.dtype != torch.bool:
            raise TypeError("completion_mask must be boolean")
        if torch.any(self.completion_mask.sum(dim=-1) == 0):
            raise ValueError("every sequence needs at least one completion target")

    @property
    def lengths(self) -> torch.Tensor:
        return self.completion_mask.sum(dim=-1)

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]


@dataclass(frozen=True)
class PreferenceBatch:
    chosen: SequenceBatch
    rejected: SequenceBatch

    def __post_init__(self) -> None:
        if self.chosen.batch_size != self.rejected.batch_size:
            raise ValueError("chosen and rejected batches must have the same size")

    @property
    def batch_size(self) -> int:
        return self.chosen.batch_size


@dataclass(frozen=True)
class PairLogps:
    """Sequence log-prob SUMS and their completion lengths for a preference batch."""

    chosen_sum: torch.Tensor
    rejected_sum: torch.Tensor
    chosen_lengths: torch.Tensor
    rejected_lengths: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.chosen_sum,
            self.rejected_sum,
            self.chosen_lengths,
            self.rejected_lengths,
        )
        if any(t.ndim != 1 for t in tensors):
            raise ValueError("all PairLogps fields must be rank-1")
        sizes = {t.shape[0] for t in tensors}
        if len(sizes) != 1:
            raise ValueError("all PairLogps fields must have the same batch size")
        if torch.any(self.chosen_lengths <= 0) or torch.any(self.rejected_lengths <= 0):
            raise ValueError("completion lengths must be positive")

    def reduced(self, length_normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not length_normalize:
            return self.chosen_sum, self.rejected_sum
        chosen = self.chosen_sum / self.chosen_lengths.to(self.chosen_sum.dtype)
        rejected = self.rejected_sum / self.rejected_lengths.to(self.rejected_sum.dtype)
        return chosen, rejected


@dataclass(frozen=True)
class DpoOutput:
    loss: torch.Tensor
    preference_logits: torch.Tensor
    logratio_margin: torch.Tensor
    chosen_logratio: torch.Tensor
    rejected_logratio: torch.Tensor


@dataclass(frozen=True)
class EvaluationCache:
    sequences: SequenceBatch
    ref_logps: torch.Tensor


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 30
    lr: float = 3e-3
    beta: float = 0.2
    label_smoothing: float = 0.0
    length_normalize: bool = False
    log_every: int = 5


def _make_sequence_batch(
    prompts: list[list[int]], completions: list[list[int]]
) -> SequenceBatch:
    if len(prompts) != len(completions) or not prompts:
        raise ValueError("prompts and completions need the same non-zero length")
    if any(not completion or completion[-1] != EOS for completion in completions):
        raise ValueError("every completion must be non-empty and end in EOS")

    full = [prompt + completion for prompt, completion in zip(prompts, completions)]
    max_len = max(map(len, full))
    input_ids = torch.full((len(full), max_len), PAD, dtype=torch.long)
    completion_mask = torch.zeros((len(full), max_len - 1), dtype=torch.bool)

    for i, (prompt, completion, sequence) in enumerate(zip(prompts, completions, full)):
        input_ids[i, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        # Mask indices align with targets input_ids[:, 1:]. The first completion token
        # is target index len(prompt)-1; EOS is included.
        start = len(prompt) - 1
        stop = start + len(completion)
        completion_mask[i, start:stop] = True

    return SequenceBatch(input_ids=input_ids, completion_mask=completion_mask)


def make_synthetic_preferences(
    *,
    length_confound: float,
    label_noise: float,
    repeats: int,
    seed: int,
) -> PreferenceBatch:
    """Create a tiny preference dataset with independent quality, length, and noise knobs.

    Utility is:

        content_quality + length_confound * number_of_non_EOS_completion_tokens

    `label_noise` flips the resulting pair label. The candidates are tokenizer-shaped
    token sequences, not scalar rewards, so the trainer must handle padding, causal
    shifting, EOS, and variable completion lengths.
    """

    if not 0.0 <= label_noise <= 0.5:
        raise ValueError("label_noise must be in [0, 0.5]")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    generator = torch.Generator().manual_seed(seed)
    candidates = [
        [GOOD, EOS],
        [GOOD, FILLER, FILLER, EOS],
        [BAD, EOS],
        [BAD, FILLER, FILLER, EOS],
    ]
    candidate_pairs = [(0, 2), (1, 3), (0, 3), (1, 2), (1, 0), (3, 2)]

    def utility(completion: list[int]) -> float:
        content = float(completion.count(GOOD) - completion.count(BAD))
        length = len(completion) - 1  # exclude EOS from the synthetic utility
        return content + length_confound * length

    chosen_prompts: list[list[int]] = []
    rejected_prompts: list[list[int]] = []
    chosen: list[list[int]] = []
    rejected: list[list[int]] = []

    for _ in range(repeats):
        for prompt_token in (PROMPT_A, PROMPT_B):
            for left, right in candidate_pairs:
                first, second = candidates[left], candidates[right]
                first_wins = utility(first) >= utility(second)
                if torch.rand((), generator=generator).item() < label_noise:
                    first_wins = not first_wins
                winner, loser = (first, second) if first_wins else (second, first)
                prompt = [BOS, prompt_token]
                chosen_prompts.append(prompt)
                rejected_prompts.append(prompt)
                chosen.append(winner)
                rejected.append(loser)

    return PreferenceBatch(
        chosen=_make_sequence_batch(chosen_prompts, chosen),
        rejected=_make_sequence_batch(rejected_prompts, rejected),
    )


def make_evaluation_sequences() -> SequenceBatch:
    candidates = [
        [GOOD, EOS],
        [GOOD, FILLER, EOS],
        [GOOD, FILLER, FILLER, EOS],
        [BAD, EOS],
        [BAD, FILLER, EOS],
        [BAD, FILLER, FILLER, EOS],
    ]
    prompts = [[BOS, PROMPT_A] for _ in candidates]
    return _make_sequence_batch(prompts, candidates)


def make_model(seed: int) -> TinyGPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(
        V=VOCAB_SIZE,
        d=24,
        n_heads=4,
        n_layers=1,
        T=8,
        mult=2,
    )
    return TinyGPT(cfg)


# ---------------------------------------------------------------------------
# Exercises (learner-owned TODO bodies)
# ---------------------------------------------------------------------------


def sequence_logprob_sum(model: nn.Module, batch: SequenceBatch) -> torch.Tensor:
    """Return one completion log-prob SUM per sequence, shape (B,), accumulated in fp32.

    Contract:

      * logits[:, t] predicts input_ids[:, t+1] -- shift exactly once;
      * gather the observed target token from `log_softmax`, never feed probabilities
        into cross entropy;
      * include completion tokens and EOS, exclude prompt and padding via the mask;
      * perform log-softmax and the reduction in fp32 even when model logits are bf16;
      * preserve gradients with respect to the policy model.

    Do not place `-inf` in values that will later be summed. Gather finite token
    log-probs first, then zero excluded positions with the boolean mask.
    """
    logits = model(batch.input_ids)[:, :-1]
    targets = batch.input_ids[:, 1:]

    all_logps = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    token_logps = all_logps.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return token_logps.masked_fill(~batch.completion_mask, 0.0).sum(dim=-1)


def sequence_logprob_mean(model: nn.Module, batch: SequenceBatch) -> torch.Tensor:
    """Return one completion log-prob MEAN per sequence, shape (B,).

    Reuse `sequence_logprob_sum`; the only mathematical difference is division by the
    number of unmasked completion targets. A second gather/log-softmax implementation is
    duplicated surface area and defeats the day's sum-vs-mean check.
    """

    sequence_sums = sequence_logprob_sum(model, batch)
    return sequence_sums / batch.lengths.to(sequence_sums.dtype)


def precompute_reference_logprobs(
    model_ref: nn.Module, dataset: PreferenceBatch
) -> PairLogps:
    """Compute and detach frozen-reference sequence log-prob SUMS once.

    Requirements:

      * evaluate under `torch.no_grad()` and `model_ref.eval()`;
      * restore the caller's original train/eval mode afterwards;
      * return detached fp32 tensors with `requires_grad=False`;
      * cache SUMS plus lengths. Mean reduction can then be selected later without
        re-running the model.

    The training function intentionally has no `model_ref` parameter. Once this cache
    exists, touching the reference again is a systems bug.
    """
    was_training = model_ref.training
    model_ref.eval()
    try:
        with torch.no_grad():
            logp_ref_chosen = sequence_logprob_sum(
                model_ref, dataset.chosen
            ).detach().clone()
            logp_ref_rejected = sequence_logprob_sum(
                model_ref, dataset.rejected
            ).detach().clone()
    finally:
        model_ref.train(was_training)

    return PairLogps(
        logp_ref_chosen,
        logp_ref_rejected,
        dataset.chosen.lengths,
        dataset.rejected.lengths,
    )


def dpo_loss(
    policy_logps: PairLogps,
    ref_logps: PairLogps,
    *,
    beta: float,
    label_smoothing: float = 0.0,
    length_normalize: bool = False,
) -> DpoOutput:
    """Compute vanilla DPO, cDPO, or length-normalized DPO.

    Let:

        Delta = (log pi_theta(y_w) - log pi_ref(y_w))
              - (log pi_theta(y_l) - log pi_ref(y_l))
        M = beta * Delta

    The conservative-DPO NLL is:

        -(1-eps) * logsigmoid(M) - eps * logsigmoid(-M)

    Return the mean scalar loss and the per-pair tensors needed for diagnostics.
    Validate `beta > 0` and `0 <= label_smoothing <= 0.5`; invalid hyperparameters
    should fail loudly rather than silently change the algorithm.
    """
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing <= 0.5:
        raise ValueError("label_smoothing must be finite and in [0, 0.5]")

    chosen_policy, rejected_policy = policy_logps.reduced(length_normalize)
    chosen_ref, rejected_ref = ref_logps.reduced(length_normalize)

    chosen_logratio = chosen_policy - chosen_ref
    rejected_logratio = rejected_policy - rejected_ref
    logratio_margin = chosen_logratio - rejected_logratio
    preference_logits = beta * logratio_margin
    per_pair_nll = (
        -(1.0 - label_smoothing) * F.logsigmoid(preference_logits)
        - label_smoothing * F.logsigmoid(-preference_logits)
    )
    return DpoOutput(
        loss=per_pair_nll.mean(),
        preference_logits=preference_logits,
        logratio_margin=logratio_margin,
        chosen_logratio=chosen_logratio,
        rejected_logratio=rejected_logratio,
    )


def sft_warmup(
    model: nn.Module,
    chosen: SequenceBatch,
    *,
    steps: int,
    lr: float,
) -> list[float]:
    """SFT the model on chosen completions and return the token-NLL history.

    Minimize total chosen negative log-likelihood divided by the number of completion
    targets. After warmup, the resulting checkpoint should be cloned into BOTH roles:
    the frozen reference and the initial trainable DPO policy.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")

    nll: list[float] = []
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            -sequence_logprob_sum(model, chosen).sum()
            / chosen.lengths.sum()
        )
        nll.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return nll


@torch.no_grad()
def update_record(
    model: nn.Module,
    dataset: PreferenceBatch,
    ref_cache: PairLogps,
    config: TrainConfig,
    step: int,
    evaluation: EvaluationCache | None,
) -> dict[str, float]:
    """Take a fresh detached snapshot of all DPO training diagnostics."""

    was_training = model.training
    model.eval()
    try:
        policy_logps = policy_pair_logps(model, dataset)
        dpo_output = dpo_loss(
            policy_logps,
            ref_cache,
            beta=config.beta,
            label_smoothing=config.label_smoothing,
            length_normalize=config.length_normalize,
        )
        chosen_logp, rejected_logp = policy_logps.reduced(
            config.length_normalize
        )
        implicit_reward_abs = config.beta * torch.cat(
            [
                dpo_output.chosen_logratio,
                dpo_output.rejected_logratio,
            ]
        ).abs().mean()
        preference_accuracy = (
            dpo_output.logratio_margin > 0
        ).float().mean()

        if evaluation is None:
            subset_kl = float("nan")
            mean_generated_length = float("nan")
        else:
            subset_kl, mean_generated_length = subset_metrics(
                model, evaluation
            )

        return {
            "step": float(step),
            "loss": dpo_output.loss.item(),
            "chosen_logp": chosen_logp.mean().item(),
            "rejected_logp": rejected_logp.mean().item(),
            "logratio_margin": dpo_output.logratio_margin.mean().item(),
            "implicit_reward_abs": implicit_reward_abs.item(),
            "preference_accuracy": preference_accuracy.item(),
            "mean_generated_length": mean_generated_length,
            "subset_kl": subset_kl,
        }
    finally:
        model.train(was_training)


def train(
    model: nn.Module,
    dataset: PreferenceBatch,
    ref_cache: PairLogps,
    *,
    config: TrainConfig,
    evaluation: EvaluationCache | None = None,
) -> list[dict[str, float]]:
    """Full-batch TinyGPT DPO training with the diagnostics the pairwise loss omits.

    Record step 0, every `log_every` updates, and the final step. Each record contains:

      step, loss, chosen_logp, rejected_logp, logratio_margin,
      implicit_reward_abs, preference_accuracy, mean_generated_length, subset_kl

    `mean_generated_length` and `subset_kl` are exact under the normalized enumerable
    evaluation subset. Use NaN for those two fields when no evaluation cache is passed.

    There is deliberately NO reference-model argument. The only reference dependency
    is the immutable cache.
    """
    if config.steps < 0:
        raise ValueError("steps must be non-negative")
    if not math.isfinite(config.lr) or config.lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    result = [
        update_record(
            model,
            dataset,
            ref_cache,
            config,
            step=0,
            evaluation=evaluation,
        )
    ]

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        policy_logps = policy_pair_logps(model, dataset)
        dpo_output = dpo_loss(
            policy_logps,
            ref_cache,
            beta=config.beta,
            label_smoothing=config.label_smoothing,
            length_normalize=config.length_normalize,
        )
        dpo_output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config.log_every == 0 or step == config.steps:
            result.append(
                update_record(
                    model,
                    dataset,
                    ref_cache,
                    config,
                    step=step,
                    evaluation=evaluation,
                )
            )

    return result


# ---------------------------------------------------------------------------
# Evaluation helpers (provided; become usable after the TODOs above)
# ---------------------------------------------------------------------------


def policy_pair_logps(model: nn.Module, dataset: PreferenceBatch) -> PairLogps:
    return PairLogps(
        chosen_sum=sequence_logprob_sum(model, dataset.chosen),
        rejected_sum=sequence_logprob_sum(model, dataset.rejected),
        chosen_lengths=dataset.chosen.lengths,
        rejected_lengths=dataset.rejected.lengths,
    )


def precompute_evaluation_cache(
    model_ref: nn.Module, sequences: SequenceBatch
) -> EvaluationCache:
    was_training = model_ref.training
    model_ref.eval()
    try:
        with torch.no_grad():
            ref_logps = sequence_logprob_sum(model_ref, sequences).detach().clone()
    finally:
        model_ref.train(was_training)
    return EvaluationCache(sequences=sequences, ref_logps=ref_logps)


def subset_metrics(
    model: nn.Module, evaluation: EvaluationCache
) -> tuple[float, float]:
    """Exact conditional KL and expected length on the enumerable candidate subset."""

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            policy_logps = sequence_logprob_sum(model, evaluation.sequences)
            log_policy = policy_logps - torch.logsumexp(policy_logps, dim=0)
            log_ref = evaluation.ref_logps - torch.logsumexp(evaluation.ref_logps, dim=0)
            policy = log_policy.exp()
            kl = torch.sum(policy * (log_policy - log_ref))
            mean_length = torch.sum(
                policy * evaluation.sequences.lengths.to(policy.dtype)
            )
    finally:
        model.train(was_training)
    return float(kl), float(mean_length)


def _scalar_reference_logprobs(
    model: nn.Module, batch: SequenceBatch
) -> torch.Tensor:
    """Slow independent oracle for the vectorized sequence-logprob exercise."""

    logits = model(batch.input_ids).float()
    values: list[torch.Tensor] = []
    for row in range(batch.batch_size):
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for position in range(batch.input_ids.shape[1] - 1):
            if bool(batch.completion_mask[row, position]):
                token = batch.input_ids[row, position + 1]
                total = total + F.log_softmax(
                    logits[row, position], dim=-1
                )[token]
        values.append(total)
    return torch.stack(values)


# ---------------------------------------------------------------------------
# Staged harness
# ---------------------------------------------------------------------------


_TALLY = {"pass": 0, "fail": 0}


def _check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    _TALLY["pass" if ok else "fail"] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def _raises_value_error(function) -> bool:
    try:
        function()
    except ValueError:
        return True
    return False


def _grad_at_margin(beta: float, eps: float, delta: float) -> tuple[float, float]:
    chosen = torch.tensor([delta], dtype=torch.float64, requires_grad=True)
    zeros = torch.zeros(1, dtype=torch.float64)
    ones = torch.ones(1, dtype=torch.long)
    policy = PairLogps(chosen, zeros, ones, ones)
    reference = PairLogps(zeros, zeros, ones, ones)
    output = dpo_loss(
        policy,
        reference,
        beta=beta,
        label_smoothing=eps,
    )
    (gradient,) = torch.autograd.grad(output.loss, chosen, create_graph=True)
    (curvature,) = torch.autograd.grad(gradient, chosen)
    return float(gradient.detach()), float(curvature.detach())


class CountingModel(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.inner(input_ids)


class FixedBFloat16LM(nn.Module):
    """Deterministic bf16 logits for the fp32-reduction contract."""

    def __init__(self, sequence_length: int, vocab_size: int):
        super().__init__()
        generator = torch.Generator().manual_seed(91)
        logits = 0.35 * torch.randn(
            sequence_length, vocab_size, generator=generator
        )
        self.register_buffer("fixed_logits", logits.to(torch.bfloat16))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, time = input_ids.shape
        return self.fixed_logits[:time].unsqueeze(0).expand(batch, -1, -1)


class Ctx:
    def __init__(self, fast: bool):
        self.fast = fast

    @cached_property
    def dataset(self) -> PreferenceBatch:
        return make_synthetic_preferences(
            length_confound=0.45,
            label_noise=0.1,
            repeats=1 if self.fast else 3,
            seed=13,
        )


def stage_logprobs(ctx: Ctx) -> None:
    """causal shift, completion mask, sum-vs-mean, gradients, and padding"""

    model = make_model(seed=0)
    dataset = ctx.dataset
    model.eval()

    actual_sum = sequence_logprob_sum(model, dataset.chosen)
    expected_sum = _scalar_reference_logprobs(model, dataset.chosen)
    actual_mean = sequence_logprob_mean(model, dataset.chosen)
    expected_mean = actual_sum / dataset.chosen.lengths.to(actual_sum.dtype)

    _check("sum shape is one scalar per sequence",
           tuple(actual_sum.shape) == (dataset.batch_size,))
    _check("sum matches independent scalar loop",
           torch.allclose(actual_sum, expected_sum, atol=1e-6, rtol=1e-6),
           f"max error={float((actual_sum - expected_sum).detach().abs().max()):.3e}")
    _check("mean differs only by completion length",
           torch.allclose(actual_mean, expected_mean, atol=1e-7, rtol=1e-7))
    _check("sequence reductions are fp32", actual_sum.dtype == torch.float32)
    _check("policy logprobs preserve gradients", actual_sum.requires_grad)

    changed = dataset.chosen.input_ids.clone()
    for row in range(dataset.chosen.batch_size):
        last_target = int(
            torch.where(dataset.chosen.completion_mask[row])[0][-1].item()
        )
        changed[row, last_target + 2:] = (changed[row, last_target + 2:] + 3) % VOCAB_SIZE
    changed_batch = SequenceBatch(changed, dataset.chosen.completion_mask)
    changed_sum = sequence_logprob_sum(model, changed_batch)
    _check("masked future padding cannot change completion logprob",
           torch.equal(actual_sum, changed_sum))

    print(
        "  lengths:",
        sorted(set(map(int, dataset.chosen.lengths.tolist()))),
        "| mean(sum/length error)=",
        f"{float((actual_mean - expected_mean).detach().abs().mean()):.3e}",
    )


def stage_cache(ctx: Ctx) -> None:
    """reference cache is exact, detached, mode-safe, and absent from train()"""

    wrapped = CountingModel(make_model(seed=1))
    wrapped.train()
    cache = precompute_reference_logprobs(wrapped, ctx.dataset)
    calls_after_cache = wrapped.forward_calls

    with torch.no_grad():
        fresh = PairLogps(
            sequence_logprob_sum(wrapped, ctx.dataset.chosen),
            sequence_logprob_sum(wrapped, ctx.dataset.rejected),
            ctx.dataset.chosen.lengths,
            ctx.dataset.rejected.lengths,
        )

    _check("cache runs exactly one chosen and one rejected forward",
           calls_after_cache == 2, f"calls={calls_after_cache}")
    _check("chosen cache is bitwise exact",
           torch.equal(cache.chosen_sum, fresh.chosen_sum))
    _check("rejected cache is bitwise exact",
           torch.equal(cache.rejected_sum, fresh.rejected_sum))
    _check("cache is detached",
           not cache.chosen_sum.requires_grad and not cache.rejected_sum.requires_grad)
    _check("cache is fp32",
           cache.chosen_sum.dtype == torch.float32
           and cache.rejected_sum.dtype == torch.float32)
    _check("precompute restores caller's train mode", wrapped.training)
    _check("trainer has no reference-model parameter",
           "model_ref" not in inspect.signature(train).parameters)

    bytes_per_pair = (
        cache.chosen_sum.element_size() + cache.rejected_sum.element_size()
    )
    print(
        f"  cached values: {cache.chosen_sum.numel() * 2} fp32 scalars"
        f" | {bytes_per_pair} bytes/pair"
        f" | reference calls after cache={calls_after_cache}"
    )


def stage_loss(ctx: Ctx) -> None:
    """stable DPO/cDPO loss, finite optimum, and corrected boundary regressions"""

    del ctx
    ones = torch.ones(2, dtype=torch.long)
    policy_chosen = torch.tensor([0.4, -0.2], requires_grad=True)
    policy_rejected = torch.tensor([-0.1, -0.3], requires_grad=True)
    ref_chosen = torch.tensor([0.1, -0.4])
    ref_rejected = torch.tensor([-0.2, -0.1])
    policy = PairLogps(policy_chosen, policy_rejected, ones, ones)
    reference = PairLogps(ref_chosen, ref_rejected, ones, ones)

    beta = 0.3
    eps = 0.1
    output = dpo_loss(
        policy,
        reference,
        beta=beta,
        label_smoothing=eps,
    )
    manual_margin = (
        (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    )
    manual_logits = beta * manual_margin
    manual_loss = (
        -(1.0 - eps) * F.logsigmoid(manual_logits)
        - eps * F.logsigmoid(-manual_logits)
    ).mean()

    _check("margin formula",
           torch.allclose(output.logratio_margin, manual_margin))
    _check("cDPO matches independent expression",
           torch.allclose(output.loss, manual_loss, atol=1e-7))
    _check("NLL is scalar and finite",
           output.loss.ndim == 0 and bool(torch.isfinite(output.loss)))

    zeros = torch.zeros(1, requires_grad=True)
    equal = PairLogps(zeros, zeros, torch.ones(1, dtype=torch.long),
                      torch.ones(1, dtype=torch.long))
    at_reference = dpo_loss(equal, PairLogps(
        torch.zeros(1), torch.zeros(1),
        torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.long)),
        beta=beta,
        label_smoothing=eps,
    )
    reference_loss = float(at_reference.loss.detach())
    _check("known-value probe: policy==reference gives +log(2)",
           abs(reference_loss - math.log(2.0)) < 1e-7,
           f"loss={reference_loss:.9f}")

    shifted_policy = PairLogps(
        policy_chosen - 9.0,
        policy_rejected - 9.0,
        ones,
        ones,
    )
    shifted = dpo_loss(
        shifted_policy,
        reference,
        beta=beta,
        label_smoothing=eps,
    )
    _check("pair loss does not control common marginal shift",
           torch.allclose(
               output.preference_logits,
               shifted.preference_logits,
               atol=1e-6,
               rtol=1e-6,
           )
           and torch.allclose(output.loss, shifted.loss, atol=1e-6, rtol=1e-6))

    _check("non-positive beta fails loudly",
           _raises_value_error(
               lambda: dpo_loss(policy, reference, beta=0.0)
           ))
    _check("invalid label smoothing fails loudly",
           _raises_value_error(
               lambda: dpo_loss(
                   policy,
                   reference,
                   beta=beta,
                   label_smoothing=0.6,
               )
           )
           and _raises_value_error(
               lambda: dpo_loss(
                   policy,
                   reference,
                   beta=beta,
                   label_smoothing=-0.1,
               )
           ))

    sweep = []
    for smoothing in (0.05, 0.10, 0.20, 0.35):
        optimum = math.log((1.0 - smoothing) / smoothing) / beta
        gradient, curvature = _grad_at_margin(beta, smoothing, optimum)
        _check(f"cDPO optimum eps={smoothing:.2f}",
               abs(gradient) < 1e-10 and curvature > 0.0,
               f"Delta*={optimum:.6f}, grad={gradient:.3e}")
        sweep.append((smoothing, optimum))

    grad_zero, _ = _grad_at_margin(beta, 0.0, 10.0 / beta)
    _check("vanilla DPO still pushes at every finite positive margin",
           grad_zero < 0.0, f"grad={grad_zero:.3e}")

    grad_small_beta, _ = _grad_at_margin(1e-7, 0.0, 0.0)
    _check("correct no-op boundary is beta -> 0",
           abs(grad_small_beta) < 1e-6, f"grad={grad_small_beta:.3e}")

    grad_tie, curvature_tie = _grad_at_margin(beta, 0.5, 0.0)
    tie_away = dpo_loss(
        PairLogps(torch.tensor([2.0]), torch.zeros(1),
                  torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.long)),
        PairLogps(torch.zeros(1), torch.zeros(1),
                  torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.long)),
        beta=beta,
        label_smoothing=0.5,
    )
    _check("eps=0.5 means a zero-margin target, not constant loss",
           abs(grad_tie) < 1e-12
           and curvature_tie > 0.0
           and float(tie_away.loss) > math.log(2.0))

    hard = dpo_loss(
        PairLogps(torch.tensor([-100.0]), torch.zeros(1),
                  torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.long)),
        PairLogps(torch.zeros(1), torch.zeros(1),
                  torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.long)),
        beta=1.0,
    )
    _check("hard negative stays finite under logsigmoid",
           bool(torch.isfinite(hard.loss)) and abs(float(hard.loss) - 100.0) < 1e-4,
           f"loss={float(hard.loss):.6f}")

    z_pos = torch.tensor(40.0, dtype=torch.float32)
    z_neg = torch.tensor(-100.0, dtype=torch.float32)
    naive_easy = torch.log(torch.sigmoid(z_pos))
    stable_easy = F.logsigmoid(z_pos)
    naive_hard_nll = -torch.log(torch.sigmoid(z_neg))
    stable_hard_nll = -F.logsigmoid(z_neg)
    _check("fp32 positive saturation: naive log-sigmoid becomes exactly zero",
           float(naive_easy) == 0.0 and float(stable_easy) < 0.0)
    _check("fp32 negative underflow: stable NLL remains finite",
           bool(torch.isinf(naive_hard_nll))
           and abs(float(stable_hard_nll) - 100.0) < 1e-6)

    print("  eps     finite Delta* at beta=0.3")
    for smoothing, optimum in sweep:
        print(f"  {smoothing:>4.2f}    {optimum:>10.6f}")


def stage_length(ctx: Ctx) -> None:
    """sum asks drift>0; mean asks drift>running average"""

    del ctx
    lengths = torch.tensor([2, 4, 6], dtype=torch.long)
    per_token_drift = 0.2
    zeros = torch.zeros(3)
    policy = PairLogps(
        chosen_sum=per_token_drift * lengths.float(),
        rejected_sum=zeros,
        chosen_lengths=lengths,
        rejected_lengths=lengths,
    )
    reference = PairLogps(
        chosen_sum=zeros,
        rejected_sum=zeros,
        chosen_lengths=lengths,
        rejected_lengths=lengths,
    )

    sum_output = dpo_loss(policy, reference, beta=0.5)
    mean_output = dpo_loss(
        policy,
        reference,
        beta=0.5,
        length_normalize=True,
    )
    sum_slope = float(
        (sum_output.logratio_margin[-1] - sum_output.logratio_margin[0])
        / (lengths[-1] - lengths[0])
    )
    mean_slope = float(
        (mean_output.logratio_margin[-1] - mean_output.logratio_margin[0])
        / (lengths[-1] - lengths[0])
    )

    _check("sum reward grows linearly with repeated positive drift",
           torch.allclose(
               sum_output.logratio_margin,
               per_token_drift * lengths.float(),
           ))
    _check("mean reward removes pure repeated-drift length slope",
           torch.allclose(
               mean_output.logratio_margin,
               torch.full((3,), per_token_drift),
           ))
    _check("measured sum-vs-mean slopes",
           abs(sum_slope - per_token_drift) < 1e-7
           and abs(mean_slope) < 1e-7,
           f"sum={sum_slope:.6f}, mean={mean_slope:.6f}")

    running_average = 0.2
    marginal_below = 0.1
    _check("marginal token can pass sum and fail mean",
           marginal_below > 0.0 and marginal_below < running_average)
    print(
        f"  added-token rule: sum accepts {marginal_below:.2f}>0"
        f" | mean rejects {marginal_below:.2f}<{running_average:.2f}"
    )


def stage_precision(ctx: Ctx) -> None:
    """fp32 log-softmax/reduction vs bf16 cancellation over 256 targets"""

    del ctx
    completion_length = 256
    tokens = [BOS] + [GOOD if i % 2 == 0 else FILLER
                      for i in range(completion_length - 1)] + [EOS]
    batch = _make_sequence_batch([[BOS]], [tokens[1:]])
    model = FixedBFloat16LM(
        sequence_length=batch.input_ids.shape[1],
        vocab_size=VOCAB_SIZE,
    )

    actual = sequence_logprob_sum(model, batch)
    logits_bf16 = model(batch.input_ids)[:, :-1]
    targets = batch.input_ids[:, 1:]
    fp32_tokens = F.log_softmax(logits_bf16.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    expected = fp32_tokens.masked_fill(~batch.completion_mask, 0.0).sum(dim=-1)

    bf16_tokens = F.log_softmax(logits_bf16, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    bf16_sum = bf16_tokens.masked_fill(~batch.completion_mask, 0.0).sum(dim=-1)
    accumulation_error = abs(float(expected - bf16_sum.float()))
    beta = 0.1
    typical_margin = 0.5

    _check("sequence function follows fp32 numerical path",
           torch.equal(actual, expected),
           f"actual={float(actual):.6f}, expected={float(expected):.6f}")
    _check("bf16 path has measurable 256-token error",
           accumulation_error > 0.0,
           f"|fp32-bf16|={accumulation_error:.6f}")
    print(
        f"  T={completion_length}: |fp32-bf16 sum|={accumulation_error:.6f}"
        f" | beta*typical_margin={beta * typical_margin:.6f}"
        f" | ratio={accumulation_error / (beta * typical_margin):.3f}x"
    )


def stage_sft(ctx: Ctx) -> None:
    """chosen SFT raises data likelihood and seeds BOTH policy and reference"""

    model = make_model(seed=3)
    model.eval()
    with torch.no_grad():
        before = float(sequence_logprob_mean(model, ctx.dataset.chosen).mean())

    history = sft_warmup(
        model,
        ctx.dataset.chosen,
        steps=10 if ctx.fast else 35,
        lr=8e-3,
    )
    model.eval()
    with torch.no_grad():
        after = float(sequence_logprob_mean(model, ctx.dataset.chosen).mean())

    _check("SFT returns one finite loss per step",
           len(history) == (10 if ctx.fast else 35)
           and all(math.isfinite(value) for value in history))
    _check("SFT token NLL decreases",
           history[-1] < history[0],
           f"{history[0]:.6f} -> {history[-1]:.6f}")
    _check("chosen completion mean logprob rises",
           after > before,
           f"{before:.6f} -> {after:.6f}")

    model_ref = copy.deepcopy(model).eval()
    policy = copy.deepcopy(model).eval()
    ref_cache = precompute_reference_logprobs(model_ref, ctx.dataset)
    with torch.no_grad():
        policy_logps = policy_pair_logps(policy, ctx.dataset)
        initial = dpo_loss(policy_logps, ref_cache, beta=0.2)
    _check("updated SFT checkpoint seeds zero DPO log-ratios",
           torch.equal(initial.chosen_logratio, torch.zeros_like(initial.chosen_logratio))
           and torch.equal(
               initial.rejected_logratio,
               torch.zeros_like(initial.rejected_logratio),
           ))
    _check("updated SFT checkpoint seeds +log(2) NLL",
           abs(float(initial.loss) - math.log(2.0)) < 1e-7)
    print(
        f"  chosen mean logp/token: {before:.5f} -> {after:.5f}"
        " | cloned to policy AND reference"
    )


def stage_train(ctx: Ctx) -> None:
    """real TinyGPT update with separate marginals, margin, length, and subset KL"""

    model_ref = make_model(seed=4).eval()
    policy = copy.deepcopy(model_ref)
    initial_state = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    ref_cache = precompute_reference_logprobs(model_ref, ctx.dataset)
    cached_before = (
        ref_cache.chosen_sum.clone(),
        ref_cache.rejected_sum.clone(),
    )
    evaluation = precompute_evaluation_cache(
        model_ref,
        make_evaluation_sequences(),
    )
    config = TrainConfig(
        steps=6 if ctx.fast else 24,
        lr=5e-3,
        beta=0.2,
        label_smoothing=0.1,
        length_normalize=False,
        log_every=4 if ctx.fast else 5,
    )
    history = train(
        policy,
        ctx.dataset,
        ref_cache,
        config=config,
        evaluation=evaluation,
    )

    required = {
        "step",
        "loss",
        "chosen_logp",
        "rejected_logp",
        "logratio_margin",
        "implicit_reward_abs",
        "preference_accuracy",
        "mean_generated_length",
        "subset_kl",
    }
    _check("history records step 0", history[0]["step"] == 0)
    _check("history records final step", history[-1]["step"] == config.steps)
    _check("history exposes every diagnostic",
           all(required <= record.keys() for record in history))
    _check("all recorded values are finite",
           all(
               math.isfinite(value)
               for record in history
               for key, value in record.items()
               if key != "step"
           ))
    _check("initial DPO NLL is +log(2)",
           abs(history[0]["loss"] - math.log(2.0)) < 1e-6,
           f"loss0={history[0]['loss']:.9f}")
    _check("training lowers preference loss",
           history[-1]["loss"] < history[0]["loss"],
           f"{history[0]['loss']:.6f} -> {history[-1]['loss']:.6f}")
    _check("policy parameters changed",
           any(
               not torch.equal(initial_state[name], value)
               for name, value in policy.state_dict().items()
           ))
    _check("reference cache remains immutable",
           torch.equal(ref_cache.chosen_sum, cached_before[0])
           and torch.equal(ref_cache.rejected_sum, cached_before[1]))
    _check("subset KL is non-negative within numerical tolerance",
           all(record["subset_kl"] >= -1e-6 for record in history))
    _check("trainer cannot touch model_ref",
           "model_ref" not in inspect.signature(train).parameters)

    first, last = history[0], history[-1]
    print(
        "  step  chosen_logp  rejected_logp  margin    |r_hat|  subset_KL  mean_len"
    )
    for record in history:
        print(
            f"  {int(record['step']):>4d}"
            f"  {record['chosen_logp']:>11.5f}"
            f"  {record['rejected_logp']:>13.5f}"
            f"  {record['logratio_margin']:>7.4f}"
            f"  {record['implicit_reward_abs']:>8.4f}"
            f"  {record['subset_kl']:>9.5f}"
            f"  {record['mean_generated_length']:>8.4f}"
        )
    print(
        "  separate-marginal deltas:"
        f" chosen={last['chosen_logp'] - first['chosen_logp']:+.5f},"
        f" rejected={last['rejected_logp'] - first['rejected_logp']:+.5f}"
    )


STAGES = [
    ("logprobs", stage_logprobs),
    ("cache", stage_cache),
    ("loss", stage_loss),
    ("length", stage_length),
    ("precision", stage_precision),
    ("sft", stage_sft),
    ("train", stage_train),
]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv:
        print("stages (run a subset by name):")
        for name, function in STAGES:
            print(f"  {name:<10} {function.__doc__}")
        return 0

    fast = "--fast" in argv
    names = [arg for arg in argv if not arg.startswith("--")]
    known = {name for name, _ in STAGES}
    unknown = [name for name in names if name not in known]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}")
        print("use --list to see valid names")
        return 2

    selected = [
        (name, function)
        for name, function in STAGES
        if not names or name in names
    ]
    ctx = Ctx(fast=fast)
    todo: list[str] = []
    failed: list[str] = []

    for name, function in selected:
        print(f"\n== stage: {name} ==")
        try:
            function(ctx)
        except NotImplementedError:
            print("  [TODO] dependency not implemented yet -- skipped")
            todo.append(name)
        except Exception:
            failed.append(name)
            traceback.print_exc()

    print(
        f"\nchecks: {_TALLY['pass']} passed, {_TALLY['fail']} failed"
        + (f" | stages TODO: {', '.join(todo)}" if todo else "")
        + (f" | stages failing: {', '.join(failed)}" if failed else "")
    )
    return 1 if failed or _TALLY["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
