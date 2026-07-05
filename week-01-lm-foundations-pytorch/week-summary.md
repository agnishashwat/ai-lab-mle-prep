# Week 01 Summary - Role Calibration, LM Foundations & PyTorch Fluency

> Interview prep material distilled from Days 1-5.

## The 5-Minute Version

Week 1 built the LM + PyTorch substrate everything else stands on. Starting from the loss
(`−(1/BT) Σ log p_θ(y|x_<t)`), I derived the softmax/cross-entropy gradient `∂CE/∂z = p − e_y`
by hand and verified it against autograd and finite differences. I made the numerics safe
(max-subtraction, `z − logsumexp(z)`, never `softmax().log()`), reasoned about mixed-precision
memory (bf16 Adam = 16 bytes/param = same 112 GB for 7B as fp32; savings are activations +
throughput, and fp16 needs loss scaling because of gradient underflow). Then I built a minimal,
reproducible training loop (seed → per-step LR schedule → grad-accum with backward *inside* the
accum loop → clip → step → zero_grad → eval under `eval()`+`no_grad()`), and finally an
evaluation + ablation layer: dual-curve logging, the overfit-one-batch sanity test, and
seed-averaged one-variable ablations. The throughline is *trustworthy measurement*: correct
gradients, correct numerics, correct eval hygiene, and results that beat seed noise.

## Key Concepts (crisp)

- CE gradient = "prediction − target," sums to zero (mass is moved, not created); carries a 1/N under a mean.
- Stable softmax + `log_softmax = z − logsumexp(z)`; two safeguards = no exp-overflow, no log-underflow.
- gather = read log-p at target (loss); scatter = write one-hot (gradient).
- Mixed precision: bf16 saves activations + 2× matmul, NOT optimizer/weight state; fp16's killer is grad underflow → GradScaler.
- Loop invariants + ORDER, with `.backward()` **inside** the grad-accum loop (memory, not math).
- Eval hygiene = two bugs: `no_grad` (graph/OOM + grad leak) and `eval()` (dropout + BN running stats).
- Ablation discipline: one variable, fixed compute budget, ≥3 seeds, report mean ± spread; effect must exceed noise.
- Overfitting needs non-generalizing signal (label noise); a noiseless consistent rule can't overfit.

## Key Equations

- LM loss: `−(1/BT) Σ log p_θ(y_{b,t} | x_{b,<t})`
- `∂CE/∂z = softmax(z) − onehot(y)`; mean-CE per-position grad `(p − e_y)/N`
- `log_softmax(z) = z − logsumexp(z)`; perplexity `= exp(CE)`
- bf16 Adam memory = 16 bytes/param (fp32 master+m+v = 12, bf16 W+grad = 4)
- `Var(g_B) = σ²/B` → LR scaling linear `η∝B` / sqrt `η∝√B`

## What I Implemented

- `fluency_probe.py` (softmax/CE/masked-mean/sgd-step/seq-logprobs), `token_lm.py` (+ analytic ce_grad),
  `autograd_shapes_amp.py` (manual backward through Linear→ReLU→Linear→CE, fp16 overflow → bf16 fix),
  `train.py` (full reproducible loop), `eval_ablation.py` (eval + ablation layer). All with self-check + pytest.

## Failure Modes I Now Understand

- Double-softmax into `F.cross_entropy` = silent underfit (no crash).
- `cross_entropy` K-dim shape bug is silent when `T==V` (square shapes hide shape bugs).
- Backward-outside-accum = correct math but blows up memory.
- `no_grad`/`eval()` omissions; grad leak into next step; BN stat corruption.
- Grad clipping tames Adam so much that LR-divergence hides until LR≈300 (clip off).
- A broken eval harness masquerades as catastrophic overfitting — verify eval first.

## Open Questions / To Revisit

- Masked-mean static form `(x*mask).sum(-1)/mask.sum(-1)` — still the recurring soft spot.
- LR↔batch critical batch size (where linear scaling breaks down).
- Bootstrap confidence intervals (previewed; formalize in Week 7 evals).
