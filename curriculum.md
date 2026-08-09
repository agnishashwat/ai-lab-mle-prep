# AI Lab MLE Curriculum Skill File

## Purpose
Use this file as persistent context for generating week-by-week 2-hour study sessions for Shashwat's preparation for Machine Learning Engineer / Research Engineer roles at frontier AI labs such as OpenAI, Anthropic, and similar labs.

The goal is not full academic coverage. The goal is to maximize interview and on-the-job readiness in 2-3 months by focusing on the highest-leverage 20% of topics: post-training, RL for LLMs, evaluations, ML systems, debugging, and strong project narratives.

## Learner profile
- Name: Shashwat Shivam.
- Background: senior software / ML engineering experience; production ML systems; semantic retrieval; LLM personalization; vector search; large-scale systems; PhD-level math/optimization/control background.
- Existing exposure: practitioner-level experience with post-training techniques including SFT, DPO, GRPO, preference data, LLM evaluation, and model distillation.
- Main gap: convert fragmented practical familiarity into frontier-lab crispness: clean first-principles understanding, strong implementation fluency, controlled experiment design, evaluation taste, and ability to debug large ML codebases.
- Target positioning: post-training + RL/reasoning + evaluations + training systems MLE / research engineer.

## Target role signals
Prepare for roles that expect:
- Strong ML engineering and research taste.
- Ability to debug and improve large ML codebases.
- Comfort with RL, post-training, model evaluations, and product/research iteration.
- Strong software engineering fundamentals: clean code, testing, performance, communication.
- Ability to build scalable ML infrastructure, evaluation systems, and training/reward pipelines.
- Bias toward shipping controlled experiments and making trustworthy conclusions.

## Curriculum format
Default cadence: 5 sessions per week, 2 hours each = 10 hours/week.
Optional 6th block: project catch-up, paper digestion, or mock interview.

Every generated 2-hour session should have this structure:
1. 10 min — Recall quiz from prior material.
2. 25 min — First-principles concept explanation.
3. 25 min — Math derivation or paper deconstruction.
4. 15 min — Interview-style explanation drill. **Mandatory, and it runs BEFORE the coding
   block.** Answers are written into `interview-bank/`.
5. 40 min — Coding / implementation / experiment task.
6. 5 min — Notes, deliverable, and next-session setup.

Each session should include:
- A clear title.
- Learning objective.
- Prerequisites.
- Minimal required reading, ideally no more than one paper section or one compact resource.
- Concrete exercises.
- A deliverable that can go into a GitHub repo / blog / interview story.
- 3-5 checkpoint questions.
- One “frontier-lab interview angle”: how the topic might show up in interviews or project deep-dives.

Avoid:
- Broad survey-style coverage.
- Too many papers in one session.
- Basic ML explanations unless needed to support a derivation.
- Passive reading-only sessions.
- LeetCode-only prep disconnected from ML engineering.

## 10-week recommended plan

### Week 1 — Role calibration, LM foundations, and PyTorch fluency
Theme: Build a precise mental model of language modeling and refresh implementation muscle.
Blocks:
1. Role map + baseline diagnostic for AI-lab MLE interviews.
2. Token-level language modeling: logits, logprobs, cross-entropy, perplexity.
3. PyTorch autograd, tensor shapes, masking, mixed precision failure modes.
4. Build a minimal transformer training loop skeleton.
5. Evaluation basics: loss curves, heldout eval, overfitting, ablation discipline.
Deliverable: tiny reproducible LM training/eval harness with tests and clean logging.

### Week 2 — Transformer internals and efficient generation
Theme: Know transformer mechanics well enough to debug training/inference issues.
Blocks:
1. Attention from first principles: Q/K/V, masks, positional encodings.
2. Transformer block implementation and numerical stability.
3. Decoding: temperature, top-k/top-p, beam, repetition, length bias.
4. KV cache, batching, memory math, inference throughput.
5. Debug lab: diagnose bad generations, NaNs, shape bugs, masking bugs.
Deliverable: annotated transformer or decoder-only mini-implementation + generation notebook.

### Week 3 — RL foundations for post-training
Theme: Derive the machinery behind PPO/GRPO rather than memorizing formulas.
Blocks:
1. MDPs, trajectories, returns, value functions, Bellman equations.
2. Policy gradient theorem and score-function estimator.
3. Baselines, advantage functions, variance reduction, entropy regularization.
4. GAE, bootstrapping, bias/variance tradeoff.
5. KL-constrained RL and why LLM post-training uses KL to a reference policy.
Deliverable: small policy-gradient implementation on a toy environment, with advantage/KL ablations.

### Week 4 — RLHF and PPO for LLMs
Theme: Understand the canonical RLHF pipeline and why PPO is engineered the way it is.
Blocks:
1. InstructGPT/RLHF pipeline: SFT, reward model, PPO, KL, evals.
2. PPO objective: ratio, clipping, old policy, multiple epochs, minibatches.
3. LLM-specific PPO: token-level logprobs, sequence rewards, KL penalties, value heads.
4. Failure modes: reward hacking, KL collapse, length exploitation, policy drift.
5. Implement/trace a toy PPO-style RLHF loop.
Deliverable: PPO/RLHF explainer notebook with toy code and failure-mode demonstrations.

### Week 5 — Preference optimization: DPO and variants
Theme: Make DPO mathematically obvious and know when it is the wrong tool.
Blocks:
1. Bradley-Terry preference model and reward modeling.
2. Derive DPO from KL-regularized RLHF.
3. DPO implementation details: chosen/rejected logprobs, reference model, beta, length effects.
4. Variants and neighboring methods: IPO, KTO, ORPO, SimPO, RPO-style methods.
5. DPO failure modes: noisy preferences, distribution shift, overoptimization, data quality.
Deliverable: DPO from-scratch trainer on a small model/dataset, with beta and data-quality ablations.

### Week 6 — GRPO, RLVR, and reasoning post-training
Theme: Understand modern reasoning-focused RL: verifiable rewards, group baselines, pass@k, curriculum.
Blocks:
1. GRPO vs PPO: group-relative advantages, no critic, memory/variance tradeoff.
2. RL with verifiable rewards: math/code tasks, reward sparsity, correctness checkers.
3. Sampling strategy: group size, temperature, pass@k, rejection sampling, self-consistency.
4. Reasoning data curriculum: easy-to-hard, zero-reward cliffs, scaffolding, contamination risk.
5. Implement a mini RLVR/GRPO-style loop on arithmetic/code/math tasks.
Deliverable: mini RLVR repo: generate N completions, score with verifier, compute group advantages, update policy or simulate update math.

### Week 7 — Evaluations, reward modeling, and safety-oriented ML
Theme: Be unusually good at eval design; this is a frontier-lab differentiator.
Blocks:
1. What makes an eval valid: construct validity, leakage, confidence intervals, regression tests.
2. LLM-as-judge: calibration, bias, pairwise ranking, rubric design.
3. Reward model design: data collection, annotator disagreement, calibration, adversarial evals.
4. Safety/misuse evals: classifiers, red teaming, prompt injection, anomaly detection.
5. Build an eval harness for post-training experiments.
Deliverable: evaluation harness with pairwise judge, bootstrap CI, failure taxonomy, and model-card-style report.

### Week 8 — Training and inference systems for post-training
Theme: Know enough systems to reason about cost, scale, and bottlenecks.
Blocks:
1. GPU memory math: parameters, optimizer states, activations, KV cache, batch/sequence tradeoffs.
2. Fine-tuning methods: LoRA/QLoRA vs full fine-tune; FSDP/ZeRO; gradient checkpointing.
3. Distributed training basics: data/model/pipeline parallelism, collectives, failure modes.
4. Inference systems: continuous batching, paged attention, quantization, throughput/latency.
5. End-to-end post-training pipeline architecture.
Deliverable: systems design doc for a scalable SFT+DPO+eval pipeline with cost/memory estimates.

### Week 9 — Capstone: post-training mini-lab
Theme: Convert learning into a strong interview artifact.
Blocks:
1. Define experiment: model, dataset, eval, success metric, baseline.
2. SFT baseline and eval harness.
3. DPO or IPO run with ablations.
4. GRPO/RLVR simulation or lightweight run with verifiable rewards.
5. Write results: what improved, what failed, what you would scale next.
Deliverable: polished GitHub repo + short technical writeup.

### Week 10 — Interview conversion and deep-dive readiness
Theme: Turn knowledge into interview performance.
Blocks:
1. ML coding drills: PyTorch tensor tasks, custom loss, dataloader, debugging.
2. ML system design: training pipeline, eval system, reward pipeline, inference service.
3. Research deep-dive: explain DPO/GRPO/PPO tradeoffs crisply.
4. Past-project narrative: Amazon/LinkedIn retrieval + post-training story arcs.
5. Mock onsite: coding + ML design + paper discussion + project deep-dive.
Deliverable: interview packet: 1-page role narrative, 3 project stories, 20 technical Q&A, capstone README.

## Optional Weeks 11-12 — Specialization and polish
Choose one track depending on target role:

### Track A — Post-training / RL reasoning
- More RLVR papers and ablations.
- Implement harder verifier tasks.
- Study reward hacking and curriculum design.
- Create a compact blog post: “DPO vs PPO vs GRPO from first principles.”

### Track B — Evaluations / safety / safeguards
- Build model behavior classifier and adversarial eval set.
- Study prompt-injection and agentic risk evals.
- Create a safety-eval harness and report.

### Track C — Training/inference systems
- Profile fine-tuning/inference bottlenecks.
- Benchmark vLLM/transformers inference.
- Build a memory/cost calculator for SFT/DPO/RLHF.

## High-value papers and resources
Prioritize these; do not overload sessions.
- InstructGPT / Training language models to follow instructions with human feedback.
- PPO: Proximal Policy Optimization Algorithms.
- DPO: Direct Preference Optimization.
- DeepSeekMath / GRPO.
- Constitutional AI and RLAIF-related Anthropic work.
- Transformer attention and efficient decoding resources.
- OpenAI Spinning Up for RL foundations.
- Selected eval papers/blogs only when directly useful for an exercise.

## Skill rubric
By the end, Shashwat should be able to:
1. Derive policy gradient, PPO, DPO, and GRPO at a whiteboard.
2. Implement small versions of SFT, DPO, PPO-like RLHF, and GRPO/RLVR loops.
3. Explain when to choose SFT vs DPO vs PPO vs GRPO vs rejection sampling.
4. Design reliable LLM evaluations with confidence intervals and failure taxonomies.
5. Estimate memory/cost bottlenecks for post-training and inference.
6. Debug PyTorch/tensor/model-training issues quickly.
7. Tell a coherent frontier-lab story connecting prior production ML systems to post-training/evals/reasoning work.

## How future chats should use this file
When Shashwat asks for “Week N sessions,” generate the full plan for that week only. Use 5 two-hour sessions unless he asks otherwise. Include readings, exercises, code tasks, deliverables, and checkpoint questions. Keep sessions intense but realistic.

If Shashwat asks to start a session, teach it interactively: explain concepts, ask checkpoint questions, provide derivations, and assign implementation tasks. Prefer depth over breadth.
