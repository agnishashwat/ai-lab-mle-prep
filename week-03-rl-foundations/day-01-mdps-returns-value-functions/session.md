# Week 03 Day 01 — MDPs, Returns & Value Functions

## Objective

Build the exact formal scaffold that PPO/GRPO/DPO sit on top of: Markov Decision
Processes, trajectories, returns, the two value functions (\(V^\pi\), \(Q^\pi\)),
the advantage \(A^\pi\), and the Bellman equations (expectation + optimality) as
the fixed-point identities everything else is derived from. By end of day you can
whiteboard "what is the object policy gradient differentiates, and what does the
critic estimate" without hand-waving.

## Prerequisites

- Expectations, conditional expectations, law of total expectation.
- Comfort with a discounted geometric series (it sets the return's scale).
- Week-2 mechanics (for the recall quiz only).

## Required Reading

- OpenAI Spinning Up — "Part 1: Key Concepts in RL" (states/actions, policies,
  trajectories, reward & return, value functions). One read, ~15 min.
- Optional: Sutton & Barto Ch. 3 (finite MDPs) §3.1–3.6 for the Bellman equations.

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Week 2 cold — spaced repetition)

1. **Attention-NaN, fastest-first (the D5 soft spot).** You see NaN loss. Give the
   temporal-clue split first, then the ordered checklist of causes + the one-line
   fix for each. (step-0 vs step-N → masked row → fp16 overflow → eps → grads → data)
2. **Teacher forcing vs free-running.** Define both. "Trains fine, generates garbage"
   — what single experiment localizes the bug to the generation path, and why?
3. **KV-cache bytes.** Write the formula. 7B, fp16, MHA, 8k context, batch 16 — number?
   What does GQA change and by how much? Why is decode bandwidth-bound (intensity ≈ ?).
4. **top-k vs top-p.** Which is adaptive and why; how does top-k fail in both the
   peaked and flat regimes; what's the `>=` boundary quirk.
5. **Two crown-jewel invariant tests.** Name them; why must the model be *random*
   (not the successor-trained one) for cached==uncached to have teeth?

### 0:10-0:35 - Concept Explanation (First Principles)

Cover, in order:

- **The RL object vs the SL object.** SL: minimize a fixed per-example loss over an
  iid dataset. RL: maximize \(J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]\) where
  the *data distribution itself depends on \(\theta\)*. This single fact is why we get
  the score-function/log-derivative trick (Day 2) instead of ordinary backprop.
- **MDP tuple** \((\mathcal S,\mathcal A,P,r,\gamma,\rho_0)\): dynamics
  \(P(s'|s,a)\), reward \(r(s,a)\) (or \(r(s,a,s')\)), discount \(\gamma\in[0,1)\),
  start distribution \(\rho_0\). **Markov property**: the future is independent of the
  past given the present state — the whole reason value functions are well-defined.
- **Policy** \(\pi(a|s)\) (stochastic). **Trajectory** \(\tau=(s_0,a_0,s_1,a_1,\dots)\)
  with \(p_\theta(\tau)=\rho_0(s_0)\prod_t \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\).
- **Return** \(G_t=\sum_{k\ge0}\gamma^k r_{t+k}\). Why discount: (i) convergence of
  infinite-horizon sums, (ii) effective horizon \(\approx 1/(1-\gamma)\), (iii) models
  uncertainty/impatience. Finite-horizon vs infinite-horizon framing.
- **Value functions.** \(V^\pi(s)=\mathbb{E}_\pi[G_t|s_t=s]\),
  \(Q^\pi(s,a)=\mathbb{E}_\pi[G_t|s_t=s,a_t=a]\). Relationship
  \(V^\pi(s)=\mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]\).
- **Advantage** \(A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)\): "how much better than the policy's
  average is this action." Note \(\mathbb{E}_{a\sim\pi}[A^\pi(s,a)]=0\) — this is *why*
  \(V\) is the natural baseline (Day 3) and what PPO/GAE estimate.
- **The LLM mapping (make it concrete):** state = prompt + tokens so far, action = next
  token, policy = the LM, \(\pi_\theta(a_t|s_t)\) = the softmax over vocab, episode =
  one generated sequence, reward = terminal (reward model / verifier) usually only at
  EOS. Per-token vs sequence-level reward is the recurring design axis all week.

### 0:35-1:00 - Derivation / Paper Deconstruction

Derive from scratch (see `derivation.md` if it grows):

1. **Bellman expectation for \(V^\pi\) and \(Q^\pi\).** Peel one step off the return
   using the Markov property + law of total expectation:
   \[V^\pi(s)=\sum_a\pi(a|s)\Big(r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^\pi(s')\Big).\]
   \[Q^\pi(s,a)=r(s,a)+\gamma\sum_{s'}P(s'|s,a)\sum_{a'}\pi(a'|s')Q^\pi(s',a').\]
   Emphasize the recursion is a *consistency* identity, not an update rule yet.
2. **Bellman as a fixed point.** The Bellman operator \(T^\pi\) is a \(\gamma\)-contraction
   in sup-norm ⇒ unique fixed point \(V^\pi\) ⇒ iterative policy evaluation converges.
   (One-line contraction argument.)
3. **Bellman optimality.** \(V^*(s)=\max_a\big(r(s,a)+\gamma\sum_{s'}P V^*(s')\big)\);
   greedy-in-\(Q^*\) is optimal. Contrast the `max` (optimality, nonlinear) with the
   `E_{a~π}` (evaluation, linear).
4. **Advantage decomposition of the return / the TD error** \(\delta_t=r_t+\gamma
   V(s_{t+1})-V(s_t)\): \(\mathbb{E}[\delta_t]=A^\pi\) under \(V=V^\pi\). This is the
   seed for GAE (Day 4) and for why the critic reduces variance (Day 3).

### 1:00-1:45 - Coding / Experiment Task

Build `week-03-rl-foundations/code/mdp.py` (stubs + self-check harness, learner
implements). A tiny finite MDP where everything is checkable in closed form:

- `class MDP` holding `P[s,a,s']`, `R[s,a]`, `gamma`.
- `policy_evaluation(mdp, pi)` two ways and assert they match:
  (a) **iterative** Bellman backup to convergence; (b) **exact linear solve**
  \(V^\pi=(I-\gamma P^\pi)^{-1} r^\pi\).
- `q_from_v(mdp, V)`, `advantage(mdp, pi, V)`; assert \(\mathbb{E}_{a\sim\pi}[A]=0\)
  per state and \(V=\mathbb{E}_a[Q]\).
- `value_iteration(mdp)` → \(V^*\), extract greedy \(\pi^*\); assert it's a fixed point
  of Bellman optimality and that its \(V^{\pi^*}\) matches \(V^*\).
- Sanity env: a 2–3 state chain / tiny gridworld with a known optimal policy.
- Self-check harness prints PASS/FAIL per invariant (contraction convergence,
  linear==iterative, advantage-sums-to-zero, VI fixed point).

### 1:45-1:55 - Interview Drill

Explain out loud, 2 min each: (a) "Why can't we just backprop through the reward like
a supervised loss?" (b) "What does the critic in PPO actually estimate, and why does
subtracting it not bias the gradient?" — pre-loading Days 2–3.

### 1:55-2:00 - Deliverable & Next Session

Commit `code/mdp.py` + tests; fill `deliverable.md`. Next: **Day 2 — Policy Gradient
Theorem** (log-derivative trick, REINFORCE, why the expectation is unbiased).

## Checkpoint Questions

1. State the MDP tuple and the Markov property; why does the Markov property make
   \(V^\pi\) well-defined?
2. Write \(p_\theta(\tau)\) and the RL objective \(J(\theta)\). What's the ONE
   structural difference from a supervised objective, and what does it force later?
3. Derive the Bellman expectation equation for \(V^\pi\) in two lines.
4. Define \(A^\pi\); prove \(\mathbb{E}_{a\sim\pi}[A^\pi(s,a)]=0\). Why does that make
   \(V^\pi\) the variance-optimal-ish baseline?
5. Contrast Bellman *expectation* vs *optimality* (the `E_π` vs `max`), and map every
   MDP element onto LLM token generation.

## Frontier-Lab Interview Angle

RL rounds almost always open with "walk me through PPO/GRPO." The candidates who lose
points hand-wave the substrate — they can't say precisely what \(A^\pi\) is, why the
baseline is unbiased, or what the value head regresses onto. Nailing this MDP/value
layer cold is what lets you *derive* PPO's clipped objective and GRPO's group baseline
later instead of reciting them. Also a classic probe: "map RLHF onto an MDP" (state =
context, action = token, sparse terminal reward) — and the follow-up on why sparse
sequence reward + long horizon makes credit assignment (hence GAE/baselines) the crux.
