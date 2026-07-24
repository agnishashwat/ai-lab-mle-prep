# Week 3 Day 1 — Deliverable: MDPs, Returns & Value Functions

## What I built

`week-03-rl-foundations/code/mdp.py` — a finite tabular MDP with the full value-function
toolkit and a self-check harness. **All 7 invariants PASS.**

- `MDP(P, R, gamma)` — validated `(S,A,S)` dynamics + `(S,A)` rewards.
- `policy_matrices` — `P_pi[s,s']=Σ_a π(a|s)P(s'|s,a)`, `r_pi[s]=Σ_a π(a|s)r(s,a)` (einsum).
- `bellman_expectation_backup` — one application of `T^π`.
- `policy_evaluation_iterative` — iterate `T^π` to sup-norm convergence (the γ-contraction, in code).
- `policy_evaluation_linear` — exact `V^π = (I − γP_π)^{-1} r_π` via `np.linalg.solve` (not `inv`).
- `q_from_v`, `advantage` (`Q − V[:,None]`, explicit broadcast).
- `value_iteration` — `max_a` optimality backup → `V^*` + one-hot greedy `π^*`.

### Result (2-state chain, γ=0.9)

- `V^* = [10, 10]` = `1/(1−γ)` — optimal policy parks in the paying state forever.
- `π^* = [[0,1],[1,0]]` = "switch to state 1, then stay."
- `V^π (random) = [4.75, 5.41]` < `V^*` (dominance holds).

## Key results / identities locked in

- **The RL vs SL structural difference:** `J(θ)=E_{τ∼p_θ}[R(τ)]` — the sampling distribution
  depends on θ. Forces the log-derivative trick (can't plain-backprop even if R were smooth).
- **Bellman expectation** (derived from the return via Markov + law of total expectation):
  `V^π(s)=Σ_a π(a|s)[r(s,a)+γ Σ_{s'} P(s'|s,a) V^π(s')]`.
- **Contraction:** `T^π` is a γ-contraction in sup-norm ⇒ unique fixed point `V^π` ⇒
  iterative eval converges geometrically. Optimality `T^*` (max_a) same, nonlinear.
- **Advantage** `A^π=Q^π−V^π`, and `E_{a∼π}[A^π(s,a)]=0` per state (verified in code) — the
  reason `V^π` is the natural baseline.
- **LLM mapping:** state = prompt+tokens, action = next token, dynamics deterministic (append),
  reward sparse & terminal (RM/verifier at EOS) ⇒ credit assignment is the hard part.

## Interview-drill grades

- Q1 (why not SL): 0.9 — both reasons (non-diff reward; θ-dependent distribution). Sharpen:
  the distribution-shift reason is the *fundamental* one even for smooth R.
- Q2 (what does the critic estimate / why unbiased): 0.6 — nailed `V^π=E_{a∼π}[Q^π]`, but
  stopped before the unbiasedness argument. **Gap → Day 3:** EGLP lemma
  `E_{a∼π}[∇log π]=0` ⇒ any state-only baseline is unbiased, variance-reducing.

## Next

Day 2 — Policy Gradient Theorem (log-derivative trick, REINFORCE, reward-to-go, why P & ρ₀
drop out). Carry in the EGLP identity as the bridge to Day-3 baselines.
