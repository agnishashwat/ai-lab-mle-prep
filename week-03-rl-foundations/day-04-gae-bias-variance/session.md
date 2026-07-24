# Week 03 Day 04 — GAE, Bootstrapping & the Bias/Variance Tradeoff

## Objective

Derive Generalized Advantage Estimation as an exponentially-weighted average of
\(k\)-step TD advantage estimators, and understand the \((\gamma,\lambda)\) knobs as the
bias (bootstrapping) ↔ variance (Monte-Carlo) dial that PPO tunes.

## Prerequisites
- Day 3: advantage, learned critic \(V_\phi\), TD error \(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\).

## Required Reading
- Schulman et al. 2015, "High-Dimensional Continuous Control Using GAE" — §3 (the
  estimator + the \(\lambda\) derivation).

## 2-Hour Plan

### 0:00-0:10 - Recall Quiz (Day 3)
EGLP + baseline unbiasedness; \(V\) vs \(Q\) baseline; entropy bonus; actor-critic bias.

### 0:10-0:35 - Concept
The estimator spectrum: Monte-Carlo return (unbiased, high variance) ↔ 1-step TD
bootstrap (low variance, biased if \(V\ne V^\pi\)). \(n\)-step returns interpolate. GAE
= geometric mixture over all \(n\). Map to LLMs: token-level credit for a terminal reward.

### 0:35-1:00 - Derivation
1. \(k\)-step advantage \(\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}\).
2. GAE: \(\hat A_t^{GAE(\gamma,\lambda)}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}\);
   derive the geometric-weight collapse.
3. Limits: \(\lambda=0\Rightarrow\delta_t\) (1-step TD, low var/high bias);
   \(\lambda=1\Rightarrow\) Monte-Carlo advantage (\(\sum\gamma^l r-V\), high var/low bias).
4. Bias source = bootstrapping on an imperfect \(V_\phi\); variance source = summing many
   noisy rewards. \(\lambda\) trades them.

### 1:00-1:45 - Coding
`code/gae.py`: `compute_gae(rewards, values, dones, gamma, lam)` (the reverse
scan), plus a bias/variance sweep over \(\lambda\in\{0,0.5,0.9,0.95,1\}\) measuring
advantage-estimate variance and (against a known-\(V^\pi\) toy MDP from Day 1) its bias.
Self-check: \(\lambda\) endpoints match the closed forms; reverse-scan matches a naive
double loop; handles episode boundaries (`dones`) correctly.

### 1:45-1:55 - Interview Drill
"Derive GAE." "What do \(\gamma\) and \(\lambda\) each control?" "Why does bootstrapping
introduce bias, and when is that a good trade?"

### 1:55-2:00 - Deliverable & Next
Next: **Day 5 — KL-constrained RL** (why post-training pins to a reference policy).

## Checkpoint Questions
1. Write \(\delta_t\) and the GAE formula; derive it as a \((\gamma\lambda)\)-weighted sum.
2. What do \(\lambda=0\) and \(\lambda=1\) reduce to? Bias/variance of each?
3. Where does the bias come from, precisely?
4. Why a reverse scan; how do `dones` reset it?
5. In LLM RLHF with a single terminal reward, what does GAE buy you per token?

## Frontier-Lab Interview Angle
GAE is the advantage estimator inside PPO (incl. LLM PPO). Interviewers ask "how do you
assign credit to individual tokens when the reward comes only at EOS?" — GAE + a value
head is the textbook answer, and the \(\lambda\) knob is a favorite "how would you debug
unstable PPO" lever. Sets up why GRPO's critic-free group baseline is an attractive
simplification when a good \(V_\phi\) is hard to fit on language.
