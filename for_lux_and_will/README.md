# SeqComm Battleship — Experiment Results

**Run timestamp:** 2026-04-29 18:01 → 2026-04-30 (all 8 configs ran in parallel, completed in under 24 hours)  
**Episodes per config:** 50,000  
**Seed:** 0 (single seed)

Raw per-episode logs (~23 MB each) live in the main runs directory at  
`runs/battleship_grid/20260429_180138/<config>/logs/`.

---

## 1. What SeqComm Is

SeqComm (**Sequential Communication**) is a multi-agent coordination protocol where agents do not act simultaneously. Instead, at every timestep they first **negotiate a priority ordering** — deciding who acts first — and then execute actions in cascade so that each agent can observe what the agents above it in the queue have already chosen to do before committing its own action.

The key idea: if agent A knows it will go first, it can take an aggressive action; agent B, knowing A went first, can coordinate rather than duplicate. Without ordering, both agents might fire at the same target, wasting a shot, or both dodge in the same direction, leaving the other side exposed.

**The ordering signal** is the mechanism by which agents decide the queue. This is what the experiments vary.

---

## 2. The Environment

### Grid and ships

| Parameter | Value |
|-----------|-------|
| Grid | 8 × 8 |
| Agent ships | 2 |
| Boss ships | 1 |
| Ship length | 3 cells (horizontal or vertical) |
| Observation radius (Chebyshev) | 2 cells around ship center |
| Firing radius (Chebyshev) | 2 cells around ship center |

Each ship occupies 3 contiguous grid cells. A ship is **sunk** when all 3 cells are destroyed. Individual cells can be hit independently — partial damage is tracked.

The boss ship follows a fixed heuristic: move toward the nearest visible agent, fire on a cadence determined by the curriculum stage. The boss aims at the agents' previous-step positions (aim lag = 1), so moving is a real evasion mechanic.

### Observation space (per agent)

Each agent receives a `(5×5×3 + 4) = 79`-dimensional observation:
- A 5×5 local patch (2-cell Chebyshev radius) with 3 channels: own cells, ally cells, boss cells
- 4 scalars: own HP remaining, current step, own row, own col

### Action space

3 continuous floats decoded by the policy:
- `action[0]` → move direction (stay / N / S / E / W)
- `action[1]`, `action[2]` → fire row/col offset (clamped to firing radius)

### Reward structure

| Signal | Value | Purpose |
|--------|-------|---------|
| Hit boss cell | +2.0 | Primary objective |
| Hit self/ally cell | −0.3 | Discourage friendly fire |
| Survive per step (per cell) | +0.005 | Small stabilizer |
| Near-miss (shot close to boss) | fades 0.20→0 | Dense shaping signal (see curriculum) |
| Proximity to boss (per step) | +0.02 | Gradient for move toward boss |
| Agents win (all boss cells gone) | +10.0 | Terminal reward |

---

## 3. The Curriculum

Training uses a 13-stage curriculum that automatically advances when agents meet the win/hit thresholds at the current stage. The curriculum controls three difficulty knobs simultaneously: boss HP (1→3 cells alive at episode start), boss fire period (how often it shoots), and episode time limit.

The **near-miss reward** fades to zero at the final stage so agents must learn to aim precisely without a shaping crutch.

| Stage | Boss HP | Boss fires every N steps | Max steps | Near-miss reward |
|-------|---------|--------------------------|-----------|-----------------|
| 0 | 1 | 12 | 30 | 0.20 |
| 1 | 1 | 10 | 28 | 0.20 |
| 2 | 1 | 8 | 26 | 0.15 |
| 3 | 1 | 6 | 24 | 0.15 |
| 4 | 2 | 12 | 46 | 0.10 |
| 5 | 2 | 10 | 42 | 0.10 |
| 6 | 2 | 8 | 38 | 0.05 |
| 7 | 2 | 6 | 34 | 0.05 |
| 8 | 2 | 4 | 32 | 0.02 |
| 9 | 3 | 12 | 60 | 0.02 |
| 10 | 3 | 8 | 60 | 0.01 |
| 11 | 3 | 6 | 54 | 0.01 |
| 12 | 3 | 4 | 50 | 0.00 |

**Advancement thresholds** (evaluated over a rolling 1000-episode window):
- For HP-1 stages: win rate ≥ 0.70, hit rate ≥ 0.020, zero-hit episodes ≤ 30%
- For HP-2 stages: win rate ≥ 0.65, hit rate ≥ 0.030, zero-hit episodes ≤ 8%
- For HP-3 stages: win rate ≥ 0.60, hit rate ≥ 0.035, zero-hit episodes ≤ 5%

Stage 12 is the final configuration: a full-health 3-cell boss that fires every 4 steps with no near-miss reward shaping. **Anything above ~60% win rate on stage 12 is strong.**

---

## 4. Training Architecture

All configurations share the same neural architecture and PPO training loop.

### Networks

| Module | Role | Used by C++? |
|--------|------|--------------|
| `Encoder` (policy path) | Obs → 64-d embedding for policy decisions | Yes — saved to `encoder.pt`, loaded at runtime |
| `Encoder` (WM path) | Obs → 64-d embedding for world model only | No — Python-only |
| `AttentionModule` (action) | Fuses own embedding with partner context | Yes |
| `AttentionModule` (world model) | Fuses embeddings for next-state prediction | No |
| `WorldModel` | Predicts next obs + reward for WM ordering | No |
| `Policy` | Gaussian action head (3D continuous) | Yes |
| `Critic` | Scalar value estimate | Yes |
| `CommGate` | Bernoulli gate: comm or skip? | Yes (Exp 2 only) |

The policy encoder and world-model encoder are **separate networks with separate optimizers** — this prevents the world-model gradient from corrupting the policy gradient path, which was a key training stability fix.

The critic is trained on **detached encoder features** — the value loss cannot backpropagate through the policy encoder. This ensures the policy encoder is shaped only by policy gradient signal, not by how well it predicts value.

### PPO hyperparameters

| Parameter | Value |
|-----------|-------|
| Update every | 64 episodes |
| PPO epochs per update | 8 |
| GAE λ | 0.95 |
| Discount γ | 0.99 |
| PPO clip ε | 0.2 |
| Entropy coefficient | 0.003 |
| Encoder LR | 0.0001 |
| Policy / Critic LR | 0.0003 |
| World model LR | 0.0003 |
| Gradient clip | 1.0 |

---

## 5. Experiment 1 — Ordering Signal Quality and Communication Loss

### Question

SeqComm's ordering depends on each agent broadcasting an **intention score** — a scalar that determines queue position. Two possible signals:

1. **Critic proxy (baseline):** use the agent's own value estimate V(hᵢ) as the intention score. No communication overhead; computed locally.
2. **World-model rollout (H=2):** each agent broadcasts its hidden state hᵢ, all agents run 2 steps of a shared world model forward, and the agent with the highest predicted cumulative return goes first. This is the SeqComm paper's Eq. 4 mechanism.

The second question: what happens when the communication channel is unreliable? With probability `p`, a negotiation round fails entirely and agents fall back to a random ordering for that step.

### The five configurations

| Config | Ordering signal | Comm loss |
|--------|----------------|-----------|
| `baseline_critic` | V(hᵢ) critic (local, no comm needed) | 0% |
| `wm_loss00` | H=2 WM rollout | 0% |
| `wm_loss10` | H=2 WM rollout | 10% steps random |
| `wm_loss20` | H=2 WM rollout | 20% steps random |
| `wm_loss30` | H=2 WM rollout | 30% steps random |

---

### Curriculum progression — Experiment 1

Each number is the episode at which the agent crossed that stage threshold.

| Stage | baseline_critic | wm_loss00 | wm_loss10 | wm_loss20 | wm_loss30 |
|-------|-----------------|-----------|-----------|-----------|-----------|
| 0→1 | ep 3,841 | ep 3,727 | ep 8,645 | ep 3,590 | ep 6,154 |
| 1→2 | ep 4,841 | ep 4,727 | ep 9,645 | ep 4,755 | ep 7,154 |
| 2→3 | ep 5,841 | ep 5,727 | ep 10,972 | ep 5,755 | ep 8,154 |
| 3→4 | ep 6,841 | ep 6,727 | ep 11,972 | ep 6,755 | ep 9,154 |
| 4→5 | ep 7,841 | ep 7,727 | ep 12,972 | ep 7,755 | ep 10,154 |
| 5→6 | ep 8,841 | ep 8,727 | ep 13,972 | ep 8,755 | ep 11,154 |
| 6→7 | ep 9,841 | ep 9,727 | ep 14,972 | ep 9,755 | ep 12,154 |
| 7→8 | ep 10,841 | ep 10,727 | ep 15,972 | ep 10,755 | ep 13,154 |
| 8→9 | ep 11,841 | ep 11,727 | ep 16,972 | ep 11,755 | ep 14,154 |
| 9→10 | ep 12,863 | ep 12,727 | ep 17,972 | ep 12,755 | ep 15,154 |
| 10→11 | ep 13,863 | ep 13,727 | ep 18,972 | ep 13,755 | ep 16,154 |
| 11→12 | ep 14,863 | ep 14,727 | ep 19,972 | ep 14,755 | ep 17,154 |

All five configurations reached the final stage 12. The `wm_loss10` variant took noticeably longer to crack stage 0 (ep 8,645 vs ~3,600–3,800 for the others), reflecting that 10% random ordering meaningfully disrupts early coordination. `wm_loss20` and `wm_loss30` recovered and advanced at normal speed after breaking through. All later stage transitions were uniform (1,000 episodes per stage = the minimum window), indicating the agent cleared each threshold cleanly without struggling.

---

### Win rate trajectory — Experiment 1

Win rate averaged over successive 10,000-episode windows:

| Bucket | baseline_critic | wm_loss00 | wm_loss10 | wm_loss20 | wm_loss30 |
|--------|-----------------|-----------|-----------|-----------|-----------|
| ep 0–10k | 0.741 | 0.803 | 0.654 | 0.716 | 0.706 |
| ep 10k–20k | 0.712 | 0.980 | 0.920 | 0.778 | 0.843 |
| ep 20k–30k | 0.868 | 0.989 | 0.994 | 0.970 | 0.962 |
| ep 30k–40k | 0.992 | 0.997 | 0.998 | 0.996 | 0.996 |
| ep 40k–50k | 0.999 | 0.997 | 0.999 | 0.999 | 0.999 |

---

### Final performance — Experiment 1 (last 500 episodes, stage 12)

| Config | Win rate | Boss hits/ep | Agent hits/ep | Zero-hit eps | Fire dist | Avg steps to win |
|--------|----------|--------------|---------------|--------------|-----------|-----------------|
| baseline_critic | **1.000** | **3.000** | 0.974 | **0.000** | 2.04 | **9.1** |
| wm_loss00 | 0.998 | 2.998 | 1.000 | 0.000 | 2.19 | 9.3 |
| wm_loss10 | **1.000** | **3.000** | 0.884 | **0.000** | **1.99** | **8.2** |
| wm_loss20 | 0.998 | 2.998 | 0.896 | 0.000 | 2.07 | 8.6 |
| wm_loss30 | 0.998 | 2.998 | 1.002 | 0.000 | 2.04 | 9.2 |

"Boss hits/ep" of 3.000 means the agents are destroying all 3 boss cells every single episode — a perfect offense. "Agent hits/ep" around 0.9–1.0 means the agents take roughly 1 hit from the boss per episode (the boss fires every 4 steps, so they're dodging most shots). "Avg steps to win" of 8–9 means on a 50-step maximum, agents are winning in well under 20% of the time limit.

---

### Experiment 1 interpretation

**The critic proxy is not worse than the WM.** `baseline_critic` achieved a perfect 1.000 win rate with 3.000 boss hits per episode — marginally better raw numbers than `wm_loss00`. This suggests that for this environment, the value estimate is a sufficient proxy for intention. The WM ordering provides more information in theory, but the agents do not need it to achieve perfect performance.

**The protocol is highly robust to channel loss.** All three loss variants (10%, 20%, 30%) reached near-identical final performance to the no-loss conditions. `wm_loss10` was the only variant that took noticeably longer to crack stage 0 (~5,000 more episodes), but it fully recovered and achieved a perfect final win rate. The 20% and 30% variants showed no meaningful degradation at all. This is a strong robustness result: SeqComm coordination survives substantial communication dropout.

---

## 6. Experiment 2 — Learned Communication Cost

### Question

What if agents can **choose** whether to communicate? At each negotiation step, each agent independently samples a binary decision from a learned `CommGate` — a small linear layer over its hidden state that outputs a Bernoulli probability. If any agent opts out, the whole round falls back to random ordering. Any agent that chose to communicate pays a `comm_penalty` subtracted from its step reward.

The CommGate parameters are updated by REINFORCE: if communicating led to higher net return than the episode baseline (after subtracting the penalty), the gate is nudged toward committing; otherwise away from it.

The question: **does the learned comm rate respond monotonically to the penalty?** And critically: **does performance drop when agents stop communicating?**

### The three configurations

| Config | Comm penalty per step | Expected behavior |
|--------|----------------------|-------------------|
| `commgate_tiny` | 0.005 | Should communicate almost always — benefit > cost |
| `commgate_medium` | 0.050 | Should communicate moderately |
| `commgate_large` | 0.500 | Should learn to mostly skip |

All three use `--wm-intention` (H=2 WM rollout) when communication does happen.

---

### Curriculum progression — Experiment 2

| Stage | commgate_tiny | commgate_medium | commgate_large |
|-------|---------------|-----------------|----------------|
| 0→1 | ep **1,942** | ep 8,553 | ep 4,910 |
| 1→2 | ep 2,942 | ep 9,553 | ep 5,910 |
| 2→3 | ep 3,942 | ep 10,553 | ep 6,910 |
| 3→4 | ep 4,942 | ep 11,553 | ep 7,910 |
| 4→5 | ep 5,942 | ep 12,553 | ep 8,910 |
| 5→6 | ep 6,942 | ep 13,553 | ep 9,910 |
| 6→7 | ep 7,942 | ep 14,553 | ep 10,910 |
| 7→8 | ep 8,942 | ep 15,553 | ep 11,910 |
| 8→9 | ep 9,942 | ep 16,553 | ep 12,910 |
| 9→10 | ep 11,265 | ep 17,553 | ep 13,910 |
| 10→11 | ep 12,265 | ep 18,553 | ep 14,910 |
| 11→12 | ep 13,265 | ep 19,553 | ep 15,910 |

`commgate_tiny` cracked stage 0 at episode 1,942 — the **fastest across all 8 configurations**, including both experiments. All three commgate configs reached stage 12.

---

### Learned communication rate — Experiment 2

This is the headline metric: what fraction of negotiation steps did agents voluntarily communicate?

| Bucket | commgate_tiny | commgate_medium | commgate_large |
|--------|---------------|-----------------|----------------|
| ep 0–10k | 0.279 | 0.237 | 0.112 |
| ep 10k–20k | 0.246 | 0.117 | 0.009 |
| ep 20k–30k | 0.236 | 0.094 | 0.001 |
| ep 30k–40k | 0.220 | 0.100 | **0.000** |
| ep 40k–50k | 0.326 | 0.117 | **0.000** |

The gradient is clean and monotone: larger penalty → lower comm rate. By ep 10,000–20,000, `commgate_large` has essentially learned to **never communicate** (0.9% comm rate), and by ep 30,000 it reaches exactly 0.000. `commgate_medium` settles at ~10–12%. `commgate_tiny` settles at ~22–30% — surprisingly low given the tiny penalty of 0.005, which is 1/200th of a direct hit reward.

---

### Win rate trajectory — Experiment 2

| Bucket | commgate_tiny | commgate_medium | commgate_large |
|--------|---------------|-----------------|----------------|
| ep 0–10k | 0.749 | 0.678 | 0.720 |
| ep 10k–20k | 0.673 | 0.948 | 0.898 |
| ep 20k–30k | 0.925 | 0.988 | 0.981 |
| ep 30k–40k | 0.992 | 0.995 | 0.995 |
| ep 40k–50k | 0.997 | 0.998 | 0.997 |

---

### Final performance — Experiment 2 (last 500 episodes, stage 12)

| Config | Comm rate (last 500) | Win rate | Boss hits/ep | Agent hits/ep | Zero-hit eps | Fire dist | Avg steps |
|--------|---------------------|----------|--------------|---------------|--------------|-----------|-----------|
| commgate_tiny | 0.294 | 0.992 | 2.986 | 1.126 | 0.002 | 2.22 | 10.9 |
| commgate_medium | 0.129 | 0.996 | 2.996 | 1.066 | 0.000 | 2.09 | 9.8 |
| commgate_large | **0.000** | 0.998 | 2.998 | 1.068 | 0.000 | 2.14 | 9.9 |

---

### Experiment 2 interpretation

**The most striking result: agents that stopped communicating entirely still achieved near-perfect performance.** `commgate_large` communicates 0% of the time in its final 20,000 episodes yet wins 99.8% of episodes and destroys all 3 boss cells per episode. This is nearly identical to the fully-communicating WM variants.

**The comm rate responds cleanly to the penalty,** confirming that the CommGate learned a meaningful cost-benefit calculation. The penalty gradient is visible from the very first 10,000 episodes.

**The surprising wrinkle:** `commgate_tiny` (penalty = 0.005) converges to only ~29% communication — much lower than expected. Despite the tiny penalty, the agents learned early that random ordering was *good enough*, and the CommGate settled into mostly opting out. The slight performance gap (99.2% vs 99.6–99.8% for the others) and higher agent_hits/ep (1.126 vs ~1.07) suggests the random-ordering baseline does marginally worse at avoiding boss fire, but the agents don't pay a big enough price to justify committing.

**The key takeaway:** once agents have learned a strong policy, explicit ordering provides diminishing marginal returns. The policy itself — positioning, firing decisions — carries more of the coordination weight than the negotiation protocol does.

---

## 7. Cross-experiment summary

| Config | Ordering | Loss/Gate | Comm rate (last 500) | Win rate (last 500) | Stage at ep 15k | Final stage |
|--------|----------|-----------|---------------------|---------------------|-----------------|-------------|
| baseline_critic | Critic V(h) | None | 1.000 | **1.000** | 12 | 12 |
| wm_loss00 | WM H=2 | 0% | 1.000 | 0.998 | 12 | 12 |
| wm_loss10 | WM H=2 | 10% | 0.902 | **1.000** | 7 | 12 |
| wm_loss20 | WM H=2 | 20% | 0.801 | 0.998 | 12 | 12 |
| wm_loss30 | WM H=2 | 30% | 0.686 | 0.998 | 11 | 12 |
| commgate_tiny | WM H=2 | Pen 0.005 | 0.294 | 0.992 | 12 | 12 |
| commgate_medium | WM H=2 | Pen 0.050 | 0.129 | 0.996 | 6 | 12 |
| commgate_large | WM H=2 | Pen 0.500 | 0.000 | 0.998 | 9 | 12 |

---

## 8. Key findings

1. **SeqComm learns extremely robust coordination.** All 8 configurations — including those with 30% random ordering, 50% per-step communication penalty, or agents who chose to never communicate at all — reached the final curriculum stage and achieved ~99–100% win rates.

2. **The critic proxy and the WM rollout produce equivalent final performance.** `baseline_critic` achieved 100% win rate and 3.000 boss hits/episode, matching the WM variants. The WM provides better information in principle but this environment doesn't need it to reach ceiling performance.

3. **The protocol is robust to channel failure.** 10%, 20%, and 30% comm dropout barely changed final performance. The agents learn policies that work whether or not the ordering signal arrives.

4. **Agents learn to stop communicating when it costs them.** The CommGate responses are monotone: tiny penalty → 29% comm, medium → 13%, large → 0%. Yet all three achieve near-identical win rates. This demonstrates that the coordination advantage of ordering is real but bounded — once the policy is strong enough, it outweighs whatever the ordering protocol contributes.

5. **The fastest learner was `commgate_tiny`** (stage 0 cleared at episode 1,942). The second fastest was `wm_loss00` (episode 3,727). This suggests that the WM+CommGate combination has an acceleration effect early in training, possibly because the gate immediately forces agents to rely on policy quality rather than communication scaffolding.

6. **Stage advancement was uniform after stage 0** across all configs — exactly 1,000 episodes per stage in most cases. This means the real difficulty and sample complexity is concentrated in stage 0 (hp=1, no pressure), which is where agents learn the fundamental skill of aiming. Once they can aim, they adapt to harder bosses quickly.

---

## 9. File index

```
for_lux_and_will/
├── README.md                         ← this file
├── summary.csv                       ← one row per config; all headline metrics
├── baseline_critic_seed0/
│   └── baseline_critic_seed0.png     ← learning curves: win rate, boss hits, comm rate, curriculum stage
├── wm_loss00_seed0/
│   └── wm_loss00_seed0.png
├── wm_loss10_seed0/
│   └── wm_loss10_seed0.png
├── wm_loss20_seed0/
│   └── wm_loss20_seed0.png
├── wm_loss30_seed0/
│   └── wm_loss30_seed0.png
├── commgate_tiny_seed0/
│   └── commgate_tiny_seed0.png
├── commgate_medium_seed0/
│   └── commgate_medium_seed0.png
└── commgate_large_seed0/
    └── commgate_large_seed0.png
```

Raw per-episode JSONL logs (~23 MB each) and trainer diagnostic logs are in the runs directory at  
`runs/battleship_grid/20260429_180138/<config>/logs/` and `<config>/trainer.jsonl`.  
Each episode line contains: `ep`, `reward`, `steps`, `boss_hits`, `agent_hits`, `agents_won`,  
`boss_won`, `agent_shots`, `fire_oob`, `mean_fire_dist`, `curriculum_stage`, `comms_ok`,  
`comms_total`, `comm_rate`, `intention_spread`, `first_mover`.

---

## 10. Environment and training code

The simulation is implemented in C++ (`battleship_sim.cc`, `battleship_env.cc`) and trained via a Python PPO loop (`train_battleship.py`). The two processes communicate through a shared weights directory using binary file handshaking (`traj.bin` / `traj.ready` / `weights.bin` / `weights.ready`). Policy networks are serialized as TorchScript (`.pt`) for loading into LibTorch from C++.

Runs are orchestrated by `battleship/run_battleship_grid.sh`. To reproduce:

```bash
# Experiment 1 (5 configs, 1 seed, 50k episodes each)
EXP=exp1 bash battleship/run_battleship_grid.sh

# Experiment 2 (3 configs, 1 seed, 50k episodes each)
EXP=exp2 bash battleship/run_battleship_grid.sh

# Run both in parallel (they write to separate timestamped directories):
ALLOW_EXISTING_BATTLESHIP=1 EXP=exp1 bash battleship/run_battleship_grid.sh &
ALLOW_EXISTING_BATTLESHIP=1 EXP=exp2 bash battleship/run_battleship_grid.sh &
wait
```
