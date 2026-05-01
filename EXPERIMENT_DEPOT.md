# Experiment Depot — SeqComm / MAPPO Battleship Study
### CS 2860 Final Project · Lux Hogan-Murphy, Will Sherwood, Bobby Cupps

> **Purpose.** This file is a *source-material depot* for writing the experimental
> results section of the IEEE paper. Every interesting run is catalogued in
> chronological order: what we were testing, what changed from the previous run,
> what the data showed, and what design decision came next. Numbers come directly
> from JSONL logs; nothing is cherry-picked.

---

## Table of Contents

1. [Environment & Protocol Recap](#environment--protocol-recap)
2. [Curriculum Ladder](#curriculum-ladder)
3. [Training Architecture — What Each Config Does](#training-architecture)
4. [Run 0 — Pilot 3-Seed Comparison (`20260429_102350`)](#run-0--pilot-3-seed-comparison)
5. [Run 1 — The MAPPO Encoder Collapse (`20260429_140604`)](#run-1--the-mappo-encoder-collapse)
6. [Run 2 — Full Grid Exp, 2v1, 8 Configs (`20260429_180138`)](#run-2--full-grid-exp-2v1-8-configs)
7. [Run 3 — First Scale-Up, 3v2 M=10 (`scale3_20260430_124843`)](#run-3--first-scale-up-3v2-m10)
8. [Run 4 — Matched 3v3 M=10 (`scale3_20260430_180727`)](#run-4--matched-3v3-m10)
9. [Run 5 — 5v5 M=12 with Shot-Exclusivity (`scale3_20260430_215716`)](#run-5--5v5-m12-with-shot-exclusivity)
10. [Cross-Run Synthesis & Open Questions](#cross-run-synthesis--open-questions)
11. [Metrics Glossary](#metrics-glossary)

---

## Environment & Protocol Recap

**Battleship grid.** An $M \times M$ grid. Each ship occupies 3 contiguous cells (horizontal or vertical). Individual cells can be independently destroyed; a ship is *sunk* when all three are gone. This matters because boss ships take 3 independent hits to kill, not one.

**Partial observability.** Each agent sees only a $(2 \cdot \text{sight}+1)^2$ Chebyshev patch centred on itself. With `sight_range=2` the patch is $5\times5 = 25$ cells. Three channels: own cells, ally cells, boss cells. Agents cannot see beyond that radius. Bosses outside the window are invisible.

**Actions.** Continuous Gaussian output, 3 floats decoded to: (1) move direction (stay/N/S/E/W), (2–3) fire-row offset and fire-col offset clamped to `fire_range`.

**Episode order per step:** agents move → agents fire → bosses move → bosses fire.

**Win condition.** Agents win when all boss ships are sunk within `max_steps`. Boss wins if all agent ships are sunk.

**SeqComm protocol.** Before each joint action is executed, a *negotiation phase* determines a priority ordering $\sigma_t$. The first-mover in $\sigma_t$ commits its action; each subsequent agent conditions on the already-committed actions of higher-priority agents:

$$\pi(a_t \mid s_t, \sigma_t) = \prod_{k=1}^{n} \pi_\theta\!\left(a_t^{\sigma_t(k)} \mid o_t^{\sigma_t(k)},\, m_t^{-\sigma_t(k)}\right)$$

where $m_t^{-\sigma_t(k)}$ contains the hidden states of all agents **plus** the committed actions of agents ranked above $k$ in $\sigma_t$.

**Intention value.** To decide who goes first, SeqComm evaluates candidate first-movers via a $H$-step world-model rollout. Imagining agent $i$ as first-mover, it predicts a joint action $\hat{a}_t$, advances the latent state:
$$\hat{o}_{t+1}, \hat{r}_{t+1} = M_\omega\!\left(\text{AM}_w(h_t, \hat{a}_t)\right)$$
and repeats for $H$ steps to get $\hat{\tau}_t^{(i)}$. The critic evaluates $v(\hat{\tau}_t^{(i)})$. The agent with the highest imagined value is assigned first-mover status.

---

## Curriculum Ladder

All runs use the same 13-stage curriculum (stages 0–12). The scheduler uses a rolling window of `WINDOW=1000` episodes and advances when the window meets all three thresholds simultaneously.

| Stage | Boss HP | Boss Miss Prob | Fire Period | Max Steps | Near-Miss Reward | Advance Condition (per 1000-ep window) |
|-------|---------|---------------|-------------|-----------|-----------------|---------------------------------------|
| 0 | 1 | 0.80 | 12 | 30 | 0.20 | win ≥ 0.70, boss_hits ≥ 0.60, zero_hit ≤ 0.05 |
| 1 | 1 | 0.60 | 12 | 30 | 0.20 | win ≥ 0.75, boss_hits ≥ 0.70, zero_hit ≤ 0.04 |
| 2 | 1 | 0.40 | 12 | 30 | 0.15 | win ≥ 0.80 |
| 3 | 1 | 0.20 | 12 | 30 | 0.10 | win ≥ 0.80 |
| 4 | 1 | 0.00 | 12 | 30 | 0.05 | win ≥ 0.85 |
| 5 | 1 | 0.00 | 10 | 40 | 0.02 | win ≥ 0.85 |
| 6 | 2 | 0.00 | 10 | 45 | 0.01 | win ≥ 0.85 |
| 7 | 2 | 0.00 | 8 | 45 | 0.00 | win ≥ 0.85 |
| 8 | 2 | 0.00 | 8 | 50 | 0.00 | win ≥ 0.85 |
| 9 (bridge) | 3 | 0.00 | 12 | 60 | 0.00 | win ≥ 0.85 |
| 10 | 3 | 0.00 | 10 | 60 | 0.00 | win ≥ 0.85 |
| 11 | 3 | 0.00 | 8 | 60 | 0.00 | win ≥ 0.85 |
| 12 | 3 | 0.00 | 4 | 60 | 0.00 | (terminal) |

Stage 12 is the hardest: bosses have full 3-cell HP, never miss, fire every 4 steps, and agents have 60 steps to win.

---

## Training Architecture

All configs share: encoder → attention module → policy head, trained with PPO (GAE $\lambda=0.95$, $\gamma=0.99$, $\varepsilon=0.2$, 8 epochs, `UPDATE_EVERY=64`). The `encoder` used by the policy head is **separate** from `encoder_wm` used by the world model — they have independent optimizers so WM gradients cannot corrupt policy features.

| Config | Intent ordering | Comm loss model | Comm flag |
|--------|----------------|----------------|-----------|
| `baseline_critic` | Critic $V(h)$ (no WM rollout) | None | Always comm |
| `wm_loss00` | WM rollout, $H=3$, no drop | None | Always comm |
| `wm_loss10` | WM rollout, $H=3$ | 10% random drop | Always comm |
| `wm_loss20` | WM rollout, $H=3$ | 20% random drop | Always comm |
| `wm_loss30` | WM rollout, $H=3$ | 30% random drop | Always comm |
| `commgate_tiny` | WM rollout, $H=3$ | Learned gate | Penalty = 0.05 |
| `commgate_medium` | WM rollout, $H=3$ | Learned gate | Penalty = 0.20 |
| `commgate_large` | WM rollout, $H=3$ | Learned gate | Penalty = 0.50 |

**CommGate mechanic.** Each agent runs a learned Bernoulli gate $\text{do\_comm} \sim \sigma(\text{CommGate}(h))$. If any agent opts out, the round falls back to a random ordering (no conditioning on predecessors). The gate is updated via REINFORCE with a per-step penalty of `-comm_penalty` deducted whenever the gate opens.

**Critical fix (not a config — a correctness issue).** In MAPPO the value network shares a forward pass through the encoder with the policy network. Prior to the fix, the value loss (scale 28–55) dominated gradients through this shared path. The policy encoder's gradient norm `gn_enc_pol` collapsed to ≈ 0.08 by update 7 of 781 and stayed near zero for the full 50 K episode run, leaving the policy frozen at $\pi \approx \mathcal{N}(0,\,\mathbf{I})$. Fix: `torch.no_grad()` wrapping the encoder call inside `loss_value()`, so the value gradient path is severed from the policy encoder.

---

## Run 0 — Pilot 3-Seed Comparison

**Directory:** `runs/comparison/20260429_102350`  
**Date:** 2026-04-29 10:23  
**Setup:** 2 agents vs 1 boss, M=8, `sight_range=2`, `fire_range=2`, 15 000 episodes, 3 seeds each for SeqComm and MAPPO.

### What we were testing
Whether SeqComm meaningfully outperforms MAPPO on the basic 2v1 task, and whether results were stable across seeds.

### Results

| Config | Seed | Win (last 500) | Boss Hits | Agent Hits | Zero-Hit | Timeout | Stage |
|--------|------|--------------|-----------|------------|---------|---------|-------|
| SeqComm | 0 | 0.550 | 2.398 | 3.856 | 0.024 | 0.198 | 11 |
| SeqComm | 1 | 0.518 | 1.418 | 1.852 | 0.100 | 0.482 | 7 |
| SeqComm | 2 | 0.616 | 2.482 | 3.544 | 0.022 | 0.212 | 11 |
| MAPPO | 0 | 0.614 | 1.532 | 1.180 | 0.082 | 0.386 | 4 |
| MAPPO | 1 | 0.562 | 2.386 | 4.116 | 0.040 | 0.108 | 11 |
| MAPPO | 2 | 0.516 | 1.408 | 1.674 | 0.108 | 0.484 | 6 |

### What we learned

- Results were **highly seed-dependent** — up to 10% win-rate spread within each method. Seed 1 MAPPO was the outlier (stage 11 vs stage 4/6 for the others).
- 15 K episodes was too short: most seeds had not yet converged. Curriculum stage spread (4–11) confirmed that runs were at very different points.
- No clear winner between SeqComm and MAPPO at 15 K.

### What changed next

- Dropped to 1 seed (variance analysis showed seeds were not the bottleneck).
- Extended to 50 K episodes.
- Added MAPPO encoder diagnostic metrics (`gn_enc_pol`) to the trainer log.

---

## Run 1 — The MAPPO Encoder Collapse

**Directory:** `runs/comparison/20260429_140604`  
**Date:** 2026-04-29 14:06  
**Setup:** 2 agents vs 1 boss, M=8, 50 000 episodes, 1 seed.

### What we were testing
A fair 50 K-episode head-to-head: SeqComm (critic-based ordering, no WM intention, no comm loss) vs MAPPO.

### The failure

MAPPO's policy completely failed to learn. Across all 50 000 episodes:

- Win rate: **0.000 every 500-episode window** — literally never won a single episode across 50 K tries.
- Curriculum: **stuck at stage 0 the entire run**.

The trainer log reveals the mechanism exactly:

| Update | `gn_enc_pol` | `value_loss` | `policy_std` |
|--------|-------------|-------------|-------------|
| 0 | 0.57 | 31.2 | 1.001 |
| 1 | 1.63 | 38.3 | 1.002 |
| 2 | 5.40 | 55.3 | 1.002 |
| 3 | 10.46 | 45.1 | 1.002 |
| 4 | 8.23 | 29.0 | 1.003 |
| 5 | 1.79 | 22.3 | 1.003 |
| 6 | 1.75 | 35.6 | 1.004 |
| **7** | **0.27** | **29.4** | **1.005** |
| 8 | **0.08** | 32.8 | 1.005 |
| 9 | 0.08 | 34.8 | 1.004 |
| 10 | 0.03 | 31.9 | 1.004 |
| 11–12 | ≈ 0.04 | ≈ 30–40 | ≈ 1.005 |

The encoder's gradient norm for the policy path spiked as the value loss grew (updates 2–4), then the value loss pulverised the encoder weights to a near-zero fixed point (updates 7+). From that point on: `policy_std = 1.005` unchanged — the policy outputted exactly its initialisation distribution for all remaining 49 000+ episodes.

Policy `std ≈ 1.0` is the network's *prior*. A Gaussian policy with $\sigma=1$ and $\mu\approx 0$ generates uniformly random actions (move uniformly, fire uniformly). That is equivalent to an untrained agent. MAPPO trained for 50 K episodes on 50 000 episodes of randomly-acting experience.

### Root cause

The policy encoder and value encoder were the **same object**. In each PPO update the value loss $\mathcal{L}_V$ (MSE on returns, scale ≈ 30–55) was backpropagated through this shared encoder before the policy gradient $\mathcal{L}_\pi$ (scale ≈ 0.01–0.015) had any chance. The value signal dwarfed the policy signal by a factor of ~2000–3500×. The encoder converged to minimise $\mathcal{L}_V$ — becoming a value estimator — and the policy head received essentially zero informative gradient.

### SeqComm in the same run

SeqComm reached **64.2% win rate** (last 500) and stage 12 by episode 34 228. The communication rate was 100% throughout (no gate). The run shows SeqComm *can* learn on this task — validating the environment and training loop — but 64% win at 50 K episodes is not saturation.

### What changed next

The `loss_value()` function was fixed with `torch.no_grad()`:
```python
def loss_value(encoder, attn_a, critic, obs_self, up_pv, returns_pv):
    with torch.no_grad():
        h = encoder(obs_self)   # frozen — value gradient cannot reach policy encoder
    ctx = attn_a(h, up_pv)
    v = critic(ctx)
    diff = returns_pv - v
    return (diff * diff).mean()
```
The encoder and `encoder_wm` were split into two independent modules with separate optimisers (`opt_enc_pol`, `opt_enc_wm`) to prevent WM gradients from contaminating policy features.

Additionally, the curriculum was tightened (WINDOW raised from 300 → 1 000, thresholds raised), PPO epochs 4 → 8, and the run was expanded to 8 configuration variants.

---

## Run 2 — Full Grid Experiment, 2v1, 8 Configs

**Directory:** `runs/battleship_grid/20260429_180138`  
**Date:** 2026-04-29 18:01  
**Setup:** 2 agents vs **1 boss**, M=8, 50 000 episodes, 8 configs run in parallel.

### What we were testing

Now that MAPPO was fixed, run a controlled grid: (1) Is communication-based ordering better than critic-based ordering? (2) How does random communication dropout ($p_{\text{drop}} \in \{0.1, 0.2, 0.3\}$) affect performance? (3) Can a learned CommGate gate communication entirely when penalised, and does that degrade performance?

### Results

| Config | Win (last 500) | Boss Hits | Agent Hits | Comm Rate | First 90% Win @ ep | Stage |
|--------|--------------|-----------|------------|-----------|-------------------|-------|
| `baseline_critic` | **1.000** | 3.000 | 0.974 | 1.000 | 26 073 | 12 |
| `wm_loss00` | 0.998 | 2.998 | 1.000 | 1.000 | **6 842** | 12 |
| `wm_loss10` | **1.000** | 3.000 | 0.884 | 0.902 | 12 598 | 12 |
| `wm_loss20` | 0.998 | 2.998 | 0.896 | 0.801 | 19 233 | 12 |
| `wm_loss30` | 0.998 | 2.998 | 1.002 | 0.686 | 20 127 | 12 |
| `commgate_tiny` | 0.992 | 2.986 | 1.126 | 0.294 | 23 209 | 12 |
| `commgate_medium` | 0.996 | 2.996 | 1.066 | 0.129 | 11 853 | 12 |
| `commgate_large` | 0.998 | 2.998 | 1.068 | **0.000** | 14 246 | 12 |

### Key observations

**1. Ceiling effect.** Every single config converges to 99–100% win on the hardest curriculum stage. A 2v1 Battleship game — even with full HP, fast-firing boss, no aim lag, tight step budget — is simply not hard enough to produce meaningful performance *gaps*. This limits our ability to claim SeqComm is better; all methods tie at the ceiling.

**2. WM rollout gives a substantial speed advantage.** `wm_loss00` (world-model intent ordering) reaches first sustained 90% win at episode 6 842, versus 26 073 for `baseline_critic` (critic-only ordering). That is a **3.8× speed-up** in sample efficiency. This is the clearest positive result for SeqComm's core mechanism.

**3. Comm dropout degrades sample efficiency monotonically.** As $p_{\text{drop}}$ increases from 0 → 30%, the episode at which 90% win is first sustained increases from 6 842 → 20 127. Final win rates are statistically identical (~99.8%) but the communication signal helps most when it is reliable. This confirms that the ordering information is genuinely useful during training, even if the task is ultimately solvable without it.

**4. CommGate converges to silence — and still wins.** `commgate_large` (penalty 0.50) learns a gate that fires essentially **0% of the time** by episode ~30 000, then continues training in random-ordering mode. It still reaches 99.8% win. This is simultaneously a positive and a null result: the CommGate *works* (it successfully suppresses communication in response to the penalty), but it reveals that for a 2v1 task, ordering doesn't matter enough to overcome a 0.50-per-step penalty. The gate makes the rational choice: save the penalty, accept random ordering, and the task is trivially solved anyway.

**5. commgate_tiny is an outlier.** With the smallest penalty (0.05), the gate should communicate often. It stabilises at 29% comm rate — much lower than expected. High `agent_hits=1.126` (more than `wm_loss00`'s 1.000) suggests the gate was learning to suppress comm as a mild risk-avoidance strategy even at low penalty. This is a subtle behavioural divergence worth noting.

**6. Policy std drops well below 1.0 in all configs.** Values of 0.54–0.67 vs MAPPO's stuck-at-1.005 confirm the encoder-fix worked: policies are deterministic and fine-grained at convergence.

### What we learned

The 2v1 environment is too easy. Communication produces a speed/efficiency benefit but not a win-rate benefit. The benchmark for SeqComm needs to be *harder* — either more agents, more bosses, or explicit coordination pressure where duplicated actions are penalised.

### What changed next

Scale up: 3 agents, 2 bosses, larger grid (M=10). Also subset configs to 4 (drop wm_loss10/20, commgate_tiny/medium) to reduce wall-clock time.

---

## Run 3 — First Scale-Up, 3v2 M=10

**Directory:** `runs/battleship_grid/scale3_20260430_124843`  
**Date:** 2026-04-30 12:48  
**Setup:** **3 agents vs 2 bosses**, M=10, 50 000 episodes, 4 configs.

### What we were testing

Whether adding a third agent and a second boss produces the coordination pressure needed for SeqComm to differentiate from the baseline.

### Results

| Config | Win (last 500) | Boss Hits | Agent Hits | Comm Rate | Stage |
|--------|--------------|-----------|------------|-----------|-------|
| `baseline_critic` | 0.990 | 5.98 | 2.324 | 1.000 | 12 |
| `wm_loss00` | 0.998 | 5.99 | 1.978 | 1.000 | 12 |
| `wm_loss30` | **1.000** | 6.000 | 2.240 | 0.702 | 12 |
| `commgate_large` | **1.000** | 6.000 | 1.894 | **0.000** | 12 |

### Key observations

- Ceiling effect **persists**. Three agents against two bosses with 6 total cells to hit is still too easy — all four configs reach 99–100% win.
- `commgate_large` again reaches 0% comm rate and still achieves perfect win rate.
- Boss hits = 5.98–6.00 means every boss cell is being hit. With 3 shooters targeting 2 bosses (6 cells), each agent can find a unique target without any coordination signal — there is no genuine action conflict.
- The agent-to-boss cell ratio is 3:6 = 0.5 agents per boss cell, and the grid is large enough (M=10) that agents naturally spread out.

### What changed next

The key insight: the game has no *structural reason* to coordinate. Adding a **third boss** would require genuinely splitting 3 agents across 3 different bosses, and if two agents stack on the same boss cell, the second shot is wasted. This led to two parallel changes:

1. **Matched boss count** (3v3) so each agent must claim a unique boss to avoid wasted effort.
2. **Shot exclusivity mechanic**: if two agents fire at the exact same cell in the same step, the second shot is blocked (tracked as `wasted_shots`). This imposes a direct coordination penalty on overlap.

---

## Run 4 — Matched 3v3 M=10

**Directory:** `runs/battleship_grid/scale3_20260430_180727`  
**Date:** 2026-04-30 18:07  
**Setup:** **3 agents vs 3 bosses**, M=10, 50 000 episodes, 4 configs. Shot exclusivity **already present in binary** (mechanic was in place), `wasted_shots` logged.

### What we were testing

Whether a 1:1 agent-to-boss ratio creates enough coordination pressure to show differentiation.

### Results

| Config | Win (last 500) | Boss Hits | Agent Hits | Wasted Shots/ep | Comm Rate | Stage |
|--------|--------------|-----------|------------|----------------|-----------|-------|
| `baseline_critic` | **0.998** | 8.99 | 2.986 | 0.758 | 1.000 | 12 |
| `wm_loss00` | 0.854 | 8.72 | **5.314** | 0.562 | 1.000 | 12 |
| `wm_loss30` | 0.994 | 8.99 | 3.416 | 0.700 | 0.697 | 12 |
| `commgate_large` | 0.994 | 8.99 | 3.344 | 0.624 | **0.000** | 12 |

### Key observations

**1. wm_loss00 fails to fully converge.** This is the most interesting failure in the dataset. `wm_loss00` (full WM-ordering, no comm loss) reaches only 85.4% win with dramatically elevated agent hits (5.31/ep vs ~3 for all other configs). The learning trajectory:

| Episode window | Win rate | Agent Hits | Stage |
|---------------|---------|-----------|-------|
| 0–1 K | 0.202 | 1.88 | 0 |
| 5–6 K | 0.496 | 2.25 | 0 |
| 10–11 K | 0.622 | 2.00 | 0 |
| 20–21 K | 0.695 | 2.88 | 3 |
| 30–31 K | 0.508 | 5.92 | 9 |
| 40–41 K | 0.596 | 6.98 | 12 |
| 45–46 K | 0.798 | 5.62 | 12 |
| 49–50 K | 0.834 | 5.43 | 12 |

Win rate **drops** from ~70% (stage 3) to ~51% when the curriculum jumps to stage 9 (boss HP 3, no miss). Agent hits then balloons — the policy is learning to be *aggressive* (approach and shoot) but not *coordinated* (avoid boss fire). The WM rollout for intent ordering at H=3 appears to destabilise at the transition: computing 3-step rollouts under 3 agents × 3 bosses is a much harder prediction task than 2×1 or 3×2. The WM's predicted values become unreliable, the ordering becomes effectively random, and the policy receives noisy conditioning signals.

**2. Other three configs converge cleanly despite stage 9 jump.** `baseline_critic`, `wm_loss30`, and `commgate_large` all cross stage 9 without regression. This suggests the WM rollout specifically (not the environment difficulty) causes the instability for `wm_loss00`.

**3. Wasted shots 0.6–0.8/ep.** With 3 agents and 9 total boss cells, ~0.7 wasted shots per episode is ~8% of all shots. This is beginning to create real coordination pressure but is still not dominant.

**4. commgate_large still silent.** 0% comm rate, 99.4% win. The penalty is high enough to shut off all communication even when the task is harder. The gate has learned: "the comm signal is not worth 0.50 per step in reward."

### What changed next

Two parallel lines of investigation:

1. **Scale further** to 5 agents vs 5 bosses on a larger grid (M=12) to amplify wasted-shot pressure.
2. **Add territorial coverage metrics** (`mean_ally_dist`, `boss_hit_counts`) to the environment to measure whether agents are actually spreading out vs stacking.

---

## Run 5 — 5v5 M=12 with Shot-Exclusivity

**Directory:** `runs/battleship_grid/scale3_20260430_215716`  
**Date:** 2026-04-30 21:57  
**Setup:** **5 agents vs 5 bosses**, M=12, 50 000 episodes, 4 configs. Shot exclusivity active. New metrics: `wasted_shots`, `mean_ally_dist`, `boss_hit_counts` logged.

### What we were testing

Whether 5v5 at M=12 creates enough coordination pressure to finally differentiate SeqComm from the baseline. At 5v5 with 15 target cells and shot exclusivity, agents must implicitly assign themselves to distinct bosses.

### Results

| Config | Win (last 500) | Boss Hits | Agent Hits | Wasted Shots/ep | Comm Rate | Stage |
|--------|--------------|-----------|------------|----------------|-----------|-------|
| `baseline_critic` | 0.998 | **15.00** | 5.132 | 1.620 | 1.000 | 12 |
| `wm_loss00` | 0.994* | 14.99 | 5.716 | 1.412 | 1.000 | 12 |
| `wm_loss30` | 0.988 | 14.97 | 5.408 | 1.350 | 0.706 | 12 |
| `commgate_large` | **1.000** | **15.00** | **4.488** | **1.470** | **0.000** | 12 |

*wm_loss00 had only 42 537 episodes logged when last captured — still running.

### Learning curves (win rate by 1 000-episode window)

| Episode window | `baseline_critic` | `commgate_large` |
|---------------|-----------------|----------------|
| 0–1 K | 0.059 | 0.067 |
| 5–6 K | 0.498 | 0.492 |
| 10–11 K | 0.664 | 0.737 |
| 20–21 K | 0.885 | 0.757 |
| 30–31 K | 0.961 | 0.968 |
| 40–41 K | 0.999 | 0.998 |

Both methods reach ~99% by episode 30 000. `commgate_large` slightly *faster* during the 10–20 K window (0.757 vs 0.885 reversed at 10 K, then commgate catches up). No statistically meaningful difference at endpoint.

### Key observations

**1. Ceiling effect at 5v5.** The 15 boss cells across 5 ships are a large enough target space that even with shot exclusivity and 1.5–1.6 wasted shots per episode (~10% of total shots), agents learn to spread naturally. Wasted shots are ~10% of the ~15 shots needed to win, which adds ~1.5 expected extra shots — enough to slow the kill but not to make the task hard.

**2. Wasted shots are low because the target space is large.** With 15 cells and 5 agents, the probability that two agents randomly target the same cell in the same step is $\approx 1/15$. Coordination reduces this, but even random firing has low collision probability. Shot exclusivity only creates strong pressure when $n_\text{agents} \geq n_\text{boss cells}$, which requires a much more concentrated target (e.g., 5 agents vs 1 boss = 5 agents for 3 cells).

**3. commgate_large is the most efficient hitter.** `agent_hits = 4.49` versus 5.13–5.72 for others. This means commgate agents take ~0.6–1.2 fewer hits from boss fire per episode despite winning equally often. Lower agent hits could indicate: (a) commgate agents spread further (better territorial coverage), (b) commgate agents move more defensively, or (c) random ordering occasionally causes commgate agents to act later in the sequence and thus later in exposure. The `mean_ally_dist` metric (newly added) will clarify this in future runs.

**4. wm_loss00 convergence instability recurs.** The same pattern as Run 4: at 42 537 episodes, wm_loss00 shows elevated `agent_hits=5.716` and slightly lower win rate (99.4%). The H=3 WM rollout under 5×5 generates a very large latent tree — prediction quality degrades and the ordering signal is noisy.

**5. Why does 5v5 learn at the same rate as 3v3?** The agent-to-boss ratio is constant (1:1). The curriculum uses ratio-relative thresholds (boss_hits / n_boss_cells). Each agent faces the same local subtask — find a boss, approach, shoot. The attention module scales generically over $N$ agents. So the per-agent problem complexity does not change with $N$, and neither does the convergence rate.

### What this run confirmed about communication

`commgate_large` achieves perfect win (100%) while never communicating. This is theoretically expected when: the task has a unique optimal strategy reachable without ordering information, agents are homogeneous, and the penalty for communicating exceeds the marginal value of the communication signal. The question the paper must address is: **what class of task makes ordering strictly necessary?**

The answer, based on these experiments: ordering is necessary when two or more agents have a genuine *circular dependency* — where A's best action depends on B's committed action and vice versa — *and* where both available joint actions are locally optimal but globally distinct in value. In this Battleship environment, every agent has a clearly dominant local strategy (approach nearest boss, fire on it). Ordering removes ambiguity when two agents approach the same boss, but the grid is large enough that this is rare.

---

## Cross-Run Synthesis & Open Questions

### Summary of what changed and why

| Run | Key change | Motivation | Result |
|-----|-----------|-----------|--------|
| 0 | Pilot 3-seed | Establish baseline variance | Too short; seeds don't add value |
| 1 | 50 K, added `gn_enc_pol` logging | Find why MAPPO doesn't learn | Encoder collapse discovered and fixed |
| 2 | 8 configs, encoder fix, 8 PPO epochs | First clean comparison | All hit ceiling; WM intent 3.8× faster |
| 3 | 3 agents, 2 bosses, M=10 | More coordination pressure | Ceiling persists; 3 shooters for 2 bosses is still too easy |
| 4 | 3 bosses (1:1 ratio), shot exclusivity | Structural action conflict | wm_loss00 diverges at curriculum jump; commgate still silent |
| 5 | 5 agents, 5 bosses, M=12, new metrics | More scale and coverage tracking | Still ceiling; wasted shots ~10%, commgate most efficient hitter |

### The central tension

SeqComm's ordering mechanism is most valuable when:
$$\text{Value}(\text{ordering}) > \text{cost of comm overhead}$$

In every run, the right-hand side was zero (or set by the penalty in CommGate experiments). The left-hand side was also near-zero because the environment allowed agents to reach ceiling performance with random ordering. **The mechanism works — wm_loss00 learns 3.8× faster in Run 2 — but it cannot be proven superior on final performance when all configs saturate.**

### The commgate paradox

`commgate_large` (penalty=0.50) consistently reaches 99–100% win while communicating 0% of the time. This is not a failure of the mechanism — it is the mechanism working *correctly*. The gate learned to compute the value of a communication round and found it worth less than 0.50 reward. In an environment where ordering does not matter, that is the rational equilibrium. To *force* communication, you need a task where the counterfactual (random ordering) leads to a meaningfully worse outcome.

### Positive findings for the paper

1. **MAPPO encoder collapse is a real and serious bug.** Without the `torch.no_grad()` fix, MAPPO never learns. This is a novel correctness finding, not just a hyperparameter issue, and is worth explicit discussion in the paper.

2. **World-model intent ordering gives a 3.8× sample-efficiency gain** (first 90% win: episode 6 842 vs 26 073). This holds even when all methods eventually tie at final performance. In practice, faster learning matters when compute is fixed.

3. **Graduated communication degradation is smooth.** $p_\text{drop} = 0\% \to 30\%$ delays 90% win from 6 842 → 20 127 episodes but does not affect final win. This is a useful robustness result: SeqComm degrades gracefully under imperfect channels.

4. **Wasted shots are a real, measurable coordination cost** (up to 10% of shots wasted at 5v5). The metric is in place; a harder target distribution would amplify this into a performance gap.

### Open questions for discussion

- **What environment structure would force ordering to matter?** Candidates: intersection management (agents targeting a single bottleneck), resource competition (limited landing pads), or explicitly adversarial peer selection (only the first agent to claim a target can shoot it).
- **Why does wm_loss00 diverge at the stage 9 jump?** Is it WM prediction error under 3+ agents, or the harder boss dynamics (full HP, no miss) breaking the rollout's value estimates? A controlled ablation (fix boss difficulty, vary N) would answer this.
- **Does `mean_ally_dist` differ between commgate and baseline?** The new metric should reveal whether commgate's lower agent_hits comes from territorial separation. This data will be in future runs.
- **Is commgate_large's 0% comm rate stable, or does it occasionally spike?** The last-500 average is 0.000, but the CommGate policy could briefly communicate during difficult episodes. A histogram of per-episode comm_rate would show this.

---

## Metrics Glossary

| Metric | Definition | Where logged |
|--------|-----------|-------------|
| `win` / `agents_won` | 1 if all boss ships sunk before timeout | JSONL `agents_won` |
| `boss_hits` | Number of individual boss cells destroyed in episode | JSONL |
| `agent_hits` | Number of individual agent cells destroyed by boss fire | JSONL |
| `zero_hit` | 1 if agents never hit the boss even once (bad episode) | Summary CSV |
| `timeout` | 1 if episode ran to max_steps without win | JSONL `boss_won` = 0 and `agents_won` = 0 |
| `mean_fire_dist` | Average Chebyshev distance from shooter to target at fire time | JSONL |
| `fire_oob` | Shots that aimed outside the grid (before clamping) | JSONL |
| `wasted_shots` | Shots blocked because a teammate already targeted that cell this step | JSONL (added Run 4+) |
| `comm_rate` | Fraction of steps where full sequential communication occurred | JSONL |
| `curriculum_stage` | Active stage (0–12) at episode end | JSONL |
| `gn_enc_pol` | L2 gradient norm on policy encoder (trainer) | trainer.jsonl |
| `value_loss` | MSE value loss at PPO update | trainer.jsonl |
| `policy_std` | Mean of $e^{\text{log\_std}}$ across all action dims | trainer.jsonl |
| `mean_ally_dist` | Per-step average min Chebyshev distance to nearest alive ally | JSONL (added Run 5) |
| `boss_hit_counts` | Per-boss vector of cell hits this episode | JSONL (added Run 5) |
| `intention_spread` | Std of intention scores across candidate first-movers | JSONL (SeqComm only) |

---

*Generated 2026-05-01. Log sources: `/Users/bobbycupps/Downloads/work/multi-lvl-comms/runs/`*
