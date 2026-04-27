# SeqComm Battleship — Paper Experiments

Two targeted experiments on the collaborative battleship environment (N=2 agent ships vs 2 boss ships on an 8×8 grid). Both use the same curriculum training baseline and differ only in the negotiation protocol tested.

Run with 2 seeds. Results land in timestamped subdirectories under `runs/`.

---

## What the baseline is

In the standard SeqComm protocol, agents negotiate a priority ordering each timestep before executing actions. The ordering is determined by an **intention signal** — a scalar score each agent broadcasts. Whoever scores highest gets to act first and their chosen action is visible to the agents below them before those agents commit.

The current baseline uses the **critic value V(h_i)** as a cheap proxy for the intention signal: encode observation → hidden state h, pass through attention with no context → scalar value estimate. This is a stand-in for the world model rollout described in the SeqComm paper.

The experiments ask two independent questions about the negotiation phase.

---

## Experiment 1 — Does the real world model help, and what happens when messages fail?

### What it tests

The SeqComm paper (Eq. 4) says agents should compute their intention using a multi-step world-model rollout: each agent broadcasts its hidden state h_i, collects all h_j from neighbours, then locally simulates H steps forward using the shared policy and world model to estimate cumulative reward. The agent with the highest predicted return goes first.

Our baseline skips this and uses the critic directly. **Experiment 1 checks whether replacing the critic proxy with the real H=2 rollout improves coordination**, and then **degrades the communication channel** by randomly dropping negotiation rounds at 10%, 20%, and 30% probability to simulate real-world packet loss or radio interference.

When a round drops, there is no negotiation — agents use a random ordering for that timestep.

### The five conditions

| Config name | Ordering signal | Fraction of steps with random ordering |
|-------------|----------------|----------------------------------------|
| `baseline_critic` | V(h_i) critic | 0% — comms always succeed |
| `wm_loss00` | H=2 WM rollout | 0% — comms always succeed |
| `wm_loss10` | H=2 WM rollout | 10% of steps random |
| `wm_loss20` | H=2 WM rollout | 20% of steps random |
| `wm_loss30` | H=2 WM rollout | 30% of steps random |

### What the numbers will show

- `baseline_critic` vs `wm_loss00`: isolated effect of real world-model ordering. If WM ordering is better, win rate should be higher.
- `wm_loss00` → `wm_loss10` → `wm_loss20` → `wm_loss30`: degradation curve. How robust is coordination to channel failure? The comm_rate column in the log directly measures the fraction of steps where ordering was real (not random) — you can confirm it matches the intended loss level.
- If baseline_critic ≈ wm_loss30, it suggests the critic proxy is roughly as informative as communicating 70% of the time via the world model.

### How to run

```bash
EXP=exp1 EPISODES=5000 SEEDS="0 1" bash battleship/run_battleship_grid.sh
```

This runs 5 configs × 2 seeds = **10 sequential runs**.

### Where results go

```
runs/battleship_grid/<timestamp>/          ← one folder per launch of the script
├── summary.csv                            ← one row per run; key columns below
├── baseline_critic_seed0/
│   ├── logs/seqcomm_<ts>.jsonl            ← per-episode JSON (win, boss_hits, comm_rate, …)
│   ├── trainer.jsonl                      ← per-update Python diagnostics
│   ├── figures/baseline_critic_seed0.png  ← learning curve plot
│   └── stdout/cpp.out, python.out         ← raw process logs
├── baseline_critic_seed1/
├── wm_loss00_seed0/
│   └── …
…
```

Key columns in `summary.csv`:

| Column | Meaning |
|--------|---------|
| `last500_win` | Win rate over final 500 episodes |
| `last500_boss_hits` | Average boss hits per episode |
| `last500_comm_rate` | Fraction of negotiation steps using real ordering |
| `last500_fire_dist` | Average fire distance from boss (lower = agents learned to close in) |
| `final_curriculum_stage` | Which curriculum stage was reached |

---

## Experiment 2 — Do agents learn to stop communicating when it costs them?

### What it tests

This experiment gives agents a **choice**: communicate in the negotiation phase, or skip it. Communication produces better ordering information but costs a small reward penalty. If any agent on the team opts out, the whole negotiation round is skipped and agents use a random ordering.

The question is: **given three different penalty sizes, does the fraction of steps where agents voluntarily communicate change as a function of the penalty?** With a tiny penalty the benefit of coordination should outweigh the cost and agents should keep communicating. With a large penalty they should learn to stop.

### The mechanism

At the start of each negotiation step:
1. Each agent independently samples `do_comm ~ Bernoulli(σ(CommGate(h)))` where `CommGate` is a tiny learned linear layer (embed_dim → 1 scalar).
2. Agents broadcast their intention or a no-comm sentinel depending on their decision.
3. If any agent broadcasts the sentinel, everyone falls back to random ordering.
4. Any agent that chose to comm has `comm_penalty` subtracted from their step reward.

The `CommGate` parameters are updated each training batch via REINFORCE: if communicating led to a higher net return (after penalty) than the episode baseline, the gate is nudged toward committing; otherwise it is nudged away.

### The three conditions

| Config name | Comm penalty per step | Expected learned behaviour |
|-------------|----------------------|---------------------------|
| `commgate_tiny` | 0.005 | Agents still comm almost always — ordering benefit exceeds cost |
| `commgate_medium` | 0.050 | Mixed — comm rate falls as policy matures and needs ordering less |
| `commgate_large` | 0.500 | Agents learn to mostly skip communication |

All three also use `--wm-intention` so that when agents do communicate, the ordering signal is the H=2 world-model rollout (same as `wm_loss00` in Experiment 1).

### What the numbers will show

The key result is the `last500_comm_rate` column across the three penalty conditions. A clean result shows a monotone decrease: tiny → ~1.0, medium → ~0.5–0.8, large → ~0.1–0.3.

Secondary result: compare `last500_win` across the three. If agents learn to skip comms under the large penalty, does the win rate also drop? This shows whether ordering information was actually valuable (it should be, since Experiment 1 tested this directly).

### How to run

```bash
EXP=exp2 EPISODES=5000 SEEDS="0 1" bash battleship/run_battleship_grid.sh
```

This runs 3 configs × 2 seeds = **6 sequential runs**.

**Note:** Experiment 2 initialises a separate `comm_gate.pt` module alongside the normal weights. Do not reuse a weights directory from a baseline or Experiment 1 run.

### Where results go

```
runs/battleship_grid/<timestamp>/
├── summary.csv
├── commgate_tiny_seed0/
│   ├── logs/seqcomm_<ts>.jsonl    ← comm_rate column shows per-episode comm frequency
│   ├── weights/comm_gate.pt       ← final learned comm gate (loadable for analysis)
│   └── …
├── commgate_tiny_seed1/
├── commgate_medium_seed0/
├── commgate_medium_seed1/
├── commgate_large_seed0/
└── commgate_large_seed1/
```

The per-episode JSONL log includes `"comm_rate": 0.72` on every line — plot this over training to see when agents learn to reduce communication.

---

## Running both experiments back to back

Each experiment produces its own timestamped folder so they never collide:

```bash
# Run Experiment 1 first (10 runs, ~8–12 hours depending on hardware)
EXP=exp1 EPISODES=5000 SEEDS="0 1" bash battleship/run_battleship_grid.sh

# Then Experiment 2 (6 runs, ~5–8 hours)
EXP=exp2 EPISODES=5000 SEEDS="0 1" bash battleship/run_battleship_grid.sh
```

Results for comparison across all runs:

```
runs/battleship_grid/
├── 20260427_<time_exp1>/summary.csv    ← Experiment 1 results
└── 20260427_<time_exp2>/summary.csv    ← Experiment 2 results
```

Load both CSVs and group by `config` to produce the paper tables.

---

## What the logs contain

Every run's JSONL log (one line per episode) looks like:

```json
{
  "ep": 1200,
  "reward": 4.21,
  "steps": 31,
  "boss_hits": 2,
  "agents_won": true,
  "comms_ok": 28,
  "comms_total": 31,
  "comm_rate": 0.903,
  "curriculum_stage": 4
}
```

- `comms_ok` — steps where ordering was determined by real negotiation (not random fallback)
- `comms_total` — total negotiation steps in this episode
- `comm_rate` — `comms_ok / comms_total`; this is the headline metric for both experiments

For Experiment 1 the `comm_rate` is mechanically set by `comms_loss_prob` and serves as a sanity check. For Experiment 2 it is a learned quantity that changes over training.
