# multi-lvl-comms

> **CS 2860 Final Project** · Lux Hogan-Murphy · Will Sherwood · Bobby Cupps  
> *No, You Go: Simulating Multi-Robot Coordination with Multi-Level Communication*

A C++ + Python implementation of **SeqComm** — multi-agent coordination via sequential communication — extended with communication dropout, learned communication gates, and a custom Battleship grid environment. Built on top of [Ding et al., NeurIPS 2024](multi-lvl-paper.pdf).

---

## What this is

Standard multi-agent RL algorithms (MAPPO, QMIX) have every agent choose its action simultaneously. This creates **circular dependencies**: agent A's best action depends on B's choice, but B's depends on A's. SeqComm breaks the cycle by running a *negotiation phase* before every timestep that establishes a priority ordering — the first-mover commits its action, and each subsequent agent conditions on what the agents above it already decided.

This repo tests three questions:

1. **Does world-model-based ordering outperform critic-based ordering?** (Yes — 3.8× faster to 90% win on 2v1)
2. **What happens when communication is unreliable or penalised?** (Dropout acts as regularisation; lower comm rate → more robust final policy)
3. **Can agents learn to decide for themselves whether to communicate?** (Yes — a learned gate suppresses all communication when the task doesn't require it)

---

## Repository layout

```
multi-lvl-comms/
│
├── battleship/                      ← main experiment
│   ├── battleship_env.hh/.cc        C++ grid environment — ships, hits, curriculum
│   ├── battleship_sim.cc            C++ training loop — IPC with Python via traj.bin
│   ├── train_battleship.py          Python trainer — PPO, world model, CommGate
│   ├── run_battleship_grid.sh       Launch one or many parallel runs (the main script)
│   ├── run_battleship_comparison.sh SeqComm vs MAPPO head-to-head
│   ├── plot_battleship.py           Per-run figure generator
│   └── make_paper_figures.py        Generates figures/fig1–fig3 for the paper
│
├── robot_sim/                       ← original seqcomm sim (intersection / gaussian)
│   ├── seqcomm_sim.cc               Single-episode demo — no libtorch needed
│   └── seqcomm_sim_trained.cc       Multi-episode C++ loop with weight reloading
│
├── cotamer/                         ← async coroutine runtime (C++20 coroutines)
│
├── training/                        ← Python-only training (older environments)
│
├── figures/                         ← paper figures (generated)
│   ├── fig1_curriculum.png
│   ├── fig2_2v1_grid.png
│   └── fig3_scale.png
│
├── runs/                            ← all training logs (gitignored)
│
├── weights_bs/                      ← TorchScript .pt files (C++ reads these)
├── final_report/
│   └── references.bib
├── EXPERIMENT_DEPOT.md              ← full run-by-run experimental narrative
├── CMakeLists.txt
├── requirements.txt
└── setup-local.sh                   ← one-shot local build script
```

---

## Architecture

### How training works

Training is a two-process loop. The **C++ simulator** (`battleship-sim`) runs episodes as fast as it can, writing completed trajectory batches to `weights_bs/traj.bin` and touching `weights_bs/traj.ready`. The **Python trainer** (`train_battleship.py`) watches for that signal, reads the batch, runs PPO updates, writes new weights to `weights_bs/weights.bin`, and signals back. The C++ process reloads and starts the next batch.

```
  ┌──────────────────────┐     traj.bin / traj.ready      ┌─────────────────────┐
  │   C++ battleship-sim │  ────────────────────────────►  │  Python PPO trainer │
  │                      │                                  │                     │
  │  • runs episodes     │  ◄────────────────────────────  │  • GAE + PPO        │
  │  • curriculum        │    weights.bin / weights.ready   │  • world model      │
  │  • logs JSONL        │                                  │  • CommGate         │
  └──────────────────────┘                                  └─────────────────────┘
```

### Neural network components

| Module | Role | Used by |
|--------|------|---------|
| `encoder` | MLP obs → 64-dim hidden state | Policy path (C++ reads) |
| `encoder_wm` | Separate MLP for world model features | WM path only |
| `attn_a` | Cross-attention over agent hidden states | Policy conditioning |
| `attn_w` | Attention for world model rollout | Intention ordering |
| `policy` | Gaussian action head | Action sampling |
| `critic` | State value estimator | GAE + critic ordering |
| `world_model` | Latent next-state + reward predictor | H-step intent rollout |
| `CommGate` | Linear → Bernoulli comm decision | CommGate configs only |

> **Critical implementation note**: `encoder` and `encoder_wm` are **separate modules with separate optimisers**. Value loss gradients cannot flow into the policy encoder. Without this, the policy encoder collapses to near-zero gradient norm by PPO update 7 and the policy never learns (stuck at `policy_std ≈ 1.0` for all 50 000 episodes).

### Configs

| Config | Ordering | Comm model |
|--------|----------|------------|
| `baseline_critic` | Critic V(h) | Always comm |
| `wm_loss00` | WM rollout H=3 | Always comm |
| `wm_loss10/20/30` | WM rollout H=3 | Random dropout 10/20/30% |
| `commgate_tiny` | WM rollout H=3 | Learned gate, penalty 0.05 |
| `commgate_medium` | WM rollout H=3 | Learned gate, penalty 0.20 |
| `commgate_large` | WM rollout H=3 | Learned gate, penalty 0.50 |

---

## Quickstart

### Prerequisites

```bash
# macOS
brew install cmake

# Python 3.10+ required (3.12 recommended)
python3 --version
```

### 1. Clone and build

```bash
git clone <repo-url> multi-lvl-comms
cd multi-lvl-comms

# Creates .venv, installs torch + deps, builds battleship-sim
./setup-local.sh
```

This does three things: creates `.venv/` with PyTorch CPU, detects the libtorch prefix from the installed wheel, then runs cmake and builds the `battleship-sim` binary into `build-local/battleship/`.

### 2. Run an experiment

```bash
# Default: 4 configs × 1 seed × 50 000 episodes (3v3, M=10 grid)
EXP=scale3 bash battleship/run_battleship_grid.sh
```

Results stream to `runs/battleship_grid/<timestamp>/` as JSONL. Each config runs two background processes (C++ sim + Python trainer); the script waits for both before starting the next.

**Common overrides:**

```bash
# Quick smoke test — 2 000 episodes, one config
EPISODES=2000 SINGLE_CONFIG=baseline_critic bash battleship/run_battleship_grid.sh

# Full 2v1 grid (8 configs — reproduces paper Exp 1+2)
EXP=exp2 EPISODES=50000 bash battleship/run_battleship_grid.sh

# 5v5 on a 12×12 grid
EXP=scale3 AGENTS=5 BOSS=5 M=12 EPISODES=50000 bash battleship/run_battleship_grid.sh

# SeqComm vs MAPPO head-to-head
bash battleship/run_battleship_comparison.sh
```

### 3. Monitor progress

```bash
tail -f runs/battleship_grid/<timestamp>/baseline_critic_seed0/logs/*.jsonl | \
  python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if 'agents_won' in d:
        print(f'ep={d[\"ep\"]:6d}  win={d[\"agents_won\"]}  boss_hits={d[\"boss_hits\"]}  stage={d.get(\"curriculum_stage\",0)}')
"
```

### 4. Generate paper figures

```bash
# Regenerates all three paper figures
.venv/bin/python battleship/make_paper_figures.py

# Single-run figure for any log file
.venv/bin/python battleship/plot_battleship.py \
    runs/battleship_grid/<timestamp>/<config>/logs/*.jsonl \
    --window 500
```

---

## Curriculum

Training uses a 13-stage curriculum (stages 0–12). A 1 000-episode rolling window must meet hit/win/zero-hit thresholds simultaneously to advance. Stage 12 is the hardest: full-HP bosses, no misses, fire period of 4 steps.

| Stages | Boss HP | Boss miss prob | Fire period |
|--------|---------|---------------|-------------|
| 0–4 | 1 | 0.80 → 0.00 | 12 |
| 5 | 1 | 0.00 | 10 |
| 6–8 | 2 | 0.00 | 10 → 8 |
| 9 (bridge) | 3 | 0.00 | 12 |
| 10–12 | 3 | 0.00 | 10 → 4 |

---

## Reproducing paper results

| Figure | Command | Expected |
|--------|---------|----------|
| Fig 1 — curriculum | `EXP=exp2 SINGLE_CONFIG=wm_loss00 EPISODES=50000 bash battleship/run_battleship_grid.sh` | Stage 12 by ~ep 34 000 |
| Fig 2 — 2v1 grid | `EXP=exp2 EPISODES=50000 bash battleship/run_battleship_grid.sh` | All 8 configs ≥ 99% win |
| Fig 3 — 3v3 | `EXP=scale3 AGENTS=3 BOSS=3 M=10 EPISODES=50000 bash battleship/run_battleship_grid.sh` | `wm_loss00` plateaus at ~85% |
| Fig 3 — 5v5 | `EXP=scale3 AGENTS=5 BOSS=5 M=12 EPISODES=50000 bash battleship/run_battleship_grid.sh` | All configs ≥ 99% win |

Full run-by-run experimental narrative with all numbers: [`EXPERIMENT_DEPOT.md`](EXPERIMENT_DEPOT.md)

---

## Reading the code

| Want to understand… | Read… |
|--------------------|-------|
| Environment dynamics | `battleship/battleship_env.hh` |
| C++ ↔ Python IPC | `battleship/battleship_sim.cc` |
| PPO + world model | `battleship/train_battleship.py` |
| CommGate | `train_battleship.py` → `class CommGate`, `update_comm_gate()` |
| Curriculum logic | `battleship/run_battleship_grid.sh` → `advance_curriculum()` |
| Math foundations | `math.md` |

---

## Citation

```bibtex
@inproceedings{Ding2024,
  author    = {Ding, Ziluo and Liu, Zeyuan and Fang, Zhirui and Su, Kefan
               and Zhu, Liwen and Lu, Zongqing},
  title     = {Multi-Agent Coordination via Multi-Level Communication},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
