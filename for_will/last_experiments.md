# Battleship SeqComm: Current Special Experiment Results

This file is the handoff note for the experiments packaged in `for_will/`.
These are not just ordinary smoke tests. They are the current "big result"
runs for two communication experiments:

1. **World-model ordering under communication loss**
2. **Learned optional communication with a communication penalty**

The baseline policy is still not learning as cleanly as we want, so the absolute
numbers should not be oversold. The important thing here is the controlled
comparison: every run uses the same Battleship task, same seed, same episode
count, same logging, and the same summary metrics.

All packaged runs here are:

- `seed=0`
- `episodes=1000`
- `update_every=16`
- `lr_policy=0.0003`
- `entropy=0.003`
- `near_boss=0.15`
- `H=2` for the world-model rollout
- `F=2` at the SeqComm call site; `F` is currently not used by the implementation

The result files live under:

```text
for_will/
```

Each run directory contains:

- `summary.csv`: one-row summary for the run
- `logs/*.jsonl`: per-episode environment metrics
- `trainer.jsonl`: optimizer / PPO / world-model diagnostics
- `figures/*.png`: the plot generated from the episode log

## Experiment 1: World-Model Ordering With Communication Loss

**Question.** If SeqComm uses a world model to choose the communication ordering,
how much does performance degrade when communication is unavailable some fraction
of the time?

**Mechanism.**

In the normal world-model ordering case, agents communicate hidden states during
the negotiation phase. Each agent uses the learned world model to roll forward
the current state and current policy for `H=2` steps, then uses the predicted
return as its ordering intention. This is the stronger ordering signal: it is
not just the critic value of the current observation, it is a short planning
rollout through the learned world model.

For the communication-loss variants, the simulator makes communication fail on
a fixed percentage of negotiation rounds:

- `wm_loss10`: communication unavailable 10% of the time
- `wm_loss20`: communication unavailable 20% of the time
- `wm_loss30`: communication unavailable 30% of the time

When communication fails, the agents cannot use the world-model intention
exchange for that round. The implementation falls back to a deterministic
no-message ordering so the launching phase can still proceed without deadlock.
The logged `comm_rate` is the fraction of negotiation rounds where real
communication ordering was used.

**Important missing baseline.**

The folder currently contains `wm_loss10`, `wm_loss20`, and `wm_loss30`, but it
does **not** contain the required `wm_loss00` run. To compute the final deltas
against "normal SeqComm with world-model ordering", we still need:

```bash
RUN_ROOT=for_will/$(date +%Y%m%d_%H%M%S)_wm_loss00 \
SINGLE_CONFIG="wm_loss00|16|0.0003|0.003|0.15" \
EXP=exp1 EPISODES=1000 SEEDS="0" \
bash battleship/run_battleship_grid.sh
```

Optional critic-only baseline:

```bash
RUN_ROOT=for_will/$(date +%Y%m%d_%H%M%S)_baseline_critic \
SINGLE_CONFIG="baseline_critic|16|0.0003|0.003|0.15" \
EXP=exp1 EPISODES=1000 SEEDS="0" \
bash battleship/run_battleship_grid.sh
```

### Current Packaged Experiment 1 Results

These are final-500-episode metrics from `summary.csv`.

| Run | Communication loss | Effective comm rate | Win rate | Boss hits / ep | Zero-hit eps | Timeout rate | Mean fire dist | Figure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `wm_loss10_seed0` | 10% | 89.9% | 60.6% | 2.502 | 1.4% | 36.4% | 2.701 | `for_will/20260427_030052/wm_loss10_seed0/figures/wm_loss10_seed0.png` |
| `wm_loss20_seed0` | 20% | 79.9% | 77.4% | 2.714 | 1.0% | 15.4% | 2.435 | `for_will/20260427_030443/wm_loss20_seed0/figures/wm_loss20_seed0.png` |
| `wm_loss30_seed0` | 30% | 69.9% | 48.8% | 2.168 | 7.6% | 49.4% | 2.863 | `for_will/20260427_030819/wm_loss30_seed0/figures/wm_loss30_seed0.png` |

**What this means right now.**

The 30% loss run clearly hurts performance: win rate drops to 48.8%, zero-hit
episodes rise, timeout rate rises, and aiming gets worse. The 20% run being
better than 10% is almost certainly single-seed variance and/or unstable early
learning, not a real claim that more message loss helps. The paper-safe claim is:
the logging is now good enough to quantify degradation, and the 30% condition
shows the expected failure mode.

Once `wm_loss00` is added, report the deltas like this:

```text
delta_win(loss X) = last500_win(loss X) - last500_win(wm_loss00)
degradation(loss X) = last500_win(wm_loss00) - last500_win(loss X)
```

The same delta should also be reported for:

- `last500_boss_hits`
- `last500_zero_hit`
- `last500_timeout`
- `last500_fire_dist`

## Experiment 2: Learned Optional Communication Gate

**Question.** Can agents learn when communication is worth paying for?

**Mechanism.**

Each agent has a learned communication gate:

```text
comm_gate(h_i) -> logit
sigmoid(logit) -> P(agent chooses to communicate)
```

During negotiation, each agent independently chooses whether to communicate.
If an agent communicates, it pays a penalty. If any agent chooses not to
communicate, the round cannot use normal world-model communication ordering and
falls back to no-message ordering before launching.

The communication penalty is varied:

- `commgate_tiny`: penalty `0.005`
- `commgate_medium`: penalty `0.050`
- `commgate_large`: penalty `0.500`

The logged `last500_comm_rate` is the effective fraction of negotiation rounds
where all agents communicated and real ordering was used. It is not simply the
per-agent probability of saying "yes"; it is the rate at which the system
actually got to use communication.

### Current Packaged Experiment 2 Results

| Run | Comm penalty | Effective comm rate | Win rate | Boss hits / ep | Zero-hit eps | Timeout rate | Mean fire dist | Figure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `commgate_tiny_seed0` | 0.005 | 20.9% | 65.2% | 2.514 | 2.4% | 32.2% | 2.705 | `for_will/20260427_032405/commgate_tiny_seed0/figures/commgate_tiny_seed0.png` |
| `commgate_medium_seed0` | 0.050 | 28.3% | 70.2% | 2.562 | 3.4% | 18.8% | 2.450 | `for_will/20260427_032405/commgate_medium_seed0/figures/commgate_medium_seed0.png` |
| `commgate_large_seed0` | 0.500 | 14.7% | 72.4% | 2.630 | 2.8% | 20.0% | 2.549 | `for_will/20260427_032405/commgate_large_seed0/figures/commgate_large_seed0.png` |

**What this means right now.**

The communication gate is doing something nontrivial. It is not communicating
all the time. The large-penalty run communicates least, but still has the best
single-seed win rate in this package. That is exactly the kind of result we
wanted to look for: sparse communication can still preserve performance when
the agents learn to use it selectively.

Do not overclaim yet. This is one seed, 1000 episodes, and the baseline learning
curve is still noisy. The strong claim we can make now is that the experiment
pipeline works and produces measurable tradeoffs between communication rate,
penalty, and performance.

## How To Turn The Results Into A Table

The easiest table source is `summary.csv`. Each row is one run. The columns
already contain the final-500-episode metrics.

Run this from the repo root to print a Markdown table from every packaged
`for_will` summary:

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

rows = []
for summary in sorted(Path("for_will").glob("*/summary.csv")):
    for row in csv.DictReader(summary.open()):
        run_dir = summary.parent / row["run_id"]
        log_path = next((run_dir / "logs").glob("*.jsonl"), None)
        meta = {}
        if log_path is not None:
            first = json.loads(log_path.open().readline())
            meta = first.get("_meta", {})
        rows.append({
            "run": row["run_id"],
            "config": row["config"],
            "episodes": row["episodes_logged"],
            "wm": meta.get("use_wm_intention", ""),
            "loss": meta.get("comms_loss_prob", ""),
            "gate": meta.get("use_comm_gate", ""),
            "penalty": meta.get("comm_penalty", ""),
            "win": 100 * float(row["last500_win"]),
            "boss_hits": float(row["last500_boss_hits"]),
            "zero": 100 * float(row["last500_zero_hit"]),
            "timeout": 100 * float(row["last500_timeout"]),
            "fire_dist": float(row["last500_fire_dist"]),
            "comm": 100 * float(row.get("last500_comm_rate", 1.0)),
        })

print("| Run | Config | Loss | Gate | Penalty | Win | Boss hits | Zero-hit | Timeout | Fire dist | Comm rate |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(
        f"| `{r['run']}` | `{r['config']}` | {r['loss']} | {r['gate']} | "
        f"{r['penalty']} | {r['win']:.1f}% | {r['boss_hits']:.3f} | "
        f"{r['zero']:.1f}% | {r['timeout']:.1f}% | {r['fire_dist']:.3f} | "
        f"{r['comm']:.1f}% |"
    )
PY
```

If `wm_loss00` is added to `for_will`, compute degradation relative to it:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

rows = []
for summary in sorted(Path("for_will").glob("*/summary.csv")):
    rows.extend(csv.DictReader(summary.open()))

base = next((r for r in rows if r["config"] == "wm_loss00"), None)
if base is None:
    raise SystemExit("No wm_loss00 row found. Run the missing baseline first.")

base_win = float(base["last500_win"])
print("| Config | Win | Delta vs wm_loss00 | Degradation |")
print("|---|---:|---:|---:|")
for r in rows:
    if not r["config"].startswith("wm_loss"):
        continue
    win = float(r["last500_win"])
    print(
        f"| `{r['config']}` | {100*win:.1f}% | "
        f"{100*(win - base_win):+.1f} pp | {100*(base_win - win):+.1f} pp |"
    )
PY
```

## What Each Metric Means

Use these interpretations when writing the results section:

- `last500_win`: fraction of the final 500 episodes where the agents sank the boss. Higher is better.
- `last500_boss_hits`: average number of boss hits per episode. The non-curriculum boss needs 3 hits to sink, so values closer to 3 are better.
- `last500_zero_hit`: fraction of episodes where agents never hit the boss. Lower is much better.
- `last500_timeout`: fraction of episodes that reached the step cap without either side winning. Lower is usually better if win rate is higher.
- `last500_fire_dist`: average firing distance to the boss. Lower means the learned aiming/navigation is better.
- `last500_oob_rate`: fraction of shots aimed out of bounds. Lower is better.
- `last_policy_std` and `last_entropy`: PPO exploration health. These are not the headline result, but they help diagnose collapse.
- `last500_comm_rate`: effective communication usage. For comm-loss runs this should roughly equal `1 - loss_prob`; for comm-gate runs it means "rounds where all agents chose to communicate."

## Recommended Final Result Framing

Use cautious wording:

```text
We evaluate two communication interventions in the Battleship SeqComm setting.
First, we replace critic-only ordering with a learned world-model ordering
signal and then ablate communication availability at 10%, 20%, and 30% loss.
Second, we add a learned communication gate, charging a penalty when an agent
chooses to communicate. We report final-500-episode performance, hit quality,
timeout rate, and effective communication rate.
```

Then state the current takeaways:

```text
The current single-seed package shows measurable degradation at 30% message loss
and shows that the learned communication gate can reduce communication usage
while preserving or improving win rate. Because the underlying baseline remains
noisy, these should be treated as preliminary but promising comparative results.
```

## Next Runs Needed Before Calling This Final

1. Add the missing normal world-model baseline:

```bash
RUN_ROOT=for_will/$(date +%Y%m%d_%H%M%S)_wm_loss00 \
SINGLE_CONFIG="wm_loss00|16|0.0003|0.003|0.15" \
EXP=exp1 EPISODES=1000 SEEDS="0" \
bash battleship/run_battleship_grid.sh
```

2. Repeat all conditions with at least `SEEDS="0 1 2"`:

```bash
EXP=exp1 EPISODES=1000 SEEDS="0 1 2" bash battleship/run_battleship_grid.sh
EXP=exp2 EPISODES=1000 SEEDS="0 1 2" bash battleship/run_battleship_grid.sh
```

3. If we want the communication-loss fallback to be literally random ordering,
change the no-message fallback in `robot_sim/agent_action.cc`. Right now it uses
a deterministic no-message ordering to avoid deadlock.
