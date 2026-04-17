# Weekend Plan: Closing the Gap to the Paper

## The core problem

Right now training lives entirely in Python and the C++ simulation uses a random stub.
That means we get none of the advantages C++ was supposed to provide:

- **Async, concurrent agent execution** via cotamer coroutines is bypassed — Python
  runs agents sequentially in a single thread
- **Network effects** (latency, packet loss, reordering via `netsim.hh`) are never
  exercised during training, so learned policies have never seen comms noise
- **Trajectory fidelity** — the Python rollout approximates SeqComm; the C++ version
  *is* SeqComm

The goal is to flip the loop: C++ collects trajectories using the real concurrent
protocol, Python updates weights, C++ loads new weights and collects again.

---

## 1. C++ → Python training loop

Replace the Python rollout with a C++ → Python pipeline:

- After each episode, `seqcomm_sim_trained.cc` serialises the trajectory buffer
  (already a `std::vector<transition>`) to a shared file or a local socket
- Python reads the trajectory, runs `compute_gae` + `train_step`, saves updated
  weights back to `weights/`
- C++ reloads weights (`torch::jit::load`) and runs the next episode

This is the paper's actual setup. The C++ side gets all the protocol fidelity;
Python does only what it's good at (autograd).

Serialisation options (simplest first):
1. **Flat binary file** — write transitions as packed floats, read with `numpy.frombuffer`
2. **JSON/MessagePack** — human-readable, easy to inspect mid-run
3. **ZeroMQ / Unix socket** — removes the file roundtrip for faster iteration

---

## 2. Environment where the SeqComm advantage is visible

The Gaussian field is too forgiving — random agents still score reasonably because
peaks are large and slow. We need an environment where *ordering matters* and
*communication pays off measurably*.

### Candidate: Narrow-corridor rendezvous

```
┌──────────────────────────────┐
│  A   A       [door]    G  G  │
│              [door]          │
└──────────────────────────────┘
```

- Two agents must pass through a single-cell doorway to reach their goals
- If both try simultaneously they collide and both fail
- SeqComm resolves this: the higher-intention agent goes first, the lower one
  waits — pure ordering benefit, no need for a learned world model
- Baseline (simultaneous / random order) win rate ≈ 50%; SeqComm should approach 100%

Win condition is binary, so **win percentage** is a clean scalar metric that shows
the curve clearly.

### Alternative: Token-passing relay

```
S → A₀ → A₁ → A₂ → A₃ → G
```

Agents must pass a token down a chain. Each agent can only forward when its
upstream neighbour has already acted. SeqComm's cascade ordering naturally induces
the correct sequence; an unordered policy randomly orders the chain and fails.

Both environments are small enough that the optimal policy is known analytically,
so we can plot **regret** (gap to optimal) rather than raw reward.

---

## 3. Full observability logging

Every event that matters should be logged with a timestep tag so we can replay
or plot any slice after the fact.

### Per-step log line (structured, one JSON object per line)

```json
{
  "ep": 12, "t": 7,
  "intentions": [2.31, 1.87, 3.02, 0.94],
  "ordering": [2, 0, 1, 3],
  "neighborhoods": {
    "0": {"upper": [2], "lower": [1, 3]},
    "1": {"upper": [2, 0], "lower": [3]},
    ...
  },
  "messages": [
    {"from": 2, "to": 0, "type": "upper_action", "value": [0.73]},
    {"from": 0, "to": 1, "type": "upper_action", "value": [-0.12]},
    ...
  ],
  "actions": [1, 3, 0, 2],
  "reward": 4.21,
  "values": [3.8, 3.1, 4.9, 2.2]
}
```

Emit from the C++ harness — agents already record all of this in `transition` structs
and the message-passing infrastructure. It's plumbing, not new logic.

### Episode summary

```json
{
  "ep": 12,
  "total_reward": 312.5,
  "win": true,
  "ordering_entropy": 1.2,
  "mean_intention_spread": 0.88
}
```

`ordering_entropy` — how varied the orderings were across timesteps; near-zero means
one agent always goes first (degenerate), high means agents regularly swap priority.

`mean_intention_spread` — std-dev of intentions at each step averaged over the
episode; low spread means the ordering is essentially random, high means the world
model is confidently differentiating agents.

---

## 4. Visualisation

### Training curves (matplotlib / wandb)

- Episode reward (rolling mean ± std)
- Win percentage (for binary-outcome envs)
- Per-loss curves: `wm`, `v`, `π` on the same plot
- Ordering entropy over training — should rise then stabilise as the model learns
  who to prioritise

### Per-episode replay (terminal or browser)

A small Python script that reads the JSONL log and renders each timestep:

```
t=7  ordering: [2 → 0 → 1 → 3]

  grid:
  . . . . .
  . A₂. . .    A₂ (priority 0)  intention=3.02
  . . A₀. .    A₀ (priority 1)  intention=2.31
  . . . A₁.    A₁ (priority 2)  intention=1.87
  . . . . A₃   A₃ (priority 3)  intention=0.94

  messages this step:
    2→0  upper_action [0.73]
    0→1  upper_action [-0.12]
    0→3  upper_action [-0.12]
    1→3  upper_action [0.41]

  reward: 4.21
```

### Win-rate dashboard

For binary-outcome environments: rolling win rate vs episode, with a horizontal
line at the random-policy baseline. The clearest possible picture of whether
SeqComm is actually helping.

---

## 5. Priority checklist

| # | Task | Needed for |
|---|------|------------|
| 1 | Serialise C++ trajectory → Python (flat binary) | Real training loop |
| 2 | Python training loop that reads trajectory file | Real training loop |
| 3 | Narrow-corridor or token-relay environment (C++ + Python mirror) | Measurable advantage |
| 4 | Structured JSONL logging from C++ harness | All visualisation |
| 5 | Episode-summary log + win-rate metric | Dashboard |
| 6 | Training-curve plots (reward, win %, losses) | Progress visibility |
| 7 | Per-episode terminal replay from JSONL | Debugging ordering behaviour |
