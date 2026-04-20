# multi-lvl-comms

C++ + Python implementation of **SeqComm** (NeurIPS 2024): multi-agent coordination
via sequential communication. Agents negotiate a priority order each timestep based
on estimated future return, then cascade actions from highest to lowest — each agent
sees what the agents above it chose before making its own decision.

---

## Repository layout

```
multi-lvl-comms/
├── CMakeLists.txt              root cmake — delegates to robot_sim/
├── requirements.txt            Python deps (torch, numpy)
│
├── training/
│   ├── train.py                Python-only training (run with: python -m training.train)
│   ├── train_from_cpp.py       Python updater for the C++ ↔ Python training loop
│   └── train_world_model.py    PyTorch modules, MAPPO losses, GAE, save_weights()
│
├── execution/
│   └── gaussian_field_env.py   Python mirror of the C++ environment (used during training)
│
├── weights/                    TorchScript .pt files written by train.py (gitignored if large)
│
├── robot_sim/
│   ├── agent_action.hh/.cc     Agent coroutine tasks + NeuralModels/Environment interfaces
│   ├── gaussian_field_env.hh/.cc  First test environment (2D Gaussian coverage)
│   ├── seqcomm_sim.cc          Single-episode demo — random stub models (no libtorch needed)
│   ├── seqcomm_sim_trained.cc  Multi-episode C++ training loop — loads/reloads weights/
│   ├── libtorch_models.hh      LibTorchNeuralModels: NeuralModels impl backed by .pt files
│   ├── trajectory_io.hh        Flat binary serializer for std::vector<transition>
│   ├── netsim.hh               Async channel<T>/port<T> message passing
│   ├── pancy_msgs.hh           SeqComm message types + std::formatter specialisations
│   ├── random_source.hh        Seeded RNG with uniform/normal/exponential/hex helpers
│   └── utils.hh                randomly_seeded<> + durational concept
│
└── cotamer/                    C++23 coroutine framework (task, event, after, loop)
```

---

## Build and run

### Stub simulation (no libtorch needed)

```bash
cmake -B build
cmake --build build --target seqcomm-sim
./build/robot_sim/seqcomm-sim
```

Requires GCC 13+ and CMake 3.16+. No external libraries.

```
SeqComm: 4 agents, T=10 H=3 F=4 obs_dim=27
...agent 0 t=0 N_upper=0 N_lower=3
...agent 3 t=0 N_upper=3 N_lower=0
Episode done — transitions: 40  total_reward: 19.380
```

`RandomNeuralModels` returns random tensors of the right shapes so the full coroutine
task graph runs before any real learning is wired in.

### Training with the C++ ↔ Python loop (recommended)

**Step 1 — install Python deps:**
```bash
pip install -r requirements.txt
```

**Step 2 — bootstrap initial weights:**
```bash
python -m training.train --episodes 100 --save-weights weights/
# writes weights/encoder.pt, attn_a.pt, attn_w.pt,
#         world_model.pt, policy.pt, critic.pt, config.json
```

**Step 3 — build the C++ training harness:**

```bash
cmake -B build -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake --build build --target seqcomm-sim-trained
```

If libtorch is not found, CMake skips `seqcomm-sim-trained` and prints a hint —
the stub `seqcomm-sim` still builds normally.

> **macOS only:** libtorch's prebuilt binaries require the LLVM OpenMP runtime.
> If you see `Library not loaded: /opt/llvm-openmp/lib/libomp.dylib` at runtime, install it:
> ```bash
> brew install libomp
> ```

**Step 4 — run the two-process training loop:**
```bash
# terminal 1 — C++ sim collects trajectories with the real SeqComm protocol
./build/robot_sim/seqcomm-sim-trained weights/ 2000

# terminal 2 — Python runs MAPPO updates and saves new weights each episode
python -m training.train_from_cpp weights/
```

```
# C++ output:
SeqComm C++ training loop: 4 agents  T=200  H=5  F=4  obs_dim=27  episodes=2000
ep    0  R=   18.34  avg=   18.34  transitions=800  waiting for Python…
  → weights reloaded
ep    1  R=   22.10  avg=   20.22  transitions=800  waiting for Python…
…

# Python output:
train_from_cpp: watching weights/  obs_dim=27  embed_dim=64  agents=4
ep    0 | R=  18.34  avg=  18.34 | wm=0.0231  v=0.1842  π=0.0034  (1.2s)
ep   10 | R=  31.50  avg=  24.11 | wm=0.0198  v=0.1531  π=0.0021  (1.1s)
…
```

### Python-only training (no libtorch required)

```bash
python -m training.train --env intersection --episodes 2000 --save-weights weights/ \
    --log-dir logs/ --seed 0
```

Uses a Python rollout instead of the C++ sim — faster to iterate on architecture
changes, but doesn't exercise the real concurrent SeqComm protocol.

Key flags:
- `--env`          — `gaussian`, `coverage`, or `intersection` (default: gaussian)
- `--mode`         — ablation variant (default: `seqcomm`); see table below
- `--log-dir DIR`  — write a JSONL log to `DIR/<env>_<mode>[_<comm>]_seed<N>.jsonl`
- `--seed N`       — fix torch + Python RNG for reproducible runs
- `--save-every N` — checkpoint weights every N episodes in addition to the final save

Communication stressor flags (all default to perfect/lossless):
- `--comm-delay N`  — message arrives N steps late; first N steps see zeros
- `--comm-drop P`   — each message is independently dropped with probability P
- `--comm-noise S`  — additive Gaussian noise with std S on every received tensor
- `--comm-bits B`   — quantise messages to 2^B uniform levels (0 = full float32)

Stressors can be combined. The comm tag is appended to the log filename only when at
least one stressor is active, e.g. `intersection_seqcomm_delay2_drop0.3_seed0.jsonl`.

Example — seqcomm under 30% packet drop:
```bash
python3 -m training.train \
  --env intersection --mode seqcomm --seed 0 \
  --episodes 2000 --log-dir logs/ \
  --comm-drop 0.3
```

### World-model accuracy stressors

These flags degrade the *intention computation* only — the env step and training
gradients always use clean observations. They simulate the sim-to-real gap that
appears when a world model trained in simulation is deployed on real hardware.

- `--obs-noise STD`  — additive Gaussian noise on each agent's own observation
                       before `compute_intention`. The world model was never trained
                       on noisy inputs, so intention estimates become less reliable.
- `--wm-H N`         — rollout horizon used in `compute_intention` at inference
                       (training default: 5). Lower → shallower rollouts → noisier
                       intention values and less accurate priority ordering.
- `--wm-F N`         — random orderings sampled per intention estimate (training
                       default: 4). Lower → higher variance in estimates.

These flags are recorded in the `_meta` header and reflected in the log filename,
e.g. `intersection_seqcomm_obsnoise0.1_H2_F1_seed0.jsonl`.

Example — degraded rollout fidelity (H=1, F=1) with sensor noise:
```bash
python3 -m training.train \
  --env intersection --mode seqcomm --seed 0 \
  --episodes 2000 --log-dir logs/ \
  --obs-noise 0.1 --wm-H 1 --wm-F 1
```

### Ablation modes

| `--mode`              | Intention negotiation | Action sharing in launch | Ordering        |
|-----------------------|-----------------------|--------------------------|-----------------|
| `seqcomm`             | ✓ world-model rollout | ✓ upper → lower          | by intention    |
| `mappo`               | ✗                     | ✗                        | fixed [0,1,2,3] |
| `seqcomm_random`      | ✗                     | ✓                        | random shuffle  |
| `seqcomm_no_action`   | ✓                     | ✗                        | by intention    |
| `seqcomm_fixed`       | ✗                     | ✓                        | fixed [0,1,2,3] |

To run all five variants for three seeds each (15 total runs):

```bash
for mode in seqcomm mappo seqcomm_random seqcomm_no_action seqcomm_fixed; do
  for seed in 0 1 2; do
    python3 -m training.train \
      --env intersection --mode $mode --seed $seed \
      --episodes 2000 --log-dir logs/
  done
done
```

### Episode logging

When `--log-dir` is set, each episode appends one JSON record to the log file:

```json
{
  "episode": 42,
  "total_reward": 37.42,
  "success": true,
  "steps_to_completion": 87,
  "deadlock": false,
  "n_collisions": 2,
  "n_goals_reached": 4,
  "n_msgs_dropped": 7,
  "order_entropy": 1.23,
  "mean_intention_spread": 0.45,
  "first_mover_counts": [12, 8, 15, 65],
  "world_model_loss": 0.023400,
  "value_loss": 0.145000,
  "policy_loss": 0.012300
}
```

- `order_entropy` — Shannon entropy of the first-mover distribution across episode steps.
  High (≈ log 4 ≈ 1.39) means all agents take turns leading; low means one agent dominates.
- `mean_intention_spread` — average per-step std of agent intention values.
  High means agents clearly differentiate their priorities each step.
- `first_mover_counts` — how many steps each agent ranked first in the priority ordering.

The first line of each log file is a `{"_meta": {...}}` record with all hyperparameters
and the run timestamp, so the file is fully self-describing.

---

## Algorithm

Each timestep runs two phases across all agents concurrently as cotamer coroutines.

### Negotiation phase

1. **Encode** — agent encodes its observation: `h_i = e(o_i)` via `NeuralModels::encode`
2. **Share hidden states** — broadcast `hidden_state_msg{id, h_i}` to the full clique, collect one from each neighbour
3. **Compute intention** (Algorithm 5) — for each of `F` sampled random orderings of the clique, simulate `H` world-model steps:
   - Act through the ordering top-down, each agent's action conditioned on the actions above it via `AM_a`
   - Step the world model: `(o′_all, r) = M(AM_w(enc_obs_all, actions_all))`
   - Bootstrap with critic `V` at the horizon
   - Return mean discounted return over `F` orderings as agent `i`'s **intention** `I_i`
4. **Share intentions** — broadcast `intention_msg{id, I_i}`, partition neighbours into `N_upper` (higher `I`) and `N_lower` (lower `I`)

### Launching phase

1. **Wait** for `upper_action_msg` from every agent in `N_upper`
2. **Sample action** — `a_i ~ π(AM_a(h_i, a_upper))`
3. **Send down** — forward `upper_action_msg{id, a_i}` to every agent in `N_lower`
4. **Synchronise** — the agent with no lower neighbours broadcasts `execute_signal`; all others wait for it
5. **Step** — call `env.submit_action(id, a_i)`; last agent to submit triggers the environment; all agents block on `env.get_result(id)` until results are ready
6. **Record** — append `transition{obs, action, upper_actions, next_obs, reward, value, log_prob, log_prob_old}` to the shared trajectory buffer

---

## How training works

There are two training paths: a Python-only path for quick iteration, and the full
C++ ↔ Python loop that uses the real concurrent SeqComm protocol for trajectory collection.

### C++ ↔ Python training loop (primary path)

C++ collects trajectories using the real concurrent protocol (coroutines, async message
passing); Python does the gradient updates. They synchronise via three files in `weights/`:

```
weights/traj.bin      — flat binary trajectory written by C++ each episode
weights/traj.ready    — sentinel: C++ touches this when traj.bin is complete
weights/weights.ready — sentinel: Python touches this when new .pt files are saved
```

**Run in two terminals:**

```bash
# terminal 1 — C++ sim (blocks after each episode waiting for Python)
./build/robot_sim/seqcomm-sim-trained weights/ 2000

# terminal 2 — Python updater (blocks between episodes waiting for C++)
python -m training.train_from_cpp weights/
```

Per episode the C++ sim:
1. Runs one full SeqComm episode with `LibtorchNeuralModels`
2. Serialises `std::vector<transition>` to `weights/traj.bin` via `trajectory_io.hh`
3. Touches `weights/traj.ready`
4. Polls for `weights/weights.ready`, then calls `LibtorchNeuralModels::reload()`

Per episode the Python updater (`training/train_from_cpp.py`):
1. Polls for `weights/traj.ready`
2. Reads `traj.bin` with `numpy.frombuffer`, reassembles all-agent tensors by timestep
3. Runs `compute_gae` + three MAPPO losses
4. Saves six updated `.pt` files via `save_weights()`
5. Touches `weights/weights.ready`

**Bootstrap:** run `train.py` once first to create initial weights before starting the loop.

```bash
python -m training.train --episodes 100 --save-weights weights/
```

### Trajectory binary format (`trajectory_io.hh`)

```
header (16 bytes):  int32 n_agents, obs_dim, action_dim, n_transitions

per transition:
  int32                           agent_id, timestep, n_upper
  float32[obs_dim]                obs
  float32[action_dim]             action
  float32[n_agents * action_dim]  upper_actions  (first n_upper slots filled)
  float32[obs_dim]                next_obs
  float32                         reward, value, log_prob, log_prob_old
```

### C++ side — trajectory collection

`seqcomm_sim_trained.cc` is the multi-episode training harness.
`seqcomm_sim.cc` is a single-episode demo with random stub models (no libtorch needed).

The `NeuralModels` interface is the only seam between the two sides:

```cpp
struct NeuralModels {
    virtual std::vector<float> encode(std::span<const float> obs) = 0;
    virtual std::vector<float> attention_a(std::span<const float> h,
                                           const std::vector<std::vector<float>>& msgs) = 0;
    virtual std::vector<float> attention_w(const std::vector<std::vector<float>>& enc,
                                           const std::vector<std::vector<float>>& acts) = 0;
    virtual std::pair<std::vector<float>, float>
                               policy_sample(std::span<const float> ctx) = 0;
    virtual float              policy_log_prob_old(std::span<const float> ctx,
                                                   std::span<const float> act) = 0;
    virtual float              critic(std::span<const float> ctx) = 0;
    virtual std::pair<std::vector<float>, float>
                               world_model(std::span<const float> ctx) = 0;
};
```

### Python side — gradient updates

`train_world_model.py` implements the three MAPPO losses from the paper. Each loss
trains a distinct subset of parameters:

| Function | Paper eq. | Trains |
|---|---|---|
| `world_model_loss` | (4) | encoder + AM_w + world model M |
| `value_loss` | (2) | encoder + AM_a + critic V |
| `ppo_loss` | (3) | encoder + AM_a + policy π |

`train_step` runs world model first (independent gradients), then value + policy jointly:

```python
# World model — eq (4)
L_w = mean || (o′, r) − M(AM_w(e(o), a)) ||²

# Value — eq (2)
L_v = mean || V(AM_a(e(o_i), a_upper)) − R̂ ||²

# Policy — eq (3), PPO-clip + GAE advantages
L_π = −mean min(ρ·A,  clip(ρ, 1±ε)·A)
```

`compute_gae` turns the raw reward/value sequence from a trajectory into
`(advantages, returns)` via generalised advantage estimation (γ=0.99, λ=0.95).

### Training loop (Python-only path, no libtorch required)

```python
env = GaussianFieldEnv()          # execution/gaussian_field_env.py

for episode in range(N_EPISODES):
    obs = env.reset()
    trajectory = []

    for t in range(T):
        # ── negotiation ──────────────────────────────────
        h = [encoder(o) for o in obs]
        intentions = [compute_intention(h, i, world_model, ...) for i in range(N)]
        order = sorted(range(N), key=lambda i: -intentions[i])

        # ── launching (cascade actions top-down) ─────────
        a_upper = {}
        actions, log_p, values = [], [], []
        for i in order:
            ctx = attn_a(h[i], [a_upper[j] for j in order if order.index(j) < order.index(i)])
            a = policy.sample(ctx)
            actions.append(a);  a_upper[i] = a
            log_p.append(policy.log_prob(ctx, a))
            values.append(critic(ctx))

        next_obs, reward, _ = env.step(actions)
        trajectory.append((obs, actions, ..., next_obs, reward, values, log_p))
        obs = next_obs

    advantages, returns = compute_gae([t.reward for t in trajectory],
                                      [t.value  for t in trajectory])
    losses = train_step(encoder, attn_a, attn_w,
                        world_model_net, policy, critic,
                        *tensorify(trajectory, advantages, returns),
                        opt_world, opt_policy)
```

`gaussian_field_env.py` is a line-for-line Python mirror of the C++ environment so
weights trained here plug directly into the C++ simulation without any adaptation.

---

## Environment 1: Gaussian Field Coverage

**Setup** — `G×G` grid (default 20×20), `K=3` Gaussian peaks moving with constant
velocity and bouncing off walls. `N` agents navigate the grid.

**Observation** `o_i` (length 27 with default `window_half=2`):
```
[row/G,  col/G,  field(r−2,c−2), ..., field(r+2,c+2)]
```
Normalised position plus a 5×5 local field window, zero-padded at borders.

**Action** — integer in `{stay, up, down, left, right}`.

**Reward** (joint, identical for all agents):
```
r = Σ_i F(p_i)  −  λ · |{(i,j) : p_i = p_j,  i < j}|
```
Sum of field values under each agent minus a penalty for any two agents sharing a cell.

**What the world model must learn** — Gaussian trajectories are deterministic (linear
plus wall bouncing) so peak positions are predictable from `H` steps of history. An
agent that correctly forecasts where peaks will be in `H` steps gets a higher intention
estimate, earns a higher priority, and stakes out high-value cells before lower-ranked
agents are committed.

**What SeqComm provides** — agents can't see the full field, only their local window.
Upper agents share their hidden state (encoded local view) before lower agents act,
giving lower agents indirect information about field conditions elsewhere in the grid.

---

## Environment 2 (roadmap): Archipelago Survival

*"Keeping things afloat"* — the team must collectively stay alive.
Coordination is existential rather than purely reward-maximising.

### Concept

The grid has `K` island cells where `K < N` (not enough islands for everyone).
Open water slowly drains each agent's survival level `s_i ∈ [0, 1]`.
An island replenishes the agent that occupies it — but only if that agent is there alone.
Two agents on the same island both take the penalty instead.

```
Each timestep, for each agent i:

  unoccupied island  →  s_i ← min(1,  s_i + δ_replenish)
  contested island   →  s_i ← max(0,  s_i − δ_penalty)
  open water         →  s_i ← max(0,  s_i − δ_drain)

Joint reward:
  r = mean_i(s_i)  −  λ · collisions
```

Agents with `s_i = 0` are lost and contribute nothing to the team score for the rest
of the episode. The team's goal is to maximise the survival integral over T timesteps.

### Why SeqComm handles this naturally

With `K < N` islands the assignment problem is strictly competitive within the team —
two agents on one island is always a net loss. SeqComm resolves it without any explicit
"I'm going here" protocol:

- **High-intention agents** (better world-model predictions of where islands will be
  in `H` steps) commit to a cell first.
- **Low-intention agents** receive those choices as `upper_action_msg` before they
  decide, and route to unclaimed islands rather than competing.
- The cascade naturally solves the assignment in priority order, one level at a time.

An agent's urgency (`s_i` close to 0) feeds through the encoder into its hidden state,
so neighbours see it during negotiation and can route around an agent that is critically
low and must reach the nearest island at all costs.

### Observation extension

```
[row/G,  col/G,  field_window,  s_i,  nearest_island_dist/G]
```

Adding `s_i` and approximate island proximity gives the world model the signal it needs
to predict multi-step survival trajectories.

### Island dynamics

Islands shift slowly each step with the same bouncing-velocity physics as the Gaussian
peaks, making them predictable (good for the world model) but requiring ongoing
tracking. This is a direct structural extension of Environment 1 — the learned world
model architecture transfers without modification.

### Generalisation path

| Variation | What it stresses |
|---|---|
| Reduce K | Harder assignment; more agents forced into water |
| Heterogeneous drain rates | Priority ordering must adapt to urgency, not just prediction quality |
| Hidden `s_j` of others | Agents must infer neighbours' urgency from negotiation hidden states |
| Island capacity > 1 | Softens hard exclusion; partial sharing with diminishing returns |
| Islands appear/disappear | World model must predict availability, not just position |

---

## Interfaces — adding a new environment

Subclass `Environment` in `agent_action.hh`. Two methods required:

```cpp
struct MyEnv : Environment {
    // Called by each agent. Last agent to call triggers the step.
    void submit_action(int agent_id, std::span<const float> action) override;

    // Coroutine: suspends until all N agents have submitted, then returns.
    cot::task<std::pair<std::vector<float>, float>> get_result(int agent_id) override;
};
```

Pass an instance to `Agent` in `seqcomm_sim.cc` — everything else (negotiation,
launching, trajectory recording) is environment-agnostic.
