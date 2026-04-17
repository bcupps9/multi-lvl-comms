# multi-lvl-comms

Implementation of **SeqComm** (Sequential Communication, NeurIPS 2024) for multi-agent reinforcement learning. Agents communicate in a dynamically-ranked order — those with higher estimated future return act and speak first, giving lower-ranked agents better context for their own decisions.

The backbone runs in C++ using cotamer coroutines for agent concurrency. Neural network weights are trained separately in Python using PyTorch.

---

## What exists

### C++ simulation (`robot_sim/`)

| File | Purpose |
|------|---------|
| `agent_action.hh/.cc` | Core Agent struct — negotiation phase, launching phase, intention rollout |
| `gaussian_field_env.hh/.cc` | First environment: cooperative Gaussian field coverage on a 2D grid |
| `netsim.hh` | Async message passing: `channel<T>` (sender) and `port<T>` (receiver) with simulated latency/loss |
| `pancy_msgs.hh` | Message variant types: `hidden_state_msg`, `intention_msg`, `upper_action_msg`, `execute_signal` |
| `random_source.hh` | Seeded RNG wrapper (uniform, normal, exponential, coin flip) |
| `seqcomm_sim.cc` | Simulation harness with random stub neural models — the runnable entry point |
| `print` | `<print>` compatibility shim for GCC 13 (implements via `<format>`) |

### Python training (`/`)

| File | Purpose |
|------|---------|
| `train_world_model.py` | All three MAPPO loss functions (eqs. 2, 3, 4), GAE, and `train_step` |
| `gaussian_field_env.py` | Python mirror of the C++ environment — identical dynamics, used during training |

### cotamer coroutine library (`cotamer/`)

Self-contained C++23 async/await framework. Provides `task<T>`, `event`, `after(duration)`, and `loop()` — the event loop that drives all agent tasks concurrently. Not modified by this project.

---

## Build

**Requirements:** GCC 13+, CMake 3.16+, C++23.

```bash
# From repo root
cmake -B build
cmake --build build --target seqcomm-sim
./build/robot_sim/seqcomm-sim
```

Expected output:
```
SeqComm: 4 agents, T=10 H=3 F=4 obs_dim=27
0.000: agent 2 t=0 N_upper=3 N_lower=0
...
Episode done — transitions: 40  total_reward: 2.644
```

The stub neural models (`RandomNeuralModels` in `seqcomm_sim.cc`) return random tensors of the correct shape — the full cotamer task flow runs end-to-end before any real learning.

---

## Algorithm overview

Each timestep has two phases run concurrently across all agents:

### Negotiation phase
1. Each agent encodes its observation: `h_i = e(o_i)`
2. All agents broadcast `h_i` to their clique (fully-connected by default)
3. Each agent runs **Algorithm 5**: for `F` sampled random orderings of the clique, simulate `H` world-model steps — cascade actions top-down through each ordering, predict `(o', r)` via the world model, bootstrap with `V` at the horizon. Average return over `F` orderings is agent `i`'s **intention** `I_i`.
4. Agents broadcast `I_i`; each partitions its clique into `N_upper` (higher intention) and `N_lower` (lower intention).

### Launching phase (Algorithm 6)
1. Agent waits for `upper_action_msg` from every agent in `N_upper`.
2. Agent samples its action: `a_i ~ pi(AM_a(h_i, a_upper))`.
3. Agent sends `upper_action_msg` to every agent in `N_lower`.
4. The agent with the smallest intention (empty `N_lower`) broadcasts `execute_signal` to trigger the environment step.
5. Every agent calls `env.submit_action(id, a)` and then suspends until all actions are in (cotamer barrier), then receives `(next_obs, reward)`.

The key insight: **higher-intention agents choose first** and their choices become context for lower-ranked agents, reducing action-space uncertainty downstream.

---

## How training works

Training is split across the C++ simulation (trajectory collection) and Python (gradient updates).

### Trajectory collection (C++)

`seqcomm_sim.cc` runs one episode and fills a shared `std::vector<transition>`. Each `transition` records:

```
agent_id, timestep, obs, action, upper_actions,
next_obs, reward, value, log_prob, log_prob_old
```

In a real training loop, replace `RandomNeuralModels` with thin wrappers that call libtorch (or load weights from a shared file), run many episodes, and write the trajectory buffer to disk or pass it to Python.

### Gradient updates (Python)

`train_world_model.py` implements the three MAPPO losses from the paper:

| Loss | Equation | What it trains |
|------|----------|----------------|
| `world_model_loss` | (4) `L(theta_w) = mean ||(o',r) - M(AM_w(e(o),a))||^2` | World model + AM_w |
| `value_loss` | (2) `L(theta_v) = mean ||V(AM_a(e(o),a_upper)) - R_hat||^2` | Critic |
| `ppo_loss` | (3) `L(theta_pi) = -mean min(rho*A, clip(rho,eps)*A)` | Policy via PPO-clip |

`train_step` runs both optimizers in sequence: world model first (independent of policy), then value + policy jointly.

**Training flow for the first environment:**

```
for episode in range(N_EPISODES):
    obs = env.reset()                   # gaussian_field_env.py
    trajectory = []

    for t in range(T):
        # --- Negotiation (Python) ---
        h = encoder(obs)                # encode all agents' obs
        intentions = compute_intention(h, world_model, ...)
        order = rank_by_intention(intentions)

        # --- Launching (Python) ---
        actions, log_p, values = cascade_policy(h, order, policy)
        next_obs, reward, _ = env.step(actions)
        trajectory.append((obs, actions, upper_actions,
                           next_obs, reward, values, log_p))
        obs = next_obs

    advantages, returns = compute_gae(trajectory)
    losses = train_step(encoder, attn_a, attn_w,
                        world_model, policy, critic, ...)
```

The Python environment mirrors the C++ dynamics exactly, so weights trained in Python transfer directly to the C++ simulation.

---

## Environment: Gaussian Field Coverage

**World:** `G x G` grid (default 20x20) with `K` moving Gaussian peaks.

**Observation** for agent `i`:
```
[row/G, col/G,  field(r-w,c-w), ..., field(r+w,c+w)]
```
Normalized position + flattened `(2w+1)^2` field window centered on the agent (zero-padded at borders). Default `w=2` gives `obs_dim = 2 + 25 = 27`.

**Action:** integer in `{0=stay, 1=up, 2=down, 3=left, 4=right}`.

**Reward (joint, shared):**
```
r = sum_i F(p_i)  -  lambda * |{(i,j) : p_i == p_j, i < j}|
```
Sum of field values at each agent's position minus overlap penalty for co-occupancy. Agents are rewarded for covering high-value regions and penalized for duplicating coverage.

**Gaussian dynamics:** Each peak moves with constant velocity and bounces off walls — deterministic but non-trivial. This gives the world model a learnable prediction target: agents that correctly anticipate where peaks will be in H steps get higher intentions and act first.

**Why this tests SeqComm:** The field is only partially observable (local window). Upper-ranked agents stake out high-value cells based on their world-model predictions; lower-ranked agents see those choices and avoid wasted overlap.

---

## Environment roadmap: Archipelago Survival

*"Keeping things afloat"* — a cooperative survival task where coordination is existential rather than just reward-maximizing.

### Concept

The grid contains `K` island cells where `K < N` (not enough islands for everyone). Open water slowly drains each agent's survival level. Islands replenish it. Agents that drain completely are "lost at sea" and freeze, dragging the team's collective score down.

```
Each step, for each agent i:

  if on unoccupied island:     s_i += delta_replenish  (cap at 1.0)
  elif on contested island:    s_i -= delta_penalty     (both lose)
  else (open water):           s_i -= delta_drain       (floor at 0.0)

Joint reward:
  r = (1/N) * sum_i(s_i)  -  lambda * collisions
```

The team's score is average survival. A drowned agent (s_i = 0) contributes zero and cannot recover unless the episode ends.

### Why SeqComm handles this well

With `K < N` islands, two agents on the same island is always a loss. The team must solve an assignment problem every time islands shift — which agent claims which island? SeqComm provides a natural answer:

- **High-intention agents** (better predictors of where islands will be in H steps) claim islands first.
- **Low-intention agents** receive those choices as `upper_action_msg` and route to unclaimed islands rather than competing.
- The priority cascade resolves conflicts *before* the environment step, with no explicit "I'm going here" message needed — the action itself carries that information.

### Observation extension

```
[row/G, col/G,  local_field_window,  s_i,  nearest_island_dist/G]
```

Adding each agent's own survival level lets the world model capture urgency: an agent with low `s_i` should act more conservatively, and other agents seeing that in the negotiation round can plan around it.

### Island dynamics

Islands move with slow momentum each episode, bouncing off walls — the same bouncing physics used for Gaussians. This means:
- Short-horizon predictions are learnable (smooth movement)
- Agents that track island trajectories have a consistent edge in intention estimates
- The world model faces the same prediction structure as in Gaussian Field Coverage, making it a natural progression

### Episode termination

Fixed `T` timesteps, or early exit when all agents reach `s_i = 0`. The reported score is the mean survival integral over the episode, not just final survival.

### Generalization path

| Modification | What it tests |
|---|---|
| Reduce K (fewer islands) | Harder assignment; more agents forced onto water |
| Moving islands with momentum | World model must predict multi-step trajectories |
| Heterogeneous drain rates | Some agents are more urgent; others can wait — priority ordering must adapt |
| Hidden `s_j` of others | Agents infer urgency from negotiation hidden states, not direct observation |
| Island capacity > 1 | Softens the hard constraint; partial sharing with diminishing returns |

---

## Abstractions and generalizability

The C++ layer is deliberately decoupled from any specific environment or neural architecture:

- **`NeuralModels`** (abstract, `agent_action.hh`) — swap in any encoder/attention/policy/world-model without touching agent logic. `seqcomm_sim.cc` shows the stub; a libtorch version plugs in identically.
- **`Environment`** (abstract, `agent_action.hh`) — any class with `submit_action` + `get_result` is a valid environment. Adding Archipelago Survival is a new `.hh/.cc` pair; the agent task code is untouched.
- **`netsim::channel<T>` / `port<T>`** — configurable latency and packet loss. The same agent code runs under perfect comms or degraded network, with no changes to the algorithm.
- **Python parity** — `gaussian_field_env.py` is an exact mirror of the C++ environment, keeping training and simulation dynamics consistent across the language boundary.
