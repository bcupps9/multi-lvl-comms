# multi-lvl-comms


check the deps that claude has put in for the training script

C++ + Python implementation of **SeqComm** (NeurIPS 2024): multi-agent coordination
via sequential communication. Agents negotiate a priority order each timestep based
on estimated future return, then cascade actions from highest to lowest — each agent
sees what the agents above it chose before making its own decision.

---

## Repository layout

```
multi-lvl-comms/
├── CMakeLists.txt              root cmake — delegates to robot_sim/
├── train_world_model.py        PyTorch MAPPO losses + GAE (equations 2, 3, 4)
├── gaussian_field_env.py       Python mirror of the C++ environment
│
├── robot_sim/
│   ├── agent_action.hh/.cc     Agent coroutine tasks + NeuralModels/Environment interfaces
│   ├── gaussian_field_env.hh/.cc  First test environment (2D Gaussian coverage)
│   ├── seqcomm_sim.cc          Simulation harness with random stub neural models
│   ├── netsim.hh               Async channel<T>/port<T> message passing
│   ├── pancy_msgs.hh           SeqComm message types + std::formatter specialisations
│   ├── random_source.hh        Seeded RNG with uniform/normal/exponential/hex helpers
│   ├── utils.hh                randomly_seeded<> + durational concept (used by random_source)
│   └── print                   <print> shim for GCC 13 (std::format backend)
│
└── cotamer/                    C++23 coroutine framework (task, event, after, loop)
```

---

## Build and run

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

The stub models (`RandomNeuralModels` in `seqcomm_sim.cc`) return random tensors of the
right shapes so the full coroutine task graph runs before any real learning is wired in.

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

### C++ side — trajectory collection

`seqcomm_sim.cc` is the harness. Replace `RandomNeuralModels` with thin wrappers that
call into libtorch (or load weights exported from the Python trainer), run episodes,
and serialise the trajectory buffer to disk or pass it to Python via a socket.

The `NeuralModels` interface is the only seam:

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
env = GaussianFieldEnv()          # gaussian_field_env.py

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
