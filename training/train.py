"""
SeqComm MAPPO training.

Supports two environments:
  gaussian  — GaussianFieldEnv (default, same as before)
  coverage  — IslandCoverageEnv (assignment/coordination stress test)

Run:
    python training/train.py
    python training/train.py --env coverage
    python training/train.py --env coverage --save-weights weights/ --episodes 2000
    python training/train.py --episodes 5     # quick smoke test

Weight export (--save-weights):
    Saves six TorchScript .pt files + config.json to the specified directory.
    The C++ sim loads these when called with --weights <dir>.
"""

import argparse
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
import torch
import torch.optim as optim

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from execution.gaussian_field_env import GaussianFieldEnv, GaussianFieldConfig
from execution.island_coverage_env import IslandCoverageEnv, IslandCoverageConfig
from execution.intersection_env import IntersectionCrossingEnv, IntersectionCrossingConfig
from training.train_world_model import (
    ObservationEncoder,
    AttentionModule,
    WorldModel,
    Policy,
    Critic,
    ScriptablePolicy,
    compute_intention,
    compute_gae,
    world_model_loss,
    value_loss,
    ppo_loss,
    save_weights,
)

# ── Fixed hyperparameters (env-independent) ────────────────────────────────────

N_AGENTS    = 4
EMBED_DIM   = 64
ACTION_DIM  = 1     # continuous output, rounded to discrete env action
H           = 5     # world-model rollout horizon
F           = 4     # random orderings per intention estimate
EPISODE_LEN = 200  # overridden by --episode-len at runtime
N_EPISODES  = 2000
GAMMA       = 0.99
LAM         = 0.95
CLIP_EPS    = 0.2
LR_WORLD    = 3e-4
LR_POLICY   = 3e-4
LOG_EVERY   = 10

# ── Ablation modes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModeConfig:
    compute_intention: bool  # run world-model rollout to score agents (Algorithm 5)
    share_actions: bool      # pass actual upper-level actions in launching phase
    ordering: str            # "intention" | "random" | "fixed"

# Each variant isolates one design choice from the full SeqComm algorithm:
#   seqcomm          — full paper method (both phases active)
#   mappo            — plain MAPPO: no communication, fixed order, no shared actions
#   seqcomm_random   — share actions but pick order randomly (no intention negotiation)
#   seqcomm_no_action— negotiate order via intentions but don't share actual actions
#   seqcomm_fixed    — share actions but always use agent-index order (no negotiation)
MODES: dict[str, ModeConfig] = {
    "seqcomm":           ModeConfig(compute_intention=True,  share_actions=True,  ordering="intention"),
    "mappo":             ModeConfig(compute_intention=False, share_actions=False, ordering="fixed"),
    "seqcomm_random":    ModeConfig(compute_intention=False, share_actions=True,  ordering="random"),
    "seqcomm_no_action": ModeConfig(compute_intention=True,  share_actions=False, ordering="intention"),
    "seqcomm_fixed":     ModeConfig(compute_intention=False, share_actions=True,  ordering="fixed"),
}


# ── Communication stressors ────────────────────────────────────────────────────

@dataclass
class CommConfig:
    """
    Controls imperfect communication between agents.

    All four stressors can be combined. They are applied in order:
      delay → drop → noise → bandwidth

    delay (int):
        Each message arrives N timesteps late. During the first N steps of
        an episode the receiver gets zeros (no prior message exists yet).
        Models real network latency or slow sensing pipelines.

    drop_prob (float in [0, 1]):
        Each message is independently lost with this probability; the
        receiver gets a zero tensor instead. Models unreliable wireless links.

    noise_std (float ≥ 0):
        Additive Gaussian noise with this std is added to every received
        tensor. Models sensor noise or lossy compression artefacts.

    bandwidth_bits (int ≥ 0):
        Quantise each message to 2^bits uniform levels (0 = lossless).
        Models limited channel capacity: 4 bits ≈ 16 discrete levels,
        8 bits ≈ 256 levels, 0 = full float32 precision.
    """
    delay:           int   = 0
    drop_prob:       float = 0.0
    noise_std:       float = 0.0
    bandwidth_bits:  int   = 0

    def is_perfect(self) -> bool:
        return (self.delay == 0 and self.drop_prob == 0.0
                and self.noise_std == 0.0 and self.bandwidth_bits == 0)

    def tag(self) -> str:
        """Short identifier for filenames; empty string when no stressor is active."""
        if self.is_perfect():
            return ""
        parts = []
        if self.delay          > 0:   parts.append(f"delay{self.delay}")
        if self.drop_prob      > 0.0: parts.append(f"drop{self.drop_prob}")
        if self.noise_std      > 0.0: parts.append(f"noise{self.noise_std}")
        if self.bandwidth_bits > 0:   parts.append(f"bits{self.bandwidth_bits}")
        return "_".join(parts)


class CommChannel:
    """
    Applies CommConfig stressors to any tensor message.

    One instance is shared for the whole run. Delay buffers are keyed by
    (sender_id, receiver_id) so each directed channel has its own history.
    Call reset() at the start of every episode to clear those buffers.

    Usage
    -----
    channel = CommChannel(comm_cfg)
    # inside run_episode:
    channel.reset()
    received = channel.transmit(tensor, sender=j, receiver=i)
    """

    def __init__(self, cfg: CommConfig):
        self.cfg = cfg
        # (sender, receiver) → deque of past messages, length = cfg.delay
        self._buffers: dict[tuple[int, int], deque] = {}
        self.n_dropped: int = 0

    def reset(self) -> None:
        self._buffers.clear()
        self.n_dropped = 0

    def transmit(
        self,
        msg: torch.Tensor,
        sender: int,
        receiver: int,
    ) -> torch.Tensor:
        """Return what receiver actually gets when sender transmits msg."""

        # 1. Delay — retrieve the message sent `delay` steps ago.
        #    Buffer holds the last `delay` messages; oldest is what we return.
        #    Starts filled with zeros so early steps see "nothing sent yet".
        if self.cfg.delay > 0:
            key = (sender, receiver)
            if key not in self._buffers:
                self._buffers[key] = deque(
                    [torch.zeros_like(msg)] * self.cfg.delay
                )
            buf = self._buffers[key]
            received = buf.popleft()   # message from `delay` steps ago
            buf.append(msg.clone())    # enqueue this step's message
        else:
            received = msg.clone()

        # 2. Drop — packet lost entirely; receiver gets zeros.
        if self.cfg.drop_prob > 0.0 and random.random() < self.cfg.drop_prob:
            self.n_dropped += 1
            return torch.zeros_like(received)

        # 3. Noise — additive Gaussian channel noise.
        if self.cfg.noise_std > 0.0:
            received = received + torch.randn_like(received) * self.cfg.noise_std

        # 4. Bandwidth — uniform quantisation to 2^bits levels.
        #    lo==hi means the tensor is constant; skip to avoid divide-by-zero.
        if self.cfg.bandwidth_bits > 0:
            levels = 2 ** self.cfg.bandwidth_bits
            lo, hi = received.min(), received.max()
            if hi > lo:
                norm     = (received - lo) / (hi - lo)
                quantised = (norm * (levels - 1)).round() / (levels - 1)
                received  = quantised * (hi - lo) + lo

        return received


# ── Logging ────────────────────────────────────────────────────────────────────

class EpisodeLogger:
    """
    Writes one JSON object per line (JSONL) to a log file.

    Each line captures everything needed to reproduce the paper's
    Figure 4-style learning curves plus robotics-specific metrics:
      - total_reward, success, steps_to_completion, deadlock
      - n_collisions (intersection env only)
      - order_entropy: Shannon entropy over who becomes first mover
        (high = ordering varies a lot; low = one agent dominates)
      - mean_intention_spread: per-step std of agent intention values,
        averaged over the episode (high = clear hierarchy each step)
      - first_mover_counts: how many steps each agent ranked first
      - training losses
    """

    def __init__(self, log_path: str, metadata: dict):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._f = open(log_path, "w")
        # Write metadata as the first line so the file is self-describing
        self._f.write(json.dumps({"_meta": metadata}) + "\n")
        self._f.flush()
        print(f"Logging to {log_path}")

    def log(self, episode: int, info: dict, losses: dict) -> None:
        ordering_history  = info["ordering_history"]   # list[list[int]]
        intention_history = info["intention_history"]  # list[list[float]]
        n_steps           = len(ordering_history)

        first_movers = [order[0] for order in ordering_history]
        n_agents = len(ordering_history[0]) if ordering_history else 4
        first_mover_counts = [first_movers.count(i) for i in range(n_agents)]

        # Use pre-computed entropy if available (avoids duplicate work with wandb path).
        order_entropy = info.get("_order_entropy") or (
            -sum(p * math.log(p) for p in [c / n_steps for c in first_mover_counts if c > 0])
            if any(c > 0 for c in first_mover_counts) else 0.0
        )

        # Per-step std of intention values → how differentiated agents are
        intention_stds = [
            float(torch.tensor(ivec).std().item()) if len(ivec) > 1 else 0.0
            for ivec in intention_history
        ]
        mean_intention_spread = sum(intention_stds) / len(intention_stds) if intention_stds else 0.0

        record = {
            "episode":               episode,
            "total_reward":          round(info["total_reward"], 4),
            "success":               info["success"],
            "steps_to_completion":   info["steps_to_completion"],
            "deadlock":              info["deadlock"],
            "n_collisions":          info["n_collisions"],
            "n_goals_reached":       info["n_goals_reached"],
            "coverage_rate":         round(info["coverage_rate"], 4),
            "n_overlaps":            info["n_overlaps"],
            "n_msgs_dropped":        info["n_msgs_dropped"],
            "order_entropy":         round(order_entropy, 4),
            "mean_intention_spread": round(mean_intention_spread, 4),
            "first_mover_counts":    first_mover_counts,
            "world_model_loss":      round(losses["world_model"], 6),
            "value_loss":            round(losses["value"], 6),
            "policy_loss":           round(losses["policy"], 6),
            "policy_entropy":        round(losses.get("entropy", 0.0), 6),
        }
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ── Environment factory ────────────────────────────────────────────────────────

def make_env(name: str, n_agents: int):
    """Return a configured environment and its obs_dim."""
    if name == "gaussian":
        cfg = GaussianFieldConfig(n_agents=n_agents)
        return GaussianFieldEnv(cfg)
    if name == "coverage":
        cfg = IslandCoverageConfig(n_agents=n_agents, n_islands=n_agents)
        return IslandCoverageEnv(cfg)
    if name == "intersection":
        cfg = IntersectionCrossingConfig(n_agents=n_agents)
        return IntersectionCrossingEnv(cfg)
    raise ValueError(f"Unknown env '{name}'. Choose 'gaussian', 'coverage', or 'intersection'.")


# ── Rollout ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    env,
    n_agents: int,
    obs_dim: int,
    encoder, attn_a, attn_w, world_model_net, policy, critic,
    mode_cfg: ModeConfig = MODES["seqcomm"],
    comm_channel: "CommChannel | None" = None,
    obs_noise_std: float = 0.0,
    H_infer: int = H,
    F_infer: int = F,
) -> tuple[list[dict], dict]:
    """
    Run one full episode.  Behaviour is controlled by mode_cfg:

      compute_intention  — whether to run world-model rollouts (Algorithm 5)
                           to score agents and determine priority order.
                           False → skip; intentions are all 0.0.
      share_actions      — whether upper-level agents' actual actions are
                           passed to lower-level agents in the launching phase.
                           False → upper_actions tensor stays zero (no cascade).
      ordering           — how to rank agents each step:
                             "intention" sort by computed intention (descending)
                             "random"    random permutation
                             "fixed"     always [0, 1, ..., n_agents-1]

    World-model accuracy stressors (applied only to intention computation,
    not to the env step or training update):
      obs_noise_std — Gaussian noise added to each agent's own observation
                      before it enters compute_intention.  Simulates sensor
                      noise not seen during training (sim-to-real gap).
      H_infer       — rollout horizon used in compute_intention at inference.
                      Lower than training H → shallower, less accurate rollouts.
      F_infer       — number of random orderings sampled in compute_intention.
                      Lower than training F → higher-variance intention estimates.

    Returns (transitions, episode_info).
      transitions  — T_actual*N dicts, one per agent per timestep
      episode_info — scalar metrics + per-step histories for logging
    """
    obs_list = env.reset()
    if comm_channel is not None:
        comm_channel.reset()
    transitions: list[dict] = []

    device = next(encoder.parameters()).device

    # Per-episode accumulators for logging
    total_reward      = 0.0
    n_collisions      = 0
    n_goals_reached   = 0
    coverage_sum      = 0    # sum of per-step coverage counts (islands with exactly 1 agent)
    n_overlaps        = 0    # total extra agents on doubled-up islands across the episode
    step_done         = EPISODE_LEN
    done              = False
    ordering_history  = []
    intention_history = []

    for t in range(EPISODE_LEN):
        obs_tensors = [torch.tensor(o, device=device) for o in obs_list]

        # ── 1. Encode ──────────────────────────────────────────────────────────
        # h is built from clean obs — sensor noise only affects intention rollouts,
        # not the actual action (policy still acts on what the sensor really reports).
        h = [encoder(o.unsqueeze(0)) for o in obs_tensors]

        # ── 2. Intention (negotiation phase) ──────────────────────────────────
        if mode_cfg.compute_intention:
            intentions = []
            for i in range(n_agents):
                # Obs noise: agent i's own sensor is corrupted (sim-to-real gap).
                # Applied before comm so the noisy reading is what gets shared too.
                if obs_noise_std > 0.0:
                    noisy = [
                        o + torch.randn_like(o) * obs_noise_std
                        for o in obs_tensors
                    ]
                else:
                    noisy = obs_tensors

                if comm_channel is not None:
                    # Each agent sees its own (possibly noisy) obs directly;
                    # neighbors' obs are further corrupted by the comm channel.
                    obs_for_i = [
                        comm_channel.transmit(noisy[j], sender=j, receiver=i)
                        if j != i else noisy[i]
                        for j in range(n_agents)
                    ]
                else:
                    obs_for_i = noisy if obs_noise_std > 0.0 else obs_tensors

                intentions.append(
                    compute_intention(
                        i, obs_for_i, encoder, attn_a, attn_w, world_model_net,
                        policy, critic, H_infer, F_infer, n_agents, obs_dim,
                        ACTION_DIM, GAMMA,
                    )
                )
        else:
            intentions = [0.0] * n_agents  # unused; ordering determined by mode

        # ── 3. Priority ordering ───────────────────────────────────────────────
        if mode_cfg.ordering == "intention":
            ordering = sorted(range(n_agents), key=lambda i: -intentions[i])
        elif mode_cfg.ordering == "random":
            ordering = list(range(n_agents))
            random.shuffle(ordering)
        else:  # "fixed"
            ordering = list(range(n_agents))

        ordering_history.append(ordering)
        intention_history.append([float(v) for v in intentions])

        # ── 4. Launching cascade ───────────────────────────────────────────────
        upper_so_far: list[tuple[torch.Tensor, int]] = []  # (action, sender_id)
        agent_data: dict[int, tuple] = {}
        for _, i in enumerate(ordering):
            up_pad = torch.zeros(1, n_agents, ACTION_DIM, device=device)
            if mode_cfg.share_actions:
                for l, (ua, sender_id) in enumerate(upper_so_far):
                    if comm_channel is not None:
                        ua = comm_channel.transmit(ua, sender=sender_id, receiver=i)
                    up_pad[0, l] = ua
            ctx  = attn_a(h[i], up_pad)
            dist = policy(ctx)
            act  = dist.sample()
            lp   = dist.log_prob(act).sum().item()
            val  = critic(ctx).item()
            agent_data[i] = (act.squeeze(0), lp, val, up_pad.squeeze(0))
            upper_so_far.append((act.squeeze(0), i))

        # ── 5. Step environment ───────────────────────────────────────────────
        n_env_actions = env.action_dim if hasattr(env, 'action_dim') else env.N_ACTIONS
        env_actions = [
            int(agent_data[i][0].round().clamp(0, n_env_actions - 1).item())
            for i in range(n_agents)
        ]
        next_obs_list, reward, done = env.step(env_actions)

        # ── 6. Accumulate episode stats ───────────────────────────────────────
        total_reward += reward
        if hasattr(env, 'last_collisions'):
            n_collisions += env.last_collisions
        if hasattr(env, 'last_goals_this_step'):
            n_goals_reached += env.last_goals_this_step
        if hasattr(env, 'last_coverage_count'):
            coverage_sum += env.last_coverage_count
        if hasattr(env, 'last_overlap_count'):
            n_overlaps += env.last_overlap_count

        if done and step_done == EPISODE_LEN:
            step_done = t + 1  # 1-indexed steps to completion

        obs_all      = torch.stack(obs_tensors, 0)
        next_obs_all = torch.stack([torch.tensor(o, device=device) for o in next_obs_list], 0)
        actions_all  = torch.stack([agent_data[i][0] for i in range(n_agents)], 0)

        for i in range(n_agents):
            act_i, lp_i, val_i, up_i = agent_data[i]
            transitions.append({
                "agent_id":    i,
                "obs_all":     obs_all,
                "action_i":    act_i,
                "actions_all": actions_all,
                "up_pad_i":    up_i,
                "next_obs_all": next_obs_all,
                "reward":      reward,
                "value":       val_i,
                "log_prob":    lp_i,
            })

        obs_list = next_obs_list

        if done:
            # Pad ordering/intention history to EPISODE_LEN so downstream
            # stats have a consistent length regardless of when done fires.
            while len(ordering_history) < EPISODE_LEN:
                ordering_history.append(ordering)
                intention_history.append([float(v) for v in intentions])
            break

    n_islands = getattr(getattr(env, 'cfg', None), 'n_islands', 0)
    actual_steps = len(ordering_history)
    episode_info = {
        "total_reward":      total_reward,
        "success":           done,
        "steps_to_completion": step_done,
        "deadlock":          step_done == EPISODE_LEN,
        "n_collisions":      n_collisions,
        "n_goals_reached":   n_goals_reached,
        "coverage_rate":     coverage_sum / (n_islands * actual_steps) if n_islands > 0 else 0.0,
        "n_overlaps":        n_overlaps,
        "n_msgs_dropped":    comm_channel.n_dropped if comm_channel is not None else 0,
        "ordering_history":  ordering_history,
        "intention_history": intention_history,
    }
    return transitions, episode_info


# ── Update ─────────────────────────────────────────────────────────────────────

def update(
    n_agents: int,
    encoder, attn_a, attn_w, world_model_net, policy, critic,
    transitions: list[dict],
    opt_world: torch.optim.Optimizer,
    opt_policy: torch.optim.Optimizer,
    entropy_coeff: float = 0.0,
    max_grad_norm: float = 0.5,
) -> dict:
    device = next(encoder.parameters()).device

    # Use actual episode length — episodes can end before EPISODE_LEN (e.g. all
    # agents reach their goal in the intersection env), and indexing with a
    # hardcoded EPISODE_LEN would cause an out-of-bounds error once learning kicks in.
    T = len([tr for tr in transitions if tr["agent_id"] == 0])

    # GAE per agent (works for any T)
    adv_by_agent: dict[int, list[float]] = {}
    ret_by_agent: dict[int, list[float]] = {}
    for i in range(n_agents):
        agent_trs = [tr for tr in transitions if tr["agent_id"] == i]
        rewards = [tr["reward"] for tr in agent_trs]
        values  = [tr["value"]  for tr in agent_trs]
        adv, ret = compute_gae(rewards, values, GAMMA, LAM)
        adv_by_agent[i] = adv
        ret_by_agent[i] = ret

    # World-model batch: one record per timestep (agent-0 carries all-agent tensors)
    trs_a0 = [tr for tr in transitions if tr["agent_id"] == 0]
    obs_wm      = torch.stack([tr["obs_all"]      for tr in trs_a0])
    actions_wm  = torch.stack([tr["actions_all"]  for tr in trs_a0])
    next_obs_wm = torch.stack([tr["next_obs_all"] for tr in trs_a0])
    rewards_wm  = torch.tensor([tr["reward"] for tr in trs_a0], dtype=torch.float32, device=device)

    # Policy/value batch — iterate over actual transitions, not a hardcoded range
    obs_pv_list, up_list, act_list, ret_list, adv_list, lp_list = [], [], [], [], [], []
    for t in range(T):
        for i in range(n_agents):
            tr = transitions[t * n_agents + i]
            others = [j for j in range(n_agents) if j != i]
            obs_row = torch.stack([tr["obs_all"][i]] + [tr["obs_all"][j] for j in others], 0)
            obs_pv_list.append(obs_row)
            up_list.append(tr["up_pad_i"])
            act_list.append(tr["action_i"])
            ret_list.append(ret_by_agent[i][t])
            adv_list.append(adv_by_agent[i][t])
            lp_list.append(tr["log_prob"])

    obs_pv       = torch.stack(obs_pv_list)
    up_pv        = torch.stack(up_list)
    actions_pv   = torch.stack(act_list)
    returns_pv   = torch.tensor(ret_list,  dtype=torch.float32, device=device)
    advantages_pv = torch.tensor(adv_list, dtype=torch.float32, device=device)
    advantages_pv = (advantages_pv - advantages_pv.mean()) / (advantages_pv.std() + 1e-8)
    log_probs_old = torch.tensor(lp_list,  dtype=torch.float32, device=device)

    # World model update — equation (4) from paper
    opt_world.zero_grad()
    lw = world_model_loss(encoder, attn_w, world_model_net,
                          obs_wm, actions_wm, next_obs_wm, rewards_wm)
    lw.backward()
    world_params = [p for g in opt_world.param_groups for p in g["params"]]
    torch.nn.utils.clip_grad_norm_(world_params, max_grad_norm)
    opt_world.step()

    # Value + policy update — equations (2) and (3) from paper
    opt_policy.zero_grad()
    lv = value_loss(encoder, attn_a, critic, obs_pv, up_pv, returns_pv)
    lp_loss, entropy = ppo_loss(encoder, attn_a, policy,
                                obs_pv[:, 0], up_pv, actions_pv,
                                advantages_pv, log_probs_old, CLIP_EPS,
                                return_entropy=True)
    total_policy_loss = lv + lp_loss - entropy_coeff * entropy
    total_policy_loss.backward()
    policy_params = [p for g in opt_policy.param_groups for p in g["params"]]
    torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
    opt_policy.step()

    return {
        "world_model": lw.item(),
        "value":       lv.item(),
        "policy":      lp_loss.item(),
        "entropy":     entropy.item(),
    }


# ── Weight export ──────────────────────────────────────────────────────────────

def save_weights(
    weights_dir: str,
    obs_dim: int,
    n_agents: int,
    encoder, attn_a, attn_w, world_model_net, policy, critic,
) -> None:
    """
    Export all six modules as TorchScript .pt files and a config.json.

    The C++ LibtorchNeuralModels class loads exactly these files.

    Why TorchScript and not plain state_dicts?
      torch.jit.save() produces a self-contained file: the C++ libtorch runtime
      can load it with torch::jit::load() without needing the Python class
      definition at all.  State dicts require re-instantiating the class.

    Files written:
      encoder.pt       — ObservationEncoder:  obs -> h
      attn_a.pt        — AttentionModule (policy/critic context)
      attn_w.pt        — AttentionModule (world model context)
      world_model.pt   — WorldModel: context -> (obs_flat, r) concatenated
      policy.pt        — ScriptablePolicy: context -> (mean, log_std)
      critic.pt        — Critic: context -> scalar value
      config.json      — dims the C++ loader needs to size its tensors
    """
    os.makedirs(weights_dir, exist_ok=True)

    # Script each module (eval mode so dropout/BN behave correctly at inference)
    encoder.eval()
    attn_a.eval()
    attn_w.eval()
    world_model_net.eval()
    policy.eval()
    critic.eval()

    torch.jit.script(encoder).save(os.path.join(weights_dir, "encoder.pt"))
    torch.jit.script(attn_a).save(os.path.join(weights_dir, "attn_a.pt"))
    torch.jit.script(attn_w).save(os.path.join(weights_dir, "attn_w.pt"))
    torch.jit.script(world_model_net).save(os.path.join(weights_dir, "world_model.pt"))
    torch.jit.script(ScriptablePolicy(policy)).save(os.path.join(weights_dir, "policy.pt"))
    torch.jit.script(critic).save(os.path.join(weights_dir, "critic.pt"))

    config = {
        "obs_dim":    obs_dim,
        "embed_dim":  EMBED_DIM,
        "action_dim": ACTION_DIM,
        "n_agents":   n_agents,
    }
    with open(os.path.join(weights_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Weights saved to {weights_dir}/  "
          f"(obs_dim={obs_dim}, embed_dim={EMBED_DIM}, action_dim={ACTION_DIM})")

    # Switch back to training mode
    encoder.train()
    attn_a.train()
    attn_w.train()
    world_model_net.train()
    policy.train()
    critic.train()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args) -> None:
    global EPISODE_LEN
    if args.episode_len is not None:
        EPISODE_LEN = args.episode_len

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    mode_cfg = MODES[args.mode]
    env      = make_env(args.env, N_AGENTS)
    obs_dim  = env.obs_dim

    comm_cfg     = CommConfig(
        delay=args.comm_delay,
        drop_prob=args.comm_drop,
        noise_std=args.comm_noise,
        bandwidth_bits=args.comm_bits,
    )
    comm_channel = None if comm_cfg.is_perfect() else CommChannel(comm_cfg)
    comm_tag     = comm_cfg.tag()

    H_infer = args.wm_H
    F_infer = args.wm_F

    use_wandb = args.wandb and _WANDB_AVAILABLE
    if args.wandb and not _WANDB_AVAILABLE:
        print("Warning: --wandb requested but wandb is not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.env}_{args.mode}_seed{args.seed}",
            config={
                "env": args.env, "mode": args.mode, "seed": args.seed,
                "episodes": args.episodes, "episode_len": EPISODE_LEN,
                "wm_H": H_infer, "wm_F": F_infer,
                "embed_dim": EMBED_DIM, "lr_world": LR_WORLD, "lr_policy": LR_POLICY,
            },
            tags=[args.env, args.mode],
        )

    print(f"Training [{args.mode}] on '{args.env}' env | "
          f"obs_dim={obs_dim}  embed_dim={EMBED_DIM}  action_dim={ACTION_DIM}  "
          f"agents={N_AGENTS}  episodes={args.episodes}  seed={args.seed}  "
          f"compute_intention={mode_cfg.compute_intention}  "
          f"share_actions={mode_cfg.share_actions}  "
          f"ordering={mode_cfg.ordering}"
          + (f"  comm={comm_tag}" if comm_tag else "")
          + (f"  obs_noise={args.obs_noise}" if args.obs_noise > 0.0 else "")
          + (f"  H={H_infer}" if H_infer != H else "")
          + (f"  F={F_infer}" if F_infer != F else ""))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    encoder         = ObservationEncoder(obs_dim, EMBED_DIM).to(device)
    attn_a          = AttentionModule(EMBED_DIM, ACTION_DIM).to(device)
    attn_w          = AttentionModule(EMBED_DIM, EMBED_DIM + ACTION_DIM).to(device)
    world_model_net = WorldModel(EMBED_DIM, N_AGENTS, obs_dim).to(device)
    policy          = Policy(EMBED_DIM, ACTION_DIM).to(device)
    critic          = Critic(EMBED_DIM).to(device)

    opt_world = optim.Adam(
        list(encoder.parameters()) +
        list(attn_w.parameters()) +
        list(world_model_net.parameters()),
        lr=LR_WORLD,
    )
    opt_policy = optim.Adam(
        list(encoder.parameters()) +
        list(attn_a.parameters()) +
        list(policy.parameters()) +
        list(critic.parameters()),
        lr=LR_POLICY,
    )

    # ── Logger setup ─────────────────────────────────────────────────────────
    logger = None
    if args.log_dir:
        wm_parts = []
        if args.obs_noise > 0.0: wm_parts.append(f"obsnoise{args.obs_noise}")
        if H_infer != H:         wm_parts.append(f"H{H_infer}")
        if F_infer != F:         wm_parts.append(f"F{F_infer}")
        wm_suffix   = ("_" + "_".join(wm_parts)) if wm_parts else ""
        comm_suffix = f"_{comm_tag}" if comm_tag else ""
        log_filename = f"{args.env}_{args.mode}{comm_suffix}{wm_suffix}_seed{args.seed}.jsonl"
        log_path = os.path.join(args.log_dir, log_filename)
        metadata = {
            "env":                args.env,
            "mode":               args.mode,
            "compute_intention":  mode_cfg.compute_intention,
            "share_actions":      mode_cfg.share_actions,
            "ordering":           mode_cfg.ordering,
            "comm_delay":         comm_cfg.delay,
            "comm_drop_prob":     comm_cfg.drop_prob,
            "comm_noise_std":     comm_cfg.noise_std,
            "comm_bandwidth_bits": comm_cfg.bandwidth_bits,
            "obs_noise_std":      args.obs_noise,
            "wm_H":               H_infer,
            "wm_F":               F_infer,
            "seed":               args.seed,
            "n_agents":   N_AGENTS,
            "obs_dim":    obs_dim,
            "embed_dim":  EMBED_DIM,
            "episodes":    args.episodes,
            "episode_len": EPISODE_LEN,
            "H_train":    H,
            "F_train":    F,
            "gamma":      GAMMA,
            "lam":        LAM,
            "lr_world":   LR_WORLD,
            "lr_policy":  LR_POLICY,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        logger = EpisodeLogger(log_path, metadata)

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(args.episodes):
        # Entropy coefficient: linear decay from args.entropy_coeff → 0
        if args.entropy_decay_eps > 0:
            ec = args.entropy_coeff * max(0.0, 1.0 - ep / args.entropy_decay_eps)
        else:
            ec = args.entropy_coeff

        transitions, episode_info = run_episode(
            env, N_AGENTS, obs_dim,
            encoder, attn_a, attn_w, world_model_net, policy, critic,
            mode_cfg=mode_cfg,
            comm_channel=comm_channel,
            obs_noise_std=args.obs_noise,
            H_infer=H_infer,
            F_infer=F_infer,
        )
        losses = update(
            N_AGENTS, encoder, attn_a, attn_w, world_model_net, policy, critic,
            transitions, opt_world, opt_policy,
            entropy_coeff=ec,
            max_grad_norm=args.max_grad_norm,
        )

        # Compute order entropy here so both logger and wandb can use it.
        oh = episode_info["ordering_history"]
        n_steps_ep = len(oh)
        first_movers = [o[0] for o in oh]
        fm_counts = [first_movers.count(i) for i in range(N_AGENTS)]
        probs = [c / n_steps_ep for c in fm_counts if c > 0]
        order_entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0
        episode_info["_order_entropy"] = order_entropy  # passed through to logger

        if logger:
            logger.log(ep, episode_info, losses)

        if use_wandb:
            wandb.log({
                "reward":         episode_info["total_reward"],
                "coverage_rate":  episode_info["coverage_rate"],
                "n_overlaps":     episode_info["n_overlaps"],
                "n_collisions":   episode_info["n_collisions"],
                "success":        int(episode_info["success"]),
                "order_entropy":  order_entropy,
                "policy_entropy": losses["entropy"],
                "entropy_coeff":  ec,
                "wm_loss":        losses["world_model"],
                "value_loss":     losses["value"],
                "policy_loss":    losses["policy"],
            }, step=ep)

        if ep % LOG_EVERY == 0:
            success_str = "✓" if episode_info["success"] else f"t={episode_info['steps_to_completion']}"
            print(
                f"ep {ep:4d} | R={episode_info['total_reward']:7.2f} | "
                f"collisions={episode_info['n_collisions']}  "
                f"done={success_str}  "
                f"wm={losses['world_model']:.4f}  "
                f"v={losses['value']:.4f}  "
                f"π={losses['policy']:.4f}"
            )

        # Periodic checkpoint
        if args.save_weights and args.save_every > 0 and ep > 0 and ep % args.save_every == 0:
            ckpt = os.path.join(args.save_weights, f"ep{ep}")
            save_weights(ckpt, obs_dim, N_AGENTS,
                         encoder, attn_a, attn_w, world_model_net, policy, critic)

    # ── Final save + cleanup ──────────────────────────────────────────────────
    if args.save_weights:
        save_weights(args.save_weights, obs_dim, N_AGENTS,
                     encoder, attn_a, attn_w, world_model_net, policy, critic)

    if logger:
        logger.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeqComm with MAPPO")
    parser.add_argument(
        "--env", choices=["gaussian", "coverage", "intersection"], default="gaussian",
        help="Environment to train on (default: gaussian)",
    )
    parser.add_argument(
        "--mode", choices=list(MODES), default="seqcomm",
        help=(
            "Training variant (default: seqcomm).  "
            "seqcomm=full paper method, "
            "mappo=no communication, "
            "seqcomm_random=random order + action sharing, "
            "seqcomm_no_action=intention order but no action sharing, "
            "seqcomm_fixed=fixed order + action sharing"
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--save-weights", metavar="DIR", default=None,
        help="Directory to export TorchScript weights after training",
    )
    parser.add_argument(
        "--save-every", type=int, default=0, metavar="N",
        help="Also save a checkpoint every N episodes (0 = only at end)",
    )
    parser.add_argument(
        "--log-dir", metavar="DIR", default=None,
        help="Directory to write JSONL training logs (one file per run)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--episode-len", type=int, default=None,
        help="Override EPISODE_LEN (default: 200). Shorter episodes speed up "
             "seqcomm by reducing compute_intention calls; coverage_rate stays comparable.",
    )
    # Communication stressors — all default to perfect (no degradation)
    parser.add_argument(
        "--comm-delay", type=int, default=0, metavar="N",
        help="Message delay in steps; early steps see zeros (default: 0)",
    )
    parser.add_argument(
        "--comm-drop", type=float, default=0.0, metavar="P",
        help="Probability each message is lost, replaced by zeros (default: 0.0)",
    )
    parser.add_argument(
        "--comm-noise", type=float, default=0.0, metavar="STD",
        help="Std of additive Gaussian noise on every message (default: 0.0)",
    )
    parser.add_argument(
        "--comm-bits", type=int, default=0, metavar="B",
        help="Quantise messages to 2^B levels (0 = full float32, default: 0)",
    )
    # World-model accuracy stressors (sim-to-real gap / rollout fidelity)
    parser.add_argument(
        "--obs-noise", type=float, default=0.0, metavar="STD",
        help=(
            "Std of Gaussian noise added to each agent's own observation before "
            "compute_intention. Simulates sensor noise unseen during training. "
            "Does NOT affect the env step or training gradients. (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--entropy-coeff", type=float, default=0.01, metavar="C",
        help="Entropy regularisation coefficient (default: 0.01). Decays linearly "
             "to 0 over the first --entropy-decay-eps episodes to allow ordering "
             "to stabilise once the world model is useful.",
    )
    parser.add_argument(
        "--entropy-decay-eps", type=int, default=150, metavar="N",
        help="Episodes over which entropy coeff decays from --entropy-coeff to 0 "
             "(default: 150). Set to 0 to use a fixed coefficient throughout.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, metavar="G",
        help="Max gradient norm for clipping in both optimisers (default: 0.5). "
             "Set to 0 to disable clipping.",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log metrics to Weights & Biases (requires wandb installed and logged in)",
    )
    parser.add_argument(
        "--wandb-project", default="multi-lvl-comms",
        help="W&B project name (default: multi-lvl-comms)",
    )
    parser.add_argument(
        "--wm-H", type=int, default=H, metavar="N",
        help=f"World-model rollout horizon used at inference (train default: {H}). "
             f"Lower → shallower intention estimates.",
    )
    parser.add_argument(
        "--wm-F", type=int, default=F, metavar="N",
        help=f"Random orderings sampled per intention estimate (train default: {F}). "
             f"Lower → higher-variance intention estimates.",
    )
    args = parser.parse_args()
    main(args)
