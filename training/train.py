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
from dataclasses import dataclass
import torch
import torch.optim as optim

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
EPISODE_LEN = 200
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

        # Who was first mover each step?
        first_movers = [order[0] for order in ordering_history]
        n_agents = len(ordering_history[0]) if ordering_history else 4
        first_mover_counts = [first_movers.count(i) for i in range(n_agents)]

        # Entropy of the first-mover distribution across the episode
        probs = [c / n_steps for c in first_mover_counts if c > 0]
        order_entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0

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
            "order_entropy":         round(order_entropy, 4),
            "mean_intention_spread": round(mean_intention_spread, 4),
            "first_mover_counts":    first_mover_counts,
            "world_model_loss":      round(losses["world_model"], 6),
            "value_loss":            round(losses["value"], 6),
            "policy_loss":           round(losses["policy"], 6),
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

    Returns (transitions, episode_info).
      transitions  — T_actual*N dicts, one per agent per timestep
      episode_info — scalar metrics + per-step histories for logging
    """
    obs_list = env.reset()
    transitions: list[dict] = []

    # Per-episode accumulators for logging
    total_reward      = 0.0
    n_collisions      = 0
    n_goals_reached   = 0
    step_done         = EPISODE_LEN
    done              = False
    ordering_history  = []
    intention_history = []

    for t in range(EPISODE_LEN):
        obs_tensors = [torch.tensor(o) for o in obs_list]

        # ── 1. Encode ──────────────────────────────────────────────────────────
        h = [encoder(o.unsqueeze(0)) for o in obs_tensors]

        # ── 2. Intention (negotiation phase) ──────────────────────────────────
        if mode_cfg.compute_intention:
            intentions = [
                compute_intention(
                    i, obs_tensors, encoder, attn_a, attn_w, world_model_net,
                    policy, critic, H, F, n_agents, obs_dim, ACTION_DIM, GAMMA,
                )
                for i in range(n_agents)
            ]
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
        upper_so_far: list[torch.Tensor] = []
        agent_data: dict[int, tuple] = {}
        for _, i in enumerate(ordering):
            up_pad = torch.zeros(1, n_agents, ACTION_DIM)
            if mode_cfg.share_actions:
                for l, ua in enumerate(upper_so_far):
                    up_pad[0, l] = ua
            ctx  = attn_a(h[i], up_pad)
            dist = policy(ctx)
            act  = dist.sample()
            lp   = dist.log_prob(act).sum().item()
            val  = critic(ctx).item()
            agent_data[i] = (act.squeeze(0), lp, val, up_pad.squeeze(0))
            upper_so_far.append(act.squeeze(0))

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

        if done and step_done == EPISODE_LEN:
            step_done = t + 1  # 1-indexed steps to completion

        obs_all      = torch.stack(obs_tensors, 0)
        next_obs_all = torch.stack([torch.tensor(o) for o in next_obs_list], 0)
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

    episode_info = {
        "total_reward":      total_reward,
        "success":           done,
        "steps_to_completion": step_done,
        "deadlock":          step_done == EPISODE_LEN,
        "n_collisions":      n_collisions,
        "n_goals_reached":   n_goals_reached,
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
) -> dict:
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
    rewards_wm  = torch.tensor([tr["reward"] for tr in trs_a0], dtype=torch.float32)

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
    returns_pv   = torch.tensor(ret_list,  dtype=torch.float32)
    advantages_pv = torch.tensor(adv_list, dtype=torch.float32)
    advantages_pv = (advantages_pv - advantages_pv.mean()) / (advantages_pv.std() + 1e-8)
    log_probs_old = torch.tensor(lp_list,  dtype=torch.float32)

    # World model update — equation (4) from paper
    opt_world.zero_grad()
    lw = world_model_loss(encoder, attn_w, world_model_net,
                          obs_wm, actions_wm, next_obs_wm, rewards_wm)
    lw.backward()
    opt_world.step()

    # Value + policy update — equations (2) and (3) from paper
    opt_policy.zero_grad()
    lv = value_loss(encoder, attn_a, critic, obs_pv, up_pv, returns_pv)
    lp_loss = ppo_loss(encoder, attn_a, policy,
                       obs_pv[:, 0], up_pv, actions_pv,
                       advantages_pv, log_probs_old, CLIP_EPS)
    (lv + lp_loss).backward()
    opt_policy.step()

    return {"world_model": lw.item(), "value": lv.item(), "policy": lp_loss.item()}


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
    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    mode_cfg = MODES[args.mode]
    env      = make_env(args.env, N_AGENTS)
    obs_dim  = env.obs_dim

    print(f"Training [{args.mode}] on '{args.env}' env | "
          f"obs_dim={obs_dim}  embed_dim={EMBED_DIM}  action_dim={ACTION_DIM}  "
          f"agents={N_AGENTS}  episodes={args.episodes}  seed={args.seed}  "
          f"compute_intention={mode_cfg.compute_intention}  "
          f"share_actions={mode_cfg.share_actions}  "
          f"ordering={mode_cfg.ordering}")

    encoder         = ObservationEncoder(obs_dim, EMBED_DIM)
    attn_a          = AttentionModule(EMBED_DIM, ACTION_DIM)
    attn_w          = AttentionModule(EMBED_DIM, EMBED_DIM + ACTION_DIM)
    world_model_net = WorldModel(EMBED_DIM, N_AGENTS, obs_dim)
    policy          = Policy(EMBED_DIM, ACTION_DIM)
    critic          = Critic(EMBED_DIM)

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
        log_filename = f"{args.env}_{args.mode}_seed{args.seed}.jsonl"
        log_path = os.path.join(args.log_dir, log_filename)
        metadata = {
            "env":                args.env,
            "mode":               args.mode,
            "compute_intention":  mode_cfg.compute_intention,
            "share_actions":      mode_cfg.share_actions,
            "ordering":           mode_cfg.ordering,
            "seed":               args.seed,
            "n_agents":   N_AGENTS,
            "obs_dim":    obs_dim,
            "embed_dim":  EMBED_DIM,
            "episodes":   args.episodes,
            "episode_len": EPISODE_LEN,
            "H":          H,
            "F":          F,
            "gamma":      GAMMA,
            "lam":        LAM,
            "lr_world":   LR_WORLD,
            "lr_policy":  LR_POLICY,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        logger = EpisodeLogger(log_path, metadata)

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(args.episodes):
        transitions, episode_info = run_episode(
            env, N_AGENTS, obs_dim,
            encoder, attn_a, attn_w, world_model_net, policy, critic,
            mode_cfg=mode_cfg,
        )
        losses = update(
            N_AGENTS, encoder, attn_a, attn_w, world_model_net, policy, critic,
            transitions, opt_world, opt_policy,
        )

        if logger:
            logger.log(ep, episode_info, losses)

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
    args = parser.parse_args()
    main(args)
