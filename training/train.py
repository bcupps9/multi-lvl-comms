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
import os
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
) -> list[dict]:
    """
    Run one full episode following the SeqComm protocol:
      1. Encode observations → hidden states h.
      2. Each agent estimates its intention via H-step world-model rollout.
      3. Sort agents by intention (descending) → priority ordering.
      4. Launching cascade: each agent conditions on zero-padded upper actions.

    Returns T*N transition dicts (one per agent per timestep).

    Intention captures how much reward the agent predicts if it goes first.
    The agent with the highest intention becomes the first mover — it commits
    to an action before lower-priority agents, who then condition on that action.
    This is Algorithm 5 in the paper.
    """
    obs_list = env.reset()
    transitions: list[dict] = []

    for _ in range(EPISODE_LEN):
        obs_tensors = [torch.tensor(o) for o in obs_list]  # list of (obs_dim,)

        # ── 1. Encode ──────────────────────────────────────────────────────────
        h = [encoder(o.unsqueeze(0)) for o in obs_tensors]  # list of (1, embed)

        # ── 2. Compute intentions (Algorithm 5 in paper) ──────────────────────
        #  Each agent simulates F random orderings × H world-model steps,
        #  treating itself as the first mover.  The average discounted return
        #  is the agent's "intention value."
        intentions = [
            compute_intention(
                i, obs_tensors, encoder, attn_a, attn_w, world_model_net,
                policy, critic, H, F, n_agents, obs_dim, ACTION_DIM, GAMMA,
            )
            for i in range(n_agents)
        ]

        # ── 3. Priority ordering: highest intention goes first ─────────────────
        ordering = sorted(range(n_agents), key=lambda i: -intentions[i])

        # ── 4. Launching cascade (Algorithm 3 in paper) ───────────────────────
        #  Each agent in priority order conditions its action on all upper
        #  agents' actual actions.  Lower agents see those actions via
        #  zero-padded upper_actions tensor.
        upper_so_far: list[torch.Tensor] = []
        agent_data: dict[int, tuple] = {}
        for _, i in enumerate(ordering):
            up_pad = torch.zeros(1, n_agents, ACTION_DIM)
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
        next_obs_list, reward, _ = env.step(env_actions)

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

    return transitions


# ── Update ─────────────────────────────────────────────────────────────────────

def update(
    n_agents: int,
    encoder, attn_a, attn_w, world_model_net, policy, critic,
    transitions: list[dict],
    opt_world: torch.optim.Optimizer,
    opt_policy: torch.optim.Optimizer,
) -> dict:
    T = EPISODE_LEN

    # GAE per agent
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

    # Policy/value batch
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
    env      = make_env(args.env, N_AGENTS)
    obs_dim  = env.obs_dim

    print(f"Training SeqComm on '{args.env}' env | "
          f"obs_dim={obs_dim}  embed_dim={EMBED_DIM}  action_dim={ACTION_DIM}  "
          f"agents={N_AGENTS}  episodes={args.episodes}")

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

    for ep in range(args.episodes):
        transitions = run_episode(
            env, N_AGENTS, obs_dim,
            encoder, attn_a, attn_w, world_model_net, policy, critic,
        )
        losses = update(
            N_AGENTS, encoder, attn_a, attn_w, world_model_net, policy, critic,
            transitions, opt_world, opt_policy,
        )

        if ep % LOG_EVERY == 0:
            ep_reward = sum(tr["reward"] for tr in transitions if tr["agent_id"] == 0)
            print(
                f"ep {ep:4d} | R={ep_reward:7.2f} | "
                f"wm={losses['world_model']:.4f}  "
                f"v={losses['value']:.4f}  "
                f"π={losses['policy']:.4f}"
            )

        # Periodic checkpoint
        if args.save_weights and args.save_every > 0 and ep > 0 and ep % args.save_every == 0:
            ckpt = os.path.join(args.save_weights, f"ep{ep}")
            save_weights(ckpt, obs_dim, N_AGENTS,
                         encoder, attn_a, attn_w, world_model_net, policy, critic)

    # Final save
    if args.save_weights:
        save_weights(args.save_weights, obs_dim, N_AGENTS,
                     encoder, attn_a, attn_w, world_model_net, policy, critic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeqComm with MAPPO")
    parser.add_argument(
        "--env", choices=["gaussian", "coverage", "intersection"], default="gaussian",
        help="Environment to train on (default: gaussian)",
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
    args = parser.parse_args()
    main(args)
