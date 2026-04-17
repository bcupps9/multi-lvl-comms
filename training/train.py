"""
SeqComm MAPPO training on GaussianFieldEnv.

Run:
    python train.py
    python train.py --episodes 5        # quick smoke test
"""

import argparse
import torch
import torch.optim as optim

from execution.gaussian_field_env import GaussianFieldEnv, GaussianFieldConfig
from training.train_world_model import (
    ObservationEncoder,
    AttentionModule,
    WorldModel,
    Policy,
    Critic,
    compute_intention,
    compute_gae,
    world_model_loss,
    value_loss,
    ppo_loss,
)

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_AGENTS    = 4
OBS_DIM     = 27    # 2 + (2*window_half+1)^2 with window_half=2
EMBED_DIM   = 64
ACTION_DIM  = 1     # single Gaussian output; rounded to int for env.step()
H           = 5     # world-model rollout horizon for intention
F           = 4     # random orderings per intention estimate
EPISODE_LEN = 200
N_EPISODES  = 2000
GAMMA       = 0.99
LAM         = 0.95
CLIP_EPS    = 0.2
LR_WORLD    = 3e-4
LR_POLICY   = 3e-4
LOG_EVERY   = 10


# ── Rollout ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    env: GaussianFieldEnv,
    encoder, attn_a, attn_w, world_model_net, policy, critic,
) -> list[dict]:
    """
    Run one full episode following the SeqComm protocol:
      1. Encode observations.
      2. Each agent computes its intention via H-step world-model rollout.
      3. Agents are sorted descending by intention → priority ordering.
      4. Launching cascade: each agent conditions on zero-padded upper actions.

    Returns a list of T*N transition dicts (one per agent per timestep).
    """
    obs_list = env.reset()
    transitions: list[dict] = []

    for _ in range(EPISODE_LEN):
        obs_tensors = [torch.tensor(o) for o in obs_list]  # list of (obs_dim,)

        # Encode
        h = [encoder(o.unsqueeze(0)) for o in obs_tensors]  # list of (1, embed)

        # Intentions
        intentions = [
            compute_intention(
                i, obs_tensors, encoder, attn_a, attn_w, world_model_net,
                policy, critic, H, F, N_AGENTS, OBS_DIM, ACTION_DIM, GAMMA,
            )
            for i in range(N_AGENTS)
        ]
        ordering = sorted(range(N_AGENTS), key=lambda i: -intentions[i])

        # Launching cascade
        upper_so_far: list[torch.Tensor] = []
        agent_data: dict[int, tuple] = {}
        for k, i in enumerate(ordering):
            up_pad = torch.zeros(1, N_AGENTS, ACTION_DIM)
            for l, ua in enumerate(upper_so_far):
                up_pad[0, l] = ua
            ctx  = attn_a(h[i], up_pad)
            dist = policy(ctx)
            act  = dist.sample()                        # (1, ACTION_DIM)
            lp   = dist.log_prob(act).sum().item()
            val  = critic(ctx).item()
            agent_data[i] = (act.squeeze(0), lp, val, up_pad.squeeze(0))
            upper_so_far.append(act.squeeze(0))

        # Step environment
        env_actions = [
            int(agent_data[i][0].round().clamp(0, N_ACTIONS - 1).item())
            for i in range(N_AGENTS)
        ]
        next_obs_list, reward, _ = env.step(env_actions)

        obs_all      = torch.stack(obs_tensors, 0)                         # (N, obs_dim)
        next_obs_all = torch.stack([torch.tensor(o) for o in next_obs_list], 0)
        actions_all  = torch.stack([agent_data[i][0] for i in range(N_AGENTS)], 0)

        for i in range(N_AGENTS):
            act_i, lp_i, val_i, up_i = agent_data[i]
            transitions.append({
                "agent_id":   i,
                "obs_all":    obs_all,          # (N, obs_dim)
                "action_i":   act_i,            # (ACTION_DIM,)
                "actions_all": actions_all,     # (N, ACTION_DIM)
                "up_pad_i":   up_i,             # (N, ACTION_DIM) zero-padded
                "next_obs_all": next_obs_all,   # (N, obs_dim)
                "reward":     reward,
                "value":      val_i,
                "log_prob":   lp_i,
            })

        obs_list = next_obs_list

    return transitions


N_ACTIONS = GaussianFieldEnv.N_ACTIONS


# ── Update ─────────────────────────────────────────────────────────────────────

def update(
    encoder, attn_a, attn_w, world_model_net, policy, critic,
    transitions: list[dict],
    opt_world: torch.optim.Optimizer,
    opt_policy: torch.optim.Optimizer,
) -> dict:
    T = EPISODE_LEN

    # ── GAE per agent ──────────────────────────────────────────────────────────
    adv_by_agent  = {}
    ret_by_agent  = {}
    for i in range(N_AGENTS):
        agent_trs = [tr for tr in transitions if tr["agent_id"] == i]
        rewards = [tr["reward"] for tr in agent_trs]
        values  = [tr["value"]  for tr in agent_trs]
        adv, ret = compute_gae(rewards, values, GAMMA, LAM)
        adv_by_agent[i] = adv
        ret_by_agent[i] = ret

    # ── World-model batch: (T, N, ...) ────────────────────────────────────────
    # Pull one sample per timestep (agent 0's record carries all-agent tensors)
    trs_a0 = [tr for tr in transitions if tr["agent_id"] == 0]
    obs_wm      = torch.stack([tr["obs_all"]      for tr in trs_a0])   # (T, N, obs_dim)
    actions_wm  = torch.stack([tr["actions_all"]  for tr in trs_a0])   # (T, N, act_dim)
    next_obs_wm = torch.stack([tr["next_obs_all"] for tr in trs_a0])   # (T, N, obs_dim)
    rewards_wm  = torch.tensor([tr["reward"] for tr in trs_a0],
                                dtype=torch.float32)                   # (T,)

    # ── Policy/value batch: (T*N, ...) ────────────────────────────────────────
    obs_pv_list, up_list, act_list, ret_list, adv_list, lp_list = [], [], [], [], [], []

    for t in range(T):
        for i in range(N_AGENTS):
            tr = transitions[t * N_AGENTS + i]
            # Self obs at index 0, others follow
            others = [j for j in range(N_AGENTS) if j != i]
            obs_row = torch.stack(
                [tr["obs_all"][i]] + [tr["obs_all"][j] for j in others], 0
            )                                                           # (N, obs_dim)
            obs_pv_list.append(obs_row)
            up_list.append(tr["up_pad_i"])
            act_list.append(tr["action_i"])
            ret_list.append(ret_by_agent[i][t])
            adv_list.append(adv_by_agent[i][t])
            lp_list.append(tr["log_prob"])

    obs_pv      = torch.stack(obs_pv_list)                             # (T*N, N, obs_dim)
    up_pv       = torch.stack(up_list)                                 # (T*N, N, act_dim)
    actions_pv  = torch.stack(act_list)                                # (T*N, act_dim)
    returns_pv  = torch.tensor(ret_list, dtype=torch.float32)
    advantages_pv = torch.tensor(adv_list, dtype=torch.float32)
    # Normalize advantages
    advantages_pv = (advantages_pv - advantages_pv.mean()) / (advantages_pv.std() + 1e-8)
    log_probs_old = torch.tensor(lp_list, dtype=torch.float32)

    # ── World model update (eq 4) ──────────────────────────────────────────────
    opt_world.zero_grad()
    lw = world_model_loss(encoder, attn_w, world_model_net,
                          obs_wm, actions_wm, next_obs_wm, rewards_wm)
    lw.backward()
    opt_world.step()

    # ── Value + policy update (eqs 2 & 3) ─────────────────────────────────────
    opt_policy.zero_grad()
    lv = value_loss(encoder, attn_a, critic, obs_pv, up_pv, returns_pv)
    lp_loss = ppo_loss(encoder, attn_a, policy,
                       obs_pv[:, 0], up_pv, actions_pv,
                       advantages_pv, log_probs_old, CLIP_EPS)
    (lv + lp_loss).backward()
    opt_policy.step()

    return {
        "world_model": lw.item(),
        "value":       lv.item(),
        "policy":      lp_loss.item(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(n_episodes: int = N_EPISODES) -> None:
    env = GaussianFieldEnv(GaussianFieldConfig(n_agents=N_AGENTS))

    encoder         = ObservationEncoder(OBS_DIM, EMBED_DIM)
    attn_a          = AttentionModule(EMBED_DIM, ACTION_DIM)
    attn_w          = AttentionModule(EMBED_DIM, EMBED_DIM + ACTION_DIM)
    world_model_net = WorldModel(EMBED_DIM, N_AGENTS, OBS_DIM)
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

    for ep in range(n_episodes):
        transitions = run_episode(
            env, encoder, attn_a, attn_w, world_model_net, policy, critic,
        )
        losses = update(
            encoder, attn_a, attn_w, world_model_net, policy, critic,
            transitions, opt_world, opt_policy,
        )

        if ep % LOG_EVERY == 0:
            ep_reward = sum(
                tr["reward"] for tr in transitions if tr["agent_id"] == 0
            )
            print(
                f"ep {ep:4d} | R={ep_reward:7.2f} | "
                f"wm={losses['world_model']:.4f}  "
                f"v={losses['value']:.4f}  "
                f"π={losses['policy']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    args = parser.parse_args()
    main(args.episodes)
