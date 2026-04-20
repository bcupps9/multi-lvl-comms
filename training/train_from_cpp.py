"""
train_from_cpp.py — Python side of the C++ ↔ Python training loop.

Waits for the C++ sim to write a trajectory, runs MAPPO update, saves new
weights, then signals C++ to reload and run the next episode.

Usage (two terminals):
    # terminal 1 — start C++ sim (it will block waiting for Python)
    ./build/robot_sim/seqcomm-sim-trained weights/ 2000

    # terminal 2 — start this script
    python -m training.train_from_cpp weights/

The two processes synchronise via two sentinel files in weights_dir:
    traj.bin     — flat binary trajectory written by C++
    traj.ready   — touched by C++ when traj.bin is complete
    weights.ready — touched here when new .pt files are ready

Binary format (matches trajectory_io.hh):
    header:  int32 n_agents, obs_dim, action_dim, n_transitions
    per record:
        int32  agent_id, timestep, n_upper
        float32[obs_dim]               obs
        float32[action_dim]            action
        float32[n_agents*action_dim]   upper_actions (first n_upper filled)
        float32[obs_dim]               next_obs
        float32                        reward, value, log_prob, log_prob_old
"""

import argparse
import os
import struct
import time
import numpy as np
import torch
import torch.optim as optim

from training.train_world_model import (
    ObservationEncoder,
    AttentionModule,
    WorldModel,
    Policy,
    Critic,
    ScriptablePolicy,
    compute_gae,
    world_model_loss,
    value_loss,
    ppo_loss,
)
from training.train import save_weights, N_AGENTS, EMBED_DIM, ACTION_DIM
from training.train import H, F, GAMMA, LAM, CLIP_EPS, LR_WORLD, LR_POLICY, LOG_EVERY

POLL_INTERVAL = 0.05   # seconds between sentinel checks - we should/could set up a sigint?


# ── Binary reader ─────────────────────────────────────────────────────────────

def read_trajectory(path: str):
    """
    Read a trajectory file written by trajectory_io.hh.
    Returns (transitions, n_agents, obs_dim, action_dim) where transitions
    is a list of dicts in the same format that train.py's run_episode() produces.
    """
    with open(path, 'rb') as f:
        n_agents, obs_dim, action_dim, n_trs = struct.unpack('<4i', f.read(16))

        raw = []
        for _ in range(n_trs):
            agent_id, timestep, n_upper = struct.unpack('<3i', f.read(12))

            obs      = _read_floats(f, obs_dim)
            action   = _read_floats(f, action_dim)
            up_flat  = _read_floats(f, n_agents * action_dim)
            next_obs = _read_floats(f, obs_dim)
            reward, value, log_prob, _lp_old = struct.unpack('<4f', f.read(16))

            raw.append({
                'agent_id': agent_id,
                'timestep': timestep,
                'n_upper':  n_upper,
                'obs':      obs,
                'action':   action,
                'up_pad':   up_flat.reshape(n_agents, action_dim),
                'next_obs': next_obs,
                'reward':   float(reward),
                'value':    float(value),
                'log_prob': float(log_prob),
            })

    transitions = _assemble(raw, n_agents, obs_dim, action_dim)
    return transitions, n_agents, obs_dim, action_dim


def _read_floats(f, n: int) -> np.ndarray:
    return np.frombuffer(f.read(n * 4), dtype='<f4').copy()


def _assemble(raw, n_agents, obs_dim, action_dim):
    """
    Group per-agent records by timestep and reconstruct the all-agent tensors
    (obs_all, actions_all, next_obs_all) that update() expects.

    Output list is ordered (t=0,a=0), (t=0,a=1), …, (t=T-1,a=N-1)
    matching the transitions[t * n_agents + i] indexing in update().
    """
    by_step: dict[int, dict[int, dict]] = {}
    for rec in raw:
        by_step.setdefault(rec['timestep'], {})[rec['agent_id']] = rec

    assembled = []
    for t in sorted(by_step.keys()):
        step = by_step[t]
        obs_all      = torch.stack([torch.tensor(step[i]['obs'])      for i in range(n_agents)])
        actions_all  = torch.stack([torch.tensor(step[i]['action'])   for i in range(n_agents)])
        next_obs_all = torch.stack([torch.tensor(step[i]['next_obs']) for i in range(n_agents)])
        reward = step[0]['reward']  # shared reward, identical across agents

        for i in range(n_agents):
            rec = step[i]
            assembled.append({
                'agent_id':    i,
                'obs_all':     obs_all,
                'action_i':    torch.tensor(rec['action']),
                'actions_all': actions_all,
                'up_pad_i':    torch.tensor(rec['up_pad']),
                'next_obs_all': next_obs_all,
                'reward':      reward,
                'value':       rec['value'],
                'log_prob':    rec['log_prob'],
            })

    return assembled


# ── MAPPO update (mirrors train.py's update()) ────────────────────────────────

def update(n_agents, episode_len,
           encoder, attn_a, attn_w, world_model_net, policy, critic,
           transitions, opt_world, opt_policy):
    T = episode_len

    adv_by_agent, ret_by_agent = {}, {}
    for i in range(n_agents):
        agent_trs = [tr for tr in transitions if tr['agent_id'] == i]
        rewards = [tr['reward'] for tr in agent_trs]
        values  = [tr['value']  for tr in agent_trs]
        adv, ret = compute_gae(rewards, values, GAMMA, LAM)
        adv_by_agent[i] = adv
        ret_by_agent[i] = ret

    trs_a0 = [tr for tr in transitions if tr['agent_id'] == 0]
    obs_wm      = torch.stack([tr['obs_all']      for tr in trs_a0])
    actions_wm  = torch.stack([tr['actions_all']  for tr in trs_a0])
    next_obs_wm = torch.stack([tr['next_obs_all'] for tr in trs_a0])
    rewards_wm  = torch.tensor([tr['reward'] for tr in trs_a0], dtype=torch.float32)

    obs_pv_list, up_list, act_list, ret_list, adv_list, lp_list = [], [], [], [], [], []
    for t in range(T):
        for i in range(n_agents):
            tr = transitions[t * n_agents + i]
            others = [j for j in range(n_agents) if j != i]
            obs_row = torch.stack([tr['obs_all'][i]] + [tr['obs_all'][j] for j in others])
            obs_pv_list.append(obs_row)
            up_list.append(tr['up_pad_i'])
            act_list.append(tr['action_i'])
            ret_list.append(ret_by_agent[i][t])
            adv_list.append(adv_by_agent[i][t])
            lp_list.append(tr['log_prob'])

    obs_pv        = torch.stack(obs_pv_list)
    up_pv         = torch.stack(up_list)
    actions_pv    = torch.stack(act_list)
    returns_pv    = torch.tensor(ret_list,  dtype=torch.float32)
    advantages_pv = torch.tensor(adv_list, dtype=torch.float32)
    advantages_pv = (advantages_pv - advantages_pv.mean()) / (advantages_pv.std() + 1e-8)
    log_probs_old = torch.tensor(lp_list,  dtype=torch.float32)

    opt_world.zero_grad()
    lw = world_model_loss(encoder, attn_w, world_model_net,
                          obs_wm, actions_wm, next_obs_wm, rewards_wm)
    lw.backward()
    opt_world.step()

    opt_policy.zero_grad()
    lv       = value_loss(encoder, attn_a, critic, obs_pv, up_pv, returns_pv)
    lp_loss  = ppo_loss(encoder, attn_a, policy,
                        obs_pv[:, 0], up_pv, actions_pv,
                        advantages_pv, log_probs_old, CLIP_EPS)
    (lv + lp_loss).backward()
    opt_policy.step()

    return {'world_model': lw.item(), 'value': lv.item(), 'policy': lp_loss.item()}


# ── Main loop ─────────────────────────────────────────────────────────────────

def main(weights_dir: str) -> None:
    traj_bin   = os.path.join(weights_dir, 'traj.bin')
    traj_ready = os.path.join(weights_dir, 'traj.ready')
    wts_ready  = os.path.join(weights_dir, 'weights.ready')

    # Infer obs_dim from config.json written by an earlier train.py run.
    import json
    cfg_path = os.path.join(weights_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found — run train.py first to bootstrap weights")
    with open(cfg_path) as f:
        cfg = json.load(f)
    obs_dim = cfg['obs_dim']

    encoder         = ObservationEncoder(obs_dim, EMBED_DIM)
    attn_a          = AttentionModule(EMBED_DIM, ACTION_DIM)
    attn_w          = AttentionModule(EMBED_DIM, EMBED_DIM + ACTION_DIM)
    world_model_net = WorldModel(EMBED_DIM, N_AGENTS, obs_dim)
    policy          = Policy(EMBED_DIM, ACTION_DIM)
    critic          = Critic(EMBED_DIM)

    # Load current weights from disk so C++ and Python start in sync.
    def _load_state(module, name):
        scripted = torch.jit.load(os.path.join(weights_dir, name))
        module.load_state_dict(
            {k: v for k, v in scripted.state_dict().items()}, strict=False)

    for mod, name in [
        (encoder,         'encoder.pt'),
        (attn_a,          'attn_a.pt'),
        (attn_w,          'attn_w.pt'),
        (world_model_net, 'world_model.pt'),
        (policy,          'policy.pt'),
        (critic,          'critic.pt'),
    ]:
        try:
            _load_state(mod, name)
        except Exception as e:
            print(f"  warning: could not load {name}: {e}")

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

    print(f"train_from_cpp: watching {weights_dir}/  obs_dim={obs_dim}  "
          f"embed_dim={EMBED_DIM}  agents={N_AGENTS}")
    print("Waiting for C++ sim…  (run: seqcomm-sim-trained weights/)\n")

    ep = 0
    reward_history = []

    while True:
        # Wait for C++ to finish an episode
        while not os.path.exists(traj_ready):
            time.sleep(POLL_INTERVAL)

        t0 = time.time()
        transitions, n_agents, obs_dim_file, action_dim = read_trajectory(traj_bin)
        episode_len = len(transitions) // n_agents

        losses = update(
            n_agents, episode_len,
            encoder, attn_a, attn_w, world_model_net, policy, critic,
            transitions, opt_world, opt_policy,
        )

        ep_reward = sum(tr['reward'] for tr in transitions if tr['agent_id'] == 0)
        reward_history.append(ep_reward)

        # Save updated weights
        save_weights(weights_dir, obs_dim, n_agents,
                     encoder, attn_a, attn_w, world_model_net, policy, critic)

        # Signal C++
        open(wts_ready, 'w').close()

        elapsed = time.time() - t0
        if ep % LOG_EVERY == 0:
            avg = sum(reward_history[-LOG_EVERY:]) / min(len(reward_history), LOG_EVERY)
            print(
                f"ep {ep:4d} | R={ep_reward:7.2f}  avg={avg:7.2f} | "
                f"wm={losses['world_model']:.4f}  "
                f"v={losses['value']:.4f}  "
                f"π={losses['policy']:.4f}  "
                f"({elapsed:.1f}s)"
            )

        ep += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Python MAPPO updater for the C++ SeqComm training loop')
    parser.add_argument('weights_dir', help='Directory containing .pt files and sentinels')
    args = parser.parse_args()
    main(args.weights_dir)
