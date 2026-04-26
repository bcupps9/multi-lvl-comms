"""
train_battleship.py — Python side of the C++ battleship training loop.

Collaborative Battleship: N_a agent ships vs N_b boss ships on an M×M grid.
The boss uses a fixed C++ heuristic (no neural training). We train agent
policy and world model only.

Architecture (Appendix D.2 of SeqComm paper):
    Encoder    : Linear → tanh → Linear → tanh   (obs_dim → embed_dim)
    AttentionA : scaled dot-product attention over upper actions
    AttentionW : scaled dot-product attention over (enc_obs, action) pairs
    Policy     : Linear → tanh → (mean, log_std)  (embed_dim → action_dim)
    Critic     : Linear → tanh → Linear            (embed_dim → 1)
    WorldModel : Linear → tanh → Linear            (embed_dim → N*obs_dim+1)

Loss functions (paper Eqs 2-4):
    Eq 2: value loss  — MSE( V(AM_a(e(o), a_upper)), R_hat )
    Eq 3: policy loss — PPO-clip surrogate
    Eq 4: WM loss     — MSE( M(AM_w(e(o), a)), (o'_flat, r) )

Synchronisation with C++ sim:
    C++ writes traj.bin then touches traj.ready.
    Python reads, trains, writes weights.bin, then touches weights.ready.

Usage:
    # Bootstrap initial weights before starting C++ sim:
    python battleship/train_battleship.py weights_bs/ --init

    # In a separate terminal, start C++ sim, then start trainer:
    python battleship/train_battleship.py weights_bs/

Default grid / ship config (must match BattleshipConfig in C++):
    M=8, n_agents=2, n_boss=2, sight_range=4  →  obs_dim=245, action_dim=3
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_AGENTS   = 2
OBS_DIM    = 245   # (2*4+1)^2 * 3 + 2  — local 9x9 patch * 3 channels + 2 scalars
ACTION_DIM = 3     # (move_dir, fire_dr, fire_dc)
EMBED_DIM  = 64

GAMMA    = 0.99
LAM      = 0.95
CLIP_EPS = 0.2
LR_ENC   = 1e-4
LR_WORLD = 3e-4
LR_POL   = 3e-4

LOG_EVERY    = 20
POLL_INTERVAL = 0.05  # seconds

# ── Neural modules ────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """e(o): raw observation → hidden state h.  FC→tanh→FC→tanh."""

    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (..., obs_dim) → (..., embed_dim)
        return self.net(obs)


class AttentionModule(nn.Module):
    """
    AM_a or AM_w: aggregate a set of messages relative to a query h_self.

    AM_a: messages are upper agent actions   (action_dim each)
    AM_w: messages are cat(enc_obs, action)  (embed_dim+action_dim each)

    Inputs:
        h_self   : (batch, embed_dim)
        messages : (batch, n, msg_dim)
    Output:
        context  : (batch, embed_dim)
    """

    def __init__(self, embed_dim: int, msg_dim: int, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim + msg_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, h_self: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        n = messages.shape[1]
        h_exp = h_self.unsqueeze(1).expand(-1, n, -1)          # (B, n, embed)
        x = torch.cat([h_exp, messages], dim=-1)                # (B, n, embed+msg)
        x = self.input_proj(x)                                   # (B, n, embed)
        attn_out = self.attn(x, x, x)[0]                        # (B, n, embed)
        context = attn_out.mean(dim=1)                           # (B, embed)
        return self.out_proj(context)


class WorldModel(nn.Module):
    """M: AM_w context → predicted (o'_all_flat, r).  FC→tanh→FC."""

    def __init__(self, embed_dim: int, n_agents: int, obs_dim: int):
        super().__init__()
        self.output_dim = n_agents * obs_dim + 1
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Tanh(),
            nn.Linear(embed_dim * 2, self.output_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class Policy(nn.Module):
    """π: AM_a context → Gaussian action.  For training (uses Normal dist)."""

    def __init__(self, embed_dim: int, action_dim: int):
        super().__init__()
        self.mean_layer = nn.Linear(embed_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, context: torch.Tensor):
        mean = self.mean_layer(context)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class ScriptablePolicy(nn.Module):
    """TorchScript-safe wrapper: returns (mean, log_std) tensors."""

    def __init__(self, policy: Policy):
        super().__init__()
        self.mean_layer = policy.mean_layer
        self.log_std = policy.log_std

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_layer(context)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


class Critic(nn.Module):
    """V: AM_a context → scalar value.  FC→tanh→FC."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context).squeeze(-1)


# ── Weight I/O ────────────────────────────────────────────────────────────────

# Must match the order read by LibtorchNeuralModels::update_from_blob() in C++.
MODULE_ORDER = ['encoder', 'attn_a', 'attn_w', 'world_model', 'policy', 'critic']


def save_torchscript(nets: dict, weights_dir: str, obs_dim: int, n_agents: int) -> None:
    """Export all modules as TorchScript .pt files + config.json."""
    os.makedirs(weights_dir, exist_ok=True)

    torch.jit.script(nets['encoder']).save(os.path.join(weights_dir, 'encoder.pt'))
    torch.jit.script(nets['attn_a']).save(os.path.join(weights_dir, 'attn_a.pt'))
    torch.jit.script(nets['attn_w']).save(os.path.join(weights_dir, 'attn_w.pt'))
    torch.jit.script(nets['world_model']).save(os.path.join(weights_dir, 'world_model.pt'))
    # Policy must be wrapped before scripting (Normal dist is not scriptable)
    torch.jit.script(ScriptablePolicy(nets['policy'])).save(
        os.path.join(weights_dir, 'policy.pt'))
    torch.jit.script(nets['critic']).save(os.path.join(weights_dir, 'critic.pt'))

    cfg = {
        'embed_dim': EMBED_DIM,
        'action_dim': ACTION_DIM,
        'obs_dim': obs_dim,
        'n_agents': n_agents,
    }
    with open(os.path.join(weights_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'TorchScript weights saved → {os.path.abspath(weights_dir)}')


def save_weights_bin(nets: dict, weights_dir: str) -> None:
    """
    Write weights.bin: magic(4B) + n_floats(4B) + float32 params.

    Parameter order must match MODULE_ORDER and LibtorchNeuralModels::update_from_blob().
    For policy, we flatten the ScriptablePolicy params (mean_layer + log_std).
    """
    all_params: list[np.ndarray] = []
    for key in MODULE_ORDER:
        mod = nets[key]
        # For policy, wrap to match what was scripted (ScriptablePolicy layout)
        if key == 'policy':
            mod = ScriptablePolicy(mod)
        for p in mod.parameters():
            all_params.append(p.detach().cpu().numpy().astype(np.float32).ravel())

    flat = np.concatenate(all_params)
    n_floats = len(flat)
    MAGIC = 0x57544253

    with open(os.path.join(weights_dir, 'weights.bin'), 'wb') as f:
        f.write(struct.pack('<II', MAGIC, n_floats))
        f.write(flat.tobytes())


# ── Trajectory I/O ────────────────────────────────────────────────────────────

def read_trajectory(path: str):
    """
    Read traj.bin written by trajectory_io.hh.

    Returns (transitions, n_agents, obs_dim, action_dim) where transitions
    is a list of assembled dicts (same format as train.py run_episode output).
    """
    with open(path, 'rb') as f:
        n_agents, obs_dim, action_dim, n_trs = struct.unpack('<4i', f.read(16))
        raw = []
        for _ in range(n_trs):
            agent_id, timestep, n_upper = struct.unpack('<3i', f.read(12))
            obs      = _rfloats(f, obs_dim)
            action   = _rfloats(f, action_dim)
            up_flat  = _rfloats(f, n_agents * action_dim)
            next_obs = _rfloats(f, obs_dim)
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

    return _assemble(raw, n_agents, obs_dim, action_dim), n_agents, obs_dim, action_dim


def _rfloats(f, n: int) -> np.ndarray:
    return np.frombuffer(f.read(n * 4), dtype='<f4').copy()


def _assemble(raw, n_agents, obs_dim, action_dim):
    """
    Group per-agent records by timestep → list of per-agent transition dicts.
    Order: (t=0,a=0), (t=0,a=1), …, (t=T-1,a=N-1).
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
        reward = step[0]['reward']

        for i in range(n_agents):
            rec = step[i]
            assembled.append({
                'agent_id':    i,
                'obs_all':     obs_all,         # (n_agents, obs_dim)
                'action_i':    torch.tensor(rec['action']),
                'actions_all': actions_all,      # (n_agents, action_dim)
                'up_pad_i':    torch.tensor(rec['up_pad']),   # (n_agents, action_dim)
                'next_obs_all': next_obs_all,
                'reward':      reward,
                'value':       rec['value'],
                'log_prob':    rec['log_prob'],
            })
    return assembled


# ── Advantage estimation ──────────────────────────────────────────────────────

def compute_gae(rewards: list, values: list,
                gamma: float = GAMMA, lam: float = LAM):
    T = len(rewards)
    adv = [0.0] * T
    ret = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        ret[t] = gae + values[t]
    return adv, ret


# ── Loss functions ────────────────────────────────────────────────────────────

def loss_value(encoder, attn_a, critic,
               obs_pv, up_pv, returns_pv):
    """Eq 2: MSE value loss."""
    h = encoder(obs_pv[:, 0])              # (B, embed)
    ctx = attn_a(h, up_pv)                # (B, embed)
    v = critic(ctx)                        # (B,)
    diff = returns_pv - v
    return (diff * diff).mean()


def loss_policy(encoder, attn_a, policy,
                obs_self, up_pv, actions_taken,
                advantages, log_probs_old, clip_eps=CLIP_EPS):
    """Eq 3: PPO-clip surrogate loss."""
    h = encoder(obs_self)                  # (B, embed)
    ctx = attn_a(h, up_pv)               # (B, embed)
    dist = policy(ctx)
    log_probs = dist.log_prob(actions_taken).sum(dim=-1)   # (B,)
    ratio = (log_probs - log_probs_old).exp()
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


def loss_world_model(encoder, attn_w, world_model,
                     obs, actions, next_obs, rewards):
    """Eq 4: MSE world-model loss."""
    batch, n_agents, obs_dim = obs.shape
    enc = encoder(obs)                               # (B, N, embed)
    msgs = torch.cat([enc, actions], dim=-1)          # (B, N, embed+act)
    h_self = enc.mean(dim=1)                         # (B, embed)
    ctx = attn_w(h_self, msgs)                       # (B, embed)
    pred = world_model(ctx)                          # (B, N*obs_dim+1)
    target_obs = next_obs.reshape(batch, -1)
    target = torch.cat([target_obs, rewards.unsqueeze(-1)], dim=-1)
    diff = target - pred
    return (diff * diff).sum(dim=-1).mean()


# ── Single-episode update ─────────────────────────────────────────────────────

def update_episode(transitions, n_agents, nets,
                   opt_enc, opt_world, opt_pol) -> dict:
    encoder     = nets['encoder']
    attn_a      = nets['attn_a']
    attn_w      = nets['attn_w']
    world_model = nets['world_model']
    policy      = nets['policy']
    critic      = nets['critic']

    # ── GAE per agent ──────────────────────────────────────────────────────────
    adv_by = {}
    ret_by = {}
    for i in range(n_agents):
        agent_trs = [tr for tr in transitions if tr['agent_id'] == i]
        rews  = [tr['reward'] for tr in agent_trs]
        vals  = [tr['value']  for tr in agent_trs]
        adv, ret = compute_gae(rews, vals)
        adv_by[i] = adv
        ret_by[i] = ret

    T = len(transitions) // n_agents

    # ── World-model batch (one record per timestep, across all agents) ─────────
    trs0 = [tr for tr in transitions if tr['agent_id'] == 0]
    obs_wm      = torch.stack([tr['obs_all']      for tr in trs0])   # (T, N, obs_dim)
    actions_wm  = torch.stack([tr['actions_all']  for tr in trs0])   # (T, N, act_dim)
    next_obs_wm = torch.stack([tr['next_obs_all'] for tr in trs0])   # (T, N, obs_dim)
    rewards_wm  = torch.tensor([tr['reward'] for tr in trs0], dtype=torch.float32)

    # ── Policy/value batch (one record per agent per timestep) ─────────────────
    obs_pv_list, up_list, obs_self_list = [], [], []
    act_list, ret_list, adv_list, lp_list = [], [], [], []

    for t in range(T):
        for i in range(n_agents):
            tr = transitions[t * n_agents + i]
            obs_pv_list.append(tr['obs_all'])          # (N, obs_dim) — self is [i]
            up_list.append(tr['up_pad_i'])              # (N, act_dim) — upper actions
            obs_self_list.append(tr['obs_all'][i])      # (obs_dim,)
            act_list.append(tr['action_i'])
            ret_list.append(ret_by[i][t])
            adv_list.append(adv_by[i][t])
            lp_list.append(tr['log_prob'])

    # Stack; obs_pv has self at index i, but for value/policy we only use self
    obs_pv      = torch.stack(obs_pv_list)             # (B, N, obs_dim)
    up_pv       = torch.stack(up_list)                 # (B, N, act_dim)
    obs_self_pv = torch.stack(obs_self_list)           # (B, obs_dim)
    actions_pv  = torch.stack(act_list)                # (B, act_dim)
    returns_pv  = torch.tensor(ret_list,  dtype=torch.float32)
    adv_pv      = torch.tensor(adv_list,  dtype=torch.float32)
    lp_old_pv   = torch.tensor(lp_list,   dtype=torch.float32)

    adv_pv = (adv_pv - adv_pv.mean()) / (adv_pv.std() + 1e-8)

    # ── World model update (Eq 4) ──────────────────────────────────────────────
    opt_enc.zero_grad()
    opt_world.zero_grad()
    lw = loss_world_model(encoder, attn_w, world_model,
                          obs_wm, actions_wm, next_obs_wm, rewards_wm)
    lw.backward()
    opt_world.step()

    # Encoder grads from world-model survive here; policy adds its own below.
    opt_pol.zero_grad()
    lv = loss_value(encoder, attn_a, critic, obs_pv, up_pv, returns_pv)
    lp = loss_policy(encoder, attn_a, policy,
                     obs_self_pv, up_pv, actions_pv,
                     adv_pv, lp_old_pv)
    (lv + lp).backward()
    opt_pol.step()
    opt_enc.step()

    return {'wm': lw.item(), 'value': lv.item(), 'policy': lp.item()}


# ── Module factory ────────────────────────────────────────────────────────────

def make_nets(obs_dim: int, n_agents: int) -> dict:
    return {
        'encoder':     Encoder(obs_dim, EMBED_DIM),
        'attn_a':      AttentionModule(EMBED_DIM, ACTION_DIM),
        'attn_w':      AttentionModule(EMBED_DIM, EMBED_DIM + ACTION_DIM),
        'world_model': WorldModel(EMBED_DIM, n_agents, obs_dim),
        'policy':      Policy(EMBED_DIM, ACTION_DIM),
        'critic':      Critic(EMBED_DIM),
    }


def make_optimizers(nets: dict):
    opt_enc = optim.Adam(nets['encoder'].parameters(), lr=LR_ENC)
    opt_world = optim.Adam(
        list(nets['attn_w'].parameters()) +
        list(nets['world_model'].parameters()),
        lr=LR_WORLD,
    )
    opt_pol = optim.Adam(
        list(nets['attn_a'].parameters()) +
        list(nets['policy'].parameters()) +
        list(nets['critic'].parameters()),
        lr=LR_POL,
    )
    return opt_enc, opt_world, opt_pol


def load_from_scripts(nets: dict, weights_dir: str) -> None:
    """Warm-start nn.Module weights from existing TorchScript .pt files."""
    mapping = [
        ('encoder',     'encoder.pt'),
        ('attn_a',      'attn_a.pt'),
        ('attn_w',      'attn_w.pt'),
        ('world_model', 'world_model.pt'),
        ('critic',      'critic.pt'),
    ]
    for key, fname in mapping:
        path = os.path.join(weights_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            scripted = torch.jit.load(path)
            nets[key].load_state_dict(
                {k: v for k, v in scripted.state_dict().items()}, strict=False)
        except Exception as e:
            print(f'  warn: could not load {fname}: {e}')

    # Policy is exported as ScriptablePolicy; recover mean_layer + log_std
    pol_path = os.path.join(weights_dir, 'policy.pt')
    if os.path.exists(pol_path):
        try:
            scripted = torch.jit.load(pol_path)
            sd = scripted.state_dict()
            # ScriptablePolicy has mean_layer.{weight,bias} and log_std
            nets['policy'].mean_layer.weight.data.copy_(sd['mean_layer.weight'])
            nets['policy'].mean_layer.bias.data.copy_(sd['mean_layer.bias'])
            nets['policy'].log_std.data.copy_(sd['log_std'])
        except Exception as e:
            print(f'  warn: could not load policy.pt: {e}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Battleship MAPPO trainer — SeqComm paper Eqs 2-4')
    parser.add_argument('weights_dir',
                        help='Directory for .pt files, weights.bin, sentinel files')
    parser.add_argument('--init', action='store_true',
                        help='Bootstrap fresh random weights and exit (run before C++)')
    parser.add_argument('--obs-dim', type=int, default=OBS_DIM,
                        help=f'Observation dim (default {OBS_DIM} for sight_range=4)')
    parser.add_argument('--n-agents', type=int, default=N_AGENTS)
    args = parser.parse_args()

    weights_dir = args.weights_dir
    obs_dim     = args.obs_dim
    n_agents    = args.n_agents

    nets = make_nets(obs_dim, n_agents)

    if args.init:
        os.makedirs(weights_dir, exist_ok=True)
        save_torchscript(nets, weights_dir, obs_dim, n_agents)
        save_weights_bin(nets, weights_dir)
        print('Initial weights written. Start the C++ sim now.')
        return

    # Warm-start if weights already exist
    load_from_scripts(nets, weights_dir)
    opt_enc, opt_world, opt_pol = make_optimizers(nets)

    traj_bin   = os.path.join(weights_dir, 'traj.bin')
    traj_ready = os.path.join(weights_dir, 'traj.ready')
    traj_done  = os.path.join(weights_dir, 'traj.done')
    wts_ready  = os.path.join(weights_dir, 'weights.ready')

    print(f'train_battleship: watching {weights_dir}/')
    print(f'  obs_dim={obs_dim}  embed_dim={EMBED_DIM}  '
          f'action_dim={ACTION_DIM}  n_agents={n_agents}')
    print('Waiting for C++ sim…  (run: battleship-sim weights_bs/ --mode seqcomm)\n')

    # Boss policy is pure C++ heuristic — no neural training needed.
    # Agent encoder, attention, policy, critic, and world model are updated here.

    ep = 0
    reward_hist: list[float] = []

    while True:
        while not os.path.exists(traj_ready):
            if os.path.exists(traj_done):
                print(f'\nC++ sim finished after {ep} episodes.')
                try:
                    os.remove(traj_done)
                except OSError:
                    pass
                return
            time.sleep(POLL_INTERVAL)

        t0 = time.time()

        try:
            os.remove(traj_ready)
        except OSError:
            pass

        transitions, n_ag, obs_d, act_d = read_trajectory(traj_bin)
        if n_ag != n_agents or obs_d != obs_dim:
            print(f'  warn: traj dims mismatch (n_agents={n_ag}, obs_dim={obs_d}); skipping')
            open(wts_ready, 'w').close()
            continue

        losses = update_episode(transitions, n_agents, nets,
                                opt_enc, opt_world, opt_pol)

        ep_reward = sum(tr['reward'] for tr in transitions if tr['agent_id'] == 0)
        reward_hist.append(ep_reward)

        # Write binary blob (fast, no TorchScript tracing overhead per episode)
        save_weights_bin(nets, weights_dir)
        open(wts_ready, 'w').close()

        elapsed = time.time() - t0
        if ep % LOG_EVERY == 0:
            recent = reward_hist[-LOG_EVERY:]
            avg = sum(recent) / len(recent)
            print(
                f'ep {ep:4d} | R={ep_reward:+7.2f}  avg={avg:+7.2f} | '
                f'wm={losses["wm"]:.4f}  '
                f'v={losses["value"]:.4f}  '
                f'π={losses["policy"]:.4f}  '
                f'({elapsed:.2f}s)'
            )
        ep += 1


if __name__ == '__main__':
    # Allow running as a script from the repo root without installing the package.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
