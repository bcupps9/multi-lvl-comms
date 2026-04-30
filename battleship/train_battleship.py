"""
train_battleship.py — Python side of the C++ battleship training loop.

Collaborative Battleship: N_a agent ships vs N_b boss ships on an M×M grid.
The boss uses a fixed C++ heuristic (no neural training). We train agent
policy and world model only.

Architecture (Appendix D.2 of SeqComm paper, with a spatial battleship encoder):
    Encoder    : Conv2d stack over local grid + scalar head  (obs_dim → embed_dim)
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
OBS_DIM    = 79    # (2*2+1)^2 * 3 + 4  — local 5x5 patch * 3 channels + 4 scalars
ACTION_DIM = 3     # (move_dir, fire_dr, fire_dc)
EMBED_DIM  = 64

GAMMA       = 0.99
LAM         = 0.95
CLIP_EPS    = 0.2
ENTROPY_COEF = 0.01   # encourages exploration; prevents policy collapse
LR_ENC      = 1e-4
LR_WORLD    = 3e-4
LR_POL      = 1e-3    # higher than WM — policy needs stronger signal
GRAD_CLIP   = 1.0

LOG_EVERY    = 20
POLL_INTERVAL = 0.05  # seconds

# ── Neural modules ────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    e(o): raw observation → hidden state h.

    Battleship observations are a local spatial patch, not an unordered feature
    vector: 9x9 cells x [own, ally, boss] plus two scalars by default.  The conv
    stack lets the policy learn "boss at offset (dr, dc)" patterns directly.
    """

    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.channels = 3
        self.scalar_dim = 4   # hp, step, row, col

        spatial_dim = obs_dim - self.scalar_dim
        patch_cells = spatial_dim // self.channels
        patch = math.isqrt(patch_cells)
        if spatial_dim <= 0 or spatial_dim % self.channels != 0 or patch * patch != patch_cells:
            raise ValueError(
                f'Battleship obs_dim={obs_dim} is not patch*patch*3 + {self.scalar_dim}')

        self.patch = patch
        self.spatial_dim = spatial_dim

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * patch * patch + self.scalar_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (obs_dim), (B, obs_dim), or (B, N, obs_dim)
        original_dim = obs.dim()
        if original_dim == 1:
            flat = obs.unsqueeze(0)
        elif original_dim == 2:
            flat = obs
        else:
            flat = obs.reshape(-1, self.obs_dim)

        spatial = flat[:, :self.spatial_dim]
        scalars = flat[:, self.spatial_dim:]
        x = spatial.reshape(-1, self.patch, self.patch, self.channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        h = self.conv(x)
        h = torch.cat([h, scalars], dim=-1)
        h = self.head(h)

        if original_dim == 1:
            return h.squeeze(0)
        if original_dim == 2:
            return h
        return h.reshape(obs.size(0), obs.size(1), self.embed_dim)


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


class CommGate(nn.Module):
    """
    Experiment 2 comm gate: h → scalar logit.

    σ(logit) = P(agent chooses to communicate this step).
    Trained via REINFORCE using per-episode return as signal.
    Saved as comm_gate.pt alongside the main TorchScript files.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Linear(embed_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)  # scalar logit


# ── Comm gate sidecar I/O (Experiment 2) ─────────────────────────────────────

COMM_GATE_MAGIC = 0x43474154  # "CGAT"

def read_comm_gate_sidecar(path: str):
    """
    Read comm_gate_sidecar.bin written by battleship_sim.cc.

    Returns (decisions, ep_reward) where decisions is a list of
    {'did_comm': bool, 'gate_logit': float} dicts (one per negotiation step).
    """
    with open(path, 'rb') as f:
        magic, n, embed_dim = struct.unpack('<3I', f.read(12))
        if magic != COMM_GATE_MAGIC:
            raise ValueError(f'Bad comm_gate_sidecar magic: {magic:#x}')
        ep_reward = struct.unpack('<f', f.read(4))[0]
        decisions = []
        for _ in range(n):
            did_comm, logit = struct.unpack('<if', f.read(8))
            h_bytes = f.read(embed_dim * 4)
            h = list(struct.unpack(f'<{embed_dim}f', h_bytes)) if embed_dim else None
            decisions.append({'did_comm': bool(did_comm),
                              'gate_logit': logit,
                              'h': h})
    return decisions, ep_reward


def update_comm_gate(comm_gate: 'CommGate', opt_comm, decisions: list,
                     ep_reward: float, comm_penalty: float,
                     reward_baseline: float = 0.0) -> dict:
    """
    REINFORCE update for the comm gate.

    Signal for each step:
        did_comm  → net reward = ep_reward - comm_penalty * n_comm_steps - baseline
        no-comm   → net reward = ep_reward - baseline

    This encourages the gate to comm only when the expected ordering benefit
    exceeds the penalty.
    """
    if not decisions:
        return {}

    n_comm = sum(1 for d in decisions if d['did_comm'])
    ep_net = ep_reward - comm_penalty * n_comm - reward_baseline

    # Re-run the Python CommGate forward pass so grad_fn exists.  The sidecar
    # stores the C++ hidden states but not their autograd history, so this is
    # the point where gradients reconnect to the gate parameters.
    n_steps = len(decisions)
    param = next(comm_gate.parameters())
    h_rows = [d.get('h') for d in decisions]
    if all(h is not None and len(h) == EMBED_DIM for h in h_rows):
        h_t = torch.tensor(h_rows, dtype=param.dtype, device=param.device)
    else:
        h_t = torch.zeros(n_steps, EMBED_DIM, dtype=param.dtype, device=param.device)
    logits = comm_gate(h_t)                          # (n_steps,) with grad_fn

    did_comm_t = torch.tensor([float(d['did_comm']) for d in decisions],
                               dtype=param.dtype, device=param.device)

    dist = torch.distributions.Bernoulli(logits=logits)
    log_probs = dist.log_prob(did_comm_t)

    loss = -(log_probs * ep_net).mean()
    opt_comm.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(comm_gate.parameters(), 1.0)
    opt_comm.step()

    return {
        'comm_gate_loss':  loss.item(),
        'comm_rate':       n_comm / max(1, len(decisions)),
        'ep_net_reward':   ep_net,
    }


# ── Weight I/O ────────────────────────────────────────────────────────────────

# Must match the order read by LibtorchNeuralModels::update_from_blob() in C++.
MODULE_ORDER = ['encoder', 'attn_a', 'attn_w', 'world_model', 'policy', 'critic']


def save_torchscript(nets: dict, weights_dir: str, obs_dim: int, n_agents: int) -> None:
    """Export all modules as TorchScript .pt files + config.json."""
    os.makedirs(weights_dir, exist_ok=True)

    torch.jit.script(nets['encoder']).save(os.path.join(weights_dir, 'encoder.pt'))
    torch.jit.script(nets['encoder_wm']).save(os.path.join(weights_dir, 'encoder_wm.pt'))
    torch.jit.script(nets['attn_a']).save(os.path.join(weights_dir, 'attn_a.pt'))
    torch.jit.script(nets['attn_w']).save(os.path.join(weights_dir, 'attn_w.pt'))
    torch.jit.script(nets['world_model']).save(os.path.join(weights_dir, 'world_model.pt'))
    # Policy must be wrapped before scripting (Normal dist is not scriptable)
    torch.jit.script(ScriptablePolicy(nets['policy'])).save(
        os.path.join(weights_dir, 'policy.pt'))
    torch.jit.script(nets['critic']).save(os.path.join(weights_dir, 'critic.pt'))
    if 'comm_gate' in nets:
        torch.jit.script(nets['comm_gate']).save(os.path.join(weights_dir, 'comm_gate.pt'))

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

        for i in range(n_agents):
            rec = step[i]
            assembled.append({
                'agent_id':    i,
                'obs_all':     obs_all,         # (n_agents, obs_dim)
                'action_i':    torch.tensor(rec['action']),
                'actions_all': actions_all,      # (n_agents, action_dim)
                'up_pad_i':    torch.tensor(rec['up_pad']),   # (n_agents, action_dim)
                'next_obs_all': next_obs_all,
                'reward':      rec['reward'],    # per-agent: hit credit goes to the shooter
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
               obs_self, up_pv, returns_pv):
    """Eq 2: MSE value loss. Detach encoder so value gradient never shapes the policy encoder."""
    with torch.no_grad():
        h = encoder(obs_self)              # (B, embed) — frozen features for value path
    ctx = attn_a(h, up_pv)                 # (B, embed)
    v = critic(ctx)                        # (B,)
    diff = returns_pv - v
    return (diff * diff).mean()


def loss_policy(encoder, attn_a, policy,
                obs_self, up_pv, actions_taken,
                advantages, log_probs_old, clip_eps=CLIP_EPS,
                entropy_coef=ENTROPY_COEF):
    """Eq 3: PPO-clip surrogate + entropy bonus."""
    h = encoder(obs_self)                  # (B, embed)
    ctx = attn_a(h, up_pv)               # (B, embed)
    dist = policy(ctx)
    log_probs = dist.log_prob(actions_taken).sum(dim=-1)   # (B,)
    entropy = dist.entropy().sum(dim=-1).mean()             # scalar
    ratio = (log_probs - log_probs_old).exp()
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    is_clipped = (ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)
    surrogate = -torch.min(ratio * advantages, clipped * advantages).mean()
    extras = {
        'entropy': entropy.item(),
        'clip_frac': is_clipped.float().mean().item(),
        'ratio_mean': ratio.mean().item(),
        'ratio_max': ratio.max().item(),
        'policy_mean': dist.mean.detach(),   # (B, action_dim) for stats
    }
    return surrogate - entropy_coef * entropy, extras


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


# ── Update ────────────────────────────────────────────────────────────────────

PPO_EPOCHS = 8   # policy/value epochs per collected batch


def update_batch(episodes, n_agents, nets,
                 opt_enc_pol, opt_enc_wm, opt_world, opt_pol,
                 entropy_coef: float = ENTROPY_COEF,
                 grad_clip: float = GRAD_CLIP) -> dict:
    encoder     = nets['encoder']       # policy path only
    encoder_wm  = nets['encoder_wm']    # world-model path only
    attn_a      = nets['attn_a']
    attn_w      = nets['attn_w']
    world_model = nets['world_model']
    policy      = nets['policy']
    critic      = nets['critic']

    # World-model batch: one record per timestep, across all agents.
    obs_wm_list, actions_wm_list, next_obs_wm_list, rewards_wm_list = [], [], [], []

    # Policy/value batch: one record per agent per timestep.
    up_list, obs_self_list = [], []
    act_list, ret_list, adv_list, lp_list = [], [], [], []

    for transitions in episodes:
        # GAE must reset at episode boundaries; batching should reduce variance
        # without pretending the last state of one episode leads into the next.
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

        for tr in transitions:
            if tr['agent_id'] == 0:
                obs_wm_list.append(tr['obs_all'])
                actions_wm_list.append(tr['actions_all'])
                next_obs_wm_list.append(tr['next_obs_all'])
                rewards_wm_list.append(tr['reward'])

        for t in range(T):
            for i in range(n_agents):
                tr = transitions[t * n_agents + i]
                up_list.append(tr['up_pad_i'])          # (N, act_dim) — upper actions
                obs_self_list.append(tr['obs_all'][i])  # (obs_dim,)
                act_list.append(tr['action_i'])
                ret_list.append(ret_by[i][t])
                adv_list.append(adv_by[i][t])
                lp_list.append(tr['log_prob'])

    obs_wm      = torch.stack(obs_wm_list)              # (T_total, N, obs_dim)
    actions_wm  = torch.stack(actions_wm_list)          # (T_total, N, act_dim)
    next_obs_wm = torch.stack(next_obs_wm_list)         # (T_total, N, obs_dim)
    rewards_wm  = torch.tensor(rewards_wm_list, dtype=torch.float32)

    up_pv       = torch.stack(up_list)                 # (B, N, act_dim)
    obs_self_pv = torch.stack(obs_self_list)           # (B, obs_dim)
    actions_pv  = torch.stack(act_list)                # (B, act_dim)
    returns_pv  = torch.tensor(ret_list,  dtype=torch.float32)
    adv_pv      = torch.tensor(adv_list,  dtype=torch.float32)
    lp_old_pv   = torch.tensor(lp_list,   dtype=torch.float32)

    # Capture pre-normalisation advantage stats for diagnostics.
    adv_raw = adv_pv.clone()
    adv_pv = (adv_pv - adv_pv.mean()) / (adv_pv.std() + 1e-8)

    def _grad_norm(params):
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
        return total ** 0.5

    # ── World model update (Eq 4) — uses encoder_wm only, no shared grad with policy ──
    opt_enc_wm.zero_grad()
    opt_world.zero_grad()
    lw = loss_world_model(encoder_wm, attn_w, world_model,
                          obs_wm, actions_wm, next_obs_wm, rewards_wm)
    lw.backward()
    gn_wm  = _grad_norm(list(attn_w.parameters()) + list(world_model.parameters()))
    gn_enc_wm = _grad_norm(encoder_wm.parameters())
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(
            list(encoder_wm.parameters()) +
            list(attn_w.parameters()) +
            list(world_model.parameters()),
            grad_clip,
        )
    opt_world.step()
    opt_enc_wm.step()

    # Policy/value: PPO_EPOCHS passes over the same batch.
    # log_probs_old are frozen from trajectory collection (lp_old_pv).
    # Re-computing log_probs each epoch lets the ratio diverge from 1 as
    # the policy updates, so the clip actually fires in later epochs.
    # Both opt_enc_pol and opt_pol are zeroed at the start of every epoch so
    # gradients never accumulate across epochs (fixes the previous explosion in
    # gn_enc_pol caused by stepping opt_enc without zeroing between epochs).
    gn_pol = gn_critic = gn_enc_pol = 0.0
    lv_last = lp_last = 0.0
    pol_extras_last = {}
    for epoch in range(PPO_EPOCHS):
        opt_enc_pol.zero_grad()
        opt_pol.zero_grad()
        lv = loss_value(encoder, attn_a, critic, obs_self_pv, up_pv, returns_pv)
        lp, pol_extras = loss_policy(encoder, attn_a, policy,
                                     obs_self_pv, up_pv, actions_pv,
                                     adv_pv, lp_old_pv,
                                     entropy_coef=entropy_coef)
        (lv + lp).backward()
        gn_pol     = _grad_norm(list(policy.parameters()) + list(attn_a.parameters()))
        gn_critic  = _grad_norm(critic.parameters())
        gn_enc_pol = _grad_norm(encoder.parameters())
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) +
                list(attn_a.parameters()) +
                list(policy.parameters()) +
                list(critic.parameters()),
                grad_clip,
            )
        opt_pol.step()
        opt_enc_pol.step()
        lv_last, lp_last, pol_extras_last = lv.item(), lp.item(), pol_extras

    # Value calibration: compare mean predicted V to mean return.
    with torch.no_grad():
        h_cal   = encoder(obs_self_pv)
        ctx_cal = attn_a(h_cal, up_pv)
        v_pred  = critic(ctx_cal)

    pm = pol_extras_last['policy_mean']
    return {
        'wm':            lw.item(),
        'value':         lv_last,
        'policy':        lp_last,
        'entropy':       pol_extras_last['entropy'],
        'clip_frac':     pol_extras_last['clip_frac'],
        'ratio_mean':    pol_extras_last['ratio_mean'],
        'ratio_max':     pol_extras_last['ratio_max'],
        'adv_mean_raw':  adv_raw.mean().item(),
        'adv_std_raw':   adv_raw.std().item(),
        'adv_max_raw':   adv_raw.max().item(),
        'adv_min_raw':   adv_raw.min().item(),
        'returns_mean':  returns_pv.mean().item(),
        'returns_std':   returns_pv.std().item(),
        'v_pred_mean':   v_pred.mean().item(),
        'v_pred_std':    v_pred.std().item(),
        'gn_wm':         gn_wm,
        'gn_enc_wm':     gn_enc_wm,
        'gn_pol':        gn_pol,
        'gn_critic':     gn_critic,
        'gn_enc_pol':    gn_enc_pol,
        'pmean_mean':    pm.mean().item(),
        'pmean_std':     pm.std().item(),
        'batch_steps':   len(adv_pv),
        'ppo_epochs':    PPO_EPOCHS,
    }


def update_episode(transitions, n_agents, nets,
                   opt_enc_pol, opt_enc_wm, opt_world, opt_pol) -> dict:
    return update_batch([transitions], n_agents, nets,
                        opt_enc_pol, opt_enc_wm, opt_world, opt_pol)


# ── Module factory ────────────────────────────────────────────────────────────

def make_nets(obs_dim: int, n_agents: int, use_comm_gate: bool = False) -> dict:
    nets = {
        'encoder':     Encoder(obs_dim, EMBED_DIM),   # policy path — saved to encoder.pt, used by C++
        'encoder_wm':  Encoder(obs_dim, EMBED_DIM),   # world-model path — Python-only, never sent to C++
        'attn_a':      AttentionModule(EMBED_DIM, ACTION_DIM),
        'attn_w':      AttentionModule(EMBED_DIM, EMBED_DIM + ACTION_DIM),
        'world_model': WorldModel(EMBED_DIM, n_agents, obs_dim),
        'policy':      Policy(EMBED_DIM, ACTION_DIM),
        'critic':      Critic(EMBED_DIM),
    }
    if use_comm_gate:
        nets['comm_gate'] = CommGate(EMBED_DIM)
    return nets


def make_optimizers(nets: dict,
                    lr_enc: float = LR_ENC,
                    lr_world: float = LR_WORLD,
                    lr_policy: float = LR_POL):
    # Two separate encoder optimizers — no shared gradient paths between WM and policy.
    opt_enc_pol = optim.Adam(nets['encoder'].parameters(), lr=lr_enc)
    opt_enc_wm  = optim.Adam(nets['encoder_wm'].parameters(), lr=lr_enc)
    opt_world = optim.Adam(
        list(nets['attn_w'].parameters()) +
        list(nets['world_model'].parameters()),
        lr=lr_world,
    )
    opt_pol = optim.Adam(
        list(nets['attn_a'].parameters()) +
        list(nets['policy'].parameters()) +
        list(nets['critic'].parameters()),
        lr=lr_policy,
    )
    opt_comm = None
    if 'comm_gate' in nets:
        opt_comm = optim.Adam(nets['comm_gate'].parameters(), lr=lr_policy)
    return opt_enc_pol, opt_enc_wm, opt_world, opt_pol, opt_comm


def load_from_scripts(nets: dict, weights_dir: str) -> None:
    """Warm-start nn.Module weights from existing TorchScript .pt files."""
    mapping = [
        ('encoder',     'encoder.pt'),
        ('encoder_wm',  'encoder_wm.pt'),
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


def write_trainer_meta(f, args, obs_dim: int, n_agents: int) -> None:
    if f is None:
        return
    f.write(json.dumps({
        '_meta': {
            'kind': 'battleship_trainer',
            'obs_dim': obs_dim,
            'n_agents': n_agents,
            'embed_dim': EMBED_DIM,
            'action_dim': ACTION_DIM,
            'update_every': max(1, args.update_every),
            'entropy_coef': args.entropy_coef,
            'grad_clip': args.grad_clip,
            'lr_enc': args.lr_enc,
            'lr_world': args.lr_world,
            'lr_policy': args.lr_policy,
        }
    }) + '\n')
    f.flush()


def write_trainer_update(f, update_idx: int, ep: int, batch_size: int,
                         losses: dict, policy: Policy, elapsed: float) -> None:
    if f is None:
        return
    log_std = policy.log_std.detach().cpu()
    row = {
        'update':             update_idx,
        'last_episode':       ep,
        'batch_size':         batch_size,
        'batch_steps':        losses.get('batch_steps', 0),
        # Losses
        'world_model_loss':   losses['wm'],
        'value_loss':         losses['value'],
        'policy_loss':        losses['policy'],
        'entropy':            losses['entropy'],
        # PPO health
        'clip_frac':          losses.get('clip_frac', 0.0),
        'ratio_mean':         losses.get('ratio_mean', 1.0),
        'ratio_max':          losses.get('ratio_max', 1.0),
        # Advantage stats (pre-normalisation) — near-zero std = no learning signal
        'adv_mean_raw':       losses.get('adv_mean_raw', 0.0),
        'adv_std_raw':        losses.get('adv_std_raw', 0.0),
        'adv_max_raw':        losses.get('adv_max_raw', 0.0),
        'adv_min_raw':        losses.get('adv_min_raw', 0.0),
        # Value calibration — v_pred_mean should track returns_mean
        'returns_mean':       losses.get('returns_mean', 0.0),
        'returns_std':        losses.get('returns_std', 0.0),
        'v_pred_mean':        losses.get('v_pred_mean', 0.0),
        'v_pred_std':         losses.get('v_pred_std', 0.0),
        # Gradient norms — near-zero = dead module, huge = exploding
        'gn_wm':              losses.get('gn_wm', 0.0),
        'gn_enc_wm':          losses.get('gn_enc_wm', 0.0),
        'gn_pol':             losses.get('gn_pol', 0.0),
        'gn_critic':          losses.get('gn_critic', 0.0),
        'gn_enc_pol':         losses.get('gn_enc_pol', 0.0),
        # Policy output statistics
        'policy_log_std_mean': float(log_std.mean()),
        'policy_log_std_min':  float(log_std.min()),
        'policy_log_std_max':  float(log_std.max()),
        'policy_std_mean':     float(log_std.exp().mean()),
        'pmean_mean':          losses.get('pmean_mean', 0.0),
        'pmean_std':           losses.get('pmean_std', 0.0),
        'elapsed_sec':         elapsed,
    }
    if 'comm_gate_loss' in losses:
        row.update({
            'comm_gate_loss': losses['comm_gate_loss'],
            'comm_gate_rate': losses.get('comm_rate', 0.0),
            'comm_gate_net_reward': losses.get('ep_net_reward', 0.0),
        })
    f.write(json.dumps(row) + '\n')
    f.flush()


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
    parser.add_argument('--update-every', type=int, default=1,
                        help='Accumulate this many complete episodes per optimizer update')
    parser.add_argument('--entropy-coef', type=float, default=ENTROPY_COEF)
    parser.add_argument('--grad-clip', type=float, default=GRAD_CLIP)
    parser.add_argument('--lr-enc', type=float, default=LR_ENC)
    parser.add_argument('--lr-world', type=float, default=LR_WORLD)
    parser.add_argument('--lr-policy', type=float, default=LR_POL)
    parser.add_argument('--trainer-log',
                        help='Optional JSONL file for Python update diagnostics')
    # Experiment 2: comm gate
    parser.add_argument('--comm-gate', action='store_true',
                        help='Enable optional comm gate (Experiment 2)')
    parser.add_argument('--comm-penalty', type=float, default=0.0,
                        help='Reward cost for communicating (Experiment 2)')
    args = parser.parse_args()

    weights_dir = args.weights_dir
    obs_dim     = args.obs_dim
    n_agents    = args.n_agents
    update_every = max(1, args.update_every)

    use_comm_gate = args.comm_gate
    comm_penalty  = args.comm_penalty

    nets = make_nets(obs_dim, n_agents, use_comm_gate=use_comm_gate)

    if args.init:
        os.makedirs(weights_dir, exist_ok=True)
        save_torchscript(nets, weights_dir, obs_dim, n_agents)
        save_weights_bin(nets, weights_dir)
        print('Initial weights written. Start the C++ sim now.')
        if use_comm_gate:
            print('  comm_gate.pt included (Experiment 2 mode).')
        return

    # Warm-start if weights already exist
    load_from_scripts(nets, weights_dir)
    opt_enc_pol, opt_enc_wm, opt_world, opt_pol, opt_comm = make_optimizers(
        nets,
        lr_enc=args.lr_enc,
        lr_world=args.lr_world,
        lr_policy=args.lr_policy,
    )

    traj_bin   = os.path.join(weights_dir, 'traj.bin')
    traj_ready = os.path.join(weights_dir, 'traj.ready')
    traj_done  = os.path.join(weights_dir, 'traj.done')
    wts_ready  = os.path.join(weights_dir, 'weights.ready')

    trainer_log = None
    if args.trainer_log:
        os.makedirs(os.path.dirname(args.trainer_log) or '.', exist_ok=True)
        trainer_log = open(args.trainer_log, 'w')
        write_trainer_meta(trainer_log, args, obs_dim, n_agents)

    print(f'train_battleship: watching {weights_dir}/')
    print(f'  obs_dim={obs_dim}  embed_dim={EMBED_DIM}  '
          f'action_dim={ACTION_DIM}  n_agents={n_agents}')
    print(f'  update_every={update_every}  entropy_coef={args.entropy_coef}  '
          f'grad_clip={args.grad_clip}  lr_policy={args.lr_policy}')
    print('Waiting for C++ sim…  (run: battleship-sim weights_bs/ --mode seqcomm)\n')

    # Boss policy is pure C++ heuristic — no neural training needed.
    # Agent encoder, attention, policy, critic, and world model are updated here.

    ep = 0
    update_idx = 0
    reward_hist: list[float] = []
    batch: list[list[dict]] = []
    last_losses = {'wm': float('nan'), 'value': float('nan'),
                   'policy': float('nan'), 'entropy': float('nan')}

    # Comm gate sidecar accumulator (Experiment 2)
    comm_sidecar_path  = os.path.join(weights_dir, 'comm_gate_sidecar.bin')
    comm_sidecar_batch: list[tuple] = []   # [(decisions, ep_reward)]
    comm_reward_hist: list[float] = []     # running baseline for REINFORCE

    while True:
        while not os.path.exists(traj_ready):
            if os.path.exists(traj_done):
                if batch:
                    t_final = time.time()
                    batch_size = len(batch)
                    losses = update_batch(batch, n_agents, nets,
                                          opt_enc_pol, opt_enc_wm, opt_world, opt_pol,
                                          entropy_coef=args.entropy_coef,
                                          grad_clip=args.grad_clip)
                    if use_comm_gate and opt_comm is not None and comm_sidecar_batch:
                        baseline = (sum(comm_reward_hist[-50:]) / max(1, len(comm_reward_hist[-50:])))
                        cg_stats = {}
                        for decisions, ep_rew in comm_sidecar_batch:
                            cg_stats = update_comm_gate(nets['comm_gate'], opt_comm, decisions,
                                                        ep_rew, comm_penalty, baseline)
                        if cg_stats:
                            losses.update(cg_stats)
                        comm_sidecar_batch.clear()
                        torch.jit.script(nets['comm_gate']).save(
                            os.path.join(weights_dir, 'comm_gate.pt'))
                    save_weights_bin(nets, weights_dir)
                    write_trainer_update(
                        trainer_log, update_idx, ep - 1, batch_size,
                        losses, nets['policy'], time.time() - t_final)
                    update_idx += 1
                    print(f'\nFinal partial update on {len(batch)} buffered episode(s): '
                          f'wm={losses["wm"]:.4f} v={losses["value"]:.4f} '
                          f'π={losses["policy"]:+.6f}')
                print(f'\nC++ sim finished after {ep} episodes.')
                if trainer_log is not None:
                    trainer_log.close()
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

        # Read comm gate sidecar (Experiment 2) — written alongside traj.bin.
        if use_comm_gate and os.path.exists(comm_sidecar_path):
            try:
                decisions, sidecar_reward = read_comm_gate_sidecar(comm_sidecar_path)
                comm_sidecar_batch.append((decisions, sidecar_reward))
                comm_reward_hist.append(sidecar_reward)
            except Exception as e:
                print(f'  warn: could not read comm_gate_sidecar: {e}')
            try:
                os.remove(comm_sidecar_path)
            except OSError:
                pass

        batch.append(transitions)
        did_update = len(batch) >= update_every
        if did_update:
            batch_size = len(batch)
            last_losses = update_batch(batch, n_agents, nets,
                                       opt_enc_pol, opt_enc_wm, opt_world, opt_pol,
                                       entropy_coef=args.entropy_coef,
                                       grad_clip=args.grad_clip)
            batch.clear()

            # Comm gate REINFORCE update (Experiment 2).
            if use_comm_gate and opt_comm is not None and comm_sidecar_batch:
                baseline = (sum(comm_reward_hist[-50:]) / max(1, len(comm_reward_hist[-50:])))
                cg_stats = {}
                for decisions, ep_rew in comm_sidecar_batch:
                    cg_stats = update_comm_gate(
                        nets['comm_gate'], opt_comm, decisions,
                        ep_rew, comm_penalty, baseline)
                comm_sidecar_batch.clear()
                # Persist updated comm gate for C++ to reload next episode.
                torch.jit.script(nets['comm_gate']).save(
                    os.path.join(weights_dir, 'comm_gate.pt'))
                if cg_stats:
                    last_losses.update(cg_stats)

        ep_reward = sum(tr['reward'] for tr in transitions if tr['agent_id'] == 0)
        reward_hist.append(ep_reward)

        # Write a fresh blob only when the optimizer actually changed weights.
        # On non-update episodes C++ reloads the previous blob and continues with
        # the same policy, giving us lower-variance multi-episode updates.
        elapsed = time.time() - t0
        if did_update:
            save_weights_bin(nets, weights_dir)
            write_trainer_update(
                trainer_log, update_idx, ep, batch_size,
                last_losses, nets['policy'], elapsed)
            update_idx += 1
        open(wts_ready, 'w').close()

        if ep % LOG_EVERY == 0:
            recent = reward_hist[-LOG_EVERY:]
            avg = sum(recent) / len(recent)
            clip  = last_losses.get('clip_frac', float('nan'))
            gn_p  = last_losses.get('gn_pol', float('nan'))
            gn_c  = last_losses.get('gn_critic', float('nan'))
            adv_s = last_losses.get('adv_std_raw', float('nan'))
            print(
                f'ep {ep:4d} | R={ep_reward:+7.2f}  avg={avg:+7.2f} | '
                f'wm={last_losses["wm"]:.4f}  '
                f'v={last_losses["value"]:.4f}  '
                f'π={last_losses["policy"]:+.6f}  '
                f'H={last_losses["entropy"]:.3f}  '
                f'clip={clip:.2f}  adv_std={adv_s:.3f}  '
                f'gn_π={gn_p:.3f}  gn_V={gn_c:.3f}  '
                f'batch={len(batch)}/{update_every}  '
                f'({elapsed:.2f}s)'
            )
        ep += 1


if __name__ == '__main__':
    # Allow running as a script from the repo root without installing the package.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
