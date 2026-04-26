"""
SeqComm MAPPO training losses (equations 2, 3, 4).

The C++ cotamer simulation collects trajectories; this module trains the
shared neural network weights on those trajectories.

Network parameters:
    theta_e  — observation encoder e
    theta_a  — attention module AM_a (negotiation + launching)
    theta_w  — attention module AM_w + world model M
    theta_v  — critic V
    theta_pi — policy π

All five modules are trained jointly from trajectory data D = {tau_k}.
"""

import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Modules ───────────────────────────────────────────────────────────────────

class ObservationEncoder(nn.Module):
    """e(o): encodes a single agent's raw observation -> hidden state h."""

    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (..., obs_dim) -> (..., embed_dim)
        return self.net(obs)


class AttentionModule(nn.Module):
    """
    AM_a or AM_w: multi-head attention over a set of messages.

    Used in two contexts:
      - AM_a: messages are neighbor hidden states (negotiation) or upper
              actions (launching); produces context for policy / critic.
      - AM_w: messages are (enc_obs, action) pairs; produces context for M.
    """

    def __init__(self, embed_dim: int, msg_dim: int, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim + msg_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, h_self: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        # h_self:   (batch, embed_dim)
        # messages: (batch, n_agents, msg_dim)
        # Expand h_self to match agent dimension, concat with messages
        n = messages.shape[1]
        h_exp = h_self.unsqueeze(1).expand(-1, n, -1)
        x = torch.cat([h_exp, messages], dim=-1)   # (batch, n, embed+msg)
        x = self.input_proj(x)                      # (batch, n, embed)
        # Index [0] instead of tuple-unpack so torch.jit.script is happy
        attn_out = self.attn(x, x, x)[0]           # (batch, n, embed)
        context = attn_out.mean(dim=1)              # (batch, embed)
        return self.out_proj(context)


class WorldModel(nn.Module):
    """M(theta_w): predicts (o'_all_flat, r) from AM_w context."""

    def __init__(self, embed_dim: int, n_agents: int, obs_dim: int):
        super().__init__()
        self.output_dim = n_agents * obs_dim + 1
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.output_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class Policy(nn.Module):
    """π(theta_pi): Gaussian policy over AM_a context."""

    def __init__(self, embed_dim: int, action_dim: int):
        super().__init__()
        self.mean = nn.Linear(embed_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    @torch.jit.ignore
    def forward(self, context: torch.Tensor):
        # Returns a Normal distribution for Python training.
        # Ignored by TorchScript — use sample() / log_prob_of() from C++.
        mean = self.mean(context)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @torch.jit.export
    def sample(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob). TorchScript-safe."""
        mean = self.mean(context)
        std  = self.log_std.exp().expand_as(mean)
        action = mean + std * torch.randn_like(mean)
        log_prob = (
            -0.5 * math.log(2.0 * math.pi)
            - std.log()
            - 0.5 * ((action - mean) / std).pow(2)
        ).sum(-1)
        return action, log_prob

    @torch.jit.export
    def log_prob_of(self, context: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
        """Log-probability of a given action under this policy."""
        mean = self.mean(context)
        std  = self.log_std.exp().expand_as(mean)
        return (
            -0.5 * math.log(2.0 * math.pi)
            - std.log()
            - 0.5 * ((action - mean) / std).pow(2)
        ).sum(-1)


class Critic(nn.Module):
    """V(theta_v): scalar value estimate from AM_a context."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context).squeeze(-1)


class ScriptablePolicy(nn.Module):
    """
    TorchScript-compatible wrapper around Policy.

    torch.distributions.Normal cannot be exported to TorchScript, so this
    module returns (mean, log_std_expanded) as plain tensors.  The C++ side
    uses these to sample:  action = mean + exp(log_std) * randn_like(mean)
    and to compute log_prob analytically.

    Usage:
        sp = ScriptablePolicy(policy)
        scripted = torch.jit.script(sp)
        scripted.save("weights/policy.pt")
    """

    def __init__(self, policy: "Policy"):
        super().__init__()
        self.mean_layer = policy.mean
        self.log_std = policy.log_std

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # context: (batch, embed_dim)
        # returns: mean (batch, action_dim), log_std (batch, action_dim)
        mean = self.mean_layer(context)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


# ── Loss functions ─────────────────────────────────────────────────────────────

def world_model_loss(
    encoder: ObservationEncoder,
    attn_w: AttentionModule,
    world_model: WorldModel,
    obs: torch.Tensor,           # (batch, n_agents, obs_dim)
    actions: torch.Tensor,       # (batch, n_agents, action_dim)
    next_obs: torch.Tensor,      # (batch, n_agents, obs_dim)
    rewards: torch.Tensor,       # (batch,)
) -> torch.Tensor:
    """
    Equation (4): L(theta_w) = (1/|S|) sum ||(o',r) - M(AM_w(e(o), a))||^2_2
    """
    batch, n_agents, obs_dim = obs.shape
    enc_obs = encoder(obs)                           # (batch, n_agents, embed)
    # For AM_w, messages = concat(enc_obs, actions) per agent
    messages = torch.cat([enc_obs, actions], dim=-1) # (batch, n_agents, embed+act)
    # Use mean of enc_obs as h_self for AM_w (world model has no single "self")
    h_self = enc_obs.mean(dim=1)                     # (batch, embed)
    context = attn_w(h_self, messages)               # (batch, embed)
    pred = world_model(context)                      # (batch, n_agents*obs_dim+1)

    target_obs = next_obs.reshape(batch, -1)         # (batch, n_agents*obs_dim)
    target = torch.cat([target_obs, rewards.unsqueeze(-1)], dim=-1)

    diff = target - pred
    return (diff * diff).sum(dim=-1).mean()


def value_loss(
    encoder: ObservationEncoder,
    attn_a: AttentionModule,
    critic: Critic,
    obs: torch.Tensor,           # (batch, n_agents, obs_dim)  self is obs[:,0]
    upper_actions: torch.Tensor, # (batch, n_upper, action_dim)
    returns: torch.Tensor,       # (batch,)  discounted R-hat
) -> torch.Tensor:
    """
    Equation (2): L(theta_v) = (1/KT) sum ||V(AM_a(e(o), a^upper)) - R-hat||^2_2
    """
    h_self = encoder(obs[:, 0])                      # (batch, embed)
    context = attn_a(h_self, upper_actions)          # (batch, embed)
    v_pred = critic(context)                         # (batch,)
    diff = returns - v_pred
    return (diff * diff).mean()


def ppo_loss(
    encoder: ObservationEncoder,
    attn_a: AttentionModule,
    policy: Policy,
    obs: torch.Tensor,           # (batch, obs_dim)  self obs only
    upper_actions: torch.Tensor, # (batch, n_upper, action_dim)
    actions_taken: torch.Tensor, # (batch, action_dim)
    advantages: torch.Tensor,    # (batch,)  GAE A_{pi_old}
    log_probs_old: torch.Tensor, # (batch,)
    clip_eps: float = 0.2,
    return_entropy: bool = False,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
    """
    Equation (3): L(theta_pi) = (1/KT) sum min(ratio*A, g(eps,A))
    where g(eps,A) = (1+eps)*A if A>=0 else (1-eps)*A

    If return_entropy=True, also returns mean policy entropy (for regularisation).
    All other callers use the default return_entropy=False so their signatures
    are unchanged.
    """
    h_self = encoder(obs)
    context = attn_a(h_self, upper_actions)
    dist = policy(context)
    log_probs = dist.log_prob(actions_taken).sum(dim=-1)

    ratio = (log_probs - log_probs_old).exp()
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    surrogate = torch.min(ratio * advantages, clipped * advantages)
    loss = -surrogate.mean()

    if return_entropy:
        entropy = dist.entropy().mean()
        return loss, entropy
    return loss


# ── Training loop ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_intention(
    agent_i: int,
    obs_all: list,          # n_agents tensors, each (obs_dim,)
    encoder: "ObservationEncoder",
    attn_a: "AttentionModule",
    attn_w: "AttentionModule",
    world_model_net: "WorldModel",
    policy: "Policy",
    critic: "Critic",
    H: int,
    F: int,
    n_agents: int,
    obs_dim: int,
    action_dim: int,
    gamma: float = 0.99,
) -> float:
    """
    Algorithm 5: estimate agent_i's future return by averaging F random
    orderings, each rolled out H steps with the world model.
    The critic bootstraps the terminal-state value (eq. in paper:
    v = 1/F * sum_f [sum_t gamma^t * r_t  +  gamma^H * V(s_H)]).
    """
    device = next(encoder.parameters()).device
    total = 0.0
    for _ in range(F):
        order = torch.randperm(n_agents).tolist()
        sim_obs = [o.clone() for o in obs_all]
        g = 1.0
        cum_r = 0.0
        for _ in range(H):
            sim_h = [encoder(o.unsqueeze(0)) for o in sim_obs]
            upper_so_far: list = []
            sim_actions: dict = {}
            for j in order:
                up_pad = torch.zeros(1, n_agents, action_dim, device=device)
                for l, ua in enumerate(upper_so_far):
                    up_pad[0, l] = ua
                ctx = attn_a(sim_h[j], up_pad)
                a = policy(ctx).sample().squeeze(0)
                sim_actions[j] = a
                upper_so_far.append(a)
            obs_t  = torch.stack(sim_obs, 0).unsqueeze(0)
            enc_t  = encoder(obs_t)
            acts_t = torch.stack(
                [sim_actions[j] for j in range(n_agents)], 0
            ).unsqueeze(0)
            msgs_w = torch.cat([enc_t, acts_t], dim=-1)
            ctx_w  = attn_w(enc_t.mean(1), msgs_w)
            pred   = world_model_net(ctx_w).squeeze(0)
            cum_r += g * pred[-1].item()
            g     *= gamma
            nxt = pred[:-1].detach()
            sim_obs = [nxt[j * obs_dim:(j + 1) * obs_dim] for j in range(n_agents)]
        h_fin  = encoder(sim_obs[agent_i].unsqueeze(0))
        up_fin = torch.zeros(1, n_agents, action_dim, device=device)
        cum_r += g * critic(attn_a(h_fin, up_fin)).item()
        total += cum_r
    return total / F


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Returns (advantages, returns) via Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns


def train_step(
    # modules
    encoder: ObservationEncoder,
    attn_a: AttentionModule,
    attn_w: AttentionModule,
    world_model_net: WorldModel,
    policy: Policy,
    critic: Critic,
    # trajectory batch (already tensorified)
    obs: torch.Tensor,
    actions: torch.Tensor,
    upper_actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    log_probs_old: torch.Tensor,
    # optimizers
    opt_world: torch.optim.Optimizer,
    opt_policy: torch.optim.Optimizer,
    clip_eps: float = 0.2,
) -> dict:
    # World model update — eq (4)
    opt_world.zero_grad()
    lw = world_model_loss(encoder, attn_w, world_model_net,
                          obs, actions, next_obs, rewards)
    lw.backward()
    opt_world.step()

    # Value + policy update — eqs (2) & (3)
    opt_policy.zero_grad()
    lv = value_loss(encoder, attn_a, critic,
                    obs, upper_actions, returns)
    lp = ppo_loss(encoder, attn_a, policy,
                  obs[:, 0], upper_actions, actions[:, 0],
                  advantages, log_probs_old, clip_eps)
    (lv + lp).backward()
    opt_policy.step()

    return {"world_model": lw.item(), "value": lv.item(), "policy": lp.item()}


# ── Weight export ──────────────────────────────────────────────────────────────

def save_weights(
    encoder: ObservationEncoder,
    attn_a: AttentionModule,
    attn_w: AttentionModule,
    world_model_net: WorldModel,
    policy: Policy,
    critic: Critic,
    weights_dir: str = "weights",
) -> None:
    """
    Export all six modules as TorchScript (.pt) files loadable by libtorch.
    Call this after training; the C++ seqcomm-sim-trained binary reads them.
    """
    os.makedirs(weights_dir, exist_ok=True)
    torch.jit.script(encoder).save(os.path.join(weights_dir, "encoder.pt"))
    torch.jit.script(attn_a).save(os.path.join(weights_dir, "attn_a.pt"))
    torch.jit.script(attn_w).save(os.path.join(weights_dir, "attn_w.pt"))
    torch.jit.script(world_model_net).save(os.path.join(weights_dir, "world_model.pt"))
    torch.jit.script(policy).save(os.path.join(weights_dir, "policy.pt"))
    torch.jit.script(critic).save(os.path.join(weights_dir, "critic.pt"))
    print(f"Weights saved → {os.path.abspath(weights_dir)}")
