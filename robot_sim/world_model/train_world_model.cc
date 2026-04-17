"""
World model training for multi-agent sequential communication.

Trains M(theta_w) as a regression model with loss (eq. 4):
    L(theta_w) = (1/|S|) * sum_{o,a,o',r in S} ||(o', r) - M(AM_w(e(o), a))||_2^2
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ObservationEncoder(nn.Module):
    """Encodes per-agent observations: e(o)."""

    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, n_agents, obs_dim) -> (batch, n_agents, embed_dim)
        return self.fc(obs)


class AttentionModule(nn.Module):
    """
    AM_w: attention module that aggregates encoded observations and actions.

    Produces a context vector used as input to the world model M.
    Query is formed from each agent's embedding; keys/values aggregate
    all agents, enabling generalisation across variable agent counts.
    """

    def __init__(self, embed_dim: int, action_dim: int, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim + action_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, enc_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # enc_obs:  (batch, n_agents, embed_dim)
        # actions:  (batch, n_agents, action_dim)
        x = torch.cat([enc_obs, actions], dim=-1)   # (batch, n_agents, embed_dim + action_dim)
        x = self.input_proj(x)                       # (batch, n_agents, embed_dim)
        attn_out, _ = self.attn(x, x, x)            # (batch, n_agents, embed_dim)
        # Mean-pool across agents for a fixed-size context vector
        context = attn_out.mean(dim=1)               # (batch, embed_dim)
        return self.out_proj(context)


class WorldModel(nn.Module):
    """
    M(theta_w): predicts joint next observations and reward.

    Input:  context vector from AM_w  (batch, embed_dim)
    Output: (o', r) concatenated       (batch, n_agents * obs_dim + 1)
    """

    def __init__(self, embed_dim: int, obs_dim: int, n_agents: int):
        super().__init__()
        self.output_dim = n_agents * obs_dim + 1  # flattened o' + scalar r
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.output_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class WorldModelNet(nn.Module):
    """
    Full pipeline: e -> AM_w -> M.

    Combines ObservationEncoder, AttentionModule, and WorldModel into a
    single module parameterised by theta_w.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        embed_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        self.encoder = ObservationEncoder(obs_dim, embed_dim)
        self.attn_module = AttentionModule(embed_dim, action_dim, n_heads)
        self.world_model = WorldModel(embed_dim, obs_dim, n_agents)
        self.n_agents = n_agents
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:     (batch, n_agents, obs_dim)  -- current observations o
            actions: (batch, n_agents, action_dim) -- actions a
        Returns:
            pred:    (batch, n_agents * obs_dim + 1) -- predicted (o', r)
        """
        enc_obs = self.encoder(obs)                      # e(o)
        context = self.attn_module(enc_obs, actions)     # AM_w(e(o), a)
        return self.world_model(context)                 # M(AM_w(e(o), a))


def world_model_loss(
    model: WorldModelNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Equation (4): L(theta_w) = (1/|S|) * sum ||(o', r) - M(AM_w(e(o), a))||_2^2

    Args:
        obs:      (batch, n_agents, obs_dim)
        actions:  (batch, n_agents, action_dim)
        next_obs: (batch, n_agents, obs_dim)
        rewards:  (batch, 1)
    Returns:
        scalar loss
    """
    pred = model(obs, actions)  # (batch, n_agents * obs_dim + 1)

    # Build target: flatten next_obs and append reward
    batch = obs.shape[0]
    target_obs = next_obs.reshape(batch, -1)            # (batch, n_agents * obs_dim)
    target = torch.cat([target_obs, rewards], dim=-1)   # (batch, n_agents * obs_dim + 1)

    # Squared L2 norm per sample, averaged over the dataset
    diff = target - pred
    loss = (diff * diff).sum(dim=-1).mean()
    return loss


def train(
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    embed_dim: int = 128,
    n_heads: int = 4,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 100,
    device: str = "cpu",
) -> WorldModelNet:
    """
    Train world model M(theta_w) on dataset S = {(o, a, o', r)}.

    Args:
        obs:        (|S|, n_agents, obs_dim)
        actions:    (|S|, n_agents, action_dim)
        next_obs:   (|S|, n_agents, obs_dim)
        rewards:    (|S|, 1)
        ...hyperparameters...
    Returns:
        Trained WorldModelNet
    """
    device = torch.device(device)
    model = WorldModelNet(obs_dim, action_dim, n_agents, embed_dim, n_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(
        obs.to(device),
        actions.to(device),
        next_obs.to(device),
        rewards.to(device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for o_batch, a_batch, o_next_batch, r_batch in loader:
            optimizer.zero_grad()
            loss = world_model_loss(model, o_batch, a_batch, o_next_batch, r_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * o_batch.shape[0]

        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    return model


if __name__ == "__main__":
    # Minimal smoke-test with random data
    N, A, OD, AD = 1000, 3, 8, 4  # samples, agents, obs_dim, action_dim
    obs = torch.randn(N, A, OD)
    actions = torch.randn(N, A, AD)
    next_obs = torch.randn(N, A, OD)
    rewards = torch.randn(N, 1)

    trained_model = train(
        obs, actions, next_obs, rewards,
        obs_dim=OD, action_dim=AD, n_agents=A,
        epochs=50,
    )
    print("Training complete.")
