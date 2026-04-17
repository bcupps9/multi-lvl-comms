"""
Python mirror of robot_sim/gaussian_field_env — used during MAPPO training.

Identical dynamics to the C++ version; synchronous since the Python training
loop steps all agents together rather than running them as cotamer tasks.

Observation o_i: [row/G, col/G, field_window_flat]
  - 2 normalised position values
  - (2*window_half+1)^2 field samples centred on agent, zero-padded at edges

Action: integer in {0,1,2,3,4} = stay, up, down, left, right

Reward (shared across all agents at each timestep):
  sum_i F(p_i)  -  overlap_penalty * |{(i,j): p_i==p_j, i<j}|
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GaussianFieldConfig:
    grid_size:       int   = 20
    window_half:     int   = 2      # window = (2*window_half+1)^2
    n_gaussians:     int   = 3
    gaussian_sigma:  float = 2.5
    move_speed:      float = 0.4    # Gaussian center displacement per step
    overlap_penalty: float = 1.0    # lambda
    n_agents:        int   = 4


class GaussianFieldEnv:
    """
    2D cooperative coverage environment.

    Agents navigate a grid to collect reward from K moving Gaussian peaks.
    They are penalised for occupying the same cell (duplicate coverage).
    The hidden field is never directly revealed — agents only see a local
    window, so the world model M must learn to predict field evolution.
    """

    N_ACTIONS = 5  # stay, up, down, left, right
    _DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, cfg: GaussianFieldConfig = GaussianFieldConfig(),
                 seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._win = 2 * cfg.window_half + 1
        # state initialised in reset()
        self.centers    = np.zeros((cfg.n_gaussians, 2), dtype=np.float32)
        self.velocities = np.zeros((cfg.n_gaussians, 2), dtype=np.float32)
        self.positions  = np.zeros((cfg.n_agents, 2), dtype=np.int32)
        self.reset()

    # ── Dimensions ────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return 2 + self._win ** 2

    @property
    def action_dim(self) -> int:
        return self.N_ACTIONS

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        """Reset field and agent positions. Returns list of obs per agent."""
        G = self.cfg.grid_size
        self.centers = self.rng.uniform(0, G, (self.cfg.n_gaussians, 2)).astype(np.float32)
        angles = self.rng.uniform(0, 2 * np.pi, self.cfg.n_gaussians)
        self.velocities = (self.cfg.move_speed *
                           np.stack([np.cos(angles), np.sin(angles)], axis=1)
                           .astype(np.float32))
        self.positions = self.rng.integers(0, G, (self.cfg.n_agents, 2)).astype(np.int32)
        return [self._obs_for(i) for i in range(self.cfg.n_agents)]

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool]:
        """
        Step the environment.

        Args:
            actions: list of int (one per agent), each in [0, N_ACTIONS)
        Returns:
            (obs_list, reward, done)
            done is always False — episodes end by timestep count
        """
        G = self.cfg.grid_size

        # Move agents
        for i, a in enumerate(actions):
            dr, dc = self._DELTAS[int(a)]
            self.positions[i, 0] = np.clip(self.positions[i, 0] + dr, 0, G - 1)
            self.positions[i, 1] = np.clip(self.positions[i, 1] + dc, 0, G - 1)

        # Reward: field value at each agent's position
        reward = sum(
            self._field_at(self.positions[i, 0], self.positions[i, 1])
            for i in range(self.cfg.n_agents)
        )

        # Overlap penalty
        for i in range(self.cfg.n_agents):
            for j in range(i + 1, self.cfg.n_agents):
                if np.array_equal(self.positions[i], self.positions[j]):
                    reward -= self.cfg.overlap_penalty

        # Advance Gaussians and bounce off walls
        self.centers += self.velocities
        for k in range(self.cfg.n_gaussians):
            for d in range(2):
                if self.centers[k, d] < 0:
                    self.centers[k, d] *= -1
                    self.velocities[k, d] = abs(self.velocities[k, d])
                elif self.centers[k, d] >= G:
                    self.centers[k, d] = 2 * G - self.centers[k, d]
                    self.velocities[k, d] = -abs(self.velocities[k, d])

        obs = [self._obs_for(i) for i in range(self.cfg.n_agents)]
        return obs, float(reward), False

    # ── Internals ─────────────────────────────────────────────────────────────

    def _field_at(self, row: float, col: float) -> float:
        inv2s2 = 1.0 / (2.0 * self.cfg.gaussian_sigma ** 2)
        dr = row - self.centers[:, 0]
        dc = col - self.centers[:, 1]
        return float(np.exp(-(dr * dr + dc * dc) * inv2s2).sum())

    def _obs_for(self, agent_id: int) -> np.ndarray:
        G   = self.cfg.grid_size
        w   = self.cfg.window_half
        row = int(self.positions[agent_id, 0])
        col = int(self.positions[agent_id, 1])

        obs = [row / G, col / G]
        for dr in range(-w, w + 1):
            for dc in range(-w, w + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < G and 0 <= nc < G:
                    obs.append(self._field_at(nr, nc))
                else:
                    obs.append(0.0)
        return np.array(obs, dtype=np.float32)

    # ── Rendering (optional) ──────────────────────────────────────────────────

    def render_field(self) -> np.ndarray:
        """Return the full field as a (grid_size, grid_size) array for viz."""
        G = self.cfg.grid_size
        grid = np.zeros((G, G), dtype=np.float32)
        for r in range(G):
            for c in range(G):
                grid[r, c] = self._field_at(r, c)
        return grid
