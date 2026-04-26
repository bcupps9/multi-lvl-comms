"""
IslandCoverageEnv — cooperative island assignment environment.

N agents on a 2D grid must each cover one of K islands (K == N by default).
An island covered by exactly one agent yields +1 reward; uncovered islands
yield -cover_penalty.  Two agents on the same island both still count as
covering it, but the overlap wastes one agent and triggers -overlap_penalty.

Why this environment stress-tests SeqComm communication:
  - With perfect comms: SeqComm assigns each island to exactly one agent via
    the priority cascade (first mover picks island, lower agents route around).
  - With packet loss: an agent misses an upper-agent's action and may pick the
    same island → overlap penalty.
  - With noisy hidden states: intention estimates degrade → priority ordering
    is wrong → wrong agent goes first, coordination collapses.
  - With delay: ordering info is stale by the time a lower agent acts.

This makes it a sharper coordination probe than GaussianFieldEnv, where agents
can still earn partial reward even when they cluster.

Observation o_i (length 2 + 2*K):
  [own_row/G, own_col/G,
   Δrow_to_island_0/G, Δcol_to_island_0/G,
   ...
   Δrow_to_island_(K-1)/G, Δcol_to_island_(K-1)/G]

Action: integer in {0,1,2,3,4} = stay, up, down, left, right
  (same discrete action space as GaussianFieldEnv — same policy head works)

Reward (shared, per timestep):
  +1.0 per island with exactly 1 agent
  -cover_penalty per island with 0 agents
  -overlap_penalty per *extra* agent on an island (count - 1) for count > 1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class IslandCoverageConfig:
    grid_size:       int   = 20
    n_agents:        int   = 4
    n_islands:       int   = 4      # K; should equal n_agents for the assignment problem
    cover_penalty:   float = 0.5    # penalty per uncovered island per step
    overlap_penalty: float = 0.5    # penalty per extra agent on a doubly-occupied island
    randomize_islands: bool = False  # re-randomize island positions each episode
    proximity_scale: float = 0.05   # shaping bonus per agent per step for being near any island
                                    # set to 0.0 to disable; keeps gradient non-zero before islands reached


class IslandCoverageEnv:
    """
    Cooperative assignment environment.

    Agents navigate a grid to cover K islands.  The challenge is pure
    coordination: no agent can know from its own observation where others
    are going, so communication is essential for a high reward.
    """

    N_ACTIONS = 5
    _DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay,up,down,left,right

    def __init__(self, cfg: IslandCoverageConfig = IslandCoverageConfig(),
                 seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Island positions: fixed by default so agents learn the assignment
        # problem rather than a tracking problem.
        self._island_positions: np.ndarray = self._make_island_positions()

        # Agent positions, initialised in reset()
        self.positions = np.zeros((cfg.n_agents, 2), dtype=np.int32)
        self.reset()

    # ── Dimensions ────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return 2 + 2 * self.cfg.n_islands

    @property
    def action_dim(self) -> int:
        return self.N_ACTIONS

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        """Reset agent positions (and optionally island positions)."""
        G = self.cfg.grid_size
        if self.cfg.randomize_islands:
            self._island_positions = self._make_island_positions()
        # Spread agents randomly; avoid placing all on the same cell.
        self.positions = self.rng.integers(0, G, (self.cfg.n_agents, 2)).astype(np.int32)
        return [self._obs_for(i) for i in range(self.cfg.n_agents)]

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool]:
        """
        Step environment.

        Args:
            actions: one integer per agent in [0, N_ACTIONS)
        Returns:
            (obs_list, reward, done=False)
        """
        G = self.cfg.grid_size

        # Move agents
        for i, a in enumerate(actions):
            dr, dc = self._DELTAS[int(a)]
            self.positions[i, 0] = int(np.clip(self.positions[i, 0] + dr, 0, G - 1))
            self.positions[i, 1] = int(np.clip(self.positions[i, 1] + dc, 0, G - 1))

        # Count agents on each island
        island_counts = np.zeros(self.cfg.n_islands, dtype=np.int32)
        for i in range(self.cfg.n_agents):
            for k in range(self.cfg.n_islands):
                if (self.positions[i, 0] == self._island_positions[k, 0] and
                        self.positions[i, 1] == self._island_positions[k, 1]):
                    island_counts[k] += 1

        # Reward:  +1 per covered island, -penalty for uncovered, -overlap
        reward = 0.0
        coverage_count = 0
        overlap_count = 0
        for k in range(self.cfg.n_islands):
            c = int(island_counts[k])
            if c == 0:
                reward -= self.cfg.cover_penalty
            else:
                reward += 1.0
                coverage_count += 1
                if c > 1:
                    extra = c - 1
                    reward -= self.cfg.overlap_penalty * extra
                    overlap_count += extra

        self.last_coverage_count = coverage_count  # islands with exactly 1 agent
        self.last_overlap_count = overlap_count    # extra agents on doubled-up islands

        # Proximity shaping: small per-agent bonus proportional to closeness to nearest island.
        # Normalized by 2*G so the maximum bonus per step across all agents is proximity_scale.
        if self.cfg.proximity_scale > 0.0:
            G = self.cfg.grid_size
            for i in range(self.cfg.n_agents):
                min_dist = min(
                    int(abs(self.positions[i, 0] - self._island_positions[k, 0]) +
                        abs(self.positions[i, 1] - self._island_positions[k, 1]))
                    for k in range(self.cfg.n_islands)
                )
                reward += self.cfg.proximity_scale * (1.0 - min_dist / (2.0 * G))

        obs = [self._obs_for(i) for i in range(self.cfg.n_agents)]
        return obs, float(reward), False

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_island_positions(self) -> np.ndarray:
        """
        Place K islands on a rough grid so they are evenly spread.
        For K=4 on a 20×20 grid this gives (5,5), (5,15), (15,5), (15,15).
        With randomize_islands=True a fresh random placement is used each call.
        """
        G = self.cfg.grid_size
        K = self.cfg.n_islands
        if self.cfg.randomize_islands:
            positions = set()
            while len(positions) < K:
                r = int(self.rng.integers(2, G - 2))
                c = int(self.rng.integers(2, G - 2))
                positions.add((r, c))
            return np.array(list(positions), dtype=np.int32)

        # Deterministic grid layout — evenly spaces K islands
        cols = max(1, int(np.ceil(np.sqrt(K))))
        rows = max(1, int(np.ceil(K / cols)))
        row_step = G // (rows + 1)
        col_step = G // (cols + 1)
        positions = []
        for ri in range(1, rows + 1):
            for ci in range(1, cols + 1):
                if len(positions) < K:
                    positions.append([ri * row_step, ci * col_step])
        return np.array(positions, dtype=np.int32)

    def _obs_for(self, agent_id: int) -> np.ndarray:
        G   = self.cfg.grid_size
        row = int(self.positions[agent_id, 0])
        col = int(self.positions[agent_id, 1])

        obs = [row / G, col / G]
        for k in range(self.cfg.n_islands):
            irow = int(self._island_positions[k, 0])
            icol = int(self._island_positions[k, 1])
            obs.append((irow - row) / G)  # signed delta: +row is down
            obs.append((icol - col) / G)  # signed delta: +col is right
        return np.array(obs, dtype=np.float32)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_grid(self) -> np.ndarray:
        """Return (grid_size, grid_size) array: 1=island, 2=agent, 3=agent-on-island."""
        G = self.cfg.grid_size
        grid = np.zeros((G, G), dtype=np.int32)
        for k in range(self.cfg.n_islands):
            grid[self._island_positions[k, 0], self._island_positions[k, 1]] = 1
        for i in range(self.cfg.n_agents):
            r, c = int(self.positions[i, 0]), int(self.positions[i, 1])
            grid[r, c] = 3 if grid[r, c] == 1 else 2
        return grid
