"""
IntersectionCrossingEnv — four robots cross a shared intersection.

Four agents approach from NSEW, each trying to reach the opposite end.
They must coordinate to cross the shared 2×2 intersection without colliding.
This directly tests SeqComm's sequential commitment mechanism: the agent with
the highest intention goes first, lower-priority agents wait.

Grid: 20×20
Fixed start/goal pairs:
  Agent 0: (1,10)  → (18,10)   [North → South]
  Agent 1: (18,10) → (1,10)    [South → North]
  Agent 2: (10,1)  → (10,18)   [West  → East ]
  Agent 3: (10,18) → (10,1)    [East  → West ]

Intersection zone: rows 9–10, cols 9–10 (four cells)

Observation o_i (obs_dim=8, partial — no other agents visible):
  [row/G, col/G, goal_row/G, goal_col/G,
   Δrow_to_center/G, Δcol_to_center/G,
   dist_to_goal/(2*G), in_intersection_flag]

Actions: 0=stay  1=up(-row)  2=down(+row)  3=left(-col)  4=right(+col)

Reward (shared across all agents each step):
  -step_penalty per agent alive (time pressure)
  +goal_reward per agent that reaches its goal this step (one-time)
  -collision_penalty per colliding pair (same cell, per step)

done: True when all agents are at their goal.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class IntersectionCrossingConfig:
    grid_size:         int   = 20
    step_penalty:      float = 0.02
    goal_reward:       float = 10.0
    collision_penalty: float = 5.0
    n_agents:          int   = 4


class IntersectionCrossingEnv:

    N_ACTIONS = 5
    _DELTAS   = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    # geometric center of the 2×2 intersection (rows 9–10, cols 9–10)
    _CENTER_ROW = 9.5
    _CENTER_COL = 9.5

    _STARTS = [(1, 10), (18, 10), (10, 1), (10, 18)]
    _GOALS  = [(18, 10), (1, 10), (10, 18), (10, 1)]

    def __init__(self, cfg: IntersectionCrossingConfig = IntersectionCrossingConfig()):
        self.cfg = cfg
        assert cfg.n_agents == 4, "IntersectionCrossingEnv requires exactly 4 agents"
        self.positions:    list = list(self._STARTS)
        self.reached_goal: list = [False] * cfg.n_agents

    # ── Dimensions ────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return 8

    @property
    def action_dim(self) -> int:
        return self.N_ACTIONS

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        self.positions    = list(self._STARTS)
        self.reached_goal = [False] * self.cfg.n_agents
        return [self._obs_for(i) for i in range(self.cfg.n_agents)]

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool]:
        G = self.cfg.grid_size

        # Move agents that haven't yet reached their goal
        for i, a in enumerate(actions):
            if self.reached_goal[i]:
                continue
            dr, dc = self._DELTAS[int(a) % self.N_ACTIONS]
            r, c = self.positions[i]
            self.positions[i] = (
                max(0, min(G - 1, r + dr)),
                max(0, min(G - 1, c + dc)),
            )

        # Shared reward
        reward = -self.cfg.step_penalty * self.cfg.n_agents

        # Goal bonus (one-time per agent)
        for i in range(self.cfg.n_agents):
            if not self.reached_goal[i] and self.positions[i] == self._GOALS[i]:
                self.reached_goal[i] = True
                reward += self.cfg.goal_reward

        # Collision penalty (per pair per step)
        for i in range(self.cfg.n_agents):
            for j in range(i + 1, self.cfg.n_agents):
                if (not self.reached_goal[i] and not self.reached_goal[j]
                        and self.positions[i] == self.positions[j]):
                    reward -= self.cfg.collision_penalty

        done = all(self.reached_goal)
        return [self._obs_for(i) for i in range(self.cfg.n_agents)], float(reward), done

    # ── Internals ─────────────────────────────────────────────────────────────

    def _obs_for(self, agent_id: int) -> np.ndarray:
        G  = float(self.cfg.grid_size)
        r, c   = self.positions[agent_id]
        gr, gc = self._GOALS[agent_id]
        dist          = (abs(r - gr) + abs(c - gc)) / (2.0 * G)
        in_intersect  = float(9 <= r <= 10 and 9 <= c <= 10)
        return np.array([
            r / G,
            c / G,
            gr / G,
            gc / G,
            (r - self._CENTER_ROW) / G,
            (c - self._CENTER_COL) / G,
            dist,
            in_intersect,
        ], dtype=np.float32)
