"""
Team Poker environment for SeqComm overgeneralization testing.

N homogeneous agents play repeated poker hands against a dealer. Each hand:
  - Agent i privately observes hand strength H_i ~ Uniform[0, 1]
  - Team wins iff H̄ > D where D ~ Uniform[0, 1] (dealer, revealed after bets)
  - Each agent bets fraction ρ_i = σ(action_i) of current coffers C_i
  - Win  (prob H̄):   C_i → C_i · (1 + M · ρ_i)
  - Lose (prob 1-H̄): C_i → C_i · (1 + α) · (1 - ρ_i)

Per-hand reward: mean log-growth across agents (shared scalar for MAPPO compat).

Episode terminates when any agent hits C_floor (bankruptcy), any agent hits
C_target (success), or t == K_max hands have been played.

Math reference: math.md — in particular:
  θ_opt = (1+α)/(M+1+α)   — optimal betting threshold under full information
  ρ*(H̄) = max(0, (H̄(M+1)-1)/M) — Kelly fraction (no α dependence!)
  Δθ    = (N-1)(M-1-α)/(2(M+1+α)) — overgeneralization gap vs independent play
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeamPokerConfig:
    n_agents: int   = 4

    # Game parameters — see math.md §4.3 for recommended values
    M: float        = 2.0    # net profit multiplier on a win
    alpha: float    = 0.1    # consolation fraction on a loss (α · (1-ρ) · C returned)

    # Coffer dynamics
    C_0: float      = 100.0  # starting coffers per agent
    C_floor: float  = 10.0   # bankruptcy threshold  (10 % of start)
    C_target: float = 1000.0 # success threshold     (10× start)
    K_max: int      = 150    # max hands per episode

    seed: int | None = None


class TeamPokerEnv:
    """
    Gym-style team poker environment.

    Interface (same as GaussianFieldEnv / IslandCoverageEnv):
        reset()  → List[np.ndarray]                          one obs per agent
        step(actions) → (List[np.ndarray], float, bool)     obs, reward, done

    Continuous actions:
        Each action_i is a raw float (policy output before any transform).
        The env applies sigmoid internally so ρ_i = σ(action_i) ∈ (0, 1).
        Set env.continuous_actions = True so train.py skips discrete rounding.

    Observation per agent i  (obs_dim = n_agents + 2):
        [H_i,                            private hand strength this hand
         log(C_0/C_0), …, log(C_{N-1}/C_0),  log-wealth relative to start (all agents)
         t / K_max]                      episode progress

    H_i is the agent's private signal. All agents' log-coffers are shared
    (visible in obs_all through the attention module), but the actual H_j for
    j ≠ i is only inferable through other agents' bet sizes in the launching
    phase — exactly the information asymmetry SeqComm exploits.
    """

    continuous_actions: bool = True  # tell train.py not to round to int

    def __init__(self, cfg: TeamPokerConfig = TeamPokerConfig()) -> None:
        self.cfg   = cfg
        self._rng  = np.random.default_rng(cfg.seed)

        # Episode state
        self._coffers = np.full(cfg.n_agents, cfg.C_0, dtype=np.float64)
        self._hands   = np.zeros(cfg.n_agents, dtype=np.float64)
        self._t       = 0

        # Per-step diagnostics (optional, checked by run_episode with hasattr)
        self.last_win:       bool  = False
        self.last_H_bar:     float = 0.0
        self.last_dealer:    float = 0.0
        self.last_bets:      list  = []   # ρ values this step
        self.last_log_growths: list = []  # per-agent log-growth this step

        # Episode-level accumulators (reset each episode, read by poker_stats())
        self._ep_wins    = 0
        self._ep_sum_bet = 0.0
        self._ep_hands   = 0

    # ── Interface properties ───────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return self.cfg.n_agents + 2  # H_i + N log-coffers + t/K_max

    @property
    def action_dim(self) -> int:
        return 1  # single continuous bet fraction (before sigmoid)

    # ── Core interface ─────────────────────────────────────────────────────────

    def reset(self) -> list[np.ndarray]:
        self._coffers[:] = self.cfg.C_0
        self._t          = 0
        self._ep_wins    = 0
        self._ep_sum_bet = 0.0
        self._ep_hands   = 0
        self._deal_hands()
        return self._obs()

    def step(
        self, actions: list[float | int]
    ) -> tuple[list[np.ndarray], float, bool]:
        """
        Resolve one hand and advance the episode.

        actions : raw policy outputs (any real number); sigmoid applied here.
        Returns : (next_obs_list, shared_reward, done)
            shared_reward = mean log-growth across agents (log-utility objective)
        """
        cfg = self.cfg
        n   = cfg.n_agents

        # Map raw policy outputs → bet fractions ∈ (0, 1)
        rho = np.array([self._sigmoid(a) for a in actions], dtype=np.float64)

        # Resolve hand
        H_bar  = self._hands.mean()
        dealer = float(self._rng.uniform())
        win    = H_bar > dealer

        # Update coffers and compute per-agent log-growth
        C_old = self._coffers.copy()
        if win:
            self._coffers *= 1.0 + cfg.M * rho
        else:
            self._coffers *= (1.0 + cfg.alpha) * (1.0 - rho)

        log_growths = np.log(self._coffers / C_old)
        reward      = float(log_growths.mean())

        # Update diagnostics
        self.last_win         = win
        self.last_H_bar       = float(H_bar)
        self.last_dealer      = dealer
        self.last_bets        = rho.tolist()
        self.last_log_growths = log_growths.tolist()

        # Accumulate episode stats
        self._ep_hands   += 1
        self._ep_wins    += int(win)
        self._ep_sum_bet += float(rho.mean())

        # Advance time
        self._t += 1

        # Termination
        done = (
            bool(np.any(self._coffers <= cfg.C_floor))
            or bool(np.any(self._coffers >= cfg.C_target))
            or self._t >= cfg.K_max
        )

        if not done:
            self._deal_hands()

        return self._obs(), reward, done

    # ── Poker-specific metrics (read after done=True) ──────────────────────────

    def poker_stats(self) -> dict:
        """Extra episode metrics appended to episode_info by run_episode."""
        n = max(1, self._ep_hands)
        return {
            "poker_win_rate":     self._ep_wins / n,
            "poker_mean_bet":     self._ep_sum_bet / n,
            "poker_final_coffers": self._coffers.tolist(),
            "poker_bankruptcy":   bool(np.any(self._coffers <= self.cfg.C_floor)),
            "poker_target_hit":   bool(np.any(self._coffers >= self.cfg.C_target)),
            "poker_hands_played": self._ep_hands,
        }

    # ── Derived quantities (static, for analysis / logging) ───────────────────

    @property
    def theta_opt(self) -> float:
        """Optimal betting threshold: bet 1 iff H̄ > θ_opt. (math.md §3.1)"""
        return (1.0 + self.cfg.alpha) / (self.cfg.M + 1.0 + self.cfg.alpha)

    @property
    def theta_ind(self) -> float:
        """Independent agent threshold for N=n_agents. (math.md §4.1)"""
        n = self.cfg.n_agents
        return n * self.theta_opt - (n - 1) / 2.0

    @property
    def theta_kelly(self) -> float:
        """Kelly threshold (log-utility objective): bet >0 iff H̄ > θ_k. (math.md §3.2)"""
        return 1.0 / (self.cfg.M + 1.0)

    def kelly_fraction(self, H_bar: float) -> float:
        """Optimal Kelly bet fraction for a given team hand strength. (math.md §3.2)"""
        return max(0.0, (H_bar * (self.cfg.M + 1.0) - 1.0) / self.cfg.M)

    def overgeneralization_gap(self) -> float:
        """Δθ = (N-1)(M-1-α) / (2(M+1+α)). (math.md §4.2)"""
        n, M, a = self.cfg.n_agents, self.cfg.M, self.cfg.alpha
        return (n - 1) * (M - 1.0 - a) / (2.0 * (M + 1.0 + a))

    def noise_tolerance(self) -> float:
        """Max obs-noise std s.t. SeqComm ordering beats random. (math.md §2.3)"""
        return self.theta_opt * math.sqrt(self.cfg.n_agents)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _deal_hands(self) -> None:
        self._hands = self._rng.uniform(0.0, 1.0, self.cfg.n_agents)

    def _obs(self) -> list[np.ndarray]:
        log_coffers = np.log(self._coffers / self.cfg.C_0).astype(np.float32)
        t_norm      = np.float32(self._t / self.cfg.K_max)
        obs_list    = []
        for i in range(self.cfg.n_agents):
            obs_list.append(
                np.concatenate([
                    [np.float32(self._hands[i])],  # private hand strength
                    log_coffers,                   # shared log-wealth (all agents)
                    [t_norm],                      # episode progress
                ])
            )
        return obs_list

    @staticmethod
    def _sigmoid(x: float | int) -> float:
        """Numerically stable sigmoid."""
        if hasattr(x, "item"):
            x = x.item()
        x = float(x)
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)
