#pragma once
// battleship_env.hh
//
// Collaborative Battleship — N_a agent ships vs N_b boss ships on an M×M grid.
//
// Grid: one flat Cell[M*M] array; every read/write goes through this single object.
//
// Ships: 3 contiguous cells (horizontal or vertical), tracked by center + orientation.
//   Partial damage is possible — individual cells are removed when hit.
//   A ship is sunk when all 3 cells are gone.
//
// Agent observations: (2·sight+1)² local Chebyshev patch × 3 channels + 2 scalars.
//   Ch0 = own ship cells, Ch1 = ally cells, Ch2 = boss cells.
//   Out-of-bounds and empty cells → all zeros.
//
// Actions: 3 floats from the policy network (Gaussian, same as poker).
//   action[0] → move direction   (continuous, decoded to N/S/E/W/stay)
//   action[1] → fire row offset  (continuous, clamped + rounded to fire_range)
//   action[2] → fire col offset  (continuous, clamped + rounded to fire_range)
//
// Boss policy: fixed heuristic — move toward nearest visible agent, then fire on
//   a predictable cadence. By default the shot is aimed at last-step agent cells
//   so moving is a real dodge mechanic.
//
// Episode order per step: agents move → agents fire → bosses move → bosses fire.

#include "agent_action.hh"
#include "random_source.hh"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace seqcomm {
namespace cot = cotamer;

// ── Config ─────────────────────────────────────────────────────────────────────

struct BattleshipConfig {
    int   M           = 8;    // grid side length
    int   n_agents    = 2;    // number of agent ships
    int   n_boss      = 2;    // number of boss ships
    int   sight_range = 4;    // Chebyshev observation radius
    int   fire_range  = 3;    // Chebyshev firing radius (agents)
    int   boss_fire_range = 3; // Chebyshev firing radius (boss heuristic)
    int   boss_fire_period = 8; // boss fires once every N boss turns
    int   boss_aim_lag = 1;     // 0=current cells, >=1=previous-step cells
    int   max_steps   = 60;   // episode cap
    float reward_hit_boss   =  1.f;
    float reward_hit_self   = -1.f;
    float reward_survive    =  0.005f; // per agent cell alive per step (small stabilizer)
    float reward_near_boss  =  0.20f;  // shaped reward for near-miss agent shots
    float reward_proximity  =  0.01f;  // per-step reward for agent being close to boss (gradient for move)
    float reward_agents_win = 10.f;    // terminal reward for sinking all boss ships

    // Curriculum knobs — updated mid-run by the simulator; env reads each episode.
    int   boss_start_hp  = 3;    // 1/2/3: how many cells the boss starts with alive
    float boss_miss_prob = 0.0f; // probability [0,1) each boss shot misses entirely
};

// ── Grid cell ──────────────────────────────────────────────────────────────────

struct Cell {
    uint8_t type;     // 0 = empty, 1 = agent ship, 2 = boss ship
    uint8_t ship_id;  // index into agents_[] or bosses_[]
    uint8_t part;     // 0 / 1 / 2 (which of the 3 cells)
};
inline constexpr Cell EMPTY_CELL{0, 0, 0};

// ── Ship ───────────────────────────────────────────────────────────────────────

struct Ship {
    int  id;
    bool is_boss;
    int  cr, cc;     // center cell (part 1)
    bool horiz;      // true = horizontal; false = vertical
    bool alive[3];   // which of the 3 cells survive

    int  hp()   const { return (int)alive[0] + alive[1] + alive[2]; }
    bool sunk() const { return hp() == 0; }

    // (row, col) of part p.
    std::array<int, 2> cell_of(int p) const {
        return horiz ? std::array<int,2>{cr, cc + p - 1}
                     : std::array<int,2>{cr + p - 1, cc};
    }
};

// ── Environment ────────────────────────────────────────────────────────────────

struct BattleshipEnv : Environment {
    explicit BattleshipEnv(BattleshipConfig cfg, random_source& rng);

    // ── Barrier-based interface (cotamer modes) ─────────────────────────────────
    void submit_action(int agent_id,
                       std::span<const float> action) override;
    cot::task<std::pair<std::vector<float>, float>>
    get_result(int agent_id) override;

    // ── Sync interface (mappo, no cotamer) ─────────────────────────────────────
    struct SyncResult {
        std::vector<std::vector<float>> next_obs;
        float reward;
    };
    SyncResult sync_step(const std::vector<std::vector<float>>& actions);

    // ── Episode control ─────────────────────────────────────────────────────────
    std::vector<std::vector<float>> reset();
    bool is_done()     const { return done_; }
    int  obs_dim()     const;   // local-patch observation dimension
    int  global_obs_dim() const;
    int  action_dim()  const { return 3; }

    // ── Observations ────────────────────────────────────────────────────────────
    std::vector<float> obs_for(int agent_id) const;
    std::vector<float> global_obs()          const;

    const BattleshipConfig& config() const { return cfg_; }

    // Update curriculum parameters; takes effect on the next reset().
    // near_miss_reward < 0 means "leave unchanged".
    void set_curriculum(int boss_hp, float miss_prob,
                        int fire_period = -1, int aim_lag = -1,
                        int max_steps = -1, float near_miss_reward = -1.f) {
        cfg_.boss_start_hp  = boss_hp;
        cfg_.boss_miss_prob = miss_prob;
        if (fire_period > 0)      cfg_.boss_fire_period = fire_period;
        if (aim_lag >= 0)         cfg_.boss_aim_lag = aim_lag;
        if (max_steps > 0)        cfg_.max_steps = max_steps;
        if (near_miss_reward >= 0.f) cfg_.reward_near_boss = near_miss_reward;
    }

    // ── Episode statistics (valid after is_done()) ──────────────────────────────
    struct EpisodeStats {
        int   steps;
        int   boss_hits;
        int   agent_hits;
        bool  agents_won;   // all boss ships sunk
        bool  boss_won;     // all agent ships sunk
        float total_reward;
        int   agent_shots;
        int   fire_oob;
        int   wasted_shots;   // shots blocked because a teammate already targeted that cell
        float mean_fire_dist;
        // Territorial coverage metrics
        float mean_ally_dist;          // avg min-Chebyshev distance to nearest ally (spread signal)
        std::vector<int> boss_hit_counts;  // per-boss: how many cells were hit this episode
        std::vector<int> fire_dist_counts;   // 0..fire_range, then >fire_range
        std::array<int, 5> move_counts;      // stay, N, S, E, W
        std::vector<int> fire_offset_counts; // row-major over [-fire_range, fire_range]^2
    };
    EpisodeStats episode_stats() const;

private:
    BattleshipConfig cfg_;
    random_source&   rng_;

    std::vector<Cell> grid_;    // M×M, row-major
    std::vector<Ship> agents_;
    std::vector<Ship> bosses_;
    std::vector<std::array<std::array<int, 2>, 3>> prev_agent_cells_;
    std::vector<std::array<bool, 3>> prev_agent_alive_;

    int   step_        = 0;
    bool  done_        = false;
    float ep_reward_   = 0.f;
    int   ep_boss_hits_  = 0;
    int   ep_agent_hits_ = 0;
    int   ep_agent_shots_ = 0;
    int   ep_fire_oob_ = 0;
    int   ep_wasted_shots_ = 0;
    // Territorial coverage accumulators
    float ep_ally_dist_sum_   = 0.f;  // sum of per-agent min-ally distances across all steps
    int   ep_ally_dist_n_     = 0;    // number of (agent, step) samples
    std::vector<int> ep_boss_hit_counts_;  // per-boss hit tally
    int   ep_fire_dist_samples_ = 0;
    float ep_fire_dist_sum_ = 0.f;
    std::vector<int> ep_fire_dist_counts_;
    std::array<int, 5> ep_move_counts_{};
    std::vector<int> ep_fire_offset_counts_;

    // Per-step output (set by step_env)
    std::vector<std::vector<float>> next_obs_;
    std::vector<float> step_reward_;   // per-agent reward for the current step

    // Cotamer barrier state
    std::vector<std::vector<float>> pending_;
    int  submitted_     = 0;
    bool results_ready_ = false;

    Cell&       at(int r, int c)       { return grid_[r * cfg_.M + c]; }
    const Cell& at(int r, int c) const { return grid_[r * cfg_.M + c]; }

    void place_ships();
    bool try_place(Ship& s, bool is_boss);
    void move_ship(Ship& s, int dr, int dc);
    void snapshot_agent_cells();
    bool fire_at(bool by_boss, int tr, int tc);   // true = hit
    int  nearest_live_boss_dist(int tr, int tc) const;
    float near_boss_reward(int tr, int tc) const;

    // Decode continuous action to integer
    static int               decode_move(float a);
    static std::pair<int,int> decode_fire(float dr_raw, float dc_raw, int range);

    std::array<int, 3> boss_policy(int boss_id) const;

    void step_env(const std::vector<std::vector<float>>& agent_actions);
};

} // namespace seqcomm
