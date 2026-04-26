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
// Boss policy: fixed heuristic — fire on nearest in-range agent cell,
//   move toward nearest visible agent (global vision).
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
    int   fire_range  = 3;    // Chebyshev firing radius
    int   max_steps   = 60;   // episode cap
    float reward_hit_boss   =  1.f;
    float reward_hit_self   = -1.f;
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

    // ── Episode statistics (valid after is_done()) ──────────────────────────────
    struct EpisodeStats {
        int   steps;
        int   boss_hits;
        int   agent_hits;
        bool  agents_won;   // all boss ships sunk
        bool  boss_won;     // all agent ships sunk
        float total_reward;
    };
    EpisodeStats episode_stats() const;

private:
    BattleshipConfig cfg_;
    random_source&   rng_;

    std::vector<Cell> grid_;    // M×M, row-major
    std::vector<Ship> agents_;
    std::vector<Ship> bosses_;

    int   step_        = 0;
    bool  done_        = false;
    float ep_reward_   = 0.f;
    int   ep_boss_hits_  = 0;
    int   ep_agent_hits_ = 0;

    // Per-step output (set by step_env)
    std::vector<std::vector<float>> next_obs_;
    float step_reward_ = 0.f;

    // Cotamer barrier state
    std::vector<std::vector<float>> pending_;
    int  submitted_     = 0;
    bool results_ready_ = false;

    Cell&       at(int r, int c)       { return grid_[r * cfg_.M + c]; }
    const Cell& at(int r, int c) const { return grid_[r * cfg_.M + c]; }

    void place_ships();
    bool try_place(Ship& s, bool is_boss);
    void move_ship(Ship& s, int dr, int dc);
    bool fire_at(bool by_boss, int tr, int tc);   // true = hit

    // Decode continuous action to integer
    static int               decode_move(float a);
    static std::pair<int,int> decode_fire(float dr_raw, float dc_raw, int range);

    std::array<int, 3> boss_policy(int boss_id) const;

    void step_env(const std::vector<std::vector<float>>& agent_actions);
};

} // namespace seqcomm
