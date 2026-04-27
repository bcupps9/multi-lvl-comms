#include "battleship_env.hh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>

namespace seqcomm {
namespace cot = cotamer;
using namespace std::chrono_literals;

// ── Construction ───────────────────────────────────────────────────────────────

BattleshipEnv::BattleshipEnv(BattleshipConfig cfg, random_source& rng)
    : cfg_(cfg), rng_(rng),
      grid_(cfg.M * cfg.M, EMPTY_CELL),
      agents_(cfg.n_agents),
      bosses_(cfg.n_boss),
      next_obs_(cfg.n_agents),
      pending_(cfg.n_agents)
{
    for (int i = 0; i < cfg.n_agents; ++i) { agents_[i].id = i; agents_[i].is_boss = false; }
    for (int i = 0; i < cfg.n_boss;   ++i) { bosses_[i].id = i; bosses_[i].is_boss = true;  }
}

// ── Reset ──────────────────────────────────────────────────────────────────────

std::vector<std::vector<float>> BattleshipEnv::reset() {
    std::fill(grid_.begin(), grid_.end(), EMPTY_CELL);
    step_          = 0;
    done_          = false;
    ep_reward_     = 0.f;
    ep_boss_hits_  = 0;
    ep_agent_hits_ = 0;
    ep_agent_shots_ = 0;
    ep_fire_oob_ = 0;
    ep_fire_dist_samples_ = 0;
    ep_fire_dist_sum_ = 0.f;
    ep_fire_dist_counts_.assign(cfg_.fire_range + 2, 0);
    ep_move_counts_.fill(0);
    int fire_span = 2 * cfg_.fire_range + 1;
    ep_fire_offset_counts_.assign(fire_span * fire_span, 0);
    submitted_     = 0;
    results_ready_ = false;

    place_ships();

    std::vector<std::vector<float>> obs;
    obs.reserve(cfg_.n_agents);
    for (int i = 0; i < cfg_.n_agents; ++i)
        obs.push_back(obs_for(i));
    return obs;
}

// ── Ship placement ─────────────────────────────────────────────────────────────

bool BattleshipEnv::try_place(Ship& s, bool is_boss) {
    int M = cfg_.M;
    s.horiz = rng_.coin_flip();

    // Center row/col bounds so all 3 cells stay in [0, M).
    int r_lo = s.horiz ? 0   : 1,   r_hi = s.horiz ? M-1 : M-2;
    int c_lo = s.horiz ? 1   : 0,   c_hi = s.horiz ? M-2 : M-1;

    s.cr = rng_.uniform(r_lo, r_hi);
    s.cc = rng_.uniform(c_lo, c_hi);

    // Check overlap with already-placed ships.
    for (int p = 0; p < 3; ++p) {
        auto [r, c] = s.cell_of(p);
        if (at(r, c).type != 0) return false;
    }

    // Stamp onto grid.
    uint8_t t = is_boss ? 2 : 1;
    for (int p = 0; p < 3; ++p) {
        auto [r, c] = s.cell_of(p);
        at(r, c) = {t, (uint8_t)s.id, (uint8_t)p};
        s.alive[p] = true;
    }
    return true;
}

void BattleshipEnv::place_ships() {
    for (auto& s : bosses_)
        for (int i = 0; i < 500 && !try_place(s, true);  ++i);
    for (auto& s : agents_)
        for (int i = 0; i < 500 && !try_place(s, false); ++i);
}

// ── Movement ───────────────────────────────────────────────────────────────────
//
// 1. Vacate all alive cells of the ship.
// 2. Check all new positions are in-bounds and unoccupied.
// 3. If valid, update center; re-stamp at new positions.
//    If invalid, re-stamp at original positions (ship stays).

void BattleshipEnv::move_ship(Ship& s, int dr, int dc) {
    if (dr == 0 && dc == 0) return;
    int M = cfg_.M;

    for (int p = 0; p < 3; ++p) {
        if (!s.alive[p]) continue;
        auto [r, c] = s.cell_of(p);
        at(r, c) = EMPTY_CELL;
    }

    bool ok = true;
    for (int p = 0; p < 3; ++p) {
        if (!s.alive[p]) continue;
        auto [r, c] = s.cell_of(p);
        int nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= M || nc < 0 || nc >= M || at(nr, nc).type != 0) {
            ok = false;
            break;
        }
    }

    if (ok) { s.cr += dr; s.cc += dc; }

    uint8_t t = s.is_boss ? 2 : 1;
    for (int p = 0; p < 3; ++p) {
        if (!s.alive[p]) continue;
        auto [r, c] = s.cell_of(p);
        at(r, c) = {t, (uint8_t)s.id, (uint8_t)p};
    }
}

// ── Firing ─────────────────────────────────────────────────────────────────────

bool BattleshipEnv::fire_at(bool by_boss, int tr, int tc) {
    int M = cfg_.M;
    if (tr < 0 || tr >= M || tc < 0 || tc >= M) return false;

    Cell& cell = at(tr, tc);
    uint8_t enemy_type = by_boss ? 1 : 2;
    if (cell.type != enemy_type) return false;

    // Destroy this cell.
    Ship& victim = by_boss ? agents_[cell.ship_id] : bosses_[cell.ship_id];
    victim.alive[cell.part] = false;
    cell = EMPTY_CELL;
    return true;
}

int BattleshipEnv::nearest_live_boss_dist(int tr, int tc) const {
    int best = INT_MAX;
    for (const Ship& boss : bosses_) {
        for (int p = 0; p < 3; ++p) {
            if (!boss.alive[p]) continue;
            auto [r, c] = boss.cell_of(p);
            int d = std::max(std::abs(r - tr), std::abs(c - tc));
            best = std::min(best, d);
        }
    }
    return best;
}

float BattleshipEnv::near_boss_reward(int tr, int tc) const {
    int M = cfg_.M;
    if (cfg_.reward_near_boss == 0.f ||
        tr < 0 || tr >= M || tc < 0 || tc >= M ||
        cfg_.fire_range <= 0) {
        return 0.f;
    }

    int d = nearest_live_boss_dist(tr, tc);
    if (d <= 0 || d > cfg_.fire_range) return 0.f;

    float closeness = static_cast<float>(cfg_.fire_range + 1 - d) /
                      static_cast<float>(cfg_.fire_range);
    return cfg_.reward_near_boss * closeness;
}

// ── Action decoding ────────────────────────────────────────────────────────────

int BattleshipEnv::decode_move(float a) {
    // 0=stay  1=N(dr=-1)  2=S(dr=+1)  3=E(dc=+1)  4=W(dc=-1)
    if (a < -0.6f) return 4;
    if (a < -0.2f) return 2;
    if (a <  0.2f) return 0;
    if (a <  0.6f) return 1;
    return 3;
}

std::pair<int, int> BattleshipEnv::decode_fire(float dr_raw, float dc_raw, int range) {
    int dr = (int)std::round(dr_raw);
    int dc = (int)std::round(dc_raw);
    dr = std::clamp(dr, -range, range);
    dc = std::clamp(dc, -range, range);
    return {dr, dc};
}

// ── Boss policy ────────────────────────────────────────────────────────────────
//
// Fire on the nearest agent cell within fire_range (Chebyshev).
// If none in range: move one step toward the nearest agent cell (global vision).
// If in range: stay put (already close enough to fire effectively).

std::array<int, 3> BattleshipEnv::boss_policy(int boss_id) const {
    const Ship& b = bosses_[boss_id];
    int R = cfg_.fire_range;

    // Nearest in-range agent cell → fire target.
    int best_dist = INT_MAX, fdr = 0, fdc = 0;
    bool found = false;
    for (const Ship& a : agents_) {
        for (int p = 0; p < 3; ++p) {
            if (!a.alive[p]) continue;
            auto [r, c] = a.cell_of(p);
            int d = std::max(std::abs(r - b.cr), std::abs(c - b.cc));
            if (d <= R && d < best_dist) {
                best_dist = d; fdr = r - b.cr; fdc = c - b.cc;
                found = true;
            }
        }
    }

    // Move toward nearest agent cell (global vision for boss).
    int move = 0;
    if (!found) {
        int md = INT_MAX, tr = b.cr, tc = b.cc;
        for (const Ship& a : agents_) {
            for (int p = 0; p < 3; ++p) {
                if (!a.alive[p]) continue;
                auto [r, c] = a.cell_of(p);
                int d = std::abs(r - b.cr) + std::abs(c - b.cc);
                if (d < md) { md = d; tr = r; tc = c; }
            }
        }
        int dr = (tr > b.cr) - (tr < b.cr);
        int dc = (tc > b.cc) - (tc < b.cc);
        // Prefer row movement; fall back to column.
        if      (dr == -1) move = 1;
        else if (dr ==  1) move = 2;
        else if (dc ==  1) move = 3;
        else if (dc == -1) move = 4;
    }

    return {move, fdr, fdc};
}

// ── Core step ──────────────────────────────────────────────────────────────────
//
// Order: agents move → agents fire → bosses move → bosses fire.

void BattleshipEnv::step_env(const std::vector<std::vector<float>>& actions) {
    static constexpr int DR[] = {0, -1, 1, 0,  0};  // indexed by move 0..4
    static constexpr int DC[] = {0,  0, 0, 1, -1};

    float reward = 0.f;

    // Agents act.
    for (int i = 0; i < cfg_.n_agents; ++i) {
        if (agents_[i].sunk()) continue;
        const auto& a = actions[i];
        int dir = decode_move(a[0]);
        ++ep_move_counts_[dir];
        move_ship(agents_[i], DR[dir], DC[dir]);

        auto [fdr, fdc] = decode_fire(a[1], a[2], cfg_.fire_range);
        int fire_span = 2 * cfg_.fire_range + 1;
        int offset_idx = (fdr + cfg_.fire_range) * fire_span + (fdc + cfg_.fire_range);
        if (offset_idx >= 0 && offset_idx < static_cast<int>(ep_fire_offset_counts_.size()))
            ++ep_fire_offset_counts_[offset_idx];

        int tr = agents_[i].cr + fdr;
        int tc = agents_[i].cc + fdc;
        ++ep_agent_shots_;
        if (tr < 0 || tr >= cfg_.M || tc < 0 || tc >= cfg_.M)
            ++ep_fire_oob_;

        int dist = nearest_live_boss_dist(tr, tc);
        if (dist != INT_MAX) {
            ep_fire_dist_sum_ += static_cast<float>(dist);
            ++ep_fire_dist_samples_;
            int bucket = std::min(dist, cfg_.fire_range + 1);
            if (bucket >= 0 && bucket < static_cast<int>(ep_fire_dist_counts_.size()))
                ++ep_fire_dist_counts_[bucket];
        }

        if (fire_at(false, tr, tc)) {
            reward += cfg_.reward_hit_boss;
            ++ep_boss_hits_;
        } else {
            reward += near_boss_reward(tr, tc);
        }
    }

    // Bosses act.
    for (int b = 0; b < cfg_.n_boss; ++b) {
        if (bosses_[b].sunk()) continue;
        auto [dir, fdr, fdc] = boss_policy(b);
        move_ship(bosses_[b], DR[dir], DC[dir]);
        if (fire_at(true, bosses_[b].cr + fdr, bosses_[b].cc + fdc)) {
            reward += cfg_.reward_hit_self;
            ++ep_agent_hits_;
        }
    }

    // Survival bonus: reward agents for keeping cells alive each step.
    if (cfg_.reward_survive != 0.f) {
        for (const auto& a : agents_)
            reward += cfg_.reward_survive * a.hp();
    }

    ++step_;

    bool all_boss  = std::all_of(bosses_.begin(), bosses_.end(),
                                 [](const Ship& s){ return s.sunk(); });
    bool all_agent = std::all_of(agents_.begin(), agents_.end(),
                                 [](const Ship& s){ return s.sunk(); });
    done_ = all_boss || all_agent || step_ >= cfg_.max_steps;
    if (all_boss)
        reward += cfg_.reward_agents_win;

    step_reward_ = reward;
    ep_reward_  += reward;

    for (int i = 0; i < cfg_.n_agents; ++i)
        next_obs_[i] = obs_for(i);
}

// ── Barrier interface ──────────────────────────────────────────────────────────

void BattleshipEnv::submit_action(int agent_id, std::span<const float> action) {
    if (submitted_ == 0) results_ready_ = false;
    pending_[agent_id].assign(action.begin(), action.end());
    if (++submitted_ == cfg_.n_agents) {
        step_env(pending_);
        submitted_     = 0;
        results_ready_ = true;
    }
}

cot::task<std::pair<std::vector<float>, float>>
BattleshipEnv::get_result(int agent_id) {
    while (!results_ready_) co_await cot::after(1us);
    co_return {next_obs_[agent_id], step_reward_};
}

// ── Sync interface ─────────────────────────────────────────────────────────────

BattleshipEnv::SyncResult
BattleshipEnv::sync_step(const std::vector<std::vector<float>>& actions) {
    step_env(actions);
    return {next_obs_, step_reward_};
}

// ── Observations ───────────────────────────────────────────────────────────────
//
// Local patch: (2·sight+1)² cells × 3 channels [own, ally, boss] + 2 scalars.
// All channels 0 for OOB and empty cells.
// Scalars: own_hp / 3, step / max_steps.

int BattleshipEnv::obs_dim() const {
    int patch = 2 * cfg_.sight_range + 1;
    return patch * patch * 3 + 2;
}

int BattleshipEnv::global_obs_dim() const {
    return cfg_.M * cfg_.M * 3 + cfg_.n_agents + cfg_.n_boss + 1;
}

std::vector<float> BattleshipEnv::obs_for(int agent_id) const {
    const Ship& me = agents_[agent_id];
    int sight = cfg_.sight_range;
    int M     = cfg_.M;

    std::vector<float> obs;
    obs.reserve(obs_dim());

    for (int dr = -sight; dr <= sight; ++dr) {
        for (int dc = -sight; dc <= sight; ++dc) {
            int r = me.cr + dr, c = me.cc + dc;
            if (r < 0 || r >= M || c < 0 || c >= M) {
                obs.insert(obs.end(), {0.f, 0.f, 0.f});
                continue;
            }
            const Cell& cell = at(r, c);
            float own  = (cell.type == 1 && cell.ship_id == (uint8_t)me.id) ? 1.f : 0.f;
            float ally = (cell.type == 1 && cell.ship_id != (uint8_t)me.id) ? 1.f : 0.f;
            float boss = (cell.type == 2)                                    ? 1.f : 0.f;
            obs.insert(obs.end(), {own, ally, boss});
        }
    }

    obs.push_back(me.hp() / 3.f);
    obs.push_back(step_ / (float)cfg_.max_steps);
    return obs;
}

// Full M×M grid: channels [agent_cell, boss_cell, padding] per cell.
// Agent/boss HP fractions and step appended.
std::vector<float> BattleshipEnv::global_obs() const {
    int M = cfg_.M;
    std::vector<float> obs;
    obs.reserve(global_obs_dim());

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < M; ++c) {
            const Cell& cell = at(r, c);
            obs.push_back(cell.type == 1 ? 1.f : 0.f);
            obs.push_back(cell.type == 2 ? 1.f : 0.f);
            obs.push_back(0.f);  // reserved
        }
    }
    for (const Ship& s : agents_) obs.push_back(s.hp() / 3.f);
    for (const Ship& s : bosses_) obs.push_back(s.hp() / 3.f);
    obs.push_back(step_ / (float)cfg_.max_steps);
    return obs;
}

// ── Episode stats ──────────────────────────────────────────────────────────────

BattleshipEnv::EpisodeStats BattleshipEnv::episode_stats() const {
    bool agents_won = std::all_of(bosses_.begin(), bosses_.end(),
                                  [](const Ship& s){ return s.sunk(); });
    bool boss_won   = std::all_of(agents_.begin(), agents_.end(),
                                  [](const Ship& s){ return s.sunk(); });
    float mean_fire_dist = ep_fire_dist_samples_ > 0
        ? ep_fire_dist_sum_ / static_cast<float>(ep_fire_dist_samples_)
        : 0.f;
    return {
        step_,
        ep_boss_hits_,
        ep_agent_hits_,
        agents_won,
        boss_won,
        ep_reward_,
        ep_agent_shots_,
        ep_fire_oob_,
        mean_fire_dist,
        ep_fire_dist_counts_,
        ep_move_counts_,
        ep_fire_offset_counts_,
    };
}

} // namespace seqcomm
