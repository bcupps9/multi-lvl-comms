#include "gaussian_field_env.hh"
#include <algorithm>
#include <cmath>
#include <print>

namespace seqcomm {
using namespace std::chrono_literals;

// ── Construction / reset ──────────────────────────────────────────────────────

GaussianFieldEnv::GaussianFieldEnv(Config cfg, random_source& rng)
    : cfg_(cfg), rng_(rng),
      gaussians_(cfg.n_gaussians),
      positions_(cfg.n_agents),
      pending_actions_(cfg.n_agents, -1),
      next_obs_(cfg.n_agents)
{
    reset();
}

std::vector<std::vector<float>> GaussianFieldEnv::reset() {
    const float G = static_cast<float>(cfg_.grid_size);

    // Randomise Gaussian centers and velocities
    for (auto& g : gaussians_) {
        g.x  = rng_.uniform(0.f, G);
        g.y  = rng_.uniform(0.f, G);
        float angle = rng_.uniform(0.f, 6.2831853f);
        g.vx = cfg_.move_speed * std::cos(angle);
        g.vy = cfg_.move_speed * std::sin(angle);
    }

    // Randomise agent positions
    for (auto& p : positions_) {
        p[0] = rng_.uniform(0, cfg_.grid_size - 1);
        p[1] = rng_.uniform(0, cfg_.grid_size - 1);
    }

    // Reset barrier
    submitted_     = 0;
    results_ready_ = false;
    std::fill(pending_actions_.begin(), pending_actions_.end(), -1);

    std::vector<std::vector<float>> obs;
    obs.reserve(cfg_.n_agents);
    for (int i = 0; i < cfg_.n_agents; ++i)
        obs.push_back(obs_for(i));
    return obs;
}

int GaussianFieldEnv::obs_dim() const {
    int w = 2 * cfg_.window_half + 1;
    return 2 + w * w;
}

int GaussianFieldEnv::action_dim() const { return N_ACTIONS; }


// ── Field ─────────────────────────────────────────────────────────────────────

float GaussianFieldEnv::field_at(float row, float col) const {
    float val = 0.f;
    float inv2s2 = 1.f / (2.f * cfg_.gaussian_sigma * cfg_.gaussian_sigma);
    for (auto& g : gaussians_) {
        float dr = row - g.x;
        float dc = col - g.y;
        val += std::exp(-(dr*dr + dc*dc) * inv2s2);
    }
    return val;
}

std::vector<float> GaussianFieldEnv::obs_for(int agent_id) const {
    const float G = static_cast<float>(cfg_.grid_size);
    const int   w = cfg_.window_half;
    int row = positions_[agent_id][0];
    int col = positions_[agent_id][1];

    std::vector<float> o;
    o.reserve(obs_dim());
    o.push_back(row / G);
    o.push_back(col / G);

    for (int dr = -w; dr <= w; ++dr) {
        for (int dc = -w; dc <= w; ++dc) {
            int nr = row + dr, nc = col + dc;
            if (nr >= 0 && nr < cfg_.grid_size && nc >= 0 && nc < cfg_.grid_size)
                o.push_back(field_at(static_cast<float>(nr),
                                     static_cast<float>(nc)));
            else
                o.push_back(0.f);
        }
    }
    return o;
}


// ── Step ──────────────────────────────────────────────────────────────────────

void GaussianFieldEnv::apply_and_step() {
    // Action deltas: stay, up, down, left, right
    constexpr int DR[5] = {0, -1, 1,  0, 0};
    constexpr int DC[5] = {0,  0, 0, -1, 1};
    const int G = cfg_.grid_size;

    for (int i = 0; i < cfg_.n_agents; ++i) {
        int a = pending_actions_[i];
        if (a < 0 || a >= N_ACTIONS) a = 0;
        positions_[i][0] = std::clamp(positions_[i][0] + DR[a], 0, G - 1);
        positions_[i][1] = std::clamp(positions_[i][1] + DC[a], 0, G - 1);
    }

    // Compute reward
    float reward = 0.f;
    for (int i = 0; i < cfg_.n_agents; ++i)
        reward += field_at(static_cast<float>(positions_[i][0]),
                           static_cast<float>(positions_[i][1]));
    for (int i = 0; i < cfg_.n_agents; ++i)
        for (int j = i + 1; j < cfg_.n_agents; ++j)
            if (positions_[i] == positions_[j])
                reward -= cfg_.overlap_penalty;
    step_reward_ = reward;

    // Advance Gaussians (bounce off walls)
    const float Gf = static_cast<float>(G);
    for (auto& g : gaussians_) {
        g.x += g.vx;
        g.y += g.vy;
        if (g.x < 0.f)  { g.x = -g.x;          g.vx = std::abs(g.vx);  }
        if (g.x >= Gf)  { g.x = 2.f*Gf - g.x;  g.vx = -std::abs(g.vx); }
        if (g.y < 0.f)  { g.y = -g.y;           g.vy = std::abs(g.vy);  }
        if (g.y >= Gf)  { g.y = 2.f*Gf - g.y;  g.vy = -std::abs(g.vy); }
    }

    // Build next observations
    for (int i = 0; i < cfg_.n_agents; ++i)
        next_obs_[i] = obs_for(i);
}


// ── Environment interface ─────────────────────────────────────────────────────

void GaussianFieldEnv::submit_action(int agent_id,
                                     std::span<const float> action) {
    // First submission of a new round: clear results from previous round
    if (submitted_ == 0)
        results_ready_ = false;

    pending_actions_[agent_id] = static_cast<int>(action[0]);
    ++submitted_;

    if (submitted_ == cfg_.n_agents) {
        apply_and_step();
        submitted_     = 0;
        results_ready_ = true;
    }
}

cot::task<std::pair<std::vector<float>, float>>
GaussianFieldEnv::get_result(int agent_id) {
    // Yield in a tight loop until all agents have submitted and the step ran.
    // Each co_await gives other cotamer tasks (other agents) a chance to run.
    while (!results_ready_)
        co_await cot::after(1us);
    co_return {next_obs_[agent_id], step_reward_};
}

} // namespace seqcomm
