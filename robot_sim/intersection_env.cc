#include "intersection_env.hh"
#include <algorithm>
#include <cmath>

namespace seqcomm {
using namespace std::chrono_literals;

// ── Construction / reset ──────────────────────────────────────────────────────

IntersectionCrossingEnv::IntersectionCrossingEnv(Config cfg)
    : cfg_(cfg),
      pending_actions_(cfg.n_agents, -1),
      next_obs_(cfg.n_agents)
{
    reset();
}

std::vector<std::vector<float>> IntersectionCrossingEnv::reset() {
    for (int i = 0; i < cfg_.n_agents; ++i) {
        positions_[i]    = STARTS[i];
        reached_goal_[i] = false;
    }
    submitted_     = 0;
    results_ready_ = false;
    std::fill(pending_actions_.begin(), pending_actions_.end(), -1);

    std::vector<std::vector<float>> obs;
    obs.reserve(cfg_.n_agents);
    for (int i = 0; i < cfg_.n_agents; ++i)
        obs.push_back(obs_for(i));
    return obs;
}


// ── Observation ───────────────────────────────────────────────────────────────

std::vector<float> IntersectionCrossingEnv::obs_for(int agent_id) const {
    const float G   = static_cast<float>(cfg_.grid_size);
    const float cr  = 9.5f;   // geometric center row of intersection zone
    const float cc  = 9.5f;   // geometric center col of intersection zone
    auto [r, c]     = positions_[agent_id];
    auto [gr, gc]   = GOALS[agent_id];
    float dist      = (std::abs(r - gr) + std::abs(c - gc)) / (2.f * G);
    float in_zone   = (r >= 9 && r <= 10 && c >= 9 && c <= 10) ? 1.f : 0.f;
    return {
        r  / G,
        c  / G,
        gr / G,
        gc / G,
        (r - cr) / G,
        (c - cc) / G,
        dist,
        in_zone,
    };
}


// ── Step ──────────────────────────────────────────────────────────────────────

void IntersectionCrossingEnv::apply_and_step() {
    constexpr int DR[5] = {0, -1, 1,  0, 0};
    constexpr int DC[5] = {0,  0, 0, -1, 1};
    const int G = cfg_.grid_size;

    // Move agents that haven't reached their goal
    for (int i = 0; i < cfg_.n_agents; ++i) {
        if (reached_goal_[i]) continue;
        int a = pending_actions_[i];
        if (a < 0 || a >= N_ACTIONS) a = 0;
        auto& [r, c] = positions_[i];
        r = std::clamp(r + DR[a], 0, G - 1);
        c = std::clamp(c + DC[a], 0, G - 1);
    }

    float reward = -cfg_.step_penalty * static_cast<float>(cfg_.n_agents);

    // Goal bonus (one-time per agent)
    for (int i = 0; i < cfg_.n_agents; ++i) {
        if (!reached_goal_[i] && positions_[i] == GOALS[i]) {
            reached_goal_[i] = true;
            reward += cfg_.goal_reward;
        }
    }

    // Collision penalty (per pair per step, only for agents still active)
    for (int i = 0; i < cfg_.n_agents; ++i) {
        if (reached_goal_[i]) continue;
        for (int j = i + 1; j < cfg_.n_agents; ++j) {
            if (!reached_goal_[j] && positions_[i] == positions_[j])
                reward -= cfg_.collision_penalty;
        }
    }

    step_reward_ = reward;

    for (int i = 0; i < cfg_.n_agents; ++i)
        next_obs_[i] = obs_for(i);
}


// ── Environment interface ─────────────────────────────────────────────────────

void IntersectionCrossingEnv::submit_action(int agent_id,
                                            std::span<const float> action) {
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
IntersectionCrossingEnv::get_result(int agent_id) {
    while (!results_ready_)
        co_await cot::after(1us);
    co_return {next_obs_[agent_id], step_reward_};
}

} // namespace seqcomm
