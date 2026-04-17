#pragma once
#include "agent_action.hh"
#include "random_source.hh"
#include <array>
#include <vector>

// gaussian_field_env.hh
//
//   Concrete Environment for testing SeqComm.
//
//   World: grid_size × grid_size 2D grid, N agents, K moving Gaussians.
//
//   Observation o_i: [row/G, col/G,  field_window_flat]
//     - normalized position (2 values)
//     - (2*window_half+1)^2 field samples centered on the agent, zero-padded
//       at borders
//
//   Action: single float cast to int — 0=stay 1=up 2=down 3=left 4=right
//
//   Reward (joint, same for all agents at a timestep):
//     sum_i F(p_i)  −  overlap_penalty × |{(i,j) : p_i == p_j, i<j}|
//
//   Gaussians bounce off walls each step, so the field is predictable but
//   non-trivial — exactly what the world model M needs to learn.
//
//   Barrier: submit_action() stores each agent's action; the last agent to
//   submit triggers the environment step. get_result() is a cotamer task
//   that polls until results are ready, yielding between polls so other
//   agents can submit.

namespace seqcomm {

struct GaussianFieldEnv : Environment {

    struct Config {
        int   grid_size       = 20;
        int   window_half     = 2;     // observation window radius; window = (2w+1)²
        int   n_gaussians     = 3;
        float gaussian_sigma  = 2.5f;
        float move_speed      = 0.4f;  // Gaussian center displacement per step
        float overlap_penalty = 1.0f;  // λ
        int   n_agents        = 4;
    };

    static constexpr int N_ACTIONS = 5;  // stay, up, down, left, right

    explicit GaussianFieldEnv(Config cfg, random_source& rng);

    // ── Environment interface ─────────────────────────────────────────────────

    // Stores action; when all n_agents have submitted, steps the environment.
    void submit_action(int agent_id,
                       std::span<const float> action) override;

    // Suspends until all agents have submitted (barrier), then returns result.
    cot::task<std::pair<std::vector<float>, float>>
    get_result(int agent_id) override;

    // ── Harness helpers ───────────────────────────────────────────────────────

    // Reset field and agent positions; returns initial obs for each agent.
    std::vector<std::vector<float>> reset();

    int obs_dim()    const;  // 2 + (2*window_half+1)^2
    int action_dim() const;  // N_ACTIONS

private:
    Config cfg_;
    random_source& rng_;

    struct Gaussian { float x, y, vx, vy; };
    std::vector<Gaussian>          gaussians_;
    std::vector<std::array<int,2>> positions_;  // [row, col] per agent

    // Barrier state — reset on first submit of each new round
    std::vector<int> pending_actions_;  // action index per agent
    int  submitted_    = 0;
    bool results_ready_ = false;

    // Computed once per step, read by all agents via get_result
    std::vector<std::vector<float>> next_obs_;
    float step_reward_ = 0.f;

    float              field_at(float row, float col) const;
    std::vector<float> obs_for(int agent_id) const;
    void               apply_and_step();
};

} // namespace seqcomm
