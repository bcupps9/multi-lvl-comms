#pragma once
#include "agent_action.hh"
#include <array>
#include <utility>
#include <vector>

// intersection_env.hh
//
//   C++ mirror of execution/intersection_env.py.
//
//   Four agents cross a shared 2×2 intersection on a 20×20 grid.
//   Each agent has a fixed start and goal on opposite sides; all four
//   routes cross the same four central cells (rows 9–10, cols 9–10).
//
//   This stresses SeqComm's priority-ordering mechanism: agents that
//   commit first (highest intention) traverse the intersection; others
//   wait or reroute to avoid collisions.
//
//   Observation o_i (obs_dim = 8, partial — no other agents visible):
//     [row/G, col/G, goal_row/G, goal_col/G,
//      Δrow_to_center/G, Δcol_to_center/G,
//      dist_to_goal/(2·G), in_intersection_flag]
//
//   Actions: 0=stay  1=up(−row)  2=down(+row)  3=left(−col)  4=right(+col)
//
//   Reward (shared):
//     −step_penalty · n_agents per step
//     +goal_reward when an agent first reaches its goal
//     −collision_penalty per colliding pair per step
//
//   done: all agents at goal — run_episode still runs for T steps; done is
//   reported through the reward signal rather than terminating the loop.

namespace seqcomm {

struct IntersectionCrossingEnv : Environment {

    struct Config {
        int   grid_size         = 20;
        float step_penalty      = 0.02f;
        float goal_reward       = 10.0f;
        float collision_penalty = 5.0f;
        int   n_agents          = 4;
    };

    static constexpr int N_ACTIONS = 5;  // stay, up, down, left, right

    explicit IntersectionCrossingEnv(Config cfg);

    // ── Environment interface ─────────────────────────────────────────────────

    void submit_action(int agent_id,
                       std::span<const float> action) override;

    cot::task<std::pair<std::vector<float>, float>>
    get_result(int agent_id) override;

    // ── Harness helpers ───────────────────────────────────────────────────────

    std::vector<std::vector<float>> reset();
    int obs_dim()    const { return 8; }
    int action_dim() const { return N_ACTIONS; }

private:
    Config cfg_;

    // starts[i] and goals[i] for agents 0–3 (N→S, S→N, W→E, E→W)
    static constexpr std::array<std::pair<int,int>, 4> STARTS = {{
        {1, 10}, {18, 10}, {10, 1}, {10, 18}
    }};
    static constexpr std::array<std::pair<int,int>, 4> GOALS = {{
        {18, 10}, {1, 10}, {10, 18}, {10, 1}
    }};

    std::array<std::pair<int,int>, 4> positions_;
    std::array<bool, 4>               reached_goal_;

    // Barrier state
    std::vector<int> pending_actions_;
    int  submitted_     = 0;
    bool results_ready_ = false;

    std::vector<std::vector<float>> next_obs_;
    float step_reward_ = 0.f;

    std::vector<float> obs_for(int agent_id) const;
    void               apply_and_step();
};

} // namespace seqcomm
