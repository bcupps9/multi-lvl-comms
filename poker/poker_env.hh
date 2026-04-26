#pragma once
// poker_env.hh
//
// Team Poker environment — implements seqcomm::Environment.
//
// Game per hand:
//   - Agent i privately observes H_i ~ U[0,1]
//   - Dealer D ~ U[0,1] revealed after bets
//   - Win (H̄ > D):  C_i *= (1 + M * rho_i)
//   - Lose (H̄ ≤ D): C_i *= (1 + alpha) * (1 - rho_i)
//   - rho_i = sigmoid(action_i)  — bet fraction
//   - Reward = mean log-growth across agents
//
// See math.md for optimal thresholds and overgeneralization gap.

#include "agent_action.hh"
#include "random_source.hh"
#include <cmath>
#include <span>
#include <vector>

namespace seqcomm {
namespace cot = cotamer;

struct PokerConfig {
    int   n_agents  = 4;
    float M         = 2.0f;     // net profit multiplier on win
    float alpha     = 0.1f;     // consolation fraction on loss
    float C_0       = 100.0f;   // initial coffers per agent
    float C_floor   = 1.0f;    // bankruptcy threshold
    float C_target  = 1000.0f;  // success threshold
    int   K_max     = 150;      // max hands per episode
};

// Per-hand record — filled by step_env() after every hand.
struct PokerHandResult {
    std::vector<float> hands;           // H_i drawn this hand
    float              H_bar;           // team mean hand strength
    float              dealer;          // D ~ U[0,1]
    bool               win;             // H_bar > dealer
    std::vector<float> rhos;            // bet fractions (post-sigmoid)
    std::vector<float> coffers_before;
    std::vector<float> coffers_after;
    float              reward;          // mean log-growth
};

struct PokerEnv : Environment {
    explicit PokerEnv(PokerConfig cfg, random_source& rng);

    // ── Barrier-based interface (for cotamer modes) ──────────────────────────
    //
    // First submit of each new hand resets results_ready_.
    // Last submit triggers step_env and sets results_ready_.
    // get_result polls until results_ready_, then returns.

    void submit_action(int agent_id,
                       std::span<const float> action) override;

    cot::task<std::pair<std::vector<float>, float>>
    get_result(int agent_id) override;

    // ── Sync interface (for mappo / oracle, no cotamer) ──────────────────────

    struct SyncResult {
        std::vector<std::vector<float>> next_obs;
        float                           reward;
    };
    // raw_actions: pre-sigmoid policy outputs (one per agent)
    SyncResult sync_step(std::span<const float> raw_actions);

    // ── Episode control ───────────────────────────────────────────────────────

    // Returns initial obs (one per agent); resets all state.
    std::vector<std::vector<float>> reset();

    int  obs_dim()    const { return cfg_.n_agents + 2; }
    int  action_dim() const { return 1; }
    bool is_done()    const { return done_; }

    // H_bar of the hands currently dealt (before this step's bets).
    // Only valid between reset()/step and the next hand being dealt.
    float current_H_bar() const {
        float s = 0.f;
        for (float h : hands_) s += h;
        return s / static_cast<float>(cfg_.n_agents);
    }

    // ── Per-step / per-episode records ───────────────────────────────────────

    PokerHandResult last_hand;                    // most recent hand
    std::vector<PokerHandResult> hand_history;    // all hands this episode

    // ── Episode stats (valid after is_done() == true) ────────────────────────

    struct EpisodeStats {
        int                hands_played;
        int                wins;
        float              mean_bet;
        bool               bankruptcy;
        bool               target_hit;
        std::vector<float> final_coffers;
    };
    EpisodeStats episode_stats() const;

    // ── Derived quantities (from math.md) ────────────────────────────────────

    float theta_opt() const {
        return (1.f + cfg_.alpha) / (cfg_.M + 1.f + cfg_.alpha);
    }
    float kelly_fraction(float H_bar) const {
        return std::max(0.f, (H_bar * (cfg_.M + 1.f) - 1.f) / cfg_.M);
    }

    const PokerConfig& config() const { return cfg_; }

private:
    PokerConfig    cfg_;
    random_source& rng_;

    // Episode state
    std::vector<float> coffers_;
    std::vector<float> hands_;
    int   t_    = 0;
    bool  done_ = false;

    // Per-step results (set by step_env, read by get_result / sync_step)
    std::vector<std::vector<float>> next_obs_;
    float                           step_reward_ = 0.f;

    // Barrier state (pending_rhos_ must be declared after next_obs_ for init order)
    std::vector<float> pending_rhos_;
    int  submitted_     = 0;
    bool results_ready_ = false;

    // Episode accumulators
    int   ep_hands_   = 0;
    int   ep_wins_    = 0;
    float ep_sum_bet_ = 0.f;

    static float sigmoid(float x);
    void deal_hands();
    void step_env(const std::vector<float>& rhos);
    std::vector<float> obs_for(int agent_id) const;
};

} // namespace seqcomm
