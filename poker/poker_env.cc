#include "poker_env.hh"
#include <chrono>
#include <numeric>

namespace seqcomm {
namespace cot = cotamer;
using namespace std::chrono_literals;

// ── Construction ───────────────────────────────────────────────────────────────

PokerEnv::PokerEnv(PokerConfig cfg, random_source& rng)
    : cfg_(cfg), rng_(rng),
      coffers_(cfg.n_agents, cfg.C_0),
      hands_(cfg.n_agents, 0.f),
      next_obs_(cfg.n_agents),
      pending_rhos_(cfg.n_agents, 0.f)
{}

// ── Reset ──────────────────────────────────────────────────────────────────────

std::vector<std::vector<float>> PokerEnv::reset() {
    std::fill(coffers_.begin(), coffers_.end(), cfg_.C_0);
    t_             = 0;
    done_          = false;
    ep_hands_      = 0;
    ep_wins_       = 0;
    ep_sum_bet_    = 0.f;
    submitted_     = 0;
    results_ready_ = false;

    hand_history.clear();
    hand_history.reserve(cfg_.K_max);

    deal_hands();

    std::vector<std::vector<float>> obs;
    obs.reserve(cfg_.n_agents);
    for (int i = 0; i < cfg_.n_agents; ++i)
        obs.push_back(obs_for(i));
    return obs;
}

// ── Barrier-based interface ────────────────────────────────────────────────────

void PokerEnv::submit_action(int agent_id, std::span<const float> action) {
    if (submitted_ == 0)
        results_ready_ = false;  // first submission of new hand resets flag

    pending_rhos_[agent_id] = sigmoid(action[0]);

    if (++submitted_ == cfg_.n_agents) {
        step_env(pending_rhos_);
        submitted_     = 0;
        results_ready_ = true;
    }
}

cot::task<std::pair<std::vector<float>, float>>
PokerEnv::get_result(int agent_id) {
    while (!results_ready_)
        co_await cot::after(1us);
    co_return {next_obs_[agent_id], step_reward_};
}

// ── Sync interface ─────────────────────────────────────────────────────────────

PokerEnv::SyncResult PokerEnv::sync_step(std::span<const float> raw_actions) {
    std::vector<float> rhos(cfg_.n_agents);
    for (int i = 0; i < cfg_.n_agents; ++i)
        rhos[i] = sigmoid(raw_actions[i]);
    step_env(rhos);
    return {next_obs_, step_reward_};
}

// ── Core step ──────────────────────────────────────────────────────────────────

void PokerEnv::step_env(const std::vector<float>& rhos) {
    const int   N = cfg_.n_agents;
    const float M = cfg_.M;
    const float a = cfg_.alpha;

    float H_bar = 0.f;
    for (float h : hands_) H_bar += h;
    H_bar /= static_cast<float>(N);

    float dealer = rng_.uniform(0.f, 1.f);
    bool  win    = H_bar > dealer;

    last_hand.hands          = hands_;
    last_hand.H_bar          = H_bar;
    last_hand.dealer         = dealer;
    last_hand.win            = win;
    last_hand.rhos           = rhos;
    last_hand.coffers_before = coffers_;

    float log_growth_sum = 0.f;
    float bet_sum        = 0.f;
    for (int i = 0; i < N; ++i) {
        float rho   = rhos[i];
        float C_old = coffers_[i];
        float C_new = win ? C_old * (1.f + M * rho)
                          : C_old * (1.f + a) * (1.f - rho);
        coffers_[i]   = C_new;
        log_growth_sum += std::log(C_new / C_old);
        bet_sum        += rho;
    }
    step_reward_ = log_growth_sum / static_cast<float>(N);

    last_hand.coffers_after = coffers_;
    last_hand.reward        = step_reward_;
    hand_history.push_back(last_hand);

    ep_hands_  += 1;
    ep_wins_   += int(win);
    ep_sum_bet_ += bet_sum / static_cast<float>(N);

    bool bankrupt   = false;
    bool target_hit = false;
    for (float c : coffers_) {
        if (c <= cfg_.C_floor)  bankrupt   = true;
        if (c >= cfg_.C_target) target_hit = true;
    }
    ++t_;
    done_ = bankrupt || target_hit || (t_ >= cfg_.K_max);

    if (!done_) deal_hands();
    for (int i = 0; i < N; ++i)
        next_obs_[i] = obs_for(i);
}

// ── Observations ──────────────────────────────────────────────────────────────
//
// obs_i = [H_i,  log(C_0/C_0), …, log(C_{N-1}/C_0),  t/K_max]
//   dim  = 1  +        N                            +    1       = N+2

std::vector<float> PokerEnv::obs_for(int agent_id) const {
    const int N = cfg_.n_agents;
    std::vector<float> o;
    o.reserve(N + 2);
    o.push_back(hands_[agent_id]);
    for (int j = 0; j < N; ++j)
        o.push_back(std::log(coffers_[j] / cfg_.C_0));
    o.push_back(static_cast<float>(t_) / static_cast<float>(cfg_.K_max));
    std::print("obs_for agent_id={} => [{}]\n", agent_id, o);
    return o;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

void PokerEnv::deal_hands() {
    for (float& h : hands_)
        h = rng_.uniform(0.f, 1.f);
}

float PokerEnv::sigmoid(float x) {
    if (x >= 0.f) return 1.f / (1.f + std::exp(-x));
    float ex = std::exp(x);
    return ex / (1.f + ex);
}

PokerEnv::EpisodeStats PokerEnv::episode_stats() const {
    int n = std::max(1, ep_hands_);
    bool bankrupt   = false;
    bool target_hit = false;
    for (float c : coffers_) {
        if (c <= cfg_.C_floor)  bankrupt   = true;
        if (c >= cfg_.C_target) target_hit = true;
    }
    return {
        .hands_played  = ep_hands_,
        .wins          = ep_wins_,
        .mean_bet      = ep_sum_bet_ / static_cast<float>(n),
        .bankruptcy    = bankrupt,
        .target_hit    = target_hit,
        .final_coffers = coffers_,
    };
}

} // namespace seqcomm
