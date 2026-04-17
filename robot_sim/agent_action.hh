#pragma once
#include "cotamer/cotamer.hh"
#include "netsim.hh"
#include "pancy_msgs.hh"
#include "random_source.hh"
#include <deque>
#include <span>
#include <unordered_map>
#include <vector>

namespace seqcomm {
namespace cot = cotamer;

struct NeuralModels {
    virtual ~NeuralModels() = default;
    virtual std::vector<float> encode(std::span<const float> obs) = 0;
    virtual std::vector<float> attention_a(
        std::span<const float> h_self,
        const std::vector<std::vector<float>>& messages) = 0;
    virtual std::vector<float> attention_w(
        const std::vector<std::vector<float>>& enc_obs,
        const std::vector<std::vector<float>>& actions) = 0;
    virtual std::pair<std::vector<float>, float>
    policy_sample(std::span<const float> context) = 0;
    virtual float policy_log_prob_old(
        std::span<const float> context,
        std::span<const float> action) = 0;
    virtual float critic(std::span<const float> context) = 0;
    virtual std::pair<std::vector<float>, float>
    world_model(std::span<const float> context_w) = 0;
};

struct Environment {
    virtual ~Environment() = default;
    virtual void submit_action(int agent_id,
                               std::span<const float> action) = 0;
    virtual std::pair<std::vector<float>, float>
    get_result(int agent_id) = 0;
};

struct transition {
    int agent_id;
    int timestep;
    std::vector<float> obs;
    std::vector<float> action;
    std::vector<std::vector<float>> upper_actions;
    std::vector<float> next_obs;
    float reward       = 0.f;
    float value        = 0.f;
    float log_prob     = 0.f;
    float log_prob_old = 0.f;
};

struct Agent {
    int id;
    int obs_dim;
    int action_dim;

    netsim::port<pancy::agent_msg> from_agents;
    std::vector<netsim::channel<pancy::agent_msg>*> to_neighbors;

    std::vector<int> clique;
    std::vector<int> N_upper;
    std::vector<int> N_lower;

    NeuralModels& models;
    Environment&  env;
    std::vector<transition>& trajectory;

    std::vector<float> obs;
    std::vector<float> h;
    std::vector<float> action;

    Agent(int id, int obs_dim, int action_dim,
          NeuralModels& models, Environment& env,
          std::vector<transition>& trajectory,
          random_source& rng);

    cot::task<void> negotiation_phase(int t, int H, int F);
    cot::task<void> launching_phase(int t);
    cot::task<void> run_episode(int T, int H, int F);

private:
    random_source& rng_;

    cot::task<float> compute_intention(
        const std::vector<pancy::hidden_state_msg>& neighbor_hs,
        int H, int F);

    template <typename MsgT>
    cot::task<MsgT> receive_typed();

    cot::task<void> send_to(int neighbor_id, pancy::agent_msg msg);
    cot::task<void> broadcast(pancy::agent_msg msg);

    std::deque<pancy::agent_msg> pending_;
};

template <typename MsgT>
cot::task<MsgT> Agent::receive_typed() {
    for (auto it = pending_.begin(); it != pending_.end(); ++it) {
        if (std::holds_alternative<MsgT>(*it)) {
            MsgT m = std::get<MsgT>(std::move(*it));
            pending_.erase(it);
            co_return m;
        }
    }
    while (true) {
        auto msg = co_await from_agents.receive();
        if (std::holds_alternative<MsgT>(msg)) {
            co_return std::get<MsgT>(std::move(msg));
        }
        pending_.push_back(std::move(msg));
    }
}

} // namespace seqcomm
