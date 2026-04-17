#include "agent_action.hh"
#include "gaussian_field_env.hh"
#include "cotamer/cotamer.hh"
#include "netsim.hh"
#include "pancy_msgs.hh"
#include "random_source.hh"
#include <list>
#include <memory>
#include <print>
#include <vector>

// seqcomm_sim.cc
//
//   Simulation harness for SeqComm on GaussianFieldEnv.
//
//   RandomNeuralModels is a stub that returns random tensors so the
//   cotamer task flow can be exercised end-to-end before plugging in
//   libtorch implementations.
//
//   Usage:
//     cmake -B build && cmake --build build --target seqcomm-sim
//     ./build/seqcomm-sim

namespace cot = cotamer;
using namespace seqcomm;

// ── Stub neural models ────────────────────────────────────────────────────────
//
//   Returns random values of the right shape. Replace with libtorch
//   implementations that actually learn; the interface stays identical.

struct RandomNeuralModels : NeuralModels {
    static constexpr int EMBED = 8;

    random_source& rng;
    int obs_dim;
    int n_agents;

    RandomNeuralModels(random_source& rng, int obs_dim, int n_agents)
        : rng(rng), obs_dim(obs_dim), n_agents(n_agents) {}

    std::vector<float> encode(std::span<const float>) override {
        std::vector<float> h(EMBED);
        for (auto& v : h) v = rng.uniform(-1.f, 1.f);
        return h;
    }

    std::vector<float> attention_a(
        std::span<const float>,
        const std::vector<std::vector<float>>&) override
    {
        std::vector<float> ctx(EMBED);
        for (auto& v : ctx) v = rng.uniform(-1.f, 1.f);
        return ctx;
    }

    std::vector<float> attention_w(
        const std::vector<std::vector<float>>&,
        const std::vector<std::vector<float>>&) override
    {
        std::vector<float> ctx(EMBED);
        for (auto& v : ctx) v = rng.uniform(-1.f, 1.f);
        return ctx;
    }

    // Action is a single-element vector holding the discrete index [0, N_ACTIONS)
    std::pair<std::vector<float>, float>
    policy_sample(std::span<const float>) override {
        float a = static_cast<float>(
            rng.uniform(0, GaussianFieldEnv::N_ACTIONS - 1));
        return {{a}, 0.f};
    }

    float policy_log_prob_old(std::span<const float>,
                              std::span<const float>) override { return 0.f; }

    float critic(std::span<const float>) override {
        return rng.uniform(-1.f, 1.f);
    }

    // World model predicts flattened (o'_all, r): n_agents*obs_dim + 1 values
    std::pair<std::vector<float>, float>
    world_model(std::span<const float>) override {
        std::vector<float> pred(n_agents * obs_dim);
        for (auto& v : pred) v = rng.uniform(-1.f, 1.f);
        return {pred, rng.uniform(-1.f, 1.f)};
    }
};


// ── Harness ───────────────────────────────────────────────────────────────────

int main() {
    constexpr int N = 4;    // agents
    constexpr int T = 10;   // timesteps per episode
    constexpr int H = 3;    // world-model rollout horizon for intention
    constexpr int F = 4;    // sampled orderings per intention estimate

    random_source rng;

    // Environment
    GaussianFieldEnv::Config env_cfg;
    env_cfg.n_agents = N;
    GaussianFieldEnv env(env_cfg, rng);

    // Shared trajectory buffer (filled by all agents during the episode)
    std::vector<transition> trajectory;

    // Neural models (stub — swap in libtorch implementations here)
    RandomNeuralModels models(rng, env.obs_dim(), N);

    // Agents — stored in a vector of unique_ptr so ports are never moved
    std::vector<std::unique_ptr<Agent>> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i)
        agents.push_back(std::make_unique<Agent>(
            i, env.obs_dim(), env.action_dim(),
            models, env, trajectory, rng));

    // Clique: fully connected (all agents in one neighbourhood).
    // clique[k] and to_neighbors[k] must have matching indices.
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (j != i) agents[i]->clique.push_back(j);

    // Channels: one per ordered pair (i→j).
    // Stored in a std::list so pointers remain stable (no reallocation).
    std::list<netsim::channel<pancy::agent_msg>> channels;
    for (int i = 0; i < N; ++i) {
        for (int j : agents[i]->clique) {
            channels.emplace_back(
                agents[j]->from_agents,
                "ch" + std::to_string(i) + "->" + std::to_string(j));
            agents[i]->to_neighbors.push_back(&channels.back());
        }
    }

    // Seed initial observations from the environment
    auto init_obs = env.reset();
    for (int i = 0; i < N; ++i)
        agents[i]->obs = init_obs[i];

    std::print("SeqComm: {} agents, T={} H={} F={} obs_dim={}\n",
               N, T, H, F, env.obs_dim());

    // Launch all agents as independent cotamer tasks and run the event loop
    for (auto& a : agents)
        a->run_episode(T, H, F).detach();

    cotamer::loop();

    // Summary
    float total_reward = 0.f;
    for (auto& tr : trajectory) total_reward += tr.reward;
    std::print("Episode done — transitions: {}  total_reward: {:.3f}\n",
               trajectory.size(), total_reward);

    return 0;
}
