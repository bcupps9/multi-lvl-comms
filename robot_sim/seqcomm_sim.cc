#include "agent_action.hh"
#include "gaussian_field_env.hh"
#include "intersection_env.hh"
#include "libtorch_models.hh"
#include "cotamer/cotamer.hh"
#include "netsim.hh"
#include "pancy_msgs.hh"
#include "random_source.hh"
#include <list>
#include <memory>
#include <print>
#include <string>
#include <vector>

// seqcomm_sim.cc
//
//   Simulation harness for SeqComm on GaussianFieldEnv.
//
//   Two model back-ends:
//     RandomNeuralModels (default) — random tensors; validates coroutine flow.
//     LibtorchNeuralModels         — loads TorchScript weights from disk.
//
//   Usage:
//     cmake -B build && cmake --build build --target seqcomm-sim
//     ./build/robot_sim/seqcomm-sim                           # random models
//
//   With libtorch (train first: python training/train.py --save-weights weights/):
//     cmake -B build -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
//     cmake --build build --target seqcomm-sim
//     ./build/robot_sim/seqcomm-sim --weights ./weights

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

int main(int argc, char* argv[]) {
    constexpr int N = 4;    // agents
    constexpr int T = 10;   // timesteps per episode
    constexpr int H = 3;    // world-model rollout horizon for intention
    constexpr int F = 4;    // sampled orderings per intention estimate

    // Parse flags: --weights <dir>  --env <gaussian|intersection>
    std::string weights_dir;
    std::string env_name = "gaussian";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--weights" && i + 1 < argc)
            weights_dir = argv[++i];
        else if (std::string(argv[i]) == "--env" && i + 1 < argc)
            env_name = argv[++i];
    }

    random_source rng;

    // Environment — choose at runtime; query dims before erasing the type.
    int obs_dim_val, action_dim_val;
    std::unique_ptr<Environment> env_ptr;
    std::vector<std::vector<float>> init_obs;

    if (env_name == "intersection") {
        IntersectionCrossingEnv::Config cfg;
        cfg.n_agents = N;
        auto e = std::make_unique<IntersectionCrossingEnv>(cfg);
        obs_dim_val    = e->obs_dim();
        action_dim_val = e->action_dim();
        init_obs       = e->reset();
        env_ptr        = std::move(e);
    } else {
        if (env_name != "gaussian")
            std::print("Warning: unknown --env '{}', defaulting to gaussian\n", env_name);
        GaussianFieldEnv::Config cfg;
        cfg.n_agents = N;
        auto e = std::make_unique<GaussianFieldEnv>(cfg, rng);
        obs_dim_val    = e->obs_dim();
        action_dim_val = e->action_dim();
        init_obs       = e->reset();
        env_ptr        = std::move(e);
    }
    Environment& env = *env_ptr;

    // Shared trajectory buffer (filled by all agents during the episode)
    std::vector<transition> trajectory;

    // Neural models — choose back-end at runtime.
    //
    // unique_ptr<NeuralModels> lets both branches share the rest of the
    // harness without duplicating agent construction or the event loop.
    std::unique_ptr<NeuralModels> models_ptr;

#ifdef USE_TORCH
    if (!weights_dir.empty()) {
        std::print("Loading TorchScript weights from '{}'\n", weights_dir);
        models_ptr = std::make_unique<LibtorchNeuralModels>(weights_dir, N);
    } else {
        std::print("USE_TORCH enabled but no --weights given; using random models\n");
        models_ptr = std::make_unique<RandomNeuralModels>(rng, obs_dim_val, N);
    }
#else
    if (!weights_dir.empty())
        std::print("Warning: --weights ignored (build with -DUSE_TORCH=ON to enable)\n");
    models_ptr = std::make_unique<RandomNeuralModels>(rng, obs_dim_val, N);
#endif

    NeuralModels& models = *models_ptr;

    // Agents — stored in a vector of unique_ptr so ports are never moved
    std::vector<std::unique_ptr<Agent>> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i)
        agents.push_back(std::make_unique<Agent>(
            i, obs_dim_val, action_dim_val,
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
    for (int i = 0; i < N; ++i)
        agents[i]->obs = init_obs[i];

    std::print("SeqComm [{}]: {} agents, T={} H={} F={} obs_dim={}\n",
               env_name, N, T, H, F, obs_dim_val);

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
