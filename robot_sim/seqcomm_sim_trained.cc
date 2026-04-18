#include "agent_action.hh"
#include "gaussian_field_env.hh"
#include "cotamer/cotamer.hh"
#include "netsim.hh"
#include "pancy_msgs.hh"
#include "random_source.hh"
#include "libtorch_models.hh"
#include <list>
#include <memory>
#include <print>
#include <vector>

// seqcomm_sim_trained.cc
//
//   Like seqcomm_sim.cc but uses LibTorchNeuralModels loaded from .pt files
//   produced by `python -m training.train --weights-dir <dir>`.
//
//   Usage:
//     cmake -B build -DCMAKE_PREFIX_PATH=$(python3 -c \
//       "import torch; print(torch.utils.cmake_prefix_path)")
//     cmake --build build --target seqcomm-sim-trained
//     ./build/robot_sim/seqcomm-sim-trained weights/

namespace cot = cotamer;
using namespace seqcomm;

// Must match the constants used during training (training/train.py).
static constexpr int N          = 4;    // N_AGENTS
static constexpr int T          = 200;  // EPISODE_LEN
static constexpr int H          = 5;    // world-model horizon
static constexpr int F          = 4;    // orderings per intention
static constexpr int EMBED_DIM  = 64;
static constexpr int ACTION_DIM = 1;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::print(stderr,
            "usage: seqcomm-sim-trained <weights_dir>\n"
            "  weights_dir: directory containing encoder.pt, attn_a.pt, …\n");
        return 1;
    }
    const std::string weights_dir = argv[1];

    random_source rng;

    GaussianFieldEnv::Config env_cfg;
    env_cfg.n_agents = N;
    GaussianFieldEnv env(env_cfg, rng);

    std::vector<transition> trajectory;

    LibTorchNeuralModels models(weights_dir, N, ACTION_DIM);

    std::vector<std::unique_ptr<Agent>> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i)
        agents.push_back(std::make_unique<Agent>(
            i, env.obs_dim(), env.action_dim(),
            models, env, trajectory, rng));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (j != i) agents[i]->clique.push_back(j);

    std::list<netsim::channel<pancy::agent_msg>> channels;
    for (int i = 0; i < N; ++i) {
        for (int j : agents[i]->clique) {
            channels.emplace_back(
                agents[j]->from_agents,
                "ch" + std::to_string(i) + "->" + std::to_string(j));
            agents[i]->to_neighbors.push_back(&channels.back());
        }
    }

    auto init_obs = env.reset();
    for (int i = 0; i < N; ++i)
        agents[i]->obs = init_obs[i];

    std::print("SeqComm (trained): {} agents  T={}  H={}  F={}  obs_dim={}\n",
               N, T, H, F, env.obs_dim());

    for (auto& a : agents)
        a->run_episode(T, H, F).detach();

    cotamer::loop();

    float total_reward = 0.f;
    for (auto& tr : trajectory) total_reward += tr.reward;
    std::print("Episode done — transitions: {}  total_reward: {:.3f}\n",
               trajectory.size(), total_reward);

    return 0;
}
