#include "agent_action.hh"
#include "gaussian_field_env.hh"
#include "cotamer/cotamer.hh"
#include "netsim.hh"
#include "pancy_msgs.hh"
#include "random_source.hh"
#include "libtorch_models.hh"
#include "trajectory_io.hh"
#include <chrono>
#include <filesystem>
#include <list>
#include <memory>
#include <print>
#include <thread>
#include <vector>

// seqcomm_sim_trained.cc
//
//   C++ side of the C++ ↔ Python training loop.
//
//   Each episode:
//     1. Run one full SeqComm episode with LibtorchNeuralModels.
//     2. Write trajectory to <weights_dir>/traj.bin.
//     3. Touch <weights_dir>/traj.ready  → signals Python to update weights.
//     4. Poll for <weights_dir>/weights.ready  → Python is done.
//     5. Delete both sentinels, reload .pt files, repeat.
//
//   Usage:
//     cmake -B build -DCMAKE_PREFIX_PATH=$(python3 -c \
//       "import torch; print(torch.utils.cmake_prefix_path)")
//     cmake --build build --target seqcomm-sim-trained
//
//     # terminal 1 — start C++ loop first
//     ./build/robot_sim/seqcomm-sim-trained weights/ [N_EPISODES]
//
//     # terminal 2 — Python reads trajectories and updates weights
//     python -m training.train_from_cpp weights/

namespace cot = cotamer;
namespace fs  = std::filesystem;
using namespace seqcomm;
using namespace std::chrono_literals;

static constexpr int N          = 4;
static constexpr int T          = 200;
static constexpr int H          = 5;
static constexpr int F          = 4;
static constexpr int ACTION_DIM = 1;


// ── Sentinel helpers ──────────────────────────────────────────────────────────

static void touch(const fs::path& p) {
    std::ofstream f(p);  // creates or truncates; Python sees non-zero size isn't required
}

static void wait_for(const fs::path& p) {
    while (!fs::exists(p))
        std::this_thread::sleep_for(100ms);
}


// ── Single episode ────────────────────────────────────────────────────────────
//
//   Agents, channels, and the cotamer scheduler are recreated each episode
//   so there is no coroutine or message-queue state leakage across episodes.

static float run_one_episode(
    GaussianFieldEnv& env,
    LibtorchNeuralModels& models,
    random_source& rng,
    std::vector<transition>& trajectory)
{
    trajectory.clear();

    std::vector<std::unique_ptr<Agent>> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i)
        agents.push_back(std::make_unique<Agent>(
            i, env.obs_dim(), ACTION_DIM,
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

    for (auto& a : agents)
        a->run_episode(T, H, F).detach();

    cotamer::loop();

    float total_reward = 0.f;
    for (auto& tr : trajectory) total_reward += tr.reward;
    return total_reward;
}


// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::print(stderr,
            "usage: seqcomm-sim-trained <weights_dir> [n_episodes]\n"
            "  weights_dir: directory with .pt files written by train.py\n"
            "  n_episodes:  default 2000\n");
        return 1;
    }

    const std::string weights_dir = argv[1];
    const int n_episodes = (argc >= 3) ? std::stoi(argv[2]) : 2000;

    random_source rng;

    GaussianFieldEnv::Config env_cfg;
    env_cfg.n_agents = N;
    GaussianFieldEnv env(env_cfg, rng);

    LibtorchNeuralModels models(weights_dir, N);

    std::vector<transition> trajectory;
    trajectory.reserve(static_cast<size_t>(T * N));

    const fs::path traj_bin   = fs::path(weights_dir) / "traj.bin";
    const fs::path traj_ready = fs::path(weights_dir) / "traj.ready";
    const fs::path wts_ready  = fs::path(weights_dir) / "weights.ready";

    std::print("SeqComm C++ training loop: {} agents  T={}  H={}  F={}  "
               "obs_dim={}  episodes={}\n",
               N, T, H, F, env.obs_dim(), n_episodes);
    std::print("Trajectory file : {}\n", traj_bin.string());
    std::print("Waiting for Python to launch (python -m training.train_from_cpp {})\n\n",
               weights_dir);

    float reward_sum = 0.f;

    for (int ep = 0; ep < n_episodes; ++ep) {
        float ep_reward = run_one_episode(env, models, rng, trajectory);
        reward_sum += ep_reward;

        // Serialize trajectory and signal Python
        write_trajectory(traj_bin.string(), trajectory,
                         N, env.obs_dim(), ACTION_DIM);
        touch(traj_ready);

        std::print("ep {:4d}  R={:8.2f}  avg={:8.2f}  "
                   "transitions={}  waiting for Python…\n",
                   ep, ep_reward, reward_sum / (ep + 1),
                   trajectory.size());

        // Wait for updated weights
        wait_for(wts_ready);
        fs::remove(traj_ready);
        fs::remove(wts_ready);

        // Reload .pt files with fresh weights
        models.reload(weights_dir);

        std::print("  → weights reloaded\n");
    }

    std::print("\nDone. {} episodes  avg_reward={:.2f}\n",
               n_episodes, reward_sum / n_episodes);
    return 0;
}
