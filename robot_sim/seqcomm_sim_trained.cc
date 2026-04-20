#include "agent_action.hh"
#include "gaussian_field_env.hh"
#include "cotamer/cotamer.hh"
#include "netsim.hh"
#include "pancy_msgs.hh"
#include "random_source.hh"
#include "libtorch_models.hh"
#include "trajectory_io.hh"
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <print>
#include <sstream>
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
//     cmake -B build -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
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


// ── Episode stats ─────────────────────────────────────────────────────────────

struct EpisodeStats {
    float total_reward;
    bool success;
    int steps_to_completion;
    bool deadlock;
    int n_collisions;
    int n_goals_reached;
    int n_msgs_dropped;
    std::array<std::array<int, N>, T> ordering_history;
    std::array<std::array<float, N>, T> intention_history;
};


static double round_to(double value, int decimals) {
    const double scale = std::pow(10.0, decimals);
    return std::round(value * scale) / scale;
}

static std::string json_number(double value, int decimals) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << round_to(value, decimals);
    std::string s = oss.str();
    auto dot = s.find('.');
    if (dot == std::string::npos) return s;

    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.push_back('0');
    return s;
}

static std::string now_iso8601_local() {
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

static std::array<int, N> first_mover_counts(const EpisodeStats& stats) {
    std::array<int, N> counts{};
    counts.fill(0);
    for (int t = 0; t < T; ++t) {
        const int first = stats.ordering_history[t][0];
        if (first >= 0 && first < N)
            counts[first]++;
    }
    return counts;
}

static double order_entropy(const EpisodeStats& stats) {
    const auto counts = first_mover_counts(stats);
    double entropy = 0.0;
    for (int c : counts) {
        if (c <= 0) continue;
        const double p = static_cast<double>(c) / static_cast<double>(T);
        entropy -= p * std::log(p);
    }
    return entropy;
}

static double mean_intention_spread(const EpisodeStats& stats) {
    if (N <= 1) return 0.0;

    double spread_sum = 0.0;
    for (int t = 0; t < T; ++t) {
        double mean = 0.0;
        for (int i = 0; i < N; ++i)
            mean += static_cast<double>(stats.intention_history[t][i]);
        mean /= static_cast<double>(N);

        double sum_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            const double d = static_cast<double>(stats.intention_history[t][i]) - mean;
            sum_sq += d * d;
        }
        spread_sum += std::sqrt(sum_sq / static_cast<double>(N - 1));
    }
    return spread_sum / static_cast<double>(T);
}

static void write_log_meta(std::ofstream& log_file, int obs_dim, int n_episodes) {
    log_file
        << "{\"_meta\": {"
        << "\"env\": \"gaussian\", "
        << "\"mode\": \"seqcomm\", "
        << "\"compute_intention\": true, "
        << "\"share_actions\": true, "
        << "\"ordering\": \"intention\", "
        << "\"comm_delay\": 0, "
        << "\"comm_drop_prob\": 0.0, "
        << "\"comm_noise_std\": 0.0, "
        << "\"comm_bandwidth_bits\": 0, "
        << "\"obs_noise_std\": 0.0, "
        << "\"wm_H\": " << H << ", "
        << "\"wm_F\": " << F << ", "
        << "\"seed\": -1, "
        << "\"n_agents\": " << N << ", "
        << "\"obs_dim\": " << obs_dim << ", "
        << "\"embed_dim\": 64, "
        << "\"episodes\": " << n_episodes << ", "
        << "\"episode_len\": " << T << ", "
        << "\"H_train\": " << H << ", "
        << "\"F_train\": " << F << ", "
        << "\"gamma\": " << json_number(0.99, 2) << ", "
        << "\"lam\": " << json_number(0.95, 2) << ", "
        << "\"lr_world\": " << json_number(3e-4, 6) << ", "
        << "\"lr_policy\": " << json_number(3e-4, 6) << ", "
        << "\"timestamp\": \"" << now_iso8601_local() << "\""
        << "}}\n";
    log_file.flush();
}

static void write_episode_log(std::ofstream& log_file, int episode, const EpisodeStats& stats) {
    const auto counts = first_mover_counts(stats);

    log_file
        << "{"
        << "\"episode\": " << episode << ", "
        << "\"total_reward\": " << json_number(stats.total_reward, 4) << ", "
        << "\"success\": " << (stats.success ? "true" : "false") << ", "
        << "\"steps_to_completion\": " << stats.steps_to_completion << ", "
        << "\"deadlock\": " << (stats.deadlock ? "true" : "false") << ", "
        << "\"n_collisions\": " << stats.n_collisions << ", "
        << "\"n_goals_reached\": " << stats.n_goals_reached << ", "
        << "\"n_msgs_dropped\": " << stats.n_msgs_dropped << ", "
        << "\"order_entropy\": " << json_number(order_entropy(stats), 4) << ", "
        << "\"mean_intention_spread\": " << json_number(mean_intention_spread(stats), 4) << ", "
        << "\"first_mover_counts\": ["
        << counts[0] << ", " << counts[1] << ", " << counts[2] << ", " << counts[3] << "], "
        << "\"world_model_loss\": " << json_number(0.0, 6) << ", "
        << "\"value_loss\": " << json_number(0.0, 6) << ", "
        << "\"policy_loss\": " << json_number(0.0, 6)
        << "}\n";
    log_file.flush();
}


// ── Single episode ────────────────────────────────────────────────────────────
//
//   Agents, channels, and the cotamer scheduler are recreated each episode
//   so there is no coroutine or message-queue state leakage across episodes.

static EpisodeStats run_one_episode(
    GaussianFieldEnv& env,
    LibtorchNeuralModels& models,
    random_source& rng,
    std::vector<transition>& trajectory,
    bool verbose)
{
    trajectory.clear();

    std::vector<std::unique_ptr<Agent>> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i)
        agents.push_back(std::make_unique<Agent>(
            i, env.obs_dim(), ACTION_DIM,
            models, env, trajectory, rng, verbose));

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

    // Wire up per-agent intention buffers for spread computation
    std::vector<std::vector<float>> agent_intentions(N);
    for (int i = 0; i < N; ++i)
        agents[i]->own_intentions = &agent_intentions[i];

    auto init_obs = env.reset();
    for (int i = 0; i < N; ++i)
        agents[i]->obs = init_obs[i];

    for (auto& a : agents)
        a->run_episode(T, H, F).detach();

    cotamer::loop();

    // ── Compute stats from trajectory ─────────────────────────────────────────
    EpisodeStats s{};
    s.success = false;               // GaussianFieldEnv never returns done=true
    s.steps_to_completion = T;       // mirrors train.py's EPISODE_LEN default
    s.deadlock = true;               // matches step_done == EPISODE_LEN in train.py
    s.n_collisions = 0;              // intersection-only metric
    s.n_goals_reached = 0;           // intersection-only metric
    s.n_msgs_dropped = 0;            // no CommChannel stressors in this C++ path
    for (auto& order : s.ordering_history) order.fill(-1);
    for (auto& intents : s.intention_history) intents.fill(0.f);

    for (auto& tr : trajectory) {
        s.total_reward += tr.reward;

        const int t = tr.timestep;
        const int rank = static_cast<int>(tr.upper_actions.size());
        if (t >= 0 && t < T && rank >= 0 && rank < N)
            s.ordering_history[t][rank] = tr.agent_id;
    }

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            if (t < static_cast<int>(agent_intentions[i].size()))
                s.intention_history[t][i] = agent_intentions[i][t];
        }
    }

    return s;
}


// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::print(stderr,
            "usage: seqcomm-sim-trained <weights_dir> [n_episodes] [--verbose]\n"
            "  weights_dir: directory with .pt files written by train.py\n"
            "  n_episodes:  default 2000\n"
            "  --verbose:   print per-timestep agent ordering\n"
            "  OMP_NUM_THREADS=N controls libtorch CPU thread count\n");
        return 1;
    }

    const std::string weights_dir = argv[1];
    int  n_episodes = 2000;
    bool verbose    = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose") verbose    = true;
        else                    n_episodes = std::stoi(arg);
    }

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
    const fs::path traj_done  = fs::path(weights_dir) / "traj.done";

    // Remove stale sentinels from any previous run so we don't misread them.
    for (auto* p : {&traj_ready, &wts_ready, &traj_done}) {
        std::error_code ec;
        fs::remove(*p, ec);
    }

    std::print("SeqComm C++ training loop: {} agents  T={}  H={}  F={}  "
               "obs_dim={}  episodes={}\n",
               N, T, H, F, env.obs_dim(), n_episodes);
    std::print("Trajectory file : {}\n", traj_bin.string());
    std::print("Waiting for Python to launch (python -m training.train_from_cpp {})\n\n",
               weights_dir);

    const fs::path log_dir  = fs::current_path() / "logs" / "cpp_runs_only";
    const fs::path log_path = log_dir / "seqcomm_sim_trained.jsonl";
    fs::create_directories(log_dir);
    std::ofstream log_file(log_path, std::ios::out | std::ios::trunc);
    write_log_meta(log_file, env.obs_dim(), n_episodes);

    float reward_sum = 0.f;

    for (int ep = 0; ep < n_episodes; ++ep) {
        auto stats = run_one_episode(env, models, rng, trajectory, verbose);
        write_episode_log(log_file, ep, stats);

        reward_sum += stats.total_reward;

        // Serialize trajectory and signal Python
        write_trajectory(traj_bin.string(), trajectory,
                         N, env.obs_dim(), ACTION_DIM);
        touch(traj_ready);

        std::print("ep {:4d}  R={:8.2f}  avg={:8.2f}  "
                   "success={}  steps={}  deadlock={}  "
                   "collisions={}  goals={}  msgs_dropped={}  "
                   "first_order=[{},{},{},{}]  "
                   "transitions={}  waiting for Python…\n",
                   ep, stats.total_reward, reward_sum / (ep + 1),
                   stats.success, stats.steps_to_completion, stats.deadlock,
                   stats.n_collisions, stats.n_goals_reached, stats.n_msgs_dropped,
                   stats.ordering_history[0][0], stats.ordering_history[0][1],
                   stats.ordering_history[0][2], stats.ordering_history[0][3],
                   trajectory.size());

        // Wait for updated weights
        wait_for(wts_ready);
        fs::remove(traj_ready);
        fs::remove(wts_ready);

        // Update weights in-place from raw binary blob (faster than full .pt reload)
        models.update_from_blob(weights_dir);

        std::print("  → weights reloaded\n");
    }

    touch(traj_done);
    std::print("\nDone. {} episodes  avg_reward={:.2f}\n",
               n_episodes, reward_sum / n_episodes);
    return 0;
}
