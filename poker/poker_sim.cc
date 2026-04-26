// poker_sim.cc
//
// Team Poker simulation harness for SeqComm overgeneralization testing.
//
// Five modes (--mode):
//   seqcomm      — Full SeqComm: intention ordering + action cascade (trains)
//   fixed_order  — Fixed ordering 0→1→2→3, action cascade, no negotiation
//   random_order — Random ordering each hand, action cascade, no negotiation
//   mappo        — Independent: each agent uses own obs only, no cascade
//   oracle       — Analytical Kelly fraction given true H̄ (upper bound)
//
// Logging:
//   Normal:  per-episode JSON lines → logs/poker/<mode>_<ts>.jsonl
//   Verbose: per-hand + per-negotiation events → logs/poker/verbose/<mode>_<ts>.jsonl
//
// Training loop (seqcomm mode only):
//   C++ writes traj.bin, touches traj.ready
//   Python updates weights, touches weights.ready
//   C++ reloads, repeats
//
// Build:
//   cmake -B build -DUSE_TORCH=ON \
//         -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
//   cmake --build build --target poker-sim
//
// Run:
//   # terminal 1
//   ./build/poker/poker-sim weights/ --mode seqcomm --episodes 2000
//   # terminal 2
//   python -m training.train_from_cpp weights/

#include "poker_env.hh"
#include "agent_action.hh"
#include "libtorch_models.hh"
#include "pancy_msgs.hh"
#include "trajectory_io.hh"
#include "random_source.hh"
#include "netsim.hh"
#include "cotamer/cotamer.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <numeric>
#include <print>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cot = cotamer;
namespace fs  = std::filesystem;
using namespace seqcomm;
using namespace std::chrono_literals;

// H=2: fast rollouts, still sufficient for SeqComm ordering signal
// F=4: four orderings sampled per intention computation
static constexpr int H_ROLLOUT   = 2;
static constexpr int F_ORDERINGS = 4;


// ── Mode ───────────────────────────────────────────────────────────────────────

enum class Mode { SEQCOMM, FIXED_ORDER, RANDOM_ORDER, MAPPO, ORACLE };

static Mode parse_mode(const std::string& s) {
    if (s == "seqcomm")      return Mode::SEQCOMM;
    if (s == "fixed_order")  return Mode::FIXED_ORDER;
    if (s == "random_order") return Mode::RANDOM_ORDER;
    if (s == "mappo")        return Mode::MAPPO;
    if (s == "oracle")       return Mode::ORACLE;
    throw std::runtime_error("unknown --mode: " + s +
        " (seqcomm|fixed_order|random_order|mappo|oracle)");
}

static std::string mode_str(Mode m) {
    switch (m) {
        case Mode::SEQCOMM:      return "seqcomm";
        case Mode::FIXED_ORDER:  return "fixed_order";
        case Mode::RANDOM_ORDER: return "random_order";
        case Mode::MAPPO:        return "mappo";
        case Mode::ORACLE:       return "oracle";
    }
    return "unknown";
}


// ── Episode stats ──────────────────────────────────────────────────────────────

struct PokerEpisodeStats {
    float total_reward          = 0.f;
    int   hands_played          = 0;
    float win_rate              = 0.f;
    float mean_bet              = 0.f;
    bool  bankruptcy            = false;
    bool  target_hit            = false;
    float order_entropy         = 0.f;
    float mean_intention_spread = 0.f;
    std::vector<int> first_mover_counts;   // size N
    float world_model_loss = 0.f;
    float value_loss       = 0.f;
    float policy_loss      = 0.f;
};


// ── JSON helpers ───────────────────────────────────────────────────────────────

static std::string jn(double v, int d = 4) {
    double s = std::pow(10.0, d);
    double r = std::round(v * s) / s;
    std::ostringstream o;
    o << std::fixed << std::setprecision(d) << r;
    std::string t = o.str();
    while (t.size() > 1 && t.back() == '0') t.pop_back();
    if (t.back() == '.') t.push_back('0');
    return t;
}

static std::string jb(bool v) { return v ? "true" : "false"; }

static std::string now_iso() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

static std::string now_ts() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

static void touch(const fs::path& p) { std::ofstream f(p); (void)f; }

static void wait_for(const fs::path& p) {
    while (!fs::exists(p)) std::this_thread::sleep_for(100ms);
}


// ── Ordering helpers ───────────────────────────────────────────────────────────

static std::vector<std::vector<int>> precompute_orderings(
    int K_max, int N, bool randomise, random_source& rng)
{
    std::vector<int> base(N);
    std::iota(base.begin(), base.end(), 0);
    std::vector<std::vector<int>> result(K_max, base);
    if (randomise)
        for (auto& o : result)
            std::shuffle(o.begin(), o.end(), rng.engine());
    return result;
}


// ── Cotamer episode tasks ──────────────────────────────────────────────────────

// Full SeqComm: negotiation phase (intention ordering) + launching phase per hand.
// Loop exits when env signals done.
static cot::task<void> poker_episode_seqcomm(Agent& agent, PokerEnv& env) {
    for (int t = 0; !env.is_done(); ++t) {
        co_await agent.negotiation_phase(t, H_ROLLOUT, F_ORDERINGS);
        co_await agent.launching_phase(t);
    }
}

// Ordered episode: ordering pre-set by harness; launching only (no negotiation).
// Each agent determines its own rank from the shared ordering vector.
static cot::task<void> poker_episode_ordered(
    Agent& agent, PokerEnv& env,
    const std::vector<std::vector<int>>& orderings)
{
    const int N = static_cast<int>(orderings.empty() ? 0 : orderings[0].size());
    for (int t = 0; !env.is_done() && t < static_cast<int>(orderings.size()); ++t) {
        const auto& ord = orderings[t];
        // Set this agent's rank in the predetermined ordering
        agent.N_upper.clear();
        agent.N_lower.clear();
        for (int r = 0; r < N; ++r) {
            if (ord[r] == agent.id) {
                for (int r2 = 0; r2 < r; ++r2)
                    agent.N_upper.push_back(ord[r2]);
                for (int r2 = r + 1; r2 < N; ++r2)
                    agent.N_lower.push_back(ord[r2]);
                break;
            }
        }
        // Manual encode (negotiation_phase normally does this)
        agent.h = agent.models.encode(agent.obs);
        co_await agent.launching_phase(t);
    }
}


// ── Cotamer episode runner ─────────────────────────────────────────────────────

struct CotamerResult {
    PokerEpisodeStats             stats;
    std::vector<transition>       trajectory;
    std::vector<std::vector<float>> intention_log;  // [t][agent_i]
    std::vector<std::vector<int>>   ordering_log;   // [t][rank] = agent_id
};

static CotamerResult run_cotamer_episode(
    PokerEnv& env,
    NeuralModels& models,
    random_source& rng,
    Mode mode,
    bool verbose)
{
    CotamerResult out;
    auto& traj = out.trajectory;  // agents record into this via reference

    const int N = env.config().n_agents;

    // Create fresh agents each episode (cotamer has per-task state)
    std::vector<std::unique_ptr<Agent>> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i)
        agents.push_back(std::make_unique<Agent>(
            i, env.obs_dim(), env.action_dim(),
            models, env, traj, rng, verbose));

    // Full clique (all-to-all)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (j != i) agents[i]->clique.push_back(j);

    // Channels: agent i → port of agent j
    std::list<netsim::channel<pancy::agent_msg>> channels;
    for (int i = 0; i < N; ++i) {
        for (int j : agents[i]->clique) {
            channels.emplace_back(
                agents[j]->from_agents,
                "pk" + std::to_string(i) + "->" + std::to_string(j));
            agents[i]->to_neighbors.push_back(&channels.back());
        }
    }

    // Intention buffers — filled by negotiation_phase for seqcomm mode
    std::vector<std::vector<float>> agent_intentions(N);
    for (int i = 0; i < N; ++i)
        agents[i]->own_intentions = &agent_intentions[i];

    auto init_obs = env.reset();
    for (int i = 0; i < N; ++i)
        agents[i]->obs = init_obs[i];

    // Detach per-agent episode tasks
    if (mode == Mode::SEQCOMM) {
        for (auto& a : agents) {
            poker_episode_seqcomm(*a, env).detach();
            std::print("releasing an agent\n");
        }
    } else {
        auto orderings = precompute_orderings(
            env.config().K_max, N,
            mode == Mode::RANDOM_ORDER, rng);
        for (auto& a : agents)
            poker_episode_ordered(*a, env, orderings).detach();
    }

    // Run event loop until all agent tasks complete
    cot::loop();

    // ── Collect stats ──────────────────────────────────────────────────────────

    auto env_s = env.episode_stats();
    int T = env_s.hands_played;

    auto& s = out.stats;
    s.hands_played = T;
    s.win_rate     = T > 0 ? static_cast<float>(env_s.wins) / T : 0.f;
    s.mean_bet     = env_s.mean_bet;
    s.bankruptcy   = env_s.bankruptcy;
    s.target_hit   = env_s.target_hit;

    // Total reward: sum per-hand reward (agent 0's transitions, one per hand)
    for (auto& tr : traj)
        if (tr.agent_id == 0)
            s.total_reward += tr.reward;

    // Ordering history from trajectory (rank = upper_actions.size())
    out.ordering_log.assign(T, std::vector<int>(N, -1));
    s.first_mover_counts.assign(N, 0);
    for (auto& tr : traj) {
        int t    = tr.timestep;
        int rank = static_cast<int>(tr.upper_actions.size());
        if (t >= 0 && t < T && rank >= 0 && rank < N)
            out.ordering_log[t][rank] = tr.agent_id;
    }
    for (int t = 0; t < T; ++t) {
        int first = out.ordering_log[t].empty() ? -1 : out.ordering_log[t][0];
        if (first >= 0 && first < N)
            s.first_mover_counts[first]++;
    }

    // Order entropy over first-mover distribution
    {
        double ent = 0.0;
        for (int c : s.first_mover_counts) {
            if (c <= 0) continue;
            double p = static_cast<double>(c) / T;
            ent -= p * std::log(p);
        }
        s.order_entropy = static_cast<float>(ent);
    }

    // Intention spread (seqcomm only; zero for ordered modes)
    out.intention_log.assign(T, std::vector<float>(N, 0.f));
    for (int i = 0; i < N; ++i) {
        auto& iv = agent_intentions[i];
        for (int t = 0; t < std::min(T, static_cast<int>(iv.size())); ++t)
            out.intention_log[t][i] = iv[t];
    }
    if (mode == Mode::SEQCOMM && T > 1) {
        double spread_sum = 0.0;
        for (int t = 0; t < T; ++t) {
            double mean = 0.0;
            for (float v : out.intention_log[t]) mean += v;
            mean /= N;
            double sq = 0.0;
            for (float v : out.intention_log[t]) sq += (v - mean) * (v - mean);
            spread_sum += std::sqrt(sq / (N - 1));
        }
        s.mean_intention_spread = static_cast<float>(spread_sum / T);
    }

    return out;
}


// ── Sync episode runner (mappo / oracle) ───────────────────────────────────────

static PokerEpisodeStats run_sync_episode(
    PokerEnv& env,
    NeuralModels* models,            // null for oracle
    std::vector<transition>& traj,
    bool /*verbose*/)
{
    const int N = env.config().n_agents;

    auto obs_all = env.reset();
    traj.clear();

    for (int t = 0; !env.is_done(); ++t) {
        std::vector<float>              raw_actions(N);
        std::vector<std::vector<float>> actions(N, std::vector<float>(1, 0.f));
        std::vector<float>              log_probs(N, 0.f);
        std::vector<float>              log_probs_old(N, 0.f);
        std::vector<float>              values(N, 0.f);

        if (models) {
            // MAPPO: each agent acts on own obs only, no upper-action context
            for (int i = 0; i < N; ++i) {
                auto h       = models->encode(obs_all[i]);
                auto ctx     = models->attention_a(h, {});
                auto [a, lp] = models->policy_sample(ctx);
                float lp_old = models->policy_log_prob_old(ctx, a);
                float v      = models->critic(ctx);
                raw_actions[i]   = a[0];
                actions[i]       = std::move(a);
                log_probs[i]     = lp;
                log_probs_old[i] = lp_old;
                values[i]        = v;
            }
        } else {
            // Oracle: analytical Kelly fraction given the true current H̄
            float H_bar  = env.current_H_bar();
            float rho    = env.kelly_fraction(H_bar);
            // Inverse sigmoid so sync_step's sigmoid gives back rho
            float raw = (rho <= 0.f) ? -10.f
                      : (rho >= 1.f) ?  10.f
                      : std::log(rho / (1.f - rho));
            std::fill(raw_actions.begin(), raw_actions.end(), raw);
            for (int i = 0; i < N; ++i) actions[i] = {raw};
        }

        auto result = env.sync_step(raw_actions);
        float reward = result.reward;

        for (int i = 0; i < N; ++i) {
            traj.push_back({
                .agent_id     = i,
                .timestep     = t,
                .obs          = obs_all[i],
                .action       = actions[i],
                .upper_actions = {},          // no cascade in these modes
                .next_obs     = result.next_obs[i],
                .reward       = reward,
                .value        = values[i],
                .log_prob     = log_probs[i],
                .log_prob_old = log_probs_old[i],
            });
        }
        obs_all = std::move(result.next_obs);
    }

    auto env_s = env.episode_stats();
    int T = env_s.hands_played;

    PokerEpisodeStats s;
    s.hands_played = T;
    s.win_rate     = T > 0 ? static_cast<float>(env_s.wins) / T : 0.f;
    s.mean_bet     = env_s.mean_bet;
    s.bankruptcy   = env_s.bankruptcy;
    s.target_hit   = env_s.target_hit;
    // Count agent-0 transitions to avoid N-fold double counting
    for (auto& tr : traj)
        if (tr.agent_id == 0)
            s.total_reward += tr.reward;
    s.first_mover_counts.assign(env.config().n_agents, 0);
    return s;
}


// ── Logging ────────────────────────────────────────────────────────────────────

static void write_meta(std::ofstream& f, Mode mode, const PokerConfig& cfg,
                       int n_episodes) {
    f << "{\"_meta\": {"
      << "\"env\": \"poker\", "
      << "\"mode\": \"" << mode_str(mode) << "\", "
      << "\"n_agents\": " << cfg.n_agents << ", "
      << "\"M\": " << jn(cfg.M, 2) << ", "
      << "\"alpha\": " << jn(cfg.alpha, 3) << ", "
      << "\"C_0\": " << jn(cfg.C_0, 1) << ", "
      << "\"C_floor\": " << jn(cfg.C_floor, 1) << ", "
      << "\"C_target\": " << jn(cfg.C_target, 1) << ", "
      << "\"K_max\": " << cfg.K_max << ", "
      << "\"H\": " << H_ROLLOUT << ", "
      << "\"F\": " << F_ORDERINGS << ", "
      << "\"episodes\": " << n_episodes << ", "
      << "\"timestamp\": \"" << now_iso() << "\""
      << "}}\n";
    f.flush();
}

static void write_episode_log(std::ofstream& f, int ep,
                               const PokerEpisodeStats& s) {
    f << "{"
      << "\"episode\": " << ep << ", "
      << "\"total_reward\": " << jn(s.total_reward) << ", "
      << "\"hands_played\": " << s.hands_played << ", "
      << "\"win_rate\": " << jn(s.win_rate) << ", "
      << "\"mean_bet\": " << jn(s.mean_bet) << ", "
      << "\"bankruptcy\": " << jb(s.bankruptcy) << ", "
      << "\"target_hit\": " << jb(s.target_hit) << ", "
      << "\"order_entropy\": " << jn(s.order_entropy) << ", "
      << "\"mean_intention_spread\": " << jn(s.mean_intention_spread) << ", "
      << "\"first_mover_counts\": [";
    for (int i = 0; i < static_cast<int>(s.first_mover_counts.size()); ++i) {
        if (i) f << ", ";
        f << s.first_mover_counts[i];
    }
    f << "], "
      << "\"world_model_loss\": " << jn(s.world_model_loss, 6) << ", "
      << "\"value_loss\": " << jn(s.value_loss, 6) << ", "
      << "\"policy_loss\": " << jn(s.policy_loss, 6)
      << "}\n";
    f.flush();
}

// Verbose: one JSON event per hand
static void write_hand_event(std::ofstream& f, int ep, int t,
                              const PokerHandResult& h) {
    f << "{\"event\": \"hand\", "
      << "\"episode\": " << ep << ", \"t\": " << t << ", "
      << "\"H_bar\": " << jn(h.H_bar) << ", "
      << "\"dealer\": " << jn(h.dealer) << ", "
      << "\"win\": " << jb(h.win) << ", "
      << "\"reward\": " << jn(h.reward) << ", "
      << "\"hands\": [";
    for (int i = 0; i < static_cast<int>(h.hands.size()); ++i) {
        if (i) f << ", ";
        f << jn(h.hands[i]);
    }
    f << "], \"rhos\": [";
    for (int i = 0; i < static_cast<int>(h.rhos.size()); ++i) {
        if (i) f << ", ";
        f << jn(h.rhos[i]);
    }
    f << "], \"coffers\": [";
    for (int i = 0; i < static_cast<int>(h.coffers_after.size()); ++i) {
        if (i) f << ", ";
        f << jn(h.coffers_after[i], 2);
    }
    f << "]}\n";
    f.flush();
}

// Verbose: negotiation outcome (seqcomm mode only)
static void write_negotiation_event(std::ofstream& f, int ep, int t,
                                     const std::vector<float>& intentions,
                                     const std::vector<int>& ordering) {
    f << "{\"event\": \"negotiation\", "
      << "\"episode\": " << ep << ", \"t\": " << t << ", "
      << "\"n_msgs\": {"
      << "\"hidden_state\": " << (intentions.size() * (intentions.size() - 1)) << ", "
      << "\"intention\": " << (intentions.size() * (intentions.size() - 1)) << ", "
      << "\"upper_action\": " << (intentions.size() * (intentions.size() - 1) / 2) << ", "
      << "\"execute\": " << (intentions.size() - 1)
      << "}, \"intentions\": [";
    for (int i = 0; i < static_cast<int>(intentions.size()); ++i) {
        if (i) f << ", ";
        f << jn(intentions[i]);
    }
    f << "], \"ordering\": [";
    for (int i = 0; i < static_cast<int>(ordering.size()); ++i) {
        if (i) f << ", ";
        f << ordering[i];
    }
    f << "]}\n";
    f.flush();
}


// ── main ───────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string weights_dir;
    Mode        mode      = Mode::SEQCOMM;
    int         n_ep      = 1000;
    bool        verbose   = false;
    bool        do_train  = true;
    std::string log_dir_str;
    int         seed      = -1;
    PokerConfig pcfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--verbose")              verbose  = true;
        else if (arg == "--no-train")             do_train = false;
        else if (arg == "--mode"     && i+1 < argc) mode        = parse_mode(argv[++i]);
        else if (arg == "--episodes" && i+1 < argc) n_ep        = std::stoi(argv[++i]);
        else if (arg == "--log-dir"  && i+1 < argc) log_dir_str = argv[++i];
        else if (arg == "--seed"     && i+1 < argc) seed        = std::stoi(argv[++i]);
        else if (arg == "--M"        && i+1 < argc) pcfg.M        = std::stof(argv[++i]);
        else if (arg == "--alpha"    && i+1 < argc) pcfg.alpha    = std::stof(argv[++i]);
        else if (arg == "--C0"       && i+1 < argc) pcfg.C_0      = std::stof(argv[++i]);
        else if (arg == "--floor"    && i+1 < argc) pcfg.C_floor  = std::stof(argv[++i]);
        else if (arg == "--target"   && i+1 < argc) pcfg.C_target = std::stof(argv[++i]);
        else if (arg == "--K"        && i+1 < argc) pcfg.K_max    = std::stoi(argv[++i]);
        else if (arg == "--agents"   && i+1 < argc) pcfg.n_agents = std::stoi(argv[++i]);
        else if (arg[0] != '-')                     weights_dir   = arg;
        else {
            std::print(stderr,
                "usage: poker-sim [<weights_dir>] [--mode seqcomm|fixed_order|"
                "random_order|mappo|oracle]\n"
                "  [--episodes N] [--no-train] [--verbose] [--log-dir PATH]\n"
                "  [--seed N] [--M F] [--alpha F] [--C0 F] [--floor F]\n"
                "  [--target F] [--K N] [--agents N]\n");
            return 1;
        }
    }

    bool need_models = (mode != Mode::ORACLE);
    if (need_models && weights_dir.empty()) {
        std::print(stderr, "error: weights_dir required for mode '{}'\n",
                   mode_str(mode));
        return 1;
    }
    if (mode != Mode::SEQCOMM) do_train = false;

    random_source rng;
    if (seed >= 0) rng.seed(static_cast<uint64_t>(seed));

    PokerEnv env(pcfg, rng);

    // Load neural models (not needed for oracle)
    std::unique_ptr<LibtorchNeuralModels> models_ptr;
    if (need_models)
        models_ptr = std::make_unique<LibtorchNeuralModels>(weights_dir, pcfg.n_agents);
    NeuralModels* models = models_ptr.get();

    // Sentinel paths (used only in training mode)
    fs::path traj_bin   = fs::path(weights_dir) / "traj.bin";
    fs::path traj_ready = fs::path(weights_dir) / "traj.ready";
    fs::path wts_ready  = fs::path(weights_dir) / "weights.ready";
    fs::path traj_done  = fs::path(weights_dir) / "traj.done";

    if (do_train) {
        for (auto* p : {&traj_ready, &wts_ready, &traj_done}) {
            std::error_code ec;
            fs::remove(*p, ec);
        }
    }

    // Log directory and files
    fs::path log_dir = log_dir_str.empty()
        ? fs::current_path() / "logs" / "poker"
        : fs::path(log_dir_str);
    fs::create_directories(log_dir);

    std::string ts = now_ts();
    fs::path log_path = log_dir / (mode_str(mode) + "_" + ts + ".jsonl");
    std::ofstream log_file(log_path, std::ios::out | std::ios::trunc);

    std::unique_ptr<std::ofstream> vlog;
    if (verbose) {
        fs::path vdir = log_dir / "verbose";
        fs::create_directories(vdir);
        vlog = std::make_unique<std::ofstream>(
            vdir / (mode_str(mode) + "_" + ts + ".jsonl"),
            std::ios::out | std::ios::trunc);
    }

    write_meta(log_file, mode, pcfg, n_ep);

    std::print(
        "poker-sim  mode={}  agents={}  M={}  alpha={}  K_max={}  "
        "H={}  F={}  episodes={}\n"
        "log: {}\n",
        mode_str(mode), pcfg.n_agents, pcfg.M, pcfg.alpha, pcfg.K_max,
        H_ROLLOUT, F_ORDERINGS, n_ep, log_path.string());

    if (do_train)
        std::print("training: start Python with: "
                   "python -m training.train_from_cpp {}\n", weights_dir);
    std::print("\n");

    float reward_sum = 0.f;
    std::vector<transition> sync_traj;

    for (int ep = 0; ep < n_ep; ++ep) {
        PokerEpisodeStats stats;

        bool is_cotamer = (mode == Mode::SEQCOMM ||
                           mode == Mode::FIXED_ORDER ||
                           mode == Mode::RANDOM_ORDER);

        if (is_cotamer) {
            auto res = run_cotamer_episode(env, *models, rng, mode, verbose);
            stats    = res.stats;

            // Verbose: log negotiation + hand events
            if (verbose && vlog) {
                int T = res.stats.hands_played;
                for (int t = 0; t < T; ++t) {
                    if (mode == Mode::SEQCOMM &&
                        t < static_cast<int>(res.intention_log.size())) {
                        write_negotiation_event(
                            *vlog, ep, t,
                            res.intention_log[t],
                            res.ordering_log[t]);
                    }
                    if (t < static_cast<int>(env.hand_history.size()))
                        write_hand_event(*vlog, ep, t, env.hand_history[t]);
                }
            }

            // Training: serialize trajectory and sync with Python
            if (do_train && !res.trajectory.empty()) {
                write_trajectory(traj_bin.string(), res.trajectory,
                                 pcfg.n_agents, env.obs_dim(), env.action_dim());
                touch(traj_ready);
                std::print("  ep {:4d}  waiting for Python…\n", ep);
                wait_for(wts_ready);
                fs::remove(traj_ready);
                fs::remove(wts_ready);
                models_ptr->update_from_blob(weights_dir);
            }

        } else {
            // Sync modes: mappo or oracle
            NeuralModels* m = (mode == Mode::ORACLE) ? nullptr : models;
            sync_traj.clear();
            stats = run_sync_episode(env, m, sync_traj, verbose);

            if (verbose && vlog) {
                for (int t = 0; t < static_cast<int>(env.hand_history.size()); ++t)
                    write_hand_event(*vlog, ep, t, env.hand_history[t]);
            }
        }

        reward_sum += stats.total_reward;
        write_episode_log(log_file, ep, stats);

        // One-line console summary
        std::string fmc = "[";
        for (int i = 0; i < static_cast<int>(stats.first_mover_counts.size()); ++i) {
            if (i) fmc += ",";
            fmc += std::to_string(stats.first_mover_counts[i]);
        }
        fmc += "]";

        std::print("ep {:4d}  R={:7.3f}  avg={:7.3f}  "
                   "hands={:3d}  win%={:5.1f}  bet={:.3f}  "
                   "B={}  T={}  ent={:.3f}  isp={:.3f}  fm={}\n",
                   ep,
                   stats.total_reward,
                   reward_sum / static_cast<float>(ep + 1),
                   stats.hands_played,
                   100.f * stats.win_rate,
                   stats.mean_bet,
                   stats.bankruptcy ? "Y" : "N",
                   stats.target_hit ? "Y" : "N",
                   stats.order_entropy,
                   stats.mean_intention_spread,
                   fmc);
    }

    if (do_train) touch(traj_done);
    std::print("\nDone. {} episodes  avg_reward={:.3f}\n",
               n_ep, reward_sum / static_cast<float>(n_ep));
    return 0;
}
