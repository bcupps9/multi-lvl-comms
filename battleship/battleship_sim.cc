// battleship_sim.cc
//
// Collaborative Battleship simulation harness.
//
// Modes (--mode):
//   seqcomm      — V(h_i) intention ordering + action cascade (learns to coordinate)
//   fixed_order  — fixed 0→1→… cascade, no negotiation (baseline: cascade only)
//   mappo        — independent, no cascade (baseline: no coordination)
//
// Logging: per-episode JSON lines → logs/battleship/<mode>_<ts>.jsonl
//
// Training loop (seqcomm / fixed_order):
//   C++ writes traj.bin + touches traj.ready
//   Python updates weights + touches weights.ready
//   C++ reloads, repeats
//
// Build:
//   cmake -B build -DUSE_TORCH=ON \
//         -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
//   cmake --build build --target battleship-sim
//
// Run:
//   ./build/battleship/battleship-sim weights/ --mode seqcomm --episodes 2000
//   python -m training.train_from_cpp weights/

#include "battleship_env.hh"
#include "agent_action.hh"
#include "libtorch_models.hh"
#include "pancy_msgs.hh"
#include "trajectory_io.hh"
#include "random_source.hh"
#include "netsim.hh"
#include "cotamer/cotamer.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <deque>
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

// ── Mode ───────────────────────────────────────────────────────────────────────

enum class Mode { SEQCOMM, FIXED_ORDER, MAPPO };

static Mode parse_mode(const std::string& s) {
    if (s == "seqcomm")     return Mode::SEQCOMM;
    if (s == "fixed_order") return Mode::FIXED_ORDER;
    if (s == "mappo")       return Mode::MAPPO;
    throw std::runtime_error("unknown --mode: " + s +
        " (seqcomm|fixed_order|mappo)");
}
static std::string mode_str(Mode m) {
    switch (m) {
        case Mode::SEQCOMM:     return "seqcomm";
        case Mode::FIXED_ORDER: return "fixed_order";
        case Mode::MAPPO:       return "mappo";
    }
    return "unknown";
}

// ── Curriculum ─────────────────────────────────────────────────────────────────

struct CurriculumStage {
    int   boss_hp;
    float boss_miss;
    int   boss_fire_period;
    int   boss_aim_lag;
    int   max_steps;
    float near_miss_reward;   // reward_near_boss set in env; fades to 0 by final stage
    const char* label;
};

// Stages progress by HP and deterministic boss tempo.  near_miss_reward starts
// high (dense aiming signal) and decreases to 0 at the final stage so the agent
// must earn only direct hits there — matching paper evaluation conditions.
// One extra hp3 stage (p12) bridges the gap between hp2/p4 and hp3/p8.
static constexpr CurriculumStage CURRICULUM[] = {
    // hp  miss  period  lag  steps  near_miss  label
    {1, 0.0f, 12, 1, 30, 0.20f, "c0:hp1/p12/t30"},
    {1, 0.0f, 10, 1, 28, 0.20f, "c1:hp1/p10/t28"},
    {1, 0.0f,  8, 1, 26, 0.15f, "c2:hp1/p8/t26"},
    {1, 0.0f,  6, 1, 24, 0.15f, "c3:hp1/p6/t24"},
    {2, 0.0f, 12, 1, 46, 0.10f, "c4:hp2/p12/t46"},
    {2, 0.0f, 10, 1, 42, 0.10f, "c5:hp2/p10/t42"},
    {2, 0.0f,  8, 1, 38, 0.05f, "c6:hp2/p8/t38"},
    {2, 0.0f,  6, 1, 34, 0.05f, "c7:hp2/p6/t34"},
    {2, 0.0f,  4, 1, 32, 0.02f, "c8:hp2/p4/t32"},
    {3, 0.0f, 12, 1, 60, 0.02f, "c9:hp3/p12/t60"},   // bridge stage
    {3, 0.0f,  8, 1, 60, 0.01f, "c10:hp3/p8/t60"},
    {3, 0.0f,  6, 1, 54, 0.01f, "c11:hp3/p6/t54"},
    {3, 0.0f,  4, 1, 50, 0.00f, "c12:hp3/p4/t50"},   // final: direct hits only
};
static constexpr int N_CURRICULUM_STAGES = static_cast<int>(
    sizeof(CURRICULUM) / sizeof(CURRICULUM[0]));

struct CurriculumTracker {
    // Advance only when the agents are winning because they can aim, not merely
    // because the current stage is forgiving.  Window resets on advance so the
    // next stage is evaluated independently.
    static constexpr int WINDOW = 1000;

    struct Sample {
        bool won;
        int boss_hits;
        int agent_shots;
    };

    int stage = 0;
    std::deque<Sample> window;
    float last_win_rate = 0.f;
    float last_hit_rate = 0.f;
    float last_zero_hit_rate = 0.f;

    const CurriculumStage& current() const { return CURRICULUM[stage]; }
    bool at_max() const { return stage >= N_CURRICULUM_STAGES - 1; }

    float target_win_rate() const {
        int hp = current().boss_hp;
        if (hp <= 1) return 0.70f;
        if (hp == 2) return 0.65f;
        return 0.60f;
    }

    float target_hit_rate() const {
        int hp = current().boss_hp;
        if (hp <= 1) return 0.020f;
        if (hp == 2) return 0.030f;
        return 0.035f;
    }

    float target_zero_hit_rate() const {
        int hp = current().boss_hp;
        if (hp <= 1) return 0.30f;
        if (hp == 2) return 0.08f;
        return 0.05f;
    }

    void update_metrics() {
        if (window.empty()) {
            last_win_rate = last_hit_rate = last_zero_hit_rate = 0.f;
            return;
        }

        int wins = 0;
        int zero_hits = 0;
        int boss_hits = 0;
        int shots = 0;
        for (const auto& s : window) {
            wins += s.won ? 1 : 0;
            zero_hits += s.boss_hits == 0 ? 1 : 0;
            boss_hits += s.boss_hits;
            shots += s.agent_shots;
        }

        last_win_rate = static_cast<float>(wins) / static_cast<float>(window.size());
        last_zero_hit_rate = static_cast<float>(zero_hits) / static_cast<float>(window.size());
        last_hit_rate = shots > 0
            ? static_cast<float>(boss_hits) / static_cast<float>(shots)
            : 0.f;
    }

    // Record episode outcome; returns true if stage advanced.
    bool record(bool won, int boss_hits, int agent_shots) {
        window.push_back({won, boss_hits, agent_shots});
        if ((int)window.size() > WINDOW) window.pop_front();
        update_metrics();

        if (at_max()) return false;
        if ((int)window.size() < WINDOW) return false;

        if (last_win_rate >= target_win_rate() &&
            last_hit_rate >= target_hit_rate() &&
            last_zero_hit_rate <= target_zero_hit_rate()) {
            ++stage;
            window.clear();
            return true;
        }
        return false;
    }
};

// ── Experiment configuration ────────────────────────────────────────────────────

struct ExpConfig {
    // Experiment 1: world-model intention ordering
    bool  use_wm_intention  = false;
    int   wm_H              = 2;     // rollout horizon
    float comms_loss_prob   = 0.f;   // P(random ordering this step)

    // Experiment 2: optional communication gate
    bool  use_comm_gate     = false;
    float comm_penalty      = 0.f;   // reward cost per comm step
};

// ── Episode stats struct ────────────────────────────────────────────────────────

struct BsEpStats {
    float total_reward       = 0.f;
    int   steps              = 0;
    int   boss_hits          = 0;
    int   agent_hits         = 0;
    bool  agents_won         = false;
    bool  boss_won           = false;
    float mean_intention_spread = 0.f;
    std::vector<int> first_mover_counts;
    int   agent_shots        = 0;
    int   fire_oob           = 0;
    int   wasted_shots       = 0;
    float mean_fire_dist     = 0.f;
    float mean_ally_dist     = 0.f;
    std::vector<int> boss_hit_counts;
    std::vector<int> fire_dist_counts;
    std::array<int, 5> move_counts{};
    std::vector<int> fire_offset_counts;
    int   curriculum_stage   = -1;  // -1 = no curriculum
    // Experiment stats
    int   comms_ok           = 0;   // negotiation steps using real ordering
    int   comms_total        = 0;   // total negotiation steps
    float comm_rate          = 0.f; // fraction of steps where >=1 agent comm'd
};

// ── Helpers ─────────────────────────────────────────────────────────────────────

static std::string jn(double v, int d = 3) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(d) << v;
    return o.str();
}
static std::string jb(bool v) { return v ? "true" : "false"; }

static void write_int_array(std::ofstream& f, const std::vector<int>& xs) {
    f << "[";
    for (int i = 0; i < (int)xs.size(); ++i) {
        if (i) f << ", ";
        f << xs[i];
    }
    f << "]";
}

template <size_t N>
static void write_int_array(std::ofstream& f, const std::array<int, N>& xs) {
    f << "[";
    for (size_t i = 0; i < N; ++i) {
        if (i) f << ", ";
        f << xs[i];
    }
    f << "]";
}

static std::string now_ts() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}
static std::string now_iso() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}
static void touch(const fs::path& p) { std::ofstream f(p); (void)f; }
static void wait_for(const fs::path& p) {
    while (!fs::exists(p)) std::this_thread::sleep_for(100ms);
}

// ── Cotamer episode tasks ───────────────────────────────────────────────────────

// SeqComm: V(h_i) ordering + launching cascade every step.
static cot::task<void> bs_seqcomm_task(Agent& agent, BattleshipEnv& env) {
    for (int t = 0; !env.is_done(); ++t) {
        co_await agent.negotiation_phase(t, 2, 2);   // F currently unused
        co_await agent.launching_phase(t);
    }
}

// Fixed-order: set rank from predetermined ordering, encode h, launch.
static cot::task<void> bs_ordered_task(Agent& agent, BattleshipEnv& env,
                                        const std::vector<int>& ordering) {
    const int N = (int)ordering.size();
    for (int t = 0; !env.is_done(); ++t) {
        agent.N_upper.clear();
        agent.N_lower.clear();
        for (int r = 0; r < N; ++r) {
            if (ordering[r] != agent.id) continue;
            for (int r2 = 0; r2 < r; ++r2) agent.N_upper.push_back(ordering[r2]);
            for (int r2 = r+1; r2 < N; ++r2) agent.N_lower.push_back(ordering[r2]);
            break;
        }
        agent.h = agent.models.encode(agent.obs);
        co_await agent.launching_phase(t);
    }
}

// ── Cotamer episode runner ──────────────────────────────────────────────────────

struct CotamerResult {
    BsEpStats             stats;
    std::vector<transition> trajectory;
    std::vector<std::vector<float>> intention_log;  // [t][agent_i]
    std::vector<std::vector<int>>   ordering_log;   // [t][rank] = agent_id
    // Experiment 2: per-agent per-step gate decisions for REINFORCE.
    std::vector<std::vector<CommGateDecision>> comm_logs;
};

static CotamerResult run_cotamer_episode(BattleshipEnv& env,
                                          NeuralModels& models,
                                          random_source& rng,
                                          Mode mode,
                                          const ExpConfig& exp = {}) {
    CotamerResult out;
    auto& traj = out.trajectory;

    const int N = env.config().n_agents;

    // Fresh agents and channels for this episode.
    std::vector<std::unique_ptr<Agent>> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i)
        agents.push_back(std::make_unique<Agent>(
            i, env.obs_dim(), env.action_dim(),
            models, env, traj, rng));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (j != i) agents[i]->clique.push_back(j);

    // Per-episode comms counters; one pair per agent (they'll all track the same
    // negotiation steps but we average later).
    std::vector<int> comms_ok_counts(N, 0), comms_total_counts(N, 0);
    // Comm gate sidecar: per-agent per-step decisions for REINFORCE.
    std::vector<std::vector<CommGateDecision>> comm_logs(N);

    // Pre-compute per-step comms failure so all agents agree on the same outcome.
    int max_t = env.config().max_steps;
    std::vector<bool> comms_failed(max_t, false);
    if (exp.comms_loss_prob > 0.f) {
        for (int t = 0; t < max_t; ++t)
            comms_failed[t] = (rng.uniform(0.f, 1.f) < exp.comms_loss_prob);
    }

    for (int i = 0; i < N; ++i) {
        auto& a = *agents[i];
        a.use_wm_intention       = exp.use_wm_intention;
        a.wm_H                   = exp.wm_H;
        a.comms_loss_prob        = exp.comms_loss_prob;
        a.comms_failed_per_step  = &comms_failed;
        a.comms_ok_count         = &comms_ok_counts[i];
        a.comms_total_count      = &comms_total_counts[i];
        a.use_comm_gate          = exp.use_comm_gate;
        a.comm_penalty           = exp.comm_penalty;
        if (exp.use_comm_gate) a.comm_log = &comm_logs[i];
    }

    std::list<netsim::channel<pancy::agent_msg>> channels;
    for (int i = 0; i < N; ++i)
        for (int j : agents[i]->clique) {
            channels.emplace_back(agents[j]->from_agents,
                                  "bs" + std::to_string(i) + "->" + std::to_string(j));
            agents[i]->to_neighbors.push_back(&channels.back());
        }

    // Intention buffers for seqcomm ordering log.
    std::vector<std::vector<float>> intentions(N);
    for (int i = 0; i < N; ++i)
        agents[i]->own_intentions = &intentions[i];

    auto init_obs = env.reset();
    for (int i = 0; i < N; ++i) agents[i]->obs = init_obs[i];

    if (mode == Mode::SEQCOMM) {
        for (auto& a : agents)
            bs_seqcomm_task(*a, env).detach();
    } else {
        std::vector<int> ordering(N);
        std::iota(ordering.begin(), ordering.end(), 0);  // fixed: 0, 1, …
        for (auto& a : agents)
            bs_ordered_task(*a, env, ordering).detach();
    }

    cot::loop();

    // ── Collect stats ──────────────────────────────────────────────────────────

    auto env_s = env.episode_stats();
    auto& s = out.stats;
    s.steps      = env_s.steps;
    s.boss_hits  = env_s.boss_hits;
    s.agent_hits = env_s.agent_hits;
    s.agents_won = env_s.agents_won;
    s.boss_won   = env_s.boss_won;
    s.agent_shots = env_s.agent_shots;
    s.fire_oob = env_s.fire_oob;
    s.wasted_shots = env_s.wasted_shots;
    s.mean_fire_dist = env_s.mean_fire_dist;
    s.mean_ally_dist = env_s.mean_ally_dist;
    s.boss_hit_counts = env_s.boss_hit_counts;
    s.fire_dist_counts = env_s.fire_dist_counts;
    s.move_counts = env_s.move_counts;
    s.fire_offset_counts = env_s.fire_offset_counts;
    s.first_mover_counts.assign(N, 0);

    for (auto& tr : traj) {
        if (tr.agent_id == 0) s.total_reward += tr.reward;
        int t    = tr.timestep;
        int rank = (int)tr.upper_actions.size();
        if (t < (int)out.ordering_log.size())
            out.ordering_log[t][rank] = tr.agent_id;
    }

    int T = env_s.steps;
    out.ordering_log.assign(T, std::vector<int>(N, -1));
    for (auto& tr : traj) {
        int t    = tr.timestep;
        int rank = (int)tr.upper_actions.size();
        if (t >= 0 && t < T && rank < N)
            out.ordering_log[t][rank] = tr.agent_id;
    }
    for (int t = 0; t < T; ++t) {
        if (!out.ordering_log[t].empty() && out.ordering_log[t][0] >= 0)
            s.first_mover_counts[out.ordering_log[t][0]]++;
    }

    // Intention spread (seqcomm only).
    out.intention_log.assign(T, std::vector<float>(N, 0.f));
    for (int i = 0; i < N; ++i) {
        auto& iv = intentions[i];
        for (int t = 0; t < std::min(T, (int)iv.size()); ++t)
            out.intention_log[t][i] = iv[t];
    }
    if (mode == Mode::SEQCOMM && N > 1 && T > 0) {
        double sum = 0.0;
        for (int t = 0; t < T; ++t) {
            double mean = 0.0;
            for (float v : out.intention_log[t]) mean += v;
            mean /= N;
            double sq = 0.0;
            for (float v : out.intention_log[t]) sq += (v-mean)*(v-mean);
            sum += std::sqrt(sq / (N-1));
        }
        s.mean_intention_spread = (float)(sum / T);
    }

    // Comms stats: average across agents (should be identical in seqcomm).
    {
        int ok_sum = 0, tot_sum = 0;
        for (int i = 0; i < N; ++i) {
            ok_sum  += comms_ok_counts[i];
            tot_sum += comms_total_counts[i];
        }
        s.comms_ok    = ok_sum / std::max(1, N);
        s.comms_total = tot_sum / std::max(1, N);
        s.comm_rate   = s.comms_total > 0
            ? static_cast<float>(s.comms_ok) / static_cast<float>(s.comms_total)
            : 1.f;
    }

    // Comm gate sidecar: stash it on out so the main loop can write it to disk.
    out.comm_logs = std::move(comm_logs);

    return out;
}

// ── Sync episode runner (mappo) ─────────────────────────────────────────────────

static BsEpStats run_sync_episode(BattleshipEnv& env, NeuralModels& models,
                                   std::vector<transition>& traj) {
    const int N = env.config().n_agents;
    auto obs_all = env.reset();
    traj.clear();

    for (int t = 0; !env.is_done(); ++t) {
        std::vector<std::vector<float>> actions(N);
        std::vector<float>              log_probs(N, 0.f), lp_old(N, 0.f), vals(N, 0.f);

        for (int i = 0; i < N; ++i) {
            auto h       = models.encode(obs_all[i]);
            auto ctx     = models.attention_a(h, {});
            auto [a, lp] = models.policy_sample(ctx);
            float lpold  = models.policy_log_prob_old(ctx, a);
            float v      = models.critic(ctx);
            actions[i]   = std::move(a);
            log_probs[i] = lp;
            lp_old[i]    = lpold;
            vals[i]      = v;
        }

        auto result = env.sync_step(actions);

        for (int i = 0; i < N; ++i)
            traj.push_back({
                .agent_id    = i,
                .timestep    = t,
                .obs         = obs_all[i],
                .action      = actions[i],
                .upper_actions = {},
                .next_obs    = result.next_obs[i],
                .reward      = result.reward,
                .value       = vals[i],
                .log_prob    = log_probs[i],
                .log_prob_old = lp_old[i],
            });

        obs_all = std::move(result.next_obs);
    }

    auto env_s = env.episode_stats();
    BsEpStats s;
    s.steps      = env_s.steps;
    s.boss_hits  = env_s.boss_hits;
    s.agent_hits = env_s.agent_hits;
    s.agents_won = env_s.agents_won;
    s.boss_won   = env_s.boss_won;
    s.agent_shots = env_s.agent_shots;
    s.fire_oob = env_s.fire_oob;
    s.wasted_shots = env_s.wasted_shots;
    s.mean_fire_dist = env_s.mean_fire_dist;
    s.mean_ally_dist = env_s.mean_ally_dist;
    s.boss_hit_counts = env_s.boss_hit_counts;
    s.fire_dist_counts = env_s.fire_dist_counts;
    s.move_counts = env_s.move_counts;
    s.fire_offset_counts = env_s.fire_offset_counts;
    s.first_mover_counts.assign(N, 0);
    for (auto& tr : traj)
        if (tr.agent_id == 0) s.total_reward += tr.reward;
    return s;
}

// ── Logging ─────────────────────────────────────────────────────────────────────

static void write_meta(std::ofstream& f, Mode mode,
                       const BattleshipConfig& cfg, int n_ep,
                       const ExpConfig& exp = {}) {
    f << "{\"_meta\": {"
      << "\"env\": \"battleship\", "
      << "\"mode\": \"" << mode_str(mode) << "\", "
      << "\"use_wm_intention\": " << (exp.use_wm_intention ? "true" : "false") << ", "
      << "\"wm_H\": " << exp.wm_H << ", "
      << "\"comms_loss_prob\": " << jn(exp.comms_loss_prob) << ", "
      << "\"use_comm_gate\": " << (exp.use_comm_gate ? "true" : "false") << ", "
      << "\"comm_penalty\": " << jn(exp.comm_penalty) << ", "
      << "\"M\": " << cfg.M << ", "
      << "\"n_agents\": " << cfg.n_agents << ", "
      << "\"n_boss\": " << cfg.n_boss << ", "
      << "\"sight_range\": " << cfg.sight_range << ", "
      << "\"fire_range\": " << cfg.fire_range << ", "
      << "\"boss_fire_range\": " << cfg.boss_fire_range << ", "
      << "\"boss_fire_period\": " << cfg.boss_fire_period << ", "
      << "\"boss_aim_lag\": " << cfg.boss_aim_lag << ", "
      << "\"boss_start_hp\": " << cfg.boss_start_hp << ", "
      << "\"boss_miss_prob\": " << jn(cfg.boss_miss_prob) << ", "
      << "\"max_steps\": " << cfg.max_steps << ", "
      << "\"reward_hit_boss\": " << jn(cfg.reward_hit_boss) << ", "
      << "\"reward_hit_self\": " << jn(cfg.reward_hit_self) << ", "
      << "\"reward_survive\": " << jn(cfg.reward_survive) << ", "
      << "\"reward_near_boss\": " << jn(cfg.reward_near_boss) << ", "
      << "\"reward_proximity\": " << jn(cfg.reward_proximity) << ", "
      << "\"reward_agents_win\": " << jn(cfg.reward_agents_win) << ", "
      << "\"episodes\": " << n_ep << ", "
      << "\"timestamp\": \"" << now_iso() << "\""
      << "}}\n";
    f.flush();
}

static void write_ep_log(std::ofstream& f, int ep, const BsEpStats& s) {
    f << "{"
      << "\"ep\": " << ep << ", "
      << "\"reward\": " << jn(s.total_reward) << ", "
      << "\"steps\": " << s.steps << ", "
      << "\"boss_hits\": " << s.boss_hits << ", "
      << "\"agent_hits\": " << s.agent_hits << ", "
      << "\"agents_won\": " << jb(s.agents_won) << ", "
      << "\"boss_won\": " << jb(s.boss_won) << ", "
      << "\"agent_shots\": " << s.agent_shots << ", "
      << "\"fire_oob\": " << s.fire_oob << ", "
      << "\"wasted_shots\": " << s.wasted_shots << ", "
      << "\"mean_fire_dist\": " << jn(s.mean_fire_dist) << ", "
      << "\"mean_ally_dist\": " << jn(s.mean_ally_dist) << ", "
      << "\"boss_hit_counts\": " << [&]{ std::string o="["; for(int i=0;i<(int)s.boss_hit_counts.size();++i){if(i)o+=","; o+=std::to_string(s.boss_hit_counts[i]);} return o+"]"; }() << ", "
      << "\"fire_dist_counts\": ";
    write_int_array(f, s.fire_dist_counts);
    f << ", \"move_counts\": ";
    write_int_array(f, s.move_counts);
    f << ", \"fire_offset_counts\": ";
    write_int_array(f, s.fire_offset_counts);
    f << ", "
      << "\"intention_spread\": " << jn(s.mean_intention_spread) << ", "
      << "\"first_mover\": [";
    for (int i = 0; i < (int)s.first_mover_counts.size(); ++i) {
        if (i) f << ", ";
        f << s.first_mover_counts[i];
    }
    f << "]";
    if (s.curriculum_stage >= 0)
        f << ", \"curriculum_stage\": " << s.curriculum_stage;
    f << ", \"comms_ok\": " << s.comms_ok
      << ", \"comms_total\": " << s.comms_total
      << ", \"comm_rate\": " << jn(s.comm_rate)
      << "}\n";
    f.flush();
}

// ── main ────────────────────────────────────────────────────────────────────────

// ── Comm gate sidecar writer ────────────────────────────────────────────────────
// Binary format:
//   MAGIC    : uint32  (0x43474154 = "CGAT")
//   n_entries: int32   — sum of per-agent comm decisions across all agents × steps
//   embed_dim: int32
//   ep_reward: float32
//   [n_entries × (did_comm:int32, gate_logit:float32, h[embed_dim]:float32)]
//
// Only agent 0's decisions are written (symmetry assumption for paper experiments).

static void write_comm_sidecar(const fs::path& path,
                                const std::vector<CommGateDecision>& log,
                                float ep_reward)
{
    constexpr uint32_t MAGIC = 0x43474154u;
    int n   = static_cast<int>(log.size());
    int edim = log.empty() ? 0 : static_cast<int>(log.front().h.size());

    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&MAGIC),     4);
    f.write(reinterpret_cast<const char*>(&n),         4);
    f.write(reinterpret_cast<const char*>(&edim),      4);
    f.write(reinterpret_cast<const char*>(&ep_reward), 4);

    std::vector<float> zero_h(edim, 0.f);
    for (const auto& rec : log) {
        int dc = static_cast<int>(rec.did_comm);
        f.write(reinterpret_cast<const char*>(&dc),    4);
        f.write(reinterpret_cast<const char*>(&rec.gate_logit), 4);
        const auto& h = static_cast<int>(rec.h.size()) == edim ? rec.h : zero_h;
        f.write(reinterpret_cast<const char*>(h.data()), edim * 4);
    }
}

int main(int argc, char* argv[]) {
    std::string      weights_dir;
    Mode             mode      = Mode::SEQCOMM;
    int              n_ep      = 1000;
    bool             do_train  = true;
    int              seed      = -1;
    std::string      log_dir_str;
    bool             use_curriculum = false;
    int              mappo_update_every = 8;
    BattleshipConfig bcfg;
    ExpConfig        exp_cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--no-train")             do_train = false;
        else if (arg == "--curriculum")           use_curriculum = true;
        else if (arg == "--mode"     && i+1<argc) mode   = parse_mode(argv[++i]);
        else if (arg == "--episodes" && i+1<argc) n_ep   = std::stoi(argv[++i]);
        else if (arg == "--seed"     && i+1<argc) seed   = std::stoi(argv[++i]);
        else if (arg == "--log-dir"  && i+1<argc) log_dir_str = argv[++i];
        else if (arg == "--M"        && i+1<argc) bcfg.M           = std::stoi(argv[++i]);
        else if (arg == "--agents"   && i+1<argc) bcfg.n_agents    = std::stoi(argv[++i]);
        else if (arg == "--boss"     && i+1<argc) bcfg.n_boss      = std::stoi(argv[++i]);
        else if (arg == "--sight"    && i+1<argc) bcfg.sight_range = std::stoi(argv[++i]);
        else if (arg == "--fire"     && i+1<argc) { bcfg.fire_range = std::stoi(argv[++i]); bcfg.boss_fire_range = bcfg.fire_range; }
        else if (arg == "--boss-fire" && i+1<argc) bcfg.boss_fire_range = std::stoi(argv[++i]);
        else if (arg == "--boss-fire-period" && i+1<argc) bcfg.boss_fire_period = std::stoi(argv[++i]);
        else if (arg == "--boss-aim-lag" && i+1<argc) bcfg.boss_aim_lag = std::stoi(argv[++i]);
        else if (arg == "--steps"    && i+1<argc) bcfg.max_steps      = std::stoi(argv[++i]);
        else if (arg == "--survive"    && i+1<argc) bcfg.reward_survive   = std::stof(argv[++i]);
        else if (arg == "--near-boss"  && i+1<argc) bcfg.reward_near_boss  = std::stof(argv[++i]);
        else if (arg == "--proximity"  && i+1<argc) bcfg.reward_proximity  = std::stof(argv[++i]);
        else if (arg == "--hit-boss"   && i+1<argc) bcfg.reward_hit_boss   = std::stof(argv[++i]);
        else if (arg == "--hit-self"   && i+1<argc) bcfg.reward_hit_self   = std::stof(argv[++i]);
        else if (arg == "--win-reward" && i+1<argc) bcfg.reward_agents_win = std::stof(argv[++i]);
        else if (arg == "--update-every"   && i+1<argc) mappo_update_every = std::stoi(argv[++i]);
        else if (arg == "--no-survive")            bcfg.reward_survive   = 0.f;
        else if (arg == "--no-near-boss")          bcfg.reward_near_boss  = 0.f;
        else if (arg == "--no-proximity")          bcfg.reward_proximity  = 0.f;
        // Experiment 1: world-model intention
        else if (arg == "--wm-intention")          exp_cfg.use_wm_intention = true;
        else if (arg == "--wm-H" && i+1<argc)      exp_cfg.wm_H = std::stoi(argv[++i]);
        else if (arg == "--comms-loss" && i+1<argc) exp_cfg.comms_loss_prob = std::stof(argv[++i]);
        // Experiment 2: comm gate
        else if (arg == "--comm-gate")             exp_cfg.use_comm_gate = true;
        else if (arg == "--comm-penalty" && i+1<argc) exp_cfg.comm_penalty = std::stof(argv[++i]);
        else if (arg[0] != '-')                   weights_dir = arg;
        else {
            std::print(stderr,
                "usage: battleship-sim [<weights_dir>] [--mode seqcomm|fixed_order|mappo]\n"
                "  [--episodes N] [--no-train] [--seed N] [--log-dir PATH]\n"
                "  [--curriculum] [--update-every N]\n"
                "  [--M N] [--agents N] [--boss N] [--sight N] [--fire N] [--steps N]\n"
                "  [--boss-fire N] [--boss-fire-period N] [--boss-aim-lag N]\n"
                "  [--survive R] [--near-boss R] [--proximity R] [--win-reward R]\n"
                "  [--hit-boss R] [--hit-self R]\n"
                "  [--no-survive] [--no-near-boss] [--no-proximity]\n"
                "  Exp1: [--wm-intention] [--wm-H N] [--comms-loss P]\n"
                "  Exp2: [--comm-gate] [--comm-penalty R]\n");
            return 1;
        }
    }

    // MAPPO trains via the same traj.bin IPC as seqcomm — just no cascade in actions.
    if (do_train && weights_dir.empty()) {
        std::print(stderr, "error: weights_dir required for training\n");
        return 1;
    }
    if (!weights_dir.empty() && !fs::exists(weights_dir)) {
        std::print(stderr, "error: weights_dir not found: {}\n", weights_dir);
        return 1;
    }

    random_source rng;
    if (seed >= 0) rng.seed((uint64_t)seed);

    CurriculumTracker curriculum;
    if (use_curriculum) {
        const auto& s = curriculum.current();
        bcfg.boss_start_hp    = s.boss_hp;
        bcfg.boss_miss_prob   = s.boss_miss;
        bcfg.boss_fire_period = s.boss_fire_period;
        bcfg.boss_aim_lag     = s.boss_aim_lag;
        bcfg.max_steps        = s.max_steps;
        bcfg.reward_near_boss = s.near_miss_reward;
    }

    BattleshipEnv env(bcfg, rng);

    // Models (required for seqcomm and fixed_order).
    std::unique_ptr<LibtorchNeuralModels> models_ptr;
    if (!weights_dir.empty())
        models_ptr = std::make_unique<LibtorchNeuralModels>(weights_dir, bcfg.n_agents);

    if (!models_ptr && mode != Mode::MAPPO) {
        std::print(stderr, "error: weights_dir required for mode '{}'\n", mode_str(mode));
        return 1;
    }
    NeuralModels* models = models_ptr.get();

    // Sentinel paths.
    fs::path traj_bin   = weights_dir.empty() ? fs::path{} : fs::path(weights_dir)/"traj.bin";
    fs::path traj_ready = weights_dir.empty() ? fs::path{} : fs::path(weights_dir)/"traj.ready";
    fs::path wts_ready  = weights_dir.empty() ? fs::path{} : fs::path(weights_dir)/"weights.ready";
    fs::path traj_done  = weights_dir.empty() ? fs::path{} : fs::path(weights_dir)/"traj.done";

    if (do_train) {
        for (auto* p : {&traj_ready, &wts_ready, &traj_done}) {
            std::error_code ec; fs::remove(*p, ec);
        }
    }

    // Logging.
    fs::path log_dir = log_dir_str.empty()
        ? fs::current_path() / "logs" / "battleship"
        : fs::path(log_dir_str);
    fs::create_directories(log_dir);

    std::string ts = now_ts();
    fs::path log_path = log_dir / (mode_str(mode) + "_" + ts + ".jsonl");
    std::ofstream log_file(log_path);
    write_meta(log_file, mode, bcfg, n_ep, exp_cfg);

    std::print("battleship-sim  mode={}  M={}  agents={}  boss={}  sight={}  fire={}  "
               "boss_period={}  boss_lag={}  steps={}  survive={}  near_boss={}  "
               "win_reward={}  episodes={}\nlog: {}\n\n",
               mode_str(mode), bcfg.M, bcfg.n_agents, bcfg.n_boss,
               bcfg.sight_range, bcfg.fire_range,
               bcfg.boss_fire_period, bcfg.boss_aim_lag, bcfg.max_steps,
               bcfg.reward_survive, bcfg.reward_near_boss,
               bcfg.reward_agents_win, n_ep,
               log_path.string());

    // Curriculum setup: start at stage 0 if enabled.
    if (use_curriculum) {
        env.set_curriculum(CURRICULUM[0].boss_hp, CURRICULUM[0].boss_miss,
                           CURRICULUM[0].boss_fire_period,
                           CURRICULUM[0].boss_aim_lag,
                           CURRICULUM[0].max_steps,
                           CURRICULUM[0].near_miss_reward);
        std::print("curriculum ON  stage={}  {}  near_miss={:.2f}\n\n",
                   curriculum.stage, CURRICULUM[0].label,
                   CURRICULUM[0].near_miss_reward);
    }

    float reward_sum = 0.f;
    std::vector<transition> mappo_batch;   // accumulates across episodes for MAPPO

    for (int ep = 0; ep < n_ep; ++ep) {
        BsEpStats stats;

        if (mode == Mode::MAPPO) {
            std::vector<transition> ep_traj;
            stats = run_sync_episode(env, *models, ep_traj);
            mappo_batch.insert(mappo_batch.end(), ep_traj.begin(), ep_traj.end());

            if (do_train && (ep + 1) % mappo_update_every == 0 && !mappo_batch.empty()) {
                write_trajectory(traj_bin.string(), mappo_batch,
                                 bcfg.n_agents, env.obs_dim(), env.action_dim());
                touch(traj_ready);
                std::print("  ep {:4d}  waiting for Python…\n", ep);
                wait_for(wts_ready);
                fs::remove(traj_ready);
                fs::remove(wts_ready);
                models_ptr->update_from_blob(weights_dir);
                mappo_batch.clear();
            }
        } else {
            auto res = run_cotamer_episode(env, *models, rng, mode, exp_cfg);
            stats    = res.stats;
            if (do_train && !res.trajectory.empty()) {
                write_trajectory(traj_bin.string(), res.trajectory,
                                 bcfg.n_agents, env.obs_dim(), env.action_dim());

                // Write comm gate sidecar for Experiment 2 REINFORCE update.
                if (exp_cfg.use_comm_gate && !res.comm_logs.empty()
                        && !res.comm_logs[0].empty()) {
                    fs::path sidecar = fs::path(weights_dir) / "comm_gate_sidecar.bin";
                    write_comm_sidecar(sidecar, res.comm_logs[0],
                                       stats.total_reward);
                }

                touch(traj_ready);
                std::print("  ep {:4d}  waiting for Python…\n", ep);
                wait_for(wts_ready);
                fs::remove(traj_ready);
                fs::remove(wts_ready);
                models_ptr->update_from_blob(weights_dir);
            }
        }

        // Curriculum auto-advance.
        if (use_curriculum) {
            stats.curriculum_stage = curriculum.stage;
            bool advanced = curriculum.record(stats.agents_won,
                                              stats.boss_hits,
                                              stats.agent_shots);
            if (advanced) {
                const auto& s = curriculum.current();
                env.set_curriculum(s.boss_hp, s.boss_miss,
                                   s.boss_fire_period, s.boss_aim_lag,
                                   s.max_steps, s.near_miss_reward);
                std::print("\n*** curriculum advance → stage {} ({})  "
                           "wr={:.0f}% hit/shot={:.1f}% zero={:.0f}%  near_miss={:.2f} ***\n\n",
                           curriculum.stage, s.label,
                           curriculum.last_win_rate * 100.f,
                           curriculum.last_hit_rate * 100.f,
                           curriculum.last_zero_hit_rate * 100.f,
                           s.near_miss_reward);
            }
        }

        reward_sum += stats.total_reward;
        write_ep_log(log_file, ep, stats);

        // One-line console summary.
        std::string fm = "[";
        for (int i = 0; i < (int)stats.first_mover_counts.size(); ++i) {
            if (i) fm += ",";
            fm += std::to_string(stats.first_mover_counts[i]);
        }
        fm += "]";
        std::string cur_tag = use_curriculum
            ? std::format("  c{}", curriculum.stage) : "";
        std::string comm_tag = "";
        if (exp_cfg.use_wm_intention || exp_cfg.use_comm_gate || exp_cfg.comms_loss_prob > 0)
            comm_tag = std::format("  cr={:.0f}%", stats.comm_rate * 100.f);
        std::print("ep {:4d}  r={:+.2f}  steps={:3d}  boss_hits={:2d}  "
                   "agent_hits={:2d}  {}  fm={}{}{}\n",
                   ep, stats.total_reward, stats.steps,
                   stats.boss_hits, stats.agent_hits,
                   stats.agents_won ? "WIN" : stats.boss_won ? "LOSS" : "time",
                   fm, cur_tag, comm_tag);
    }

    std::print("\navg reward: {:.3f}\n", reward_sum / n_ep);
    if (do_train)
        touch(traj_done);
    return 0;
}
