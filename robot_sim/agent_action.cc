#include "agent_action.hh"
#include <print>

namespace seqcomm {

// ── Construction ──────────────────────────────────────────────────────────────

Agent::Agent(int id, int obs_dim, int action_dim,
             NeuralModels& models, Environment& env,
             std::vector<transition>& trajectory,
             random_source& rng,
             bool verbose)
    : id(id), obs_dim(obs_dim), action_dim(action_dim),
      from_agents(rng, "agent-" + std::to_string(id)),
      models(models), env(env), trajectory(trajectory),
      verbose(verbose), rng_(rng) {}


// ── Helpers ───────────────────────────────────────────────────────────────────

cot::task<void> Agent::send_to(int neighbor_id, pancy::agent_msg msg) {
    for (size_t k = 0; k < clique.size(); ++k) {
        if (clique[k] == neighbor_id) {
            co_await to_neighbors[k]->send(std::move(msg));
            co_return;
        }
    }
}

cot::task<void> Agent::broadcast(pancy::agent_msg msg) {
    for (auto* ch : to_neighbors)
        co_await ch->send(msg);
}



// ── negotiation_phase ─────────────────────────────────────────────────────────
//
// Encode obs → h_i, compute ordering intention, broadcast, receive, set
// N_upper/N_lower.  Three modes:
//
//   use_wm_intention=false  (default)
//       Critic proxy: V(AM_a(h_i, {})).  Single-round broadcast of a scalar.
//
//   use_wm_intention=true   (Experiment 1)
//       H-step world-model rollout.  First broadcasts h_i (hidden_state_msg)
//       so every agent has all h_j for a LOCAL rollout, then broadcasts the
//       resulting cumulative predicted reward as the intention scalar.
//       Two cotamer rounds per negotiation step.
//
//   comms_loss_prob > 0     (Experiment 1 ablation)
//       With probability comms_loss_prob, skip negotiation entirely and assign
//       random N_upper / N_lower.  Simulates channel failure.
//
//   use_comm_gate=true      (Experiment 2)
//       Each agent independently samples do_comm ~ Bernoulli(σ(gate(h))).
//       If any agent opts out it broadcasts a sentinel (-1e30) rather than its
//       real intention; receivers detect this and fall back to random ordering.
//       Agents that chose to comm will have their reward reduced by comm_penalty
//       in launching_phase.
//
// Tie-breaking by agent ID guarantees a strict total order even with identical
// intentions, preventing execute_signal deadlock.

cot::task<void> Agent::negotiation_phase(int t, int H, int /*F*/) {
    h = models.encode(obs);

    // ── Comms-loss: random ordering, skip all negotiation ─────────────────────
    // Decision is pre-computed by the harness and shared across ALL agents for
    // this timestep — avoids deadlock from independent per-agent dice rolls.
    bool channel_failed = comms_failed_per_step &&
                          t < (int)comms_failed_per_step->size() &&
                          (*comms_failed_per_step)[t];
    if (channel_failed) {
        // Fallback ordering: lower agent ID acts first (consistent across all
        // agents with no message passing — avoids launching_phase deadlock).
        N_upper.clear();
        N_lower.clear();
        for (int nid : clique) {
            if (nid < id) N_upper.push_back(nid);
            else           N_lower.push_back(nid);
        }
        if (comms_total_count) ++(*comms_total_count);
        co_return;
    }

    // ── Comm gate: each agent independently decides whether to communicate ────
    bool do_comm    = true;
    float comm_logit = 1e9f;
    if (use_comm_gate) {
        comm_logit = models.comm_gate(h);
        float p = 1.f / (1.f + std::exp(-comm_logit));
        do_comm = (rng_.uniform(0.f, 1.f) < p);
        did_comm_this_step = do_comm;
        if (comm_log) comm_log->push_back({do_comm, comm_logit, h});
    }

    // ── Compute ordering intention ─────────────────────────────────────────────
    float intention = 0.f;

    if (use_wm_intention && !clique.empty()) {
        // Round 1: ALL agents broadcast h unconditionally — including agents that
        // later opt out via the comm gate.  This prevents deadlock when do_comm
        // differs across agents: everyone participates in round 1, the comm
        // decision only affects what is sent in round 2 (intention vs sentinel).
        co_await broadcast(pancy::hidden_state_msg{id, h});

        std::vector<std::vector<float>> all_h;
        all_h.reserve(1 + clique.size());
        all_h.push_back(h);
        for (size_t k = 0; k < clique.size(); ++k) {
            auto msg = co_await receive_typed<pancy::hidden_state_msg>();
            all_h.push_back(msg.h);
        }

        // Only run the expensive rollout if this agent is going to comm.
        // If opting out we still need to reach the intention_msg broadcast below
        // (as a sentinel), so fall back to a cheap critic call.
        if (do_comm) {
            auto cur_h = all_h;
            float total_pred = 0.f;
            const int N = static_cast<int>(cur_h.size());
            const int rollout = (H > 0) ? H : wm_H;
            for (int s = 0; s < rollout; ++s) {
                std::vector<std::vector<float>> tent_a;
                tent_a.reserve(N);
                for (auto& hj : cur_h) {
                    auto ctx_j      = models.attention_a(hj, {});
                    auto [a_j, _lp] = models.policy_sample(ctx_j);
                    tent_a.push_back(std::move(a_j));
                }
                auto ctx_w          = models.attention_w(cur_h, tent_a);
                auto [nobs_flat, r] = models.world_model(ctx_w);
                total_pred += r;
                if (s < rollout - 1) {
                    int od = static_cast<int>(nobs_flat.size()) / N;
                    for (int j = 0; j < N; ++j)
                        cur_h[j] = models.encode(
                            std::span<const float>{nobs_flat.data() + j * od,
                                                   static_cast<size_t>(od)});
                }
            }
            intention = total_pred;
        } else {
            auto ctx  = models.attention_a(h, {});
            intention = models.critic(ctx);
        }
    } else {
        // Critic proxy: V(AM_a(h, {})) — fast single forward pass.
        auto ctx  = models.attention_a(h, {});
        intention = models.critic(ctx);
    }

    if (own_intentions) own_intentions->push_back(intention);

    // ── Broadcast intention (sentinel when agent opts out of comm gate) ────────
    // Sentinel: a value so large-negative that no real critic/WM score reaches it.
    constexpr float NO_COMM_SENTINEL = -1e30f;
    co_await broadcast(pancy::intention_msg{
        id, do_comm ? intention : NO_COMM_SENTINEL});

    // ── Collect neighbours' intentions and set N_upper / N_lower ─────────────
    N_upper.clear();
    N_lower.clear();
    bool any_no_comm = !do_comm;
    std::vector<std::pair<int, float>> received;
    received.reserve(clique.size());

    for (size_t k = 0; k < clique.size(); ++k) {
        auto msg = co_await receive_typed<pancy::intention_msg>();
        if (msg.intention <= NO_COMM_SENTINEL + 1e20f) any_no_comm = true;
        received.emplace_back(msg.sender_id, msg.intention);
    }

    if (comms_total_count) ++(*comms_total_count);

    if (any_no_comm) {
        // At least one agent opted out → fixed ordering by agent ID.
        // Must be deterministic and consistent across all agents so launching_phase
        // doesn't deadlock (each agent uses the same rule independently).
        for (auto [rid, _] : received) {
            if (rid < id) N_upper.push_back(rid);
            else           N_lower.push_back(rid);
        }
    } else {
        // All agents communicated — use real ordering.
        if (comms_ok_count) ++(*comms_ok_count);
        for (auto [rid, rint] : received) {
            bool is_upper = rint > intention ||
                (rint == intention && rid > id);
            if (is_upper) N_upper.push_back(rid);
            else           N_lower.push_back(rid);
        }
    }

    if (verbose) {
        float t_sec = std::chrono::duration<float>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::print("{:.3f}: agent {} t={} N_upper={} N_lower={}\n",
                   t_sec, id, t, N_upper.size(), N_lower.size());
    }
}


// ── launching_phase ───────────────────────────────────────────────────────────
//
//   Algorithm 6:
//     1. Wait for upper_action_msg from every agent in N_upper.
//     2. Sample own action via π(AM_a(h, a_upper)).
//     3. Send upper_action_msg to every agent in N_lower.
//     4. If lowest in clique (N_lower empty): broadcast execute_signal.
//     5. Otherwise: wait for execute_signal, then step environment.

cot::task<void> Agent::launching_phase(int t) {
    // Collect upper-level actions
    std::vector<std::vector<float>> a_upper;
    a_upper.reserve(N_upper.size());
    for (size_t k = 0; k < N_upper.size(); ++k) {
        auto msg = co_await receive_typed<pancy::upper_action_msg>();
        a_upper.push_back(std::move(msg.action));
    }

    // Sample own action
    auto context       = models.attention_a(h, a_upper);
    auto [a, log_p]    = models.policy_sample(context);
    float log_p_old    = models.policy_log_prob_old(context, a);
    float v            = models.critic(context);
    action             = a;

    // Broadcast action to lower neighbors
    for (int lower_id : N_lower)
        co_await send_to(lower_id, pancy::upper_action_msg{id, action});

    // Trigger or wait for execute
    if (N_lower.empty()) {
        co_await broadcast(pancy::execute_signal{id, t});
    } else {
        co_await receive_typed<pancy::execute_signal>();
    }

    // Environment step
    env.submit_action(id, action);
    auto [next_obs, reward] = co_await env.get_result(id);

    // Comm gate: communicating has a cost.  Subtract penalty here so the PPO
    // gradient sees the true net reward and learns to trade off ordering
    // quality against communication overhead.
    float net_reward = reward;
    if (comm_penalty > 0.f && did_comm_this_step) {
        net_reward -= comm_penalty;
    }

    // Record transition
    trajectory.push_back({
        .agent_id    = id,
        .timestep    = t,
        .obs         = obs,
        .action      = action,
        .upper_actions = a_upper,
        .next_obs    = next_obs,
        .reward      = net_reward,
        .value       = v,
        .log_prob    = log_p,
        .log_prob_old = log_p_old,
    });

    obs = std::move(next_obs);
}


// ── run_episode ───────────────────────────────────────────────────────────────
//
//   Run T timesteps: negotiation then launching.
//   Neighborhoods (clique, to_neighbors) must be wired up by the harness
//   before calling this task.
//
//   Training: harness calls run_episode on all agents concurrently, collects
//             trajectory, then runs MAPPO gradient updates (eqs 2, 3, 4).
//   Test:     same task, just no gradient updates afterwards.

cot::task<void> Agent::run_episode(int T, int H, int F) {
    for (int t = 0; t < T; ++t) {
        co_await negotiation_phase(t, H, F);
        co_await launching_phase(t);
    }
}

} // namespace seqcomm
