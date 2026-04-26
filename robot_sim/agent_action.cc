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
//   Encode obs → h_i, compute intention as V(h_i) (critic value with no
//   upper context), broadcast scalar intention, receive all, set N_upper/N_lower.
//
//   Deliberately shares only the scalar intention — NOT h_i — so the cascade
//   remains the only inter-agent information channel at execution time.
//   H_bar can only be inferred from bet sizes, keeping the coordination
//   problem non-trivial.
//
//   Tie-breaking by agent ID ensures a total order even with untrained
//   (near-uniform) critics, preventing execute_signal deadlock.

cot::task<void> Agent::negotiation_phase(int t, int /*H*/, int /*F*/) {
    h = models.encode(obs);

    // Intention proxy: critic value at own state with no upper context.
    // High H_i → high expected return → high V → correct ordering signal.
    auto ctx      = models.attention_a(h, {});
    float intention = models.critic(ctx);
    if (own_intentions) own_intentions->push_back(intention);

    co_await broadcast(pancy::intention_msg{id, intention});

    N_upper.clear();
    N_lower.clear();
    for (size_t k = 0; k < clique.size(); ++k) {
        auto msg = co_await receive_typed<pancy::intention_msg>();
        // Tiebreak by agent ID → guaranteed total order even with identical
        // intentions (e.g. untrained models), preventing deadlock.
        bool is_upper = msg.intention > intention ||
            (msg.intention == intention && msg.sender_id > id);
        if (is_upper)
            N_upper.push_back(msg.sender_id);
        else
            N_lower.push_back(msg.sender_id);
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

    // Record transition
    trajectory.push_back({
        .agent_id    = id,
        .timestep    = t,
        .obs         = obs,
        .action      = action,
        .upper_actions = a_upper,
        .next_obs    = next_obs,
        .reward      = reward,
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
