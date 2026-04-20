#include "agent_action.hh"
#include <algorithm>
#include <numeric>
#include <random>
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


// ── compute_intention ─────────────────────────────────────────────────────────
//
//   Algorithm 5: for each of F sampled orderings, simulate H world-model steps.
//   At each step the full action cascade runs: agents act in sampled priority
//   order, upper actions flow down, world model predicts (o', r), repeat.
//   Bootstrap with V at the end of the rollout. Return mean return over F.

cot::task<float> Agent::compute_intention(
    const std::vector<pancy::hidden_state_msg>& neighbor_hs,
    int H, int F)
{
    constexpr float gamma = 0.99f;
    float total_return = 0.f;

    // Build lookup: neighbor id → their current hidden state
    std::unordered_map<int, const std::vector<float>*> h_map;
    for (auto& nh : neighbor_hs)
        h_map[nh.sender_id] = &nh.h;

    for (int f = 0; f < F; ++f) {
        // Sample a random ordering of all clique members
        auto ordering = clique;
        // TODO: use rng_ for reproducible shuffles once random_source exposes
        //       an std::mt19937-compatible interface; placeholder below
        std::shuffle(ordering.begin(), ordering.end(), rng_.engine());

        float discounted_return = 0.f;
        float gamma_t = 1.f;
        std::vector<float> current_obs = obs; // agent's own rolling obs

        for (int step = 0; step < H; ++step) {
            std::vector<std::vector<float>> all_enc;
            std::vector<std::vector<float>> all_actions;
            std::vector<std::vector<float>> upper_actions_for_self;

            bool self_acted = false;

            for (int agent_idx : ordering) {
                std::vector<float> enc_j;
                if (agent_idx == id) {
                    enc_j = models.encode(current_obs);
                } else {
                    // Use hidden state shared in negotiation round 1
                    auto it = h_map.find(agent_idx);
                    enc_j = (it != h_map.end()) ? *it->second : std::vector<float>(h.size(), 0.f);
                }

                // upper_actions_for_self accumulates only agents that are
                // upper relative to self in *this* sampled ordering
                std::vector<std::vector<float>> upper_for_j =
                    (agent_idx == id) ? upper_actions_for_self
                                      : std::vector<std::vector<float>>{};

                auto context_j = models.attention_a(enc_j, upper_for_j);
                auto [a_j, _lp] = models.policy_sample(context_j);

                all_enc.push_back(enc_j);
                all_actions.push_back(a_j);

                // Agents before self in this ordering count as its upper agents
                if (!self_acted && agent_idx != id)
                    upper_actions_for_self.push_back(a_j);
                if (agent_idx == id)
                    self_acted = true;
            }

            // World-model step: M(AM_w(e(o), a)) → (o'_all_flat, r)
            auto ctx_w = models.attention_w(all_enc, all_actions);
            auto [next_obs_flat, reward] = models.world_model(ctx_w);

            discounted_return += gamma_t * reward;
            gamma_t *= gamma;

            // Advance self's obs (first obs_dim floats of the flat output)
            current_obs.assign(next_obs_flat.begin(),
                                next_obs_flat.begin() + obs_dim);
        }

        // Bootstrap: V(AM_a(e(o_H), {})) — no upper actions at bootstrap
        auto enc_final = models.encode(current_obs);
        auto ctx_final = models.attention_a(enc_final, {});
        float v_final  = models.critic(ctx_final);
        discounted_return += gamma_t * v_final;

        total_return += discounted_return;
    }

    co_return total_return / static_cast<float>(F);
}


// ── negotiation_phase ─────────────────────────────────────────────────────────
//
//   Round 1: encode obs → h_i, broadcast hidden_state_msg to clique.
//   Round 2: compute intention via H-step world-model rollouts,
//            broadcast intention_msg, receive all intentions, set N_upper/N_lower.

cot::task<void> Agent::negotiation_phase(int t, int H, int F) {
    // Round 1 ─ share encoded hidden state
    h = models.encode(obs);
    co_await broadcast(pancy::hidden_state_msg{id, h});

    std::vector<pancy::hidden_state_msg> neighbor_hs;
    neighbor_hs.reserve(clique.size());
    for (size_t k = 0; k < clique.size(); ++k)
        neighbor_hs.push_back(co_await receive_typed<pancy::hidden_state_msg>());

    // Compute intention (Algorithm 5)
    float intention = co_await compute_intention(neighbor_hs, H, F);

    // Round 2 ─ share intention value
    co_await broadcast(pancy::intention_msg{id, intention});

    N_upper.clear();
    N_lower.clear();
    for (size_t k = 0; k < clique.size(); ++k) {
        auto msg = co_await receive_typed<pancy::intention_msg>();
        if (msg.intention > intention)
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
