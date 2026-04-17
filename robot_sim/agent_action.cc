#include "agent_action.hh"
#include <algorithm>
#include <numeric>
#include <random>
#include <print>

namespace seqcomm {

Agent::Agent(int id, int obs_dim, int action_dim,
             NeuralModels& models, Environment& env,
             std::vector<transition>& trajectory,
             random_source& rng)
    : id(id), obs_dim(obs_dim), action_dim(action_dim),
      from_agents(rng, "agent-" + std::to_string(id)),
      models(models), env(env), trajectory(trajectory),
      rng_(rng) {}

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

cot::task<float> Agent::compute_intention(
    const std::vector<pancy::hidden_state_msg>& neighbor_hs,
    int H, int F)
{
    constexpr float gamma = 0.99f;
    float total_return = 0.f;

    std::unordered_map<int, const std::vector<float>*> h_map;
    for (auto& nh : neighbor_hs)
        h_map[nh.sender_id] = &nh.h;

    for (int f = 0; f < F; ++f) {
        auto ordering = clique;
        std::shuffle(ordering.begin(), ordering.end(),
                     std::default_random_engine(rng_.uniform32()));

        float discounted_return = 0.f;
        float gamma_t = 1.f;
        std::vector<float> current_obs = obs;

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
                    auto it = h_map.find(agent_idx);
                    enc_j = (it != h_map.end()) ? *it->second
                                                : std::vector<float>(h.size(), 0.f);
                }

                std::vector<std::vector<float>> upper_for_j =
                    (agent_idx == id) ? upper_actions_for_self
                                      : std::vector<std::vector<float>>{};

                auto context_j = models.attention_a(enc_j, upper_for_j);
                auto [a_j, _lp] = models.policy_sample(context_j);

                all_enc.push_back(enc_j);
                all_actions.push_back(a_j);

                if (!self_acted && agent_idx != id)
                    upper_actions_for_self.push_back(a_j);
                if (agent_idx == id)
                    self_acted = true;
            }

            auto ctx_w = models.attention_w(all_enc, all_actions);
            auto [next_obs_flat, reward] = models.world_model(ctx_w);

            discounted_return += gamma_t * reward;
            gamma_t *= gamma;

            current_obs.assign(next_obs_flat.begin(),
                                next_obs_flat.begin() + obs_dim);
        }

        auto enc_final = models.encode(current_obs);
        auto ctx_final = models.attention_a(enc_final, {});
        float v_final  = models.critic(ctx_final);
        discounted_return += gamma_t * v_final;
        total_return += discounted_return;
    }

    co_return total_return / static_cast<float>(F);
}

cot::task<void> Agent::negotiation_phase(int t, int H, int F) {
    h = models.encode(obs);
    co_await broadcast(pancy::hidden_state_msg{id, h});

    std::vector<pancy::hidden_state_msg> neighbor_hs;
    neighbor_hs.reserve(clique.size());
    for (size_t k = 0; k < clique.size(); ++k)
        neighbor_hs.push_back(co_await receive_typed<pancy::hidden_state_msg>());

    float intention = co_await compute_intention(neighbor_hs, H, F);

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

    std::print("{:.3f}: agent {} t={} N_upper={} N_lower={}\n",
               cot::now(), id, t, N_upper.size(), N_lower.size());
}

cot::task<void> Agent::launching_phase(int t) {
    std::vector<std::vector<float>> a_upper;
    a_upper.reserve(N_upper.size());
    for (size_t k = 0; k < N_upper.size(); ++k) {
        auto msg = co_await receive_typed<pancy::upper_action_msg>();
        a_upper.push_back(std::move(msg.action));
    }

    auto context    = models.attention_a(h, a_upper);
    auto [a, log_p] = models.policy_sample(context);
    float log_p_old = models.policy_log_prob_old(context, a);
    float v         = models.critic(context);
    action          = a;

    for (int lower_id : N_lower)
        co_await send_to(lower_id, pancy::upper_action_msg{id, action});

    if (N_lower.empty()) {
        co_await broadcast(pancy::execute_signal{id, t});
    } else {
        co_await receive_typed<pancy::execute_signal>();
    }

    env.submit_action(id, action);
    auto [next_obs, reward] = env.get_result(id);

    trajectory.push_back({
        .agent_id      = id,
        .timestep      = t,
        .obs           = obs,
        .action        = action,
        .upper_actions = a_upper,
        .next_obs      = next_obs,
        .reward        = reward,
        .value         = v,
        .log_prob      = log_p,
        .log_prob_old  = log_p_old,
    });

    obs = std::move(next_obs);
}

cot::task<void> Agent::run_episode(int T, int H, int F) {
    for (int t = 0; t < T; ++t) {
        co_await negotiation_phase(t, H, F);
        co_await launching_phase(t);
    }
}

} // namespace seqcomm
