#pragma once
#include "cotamer/cotamer.hh"
#include "netsim.hh"
#include "pancy_msgs.hh"
#include "random_source.hh"
#include <deque>
#include <span>
#include <unordered_map>
#include <vector>

// agent_action.hh
//
//   SeqComm agent with two cotamer phases per timestep:
//     negotiation_phase  — share hidden states, compute intentions, set priority
//     launching_phase    — cascade actions top-down, trigger environment step
//
//   NeuralModels and Environment are abstract; swap in libtorch implementations.

namespace seqcomm {
namespace cot = cotamer;


// ── Abstract neural model interface ──────────────────────────────────────────
//
//   All tensors passed as std::vector<float> for portability.
//   Replace with torch::Tensor overloads when wiring up libtorch.

struct NeuralModels {
    virtual ~NeuralModels() = default;

    // e(o): encode raw observation → hidden state h
    virtual std::vector<float> encode(std::span<const float> obs) = 0;

    // AM_a(h_self, messages): attention over hidden states (negotiation) or
    //   upper actions (launching) → context vector for policy / critic
    virtual std::vector<float> attention_a(
        std::span<const float> h_self,
        const std::vector<std::vector<float>>& messages) = 0;

    // AM_w(enc_obs_all, actions_all): attention for world model input
    virtual std::vector<float> attention_w(
        const std::vector<std::vector<float>>& enc_obs,
        const std::vector<std::vector<float>>& actions) = 0;

    // π(·|context): sample action and return (action, log_prob)
    virtual std::pair<std::vector<float>, float>
    policy_sample(std::span<const float> context) = 0;

    // π_old log prob of a given action under the old policy (for PPO ratio)
    virtual float policy_log_prob_old(
        std::span<const float> context,
        std::span<const float> action) = 0;

    // V(context): scalar value estimate
    virtual float critic(std::span<const float> context) = 0;

    // M(context_w): world model → (next_obs_all_agents_flat, reward)
    virtual std::pair<std::vector<float>, float>
    world_model(std::span<const float> context_w) = 0;

    // Comm gate logit: σ(comm_gate(h)) = P(agent chooses to communicate).
    // Default returns +inf so agents always communicate when no gate is loaded.
    virtual float comm_gate(std::span<const float> /*h*/) { return 1e9f; }
};


// ── Abstract environment interface ───────────────────────────────────────────
//
//   The simulation harness implements this. After all agents call
//   submit_action(), get_result() blocks until the environment has stepped
//   and returns (next_obs, reward) for this agent.

struct Environment {
    virtual ~Environment() = default;
    virtual void submit_action(int agent_id,
                               std::span<const float> action) = 0;
    // Async: suspends until all agents have submitted and the env has stepped.
    virtual cot::task<std::pair<std::vector<float>, float>>
    get_result(int agent_id) = 0;
};


// ── Trajectory data ───────────────────────────────────────────────────────────

struct transition {
    int agent_id;
    int timestep;
    std::vector<float> obs;
    std::vector<float> action;
    std::vector<std::vector<float>> upper_actions; // {a_j : j ∈ N_upper}
    std::vector<float> next_obs;
    float reward    = 0.f;
    float value     = 0.f;   // V estimate at t, used for GAE
    float log_prob  = 0.f;   // log π(a|context), used for PPO ratio
    float log_prob_old = 0.f;
};

struct CommGateDecision {
    bool did_comm = false;
    float gate_logit = 0.f;
    std::vector<float> h;
};


// ── Agent ─────────────────────────────────────────────────────────────────────

struct Agent {
    int id;
    int obs_dim;
    int action_dim;

    // Inbound port — every neighbor sends into this one port (variant-typed)
    netsim::port<pancy::agent_msg> from_agents;

    // Outbound channels — non-owning; harness creates and connects these.
    // to_neighbors[k] is the channel from this agent to clique[k]'s port.
    // Invariant: to_neighbors.size() == clique.size()
    std::vector<netsim::channel<pancy::agent_msg>*> to_neighbors;

    std::vector<int> clique;    // all agent ids in comm range (populated at init)
    std::vector<int> N_upper;   // ids with higher intention this timestep
    std::vector<int> N_lower;   // ids with lower intention this timestep

    NeuralModels& models;
    Environment&  env;
    std::vector<transition>& trajectory; // shared buffer; appended each step

    // Per-timestep state (refreshed at start of each negotiation_phase)
    std::vector<float> obs;
    std::vector<float> h;      // e(obs), set in negotiation_phase
    std::vector<float> action; // set in launching_phase

    bool verbose = false;

    // Optional: filled by negotiation_phase each timestep so the harness can
    // compute per-step intention spread without re-running inference.
    std::vector<float>* own_intentions = nullptr;

    // ── Experiment 1: World Model intention ordering ──────────────────────────
    // When true, negotiation_phase broadcasts h, collects all neighbours' h, and
    // runs a wm_H-step world-model rollout to produce the ordering signal.
    bool  use_wm_intention = false;
    int   wm_H             = 2;    // rollout horizon (paper default: H=2)

    // Comms loss: per-step flag set by the harness before each negotiation call.
    // When true, negotiation is skipped and ordering is random.
    // The harness pre-computes one decision per timestep shared across all agents
    // so they always agree on whether the channel failed.
    const std::vector<bool>* comms_failed_per_step = nullptr;
    float comms_loss_prob  = 0.f;  // kept for logging only; harness uses the vector

    // Per-episode counters updated by negotiation_phase (harness owns storage).
    int*  comms_ok_count    = nullptr;  // steps where ordering used real comms
    int*  comms_total_count = nullptr;  // total negotiation steps attempted

    // ── Experiment 2: Optional communication gate ─────────────────────────────
    // When true, each agent independently samples a comm decision at the start
    // of each negotiation step. If any agent opts out, ordering falls back to
    // random. Agents that choose to comm incur comm_penalty in their reward.
    bool  use_comm_gate      = false;
    float comm_penalty       = 0.f;   // reward cost for choosing to communicate
    bool  did_comm_this_step = false;  // set in negotiation, read in launching

    // Per-step sidecar log for Python REINFORCE.
    // Harness owns the vector; nullptr = no logging.
    std::vector<CommGateDecision>* comm_log = nullptr;

    Agent(int id, int obs_dim, int action_dim,
          NeuralModels& models, Environment& env,
          std::vector<transition>& trajectory,
          random_source& rng,
          bool verbose = false);

    // ── Cotamer tasks ─────────────────────────────────────────────────────────

    // Encode obs, compute V(h_i) as ordering signal, set N_upper/N_lower.
    // H and F params accepted but unused (retained for call-site compatibility).
    cot::task<void> negotiation_phase(int t, int H, int F);

    // Wait for upper actions, sample own action, broadcast down, execute
    cot::task<void> launching_phase(int t);

    // Run one full episode of T timesteps
    cot::task<void> run_episode(int T, int H, int F);

private:
    random_source& rng_;

    // Receive the next message of type MsgT; buffer anything else
    template <typename MsgT>
    cot::task<MsgT> receive_typed();

    // Send msg to the channel for a specific neighbor id
    cot::task<void> send_to(int neighbor_id, pancy::agent_msg msg);

    // Broadcast msg to all neighbors
    cot::task<void> broadcast(pancy::agent_msg msg);

    std::deque<pancy::agent_msg> pending_; // messages waiting to be consumed
};


// ── receive_typed — defined here because it is a template ────────────────────

template <typename MsgT>
cot::task<MsgT> Agent::receive_typed() {
    // drain pending buffer first
    for (auto it = pending_.begin(); it != pending_.end(); ++it) {
        if (std::holds_alternative<MsgT>(*it)) {
            MsgT m = std::get<MsgT>(std::move(*it));
            pending_.erase(it);
            co_return m;
        }
    }
    // wait for a new message
    while (true) {
        auto msg = co_await from_agents.receive();
        if (std::holds_alternative<MsgT>(msg)) {
            co_return std::get<MsgT>(std::move(msg));
        }
        pending_.push_back(std::move(msg));
    }
}

} // namespace seqcomm
