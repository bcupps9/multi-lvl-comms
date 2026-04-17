#pragma once
#include <format>
#include <variant>
#include <vector>

// pancy_msgs.hh
//
//   Agent-to-agent message types for the SeqComm protocol.
//
//   All four message types share a single port<agent_msg> per agent.
//   receive_typed<MsgT>() in agent_action.hh dispatches by variant index.

namespace pancy {

// ── SeqComm messages ──────────────────────────────────────────────────────────

// Negotiation round 1: each agent broadcasts its encoded observation.
struct hidden_state_msg {
    int sender_id;
    std::vector<float> h;       // e(o_i), length = embed_dim
};

// Negotiation round 2: each agent broadcasts its estimated intention (return).
struct intention_msg {
    int sender_id;
    float intention;
};

// Launching phase: upper-ranked agent sends its chosen action downward.
struct upper_action_msg {
    int sender_id;
    std::vector<float> action;  // a_i sampled from pi
};

// Launching phase: lowest-ranked agent broadcasts to trigger environment step.
struct execute_signal {
    int sender_id;
    int timestep;
};

// Single variant carried on one port<agent_msg> per agent.
using agent_msg = std::variant<
    hidden_state_msg,
    intention_msg,
    upper_action_msg,
    execute_signal
>;

} // namespace pancy


// ── Formatters ────────────────────────────────────────────────────────────────

template <typename CharT>
struct std::formatter<pancy::hidden_state_msg, CharT> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    template <typename Ctx>
    auto format(const pancy::hidden_state_msg& m, Ctx& ctx) const {
        return std::format_to(ctx.out(), "HIDDEN_STATE(from={}, dim={})",
                              m.sender_id, m.h.size());
    }
};

template <typename CharT>
struct std::formatter<pancy::intention_msg, CharT> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    template <typename Ctx>
    auto format(const pancy::intention_msg& m, Ctx& ctx) const {
        return std::format_to(ctx.out(), "INTENTION(from={}, I={:.4f})",
                              m.sender_id, m.intention);
    }
};

template <typename CharT>
struct std::formatter<pancy::upper_action_msg, CharT> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    template <typename Ctx>
    auto format(const pancy::upper_action_msg& m, Ctx& ctx) const {
        return std::format_to(ctx.out(), "UPPER_ACTION(from={}, dim={})",
                              m.sender_id, m.action.size());
    }
};

template <typename CharT>
struct std::formatter<pancy::execute_signal, CharT> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    template <typename Ctx>
    auto format(const pancy::execute_signal& m, Ctx& ctx) const {
        return std::format_to(ctx.out(), "EXECUTE(from={}, t={})",
                              m.sender_id, m.timestep);
    }
};
