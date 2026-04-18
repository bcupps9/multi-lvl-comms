#pragma once
// trajectory_io.hh
//
//   Flat binary serialization of std::vector<transition> for hand-off to Python.
//
// File layout
// ───────────
//   [header — 16 bytes]
//     int32  n_agents
//     int32  obs_dim
//     int32  action_dim
//     int32  n_transitions
//
//   [n_transitions records, each:]
//     int32                           agent_id
//     int32                           timestep
//     int32                           n_upper
//     float32[obs_dim]                obs
//     float32[action_dim]             action
//     float32[n_agents * action_dim]  upper_actions  (first n_upper slots filled,
//                                                      rest zero-padded)
//     float32[obs_dim]                next_obs
//     float32                         reward
//     float32                         value
//     float32                         log_prob
//     float32                         log_prob_old
//
//   Python side: read with numpy.frombuffer, reassemble per-agent tensors
//   into the same dict format that training/train.py's run_episode() produces.

#include "agent_action.hh"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace seqcomm {

inline void write_trajectory(
    const std::string& path,
    const std::vector<transition>& traj,
    int n_agents,
    int obs_dim,
    int action_dim)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open for writing: " + path);

    auto wi  = [&](int32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto wf  = [&](float v)   { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto wfv = [&](const std::vector<float>& v) {
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * 4));
    };

    // Header
    wi(n_agents);
    wi(obs_dim);
    wi(action_dim);
    wi(static_cast<int32_t>(traj.size()));

    for (const auto& tr : traj) {
        wi(tr.agent_id);
        wi(tr.timestep);
        wi(static_cast<int32_t>(tr.upper_actions.size()));

        wfv(tr.obs);
        wfv(tr.action);

        // Upper actions padded to n_agents slots (first n_upper filled).
        // Matches Python's up_pad layout: slot l = l-th upper agent's action.
        for (int slot = 0; slot < n_agents; ++slot) {
            if (slot < static_cast<int>(tr.upper_actions.size()))
                wfv(tr.upper_actions[slot]);
            else
                for (int d = 0; d < action_dim; ++d) wf(0.f);
        }

        wfv(tr.next_obs);
        wf(tr.reward);
        wf(tr.value);
        wf(tr.log_prob);
        wf(tr.log_prob_old);
    }
}

} // namespace seqcomm
