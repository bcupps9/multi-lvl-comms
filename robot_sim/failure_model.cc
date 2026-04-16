// we have access to set_loss

namespace cot = cotamer;
using namespace std::chrono_literals;

// per_channel_schedule
//    Walks through a pre-generated sequence of (quiet_gap, loss_duration)
//    pairs, toggling set_loss on the channel for each window.
static cot::task<> per_channel_schedule(
    netsim::channel<paxos_message>& chan,
    std::vector<std::pair<std::chrono::milliseconds, std::chrono::milliseconds>> windows,
    double restore_loss
) {
    for (auto [gap, dur] : windows) {
        co_await cot::after(gap);
        chan.set_loss(1.0);
        co_await cot::after(dur);
        chan.set_loss(restore_loss);
    }
}

// initialize
//    For each directed link src->dst, generate an expo
//    sequence of (quiet, loss) windows that fits within total_duration, then
//    detach a per_channel_schedule coroutine for it.  All coroutines start
//    at simulation time 0 and run concurrently.
cot::task<> initialize_failure_model(pt_paxos_instance& inst, bool verbose,
        std::chrono::milliseconds total_duration) {
    auto& tester = inst.tester;
    auto total = total_duration;
    
    for (size_t src = 0; src < inst.replicas.size(); ++src) {
        for (size_t dst = 0; dst < inst.replicas.size(); ++dst) {
            if (src == dst) {
                continue;
            }

            // Build the on/off schedule for this channel as (gap, loss_dur) pairs.
            // Generate expos until we surpass the time for this round.
            std::vector<std::pair<std::chrono::milliseconds,
                                  std::chrono::milliseconds>> windows;
            std::chrono::milliseconds t = 0ms;
            while (t < total) {
                auto gap = tester.randomness.exponential(10000ms); // mean quiet: 10 s
                t += gap;
                if (t >= total) {
                    break;
                }
                auto dur = tester.randomness.exponential(3000ms);  // mean loss:  3 s
                if (t + dur > total) {
                    break;
                }
                windows.emplace_back(gap, dur);
                t += dur;
            }
            if (verbose) {
                // add debugging here to print windows
                for (auto& window : windows) {
                    std::print("src: {}, dest {}, start, dur\n", src, dst, window.first, window.second);
                }
            }

            // Detach one coroutine per channel — they all run concurrently.
            auto& chan = *inst.replicas[src]->to_replicas_[dst];
            per_channel_schedule(chan, std::move(windows), tester.loss).detach();
        }
    }
    co_return;
}
