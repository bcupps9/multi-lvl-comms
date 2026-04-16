#include "client_model.hh"

// client_model.cc
//
//    Method bodies for `client_model`.

namespace cot = cotamer;
using namespace std::chrono_literals;

client_model::client_model(size_t nreplicas, random_source& randomness)
    : randomness_(randomness), out_(nreplicas), in_(randomness, "clients") {
    if (nreplicas == 0) {
        throw std::out_of_range("nreplicas must not be zero");
    }
    for (size_t i = 0; i != nreplicas; ++i) {
        out_[i].reset(new netsim::channel<pancy::request>(randomness, "clients"));
    }
    in_task_ = receive_task();
}

void client_model::connect_replica(size_t replicaid,
                netsim::port<pancy::request>& request_input,
                netsim::channel<pancy::response>& response_channel) {
    if (replicaid >= out_.size()) {
        throw std::out_of_range("replica out of range");
    }
    out_[replicaid]->connect(request_input);
    response_channel.connect(in_);
}

cotamer::task<> client_model::receive_task() {
    while (true) {
        auto msg = co_await in_.receive();

        // check whether client exists
        size_t client_id = pancy::message_serial(msg) & client_mask();
        if (client_id >= client_events_.size()) {
            continue;
        }

        // wake client, record message
        client_events_[client_id].trigger();
        inq_.push_back(std::move(msg));
    }
}

void client_model::stop() {
}
