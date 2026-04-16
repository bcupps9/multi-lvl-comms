#pragma once
#include "pancy_msgs.hh"
#include "pancydb.hh"
#include "netsim.hh"
#include "cotamer/cotamer.hh"
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

// client_model
//
//    Models a set of Pancy clients.
//
//    The `client_model` abstract class knows how to connect to a set of
//    replicas. It encapsulates ports, channels, and a source of randomness,
//    and has convenience functions for sending and receiving messages.
//
//    Derived classes are expected to implement specific test protocols.
//    `start()` starts the test protocol, and `stop()` stops it (as does
//    destroying the `client_model`).
//
//    We provide one derived test protocol in `lockseq_model.hh/cc`.

class client_model {
public:
    client_model(size_t nreplicas, random_source& randomness);
    virtual ~client_model() = default;
    client_model(const client_model&) = delete;
    client_model(client_model&&) = delete;
    client_model& operator=(const client_model&) = delete;
    client_model& operator=(client_model&&) = delete;

    // - Source of randomness
    random_source& randomness() { return randomness_; }

    // - Configured number of replicas
    size_t nreplicas() const noexcept { return out_.size(); }
    // - Connect a replica
    void connect_replica(size_t replicaid,
                         netsim::port<pancy::request>& request_input,
                         netsim::channel<pancy::response>& response_channel);
    // - Return a random replica index
    size_t random_replica() { return randomness_.uniform(size_t(0), nreplicas() - 1); }
    // - Return the channel to a replica
    netsim::channel<pancy::request>& request_channel(size_t replicaid);

    // - Maximum number of simulated clients; must be a power of 2
    size_t max_clients() const noexcept { return 4096; }
    // - Set number of clients
    inline void set_nclients(size_t n);
    // - Client mask: these bits of a request serial number equal client ID;
    //   a model supports at most `client_mask() + 1` simulated clients
    uint64_t client_mask() const noexcept { return max_clients() - 1; }
    // - Serial step: Each subsequent message from a simulated client should
    //   advance `serial` by this amount
    size_t serial_step() const noexcept { return 4096; }
    // - Return client event corresponding to received messages
    cotamer::event& client_event(size_t cid) { return client_events_[cid]; }

    // - Send a request
    template <pancy::request_type Req, typename... Args>
    cotamer::task<> send_request(
        size_t replicaid, uint64_t serial, Args... args
    );
    // - Receive a response, potentially by redirecting `replicaid`
    template <pancy::response_type Resp>
    cotamer::task<std::optional<Resp>> receive_response(
        size_t& replicaid, uint64_t serial
    );


    // - Start the model
    virtual void start() = 0;
    // - Stop the model
    virtual void stop();

    // - Test whether `db` could have been created by clients of this model.
    //   Returns `std::nullopt` on success, and a bad key on failure.
    virtual std::optional<std::string> check(const pancy::pancydb& db) = 0;

private:
    random_source& randomness_;

    // channels for sending requests to replicas
    std::vector<std::unique_ptr<netsim::channel<pancy::request>>> out_;

    // port for receiving responses from replicas
    netsim::port<pancy::response> in_;
    std::list<pancy::response> inq_;
    cotamer::task<> in_task_;

    // receiver events
    std::vector<cotamer::event> client_events_;

    cotamer::task<> receive_task();
};

inline netsim::channel<pancy::request>& client_model::request_channel(size_t replicaid) {
    if (replicaid >= out_.size()) {
        throw std::out_of_range("replicaid out of range");
    }
    return *out_[replicaid];
}

template <pancy::request_type Req, typename... Args>
inline cotamer::task<> client_model::send_request(
    size_t replicaid, uint64_t serial, Args... args
) {
    return out_[replicaid]->send(Req{{serial}, std::forward<Args>(args)...});
}

template <pancy::response_type Resp>
inline cotamer::task<std::optional<Resp>> client_model::receive_response(
    size_t& replicaid, uint64_t serial
) {
    size_t cid = serial & client_mask();
    while (true) {
        // check incoming queue for response
        for (auto it = inq_.begin(); it != inq_.end(); ) {
            auto in_serial = message_serial(*it);
            if ((in_serial & client_mask()) != cid) {
                ++it;
            } else if (in_serial != serial) {
                it = inq_.erase(it);
            } else if (auto rp = std::get_if<pancy::redirection_response>(&*it)) {
                replicaid = rp->redirection;
                it = inq_.erase(it);
                co_return std::nullopt;
            } else if (auto mp = std::get_if<Resp>(&*it)) {
                auto m = std::move(*mp);
                it = inq_.erase(it);
                co_return m;
            } else {
                it = inq_.erase(it);
                co_return std::nullopt;
            }
        }
        // wait until message arrives
        cotamer::driver_guard guard;
        co_await client_events_[cid].arm();
    }
}

inline void client_model::set_nclients(size_t n) {
    if (n > max_clients()) {
        throw std::out_of_range("nclients out of range");
    }
    client_events_.resize(n);
}
