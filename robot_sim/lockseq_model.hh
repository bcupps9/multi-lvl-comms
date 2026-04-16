#pragma once
#include "client_model.hh"

// lockseq_model
//
//    Models a particular Pancy client-based test protocol. 32 clients
//    repeatedly lock a randomly-chosen range of keys (via compare-and-swap),
//    overwrite a sequence of keys within that range, remove vestiges of any
//    previous sequence, and then unlock. The `check(pancydb&)` function
//    validates that a pancydb database could have been created by a valid
//    client execution, and can find bugs in consensus implementations.

class lockseq_model : public client_model {
public:
    lockseq_model(size_t nreplicas, random_source& randomness);

    void start();
    void stop();

    std::optional<std::string> check(const pancy::pancydb& db);

    static inline std::string make_client_key(unsigned cid);
    static inline std::string make_group_key(unsigned gid);
    static inline std::string make_lock_key(unsigned gid);
    static inline std::string make_value_key(unsigned gid, unsigned vid);

    // phase completion counters (across all clients)
    unsigned long lock_complete = 0;
    unsigned long write_complete = 0;
    unsigned long clear_complete = 0;
    unsigned long unlock_complete = 0;

private:
    struct cstate { // state for a single "client"
        cotamer::task<> task;   // the client() task (to facilitate cleanup)
        size_t leader;          // this client’s idea of the current leader
    };

    size_t nclients_ = 32;
    unsigned ngroups_ = 100;
    unsigned max_sequence_length_ = 100;
    std::vector<std::unique_ptr<cstate>> clients_;

    cotamer::task<> client(unsigned cid);
};

inline std::string lockseq_model::make_client_key(unsigned cid) {
    return std::format("c{}", cid);
}
inline std::string lockseq_model::make_group_key(unsigned gid) {
    return std::format("g{}", gid);
}
inline std::string lockseq_model::make_lock_key(unsigned gid) {
    return std::format("g{}/lock", gid);
}
inline std::string lockseq_model::make_value_key(unsigned gid, unsigned vid) {
    return std::format("g{}/v{:03}", gid, vid);
}
