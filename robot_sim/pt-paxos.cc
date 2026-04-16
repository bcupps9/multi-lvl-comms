#include "lockseq_model.hh"
#include "pancydb.hh"
#include "netsim.hh"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <set>

namespace cot = cotamer;
using namespace std::chrono_literals;

// testinfo
//    Holds configuration information about this test.

struct testinfo {
    random_source randomness;
    double loss = 0.0;
    bool verbose = false;
    bool print_db = false;
    size_t nreplicas = 3;
    size_t initial_leader = 0;
    // hook to set a channel to desired loss rate for given period
    struct scheduled_link_loss {
        size_t src = 0;
        size_t dst = 0;
        std::chrono::milliseconds start = 0ms;
        std::chrono::milliseconds duration = 0ms;
        double loss = 1.0;
    };
    // storage for all of the losses that will occur
    std::vector<scheduled_link_loss> link_losses;

    template <typename T>
    void configure_port(netsim::port<T>& port) {
        port.set_verbose(verbose);
    }
    template <typename T>
    void configure_channel(netsim::channel<T>& chan) {
        chan.set_loss(loss);
        chan.set_verbose(verbose);
    }
    template <typename T>
    void configure_quiet_channel(netsim::channel<T>& chan) {
        chan.set_loss(loss);
    }
};

// conversion from raw CLI input into an interpretable format for cot::await and set_loss
// turns into a struct of the above type
testinfo::scheduled_link_loss parse_link_loss_spec(const std::string& spec) {
    std::vector<std::string> parts;
    size_t begin = 0;
    while (begin <= spec.size()) {
        auto comma = spec.find(',', begin);
        // get to end of string
        if (comma == std::string::npos) {
            // push back the rest
            parts.push_back(spec.substr(begin));
            break;
        }
        parts.push_back(spec.substr(begin, comma - begin));
        begin = comma + 1;
    }

    if (parts.size() != 4 && parts.size() != 5) {
        throw std::invalid_argument(
            "expected SRC,DST,START_MS,DURATION_MS[,LOSS]");
    }

    auto src = from_str_chars<size_t>(parts[0]);
    auto dst = from_str_chars<size_t>(parts[1]);
    auto start_ms = from_str_chars<long long>(parts[2]);
    auto duration_ms = from_str_chars<long long>(parts[3]);
    double loss = parts.size() == 5 ? from_str_chars<double>(parts[4]) : 1.0;

    if (start_ms < 0) {
        throw std::invalid_argument("START_MS must be nonnegative");
    }
    if (duration_ms < 0) {
        throw std::invalid_argument("DURATION_MS must be nonnegative");
    }
    if (loss < 0.0 || loss > 1.0) {
        throw std::invalid_argument("LOSS must be between 0 and 1");
    }

    return {
        src,
        dst,
        std::chrono::milliseconds(start_ms),
        std::chrono::milliseconds(duration_ms),
        loss
    };
}

// RULES
// sort by src first, then by dst, then by start. All increasing. 
void validate_link_losses(const testinfo& tester) {
    auto link_losses = tester.link_losses;
    std::sort(link_losses.begin(), link_losses.end(),
              [](const auto& a, const auto& b) {
                  if (a.src != b.src) {
                      return a.src < b.src;
                  }
                  if (a.dst != b.dst) {
                      return a.dst < b.dst;
                  }
                  return a.start < b.start;
              });

    for (size_t i = 0; i != link_losses.size(); ++i) {
        const auto& link_loss = link_losses[i];
        if (link_loss.src == link_loss.dst) {
            throw std::invalid_argument("SRC and DST must name different replicas");
        }
        if (link_loss.src >= tester.nreplicas || link_loss.dst >= tester.nreplicas) {
            throw std::invalid_argument(std::format(
                "replica index out of range for {} replicas", tester.nreplicas));
        }
        if (i > 0) {
            const auto& prev = link_losses[i - 1];
            if (prev.src == link_loss.src && prev.dst == link_loss.dst) {
                auto prev_end = prev.start + prev.duration;
                if (link_loss.start < prev_end) {
                    throw std::invalid_argument(std::format(
                        "overlapping windows on replica link {} -> {}", prev.src, prev.dst));
                }
            }
        }
    }
}


// pt_paxos_replica, pt_paxos_instance
//    Manage a test of a Paxos-based Pancy service.
//    Initialization is more complicated than in the simpler settings;
//    we have a type, `pt_paxos_replica`, that represents a single replica,
//    and another, `pt_paxos_instance`, that constructs the replica set.

struct pt_paxos_instance;

// waiters that enable triggers when each fallback path triggers
// quorum -> recovery -> do_view_change -> start_view_change is the fallback order
struct quorum_waiter {
    int ack_count = 1;
    cotamer::event new_ack;
};

struct recovery_waiter {
    int ack_count = 0;
    std::unordered_set<size_t> senders;
    cotamer::event new_ack;
    pancy::view_id_t max_view_seen = -1;
    bool have_primary_reply_for_max_view = false;
    pancy::view_id_t best_view_id = -1;
    pancy::commit_num_t best_commit_num = 0;
    std::vector<pancy::log_entry> best_log;
};

struct start_view_change_waiter {
    std::unordered_set<size_t> senders;
    cotamer::event new_msg;
    bool started = false;
    bool broadcasted = false;
    bool sent_do_view_change = false;
};

struct do_view_change_waiter {
    std::unordered_set<size_t> senders;
    cotamer::event new_msg;
    bool started = false;
    bool sent_start_view = false;
    pancy::view_id_t best_last_normal_view = -1;
    pancy::commit_num_t best_commit_num = 0;
    std::vector<pancy::log_entry> best_log;
};

// The type for inter-replica messages. You will change this!
// initialially send everything
using paxos_message = pancy::inter_replica_msg;

struct pt_paxos_replica {
    size_t index_;           // index of this replica in the replica set
    size_t nreplicas_;       // number of replicas -- ids 0 through nreplicas-1
    size_t initial_leader_ = 0;
    size_t leader_index_ = 0;    // this replica’s idea of the current leader
    random_source& randomness_;
    
    // new elements i've added
    pancy::status status_ = pancy::status::active; // status from VR paper
    std::vector<pancy::client_prog> client_table_; // tracks the most recent request from each client and its response
    pancy::op_num_t op_num_ = 0; // operation number for the next op
    pancy::view_id_t view_id_ = 0; // current view id, increases on each view change
    pancy::commit_num_t commit_num_ = 0; // one greater than highest op_num that has been committed
    std::vector<pancy::log_entry> log_; // the log of operations that have been proposed
    std::unordered_map<pancy::op_num_t, quorum_waiter> pending_quorums_; // wait for prepare_oks  
    std::unordered_map<std::string, recovery_waiter> pending_recovery_quorums_; // wait for recover_oks
    std::unordered_map<pancy::view_id_t, start_view_change_waiter> pending_start_view_changes_; // wait for start view change
    std::unordered_map<pancy::view_id_t, do_view_change_waiter> pending_do_view_changes_; // wait for do view changes
    
    // when waiting for recovery, nonces must match in case stale recover_ok received
    std::optional<std::string> active_recovery_nonce_;
    // in recovery and do_view_change, first progress tiebreaker
    pancy::view_id_t last_good_view_id_ = 0;


    // Timeout Logic
    // In simulation, doesn't matter so much
    // ORDERING: 
    // 1. VIEW_CHANGE - most extreme, try recovery first to keep most servers active
    // 2. PREPARE - two chances for the presumed leader to send a message, then move on
    // 3. PREPAREOK - shorter than prepare because no retry mechanism under it but gives a bit of buffer to steady over commit timeout
    // 4. FROM_CLIENT_COMMIT - We'd rather send another commit than start a view change
    static constexpr auto prepare_timeout = 100ms; // clients change views if have nothign from leader
    static constexpr auto prepareok_timeout = 90ms; // leader move views if has nothing from clients post prepare
    static constexpr auto from_client_commit_timeout = 40ms; // leader send a commit if has nothing from clients
    static constexpr auto view_change_timeout = 200ms; // ultimate fallback, message not coming regardless of role. Already in a view_change stuff slows

    netsim::port<pancy::request> from_clients_;   // port for client messages
    netsim::port<paxos_message> from_replicas_;   // port for inter-replica messages
    netsim::channel<pancy::response> to_clients_; // channel for client responses
    // channels for inter-replica messages:
    std::vector<std::unique_ptr<netsim::channel<paxos_message>>> to_replicas_;
    pancy::pancydb db_;      // our copy of the database
    std::unordered_map<size_t, pancy::pancydb> db_map_;
    pt_paxos_replica(size_t index, size_t nreplicas, random_source&);
    void initialize(pt_paxos_instance&);

    // coroutines
    cot::task<> run_client_interaction(); // get messages from client. Handle if leader redist if replica
    cot::task<> run_inter_replica_interaction(); // interaction between replicas, switch statements to do handle the coroutines that follow
    cot::task<> wait_for_quorum(pancy::op_num_t op_num); // leader track prepareoks
    cot::task<> wait_for_recovery_quorum(const std::string& nonce); // recovering node track recoveryoks
    cot::task<> recover(std::string nonce); // go into recovery mode, only cares about itself. Receive okays or do view change
    size_t primary_for_view(pancy::view_id_t view_id) const; // helper to return primary for the view
    void apply_commits_through(pancy::commit_num_t target_commit_num); // after commit apply to db
    void rebuild_state_from_log(const std::vector<pancy::log_entry>& log, 
                                pancy::commit_num_t target_commit_num); // take in log and apply commits up to commit number. Rep now has that state
    
    // VIEW CHANGES
    // start view change: is a route into phase one that marks the an attempt at a view change
    // FOR a replica and then detaches the phase one which starts the wait for a quorum to broadcast
    // if it gets it then begins the do quorum and proceeds

    // consider do view change: simply does the comparisons in view_id and commit num
    // First checks whether last good incoming view is higher
    // Then checks size of the log 
    // Then checks commit number (unecessary to commit as much as possible in one go)
    
    // phase one: for a node marks it's waiter for the view change as started so we don't
    // redetach another phase one starter
    // clears it's pending quorums, resets status cutting recovery and puts to view_change
    // Once it gets a start view change sends all out and marks broadcasted
    // f + 1 start responses then does phase two if leader or sends a do_view
    // to the leader if not the leader, but marks do_view_change waiter for this 
    // view for this replica as started. The start markers in this single threaded
    // code assure that we do not detach a extra phase one or phase two cots 

    // phase two: The do view_change marker is marked active either because the leader receives
    // a do view change before it got to a quorum of okays for the start view change or gets to quorum of starts itself
    // if direct receive do - sets up waiter and marks as started
    // if gets there itself - waiter has been set up and we have marked as started in phase one
    // we wait for a quorum of dos all of which update the best information that is stored in our waiters
    // we get to a quorum and both update ourselves and then start view with the best information that we have
    // stored in the waiters
    void start_view_change_if_needed(pancy::view_id_t new_view_id); // just activates phase one
    void consider_do_view_change_state(do_view_change_waiter& waiter,
                                       pancy::commit_num_t commit_num,
                                       const std::vector<pancy::log_entry>& log,
                                       pancy::view_id_t last_normal_view);
    cot::task<> phase_one_view_change(pancy::view_id_t new_view_id);
    cot::task<> phase_two_view_change(pancy::view_id_t new_view_id);
};

struct pt_paxos_instance {
    testinfo& tester;
    client_model& clients;
    std::vector<std::unique_ptr<pt_paxos_replica>> replicas; // can be added to -- class
    // ...plus anything you want to add
    // we don't need anything here, The idea is dist so hopefully nothing is here
    std::set<size_t> failed_replicas = {};
    pt_paxos_instance(testinfo&, client_model&);
};


// Configuration and initialization
// these replicas though if we were viewstamped would be grouped so that cohorts aren't associated
pt_paxos_replica::pt_paxos_replica(size_t index, size_t nreplicas, random_source& randomness)
    : index_(index),
      nreplicas_(nreplicas),
      initial_leader_(0),
      randomness_(randomness),
      from_clients_(randomness, std::format("R{}", index_)),
      from_replicas_(randomness, std::format("R{}/r", index_)),
      to_clients_(randomness, from_clients_.id()), // the port has an id on it
      to_replicas_(nreplicas) {
    for (size_t s = 0UL; s != nreplicas_; ++s) {
        to_replicas_[s].reset(new netsim::channel<paxos_message>(
            randomness, from_clients_.id()
        ));
    }
    client_table_.resize(4096); 
}

void pt_paxos_replica::initialize(pt_paxos_instance& inst) {
    initial_leader_ = inst.tester.initial_leader;
    leader_index_ = inst.tester.initial_leader;
    status_ = pancy::status::active;
    inst.clients.connect_replica(index_, from_clients_, to_clients_);
    inst.tester.configure_port(from_clients_);
    inst.tester.configure_port(from_replicas_);
    inst.tester.configure_channel(to_clients_);
    inst.tester.configure_quiet_channel(inst.clients.request_channel(index_));
    for (size_t s = 0UL; s != nreplicas_; ++s) {
        // might adjust this dep on where you want stuff sent
        to_replicas_[s]->connect(inst.replicas[s]->from_replicas_);
        inst.tester.configure_channel(*to_replicas_[s]);
    }
}

size_t pt_paxos_replica::primary_for_view(pancy::view_id_t view_id) const {
    return (initial_leader_ + static_cast<size_t>(view_id)) % nreplicas_;
}

void pt_paxos_replica::apply_commits_through(pancy::commit_num_t target_commit_num) {
    auto bounded_target = std::min(target_commit_num, static_cast<pancy::commit_num_t>(log_.size()));
    while (commit_num_ < bounded_target) {
        auto& entry = log_[commit_num_];
        size_t client_id = message_serial(entry.request) & 4095;
        client_table_[client_id].op_num = entry.op_num;
        client_table_[client_id].serial = message_serial(entry.request);
        client_table_[client_id].response = db_.process_req(entry.request);
        ++commit_num_;
    }
}

void pt_paxos_replica::rebuild_state_from_log(const std::vector<pancy::log_entry>& log,
                                              pancy::commit_num_t target_commit_num) {
    log_ = log;
    op_num_ = static_cast<pancy::op_num_t>(log_.size());
    commit_num_ = 0;
    pending_quorums_.clear();
    for (auto& entry : client_table_) {
        entry = {};
    }
    for (const auto& entry : log_) {
        size_t client_id = message_serial(entry.request) & 4095;
        client_table_[client_id].op_num = entry.op_num;
        client_table_[client_id].serial = message_serial(entry.request);
        client_table_[client_id].response = std::nullopt;
    }
    std::destroy_at(&db_);
    std::construct_at(&db_);
    apply_commits_through(target_commit_num);
    // keeeping a vector of dbs
    // db_vector.append(db_);
}

void pt_paxos_replica::start_view_change_if_needed(pancy::view_id_t new_view_id) {
    if (new_view_id < view_id_) {
        return;
    }
    auto& waiter = pending_start_view_changes_[new_view_id];
    if (!waiter.started) {
        waiter.started = true;
        phase_one_view_change(new_view_id).detach();
    }
}

void pt_paxos_replica::consider_do_view_change_state(do_view_change_waiter& waiter,
                                                     pancy::commit_num_t commit_num,
                                                     const std::vector<pancy::log_entry>& log,
                                                     pancy::view_id_t last_normal_view) {
    bool better = false;
    if (last_normal_view > waiter.best_last_normal_view) {
        better = true;
    } else if (last_normal_view == waiter.best_last_normal_view) {
        if (log.size() > waiter.best_log.size()) {
            better = true;
        } else if (log.size() == waiter.best_log.size()
                   && commit_num > waiter.best_commit_num) {
            better = true;
        }
    }

    if (better) {
        waiter.best_last_normal_view = last_normal_view;
        waiter.best_commit_num = commit_num;
        waiter.best_log = log;
    }
}

pt_paxos_instance::pt_paxos_instance(testinfo& tester, client_model& clients)
    : tester(tester), clients(clients), replicas(tester.nreplicas) {
    for (size_t s = 0UL; s != tester.nreplicas; ++s) {
        replicas[s].reset(new pt_paxos_replica(s, tester.nreplicas, tester.randomness));
    }
    for (size_t s = 0UL; s != tester.nreplicas; ++s) {
        replicas[s]->initialize(*this);
    }
}



// ********** PANCY SERVICE CODE **********

// where are we actually counting prepare_oks that come in? // but we will have more and more prepare_oks come, why don't we j process
cot::task<> pt_paxos_replica::wait_for_quorum(pancy::op_num_t op_num) {
    // create on first call
    if (pending_quorums_.find(op_num) == pending_quorums_.end()) {
        pending_quorums_[op_num] = quorum_waiter();
    }
    quorum_waiter& waiter = pending_quorums_[op_num];
    while ((size_t) waiter.ack_count < nreplicas_ / 2 + 1) {
        co_await waiter.new_ack.arm();
    }
    // now we have a quorum, we shouldn't allow this task to be deleted as we process
    co_return;
}

cot::task<> pt_paxos_replica::wait_for_recovery_quorum(const std::string& nonce) {
    // add wainting messages to the quorum iff nonce matches
    if (pending_recovery_quorums_.find(nonce) == pending_recovery_quorums_.end()) {
        pending_recovery_quorums_[nonce] = recovery_waiter();
    }
    auto& waiter = pending_recovery_quorums_[nonce];
    while ((size_t) waiter.ack_count < nreplicas_ / 2 + 1
           || !waiter.have_primary_reply_for_max_view) {
        co_await waiter.new_ack.arm();
    }
    if (status_ != pancy::status::recovering
        || !active_recovery_nonce_
        || *active_recovery_nonce_ != nonce) {
        co_return;
    }
    // now we have a quorum
    if (waiter.best_view_id >= 0) {
        view_id_ = waiter.best_view_id;
    } else if (!log_.empty()) {
        view_id_ = log_.back().view_id;
    }
    leader_index_ = primary_for_view(view_id_);
    rebuild_state_from_log(waiter.best_log, waiter.best_commit_num);
    status_ = pancy::status::active;
    last_good_view_id_ = view_id_;
    active_recovery_nonce_.reset();
    co_return;
}

cot::task<> pt_paxos_replica::run_client_interaction() {
    // Your code here! The handout code just implements a single primary.
    while (true) {
        auto potential_req = co_await cot::attempt(
                from_clients_.receive(),
                cot::after(from_client_commit_timeout)
            );
        if (!potential_req) {
            // can we handle this we a couple of retries on the message before?
            if (status_ == pancy::status::active && index_ == leader_index_) {
                for (size_t s = 0; s < nreplicas_; ++s) {
                    // don't send to self
                    if (s == index_) {
                        continue; // don't send to self
                    }
                    // send effectively a commit message to all reps
                    co_await to_replicas_[s]->send(paxos_message(commit_num_, view_id_, pancy::vr_msg_type::commit));
                }
            }
            continue;
        }
        const auto& req = *potential_req;
        
        // if not leader, redirect - (that is the error that is specified)
        if (status_ != pancy::status::active || index_ != leader_index_) {
            // std::print("index {}, leader index {}\n", index_, leader_index_);
            co_await to_clients_.send(pancy::redirection_response{
                pancy::response_header(req, pancy::errc::redirect), leader_index_
            });
            continue;
        }

        uint64_t serial = pancy::message_serial(req);
        size_t client_id = serial & 4095; // same mask idea used by client_model

        if (client_table_[client_id].serial == serial) {
            // if we've already processed this request, resend the response
            if (client_table_[client_id].response) {
                // construct the response properly. Does response constructor feed into the response of it's variants.
                // printf("gave resp\n"); // never does this trigger
                co_await to_clients_.send(*client_table_[client_id].response);
            } else {
                // if we haven't processed this request, but it's a duplicate, we can ignore it
                // we might be sending prepares going backwards?
            }
        } else {
            // if this is a new request, we need to process it and send the response 
            op_num_++;
            int request_op_num = op_num_;
            log_.push_back({req, op_num_, view_id_});
            client_table_[client_id] = {op_num_, serial, std::nullopt}; // mark as in process
            // send prepare to backups
            for (size_t s = 0; s < nreplicas_; ++s) {
                // don't send to self
                if (s == index_) {
                    continue; // don't send to self
                }
                co_await to_replicas_[s]->send(paxos_message(req, op_num_, commit_num_, view_id_, log_));
            }
            // wait for prepare_ok from majority of backups, if we get prepare_ok from majority, we can commit 
            auto result = co_await cot::attempt(
                wait_for_quorum(op_num_),
                cot::after(prepareok_timeout) // what if the timeout happens after quorum before process? is okay bc is "committed"
            );
            if (!result) {
                pending_quorums_.erase(request_op_num);
                if (client_table_[client_id].serial == serial
                    && !client_table_[client_id].response) {
                    client_table_[client_id] = {};
                }
                start_view_change_if_needed(view_id_ + 1);
                continue;
            }
            // if we don't get prepare_ok from majority, we need to retry or do view change
            // for now we just wait for prepare_ok from all backups
            
            pending_quorums_.erase(request_op_num); // clean up quorum waiter
            apply_commits_through(request_op_num);
            auto response = *client_table_[client_id].response;
            for (size_t s = 0; s < nreplicas_; ++s) {
                // don't send to self
                if (s == index_) {
                    continue; // don't send to self
                }
                co_await to_replicas_[s]->send(paxos_message(commit_num_, view_id_, pancy::vr_msg_type::commit));
            }
            // std::print("sent response to clients");
            co_await to_clients_.send(response);
            
            continue;
        }
    }
}

cot::task<> pt_paxos_replica::run_inter_replica_interaction() {
    while (true) {
        // receive message
        //auto potential_req = co_await from_replicas_.receive();
        auto timeout = status_ == pancy::status::viewchange
            ? view_change_timeout
            : prepare_timeout;
        auto potential_req = co_await cot::attempt(
                from_replicas_.receive(),
                cot::after(timeout)
            );
        if (!potential_req) {
            if (status_ != pancy::status::recovering) {
                start_view_change_if_needed(view_id_ + 1);
            }
            continue;
        }
        const auto& req = *potential_req;
        
        // decode what kind of msg it is
        pancy::vr_msg_type msg_type = req.type;
        switch (msg_type) {
            // in own mind
            case pancy::vr_msg_type::prepare: {
                if (status_ != pancy::status::active) break;
                if (req.view_id < view_id_) {
                    break;
                }
                if (req.view_id > view_id_) {
                    start_view_change_if_needed(req.view_id);
                    break;
                }
                if (req.op_num == op_num_ + 1) {
                    log_.push_back({req.op, req.op_num, req.view_id});
                    op_num_++;
                    co_await to_replicas_[leader_index_]->send(paxos_message(req.op_num, req.view_id));
                    if (log_.size() < req.log_.size()) {
                        log_ = req.log_;
                        op_num_ = static_cast<pancy::op_num_t>(log_.size());
                    }
                    apply_commits_through(req.commit_num);
                } else if (req.op_num <= op_num_) {
                    if (log_.size() < req.log_.size()) {
                        log_ = req.log_;
                        op_num_ = static_cast<pancy::op_num_t>(log_.size());
                    }
                    apply_commits_through(req.commit_num);
                } else {
                    if (!active_recovery_nonce_) {
                        std::string nonce = randomness_.uniform_hex(64);
                        status_ = pancy::status::recovering;
                        active_recovery_nonce_ = nonce;
                        recover(nonce).detach();
                    }
                }
                break;
            }
            case pancy::vr_msg_type::prepare_ok:
                if (status_ != pancy::status::active) break;
                if (index_ == leader_index_ && req.view_id == view_id_) {
                    auto it = pending_quorums_.find(req.op_num);
                    if (it != pending_quorums_.end()) {
                        it->second.ack_count++;
                        it->second.new_ack.trigger();
                    }
                }
                break;
            case pancy::vr_msg_type::commit:
                if (status_ != pancy::status::active) break;
                if (req.view_id < view_id_) {
                    break;
                }
                if (req.view_id > view_id_) {
                    start_view_change_if_needed(req.view_id);
                    break;
                }
                if (req.commit_num > static_cast<pancy::commit_num_t>(log_.size()) && !active_recovery_nonce_) {
                    std::string nonce = randomness_.uniform_hex(64);
                    status_ = pancy::status::recovering;
                    active_recovery_nonce_ = nonce;
                    recover(nonce).detach();
                    break;
                }
                apply_commits_through(req.commit_num);
                break;
            // handled by sending us to execute on the first receive for a given view then rest handled updating waiters here
            case pancy::vr_msg_type::start_view_change: {
                if (req.view_id < view_id_) {
                    break;
                }
                if (status_ == pancy::status::recovering) {
                    active_recovery_nonce_.reset();
                }

                // when does this trigger? first is hash node second is bool value?
                if (pending_start_view_changes_[req.view_id].senders.insert(req.index).second) {
                    pending_start_view_changes_[req.view_id].new_msg.trigger();
                }

                auto& start_waiter = pending_start_view_changes_[req.view_id];
                if (!start_waiter.started) {
                    start_waiter.started = true;
                    phase_one_view_change(req.view_id).detach();
                }
                break;
            }
            case pancy::vr_msg_type::do_view_change:
                if (req.view_id < view_id_) {
                    break;
                }
                if (status_ == pancy::status::recovering) {
                    active_recovery_nonce_.reset();
                }
                if (index_ == primary_for_view(req.view_id)) {
                    auto& waiter = pending_do_view_changes_[req.view_id];
                    if (waiter.senders.insert(req.index).second) {
                        consider_do_view_change_state(waiter, req.commit_num, req.log_, req.last_normal_view);
                        waiter.new_msg.trigger();
                    }
                    if (!waiter.started) {
                        waiter.started = true;
                        phase_two_view_change(req.view_id).detach();
                    }
                }
                break;
            // keep even with execute
            case pancy::vr_msg_type::start_view:
                if (req.view_id < view_id_) {
                    break;
                }
                active_recovery_nonce_.reset();
                view_id_ = req.view_id;
                leader_index_ = primary_for_view(view_id_);
                status_ = pancy::status::active;
                last_good_view_id_ = view_id_;
                rebuild_state_from_log(req.log_, req.commit_num);
                break;
            case pancy::vr_msg_type::recovery:
                if (status_ == pancy::status::recovering) break;
                // when receive a recovery message, we construct a recovery resp
                co_await to_replicas_[req.index]->send(paxos_message(view_id_, commit_num_, index_, req.nonce, log_));
                break;
            case pancy::vr_msg_type::recover_ok:
                if (pending_recovery_quorums_.find(req.nonce) != pending_recovery_quorums_.end()) {
                    auto& waiter = pending_recovery_quorums_[req.nonce];
                    if (waiter.senders.insert(req.index).second) {
                        if (req.view_id > waiter.max_view_seen) {
                            waiter.max_view_seen = req.view_id;
                            waiter.have_primary_reply_for_max_view = false;
                            waiter.best_view_id = -1;
                            waiter.best_commit_num = 0;
                            waiter.best_log.clear();
                        }

                        if (req.view_id == waiter.max_view_seen
                            && req.index == primary_for_view(req.view_id)) {
                            waiter.have_primary_reply_for_max_view = true;
                            waiter.best_view_id = req.view_id;
                            waiter.best_commit_num = req.commit_num;
                            waiter.best_log = req.log_;
                        }
                        waiter.ack_count = static_cast<int>(waiter.senders.size());
                        waiter.new_ack.trigger();
                    }
                }
                break;
        }
    }
}

// this function is fed by the waiters
cot::task<> pt_paxos_replica::phase_one_view_change(pancy::view_id_t new_view_id) {
    if (new_view_id < view_id_) {
        co_return;
    }
    active_recovery_nonce_.reset();
    pending_quorums_.clear();
    status_ = pancy::status::viewchange;
    view_id_ = new_view_id;
    leader_index_ = primary_for_view(new_view_id);

    if (pending_start_view_changes_.find(new_view_id) == pending_start_view_changes_.end()) {
        pending_start_view_changes_[new_view_id] = start_view_change_waiter();
    }
    start_view_change_waiter& waiter = pending_start_view_changes_[new_view_id];
    
    waiter.senders.insert(index_);
    if (!waiter.broadcasted) {
        waiter.broadcasted = true;
        for (size_t i = 0; i < nreplicas_; ++i) {
            if (i == index_) {
                continue;
            }
            co_await to_replicas_[i]->send(paxos_message(view_id_, index_));
        }
    }

    while ((size_t) waiter.senders.size() < nreplicas_ / 2 + 1) {
        co_await waiter.new_msg.arm();
    }

    if (!waiter.sent_do_view_change) {
        waiter.sent_do_view_change = true;
        if (index_ == primary_for_view(new_view_id)) {
            auto& do_waiter = pending_do_view_changes_[new_view_id];
            // if inserting a new sender
            if (do_waiter.senders.insert(index_).second) {
                consider_do_view_change_state(do_waiter, commit_num_, log_, last_good_view_id_);
                do_waiter.new_msg.trigger();
            }
            if (!do_waiter.started) {
                do_waiter.started = true;
                phase_two_view_change(new_view_id).detach();
            }
        } else {
            // send a do_view_change
            co_await to_replicas_[primary_for_view(view_id_)]->send(
                paxos_message(view_id_, commit_num_, log_, index_, last_good_view_id_)
            );
        }
    }
    co_return;
}

cot::task<> pt_paxos_replica::phase_two_view_change(pancy::view_id_t new_view_id) {
    if (new_view_id < view_id_ || index_ != primary_for_view(new_view_id)) {
        co_return;
    }
    status_ = pancy::status::viewchange;
    view_id_ = new_view_id;
    leader_index_ = primary_for_view(new_view_id);
    if (pending_do_view_changes_.find(new_view_id) == pending_do_view_changes_.end()) {
        pending_do_view_changes_[new_view_id] = do_view_change_waiter();
    }
    do_view_change_waiter& waiter = pending_do_view_changes_[new_view_id];
    while ((size_t) waiter.senders.size() < nreplicas_ / 2 + 1) {
        co_await waiter.new_msg.arm();
    }
    if (waiter.sent_start_view) {
        co_return;
    }
    rebuild_state_from_log(waiter.best_log, waiter.best_commit_num);
    status_ = pancy::status::active;
    last_good_view_id_ = view_id_;
    waiter.sent_start_view = true;
    for (size_t i = 0; i < nreplicas_; ++i) {
        if (i == index_) {
            continue;
        }
        // send a start view
        co_await to_replicas_[i]->send(paxos_message(view_id_, commit_num_, log_));
    }       
    co_return;
}

cot::task<> pt_paxos_replica::recover(std::string nonce) {
    // send out recovery message to all other replicas, wait for response from one of them, and then update our state based on their log
    for (size_t s = 0; s < nreplicas_; ++s) {
        // don't send to self
        if (s == index_) {
            continue; // don't send to self
        }
        co_await to_replicas_[s]->send(paxos_message(index_, nonce, pancy::vr_msg_type::recovery));
    }

    // will return the best log. Then, we set op_num and commit num accordingly. 
    auto proceed = co_await cot::attempt(wait_for_recovery_quorum(nonce), cot::after(5s)); // need quorum of responses (including leader) - TODO, adjust timeout
    if (!proceed) {
        pending_recovery_quorums_.erase(nonce);
        if (status_ == pancy::status::recovering
            && active_recovery_nonce_
            && *active_recovery_nonce_ == nonce) {
            active_recovery_nonce_.reset();
            start_view_change_if_needed(view_id_ + 1);
        }
        co_return;
    }
    // DO NOT CHANGE Commit number initially. This will change automatically.
    // new log, view_id, op_num are sent in the wait if successful
    // pending recovery quorum waiter delete
    // does this separate old vs new?
    if (active_recovery_nonce_ && *active_recovery_nonce_ == nonce) {
        active_recovery_nonce_.reset();
    }
    pending_recovery_quorums_.erase(nonce);
    co_return;
} 
// ******** end Pancy service code ********



// Test functions

cot::task<> clear_after(cot::duration d) {
    co_await cot::after(d);
    cot::clear();
}

cot::task<> apply_link_loss_window(pt_paxos_instance& inst,
                                   const testinfo::scheduled_link_loss& link_loss) {
    co_await cot::after(link_loss.start);

    auto& chan = *inst.replicas[link_loss.src]->to_replicas_[link_loss.dst];
    if (inst.tester.verbose) {
        std::print("{}: link R{} -> R{}/r loss set to {}\n",
                   cot::now(), link_loss.src, link_loss.dst, link_loss.loss);
    }
    chan.set_loss(link_loss.loss);

    co_await cot::after(link_loss.duration);

    chan.set_loss(inst.tester.loss);
    if (inst.tester.verbose) {
        std::print("{}: link R{} -> R{}/r loss restored to {}\n",
                   cot::now(), link_loss.src, link_loss.dst, inst.tester.loss);
    }
}

// Second gen of failure model
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


/*
Annie Code failure
*/
void kill_replica(pt_paxos_instance& inst, size_t r) {
    inst.failed_replicas.insert(r);
    for (size_t i = 0; i < inst.replicas.size(); ++i) {
        if (i == r) continue;
        inst.replicas[r]->to_replicas_[i]->set_loss(1.0);
        inst.replicas[i]->to_replicas_[r]->set_loss(1.0);
    }
    std::print("como estas");
    inst.replicas[r]->to_clients_.set_loss(1.0);
    inst.clients.request_channel(r).set_loss(1.0);
}

void restore_replica(pt_paxos_instance& inst, size_t r, double loss) {
    inst.failed_replicas.erase(r);
    for (size_t i = 0; i < inst.replicas.size(); ++i) {
        if (i == r) continue;
        inst.replicas[r]->to_replicas_[i]->set_loss(loss);
        inst.replicas[i]->to_replicas_[r]->set_loss(loss);
        
    }
    std::print("hello {}\n", loss);
    inst.replicas[r]->to_clients_.set_loss(loss);
    inst.clients.request_channel(r).set_loss(loss);
}

cot::task<> fail_random(pt_paxos_instance& inst, double loss) {
    size_t n = inst.replicas.size();
    size_t max_failed = n / 2;  // never kill a majority
    std::print("running fail random with loss: {}\n", loss);

    co_await cot::after(inst.tester.randomness.exponential(10s));
    while (true) {
        // Kill a random non-failed replica (if under limit)
        if (inst.failed_replicas.size() < max_failed) {
            size_t r = inst.tester.randomness.uniform(size_t(0), n - 1);
            if (!inst.failed_replicas.count(r)) {
                std::print("killing replica {}\n", r);
                kill_replica(inst, r);
            }
        }
        co_await cot::after(inst.tester.randomness.exponential(15s));

        // Recover a random failed replica
        if (!inst.failed_replicas.empty()) {
            auto it = inst.failed_replicas.begin();
            std::advance(it, inst.tester.randomness.uniform(
                size_t(0), inst.failed_replicas.size() - 1));
            restore_replica(inst, *it, loss);
        }
        co_await cot::after(inst.tester.randomness.exponential(10s));
    }
}
/*END annie failure*/


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

bool try_one_seed(testinfo& tester, unsigned long seed) {
    cot::reset();   // clear old events and coroutines
    tester.randomness.seed(seed);

    // Create client generator and test instance
    lockseq_model clients(tester.nreplicas, tester.randomness);
    pt_paxos_instance inst(tester, clients); // insteance constructed as we desired

    // Start coroutines
    clients.start();
    std::vector<cot::task<>> tasks;

    //initialize_failure_model(inst, true, 100s);
    fail_random(inst, tester.loss);
    for (size_t s = 0UL; s != tester.nreplicas; ++s) {
        tasks.push_back(inst.replicas[s]->run_client_interaction());
        tasks.push_back(inst.replicas[s]->run_inter_replica_interaction());
    }
    for (const auto& link_loss : tester.link_losses) {
        tasks.push_back(apply_link_loss_window(inst, link_loss));
    }
    cot::task<> timeout_task = clear_after(100s);

    // Wait for `timeout_task`
    cot::loop();

    // Check database
    if (tester.verbose) {
        std::print("{} lock, {} write, {} clear, {} unlock\n",
                clients.lock_complete, clients.write_complete,
                clients.clear_complete, clients.unlock_complete);
    }
    // so we are checking just against the initial leader db. Check checks structure
    pancy::pancydb& db = inst.replicas[tester.initial_leader]->db_;
    for (int i = 0; (size_t) i < tester.nreplicas; i++) {
        if ((size_t) i == tester.initial_leader) {
            // continue; // skip leader, we are checking against it
        }
        // we are more than 1000 off, something is wrong
        if (auto diff_key = inst.replicas[tester.initial_leader]->db_.diff(inst.replicas[i]->db_, 10)) {
            auto leader_vv = inst.replicas[tester.initial_leader]->db_.get(*diff_key);
            auto replica_vv = inst.replicas[i]->db_.get(*diff_key);
            std::print(std::clog,
                       "*** FAILURE on seed {}: replica {} has different db at key {} (leader view={}, commit_num={}, replica view={}, commit_num={})\n",
                       seed, i, *diff_key,
                       inst.replicas[tester.initial_leader]->view_id_,
                       inst.replicas[tester.initial_leader]->commit_num_,
                       inst.replicas[i]->view_id_,
                       inst.replicas[i]->commit_num_);
            if (leader_vv) {
                std::print(std::clog, " leader value={} version={}\n", leader_vv->value, leader_vv->version);
            } else {
                std::print(std::clog, " leader value=<missing>\n");
            }
            if (replica_vv) {
                std::print(std::clog, " replica value={} version={}\n", replica_vv->value, replica_vv->version);
            } else {
                std::print(std::clog, " replica value=<missing>\n");
            }
            auto& leader_log = inst.replicas[tester.initial_leader]->log_;
            auto& replica_log = inst.replicas[i]->log_;
            size_t mismatch = 0;
            while (mismatch < leader_log.size()
                   && mismatch < replica_log.size()
                   && leader_log[mismatch].op_num == replica_log[mismatch].op_num
                   && leader_log[mismatch].view_id == replica_log[mismatch].view_id
                   && std::format("{}", leader_log[mismatch].request)
                       == std::format("{}", replica_log[mismatch].request)) {
                ++mismatch;
            }
            if (mismatch < leader_log.size() || mismatch < replica_log.size()) {
                std::print(std::clog,
                           "first log mismatch at index {} (leader log size={}, replica log size={})\n",
                           mismatch, leader_log.size(), replica_log.size());
                if (mismatch < leader_log.size()) {
                    std::print(std::clog, " leader: op_num={} view={} req={}\n",
                               leader_log[mismatch].op_num,
                               leader_log[mismatch].view_id,
                               leader_log[mismatch].request);
                }
                if (mismatch < replica_log.size()) {
                    std::print(std::clog, " replica: op_num={} view={} req={}\n",
                               replica_log[mismatch].op_num,
                               replica_log[mismatch].view_id,
                               replica_log[mismatch].request);
                }
            }
            return false;
        }
    }
    if (auto problem = clients.check(db)) {
        std::print(std::clog, "*** FAILURE on seed {} at key {}\n", seed, *problem);
        db.print_near(*problem, std::clog);
        return false;
    } else if (tester.print_db) {
        db.print(std::cout);
    }
    return true;
}


// Argument parsing

static struct option options[] = {
    { "count", required_argument, nullptr, 'n' },
    { "seed", required_argument, nullptr, 'S' },
    { "random-seeds", required_argument, nullptr, 'R' },
    { "loss", required_argument, nullptr, 'l' },
    { "link-loss", required_argument, nullptr, 'L' },
    { "verbose", no_argument, nullptr, 'V' },
    { "print-db", no_argument, nullptr, 'p' },
    { "quiet", no_argument, nullptr, 'q' },
    { nullptr, 0, nullptr, 0 }
};

int main(int argc, char* argv[]) {
    testinfo tester;

    std::optional<unsigned long> first_seed;
    unsigned long seed_count = 1;

    auto shortopts = short_options_for(options);
    int ch;
    while ((ch = getopt_long(argc, argv, shortopts.c_str(), options, nullptr)) != -1) {
        if (ch == 'S') {
            first_seed = from_str_chars<unsigned long>(optarg);
        } else if (ch == 'R') {
            seed_count = from_str_chars<unsigned long>(optarg);
        } else if (ch == 'l') {
            tester.loss = from_str_chars<double>(optarg);
        } else if (ch == 'L') {
            try {
                tester.link_losses.push_back(parse_link_loss_spec(optarg));
            } catch (const std::exception& e) {
                std::print(std::cerr, "Invalid --link-loss spec '{}': {}\n",
                           optarg, e.what());
                return 1;
            }
        } else if (ch == 'n') {
            tester.nreplicas = from_str_chars<size_t>(optarg);
        } else if (ch == 'V') {
            tester.verbose = true;
        } else if (ch == 'p') {
            tester.print_db = true;
        } else {
            std::print(std::cerr, "Unknown option\n");
            return 1;
        }
    }

    try {
        validate_link_losses(tester);
    } catch (const std::exception& e) {
        std::print(std::cerr, "Invalid link loss configuration: {}\n", e.what());
        return 1;
    }

    bool ok;
    if (first_seed) {
        ok = try_one_seed(tester, *first_seed);
    } else {
        std::mt19937_64 seed_generator = randomly_seeded<std::mt19937_64>();
        for (unsigned long i = 0; i != seed_count; ++i) {
            if (i > 0 && i % 1000 == 0) {
                std::print(std::cerr, ".");
            }
            unsigned long seed = seed_generator();
            ok = try_one_seed(tester, seed);
            if (!ok) {
                break;
            }
        }
        if (ok && seed_count >= 1000) {
            std::print(std::cerr, "\n");
        }
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
