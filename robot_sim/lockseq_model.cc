#include "lockseq_model.hh"
#include <cassert>
#include <format>
#include <string>

// lockseq_model.cc
//
//    The most important code here is the `lockseq_model::client()` coroutine
//    that defines client behavior.

namespace cot = cotamer;
using namespace std::chrono_literals;

lockseq_model::lockseq_model(size_t nreplicas, random_source& randomness)
    : client_model(nreplicas, randomness) {
}

void lockseq_model::start() {
    assert(clients_.empty());
    set_nclients(nclients_);
    clients_.reserve(nclients_);
    while (clients_.size() < nclients_) {
        cstate* cs = new cstate;
        clients_.emplace_back(cs);
        cs->leader = random_replica();
        cs->task = client(clients_.size() - 1);
    }
}

void lockseq_model::stop() {
    client_model::stop();
    clients_.clear();
}



// lockseq_model::client
//    Each distinct simulated `lockseq_model` client runs a copy of the
//    client() coroutine.

cot::task<> lockseq_model::client(unsigned cid) {
    cstate& cs = *clients_[cid];
    uint64_t serial = cid;
    std::string client_key = make_client_key(cid);
    std::string client_key_prefix = client_key + " ";

    while (true) {
        // delay
        co_await cot::after(randomness().normal(1s, 100ms));

        // pick a random group to modify
        unsigned gid = randomness().uniform(0U, ngroups_ - 1);

        // compute the lock key, a value to install, and our sequence length
        std::string lock_key = make_lock_key(gid);
        std::string value = client_key_prefix + randomness().uniform_hex(7);
        unsigned nvalues = randomness().uniform(1U, max_sequence_length_);
        pancy::version_type lock_version = 0;

        // PHASE 1 (lock): Obtain lock with compare-and-swap loop
        serial += serial_step();
        for (unsigned tries = 0; true; ++tries) {
            co_await send_request<pancy::cas_request>(
                cs.leader, serial, lock_key, "", value
            );
            auto resp = co_await cot::attempt(
                receive_response<pancy::cas_response>(cs.leader, serial),
                cot::after(randomness().normal(3s, 1s))
            );
            if (!resp) { // timeout; try a new leader every 3 retries
                cs.leader = tries % 3 == 2 ? random_replica() : cs.leader;
                continue;
            } else if (resp->errcode == pancy::errc::ok) {
                // CAS success: we obtained the lock
                lock_version = resp->version;
                goto write_phase;
            }
            // otherwise CAS failed
            if (resp->errcode != pancy::errc::no_match) {
                throw resp->errcode;
            }
            if (resp->actual == value && tries > 0) {
                // an earlier CAS succeeded, but the reply was lost
                lock_version = resp->version;
                goto write_phase;
            }
            // must not be waiting for ourselves
            if (!resp->actual.starts_with(client_key_prefix)) {
                throw pancy::errc::no_match;
            }
            // delay then retry
            co_await cot::after(randomness().uniform(50ms, 200ms));
        }

        // PHASE 2 (write): Write value sequence -- how can we go to write phase and still b trying diff leaders?
    write_phase:
        ++lock_complete;
        unsigned n = 0;
        while (n != nvalues) {
            std::string value_key = make_value_key(gid, n);
            serial += serial_step();
            for (unsigned tries = 0; true; ++tries) {
                co_await send_request<pancy::put_request>(
                    cs.leader, serial, value_key, value
                );
                auto resp = co_await cot::attempt(
                    receive_response<pancy::put_response>(cs.leader, serial),
                    cot::after(randomness().normal(3s, 1s))
                );
                if (!resp) { // timeout; try a new leader every 3 retries
                    cs.leader = tries % 3 == 2 ? random_replica() : cs.leader;
                    continue;
                }

                // must succeed
                if (resp->errcode != pancy::errc::ok) {
                    throw resp->errcode;
                }
                break;
            }
            ++n;
            co_await cot::after(1ms);
        }

        // PHASE 3 (clear): Remove vestige of previous sequence, if any
    __attribute__((unused))
    clear_phase:
        ++write_complete;
        while (n < max_sequence_length_) {
            std::string value_key = make_value_key(gid, n);
            serial += serial_step();
            for (unsigned tries = 0; true; ++tries) {
                co_await send_request<pancy::remove_request>(
                    cs.leader, serial, value_key
                );
                auto resp = co_await cot::attempt(
                    receive_response<pancy::remove_response>(cs.leader, serial),
                    cot::after(randomness().normal(3s, 1s))
                );
                if (!resp) { // timeout; try a new leader every 3 retries
                    cs.leader = tries % 3 == 2 ? random_replica() : cs.leader;
                    continue;
                } else if (resp->errcode == pancy::errc::ok) {
                    // successful remove; go to next `n`
                    break;
                }
                // otherwise, there was nothing there
                if (resp->errcode != pancy::errc::not_found) {
                    throw resp->errcode;
                }
                // if this is our first try and there was nothing there,
                // then we must have reached the end of the previous client’s
                // values
                if (tries == 0) {
                    goto unlock_phase;
                }
                // if this is NOT our first try, then our previous attempt may
                // have succeeded with a lost reply. Have to try next `n`
                break;
            }
            ++n;
            co_await cot::after(1ms);
        }

        // PHASE 4 (unlock): Release lock with compare-and-swap loop
        // (CAS is required because replies might be dropped.)
    unlock_phase:
        ++clear_complete;
        serial += serial_step();
        for (unsigned tries = 0; true; ++tries) {
            co_await send_request<pancy::remove_request>(
                cs.leader, serial, lock_key, lock_version
            );
            auto resp = co_await cot::attempt(
                receive_response<pancy::remove_response>(cs.leader, serial),
                cot::after(randomness().normal(3s, 1s))
            );
            if (!resp) { // timeout; try a new leader every 3 retries
                cs.leader = tries % 3 == 2 ? random_replica() : cs.leader;
                continue;
            }

            // Any response indicates success!
            // - pancy::errc::ok: Successfully unlocked
            // - pancy::errc::no_match: A previous attempt successfully
            //   unlocked, but the reply was lost
            // But if this is the first try, only ok is allowed.
            if (resp->errcode != pancy::errc::ok && tries == 0) {
                throw resp->errcode;
            }
            break;
        }

        ++unlock_complete;
    }
}



// lockseq_model::check
//
//    Check that a `pancydb` database could have been created by some sequence
//    of `lockseq_model` client operations.

static unsigned parse_digits(const std::string& s, size_t from, size_t to) {
    // Parse a decimal number from `s[from]` up to, but not including,
    // `s[to]`. Return unsigned(-1) on error.
    if (from >= to) {
        return unsigned(-1);
    }
    unsigned v = 0;
    while (from < to) {
        char ch = s[from];
        if (ch < '0' || ch > '9') {
            return unsigned(-1);
        }
        unsigned vv = v * 10 + ch - '0';
        if (vv < v) {
            return unsigned(-1);
        }
        v = vv;
        ++from;
    }
    return v;
}

std::optional<std::string> lockseq_model::check(const pancy::pancydb& db) {
    // Keys (in lexicographic order) should consist of a series of locked and
    // unlocked groups. Each group starts with an optional `g{G}/lock` key,
    // and is followed by a sequence of value keys `g{G}/v000`, `g{G}/v001`,
    // ... `g{G}/vXXX` (where XXX <= 999).

    // An *unlocked group* comprises, in order:
    // * Zero or more `g{G}/v{I}` keys, with no gaps in key space, all with
    //   the same value

    // A *locked group* comprises, in order:
    // * A `g{G}/lock` key whose value is a client id and random string
    //   `c{N} {HEXDIGITS}`
    // * Zero or more `g{G}/v{I}` keys, with no index gaps, all with the same
    //   value as the lock
    // * Zero or more missing `g{G}/v{I}` keys
    // * Zero or more `g{G}/v{I}` keys, with no index gaps, all with the same
    //   value

    std::string last_group;
    size_t glen;
    std::string last_lock;
    std::string expected_value;
    unsigned expected_index = 0;
    enum { ls_unlocked, ls_locked_start, ls_locked_rest } lock_state = ls_unlocked;

    for (auto it = db.begin(); it != db.end(); ++it) {
        const std::string& key = it->first;
        const std::string& value = it->second.value;

        // Check for new group
        if (last_group.empty() || !key.starts_with(last_group)) {
            // Must start with `g{G}`
            if (key.size() < 3 || key[0] != 'g') {
                return key;
            }
            auto slash = key.find('/');
            if (slash == std::string::npos) {
                return key;
            }
            unsigned gid = parse_digits(key, 1, slash);
            if (gid == unsigned(-1)) {
                return key;
            }
            last_group = key.substr(0, slash + 1);
            glen = slash + 1;
            expected_value = "";
            expected_index = 0;
            lock_state = ls_unlocked;
        }

        // Check for lock key
        if (key.size() == glen + 4
            && key.ends_with("lock")) {
            if (expected_index != 0 || lock_state != ls_unlocked) {
                return key;
            }
            if (value.length() == 0) {
                continue;
            }
            if (value.length() < 4
                || value[0] != 'c') {
                return key;
            }
            last_lock = value;
            lock_state = ls_locked_start;
            expected_value = value;
            continue;
        }

        // Otherwise, must be value key
        unsigned vindex;
        if (key.size() != glen + 4
            || key[glen] != 'v'
            || (vindex = parse_digits(key, glen + 1, glen + 4)) == unsigned(-1)
            || value.empty()) {
            return key;
        }

        // Check key
        if (vindex != expected_index) {
            if (lock_state == ls_unlocked
                || lock_state == ls_locked_rest
                || vindex < expected_index) {
                return key;
            }
            lock_state = ls_locked_rest;
            expected_index = vindex;
            expected_value = "";
        }

        // Check value
        if (expected_value != "" && value != expected_value) {
            if (lock_state != ls_locked_start) {
                return key;
            }
            lock_state = ls_locked_rest;
            expected_value = value;
        }
        expected_value = value;
        ++expected_index;
    }
    return std::nullopt;
}
