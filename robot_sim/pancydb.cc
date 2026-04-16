#include "pancydb.hh"

namespace pancy {

vv vv::nonexistent_vv{"", nonexistent_version};

get_response pancydb::process(const get_request& req) const {
    auto vvp = find_vvptr(req.key);
    if (!vvp) {
        return {response_header(req, errc::not_found), "", nonexistent_version};
    }
    return {response_header(req), vvp->value, vvp->version};
}

template <request_type T>
inline auto pancydb::mod_no_match(const T& req, const vv* vvp) -> typename T::response_type {
    return {
        response_header(req, errc::no_match),
        vvp->value, vvp->version, vvp->version
    };
}

template <request_type T>
inline auto pancydb::mod_ok(const T& req, version_type previous_version, const vv* vvp) -> typename T::response_type {
    return {
        response_header(req),
        "", previous_version, vvp->version
    };
}

put_response pancydb::process(const put_request& req) {
    vv* vvp = find_vvptr(req.key);
    if (req.version_match >= 0 && req.version_match != vvp->version) {
        return mod_no_match(req, vvp);
    } else if (vvp->version == nonexistent_version) {
        std::tie(vvp, std::ignore) = emplace_vvptr(req.key, req.value);
        ++next_version_;
        return mod_ok(req, nonexistent_version, vvp);
    }
    auto previous_version = vvp->version;
    vvp->value = req.value;
    vvp->version = next_version_;
    ++next_version_;
    return mod_ok(req, previous_version, vvp);
}

cas_response pancydb::process(const cas_request& req) {
    vv* vvp = find_vvptr(req.key);
    if (req.expected != vvp->value
        || (req.version_match >= 0 && req.version_match != vvp->version)) {
        return mod_no_match(req, vvp);
    } else if (vvp->version == nonexistent_version) {
        std::tie(vvp, std::ignore) = emplace_vvptr(req.key, req.desired);
        ++next_version_;
        return mod_ok(req, nonexistent_version, vvp);
    }
    auto previous_version = vvp->version;
    vvp->value = req.desired;
    vvp->version = next_version_;
    ++next_version_;
    return mod_ok(req, previous_version, vvp);
}

remove_response pancydb::process(const remove_request& req) {
    auto it = db_.find(req.key);
    if (it == db_.end()) {
        return {response_header(req, errc::not_found), "", nonexistent_version, nonexistent_version};
    } else if (req.version_match >= 0 && req.version_match != it->second.version) {
        return mod_no_match(req, &it->second);
    }
    auto previous_version = it->second.version;
    do_remove(it);
    return mod_ok(req, previous_version, &vv::nonexistent_vv);
}

response pancydb::process_req(const request& req) {
    return std::visit([&](auto&& reqt) -> response {
        return process(reqt);
    }, req);
}


// pancydb::diff(db)
//
//    Checks whether this database and `db` are identical up to the given
//    `version_skew`. If `version_skew == 0`, the databases must exactly equal;
//    otherwise, the older database may be up to `version_skew` versions behind.
//    Returns `std::nullopt` if the databases are equivalent to within
//    `version_skew`, or a key at which the two databases differ.

const vv* pancydb::find_recent_remove(const_iterator desired,
                                      version_type min_version) const {
    for (auto& rr : recent_removes_) {
        if (rr.pair.first == desired->first
            && rr.remove_version >= min_version) {
            return &rr.pair.second;
        }
    }
    return nullptr;
}

std::optional<std::string> pancydb::diff(const pancydb& db, size_t version_skew) const {
    // ensure that `*this` is newer than `db`
    if (next_version_ < db.next_version_) {
        return db.diff(*this, version_skew);
    }
    auto ait = begin();
    auto bit = db.begin();
    version_type min_version = next_version_ - version_skew;
    // The recent_removes_ deque is complete for the skew window if it
    // hasn't reached capacity, or its oldest entry predates the window.
    bool a_removes_complete = recent_removes_.size() < max_recent_removes
        || recent_removes_.back().remove_version <= min_version;
    while (ait != end() || bit != db.end()) {
        // Which key is greater?
        std::strong_ordering keycmp = std::strong_ordering::equivalent;
        if (ait == end()) {
            keycmp = std::strong_ordering::greater;
        } else if (bit == db.end()) {
            keycmp = std::strong_ordering::less;
        } else {
            keycmp = ait->first <=> bit->first;
        }
        // Find versions
        const vv* avvp = keycmp <= 0
            ? &ait->second
            : find_recent_remove(bit, min_version);
        const vv* bvvp = keycmp >= 0
            ? &bit->second
            : db.find_recent_remove(ait, min_version);
        // Check version correspondence
        if (avvp && !bvvp && avvp->version >= min_version) {
            // key missing from `db`, but maybe added within version_skew: OK
        } else if (!avvp && !a_removes_complete) {
            // key missing from `*this`, but maybe evicted remove: OK
        } else if (!avvp
                   || !bvvp
                   || avvp->version < bvvp->version
                   || (avvp->version > bvvp->version
                       && avvp->version < min_version)
                   || (avvp->version == bvvp->version
                       && avvp->value != bvvp->value)) {
            // Error!
            return keycmp <= 0 ? ait->first : bit->first;
        }
        // Advance
        if (keycmp <= 0) {
            ++ait;
        }
        if (keycmp >= 0) {
            ++bit;
        }
    }
    return std::nullopt;
}


void pancydb::print(std::ostream& ios) {
    for (auto it = begin(); it != end(); ++it) {
        std::print(ios, "{} [V{}] {}\n", it->first, it->second.version, it->second.value);
    }
    std::print(ios, "\n");
}

void pancydb::print_near(const std::string& key, std::ostream& ios) {
    auto it = db_.lower_bound(key);
    for (int n = 0; n != 5 && it != db_.begin(); ++n) {
        --it;
    }
    if (it != db_.begin()) {
        std::print(ios, "   ...\n");
    }
    for (int n = 0; n != 11 && it != db_.end(); ++n, ++it) {
        std::print(ios, " {} {} [V{}] {}\n", it->first == key ? "*" : " ",
                   it->first, it->second.version, it->second.value);
    }
    if (it != db_.end()) {
        std::print(ios, "   ...\n");
    }
}

}
