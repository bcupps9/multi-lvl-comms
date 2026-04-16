#pragma once
#include "cotamer/circular_int.hh"
#include "pancy_msgs.hh"
#include <deque>
#include <map>

// pancydb.hh
//    Defines `pancy::pancydb`, a simple database that can process Pancy
//    messages (see `pancy_msgs.hh`).

namespace pancy {

struct vv { // a “Versioned Value” stored for each key
    std::string value;
    version_type version;

    static vv nonexistent_vv;
};

class pancydb {
public:
    using map_type = std::map<std::string, vv>;
    using iterator = map_type::iterator;
    using const_iterator = map_type::const_iterator;

    pancydb() = default;
    pancydb(const pancydb&) = delete;
    pancydb(pancydb&&) = delete;
    pancydb& operator=(const pancydb&) = delete;
    pancydb& operator=(pancydb&&) = delete;

    // Direct API
    inline version_type version(const std::string& key) const;
    inline bool exists(const std::string& key) const;
    inline std::optional<vv> get(const std::string& key) const;
    inline void put(const std::string& key, const std::string& value);
    inline bool remove(const std::string& key);

    // Message API
    get_response process(const get_request&) const;
    put_response process(const put_request&);
    cas_response process(const cas_request&);
    remove_response process(const remove_request&);
    response process_req(const request&);

    // Iteration
    const_iterator begin() const { return db_.begin(); }
    const_iterator end() const { return db_.end(); }

    // Compare databases
    std::optional<std::string> diff(const pancydb&, size_t version_skew = 0) const;

    // Printing
    void print(std::ostream&);
    void print_near(const std::string& key, std::ostream&);

private:
    version_type next_version_ = 1;
    map_type db_;

    struct recent_remove {
        std::pair<std::string, vv> pair;
        version_type remove_version;
    };
    std::deque<recent_remove> recent_removes_;
    static constexpr size_t max_recent_removes = 100;

    inline vv* find_vvptr(const std::string& key);
    inline const vv* find_vvptr(const std::string& key) const;
    inline std::pair<vv*, bool> emplace_vvptr(const std::string& key, const std::string& value);
    inline void do_remove(iterator it);
    const vv* find_recent_remove(const_iterator desired, version_type min_version) const;

    // helpers to generate modification responses
    template <request_type T>
    static inline auto mod_no_match(const T&, const vv*) -> typename T::response_type;
    template <request_type T>
    static inline auto mod_ok(const T&, version_type, const vv*) -> typename T::response_type;
};


inline vv* pancydb::find_vvptr(const std::string& key) {
    auto it = db_.find(key);
    return it != db_.end() ? &it->second : &vv::nonexistent_vv;
}

inline const vv* pancydb::find_vvptr(const std::string& key) const {
    auto it = db_.find(key);
    return it != db_.end() ? &it->second : &vv::nonexistent_vv;
}

inline std::pair<vv*, bool> pancydb::emplace_vvptr(const std::string& key, const std::string& value) {
    auto [it, inserted] = db_.emplace(key, vv{value, next_version_});
    return {&it->second, inserted};
}

inline void pancydb::do_remove(iterator it) {
    recent_removes_.emplace_front(*it, next_version_);
    while (recent_removes_.size() > max_recent_removes) {
        recent_removes_.pop_back();
    }
    ++next_version_;
    db_.erase(it);
}


inline version_type pancydb::version(const std::string& key) const {
    return find_vvptr(key)->version;
}

inline bool pancydb::exists(const std::string& key) const {
    return db_.contains(key);
}

inline std::optional<vv> pancydb::get(const std::string& key) const {
    auto vvp = find_vvptr(key);
    return vvp->version ? std::optional<vv>(*vvp) : std::nullopt;
}

inline void pancydb::put(const std::string& key, const std::string& value) {
    auto [vvp, inserted] = emplace_vvptr(key, value);
    if (!inserted) {
        vvp->value = value;
        vvp->version = next_version_;
    }
    ++next_version_;
}

inline bool pancydb::remove(const std::string& key) {
    auto it = db_.find(key);
    if (it == db_.end()) {
        return false;
    }
    do_remove(it);
    return true;
}

}
