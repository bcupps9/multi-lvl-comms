#pragma once
#include <cstddef>

// event_handle.hh
//    Declares the cotamer::detail::event_handle class, a smart pointer
//    specialized for Cotamer events. A full definition of this class is
//    required in order to declare the public cotamer::event class.

namespace cotamer {
class driver;
class fd;
namespace detail {
struct event_body;
struct fd_body;
struct quorum_event_body;
template <typename T> struct task_promise;
template <typename T> struct task_awaiter;
template <typename T> struct task_event_awaiter;
template <typename T, bool shared> struct task_mutex_event_awaiter;
template <typename T> struct task_final_awaiter;
struct interest_event_awaiter;

class event_handle {
public:
    event_handle() = default;
    explicit inline event_handle(event_body*) noexcept;
    inline event_handle(const event_handle&) noexcept;
    inline event_handle(event_handle&&) noexcept;
    inline event_handle& operator=(const event_handle&);
    inline event_handle& operator=(event_handle&&) noexcept;
    inline event_handle& operator=(std::nullptr_t);
    inline void swap(event_handle&) noexcept;
    inline ~event_handle();

    explicit operator bool() const noexcept { return eb_ != nullptr; }
    inline bool empty() const noexcept;
    inline bool idle() const noexcept;
    event_body* get() const noexcept { return eb_; }
    event_body& operator*() const { return *eb_; }
    event_body* operator->() const noexcept { return eb_; }

private:
    event_body* eb_ = nullptr;
};

inline void swap(event_handle& a, event_handle& b) noexcept { a.swap(b); }


struct fd_update {
    int fd;
    int mask;
    unsigned epoch;
};

class fd_event_set {
public:
    fd_event_set() = default;
    fd_event_set(const fd_event_set&) = delete;
    fd_event_set(fd_event_set&&) = delete;
    fd_event_set& operator=(const fd_event_set&) = delete;
    fd_event_set& operator=(fd_event_set&&) = delete;
    void deref_all(driver*);
    ~fd_event_set();

    static constexpr unsigned empty_epoch = 0;
    static constexpr unsigned internal_epoch = 1;
    static constexpr unsigned user_epoch = 2;

    inline event_handle watch(int fd, int type, fd_body* body, driver*);
    inline event_handle take(int fd, int type, unsigned epoch);
    inline std::optional<std::pair<fd_body*, unsigned>> check_fd_close(int fd);

    inline bool has_update() const noexcept;
    inline std::optional<fd_update> pop_update() noexcept;
    inline std::optional<fd_update> next_nonempty(int fd) const noexcept;

private:
    static constexpr unsigned first_capacity = 128;
    static constexpr unsigned block_capacity = 1024;

    // update_link_ values: singly-linked list of fdrecs with interest
    // potentially updated relative to the OS kernel event notifier. Link values
    // are fd+1 (so fd 0 maps to 1); update_clean means not in list,
    // update_sentinel means end of list or empty list head.
    static constexpr unsigned update_sentinel = -1;
    static constexpr unsigned update_clean = 0;

    struct fdrec {
        event_handle ev[3];         // 0: readable, 1: writable, 2: closed
        fd_body* body = nullptr;    // weak ref to owning fd_body
        unsigned update_link_ = update_clean;
        unsigned epoch = 0;

        inline int mask() const noexcept {
            return (ev[0] ? 1 : 0) | (ev[1] ? 2 : 0) | (ev[2] ? 4 : 0);
        }
    };

    fdrec* fdrs_ = nullptr;
    unsigned capacity_ = 0;
    unsigned update_link_ = -1;   // head of update list; see encoding above

    void hard_ensure(unsigned fd);
};

struct fd_batch;

}
}
