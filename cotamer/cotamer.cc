#include "cotamer/cotamer.hh"
#include <iterator>
#include <memory>
#include <fcntl.h>
#include <thread>
#include <unistd.h>

using namespace std::chrono_literals;

namespace cotamer {

#if COTAMER_STATS
statistics stats;
#endif


inline void driver::migrate_wake() {
#if !COTAMER_USE_POLL
    int wakefd = wakefd_.load(std::memory_order_seq_cst);
    if (wakefd >= 0) {
# if COTAMER_USE_KQUEUE
        struct kevent ev;
        EV_SET(&ev, -1, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
        kevent(wakefd, &ev, 1, nullptr, 0, nullptr);
# elif COTAMER_USE_EPOLL
        uint64_t val = 1;
        ssize_t nw = ::write(wakefd, &val, sizeof(val));
        (void) nw;
# endif
    }
#endif
}

void driver::migrate_asap(detail::event_handle eh) {
    auto df = lock();
    migrate_.emplace_back(std::move(eh));
    unlock(df | driver::df_nonempty);
    migrate_wake();
}

void driver::migrate_fd_close(int base_fd) {
    auto df = lock();
    migrate_fd_close_.emplace_back(base_fd);
    unlock(df | driver::df_nonempty);
    migrate_wake();
}


// driver methods

thread_local std::unique_ptr<driver> driver::current{new driver};
std::atomic<bool> driver::global_real_time;

driver::driver()
    : virtual_epoch_(std::chrono::system_clock::from_time_t(1634070069)),
      real_time_(global_real_time.load(std::memory_order_relaxed)) {
}

driver::~driver() {
    if (!asap_.empty()
        || !timed_.empty()
        || fds_.has_update()
        || nfdctl_ != 0
        || lock_.load(std::memory_order_relaxed)
        || guard_count_ > 0
        || !keepalives_.empty()) {
        // Clear any remaining events and coroutines
        clear();
        loop();
    }
    fds_.deref_all(this);
    if (epoll_wakefd_ >= 0) {
        ::close(epoll_wakefd_);
    }
    if (pollfd_ >= 0) {
        ::close(pollfd_);
    }
}

bool driver::loop(looptype lt) {
    detail::fd_batch fdb;

    while (true) {
        // import migrated tasks and fd close events
        if (lock_.load(std::memory_order_relaxed) != 0) {
            finish_migrate();
        }

        // process an aliquot of asap and migrated tasks
        for (size_t n = asap_quota; n != 0 && !asap_.empty(); --n) {
            auto eh = std::move(asap_.front());
            asap_.pop_front();
            while (auto coh = eh->driver_trigger(this)) {
                coh();
                step_time();
            }
        }

        // register changes in interested file descriptor set
        while (auto fdu = fds_.pop_update()) {
            apply_fd_update(fdb, *fdu);
        }

        // remove dead keepalives
        while (!keepalives_.empty() && keepalives_.back()->triggered()) {
            keepalives_.pop_back();
        }

        // exit if nothing to do
        timed_.cull();
        if (timed_.empty()
            && nfdctl_ == 0
            && !lock_.load(std::memory_order_relaxed)
            && guard_count_ == 0
            && keepalives_.empty()) {
            clearing_ = false;
            return false;
        }

        // if clearing, empty fds, keepalives, and timeouts
        if (clearing_) {
            process_clearing();
        }

        // compute timeout
        duration timeout;
        if (!asap_.empty()
            || !real_time_
            || lock_.load(std::memory_order_relaxed)
            || lt == looptype::poll
            || clearing_) {
            timeout = duration::zero();
        } else if (!timed_.empty()) {
            timeout = timed_.top_time() - steady_now();
        } else {
            timeout = duration(1h);
        }

        // block or poll for file descriptor events
        bool had_fd_event = false;
        if (nfdctl_ == 0 && timeout <= duration::zero()) {
            fdb.clear(pollfd_);
        } else {
            // call kqueue/epoll/poll, process batch of returned events
            had_fd_event = watch_fds(fdb, timeout);
        }

        // update time
        if (real_time_) {
            snow_ = steady_now();
        } else if (!timed_.empty()
                   && asap_.empty()
                   && !had_fd_event) {
            snow_ = timed_.top_time();
        }

        // process timers
        while (!timed_.empty()
               && timed_.top_time() <= snow_) {
            auto eh = std::move(timed_.top());
            timed_.pop();
            while (auto coh = eh->driver_trigger(this)) {
                coh();
                step_time();
            }
        }

        // exit if polling
        if (lt == looptype::poll) {
            return true;
        }
    }
}

void driver::finish_migrate() {
    std::vector<detail::event_handle> migrate;
    std::vector<int> migrate_fd_close;

    uint32_t flags = lock();
    migrate_.swap(migrate);
    migrate_fd_close_.swap(migrate_fd_close);
    unlock(flags & ~df_nonempty);

    std::move(migrate.begin(), migrate.end(), std::back_inserter(asap_));
    for (auto base_fd : migrate_fd_close) {
        notify_close(base_fd);
    }
}

void driver::clear() {
    clearing_ = true;
}

void driver::process_clearing() {
    assert(clearing_);
    keepalives_.clear();
    guard_count_ = 0;

    // trigger all fd events (but coroutines throw rather than running)
    int fd = -1;
    while (auto fdu = fds_.next_nonempty(fd)) {
        fd = fdu->fd;
        for (int interest = 0; interest < 3; ++interest) {
            if (auto eh = fds_.take(fd, interest, 0)) {
                while (auto coh = eh->driver_trigger(this)) {
                    coh();
                }
            }
        }
    }

    // trigger all timers (but coroutines throw rather than running)
    while (!timed_.empty()) {
        auto eh = std::move(timed_.top());
        timed_.pop();
        while (auto coh = eh->driver_trigger(this)) {
            coh();
        }
    }
}

void reset() {
    driver::current.reset(new driver);
}



// event functions

std::string event::debug_info() const {
    return std::format("#<event {}{}>", static_cast<void*>(handle().get()),
                       triggered() ? " triggered" : "");
}


// mutex functions

inline bool mutex::allow(bool is_shared, latch_type l) const noexcept {
    return is_shared ? !(l & mf_lock_excl) : l < mf_lock_excl;
}

inline auto mutex::notify_locked(latch_type l) -> latch_type {
    while (!waiters_.empty()) {
        auto& fw = waiters_.front();
        if (!fw.empty()) {
            bool fws = waiter_shared(fw);
            if (!allow(fws, l)) {
                break;
            }
            if (fw->trigger()) {
                l += fws ? mf_lock_shared : mf_lock_excl;
            }
        }
        waiters_.pop_front();
    }
    return l;
}

void mutex::lock_impl(bool is_shared, detail::event_handle& e) {
    latch_type l = latch();
    l = notify_locked(l);
    if (waiters_.empty() && allow(is_shared, l)) {
        if (e) {
            e->trigger();
        }
        l += is_shared ? mf_lock_shared : mf_lock_excl;
    } else {
        if (!e) {
            e = detail::event_handle(new detail::event_body);
        }
        if (is_shared) {
            e->set_user_flags(detail::ef_user);
        }
        waiters_.push_back(e);
    }
    unlatch(l);
}

void mutex::unlock_impl(bool is_shared) {
    latch_type l = latch();
    l = notify_locked(l - (is_shared ? mf_lock_shared : mf_lock_excl));
    unlatch(l);
}


// error functions

cotamer_error::cotamer_error(cotamer_errc ec)
    : std::logic_error(message(ec)), errc_(ec) {
}

constexpr const char* cotamer_error::message(cotamer_errc ec) noexcept {
    switch (ec) {
    case cotamer_errc::cross_driver_await:
        return "cannot co_await a task created on a different driver";
    default:
        return "unknown cotamer error";
    }
}

}
