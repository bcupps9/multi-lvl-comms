#include "cotamer/io.hh"
#include <sstream>
#include <thread>
#include <netdb.h>

namespace cotamer {

struct timespec duration_timespec(duration d) {
    if (d <= duration::zero()) {
        return {0, 0};
    }
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(d);
    auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(d - sec);
    return {sec.count(), nsec.count()};
}

int duration_milliseconds(duration d) {
    if (d <= duration::zero()) {
        return 0;
    }
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(d);
    return msec.count();
}


// file descriptor functions

namespace detail {

void fd_event_set::deref_all(driver* drv) {
    // must be called before destroying `fd_event_set`
    if (!capacity_) {
        return;
    }
    for (fdrec* fdr = fdrs_; fdr != fdrs_ + capacity_; ++fdr) {
        if (fdr->body) {
            fdr->body->remove_listener(drv);
        }
    }
}

void fd_event_set::hard_ensure(unsigned ufd) {
    assert(ufd >= capacity_);
    // choose new capacity to fit; double up to `block_capacity`
    unsigned ncapacity;
    if (ufd >= block_capacity) {
        ncapacity = ((ufd / block_capacity) + 1) * block_capacity;
    } else {
        ncapacity = std::max(first_capacity, capacity_ * 2);
        while (ufd >= ncapacity) {
            ncapacity *= 2;
        }
    }
    // allocate, copy, initialize
    fdrec* nfdrs = std::allocator<fdrec>().allocate(ncapacity);
    std::uninitialized_move(fdrs_, fdrs_ + capacity_, nfdrs);
    std::uninitialized_default_construct(nfdrs + capacity_, nfdrs + ncapacity);
    if (capacity_) {
        std::destroy(fdrs_, fdrs_ + capacity_);
        std::allocator<fdrec>().deallocate(fdrs_, capacity_);
    }
    // assign
    fdrs_ = nfdrs;
    capacity_ = ncapacity;
}

void fd_body::deref_close(bool deref) {
    lock();
    // obtain local copies of `drivers_` and `base_fd_` (cannot refer to `this`
    // after unlock, because another thread might delete this)
    small_vector<driver*, 4> local_drivers;
    for (auto dx : drivers_) {
        local_drivers.push_back(dx);
    }
    int local_base_fd = base_fd_;
    // mark as closed, but only once
    bool need_close = fd_.load(std::memory_order_relaxed) >= 0;
    if (need_close) {
        fd_.store(-1, std::memory_order_relaxed);
    }
    // actually dereference
    if (deref) {
        ref_.fetch_sub(1, std::memory_order_release);
    }
    unlock();

    // notify drivers
    if (local_drivers.empty() && deref) {
        delete this;
    } else if (need_close) {
        driver* my_driver = driver::current.get();
        for (auto dx : local_drivers) {
            if (dx == my_driver) {
                dx->notify_close(local_base_fd);
            } else {
                dx->migrate_fd_close(local_base_fd);
            }
        }
    }
}

}


namespace detail {

void fd_batch::print_changes() {
#if COTAMER_USE_KQUEUE
    for (int i = 0; i != changes; ++i) {
        const auto& e = ev[i];
        std::string t;
        switch (e.filter) {
        case EVFILT_READ:   t = std::format("READ {}", e.ident);    break;
        case EVFILT_EXCEPT: t = std::format("EXCEPT {}", e.ident);  break;
        case EVFILT_WRITE:  t = std::format("WRITE {}", e.ident);   break;
        case EVFILT_VNODE:  t = std::format("VNODE {}", e.ident);   break;
        case EVFILT_PROC:   t = std::format("PROC {}", e.ident);    break;
        case EVFILT_SIGNAL: t = std::format("SIGNAL {}", e.ident);  break;
        case EVFILT_TIMER:  t = std::format("TIMER {}", e.ident);   break;
        case EVFILT_USER:   t = "USER";                             break;
        default:            t = std::format("#{} {}", e.filter, e.ident); break;
        }
        if (e.flags & EV_ADD) {
            t += " ADD";
        }
        if (e.flags & EV_DELETE) {
            t += " DELETE";
        }
        std::print(std::cerr, "<{} U{}>\n", t, e.udata);
    }
#endif
}

inline int fd_batch::mask_out(int mask) noexcept {
#if COTAMER_USE_KQUEUE
    // should never be called
    (void) mask;
    return 0;
#elif COTAMER_USE_EPOLL
    return (mask & 1 ? int(EPOLLIN | EPOLLRDHUP) : 0)
        | (mask & 2 ? int(EPOLLOUT) : 0)
        | (mask & 4 ? int(EPOLLRDHUP) : 0);
#else
    return (mask & 1 ? int(POLLIN) : 0)
        | (mask & 2 ? int(POLLOUT) : 0);
#endif
}

inline int fd_batch::mask_in(const system_event_type& se) noexcept {
#if COTAMER_USE_KQUEUE
    return (se.filter == EVFILT_READ ? 1 : 0)
        | (se.filter == EVFILT_WRITE ? 2 : 0)
        | (se.flags & EV_EOF ? 7 : 0);
#elif COTAMER_USE_EPOLL
    return (se.events & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR) ? 1 : 0)
        | (se.events & (EPOLLOUT | EPOLLHUP | EPOLLERR) ? 2 : 0)
        | (se.events & (EPOLLRDHUP | EPOLLHUP | EPOLLERR) ? 4 : 0);
#else
    return (se.revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL) ? 1 : 0)
        | (se.revents & (POLLOUT | POLLERR | POLLHUP | POLLNVAL) ? 2 : 0)
        | (se.revents & (POLLERR | POLLHUP | POLLNVAL) ? 4 : 0);
#endif
}

inline void fd_batch::add(int pollfd, const fd_update& fdu, int old_mask) {
#if COTAMER_USE_KQUEUE
    if (changes == capacity) {
        clear(pollfd);
    }
    void* udata = reinterpret_cast<void*>(uintptr_t(fdu.epoch));
    if ((fdu.mask & 5) && !(old_mask & 5)) {
        EV_SET(&ev[changes], fdu.fd, EVFILT_READ, EV_ADD, 0, 0, udata);
        ++changes;
    } else if (!(fdu.mask & 5) && (old_mask & 5)) {
        EV_SET(&ev[changes], fdu.fd, EVFILT_READ, EV_DELETE, 0, 0, udata);
        ++changes;
    }
    if (bool(fdu.mask & 6) == bool(old_mask & 6)) {
        return;
    }
    if (changes == capacity) {
        clear(pollfd);
    }
    if ((fdu.mask & 6) && !(old_mask & 6)) {
        EV_SET(&ev[changes], fdu.fd, EVFILT_WRITE, EV_ADD, 0, 0, udata);
        ++changes;
    } else if (!(fdu.mask & 6) && (old_mask & 6)) {
        EV_SET(&ev[changes], fdu.fd, EVFILT_WRITE, EV_DELETE, 0, 0, udata);
        ++changes;
    }
#elif COTAMER_USE_EPOLL
    epoll_event epev;
    epev.events = mask_out(fdu.mask);
    epev.data.u64 = fdu.fd | (uint64_t(fdu.epoch) << 32);
    int op = fdu.mask ? (old_mask ? EPOLL_CTL_MOD : EPOLL_CTL_ADD) : EPOLL_CTL_DEL;
    if (epoll_ctl(pollfd, op, fdu.fd, &epev) < 0) {
        throw errno_error();
    }
#else
    (void) pollfd, (void) fdu, (void) old_mask;
#endif
}

inline std::optional<fd_update> fd_batch::pop() noexcept {
    while (index < size) {
        auto& e = ev[index];
        ++index;
#if COTAMER_USE_KQUEUE
        int fd = e.ident;
        int mask = mask_in(e);
        unsigned epoch = reinterpret_cast<uintptr_t>(e.udata);
#elif COTAMER_USE_EPOLL
        int fd = e.data.u64 & 0xFFFF'FFFF;
        int mask = mask_in(e);
        unsigned epoch = e.data.u64 >> 32;
#else
        int fd = e.fd;
        int mask = mask_in(e);
        unsigned epoch = fd_event_set::empty_epoch;
#endif
        if (mask) {
            return {{fd, mask, epoch}};
        }
    }
    return std::nullopt;
}

}

void driver::hard_pollfd() {
#if COTAMER_USE_KQUEUE
    if ((pollfd_ = kqueue()) < 0) {
        throw errno_error();
    }
    fcntl(pollfd_, F_SETFD, FD_CLOEXEC);
    // prepare wake event for cross-thread wake
    struct kevent ev;
    EV_SET(&ev, -1, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
    if (kevent(pollfd_, &ev, 1, nullptr, 0, nullptr) < 0) {
        throw errno_error();
    }
#elif COTAMER_USE_EPOLL
    if ((pollfd_ = epoll_create1(EPOLL_CLOEXEC)) < 0) {
        throw errno_error();
    }
    if ((epoll_wakefd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) < 0) {
        int saved = errno;
        ::close(pollfd_);
        pollfd_ = -1;
        throw std::system_error(saved, std::generic_category());
    }
    epoll_event epev;
    epev.events = EPOLLIN;
    epev.data.u64 = epoll_wakefd_ | (uint64_t(detail::fd_event_set::internal_epoch) << 32);
    if (epoll_ctl(pollfd_, EPOLL_CTL_ADD, epoll_wakefd_, &epev) < 0) {
        throw errno_error();
    }
#endif
}

void driver::apply_fd_update(detail::fd_batch& batch,
                             const detail::fd_update& fdu) {
    // look up events currently registered on our event notification fd
    // for file `fdu.fd`; we store them in `fdctl_` in bit-packed form
    unsigned fdci = unsigned(fdu.fd) / 16,
        fdcs = (unsigned(fdu.fd) % 16) * 4;
    if (fdci >= fdctl_.size()) {
        fdctl_.resize(fdci + 1, uint64_t(0));
    }

    // return if no change (e.g., firing a read event removed the readable
    // watch, but application code re-installed that watch)
    int old_mask = (fdctl_[fdci] >> fdcs) & 15;
    if (old_mask == fdu.mask) {
        return;
    }

    // add to the batch of event notification fd updates
    batch.add(pollfd(), fdu, old_mask);

    // record the new notification state in `fdctl_`
    fdctl_[fdci] ^= (old_mask ^ fdu.mask) << fdcs;
    if (old_mask == 0) {
        ++nfdctl_;
    } else if (fdu.mask == 0) {
        --nfdctl_;
    }
}

bool driver::watch_fds(detail::fd_batch& batch, duration timeout) {
    // Ensure pollfd if we are asked to block before any fd registrations
    (void) pollfd();

#if COTAMER_USE_KQUEUE || COTAMER_USE_EPOLL
    wakefd_.store(pollfd_, std::memory_order_seq_cst);
    if (lock_.load(std::memory_order_seq_cst)) {
        timeout = duration::zero();
    }
#endif

#if COTAMER_USE_KQUEUE
    // block in kernel
    struct timespec ts = duration_timespec(timeout);
    batch.size = kevent(pollfd_, batch.ev, batch.changes, batch.ev, batch.capacity, &ts);
    batch.changes = 0;
#elif COTAMER_USE_EPOLL
    batch.size = epoll_wait(pollfd_, batch.ev, batch.capacity, duration_milliseconds(timeout));
#else
    // The poll() fallback has no cross-thread wake mechanism (no equivalent
    // of EVFILT_USER or eventfd). Cross-thread triggers will be delayed
    // until the poll timeout expires. This is an acceptable tradeoff for a
    // portability fallback; kqueue/epoll should be used on production systems.
    //
    // NB: rebuilds the entire pollfd array on every call (O(max_fd) scan).
    // kqueue/epoll maintain kernel-side state and avoid this cost.
    batch.ev.clear();
    int fd = -1;
    while (auto fdu = fds_.next_nonempty(fd)) {
        batch.ev.emplace_back(fdu->fd, batch.mask_out(fdu->mask), 0);
        fd = fdu->fd;
    }
    poll(batch.ev.begin(), batch.ev.size(), duration_milliseconds(timeout));
    batch.size = batch.ev.size();
#endif
    batch.index = 0;

    // check for unexpected error
    if (batch.size < 0 && errno != EINTR) {
        throw errno_error();
    }

#if COTAMER_USE_KQUEUE || COTAMER_USE_EPOLL
    wakefd_.store(-1, std::memory_order_relaxed);
#endif

    // process returned events
    while (auto fdu = batch.pop()) {
        if (fdu->mask & 1) {
            if (auto eh = fds_.take(fdu->fd, 0, fdu->epoch)) {
                while (auto coh = eh->driver_trigger(this)) {
                    coh();
                    step_time();
                }
            }
#if COTAMER_USE_EPOLL
            else if (fdu->fd == epoll_wakefd_) {
                uint64_t v;
                ssize_t nr = ::read(epoll_wakefd_, &v, sizeof(v));
                (void) nr;
            }
#endif
        }
        if (fdu->mask & 2) {
            if (auto eh = fds_.take(fdu->fd, 1, fdu->epoch)) {
                while (auto coh = eh->driver_trigger(this)) {
                    coh();
                    step_time();
                }
            }
        }
        if (fdu->mask & 4) {
            if (auto eh = fds_.take(fdu->fd, 2, fdu->epoch)) {
                while (auto coh = eh->driver_trigger(this)) {
                    coh();
                    step_time();
                }
            }
        }
    }

    // return true if we saw any fd activity, possibly including internal
    // activity (like a notification from another thread)
    return batch.size > 0;
}


namespace {
// ensure that the return value from `getaddrinfo` is properly freed
struct getaddrinfo_value {
    struct addrinfo hints{};
    struct addrinfo* ai = nullptr;
    int status = 0;
    ~getaddrinfo_value() {
        if (status == 0 && ai) {
            freeaddrinfo(ai);
        }
    }
};
}

static void getaddrinfo_thread(std::string address,
                               std::shared_ptr<getaddrinfo_value> res,
                               event notifier) {
    auto colon = address.rfind(':');
    if (colon == std::string::npos) {
        res->status = EAI_NONAME;
        notifier.trigger();
        return;
    }
    auto host = address.substr(0, colon);
    auto port = address.substr(colon + 1);
    res->status = getaddrinfo(host.empty() ? nullptr : host.c_str(),
                              port.c_str(), &res->hints, &res->ai);
    notifier.trigger();
}

task<cotamer::fd> tcp_listen(std::string address, int backlog) {
    // DNS lookup can block, so do it on a separate thread
    auto res = std::make_shared<getaddrinfo_value>();
    res->hints.ai_family = AF_UNSPEC;
    res->hints.ai_socktype = SOCK_STREAM;
    res->hints.ai_flags = AI_PASSIVE;

    // start thread for lookup, co_await result
    event notifier;
    driver_guard guard;
    std::thread(getaddrinfo_thread, std::move(address), res, notifier).detach();
    co_await notifier;
    if (res->status != 0) {
        throw std::runtime_error(gai_strerror(res->status));
    }

    // `getaddrinfo` can return multiple addresses (e.g., IPv4 and IPv6);
    // try each one in turn
    int last_errno = EADDRNOTAVAIL;
    for (auto ai = res->ai; ai; ai = ai->ai_next) {
        int fileno = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fileno < 0) {
            last_errno = errno;
            continue;
        }

        set_nonblocking(fileno);
        int flag = 1;
        setsockopt(fileno, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
        if (ai->ai_family == AF_INET6) {
            setsockopt(fileno, IPPROTO_IPV6, IPV6_V6ONLY, &flag, sizeof(flag));
        }

        if (bind(fileno, ai->ai_addr, ai->ai_addrlen) == 0
            && listen(fileno, backlog) == 0) {
            // success
            co_return cotamer::fd(fileno);
        }

        last_errno = errno;
        ::close(fileno);
        fileno = -1;
    }
    throw std::system_error(last_errno, std::generic_category());
}

task<cotamer::fd> tcp_connect(std::string address) {
    auto res = std::make_shared<getaddrinfo_value>();
    res->hints.ai_family = AF_UNSPEC;
    res->hints.ai_socktype = SOCK_STREAM;

    // start thread for lookup, co_await result
    event notifier;
    driver_guard guard;
    std::thread(getaddrinfo_thread, std::move(address), res, notifier).detach();
    co_await notifier;
    if (res->status) {
        throw std::runtime_error(gai_strerror(res->status));
    }

    // try each address in turn
    std::exception_ptr last_err;
    for (auto ai = res->ai; ai; ai = ai->ai_next) {
        int fileno = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fileno < 0) {
            last_err = std::make_exception_ptr(errno_error());
            continue;
        }

        set_nonblocking(fileno);
        int flag = 1;
        setsockopt(fileno, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        cotamer::fd f(fileno);
        try {
            co_await connect(f, ai->ai_addr, ai->ai_addrlen);
            co_return std::move(f);
        } catch (...) {
            last_err = std::current_exception();
        }
    }
    std::rethrow_exception(last_err);
}

}
