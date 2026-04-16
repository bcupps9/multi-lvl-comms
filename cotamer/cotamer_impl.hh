#pragma once
#include "cotamer/small_vector.hh"
#include <unistd.h>
#include <system_error>
#if defined(__x86_64__)
# include <xmmintrin.h>
#endif


// cotamer_impl.hh
//    This file defines the gory details of the Cotamer implementation,
//    including C++ coroutine machinery.

namespace cotamer {
namespace detail {

#if COTAMER_STATS
# define COTAMER_STAT_INCR(x) cotamer::stats.x.fetch_add(1, std::memory_order_relaxed)
#else
# define COTAMER_STAT_INCR(x)
#endif

// mark a listener as a quorum
constexpr uintptr_t lf_quorum = uintptr_t(1);

// event_body::flags_
constexpr uint32_t ef_quorum = 1;         // this is a quorum_event_body
constexpr uint32_t ef_lock = 2;           // locked
constexpr uint32_t ef_empty = 4;          // has no listeners
constexpr uint32_t ef_empty_members = 8;  // has no members
constexpr uint32_t ef_triggered = 16;     // has been triggered
constexpr uint32_t ef_want_interest = 32; // transitive quorum member has interest{}
constexpr uint32_t ef_user = 64;          // first user flag
constexpr uint32_t ef_nuser = 4;          // number of user flags
constexpr uint32_t efm_user = 0x3C0;      // mask of user flags
constexpr uint32_t ef_interest = 1024;    // this quorum has 1 interest{}
                                          // (added once per interest{})

// exception thrown during driver::clearing()
struct clearing_exception {};

inline void spinlock_hint() {
#if defined(__x86_64__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

// fd_body
//    Reference-counted body for cotamer::fd. Strong refs represent ownership;
//    weak refs are held by fd_event_set::fdrec entries. When the last strong
//    ref drops, the fd is closed and all associated events are triggered.

struct fd_body {
    std::atomic<int> fd_;
    int base_fd_;
    std::atomic<uint32_t> ref_ = 1;
    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    small_vector<driver*, 1> drivers_;

    static constexpr uint32_t wr_lock = 1;

    explicit fd_body(int raw) : fd_(raw), base_fd_(raw) {}
    fd_body(const fd_body&) = delete;
    fd_body(fd_body&&) = delete;
    fd_body& operator=(const fd_body&) = delete;
    fd_body& operator=(fd_body&&) = delete;
    ~fd_body() {
        if (base_fd_ >= 0) {
            ::close(base_fd_);
        }
    }

    inline int base_fileno() const noexcept {
        return base_fd_;
    }

    inline int fileno() const noexcept {
        return fd_.load(std::memory_order_relaxed);
    }

    inline void ref() noexcept {
        ref_.fetch_add(1, std::memory_order_relaxed);
    }

    inline void add_listener(driver* d) noexcept {
        lock();
        drivers_.push_back(d);
        unlock();
    }

    inline void remove_listener(driver* d) noexcept {
        lock();
        for (auto& dx : drivers_) {
            if (dx == d) {
                dx = drivers_.back();
                drivers_.pop_back();
                break;
            }
        }
        bool del = drivers_.empty()
            && ref_.load(std::memory_order_acquire) == 0;
        unlock();
        if (del) {
            delete this;
        }
    }

    inline void deref() noexcept {
        if (ref_.load(std::memory_order_acquire) == 1) {
            deref_close(true);
        } else {
            ref_.fetch_sub(1, std::memory_order_release);
        }
    }

    void deref_close(bool deref);         // non-inline, in cotamer.cc

    inline void lock() {
        while (lock_.test_and_set(std::memory_order_acquire)) {
            spinlock_hint();
        }
    }

    inline void unlock() {
        lock_.clear(std::memory_order_release);
    }
};


// task_promise<T>
//    This structure is part of the C++ coroutine machinery. Coroutines don’t
//    actually use the stack for local variables; they can’t, because they can
//    suspend themselves and resume later. When a coroutine is first called, the
//    C++ runtime allocates a heap structure for the function’s locals. That
//    heap structure also contains this *promise*. The promise interface is
//    defined by the C++ language standard; the runtime calls its methods in
//    specific situations, such as when a `co_await` expression is evaluated.

struct task_promise_base {
    bool detached_ = false;                // is this task detached?
    bool has_interest_ = false;            // has interest been requested?
    driver* home_;                         // coroutine home driver
    event_handle completion_;              // completion event (lazily created)
    event_handle interest_;                // interest event (lazily created)
    // coroutine awaiting me, if any
    std::coroutine_handle<task_promise_base> continuation_;

    inline task_promise_base()
        : home_(driver::current.get()) {
        COTAMER_STAT_INCR(promises_allocated);
    }
    inline ~task_promise_base() {
        COTAMER_STAT_INCR(promises_destroyed);
    }

    inline event_handle& make_interest();
};

template <typename T>
struct task_promise : public task_promise_base {
    // Functions required by the C++ runtime
    // - Initialize the task<T> return value that manages the coroutine:
    inline task<T> get_return_object() noexcept;
    // - Behavior when coroutine starts (here, run eagerly):
    std::suspend_never initial_suspend() noexcept { return {}; }
    // - Handle `co_await E` for different `E` types:
    task_event_awaiter<T> await_transform(event ev);
    template <bool shared>
    task_mutex_event_awaiter<T, shared> await_transform(mutex_event<shared> ev);
    task_mutex_event_awaiter<T, false> await_transform(mutex&);
    inline task_event_awaiter<T> await_transform(interest);
    inline interest_event_awaiter await_transform(interest_event);
    template <typename Aw>
    Aw&& await_transform(Aw&& aw) noexcept { return std::forward<Aw>(aw); }
    // - Handle `co_return V` or throwing an exception in the coroutine:
    void return_value(T value) { result_.template emplace<1>(std::move(value)); }
    void unhandled_exception() noexcept { result_.template emplace<2>(std::current_exception()); }
    // - Behavior after coroutine exits:
    task_final_awaiter<T> final_suspend() noexcept;
    // - Export coroutine return value to `co_await`er:
    inline T result();

    std::variant<std::monostate, T, std::exception_ptr> result_;
};

template <typename T>
inline task<T> task_promise<T>::get_return_object() noexcept {
    // When an event schedules a task_promise, it needs the task's home driver.
    // It uses coroutine_handle<task_promise_base>::from_address() to get it.
    // That is strictly speaking UB -- one can only call
    // coroutine_handle<T>::from_address() if T is the actual promise type or
    // void -- but it works on GCC and Clang when the actual promise type
    // and the base type have the same alignment.
    static_assert(alignof(task_promise<T>) == alignof(task_promise_base));
    return task<T>{std::coroutine_handle<task_promise<T>>::from_promise(*this)};
}

template <typename T>
T task_promise<T>::result() {
    if (result_.index() == 2) {
        std::rethrow_exception(std::move(std::get<2>(result_)));
    }
    return std::move(std::get<1>(result_));
}


// task_promise<void>
//    Similar, but no value is returned.

template <>
struct task_promise<void> : public task_promise_base {
    inline task<void> get_return_object() noexcept;
    std::suspend_never initial_suspend() noexcept { return {}; }
    task_event_awaiter<void> await_transform(event ev);
    template <bool shared>
    task_mutex_event_awaiter<void, shared> await_transform(mutex_event<shared> ev);
    inline task_mutex_event_awaiter<void, false> await_transform(mutex&);
    inline task_event_awaiter<void> await_transform(interest);
    inline interest_event_awaiter await_transform(interest_event);
    template <typename Aw>
    Aw&& await_transform(Aw&& aw) noexcept { return std::forward<Aw>(aw); }
    void return_void() noexcept { }
    void unhandled_exception() noexcept { exception_ = std::current_exception(); }
    void result() {
        if (exception_) {
            std::rethrow_exception(std::move(exception_));
        }
    }
    inline task_final_awaiter<void> final_suspend() noexcept;

    std::exception_ptr exception_;
};

inline task<void> task_promise<void>::get_return_object() noexcept {
    // See comment in task_promise<T>::get_return_object.
    static_assert(alignof(task_promise<void>) == alignof(task_promise_base));
    return task<void>{std::coroutine_handle<task_promise<void>>::from_promise(*this)};
}


// task_awaiter<T>
//    This structure is also part of the C++ coroutine machinery. “Awaiter”
//    objects are created as part of `co_await` expression evaluation; the
//    C++ runtime calls their methods to determine whether to suspend, how
//    to handle a suspension (including what to run next), and how to resume.
//
//    task_awaiter<T> awaits a task. We also define task_final_awaiter<T>,
//    which handles the implicit final suspension when a coroutine completes;
//    task_event_awaiter<T>, which awaits an event; and a few others.

template <typename T>
struct task_awaiter {
    // - Return true if `co_await` should not suspend
    bool await_ready() noexcept {
        return self_.done();
    }
    // - Suspend this coroutine and return the next coroutine to execute
    template <typename U>
    std::coroutine_handle<> await_suspend(std::coroutine_handle<task_promise<U>> awaiting) {
        if (awaiting.promise().home_ != self_.promise().home_) {
            throw cotamer_error(cotamer_errc::cross_driver_await);
        }
        // XXX UB, but see task_promise<T>::get_return_object
        std::coroutine_handle<task_promise_base> base_awaiting =
            std::coroutine_handle<task_promise_base>::from_address(awaiting.address());
        self_.promise().continuation_ = base_awaiting;
        if (self_.promise().interest_) {
            self_.promise().interest_->trigger();
        }
        return std::noop_coroutine();
    }
    // - Resume this coroutine, returning the `co_await` expression’s result
    T await_resume() {
        return self_.promise().result();
    }

    std::coroutine_handle<task_promise<T>> self_;
};


// Awaiter for the implicit final suspension when a coroutine completes.
template <typename T>
struct task_final_awaiter {
    bool await_ready() noexcept {
        return false;
    }
    inline std::coroutine_handle<> await_suspend(std::coroutine_handle<task_promise<T>> self) noexcept {
        auto& promise = self.promise();
        // trigger completion event, since the task is done
        if (promise.completion_) {
            promise.completion_->trigger();
        }
        // if another coroutine wants this task's result, resume it directly
        // (cross-driver awaits are rejected at co_await time, so the
        // continuation is always on the same driver)
        if (promise.continuation_) {
            return std::exchange(promise.continuation_, nullptr);
        }
        // destroy if detached and then return to event loop
        if (promise.detached_) {
            self.destroy();
        }
        return std::noop_coroutine();
    }
    void await_resume() noexcept {
    }
};

template <typename T>
inline task_final_awaiter<T> task_promise<T>::final_suspend() noexcept {
    return {};
}

inline task_final_awaiter<void> task_promise<void>::final_suspend() noexcept {
    return {};
}


// make_event: converts various types into events.

template <typename T>
inline event make_event(T resumable) {
    co_await resumable;
}

inline event& make_event(event& ev) {
    return ev;
}

inline event&& make_event(event&& ev) {
    return std::move(ev);
}

template <bool shared>
inline event make_event(mutex_event<shared>& ev) {
    return event(ev.handle());
}

template <bool shared>
inline event make_event(mutex_event<shared>&& ev) {
    return event(std::move(ev.handle()));
}

inline event make_event(interest);


// event_body
//    The heap-allocated state behind an event. Managed by reference-counted
//    event_handle smart pointers. Each event_body has a list of *listeners*
//    (coroutines or quorum bodies) that are notified when the event triggers.
//
//    Event bodies can be accessed concurrently from multiple threads. The lock
//    is a bit in `flags_`. Difficulties arise with quorum events (coroutines
//    are naturally linear). Sometimes a member of a quorum event needs to lock
//    its quorum, and sometimes a quorum needs to lock its members; deadlock is
//    avoided because when a quorum locks its members, it gives up trying as
//    soon as member becomes triggered. Logic in trigger_unlock() and around
//    reference counts ensures everything works out.

struct event_body {
    inline event_body()    { COTAMER_STAT_INCR(events_allocated); }
    inline ~event_body()   { COTAMER_STAT_INCR(events_destroyed); }
    event_body(const event_body&) = delete;
    event_body(event_body&&) = delete;
    event_body& operator=(const event_body&) = delete;
    event_body& operator=(event_body&&) = delete;

    void ref(uint32_t n = 1) noexcept {
        refcount_.fetch_add(n, std::memory_order_relaxed);
    }

    uint32_t relaxed_flags() const noexcept {
        return flags_.load(std::memory_order_relaxed);
    }

    void set_user_flags(uint32_t userf) {
        assert((userf & ~efm_user) == 0);
        while (true) {
            uint32_t flags = relaxed_flags();
            if (flags_.compare_exchange_weak(flags, (flags & ~efm_user) | userf, std::memory_order_relaxed)) {
                return;
            }
            spinlock_hint();
        }
    }

    // This event can be garbage collected: it has triggered, or it has no
    // listeners and no other references.
    bool empty() const noexcept {
        auto f = relaxed_flags();
        return (f & ef_triggered)
            || ((f & ef_empty) && refcount_.load(std::memory_order_relaxed) == 1);
    }

    // This event has no listeners.
    bool idle() const noexcept {
        return (relaxed_flags() & ef_empty) != 0;
    }

    // The event has triggered.
    bool triggered() const noexcept {
        return (relaxed_flags() & ef_triggered) != 0;
    }

    inline uint32_t lock() {
        while (true) {
            uint32_t flags = relaxed_flags();
            if ((flags & ef_lock) == 0
                && flags_.compare_exchange_weak(flags, flags | ef_lock, std::memory_order_acquire, std::memory_order_relaxed)) {
                return flags;
            }
            spinlock_hint();
        }
    }

    inline uint32_t untriggered_lock() {
        while (true) {
            uint32_t flags = relaxed_flags();
            if ((flags & ef_triggered)
                || ((flags & ef_lock) == 0
                    && flags_.compare_exchange_weak(flags, flags | ef_lock, std::memory_order_acquire, std::memory_order_relaxed))) {
                return flags;
            }
            spinlock_hint();
        }
    }

    inline void unlock(uint32_t flags) {
        flags_.store(flags, std::memory_order_release);
    }

    template <typename T>
    inline void add_listener_unlock(std::coroutine_handle<task_promise<T>> coroutine,
                                    uint32_t flags) {
        add_listener_unlock(reinterpret_cast<uintptr_t>(coroutine.address()), flags);
    }

    inline void add_listener_unlock(quorum_event_body* qb, uint32_t flags) {
        add_listener_unlock(reinterpret_cast<uintptr_t>(qb) | lf_quorum, flags);
    }

    template <typename T>
    inline void remove_listener(std::coroutine_handle<task_promise<T>> coroutine) {
        remove_listener_unlock(reinterpret_cast<uintptr_t>(coroutine.address()), lock());
    }

    inline void remove_listener_unlock(quorum_event_body* qb, uint32_t flags) {
        remove_listener_unlock(reinterpret_cast<uintptr_t>(qb) | lf_quorum, flags);
    }

    inline bool trigger() {
        auto f = untriggered_lock();
        return !(f & ef_triggered) && trigger_unlock(f);
    }

    inline std::coroutine_handle<> driver_trigger(driver* drv);

    inline bool trigger_unlock(uint32_t flags, driver* drv = nullptr,
                               std::coroutine_handle<>* cot = nullptr);


    std::atomic<uint32_t> refcount_ = 1;
    std::atomic<uint32_t> flags_ = ef_empty;
    small_vector<uintptr_t, 3> listeners_;

private:
    void add_listener_unlock(uintptr_t listener, uint32_t flags) {
        // A listener is either a `coroutine_handle<T>::address()` or the
        // address of a `quorum_event_body`. Quorum bodies are distinguished by
        // setting the `lf_quorum` bit, bit 1; this is safe because coroutines
        // and quorum bodies are both aligned.
        assert(listener && (flags & ef_triggered) == 0);
        listeners_.push_back(listener);
        unlock(flags & ~ef_empty);
    }

    void remove_listener_unlock(uintptr_t listener, uint32_t flags) {
        // Remove a listener. It might have been added multiple times;
        // `remove_listener_unlock` will be called the same number of times.
        for (auto& l : listeners_) {
            if (l == listener) {
                l = listeners_.back();
                listeners_.pop_back();
                break;
            }
        }
        unlock(flags | (listeners_.empty() ? ef_empty : 0));
    }

    static std::coroutine_handle<task_promise_base> listener_coroutine(uintptr_t l) noexcept {
        return std::coroutine_handle<task_promise_base>::from_address(reinterpret_cast<void*>(l));
    }

    static quorum_event_body* listener_quorum(uintptr_t l) noexcept {
        return reinterpret_cast<quorum_event_body*>(l & ~lf_quorum);
    }
};


// quorum_event_body
//    A subclass of event_body. Implements `any()` and `all()` by tracking
//    member events, counting the number that have triggered, and triggering its
//    own event (the event_body base type) once a quorum is reached.
//
//    The `ef_interest` and `ef_want_interest` flags implement an optimization
//    that avoids allocating separate memory for `interest{}`.

struct quorum_event_body : event_body {
    template<typename... Es>
    quorum_event_body(size_t quorum, Es&&... es)
        : quorum_(quorum) {
        uint32_t qf = ef_quorum | ef_empty | ef_empty_members;
        flags_.store(qf | ef_lock, std::memory_order_release);
        ((qf = add_member(qf, std::forward<Es>(es))), ...);
        if (triggered_ >= quorum_) {
            trigger_unlock(qf);
        } else {
            unlock(qf);
        }
    }

    uint32_t add_member(uint32_t qf, event_handle eh) {
        uint32_t ef;
        if (!eh || ((ef = eh->untriggered_lock()) & ef_triggered)) {
            ++triggered_;
            return qf;
        }
        eh->add_listener_unlock(this, ef);
        if (ef & ef_want_interest) {
            qf |= ef_want_interest;
        }
        members_.push_back(std::move(eh));
        return qf & ~ef_empty_members;
    }

    template <typename E>
    inline uint32_t add_member(uint32_t qf, E&& e) {
        return add_member(qf, make_event(std::forward<E>(e)).handle());
    }

    inline uint32_t add_member(uint32_t qf, interest) {
        return (qf | ef_want_interest) + ef_interest;
    }

    // Called by a member event when it triggers. Removes that event from
    // members_ (once), increases the triggered_ count, and potentially triggers
    // this event.
    void trigger_member(event_body* e, driver* drv, std::coroutine_handle<>* coh) {
        auto qf = lock();
        for (auto& mem : members_) {
            if (mem.get() == e) {
                ++triggered_;
                mem.swap(members_.back());
                members_.pop_back();
                if (members_.empty()) {
                    qf |= ef_empty_members;
                }
                break;
            }
        }
        if (!(qf & ef_triggered) && triggered_ >= quorum_) {
            trigger_unlock(qf, drv, coh);
        } else if ((qf & ef_empty_members)
                   && refcount_.load(std::memory_order_relaxed) == 0) {
            delete this;
        } else {
            unlock(qf);
        }
    }

    inline void fix_want_interest(event_handle& ievent);

    uint32_t cull_members(uint32_t qf) {
        for (auto it = members_.begin(); it != members_.end(); ) {
            auto f = (*it)->untriggered_lock();
            if (f & ef_triggered) {
                // The triggered event `*it` is still in our members list. This
                // only happens when the event is still triggering & hasn't yet
                // gotten around to removing itself from our members list. We
                // must wait for them to remove themselves.
                ++it;
            } else {
                (*it)->remove_listener_unlock(this, f);
                it->swap(members_.back());
                members_.pop_back();
            }
        }
        return members_.empty() ? qf | ef_empty_members : qf;
    }

    void hard_deref() {
        // This may not actually delete!
        auto f = lock();
        if (refcount_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
            unlock(f);
            return;
        }
        if (!(f & ef_triggered)) {
            f = cull_members(f);
        }
        if (f & ef_empty_members) {
            delete this;
        } else {
            unlock(f); // One of the concurrent trigger()s will delete us.
        }
    }


    small_vector<event_handle, 3> members_;
    uint32_t triggered_ = 0;
    uint32_t quorum_;
};

// event_body::trigger_unlock: the key trigger machinery

inline bool event_body::trigger_unlock(uint32_t f, driver* drv,
                                       std::coroutine_handle<>* coh) {
    bool result = !(f & ef_empty)
        || refcount_.load(std::memory_order_acquire) > 1;
    // Triggering a quorum removes it from its members' listener lists, but that
    // might cause recursive triggers and eventually drop the last remaining
    // reference to `this`. Add a temporary reference so the quorum survives
    // until the end.
    if (f & ef_quorum) {
        ref();
        auto qbody = static_cast<quorum_event_body*>(this);
        f = qbody->cull_members(f);
    }
    // Process listeners: remove quorums, maybe claim 1 `drv` coroutine,
    // and collect all other interested drivers with interested coroutines.
    small_vector<uintptr_t, 3> quorums;
    small_vector<driver*, 3> drivers;
    auto lit = listeners_.begin(), oit = lit, eit = listeners_.end();
    for (; lit != eit; ++lit) {
        if (*lit & lf_quorum) {
            quorums.push_back(*lit);
            continue;
        }
        auto lcoh = listener_coroutine(*lit);
        driver* ldrv = lcoh.promise().home_;
        if (ldrv == drv) {
            // The coroutine `lcoh` should run on driver `drv`, which called
            // us via `driver_trigger`. No need to post this event to
            // `drivers`: our caller will run it to completion.
            if (!*coh) {
                *coh = lcoh;
                continue;
            }
        } else if (std::find(drivers.begin(), drivers.end(), ldrv) == drivers.end()) {
            drivers.push_back(ldrv);
        }
        if (oit != lit) {
            *oit = *lit;
        }
        ++oit;
    }
    listeners_.truncate(oit);
    // Unlock before triggering quorums to avoid deadlock with quorum
    // trigger_member (which acquires our lock).
    unlock(f | ef_triggered | (listeners_.empty() ? ef_empty : 0));
    // Inform quorums.
    for (auto listener : quorums) {
        auto qb = listener_quorum(listener);
        qb->trigger_member(this, drv, coh);
    }
    // Store references on the `migrate_` lists of remote drivers so they run
    // the relevant coroutines.
    if (!drivers.empty()) {
        ref(drivers.size());
        if (!drv) {
            drv = driver::current.get();
        }
        for (auto* d : drivers) {
            if (d == drv) {
                d->asap_.emplace_back(this);
            } else {
                d->migrate_asap(event_handle{this});
            }
        }
    }
    if (f & ef_quorum) {
        event_handle{this}; // release temporary reference
    }
    return result;
}

inline std::coroutine_handle<> event_body::driver_trigger(driver* drv) {
    std::coroutine_handle<> coh(nullptr);
    if ((flags_.load(std::memory_order_relaxed) & (ef_empty | ef_triggered)) == (ef_empty | ef_triggered)) {
        // definitely nothing left to do, don't bother locking
        return coh;
    }
    auto f = lock();
    if (!(f & ef_triggered)) {
        trigger_unlock(f, drv, &coh);
        return coh;
    }
    // Already triggered; search for a coroutine on this driver.
    for (auto& l : listeners_) {
        if (listener_coroutine(l).promise().home_ == drv) {
            coh = listener_coroutine(l);
            l = listeners_.back();
            listeners_.pop_back();
            break;
        }
    }
    unlock(f | (listeners_.empty() ? ef_empty : 0));
    return coh;
}


// event_handle implementation
//    Reference-counted smart pointer for event_body

inline event_handle::event_handle(event_body* eb) noexcept
    : eb_(eb) {
}

inline event_handle::event_handle(const event_handle& x) noexcept
    : eb_(x.eb_) {
    if (eb_) {
        eb_->ref();
    }
}

inline event_handle::event_handle(event_handle&& x) noexcept
    : eb_(std::exchange(x.eb_, nullptr)) {
}

inline event_handle& event_handle::operator=(std::nullptr_t) {
    event_handle tmp;
    std::swap(eb_, tmp.eb_);
    return *this;
}

inline event_handle& event_handle::operator=(const event_handle& x) {
    if (this != &x) {
        event_handle tmp(x);
        std::swap(eb_, tmp.eb_);
    }
    return *this;
}

inline event_handle& event_handle::operator=(event_handle&& x) noexcept {
    if (this != &x) {
        event_handle tmp(std::move(x));
        std::swap(eb_, tmp.eb_);
    }
    return *this;
}

inline event_handle::~event_handle() {
    if (!eb_) {
        return;
    }
    auto f = eb_->relaxed_flags();
    if ((f & (ef_quorum | ef_empty_members)) == ef_quorum) {
        static_cast<quorum_event_body*>(eb_)->hard_deref();
    } else if (eb_->refcount_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        // do nothing
    } else if (f & ef_quorum) {
        delete static_cast<quorum_event_body*>(eb_);
    } else {
        delete eb_;
    }
}

inline void event_handle::swap(event_handle& x) noexcept {
    auto tmp = eb_;
    eb_ = x.eb_;
    x.eb_ = tmp;
}

inline bool event_handle::empty() const noexcept {
    return !eb_ || eb_->empty();
}



// task_event_awaiter<T>
//    Awaiter for `co_await event` inside a task.

template <typename T>
struct task_event_awaiter {
    event_handle eh_;
    std::coroutine_handle<task_promise<T>> coroutine_;

    ~task_event_awaiter() {
        if (coroutine_) {
            eh_->remove_listener(coroutine_);
        }
    }
    bool await_ready() noexcept {
        return !eh_ || eh_->triggered();
    }
    bool await_suspend(std::coroutine_handle<task_promise<T>> awaiting) noexcept {
        event_body* eb = eh_.get();
        // apply interest{} if necessary, which might trigger `eb`
        if (eb->relaxed_flags() & ef_want_interest) {
            static_cast<quorum_event_body*>(eb)->fix_want_interest(awaiting.promise().make_interest());
        }
        uint32_t ef = eb->untriggered_lock();
        if (ef & ef_triggered) {
            return false;
        }
        coroutine_ = awaiting;
        eb->add_listener_unlock(coroutine_, ef);
        return true;
    }
    void await_resume() {
        // Check if our driver is being cleared, which can only happen if we
        // suspended ((bool) coroutine_).
        bool clearing = coroutine_ && coroutine_.promise().home_->clearing();
        // Optimization: Don't call remove_listener, which will do nothing
        coroutine_ = nullptr;
        // Recover memory when clearing a driver (for instance, if a test exits
        // early). driver::clear() triggers all outstanding events and unblocks
        // their waiting coroutines, but those might have other coroutines
        // waiting for their results. We destroy the whole chain by forcing the
        // event-unblocked coroutines to throw an exception; that exception is
        // propagated through their awaiters.
        if (clearing) {
            throw clearing_exception{};
        }
    }
};

template <typename T>
inline task_event_awaiter<T> task_promise<T>::await_transform(event ev) {
    return task_event_awaiter<T>{std::move(ev).handle(), nullptr};
}

inline task_event_awaiter<void> task_promise<void>::await_transform(event ev) {
    return task_event_awaiter<void>{std::move(ev).handle(), nullptr};
}


// Support `interest{}` and `interest_event{}`

inline event_handle& task_promise_base::make_interest() {
    if (!has_interest_) {
        interest_ = event_handle(new event_body);
        has_interest_ = true;
    }
    return interest_;
}

template <typename T>
inline task_event_awaiter<T> task_promise<T>::await_transform(interest) {
    return task_event_awaiter<T>{make_interest(), nullptr};
}

inline task_event_awaiter<void> task_promise<void>::await_transform(interest) {
    return task_event_awaiter<void>{make_interest(), nullptr};
}

// make_event(interest)
//    Wrap bare `interest{}` in a single-member quorum. This is used when
//    `interest{}` appears as the sole argument to any()/all().

inline event make_event(interest) {
    auto q = new detail::quorum_event_body(1, interest{});
    return event_handle(q);
}

// interest_event_awaiter: for `co_await interest_event{}`

struct interest_event_awaiter {
    event_handle handle_;
    bool await_ready() noexcept { return true; }
    void await_suspend(std::coroutine_handle<>) noexcept { }
    event await_resume() { return event(std::move(handle_)); }
};

template <typename T>
inline interest_event_awaiter task_promise<T>::await_transform(interest_event) {
    return interest_event_awaiter{make_interest()};
}

inline interest_event_awaiter task_promise<void>::await_transform(interest_event) {
    return interest_event_awaiter{make_interest()};
}

// fix_want_interest(ievent)
//    Called when this quorum and/or its transitive members need to apply
//    interest{}. The `ievent` is the lazily-created event_handle representing
//    interest; it may have already triggered.

inline void quorum_event_body::fix_want_interest(event_handle& ievent) {
    uint32_t qf = lock();
    qf &= ~ef_want_interest;
    if (qf & ef_triggered) {
        unlock(qf);
        return;
    }
    // Apply local interest (this quorum has one or more `interest{}` members)
    while (qf >= ef_interest) {
        qf = add_member(qf - ef_interest, ievent);
    }
    // That might trigger `this`. If it does, exit now, *without* propagating
    // interest{} to members. They will need to be awaited explicitly to get
    // the interest notification.
    if (triggered_ >= quorum_) {
        trigger_unlock(qf);
        return;
    }
    // Update members who want interest. Just as with `trigger_unlock()`, we
    // cannot call `mem.fix_want_interest()` directly, since those calls might
    // eventually delete `this`.
    small_vector<event_handle, 3> wi_members;
    for (auto& mem : members_) {
        if (mem->relaxed_flags() & ef_want_interest) {
            wi_members.push_back(mem);
        }
    }
    unlock(qf);
    for (auto& mem : wi_members) {
        static_cast<quorum_event_body*>(mem.get())->fix_want_interest(ievent);
    }
}


// task_mutex_event_awaiter<T, shared>
//    Awaiter for `co_await mutex_event` inside a task.

template <typename T, bool shared>
struct task_mutex_event_awaiter : public task_event_awaiter<T> {
    using parent = task_event_awaiter<T>;
    mutex* m_;

    locked_mutex_t<shared> await_resume() {
        parent::await_resume();
        return locked_mutex_t<shared>{m_};
    }
};

template <typename T>
template <bool shared>
inline task_mutex_event_awaiter<T, shared> task_promise<T>::await_transform(mutex_event<shared> ev) {
    return task_mutex_event_awaiter<T, shared>{{std::move(ev).handle(), nullptr}, ev.mutex()};
}

template <bool shared>
inline task_mutex_event_awaiter<void, shared> task_promise<void>::await_transform(mutex_event<shared> ev) {
    return task_mutex_event_awaiter<void, shared>{{std::move(ev).handle(), nullptr}, ev.mutex()};
}

}


// event methods

inline event::event()
    : ep_(new detail::event_body) {
}

inline event::event(detail::event_handle ev)
    : ep_(std::move(ev)) {
}

inline event::event(std::nullptr_t) {
}

inline bool event::empty() const noexcept {
    return !ep_ || ep_->empty();
}

inline bool event::idle() const noexcept {
    return !ep_ || ep_->idle();
}

inline bool event::triggered() const noexcept {
    return !ep_ || ep_->triggered();
}

inline bool event::trigger() {
    return ep_ && ep_->trigger();
}

inline event& event::arm() {
    if (!ep_ || ep_->triggered()) {
        std::exchange(ep_, detail::event_handle{new detail::event_body});
    }
    return *this;
}

inline bool event::operator==(const event& x) const noexcept {
    return ep_.get() == x.ep_.get();
}

inline bool event::operator!=(const event& x) const noexcept {
    return ep_.get() != x.ep_.get();
}

inline const detail::event_handle& event::handle() const& noexcept {
    return ep_;
}

inline detail::event_handle&& event::handle() && noexcept {
    return std::move(ep_);
}


// task methods

template <typename T>
inline task<T>::task(handle_type handle) noexcept
    : handle_(handle) {
}

template <typename T>
inline task<T>::task(task&& x) noexcept
    : handle_(std::exchange(x.handle_, nullptr)) {
}

template <typename T>
inline task<T>& task<T>::operator=(task&& x) noexcept {
    if (this != &x) {
        if (handle_) {
            handle_.destroy();
        }
        handle_ = std::exchange(x.handle_, nullptr);
    }
    return *this;
}

template <typename T>
inline task<T>::~task() {
    if (handle_) {
        handle_.destroy();
    }
}

template <typename T>
inline task<T>::operator bool() const noexcept {
    return handle_;
}

template <typename T>
inline bool task<T>::empty() const noexcept {
    return !handle_;
}

template <typename T>
inline bool task<T>::done() const {
    return handle_ && handle_.done();
}

template <typename T>
inline event task<T>::completion() {
    if (!handle_) {
        return event();
    } else if (handle_.done()) {
        return event(nullptr);
    }
    auto& p = handle_.promise();
    if (!p.completion_) {
        p.completion_ = detail::event_handle(new detail::event_body);
    }
    return event(p.completion_);
}

template <typename T>
inline void task<T>::start() {
    if (!handle_ || handle_.done()) {
        return;
    }
    auto& p = handle_.promise();
    if (p.interest_) {
        p.interest_->trigger();
    } else {
        p.has_interest_ = true;
    }
}

template <typename T>
inline void task<T>::detach() {
    if (!handle_) {
        return;
    }
    handle_.promise().detached_ = true;
    if (handle_.done()) {
        handle_.destroy();
    }
    handle_ = nullptr;
}

template <typename T>
inline void task<T>::destroy() {
    if (handle_) {
        handle_.destroy();
    }
    handle_ = nullptr;
}

template <typename T>
inline detail::task_awaiter<T> task<T>::operator co_await() const noexcept {
    return detail::task_awaiter<T>{handle_};
}


// driver methods

inline clock driver::clock() const noexcept {
    return real_time_ ? cotamer::clock::real_time : cotamer::clock::virtual_time;
}

inline void driver::set_clock(cotamer::clock ct) {
    real_time_ = (ct == cotamer::clock::real_time);
}

inline system_time_point driver::now() noexcept {
    if (real_time_) {
        return std::chrono::system_clock::now();
    }
    return virtual_epoch_ + std::chrono::duration_cast<std::chrono::system_clock::duration>(snow_.time_since_epoch());
}

inline steady_time_point driver::steady_now() noexcept {
    if (real_time_) {
        return std::chrono::steady_clock::now();
    }
    return snow_;
}

inline void driver::step_time() noexcept {
    if (!real_time_) {
        snow_ += duration{1};
    }
}

inline bool driver::empty() const noexcept {
    return timed_.empty()
        && nfdctl_ == 0
        && !fds_.has_update()
        && !lock_.load(std::memory_order_relaxed)
        && guard_count_ == 0
        && keepalives_.empty();
}

inline bool driver::clearing() const noexcept {
    return clearing_;
}

inline void driver::keepalive(event e) {
    if (!e.triggered()) {
        keepalives_.emplace_back(std::move(e).handle());
    }
}

inline void driver::asap(event e) {
    if (e.handle()) {
        asap_.emplace_back(std::move(e).handle());
    }
}

inline event driver::asap() {
    event e;
    asap(e);
    return e;
}

inline void driver::at(steady_time_point t, event e) {
    if (e.handle()) {
        timed_.emplace(t, std::move(e).handle());
    }
}

inline event driver::at(steady_time_point t) {
    if (!real_time_ && t <= snow_) {
        return event(nullptr);
    }
    event e;
    at(t, e);
    return e;
}

inline void driver::at(system_time_point t, event e) {
    after(t - now(), std::move(e));
}

inline event driver::at(system_time_point t) {
    return after(t - now());
}

inline void driver::after(duration d, event e) {
    at(steady_now() + d, std::move(e));
}

inline event driver::after(duration d) {
    return at(steady_now() + d);
}

template <typename Rep, typename Period>
inline event driver::after(const std::chrono::duration<Rep, Period>& d) {
    return at(steady_now() + std::chrono::duration_cast<duration>(d));
}

template <typename Rep, typename Period>
inline void driver::after(const std::chrono::duration<Rep, Period>& d, event e) {
    at(steady_now() + std::chrono::duration_cast<duration>(d), std::move(e));
}

inline event driver::file_event(const cotamer::fd& f, fdevent type) {
    return fds_.watch(f.fileno(), int(type), f.body(), this);
}

inline void driver::loop() {
    loop(looptype::complete);
}

inline bool driver::poll() {
    return loop(looptype::poll);
}


// driver_guard

inline driver_guard::driver_guard() {
    auto drv = driver::current.get();
    if (drv && !drv->clearing()) {
        drv_ = drv;
        ++drv->guard_count_;
    } else {
        drv_ = nullptr;
    }
}

inline driver_guard::~driver_guard() {
    if (drv_ && !drv_->clearing()) {
        --drv_->guard_count_;
    }
}

inline driver_guard::driver_guard(driver_guard&& x)
    : drv_(std::exchange(x.drv_, nullptr)) {
}

inline driver_guard& driver_guard::operator=(driver_guard&& x) {
    if (drv_ && !drv_->clearing()) {
        --drv_->guard_count_;
    }
    drv_ = std::exchange(x.drv_, nullptr);
    return *this;
}


// free functions

inline void set_clock(cotamer::clock ct) {
    bool is_real = ct == cotamer::clock::real_time;
    driver::global_real_time = is_real;
    driver::current->set_clock(ct);
}

inline system_time_point now() noexcept {
    return driver::current->now();
}

inline steady_time_point steady_now() noexcept {
    return driver::current->steady_now();
}

inline void step_time() noexcept {
    driver::current->step_time();
}

inline void keepalive(event e) {
    driver::current->keepalive(std::move(e));
}

inline event asap() {
    return driver::current->asap();
}

inline event at(steady_time_point t) {
    return driver::current->at(t);
}

inline event at(system_time_point t) {
    return driver::current->at(t);
}

inline event after(duration d) {
    return driver::current->after(d);
}

template <typename Rep, typename Period>
inline event after(const std::chrono::duration<Rep, Period>& d) {
    return driver::current->after(d);
}

inline event readable(const fd& f) {
    return driver::current->file_event(f, fdevent::read);
}

inline event writable(const fd& f) {
    return driver::current->file_event(f, fdevent::write);
}

inline event closed(const fd& f) {
    return driver::current->file_event(f, fdevent::close);
}


// Event combinators

// any(), all()
//    Multi-argument forms create a quorum_event_body. Single-argument forms
//    pass through to make_event (no quorum needed). Zero-argument forms
//    return an already-triggered event.

template <typename E0, typename... Es>
inline event any(E0 e0, Es&&... es) {
    auto q = new detail::quorum_event_body(1, std::forward<E0>(e0), std::forward<Es>(es)...);
    return detail::event_handle(q);
}

template <typename E>
inline event any(E&& e) {
    return detail::make_event(std::forward<E>(e));
}

inline event any() {
    // any() with no arguments returns an already-triggered event.
    // An alternate design would treat any() as an untriggered event (like
    // how false is the identity for logical or).
    return event(nullptr);
}


template <typename... Es>
inline event all(Es&&... es) {
    auto q = new detail::quorum_event_body(sizeof...(Es), std::forward<Es>(es)...);
    return detail::event_handle(q);
}

template <typename E>
inline event all(E&& e) {
    return detail::make_event(std::forward<E>(e));
}

inline event all() {
    return event(nullptr);
}


// attempt(t, e...)
//    Runs a `task<T>` (the first argument) with cancellation (the other
//    arguments). Returns `task<std::optional<T>>`, which is `nullopt` if the
//    task was cancelled.

template <typename T, typename... Es>
task<std::optional<T>> attempt(task<T> t, Es&&... es) {
    if (!t.done()) {
        t.start();
        co_await any(t.completion(), std::forward<Es>(es)...);
    }
    if (t.done()) {
        co_return co_await t;
    }
    // `t` is a parameter, so its destructor will not run immediately upon
    // co_return (it is destroyed with the coroutine state). But we want to
    // destroy it now, because no one cares about its result.
    t.destroy();
    co_return std::nullopt;
}

template <typename T, typename... Es>
task<std::optional<T>> attempt(task<std::optional<T>> t, Es&&... es) {
    if (!t.done()) {
        t.start();
        co_await any(t.completion(), std::forward<Es>(es)...);
    }
    if (t.done()) {
        co_return co_await t;
    }
    t.destroy();
    co_return std::nullopt;
}

template <typename... Es>
task<std::optional<std::monostate>> attempt(task<void> t, Es&&... es) {
    if (!t.done()) {
        t.start();
        co_await any(t.completion(), std::forward<Es>(es)...);
    }
    if (t.done()) {
        co_await t;
        co_return std::monostate{};
    }
    t.destroy();
    co_return std::nullopt;
}

template <bool shared, typename... Es>
task<std::optional<locked_mutex_t<shared>>> attempt(mutex_event<shared> e, Es&&... es) {
    if (!e.triggered()) {
        co_await any(event(e.handle()), std::forward<Es>(es)...);
    }
    if (e.triggered()) {
        co_return locked_mutex_t<shared>{e.mutex()};
    }
    co_return std::nullopt;
}


// first(t, ...)
//    Runs several tasks in parallel, and returns the result of the first
//    to complete, cancelling the others. Returns `std::variant<T...>`.

namespace detail {

// Forward declarations for mixed task/event recursion.
template <typename... Trest>
inline size_t first_done(size_t offset, event&& e0, Trest&&... trest);
template <typename Variant, size_t I, typename... Trest>
inline task<Variant> first_complete(size_t index, event&, Trest&... trest);
template <size_t I, typename... Trest>
inline task<> race_complete(size_t index, event&, Trest&... trest);

inline size_t first_done(size_t offset) {
    return offset;
}

template <typename T, typename... Trest>
inline size_t first_done(size_t offset, task<T>&& t0, Trest&&... trest) {
    if (t0.done()) {
        return offset;
    }
    return first_done(offset + 1, std::forward<Trest>(trest)...);
}

template <typename... Trest>
inline size_t first_done(size_t offset, event&& e0, Trest&&... trest) {
    if (e0.triggered()) {
        return offset;
    }
    return first_done(offset + 1, std::forward<Trest>(trest)...);
}

template <typename T>
inline void destroy_task(task<T>& t) { t.destroy(); }
inline void destroy_task(event&) { }

template <typename T>
inline void start_task(task<T>& t) { t.start(); }
inline void start_task(event&) { }

template <typename T>
inline event make_completion(task<T>& t) { return t.completion(); }
inline event make_completion(event& e) { return e; }

template <typename Variant, size_t I, typename T>
inline task<Variant> first_complete(size_t, task<T>& t0) {
    if constexpr (std::is_void_v<T>) {
        co_await t0;
        co_return Variant{std::in_place_index<I>, std::monostate{}};
    } else {
        co_return Variant{std::in_place_index<I>, co_await t0};
    }
}

template <typename Variant, size_t I>
inline task<Variant> first_complete(size_t, event&) {
    co_return Variant{std::in_place_index<I>, std::monostate{}};
}

template <typename Variant, size_t I, typename T, typename... Trest>
inline task<Variant> first_complete(size_t index, task<T>& t0, Trest&... trest) {
    if (index == I) {
        ((destroy_task(trest)), ...);
        if constexpr (std::is_void_v<T>) {
            co_await t0;
            co_return Variant{std::in_place_index<I>, std::monostate{}};
        } else {
            co_return Variant{std::in_place_index<I>, co_await t0};
        }
    } else {
        t0.destroy();
        co_return co_await first_complete<Variant, I+1>(index, std::forward<Trest&>(trest)...);
    }
}

template <typename Variant, size_t I, typename... Trest>
inline task<Variant> first_complete(size_t index, event&, Trest&... trest) {
    if (index == I) {
        ((destroy_task(trest)), ...);
        co_return Variant{std::in_place_index<I>, std::monostate{}};
    } else {
        co_return co_await first_complete<Variant, I+1>(index, std::forward<Trest&>(trest)...);
    }
}

template <size_t I, typename T>
inline task<T> race_complete(size_t, task<T>& t0) {
    co_return co_await t0;
}

template <size_t I>
inline task<> race_complete(size_t, event&) {
    co_return;
}

template <size_t I, typename T, typename... Trest>
inline task<T> race_complete(size_t index, task<T>& t0, Trest&... trest) {
    if (index == I) {
        ((destroy_task(trest)), ...);
        co_return co_await t0;
    } else {
        t0.destroy();
        co_return co_await race_complete<I+1>(index, std::forward<Trest&>(trest)...);
    }
}

template <size_t I, typename... Trest>
inline task<> race_complete(size_t index, event&, Trest&... trest) {
    if (index == I) {
        ((destroy_task(trest)), ...);
        co_return;
    } else {
        co_return co_await race_complete<I+1>(index, std::forward<Trest&>(trest)...);
    }
}

}

inline task<> first() {
    return task<>();
}

template <typename... Ts>
task<std::variant<task_return_type_t<Ts>...>> first(Ts... ts) {
    using Variant = std::variant<task_return_type_t<Ts>...>;
    size_t done = detail::first_done(0, std::forward<Ts>(ts)...);
    if (done == sizeof...(ts)) {
        ((detail::start_task(ts)), ...);
        co_await any(detail::make_completion(ts)...);
        done = detail::first_done(0, std::forward<Ts>(ts)...);
    }
    co_return co_await detail::first_complete<Variant, 0>(done, std::forward<Ts&>(ts)...);
}

inline task<> race() {
    return task<>();
}

template <typename T, typename... Trest>
task<T> race(task<T> t0, Trest... ts) {
    size_t done = detail::first_done(0, std::forward<task<T>>(t0), std::forward<Trest>(ts)...);
    if (done == 1 + sizeof...(ts)) {
        t0.start();
        ((detail::start_task(ts)), ...);
        co_await any(t0.completion(), detail::make_completion(ts)...);
        done = detail::first_done(0, std::forward<task<T>>(t0), std::forward<Trest>(ts)...);
    }
    co_return co_await detail::race_complete<0>(done, t0, std::forward<Trest&>(ts)...);
}

template <typename... Trest>
task<> race(event e0, Trest... ts) {
    size_t done = detail::first_done(0, std::forward<event>(e0), std::forward<Trest>(ts)...);
    if (done == 1 + sizeof...(ts)) {
        ((detail::start_task(ts)), ...);
        co_await any(e0, detail::make_completion(ts)...);
        done = detail::first_done(0, std::forward<event>(e0), std::forward<Trest>(ts)...);
    }
    co_return co_await detail::race_complete<0>(done, e0, std::forward<Trest&>(ts)...);
}


// driver functions

inline void loop() {
    driver::current->loop();
}

inline bool poll() {
    return driver::current->poll();
}

inline void clear() {
    driver::current->clear();
}

inline uint32_t driver::lock() {
    while (true) {
        uint32_t flags = lock_.load(std::memory_order_relaxed);
        if ((flags & df_lock) == 0
            && lock_.compare_exchange_weak(flags, flags | df_lock, std::memory_order_acquire, std::memory_order_relaxed)) {
            return flags;
        }
        detail::spinlock_hint();
    }
}

inline void driver::unlock(uint32_t flags) {
    lock_.store(flags, std::memory_order_release);
}

inline size_t driver::timer_size() const noexcept {
    return timed_.size();
}


// file descriptor functions

namespace detail {

inline event_handle fd_event_set::watch(int fd, int interest, fd_body* body,
                                        driver* drv) {
    if (fd < 0) {
        return event_handle();
    }
    unsigned ufd = fd;
    if (ufd >= capacity_) {
        hard_ensure(ufd);
    }
    fdrec& fdi = fdrs_[ufd];
    if (fdi.body != body) {
        if (fdi.body) {
            for (int ix = 0; ix != 3; ++ix) {
                if (fdi.ev[ix]) {
                    fdi.ev[ix]->trigger();
                    fdi.ev[ix] = nullptr;
                }
            }
            fdi.body->remove_listener(drv);
        }
        fdi.body = body;
        body->add_listener(drv);
    }
    if (!fdi.ev[interest]) {
        if (fdi.update_link_ == update_clean) {
            fdi.update_link_ = update_link_;
            update_link_ = ufd + 1;
        }
        fdi.ev[interest] = event_handle{new event_body};
    }
    return fdi.ev[interest];
}

inline event_handle fd_event_set::take(int fd, int interest, unsigned epoch) {
    unsigned ufd = fd;
    if (ufd >= capacity_) {
        return event_handle();
    }
    fdrec& fdi = fdrs_[ufd];
    if (!fdi.ev[interest]
        || fdi.ev[interest]->empty()
        || (epoch && epoch != fdi.epoch)) {
        return event_handle();
    }
    if (fdi.update_link_ == update_clean) {
        fdi.update_link_ = update_link_;
        update_link_ = ufd + 1;
    }
    return std::exchange(fdi.ev[interest], nullptr);
}

inline std::optional<std::pair<fd_body*, unsigned>> fd_event_set::check_fd_close(int fd) {
    unsigned ufd = fd;
    if (ufd >= capacity_) {
        return std::nullopt;
    }
    fdrec& fdi = fdrs_[ufd];
    if (!fdi.body || fdi.body->fileno() >= 0) {
        return std::nullopt;
    }
    for (int ix = 0; ix != 3; ++ix) {
        if (fdi.ev[ix]) {
            fdi.ev[ix]->trigger();
            fdi.ev[ix] = nullptr;
        }
    }
    ++fdi.epoch;
    return {{std::exchange(fdi.body, nullptr), fdi.epoch - 1}};
}

inline bool fd_event_set::has_update() const noexcept {
    return update_link_ != update_sentinel;
}

inline std::optional<fd_update> fd_event_set::pop_update() noexcept {
    if (update_link_ == update_sentinel) {
        return std::nullopt;
    }
    unsigned ufd = update_link_ - 1;
    fdrec& fdi = fdrs_[ufd];
    update_link_ = fdi.update_link_;
    fdi.update_link_ = update_clean;
    auto mask = fdi.mask();
    unsigned epoch = fdi.epoch;
    if (!mask) {
        ++fdi.epoch;
    } else if (epoch < user_epoch) { // epoch 1 is reserved for internal FDs
        fdi.epoch = epoch = user_epoch;
    }
    return {{int(ufd), mask, epoch}};
}

inline std::optional<fd_update> fd_event_set::next_nonempty(int fd) const noexcept {
    unsigned ufd = fd + 1;
    if (ufd >= capacity_) {
        return std::nullopt;
    }
    const fdrec* fdrp = fdrs_ + ufd;
    while (true) {
        if (int mask = fdrp->mask()) {
            return {{int(ufd), mask, fdrp->epoch}};
        }
        if (++ufd == capacity_) {
            return std::nullopt;
        }
        ++fdrp;
    }
}

}


// fd methods

inline fd::fd(int fileno)
    : body_(fileno >= 0 ? new detail::fd_body(fileno) : nullptr) {
}

inline fd::fd(const fd& x) noexcept
    : body_(x.body_) {
    if (body_) {
        body_->ref();
    }
}

inline fd::fd(fd&& x) noexcept
    : body_(std::exchange(x.body_, nullptr)) {
}

inline fd& fd::operator=(const fd& x) {
    if (body_ != x.body_) {
        if (x.body_) {
            x.body_->ref();
        }
        if (body_) {
            body_->deref();
        }
        body_ = x.body_;
    }
    return *this;
}

inline fd& fd::operator=(fd&& x) noexcept {
    if (this != &x) {
        if (body_) {
            body_->deref();
        }
        body_ = std::exchange(x.body_, nullptr);
    }
    return *this;
}

inline fd::~fd() {
    if (body_) {
        body_->deref();
    }
}

inline int fd::fileno() const noexcept {
    return body_ ? body_->fileno() : -1;
}

inline bool fd::valid() const noexcept {
    return fileno() >= 0;
}

inline fd::operator bool() const noexcept {
    return fileno() >= 0;
}

inline void fd::close() {
    if (body_) {
        body_->deref_close(false);
    }
}


// mutex functions

template <bool shared>
inline mutex_event<shared>::mutex_event(mutex_type* m)
    : m_(m) {
}

template <bool shared>
inline bool mutex_event<shared>::triggered() const noexcept {
    return !ep_ || ep_->triggered();
}

template <bool shared>
inline auto mutex_event<shared>::mutex() const noexcept -> mutex_type* {
    return m_;
}

template <bool shared>
inline const detail::event_handle& mutex_event<shared>::handle() const& noexcept {
    return ep_;
}

template <bool shared>
inline detail::event_handle&& mutex_event<shared>::handle() && noexcept {
    return std::move(ep_);
}


inline mutex_event<false> mutex::lock() {
    mutex_event<false> me(this);
    lock_impl(false, me.ep_);
    return me;
}

inline bool mutex::try_lock() {
    latch_type l = 0;
    return latch_.compare_exchange_strong(l, mf_lock_excl, std::memory_order_acquire, std::memory_order_relaxed);
}

inline void mutex::unlock() {
    unsigned l = latch_.load(std::memory_order_relaxed);
    // Fast-path unlock: not latched, no waiter.
    if (!(l & (mf_latch | mfm_next))
        && latch_.compare_exchange_strong(l, l - mf_lock_excl, std::memory_order_release, std::memory_order_relaxed)) {
        return;
    }
    unlock_impl(false);
}

inline mutex_event<true> mutex::lock_shared() {
    mutex_event<true> me(this);
    lock_impl(true, me.ep_);
    return me;
}

inline bool mutex::try_lock_shared() {
    latch_type l = latch_.load(std::memory_order_relaxed);
    if (l & (mf_latch | mfm_next | mf_lock_excl)) {
        return false;
    }
    return latch_.compare_exchange_strong(l, l + mf_lock_shared, std::memory_order_acquire, std::memory_order_relaxed);
}

inline void mutex::unlock_shared() {
    unsigned l = latch_.load(std::memory_order_relaxed);
    // Fast-path unlock: not latched, no shared waiter, and either no
    // exclusive waiter or the shared lock is still held.
    if (!(l & (mf_latch | mf_next_shared))
        && (!(l & mf_next_excl) || l >= 2 * mf_lock_shared)
        && latch_.compare_exchange_strong(l, l - mf_lock_shared, std::memory_order_release, std::memory_order_relaxed)) {
        return;
    }
    unlock_impl(true);
}

inline bool mutex::waiter_shared(const detail::event_handle& eh) const noexcept {
    return eh.get()->relaxed_flags() & detail::ef_user;
}

inline auto mutex::latch() -> latch_type {
    while (true) {
        latch_type l = latch_.load(std::memory_order_relaxed);
        if (!(l & mf_latch)
            && latch_.compare_exchange_weak(l, l | mf_latch, std::memory_order_acquire, std::memory_order_relaxed)) {
            return l & ~mfm_next;
        }
        detail::spinlock_hint();
    }
}

inline void mutex::unlatch(latch_type l) {
    if (!waiters_.empty()) {
        l += waiter_shared(waiters_.front()) ? mf_next_shared : mf_next_excl;
    }
    latch_.store(l, std::memory_order_release);
}


inline unique_lock::unique_lock(locked_mutex_t<false> t) noexcept
    : m_(t.m), owned_(true) {
}

inline unique_lock::~unique_lock() {
    if (owned_) {
        m_->unlock();
    }
}

inline unique_lock::unique_lock(unique_lock&& x) noexcept
    : m_(std::exchange(x.m_, nullptr)), owned_(std::exchange(x.owned_, false)) {
}

inline unique_lock::unique_lock(mutex_type& m, std::defer_lock_t) noexcept
    : m_(&m), owned_(false) {
}

inline unique_lock::unique_lock(mutex_type& m, std::try_to_lock_t) noexcept
    : m_(&m), owned_(m.try_lock()) {
}

inline unique_lock::unique_lock(mutex_type& m, std::adopt_lock_t) noexcept
    : m_(&m), owned_(true) {
}

inline unique_lock& unique_lock::operator=(unique_lock&& x) noexcept {
    if (owned_) {
        m_->unlock();
    }
    m_ = std::exchange(x.m_, nullptr);
    owned_ = std::exchange(x.owned_, false);
    return *this;
}

inline void unique_lock::swap(unique_lock& x) noexcept {
    if (&x != this) {
        std::swap(m_, x.m_);
        std::swap(owned_, x.owned_);
    }
}

inline task<> unique_lock::lock() {
    if (!m_) {
        throw std::system_error(std::make_error_code(std::errc::operation_not_permitted));
    } else if (owned_) {
        throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur));
    }
    co_await m_->lock();
    owned_ = true;
}

inline bool unique_lock::try_lock() {
    if (!m_) {
        throw std::system_error(std::make_error_code(std::errc::operation_not_permitted));
    } else if (owned_) {
        throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur));
    }
    owned_ = m_->try_lock();
    return owned_;
}

inline void unique_lock::unlock() {
    if (!owned_) {
        throw std::system_error(std::make_error_code(std::errc::operation_not_permitted));
    }
    m_->unlock();
    owned_ = false;
}

inline auto unique_lock::release() noexcept -> mutex_type* {
    owned_ = false;
    return std::exchange(m_, nullptr);
}


inline shared_lock::shared_lock(locked_mutex_t<true> t) noexcept
    : m_(t.m), owned_(true) {
}

inline shared_lock::~shared_lock() {
    if (owned_) {
        m_->unlock_shared();
    }
}

inline shared_lock::shared_lock(mutex_type& m, std::defer_lock_t) noexcept
    : m_(&m), owned_(false) {
}

inline shared_lock::shared_lock(mutex_type& m, std::try_to_lock_t) noexcept
    : m_(&m), owned_(m.try_lock_shared()) {
}

inline shared_lock::shared_lock(mutex_type& m, std::adopt_lock_t) noexcept
    : m_(&m), owned_(true) {
}

inline shared_lock::shared_lock(shared_lock&& x) noexcept
    : m_(std::exchange(x.m_, nullptr)), owned_(std::exchange(x.owned_, false)) {
}

inline shared_lock& shared_lock::operator=(shared_lock&& x) noexcept {
    if (owned_) {
        m_->unlock_shared();
    }
    m_ = std::exchange(x.m_, nullptr);
    owned_ = std::exchange(x.owned_, false);
    return *this;
}

inline void shared_lock::swap(shared_lock& x) noexcept {
    if (&x != this) {
        std::swap(m_, x.m_);
        std::swap(owned_, x.owned_);
    }
}

inline task<> shared_lock::lock() {
    if (!m_) {
        throw std::system_error(std::make_error_code(std::errc::operation_not_permitted));
    } else if (owned_) {
        throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur));
    }
    co_await m_->lock_shared();
    owned_ = true;
}

inline bool shared_lock::try_lock() {
    if (!m_) {
        throw std::system_error(std::make_error_code(std::errc::operation_not_permitted));
    } else if (owned_) {
        throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur));
    }
    owned_ = m_->try_lock_shared();
    return owned_;
}

inline void shared_lock::unlock() {
    if (!owned_) {
        throw std::system_error(std::make_error_code(std::errc::operation_not_permitted));
    }
    m_->unlock_shared();
    owned_ = false;
}

inline auto shared_lock::release() noexcept -> mutex_type* {
    owned_ = false;
    return std::exchange(m_, nullptr);
}

namespace detail {
template <typename T>
inline task_mutex_event_awaiter<T, false> task_promise<T>::await_transform(mutex& m) {
    return await_transform(m.lock());
}

inline task_mutex_event_awaiter<void, false> task_promise<void>::await_transform(mutex& m) {
    return await_transform(m.lock());
}
}

}
