#pragma once
#include "cotamer/cotamer.hh"
#include "random_source.hh"
#include <print>

// netsim.hh
//    Network simulator for messages of type T.
//
//    * channel<T> - represents a link between two servers
//    * port<T> - represents a receiving port on a server


namespace netsim {
namespace cot = cotamer;
using namespace std::chrono_literals;
template <typename T> struct channel;
template <typename T> struct message_traits;
extern const std::string disconnected_id;

template <typename T, typename U>
concept smart_pointer_to =
    std::same_as<T, std::unique_ptr<U>> || std::same_as<T, std::shared_ptr<U>>;


// port<T>
//    An input interface on a server.
//
//    The critical function is `receive`, which receives a message. `receive`
//    is a coroutine; it suspends processing until a message becomes available.

template <typename T>
struct port {
    using message_type = T;
    using message_traits_type = message_traits<T>;

    inline port(random_source&);
    inline port(random_source&, const std::string& id);
    ~port() {
        // wake up any `receive` coroutines so that the driver cleanup code
        // will free their memory
        receivable_.trigger();
    }
    port(const port<T>&) = delete;
    port(port<T>&&) = delete;
    port<T>& operator=(const port<T>&) = delete;
    port<T>& operator=(port<T>&&) = delete;

    const std::string& id() const noexcept { return id_; }
    void set_id(const std::string& id) noexcept { id_ = id; }

    bool verbose() const noexcept { return verbose_; }
    void set_verbose(bool verbose) noexcept { verbose_ = verbose; }

    // receive a message on this port
    cot::task<T> receive();
    cot::task<std::pair<T, std::string>> receive_with_id();


private:
    friend struct channel<T>;

    std::deque<std::pair<message_type, std::string>> messageq_;
    cot::event receivable_;
    cot::duration receive_delay_ = 1ms;  // time before receiver can continue

    random_source& randomness_;
    std::string id_;
    bool verbose_ = false;
};


// channel<T>
//    A link from one server to another.
//
//    The critical function is `send`, which sends a message on the link to
//    the destination server. `send` is a coroutine; it returns when this
//    server (`source`) can send another message. `send` also starts a detached
//    coroutine to actually deliver the message after some delay.

template <typename T>
struct channel {
    using message_type = T;
    using message_traits_type = message_traits<T>;

    inline channel(random_source&);
    inline channel(random_source&, const std::string& source_id);
    inline channel(port<T>&);
    inline channel(port<T>&, const std::string& source_id);
    channel(const channel<T>&) = delete;
    channel(channel<T>&&) = delete;
    channel<T>& operator=(const channel<T>&) = delete;
    channel<T>& operator=(channel<T>&&) = delete;

    void connect(port<T>&);
    void disconnect();

    const std::string& source_id() const noexcept { return source_id_; }
    void set_source_id(const std::string& id) noexcept { source_id_ = id; }
    inline const std::string& destination_id() const noexcept;

    bool verbose() const noexcept { return verbose_; }
    void set_verbose(bool verbose) noexcept { verbose_ = verbose; }

    double loss() const noexcept { return loss_; }
    void set_loss(double loss) noexcept { loss_ = loss; }

    // send a message on this channel
    cot::task<> send(message_type m);

    // convenience: when `message_type` is a smart pointer, allow
    // `send(ELEMENT_TYPE)`
    template <typename U>
    requires smart_pointer_to<message_type, U>
    cot::task<> send(U value) {
        if constexpr (std::same_as<message_type, std::unique_ptr<U>>) {
            return send(std::make_unique<U>(std::move(value)));
        }
        return send(std::make_shared<U>(std::move(value)));
    }


private:
    port<T>* destination_ = nullptr;
    unsigned destination_version_ = 0;
    cot::duration link_delay_ = 5ms; // time for message to arrive
    cot::duration send_delay_ = 1ms; // time before sender can continue
    double loss_ = 0.0;

    random_source& randomness_;
    std::string source_id_;
    bool verbose_ = false;

    cot::task<> send_after(cot::duration, message_type);

    inline random_source& randomness() { return destination_->randomness(); }
};


// The central coroutines
// ======================

// channel<T>::send(m)
//    Send a message from source() to destination().

template <typename T>
cot::task<> channel<T>::send(message_type m) {
    if (destination_ && !randomness_.coin_flip(loss_)) {
        if (verbose_) {
            std::print("{}: {} → {} {}\n", cot::now(),
                       source_id(), destination_id(),
                       message_traits_type::print_transform(m));
        }

        // after `link_delay_`, place the message in the receiver’s queue
        send_after(link_delay_, std::move(m)).detach();
    }

    // sending a message takes time
    co_await cot::after(send_delay_);
}


// channel<T>::send_after(delay, m)
//    Delay for `delay`, then enqueue `m` on the destination port.

template <typename T>
cot::task<> channel<T>::send_after(cot::duration delay, message_type m) {
    unsigned destination_version = destination_version_;

    co_await cot::after(delay);

    // drop if disconnected
    if (!destination_ || destination_version != destination_version_) {
        co_return;
    }

    // record this message in the destination message queue
    destination_->messageq_.emplace_back(std::move(m), source_id_);

    // wake up a blocked receiver
    destination_->receivable_.trigger();
}


// port<T>::receive()
//    Suspend until a message is available, then dequeue and return it.

template <typename T>
auto port<T>::receive_with_id() -> cot::task<std::pair<T, std::string>> {
    // sleep until there’s a message
    while (messageq_.empty()) {
        cot::driver_guard guard;
        // Register an event that senders will trigger on delivery.
        // Need a new one every time because events are one-shot.
        co_await receivable_.arm();
        co_await cot::after(receive_delay_);
    }

    auto m = std::move(messageq_.front());
    messageq_.pop_front();

    if (verbose_) {
        std::print("{}: {} ← {} {}\n", cot::now(), id(), m.second,
                   message_traits_type::print_transform(m.first));
    }

    co_return std::move(m);
}

template <typename T>
cot::task<T> port<T>::receive() {
    co_return (co_await receive_with_id()).first;
}


// Construction and initialization functions

template <typename T>
inline port<T>::port(random_source& randomness)
    : randomness_(randomness), id_("port" + randomness.uniform_hex(4)) {
}

template <typename T>
inline port<T>::port(random_source& randomness, const std::string& id)
    : randomness_(randomness), id_(id) {
}

template <typename T>
inline channel<T>::channel(random_source& randomness)
    : destination_(nullptr), randomness_(randomness),
      source_id_("chan" + randomness_.uniform_hex(4)) {
}

template <typename T>
inline channel<T>::channel(random_source& randomness, const std::string& source_id)
    : destination_(nullptr), randomness_(randomness), source_id_(source_id) {
}

template <typename T>
inline channel<T>::channel(port<T>& port)
    : destination_(&port), randomness_(port.randomness_),
      source_id_("chan" + randomness_.uniform_hex(4)) {
}

template <typename T>
inline channel<T>::channel(port<T>& port, const std::string& source_id)
    : destination_(&port), randomness_(port.randomness_), source_id_(source_id) {
}

template <typename T>
void channel<T>::connect(port<T>& destination) {
    destination_ = &destination;
    ++destination_version_;
}

template <typename T>
void channel<T>::disconnect() {
    destination_ = nullptr;
    ++destination_version_;
}

template <typename T>
const std::string& channel<T>::destination_id() const noexcept {
    return destination_ ? destination_->id() : disconnected_id;
}


// message_traits<T>
//    This template lets us change the behavior of network functions based on
//    message type. We provide specializations that allow you to print
//    messages that are stored as pointers.

template <typename T>
struct message_traits {
    static decltype(auto) print_transform(const T& x) {
        if constexpr (std::formattable<T, char>) {
            return x;
        } else {
            return "<message>";
        }
    }
};

template <typename T>
struct message_traits<T*> {
    static inline const T& print_transform(T* x) {
        return message_traits<T>::print_transform(*x);
    }
};

template <typename T>
struct message_traits<std::unique_ptr<T>> {
    static inline const T& print_transform(const std::unique_ptr<T>& x) {
        return message_traits<T>::print_transform(*x);
    }
};

template <typename T>
struct message_traits<std::shared_ptr<T>> {
    static inline const T& print_transform(const std::shared_ptr<T>& x) {
        return message_traits<T>::print_transform(*x);
    }
};

}
