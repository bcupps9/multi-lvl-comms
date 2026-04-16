#pragma once
#include "cotamer/circular_int.hh"
#include <chrono>

template <typename T>
struct timer_heap_traits {
    struct empty_type {
        bool operator()(const T& x) const {
            return x.empty();
        }
    };
    using time_point_type = std::chrono::steady_clock::time_point;
    static constexpr int arity = 4;
};

template <typename T>
struct timer_heap {
    using traits_type = timer_heap_traits<T>;
    static constexpr int arity = traits_type::arity;
    using empty_type = typename traits_type::empty_type;
    using time_point_type = typename traits_type::time_point_type;
    using value_type = T;
    using reference = T&;
    using size_type = unsigned;
    static_assert(arity >= 2);

    timer_heap() = default;
    timer_heap(const timer_heap&) = delete;
    timer_heap(timer_heap&&) = delete;
    timer_heap& operator=(const timer_heap&) = delete;
    timer_heap& operator=(timer_heap&&) = delete;
    inline ~timer_heap();

    inline bool empty() const noexcept;
    inline unsigned size() const noexcept;
    inline time_point_type top_time() const;
    inline T& top() &;
    inline const T& top() const&;
    void emplace(time_point_type t, T&& value);
    inline void pop();
    inline void cull();
    void clear();

  private:
    struct element {
        time_point_type when;
        circular_int<unsigned> order;
        value_type value;

        inline bool operator<(const element &x) const noexcept;
    };

    element* es_ = nullptr;
    unsigned size_ = 0;
    unsigned capacity_ = 0;
    unsigned order_ = 0;         // next `order` to insert
    unsigned cull_rand_ = 8173;  // random seed for `cull`

    static inline unsigned heap_parent(unsigned i) noexcept;
    static inline unsigned heap_first_child(unsigned i) noexcept;
    inline unsigned heap_last_child(unsigned i) const noexcept;
    void hard_cull(unsigned pos);
    void expand();
};


template <typename T>
inline timer_heap<T>::~timer_heap() {
    std::destroy_n(es_, size_);
    std::allocator<element> alloc;
    alloc.deallocate(es_, capacity_);
}

template <typename T>
inline bool timer_heap<T>::empty() const noexcept {
    return size_ == 0;
}

template <typename T>
inline unsigned timer_heap<T>::size() const noexcept {
    return size_;
}

template <typename T>
inline auto timer_heap<T>::top_time() const -> time_point_type {
    assert(size_ != 0);
    return es_[0].when;
}

template <typename T>
inline T& timer_heap<T>::top() & {
    assert(size_ != 0);
    return es_[0].value;
}

template <typename T>
inline const T& timer_heap<T>::top() const& {
    assert(size_ != 0);
    return es_[0].value;
}

template <typename T>
inline void timer_heap<T>::cull() {
    while (size_ != 0 && empty_type{}(es_[0].value)) {
        hard_cull(0);
    }
}

template <typename T>
inline bool timer_heap<T>::element::operator<(const element &x) const noexcept {
    auto cmp = when <=> x.when;
    return cmp < 0 || (cmp == 0 && order < x.order);
}

template <typename T>
inline unsigned timer_heap<T>::heap_parent(unsigned i) noexcept {
    return (i - (arity == 2)) / arity;
}

template <typename T>
inline unsigned timer_heap<T>::heap_first_child(unsigned i) noexcept {
    return i * arity + (arity == 2 || i == 0);
}

template <typename T>
inline unsigned timer_heap<T>::heap_last_child(unsigned i) const noexcept {
    unsigned p = i * arity + arity + (arity == 2);
    return p < size_ ? p : size_;
}

template <typename T>
inline void timer_heap<T>::pop() {
    hard_cull(0);
}

template <typename T>
void timer_heap<T>::clear() {
    std::destroy_n(es_, size_);
    size_ = 0;
}

template <typename T>
void timer_heap<T>::expand() {
    unsigned ncap = (capacity_ ? (capacity_ * 4) + 3 : 31);
    std::allocator<element> alloc;
    element* nes = alloc.allocate(ncap);
    if (es_) {
        std::uninitialized_move_n(es_, size_, nes);
        std::destroy_n(es_, size_);
        alloc.deallocate(es_, capacity_);
    }
    es_ = nes;
    capacity_ = ncap;
}

template <typename T>
void timer_heap<T>::emplace(time_point_type when, T&& value) {
    using std::swap;

    // If heap is largish, check to see if a random element is empty.
    // If it's empty, remove it, and look for another empty element.
    // This should help keep the timer heap small even if we set many
    // more timers than get a chance to fire.
    while (size_ >= 32) {
        unsigned pos = cull_rand_ % size_;
        cull_rand_ = cull_rand_ * 1664525 + 1013904223U; // Numerical Recipes LCG
        if (!empty_type{}(es_[pos].value)) {
            break;
        }
        hard_cull(pos);
    }

    // Append new trec
    unsigned pos = size_;
    if (pos == capacity_) {
        expand();
    }
    std::construct_at(es_ + pos, when, ++order_, std::move(value));
    ++size_;

    // Swap trec to proper position in heap
    while (pos != 0) {
        unsigned p = heap_parent(pos);
        if (!(es_[pos] < es_[p])) {
            break;
        }
        swap(es_[pos], es_[p]);
        pos = p;
    }
}

template <typename T>
void timer_heap<T>::hard_cull(unsigned pos) {
    using std::swap;
    assert(size_ != 0);

    --size_;
    if (pos == size_) {
        std::destroy_at(es_ + pos);
        return;
    }
    swap(es_[size_], es_[pos]);
    std::destroy_at(es_ + size_);

    if (pos == 0 || !(es_[pos] < es_[heap_parent(pos)])) {
        while (true) {
            unsigned smallest = pos;
            for (unsigned t = heap_first_child(pos);
                 t < heap_last_child(pos);
                 ++t) {
                if (es_[t] < es_[smallest]) {
                    smallest = t;
                }
            }
            if (smallest == pos) {
                break;
            }
            swap(es_[pos], es_[smallest]);
            pos = smallest;
        }
    } else {
        do {
            unsigned p = heap_parent(pos);
            swap(es_[pos], es_[p]);
            pos = p;
        } while (pos && es_[pos] < es_[heap_parent(pos)]);
    }
}
