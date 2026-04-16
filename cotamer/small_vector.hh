#pragma once
#include <cstddef>
#include <memory>

template <typename T, size_t N>
struct small_vector {
    static_assert(N > 0 && N <= 128);

    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using iterator = T*;
    using const_iterator = const T*;

    small_vector() = default;
    small_vector(const small_vector<T, N>&) = delete;
    small_vector(small_vector<T, N>&& x)
        : sz_(x.sz_), cap_(x.cap_) {
        if (x.cap_ > N) {
            u_.out = x.u_.out;
            x.sz_ = x.cap_ = 0;
        } else {
            std::uninitialized_move_n(x.begin(), x.sz_, begin());
        }
    }
    small_vector<T, N>& operator=(const small_vector<T, N>&) = delete;
    small_vector<T, N>& operator=(small_vector<T, N>&&) = delete;
    ~small_vector() {
        clear_capacity();
    }

    size_t size() const {
        return sz_;
    }
    bool empty() const {
        return size() == 0;
    }
    void clear() {
        std::destroy_n(begin(), sz_);
        sz_ = 0;
    }
    void clear_capacity() {
        std::destroy_n(begin(), sz_);
        if (cap_ > N) {
            std::allocator<T> alloc;
            alloc.deallocate(u_.out, cap_);
        }
        sz_ = cap_ = 0;
    }
    iterator erase(const_iterator first, const_iterator last) {
        if (first != last) {
            auto eit = end(), delsz = last - first;
            std::move(const_cast<iterator>(last), eit, const_cast<iterator>(first));
            sz_ -= delsz;
            std::destroy(end(), eit);
        }
        return const_cast<iterator>(first);
    }
    void truncate(const_iterator first) {
        std::destroy(const_cast<iterator>(first), end());
        sz_ = first - begin();
    }

    T* begin() {
        return cap_ <= N ? reinterpret_cast<T*>(u_.inbuf) : u_.out;
    }
    const T* begin() const {
        return cap_ <= N ? reinterpret_cast<const T*>(u_.inbuf) : u_.out;
    }
    T* end() {
        return begin() + size();
    }
    const T* end() const {
        return begin() + size();
    }

    T* push_space() {
        if (cap_ == 0) {
            cap_ = N;
        }
        if (sz_ < cap_) {
            return end();
        }
        std::allocator<T> alloc;
        T* newptr = alloc.allocate(cap_ * 2);
        std::uninitialized_move_n(begin(), sz_, newptr);
        std::destroy_n(begin(), sz_);
        if (cap_ > N) {
            alloc.deallocate(u_.out, cap_);
        }
        u_.out = newptr;
        cap_ *= 2;
        return u_.out + sz_;
    }

    T& front() {
        return *begin();
    }
    const T& front() const {
        return *begin();
    }
    T& back() {
        return *(end() - 1);
    }
    const T& back() const {
        return *(end() - 1);
    }
    T& operator[](size_type i) {
        return begin()[i];
    }
    const T& operator[](size_type i) const {
        return begin()[i];
    }

    void push_back(const T& v) {
        std::construct_at(push_space(), v);
        ++sz_;
    }
    void push_back(T&& v) {
        std::construct_at(push_space(), std::move(v));
        ++sz_;
    }
    template <typename... Args>
    T& emplace_back(Args&&... args) {
        auto sp = push_space();
        std::construct_at(sp, std::forward<Args>(args)...);
        ++sz_;
        return *sp;
    }
    void pop_back() {
        std::destroy_at(end() - 1);
        --sz_;
    }

private:
    uint32_t sz_ = 0;
    uint32_t cap_ = N;
    union {
        alignas(T) std::byte inbuf[sizeof(T) * N];
        T* out;
    } u_;
};
