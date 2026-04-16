#pragma once
#include <format>
#include <iostream>
#include <type_traits>

template <typename T = unsigned long>
class circular_int {
  public:
    using value_type = typename std::make_unsigned<T>::type;
    using difference_type = typename std::make_signed<T>::type;

    circular_int()
        : v_() {
    }
    circular_int(T x)
        : v_(x) {
    }

    value_type value() const {
        return v_;
    }

    circular_int<T> &operator++() {
        ++v_;
        return *this;
    }
    circular_int<T> operator++(int) {
        ++v_;
        return circular_int<T>(v_ - 1);
    }
    circular_int<T> &operator--() {
        --v_;
        return *this;
    }
    circular_int<T> operator--(int) {
        --v_;
        return circular_int<T>(v_ + 1);
    }
    circular_int<T> &operator+=(unsigned x) {
        v_ += x;
        return *this;
    }
    circular_int<T> &operator+=(int x) {
        v_ += x;
        return *this;
    }
    circular_int<T> &operator-=(unsigned x) {
        v_ -= x;
        return *this;
    }
    circular_int<T> &operator-=(int x) {
        v_ -= x;
        return *this;
    }

    explicit operator bool() const {
        return v_ != 0;
    }
    bool operator!() const {
        return v_ == 0;
    }
    operator value_type() const {
        return v_;
    }

    circular_int<T> operator+(unsigned x) const {
        return circular_int<T>(v_ + x);
    }
    circular_int<T> operator+(int x) const {
        return circular_int<T>(v_ + x);
    }
    circular_int<T> operator+(unsigned long x) const {
        return circular_int<T>(v_ + x);
    }
    circular_int<T> operator+(long x) const {
        return circular_int<T>(v_ + x);
    }
    circular_int<T> operator+(unsigned long long x) const {
        return circular_int<T>(v_ + x);
    }
    circular_int<T> operator+(long long x) const {
        return circular_int<T>(v_ + x);
    }
    circular_int<T> next_nonzero() const {
        value_type v = v_ + 1;
        return circular_int<T>(v + !v);
    }
    static value_type next_nonzero(value_type x) {
        ++x;
        return x + !x;
    }
    circular_int<T> operator-(unsigned x) const {
        return circular_int<T>(v_ - x);
    }
    circular_int<T> operator-(int x) const {
        return circular_int<T>(v_ - x);
    }
    circular_int<T> operator-(unsigned long x) const {
        return circular_int<T>(v_ - x);
    }
    circular_int<T> operator-(long x) const {
        return circular_int<T>(v_ - x);
    }
    circular_int<T> operator-(unsigned long long x) const {
        return circular_int<T>(v_ - x);
    }
    circular_int<T> operator-(long long x) const {
        return circular_int<T>(v_ - x);
    }
    difference_type operator-(circular_int<T> x) const {
        return v_ - x.v_;
    }

    value_type operator%(int x) const {
        return v_ % x;
    }
    value_type operator%(unsigned x) const {
        return v_ % x;
    }
    value_type operator%(long x) const {
        return v_ % x;
    }
    value_type operator%(unsigned long x) const {
        return v_ % x;
    }


    bool operator==(circular_int<T> x) const {
        return v_ == x.v_;
    }
    bool operator!=(circular_int<T> x) const {
        return !(*this == x);
    }
    auto operator<=>(circular_int<T> x) const {
        if (v_ == x.v_) {
            return std::strong_ordering::equal;
        } else if (difference_type(v_ - x.v_) < 0) {
            return std::strong_ordering::less;
        }
        return std::strong_ordering::greater;
    }
    static bool less(value_type a, value_type b) {
        return difference_type(a - b) < 0;
    }
    static bool less_equal(value_type a, value_type b) {
        return difference_type(a - b) <= 0;
    }

  private:
    value_type v_;
};


// Support iostreams, std::format, and std::print

template <typename T>
inline std::ostream& operator<<(std::ostream& str, circular_int<T> x) {
    return str << x.value();
}

template <typename T, typename CharT>
struct std::formatter<circular_int<T>, CharT>
    : std::formatter<typename circular_int<T>::value_type, CharT> {
    template <typename FormatContext>
    auto format(circular_int<T> x, FormatContext& ctx) const {
        return std::formatter<typename circular_int<T>::value_type, CharT>::format(x.value(), ctx);
    }
};
