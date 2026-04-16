#pragma once
#include "utils.hh"

// random_source.hh
//    Source of randomness, with helpers for different distributions.

class random_source {
public:
    using random_engine_type = std::mt19937_64;

    inline random_source();
    explicit inline random_source(const random_engine_type&);
    random_source(const random_source&) = delete;
    random_source(random_source&&) = delete;
    random_source& operator=(random_source&) = delete;
    random_source& operator=(random_source&&) = delete;

    random_engine_type& engine() { return engine_; }
    inline void seed(random_engine_type::result_type value);

    // - returning bool
    inline bool coin_flip();     // returns true with P = 0.5
    inline bool coin_flip(double probability_of_true);
    // - uniform distributions: select from a list of items or choose within a
    //   range
    template <typename U>
    inline U uniform(std::initializer_list<U> list);
    template <std::integral I>
    inline I uniform(I min, I max);
    template <std::integral I, std::integral J>
    inline std::common_type_t<I, J> uniform(I min, J max);
    template <std::floating_point FP>
    inline FP uniform(FP min, FP max);
    template <durational D>
    inline D uniform(D min, D max);
    template <durational D1, durational D2>
    inline std::common_type_t<D1, D2> uniform(D1 min, D2 max);
    // - exponential distributions: useful for network delay, which can have
    //   occasional long tails
    template <std::floating_point FP>
    inline FP exponential(FP mean);
    template <durational D>
    inline D exponential(D mean);
    // - normal (Gaussian) distributions: useful for jitter around a mean
    //   delay; the duration overload clamps negative results to zero
    template <std::floating_point FP>
    inline FP normal(FP mean, FP stddev);
    template <durational D>
    inline D normal(D mean, D stddev);
    template <durational D1, durational D2>
    inline std::common_type_t<D1, D2> normal(D1 mean, D2 stddev);
    // - lower-case hexadecimal string with `n` hex digits
    inline std::string uniform_hex(unsigned n);

private:
    random_engine_type engine_;
};


inline random_source::random_source()
    : engine_(randomly_seeded<random_engine_type>()) {
}

inline random_source::random_source(const random_engine_type& engine)
    : engine_(engine) {
}

inline void random_source::seed(random_engine_type::result_type value) {
    engine_.seed(value);
}

inline bool random_source::coin_flip() {
    return std::uniform_int_distribution<int>(0, 1)(engine_);
}

inline bool random_source::coin_flip(double probability_of_true) {
    if (probability_of_true <= 0.0) {
        return false;
    } else if (probability_of_true >= 1.0) {
        return true;
    }
    constexpr uint64_t one = uint64_t(1) << 53;
    auto val = std::uniform_int_distribution<uint64_t>(0, one - 1)(engine_);
    return val < static_cast<uint64_t>(probability_of_true * one);
}

template <typename U>
inline U random_source::uniform(std::initializer_list<U> list) {
    assert(list.size() > 0);
    auto idx = std::uniform_int_distribution<size_t>(0, list.size() - 1)(engine_);
    return list.begin()[idx];
}

template <std::integral I>
inline I random_source::uniform(I min, I max) {
    if (min >= max) {
        return min;
    }
    return std::uniform_int_distribution<I>(min, max)(engine_);
}

template <std::integral I, std::integral J>
inline std::common_type_t<I, J> random_source::uniform(I min, J max) {
    if (min >= max) {
        return min;
    }
    using T = std::common_type_t<I, J>;
    return std::uniform_int_distribution<T>(T(min), T(max))(engine_);
}

template <std::floating_point FP>
inline FP random_source::uniform(FP min, FP max) {
    if (min >= max) {
        return min;
    }
    return std::uniform_real_distribution<FP>(min, max)(engine_);
}

template <durational D>
inline D random_source::uniform(D min, D max) {
    if (min >= max) {
        return min;
    }
    std::uniform_int_distribution<typename D::rep> dist(min.count(), max.count());
    return D(dist(engine_));
}

template <durational D1, durational D2>
inline std::common_type_t<D1, D2> random_source::uniform(D1 min, D2 max) {
    using D = std::common_type_t<D1, D2>;
    return uniform(D(min), D(max));
}

template <std::floating_point FP>
inline FP random_source::exponential(FP mean) {
    return std::exponential_distribution<FP>(1.0 / mean)(engine_);
}

template <durational D>
inline D random_source::exponential(D mean) {
    std::exponential_distribution<double> dist(1.0 / mean.count());
    return D(static_cast<typename D::rep>(dist(engine_)));
}

template <std::floating_point FP>
inline FP random_source::normal(FP mean, FP stddev) {
    return std::normal_distribution<FP>(mean, stddev)(engine_);
}

template <durational D>
inline D random_source::normal(D mean, D stddev) {
    std::normal_distribution<double> dist(mean.count(), stddev.count());
    return D(static_cast<typename D::rep>(std::max(dist(engine_), 0.0)));
}

template <durational D1, durational D2>
inline std::common_type_t<D1, D2> random_source::normal(D1 mean, D2 stddev) {
    using D = typename std::common_type_t<D1, D2>;
    return normal(D(mean), D(stddev));
}

inline std::string random_source::uniform_hex(unsigned n) {
    std::string s;
    while (n >= 8) {
        s += std::format("{:08x}", uniform(0U, 0xFFFFFFFFU));
        n -= 8;
    }
    if (n > 0) {
        s += std::format("{:0{}x}", uniform(0U, (1U << (n * 4)) - 1), n);
    }
    return s;
}
