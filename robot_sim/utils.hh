#pragma once
#include <charconv>
#include <chrono>
#include <random>
#include <string>
#include <system_error>
#include <getopt.h>

// - perform std::from_chars on `s`; all of `s` must be parsed
template <typename T, typename... Args>
inline std::errc from_str_chars(const std::string& s, T& value, Args... rest) {
    auto [next, ec] = std::from_chars(s.data(), s.data() + s.size(), value, rest...);
    if (next != s.data() + s.size()) {
        ec = std::errc::invalid_argument;
    }
    return ec;
}

// - perform std::from_chars on `s`; all of `s` must be parsed. return parsed
//   value, or throw exception if parsing fails
template <typename T, typename... Args>
inline T from_str_chars(const std::string& s, Args... rest) {
    T value;
    auto ec = from_str_chars(s, value, rest...);
    if (ec != std::errc()) {
        throw std::invalid_argument(std::make_error_code(ec).message());
    }
    return value;
}


// - construct a randomly-seeded generator
template <typename RNG>
inline RNG randomly_seeded() {
    std::random_device device;
    std::array<unsigned, RNG::state_size> seed_data;
    for (unsigned i = 0; i != RNG::state_size; ++i) {
        seed_data[i] = device();
    }
    std::seed_seq seq(seed_data.begin(), seed_data.end());
    return RNG(seq);
}


// - compute a `short_options` string corresponding to `longopts`
std::string short_options_for(const struct option* longopts);


template <typename T>
struct is_duration_t : public std::false_type {};
template <typename Rep, typename Period>
struct is_duration_t<std::chrono::duration<Rep, Period>> : public std::true_type {};
template <typename T>
concept durational = is_duration_t<std::remove_cvref_t<T>>::value;
