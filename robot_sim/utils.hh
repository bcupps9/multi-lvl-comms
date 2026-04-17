#pragma once
#include <chrono>
#include <random>
#include <string>

template <typename RNG>
inline RNG randomly_seeded() {
    std::random_device device;
    std::array<unsigned, RNG::state_size> seed_data;
    for (unsigned i = 0; i != RNG::state_size; ++i)
        seed_data[i] = device();
    std::seed_seq seq(seed_data.begin(), seed_data.end());
    return RNG(seq);
}

template <typename T>
struct is_duration_t : public std::false_type {};
template <typename Rep, typename Period>
struct is_duration_t<std::chrono::duration<Rep, Period>> : public std::true_type {};
template <typename T>
concept durational = is_duration_t<std::remove_cvref_t<T>>::value;
