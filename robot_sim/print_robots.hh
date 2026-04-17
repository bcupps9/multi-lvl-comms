#pragma once
// Compatibility shim: GCC 13 has <format> but not <print>.
#include <format>
#include <cstdio>
#include <string>

namespace std {

template <typename... Args>
void print(std::format_string<Args...> fmt, Args&&... args) {
    auto s = std::format(fmt, std::forward<Args>(args)...);
    std::fputs(s.c_str(), stdout);
}

template <typename... Args>
void print(std::FILE* f, std::format_string<Args...> fmt, Args&&... args) {
    auto s = std::format(fmt, std::forward<Args>(args)...);
    std::fputs(s.c_str(), f);
}

template <typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args) {
    auto s = std::format(fmt, std::forward<Args>(args)...);
    s += '\n';
    std::fputs(s.c_str(), stdout);
}

} // namespace std
