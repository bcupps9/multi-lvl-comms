#include "utils.hh"
#include <cctype>
#include <sstream>

std::string short_options_for(const struct option* longopts) {
    std::string str;
    while (longopts->name) {
        if (longopts->val > 0
            && longopts->val < 128
            && (isalpha(longopts->val) || isdigit(longopts->val))) {
            str += (char) longopts->val;
            if (longopts->has_arg == required_argument) {
                str += ':';
            } else if (longopts->has_arg == optional_argument) {
                str += "::";
            }
        }
        ++longopts;
    }
    return str;
}

namespace netsim {
extern const std::string disconnected_id;
[[maybe_unused]] const std::string disconnected_id{"disconnected"};
}
