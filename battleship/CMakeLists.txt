cmake_minimum_required(VERSION 3.16)
project(battleship CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../robot_sim
)

add_compile_options(-g -O3 -march=native -W -Wall)
add_link_options(-g -O3 -march=native -W -Wall)

# ── battleship-sim (requires libtorch) ───────────────────────────────────────
#
# Enable with:
#   cmake -B build -DUSE_TORCH=ON \
#         -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
#   cmake --build build --target battleship-sim

option(USE_TORCH "Build with libtorch" OFF)

if(USE_TORCH)
    find_package(Torch REQUIRED)

    add_executable(battleship-sim
        battleship_env.cc
        battleship_sim.cc
        ../robot_sim/agent_action.cc
        ../cotamer/cotamer.cc
        ../cotamer/io.cc
    )

    target_compile_definitions(battleship-sim PRIVATE USE_TORCH)
    target_link_libraries(battleship-sim "${TORCH_LIBRARIES}")
    set_property(TARGET battleship-sim PROPERTY CXX_STANDARD 23)

    message(STATUS "battleship-sim enabled")
    message(STATUS "  Run: ./build/battleship/battleship-sim weights/ --mode seqcomm --episodes 2000")
else()
    message(STATUS "battleship-sim disabled (pass -DUSE_TORCH=ON to enable)")
endif()
