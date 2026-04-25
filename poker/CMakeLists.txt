cmake_minimum_required(VERSION 3.16)
project(poker CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include paths: this dir, project root (for cotamer/), robot_sim/ (for shared headers)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../robot_sim
)

add_compile_options(-g -O3 -march=native -W -Wall)
add_link_options(-g -O3 -march=native -W -Wall)

# ── poker-sim (requires libtorch) ────────────────────────────────────────────
#
# Enable with:
#   cmake -B build -DUSE_TORCH=ON \
#         -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
#   cmake --build build --target poker-sim

option(USE_TORCH "Build poker-sim with libtorch" OFF)

if(USE_TORCH)
    find_package(Torch REQUIRED)

    add_executable(poker-sim
        poker_env.cc
        poker_sim.cc
        # Shared SeqComm agent logic
        ../robot_sim/agent_action.cc
        # Cotamer coroutine runtime
        ../cotamer/cotamer.cc
        ../cotamer/io.cc
    )

    target_compile_definitions(poker-sim PRIVATE USE_TORCH)
    target_link_libraries(poker-sim "${TORCH_LIBRARIES}")
    set_property(TARGET poker-sim PROPERTY CXX_STANDARD 23)

    message(STATUS "libtorch found — poker-sim enabled")
    message(STATUS "  Run: ./build/poker/poker-sim weights/ --mode seqcomm --episodes 2000")
else()
    message(STATUS "poker-sim disabled (pass -DUSE_TORCH=ON to enable)")
endif()
