cmake_minimum_required(VERSION 3.16)
project(cs2620-pset3 CXX)

option(SAN "Enable AddressSanitizer+UBSanitizer" OFF)
option(ASAN "Enable AddressSanitizer" OFF)
option(UBSAN "Enable UBSanitizer" OFF)
option(TSAN "Enable ThreadSanitizer" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_compile_options(-g -O3 -W -Wall)
add_link_options(-g -O3 -W -Wall)

if((SAN AND NOT TSAN) OR UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()
if((SAN AND NOT TSAN) OR ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()
if(TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()


add_library(Cotamer OBJECT
    ../cotamer/cotamer.cc
    ../cotamer/io.cc
)

add_library(SeqComm OBJECT
    agent_action.cc
    gaussian_field_env.cc
    intersection_env.cc
)

add_executable(seqcomm-sim
    seqcomm_sim.cc
    $<TARGET_OBJECTS:Cotamer>
    $<TARGET_OBJECTS:SeqComm>
)

# ── Optional: link libtorch for real neural model inference ───────────────────
#
# Enable with:
#   cmake -B build -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
#
# Download libtorch from https://pytorch.org/get-started/locally/
# (select LibTorch, C++/Java, your platform, same version as your Python torch)
#
# Once built:
#   ./build/robot_sim/seqcomm-sim --weights ./weights
option(USE_TORCH "Link against libtorch to run trained TorchScript models" OFF)
if(USE_TORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(seqcomm-sim PRIVATE USE_TORCH)
    target_link_libraries(seqcomm-sim "${TORCH_LIBRARIES}")
    # libtorch requires C++17 ABI; the project already uses C++23 which is fine.
    set_property(TARGET seqcomm-sim PROPERTY CXX_STANDARD 23)
    message(STATUS "libtorch found — LibtorchNeuralModels enabled")
endif()
