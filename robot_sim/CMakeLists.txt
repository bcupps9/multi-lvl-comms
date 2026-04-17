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
)

add_executable(seqcomm-sim
    seqcomm_sim.cc
    $<TARGET_OBJECTS:Cotamer>
    $<TARGET_OBJECTS:SeqComm>
)

# ── Trained simulation (requires libtorch) ────────────────────────────────────
#
# Provide libtorch via CMAKE_PREFIX_PATH, e.g.:
#   cmake -B build -DCMAKE_PREFIX_PATH=$(python3 -c \
#     "import torch; print(torch.utils.cmake_prefix_path)")
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "libtorch found — building seqcomm-sim-trained")
    add_executable(seqcomm-sim-trained
        seqcomm_sim_trained.cc
        $<TARGET_OBJECTS:Cotamer>
        $<TARGET_OBJECTS:SeqComm>
    )
    target_link_libraries(seqcomm-sim-trained ${TORCH_LIBRARIES})
    target_include_directories(seqcomm-sim-trained PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(STATUS "libtorch not found — skipping seqcomm-sim-trained")
    message(STATUS "  hint: cmake -DCMAKE_PREFIX_PATH=\$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..")
endif()
