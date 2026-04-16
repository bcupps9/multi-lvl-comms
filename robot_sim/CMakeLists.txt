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

# Find xxhash
find_path(XXHASH_INCLUDE_DIR xxhash.h HINTS /opt/homebrew/include)
find_library(XXHASH_LIBRARY xxhash HINTS /opt/homebrew/lib)

if(NOT XXHASH_INCLUDE_DIR OR NOT XXHASH_LIBRARY)
    message(FATAL_ERROR "xxhash not found. Install with: brew install xxhash")
endif()

add_library(Cotamer OBJECT
    ../cotamer/cotamer.cc
    ../cotamer/io.cc
)

add_library(Pancy OBJECT
    pancydb.cc
    client_model.cc
    lockseq_model.cc
    utils.cc
)

add_executable(pt-single
    pt-single.cc
    $<TARGET_OBJECTS:Cotamer>
    $<TARGET_OBJECTS:Pancy>
)

add_executable(pt-backup
    pt-backup.cc
    $<TARGET_OBJECTS:Cotamer>
    $<TARGET_OBJECTS:Pancy>
)

add_executable(pt-paxos
    pt-paxos.cc
    $<TARGET_OBJECTS:Cotamer>
    $<TARGET_OBJECTS:Pancy>
)
