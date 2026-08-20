cmake_minimum_required(VERSION 3.20)

project(miniccl
    VERSION 0.1.0
    DESCRIPTION "A small collective communication library, built to learn how NCCL works"
    LANGUAGES CXX)

# No CUDA language enabled yet. It gets added in Milestone 2, at which point
# this becomes: project(... LANGUAGES CXX CUDA).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17, not -std=gnu++17: keeps us portable

# Default to Release so benchmark numbers are not accidentally measured on an
# unoptimised build. Single-config generators only; Xcode/VS pick at build time.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MINICCL_BUILD_TESTS "Build MiniCCL tests" ON)
option(MINICCL_BUILD_BENCHMARKS "Build MiniCCL benchmarks" ON)

# ---------------------------------------------------------------------------
# The library itself
# ---------------------------------------------------------------------------
# INTERFACE, because every function so far is a template living in a header.
# There is no .cpp to compile, so there is no object file and no .a/.so -- the
# target exists purely to carry the include path and the C++17 requirement to
# anything that links it.
add_library(miniccl INTERFACE)
add_library(miniccl::miniccl ALIAS miniccl)

target_include_directories(miniccl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_features(miniccl INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# Warning flags
# ---------------------------------------------------------------------------
# A separate INTERFACE target rather than flags on `miniccl`. If warnings rode
# on the library target, every downstream consumer would inherit our strictness
# and get warnings in *their* code. We opt our own targets in, nobody else's.
add_library(miniccl_warnings INTERFACE)
if(MSVC)
    target_compile_options(miniccl_warnings INTERFACE /W4)
else()
    target_compile_options(miniccl_warnings INTERFACE
        -Wall       # the common mistakes
        -Wextra     # the ones -Wall leaves out
        -Wpedantic  # anything outside strict ISO C++
        -Wshadow)   # an inner name hiding an outer one -- a classic silent bug
endif()

if(MINICCL_BUILD_TESTS)
    enable_testing()  # must be called in the top-level file for CTest to work
    add_subdirectory(tests)
endif()

if(MINICCL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

message(STATUS "MiniCCL ${PROJECT_VERSION} | build type: ${CMAKE_BUILD_TYPE} | "
               "tests: ${MINICCL_BUILD_TESTS} | benchmarks: ${MINICCL_BUILD_BENCHMARKS}")
