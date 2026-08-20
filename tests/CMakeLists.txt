# Use a GoogleTest already installed on the machine, otherwise download one.
#
# This defaults to OFF rather than auto-detecting, which is deliberate.
# find_package(GTest) searches every prefix on PATH, and package managers such
# as Anaconda ship GoogleTest as a transitive dependency of unrelated packages
# (yaml-cpp, for one). Those copies link successfully but have no rpath, so the
# test binary builds and then dies at startup with "Library not loaded". An
# opt-in flag makes the dependency explicit instead of dependent on what else
# happens to be installed. Turn it ON if you have a real system GoogleTest.
option(MINICCL_USE_SYSTEM_GTEST "Use an installed GoogleTest instead of fetching one" OFF)

if(MINICCL_USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
endif()

if(GTest_FOUND)
    message(STATUS "GoogleTest: using system installation")
else()
    message(STATUS "GoogleTest: not found locally, fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2  # pinned: a moving tag makes builds unreproducible
        GIT_SHALLOW    TRUE)
    # Keeps GoogleTest from overriding this project's compiler/linker settings.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_cpu_reduce test_cpu_reduce.cpp)

target_link_libraries(test_cpu_reduce
    PRIVATE
        miniccl::miniccl
        miniccl_warnings
        GTest::gtest_main)  # gtest_main supplies main(), so the test file has none

# Runs the test binary at build time to enumerate every TEST() and register
# each one with CTest individually. A failure then names the specific case
# rather than just "the test executable failed".
include(GoogleTest)
gtest_discover_tests(test_cpu_reduce)
