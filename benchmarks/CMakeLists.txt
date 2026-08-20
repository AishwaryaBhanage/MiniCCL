add_executable(benchmark_cpu_reduce benchmark_cpu_reduce.cpp)

target_link_libraries(benchmark_cpu_reduce
    PRIVATE
        miniccl::miniccl
        miniccl_warnings)
