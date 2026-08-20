#ifndef MINICCL_CPU_REDUCE_HPP
#define MINICCL_CPU_REDUCE_HPP

#include <cstddef>
#include <stdexcept>

namespace miniccl {

/// Sums `count` elements starting at `data` and returns the total.
///
/// This is the reference implementation for every reduction MiniCCL will
/// later perform on the GPU. It is deliberately the simplest correct thing:
/// a single sequential loop with no vectorisation hints and no parallelism.
/// Its job is to be *obviously* right, so that when a CUDA kernel disagrees
/// with it, the kernel is the suspect.
///
/// Contract:
///   count == 0                 -> returns T{} (zero), regardless of `data`
///   data == nullptr, count > 0 -> throws std::invalid_argument
///
/// Precision note: the accumulator has type T, so summing many float values
/// loses precision as the running total grows relative to each addend. This
/// is intentional for now -- it matches what a naive GPU kernel does. When
/// MiniCCL compares CPU and GPU results, tolerance must account for it.
template <typename T>
T cpu_reduce_sum(const T* data, std::size_t count) {
    // Checked before the null test on purpose: an empty range reads no
    // memory, so a null pointer is harmless and the answer is well-defined.
    // This mirrors how the C++ standard library treats empty ranges.
    if (count == 0) {
        return T{};
    }

    if (data == nullptr) {
        throw std::invalid_argument(
            "miniccl::cpu_reduce_sum: data is null but count > 0");
    }

    T sum = T{};
    for (std::size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

}  // namespace miniccl

#endif  // MINICCL_CPU_REDUCE_HPP
