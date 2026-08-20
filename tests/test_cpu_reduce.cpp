#include <miniccl/cpu_reduce.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

TEST(CpuReduceSum, SumsSmallIntegerArray) {
    const std::vector<int> data{1, 2, 3, 4, 5};
    EXPECT_EQ(miniccl::cpu_reduce_sum(data.data(), data.size()), 15);
}

TEST(CpuReduceSum, HandlesNegativeValues) {
    const std::vector<int> data{-5, 10, -3, -2};
    EXPECT_EQ(miniccl::cpu_reduce_sum(data.data(), data.size()), 0);

    const std::vector<int> all_negative{-1, -2, -3};
    EXPECT_EQ(miniccl::cpu_reduce_sum(all_negative.data(), all_negative.size()), -6);
}

TEST(CpuReduceSum, SumsFloatingPointValues) {
    const std::vector<float> floats{1.5f, 2.25f, -0.75f};
    // EXPECT_NEAR, not EXPECT_EQ: floating-point addition is not exact, and
    // testing == on a computed float is a habit worth never forming.
    EXPECT_NEAR(miniccl::cpu_reduce_sum(floats.data(), floats.size()), 3.0f, 1e-5f);

    const std::vector<double> doubles{0.1, 0.2, 0.3};
    EXPECT_NEAR(miniccl::cpu_reduce_sum(doubles.data(), doubles.size()), 0.6, 1e-12);
}

TEST(CpuReduceSum, EmptyInputReturnsZero) {
    const std::vector<int> empty;
    EXPECT_EQ(miniccl::cpu_reduce_sum(empty.data(), 0u), 0);

    // count == 0 wins over a null pointer: no memory is read, so there is
    // nothing to reject.
    EXPECT_EQ(miniccl::cpu_reduce_sum<double>(nullptr, 0u), 0.0);
}

TEST(CpuReduceSum, NullPointerWithPositiveCountThrows) {
    EXPECT_THROW(miniccl::cpu_reduce_sum<int>(nullptr, 5u), std::invalid_argument);
    EXPECT_THROW(miniccl::cpu_reduce_sum<float>(nullptr, 1u), std::invalid_argument);
}

TEST(CpuReduceSum, MatchesStdAccumulateOnLargeInput) {
    constexpr std::size_t kCount = 1'000'000;
    std::vector<double> data(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        data[i] = static_cast<double>(i % 100) * 0.5;
    }

    const double expected =
        std::accumulate(data.begin(), data.end(), 0.0);
    const double actual = miniccl::cpu_reduce_sum(data.data(), data.size());

    // std::accumulate performs the same left-to-right sequential addition in
    // the same order, so the results should be bit-identical here. The
    // tolerance guards against a future change to either side's summation
    // order rather than against expected drift.
    EXPECT_NEAR(actual, expected, 1e-6);
}

}  // namespace
