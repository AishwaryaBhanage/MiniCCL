// Lightweight timing harness for miniccl::cpu_reduce_sum.
//
// This is NOT Google Benchmark. It does no statistical analysis, no outlier
// rejection, and no automatic iteration scaling. It reports a plain mean over
// a fixed number of runs. Treat the numbers as a rough baseline for the CUDA
// work later, not as a rigorous measurement.

#include <miniccl/cpu_reduce.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDefaultElements = 10'000'000;
constexpr int kWarmupIterations = 3;
constexpr int kMeasuredIterations = 10;

// Fixed seed so the input is byte-identical on every run and every machine.
// A benchmark whose input changes between runs cannot detect a regression.
std::vector<float> make_deterministic_input(std::size_t count) {
    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = dist(rng);
    }
    return values;
}

void print_usage(const char* program_name) {
    std::cerr << "usage: " << program_name << " [element_count]\n"
              << "  element_count  number of floats to reduce (default "
              << kDefaultElements << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t element_count = kDefaultElements;

    if (argc > 2) {
        print_usage(argv[0]);
        return 1;
    }
    if (argc == 2) {
        try {
            element_count = static_cast<std::size_t>(std::stoull(argv[1]));
        } catch (const std::exception&) {
            std::cerr << "error: could not parse '" << argv[1]
                      << "' as an element count\n";
            print_usage(argv[0]);
            return 1;
        }
        if (element_count == 0) {
            std::cerr << "error: element count must be greater than zero\n";
            return 1;
        }
    }

    const std::vector<float> input = make_deterministic_input(element_count);

    // The compiler can see that cpu_reduce_sum has no side effects and that we
    // ignore the result -- which entitles it to delete the entire call and
    // report a timing of zero. Writing each result into a volatile variable
    // forces the store to actually happen, so the work cannot be elided.
    volatile float sink = 0.0f;

    // Warm-up: the first passes fault in pages and pull the array into cache.
    // Including them in the average would measure memory setup, not the loop.
    for (int i = 0; i < kWarmupIterations; ++i) {
        sink = miniccl::cpu_reduce_sum(input.data(), input.size());
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kMeasuredIterations; ++i) {
        sink = miniccl::cpu_reduce_sum(input.data(), input.size());
    }
    const auto end = std::chrono::steady_clock::now();

    // Read the sink after timing. Without this the variable is only ever
    // written, which trips -Wunused-but-set-variable; reading it also gives us
    // the value to report, so nothing is computed twice.
    const float result = sink;

    const double total_seconds =
        std::chrono::duration<double>(end - start).count();
    const double average_seconds =
        total_seconds / static_cast<double>(kMeasuredIterations);
    const double average_ms = average_seconds * 1000.0;

    const double bytes_read =
        static_cast<double>(element_count) * static_cast<double>(sizeof(float));
    // "Approximate" because this counts only the bytes the array occupies. It
    // ignores cache hits, so for small inputs the figure will exceed what the
    // DRAM bus could actually deliver.
    const double gb_per_second =
        (average_seconds > 0.0) ? (bytes_read / average_seconds) / 1e9 : 0.0;

    std::cout << "MiniCCL CPU reduction benchmark\n"
              << "  (lightweight std::chrono harness, not Google Benchmark)\n\n";
    std::cout << "  Type:            float\n";
    std::cout << "  Elements:        " << element_count << "\n";
    std::cout << "  Input size:      " << std::fixed << std::setprecision(2)
              << bytes_read / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "  Warm-up iters:   " << kWarmupIterations << "\n";
    std::cout << "  Measured iters:  " << kMeasuredIterations << "\n";
    std::cout << "  Result:          " << std::setprecision(4) << result << "\n";
    std::cout << "  Average time:    " << std::setprecision(4) << average_ms
              << " ms\n";
    std::cout << "  Approx bandwidth: " << std::setprecision(2) << gb_per_second
              << " GB/s\n";

    return 0;
}
