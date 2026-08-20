# MiniCCL

A small collective communication library in C++/CUDA, built to understand how
NVIDIA's NCCL works by reimplementing its core.

## Goal

When a neural network trains on several GPUs, each GPU computes gradients from a
different slice of the batch. Before the optimizer step, every GPU must hold the
*sum* of all GPUs' gradients — otherwise the model copies diverge. That operation
is called **all-reduce**, and NCCL is the library that performs it.

MiniCCL rebuilds that operation from scratch: a ring all-reduce across multiple
GPUs, benchmarked against NCCL itself.

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the full learning plan.

## Current milestone

**Milestone 1 — CPU reference reduction.** Complete.

A header-only, templated sequential sum plus the project skeleton: CMake build,
GoogleTest suite, and a timing harness. There is **no CUDA code yet**. This
milestone exists to establish the correctness oracle — every GPU kernel written
later is checked against `cpu_reduce_sum`.

```cpp
#include <miniccl/cpu_reduce.hpp>

std::vector<float> data{1.0f, 2.0f, 3.0f};
float total = miniccl::cpu_reduce_sum(data.data(), data.size());  // 6.0f
```

Behaviour:

| Input | Result |
|---|---|
| `count == 0` | returns `T{}`, even if `data` is null |
| `data == nullptr`, `count > 0` | throws `std::invalid_argument` |

## Prerequisites

- CMake 3.20 or newer
- A C++17 compiler (tested with Apple clang 17)
- Git — used by CMake to fetch GoogleTest if it isn't installed locally
- Network access on the first configure, only if GoogleTest isn't already present

No CUDA toolkit and no GPU are required for this milestone.

## Build and run

Configure:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build
```

Run the tests:

```bash
ctest --test-dir build --output-on-failure
```

Run the benchmark (element count is optional, defaults to 10,000,000):

```bash
./build/benchmarks/benchmark_cpu_reduce 1000000
```

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | CPU reference reduction + project skeleton | ✅ done |
| 2 | CUDA reduction kernel (naive → shared memory → warp shuffle) | planned |
| 3 | Ring all-reduce simulated on CPU | planned |
| 4 | Multi-GPU ring all-reduce (reduce-scatter + all-gather) | planned |
| 5 | Benchmark against NCCL | planned |

## Limitations

Stated plainly, because the gap between this and NCCL is the point of the
project:

- **Sum only.** No min, max, or product reduction operations.
- **Single-threaded.** No SIMD intrinsics, no OpenMP, no multicore. The loop is
  written for clarity, not speed.
- **Accumulates in `T`.** Summing a large `float` array loses precision as the
  running total grows relative to each addend. A future milestone may add
  pairwise or Kahan summation; for now the behaviour deliberately matches what a
  naive GPU kernel does.
- **Contiguous raw pointers only.** No iterator or range interface.
- **The benchmark is not Google Benchmark.** It reports a plain mean over a fixed
  iteration count, with no statistical analysis or outlier rejection. Its
  bandwidth figure counts only the array's bytes and ignores cache effects, so
  small inputs will report a number above what DRAM could actually deliver.
- **No CUDA, no multi-GPU, no networking** — that is the whole point of
  milestones 2 through 5.
