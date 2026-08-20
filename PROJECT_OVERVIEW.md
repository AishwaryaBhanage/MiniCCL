# MiniCCL — Project Overview & Learning Map

> Read this file first, every time you come back to the project after a break.
> It answers three questions: **what is this**, **what are we building**, and
> **what do I need to know / what will I learn**.

---

## 1. What is this project?

**MiniCCL is a from-scratch reimplementation of the core of NVIDIA's NCCL library.**

NCCL (NVIDIA Collective Communications Library, pronounced "nickel") is the piece
of software that makes multi-GPU training possible. Every time someone runs
PyTorch on 8 GPUs, NCCL is doing the invisible work underneath. We are rebuilding
the most important part of it — **all-reduce** — so that we understand it instead
of just using it.

### The problem it solves

Train a neural network on 4 GPUs. Each GPU holds a full copy of the model and
processes a different slice of the batch. After the backward pass:

```
GPU 0 has gradients:  [0.1, 0.5, 0.3, ...]
GPU 1 has gradients:  [0.2, 0.1, 0.9, ...]
GPU 2 has gradients:  [0.4, 0.3, 0.1, ...]
GPU 3 has gradients:  [0.1, 0.2, 0.2, ...]
```

These are **different**, because each GPU saw different data. But the model
weights must stay identical across GPUs, otherwise you no longer have one model —
you have four diverging ones. So before the optimizer step, every GPU must end up
holding the **sum** (then average) of all four gradient vectors:

```
Every GPU now has:    [0.8, 1.1, 1.5, ...]
```

That operation — *"combine a value across N devices, and give the combined result
back to all N devices"* — is called **all-reduce**. It is the single most
performance-critical collective in distributed training, because it runs on every
single training step, on hundreds of megabytes of gradients.

**MiniCCL is a library that performs all-reduce across multiple GPUs, efficiently.**

### Why this is a good project

- It sits at the exact intersection of **CUDA**, **C++ systems design**, and
  **distributed systems** — the three things ML-infrastructure interviews probe.
- It is *demonstrable*: you get real numbers on real hardware, compared against
  the industry-standard library.
- Almost nobody writes one. "I used PyTorch DDP" is common; "I implemented the
  collective underneath DDP and benchmarked it against NCCL" is not.

---

## 2. What are we actually building?

A C++/CUDA library exposing an API roughly like:

```cpp
miniccl::Communicator comm({0, 1, 2, 3});   // use GPUs 0..3

comm.all_reduce<float>(buffers, element_count, miniccl::ReduceOp::Sum);
// on return, every GPU's buffer holds the element-wise sum across all 4 GPUs
```

Plus: correctness tests, benchmarks against NCCL, and a README that makes the
results legible to someone who has 60 seconds.

### The core algorithm: ring all-reduce

The naive approach — every GPU sends its whole buffer to every other GPU — costs
O(N²) traffic and melts the interconnect. The ring algorithm is the standard
trick. Arrange the GPUs in a logical circle, split each buffer into N chunks, and
run two phases:

**Phase 1 — reduce-scatter** (N−1 steps). Each GPU sends one chunk to its right
neighbour and adds the chunk it receives from its left neighbour. After N−1 steps,
*each GPU owns exactly one chunk that is fully summed across all GPUs* — but no
GPU has the whole answer yet.

**Phase 2 — all-gather** (N−1 steps). Those finished chunks now circulate around
the ring, being copied (not added) until every GPU holds every finished chunk.

```
Start (4 ranks, 4 chunks each):

  rank 0: [A0 A1 A2 A3]
  rank 1: [B0 B1 B2 B3]
  rank 2: [C0 C1 C2 C3]
  rank 3: [D0 D1 D2 D3]

After reduce-scatter — each rank owns ONE complete sum:

  rank 0: [ ..  Σ1  ..  .. ]        Σk = Ak+Bk+Ck+Dk
  rank 1: [ ..  ..  Σ2  .. ]
  rank 2: [ ..  ..  ..  Σ3 ]
  rank 3: [ Σ0  ..  ..  .. ]

After all-gather — everyone has everything:

  every rank: [Σ0 Σ1 Σ2 Σ3]
```

**Why this is the right algorithm:** each GPU sends and receives only
`2·(N−1)/N · buffer_size` bytes total — which approaches `2·buffer_size` and is
*independent of N* as GPU count grows. It is provably bandwidth-optimal. The cost
is latency: 2(N−1) sequential steps, which is why small messages are better served
by a tree algorithm (a good thing to know, and a good "what would you do next"
answer).

### Scope boundary

**In scope (Stages 0–5):** sum reduction, FP32/FP64, multiple GPUs inside one
machine, peer-to-peer transfers over PCIe/NVLink.

**Out of scope for now (Stage 6, optional):** multiple machines, TCP sockets,
rank discovery, fault tolerance, InfiniBand/RDMA.

Being explicit about this boundary in the README is a *strength*, not an
admission of weakness. A polished 2-GPU implementation beats a half-broken
multi-node one.

---

## 3. Knowledge map — what you need, and what you'll gain

The stages are ordered so each one only requires what the previous one taught you.
Do not skip ahead; the debugging cost of skipping is brutal, because a wrong
multi-GPU program looks exactly like a correct one until you check the numbers.

### Stage 0 — Foundations (~1 week)

| Need going in | Gain coming out |
|---|---|
| Basic C++ syntax | Pointers vs. references vs. values, when each is right |
| Willingness to read compiler errors | Templates: writing code once for `float` and `double` |
| — | RAII and smart pointers — why destructors are the answer to leaks |
| — | The host/device split: CPU RAM and GPU VRAM are separate universes |
| — | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| — | The CUDA execution model: threads → blocks → grid |
| — | CMake enough to build a mixed C++/CUDA target |

**Deliverables:** C++ vector add, CUDA vector add, CUDA array sum.

**The gate:** you may not proceed until you can explain, unprompted, every term in

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
```

*(Why: each thread must compute a globally unique index into the array. `threadIdx.x`
is your position inside your block; `blockDim.x` is how many threads a block holds;
`blockIdx.x` is which block you're in. Multiply and add → a unique global slot.
If you can't derive this, every later kernel will be guesswork.)*

---

### Stage 1 — CUDA reduction kernel (~1–2 weeks)

Sum an array on the GPU. `[1,2,3,4,5,6,7,8] → 36`. Sounds trivial; it is the
canonical hard CUDA problem, because the answer lives in one place but the data is
spread across thousands of threads.

Build it in escalating versions — **keep every version**, they become your
before/after optimization story:

1. CPU reference
2. Naive CUDA
3. Shared-memory tree reduction
4. Warp-shuffle optimization
5. Templated over `float` / `double`
6. Benchmarked against CUB (NVIDIA's own primitive library)

| Need going in | Gain coming out |
|---|---|
| Stage 0 | **Coalesced memory access** — why thread *i* must read element *i* |
| | **Shared memory** — the fast scratchpad each block owns |
| | **Warps** — 32 threads that execute in lockstep, and why that changes everything |
| | `__syncthreads()` and why a missing one silently corrupts results |
| | Warp shuffles (`__shfl_down_sync`) — cross-thread data movement with no memory at all |
| | Choosing grid/block dimensions |
| | Measuring **bandwidth**, and realising you are memory-bound, not compute-bound |
| | That PCIe transfer often dominates the kernel itself |

**Deliverable:**

```
$ ./reduce_benchmark --elements 10000000
Elements:     10,000,000
CPU result:   4,998,231.5
CUDA result:  4,998,231.5
CUDA time:    0.42 ms
Bandwidth:    95.2 GB/s
Correctness:  PASS
```

---

### Stage 2 — Ring all-reduce, simulated on CPU (~3–5 days)

Implement the ring using plain `std::vector`s and a loop over "ranks". No GPU, no
CUDA, no async. Print the buffers after every step.

This stage feels like a detour. It is the highest-value 4 days in the whole
project. Once the multi-GPU version exists, a wrong answer could be caused by
algorithm error, wrong chunk indexing, a missing stream sync, a peer-access
failure, or a race — and you will not be able to tell which. Doing it on the CPU
first permanently eliminates the first two.

| Need going in | Gain coming out |
|---|---|
| Basic C++ containers | The reduce-scatter / all-gather decomposition, deeply |
| | Chunk index arithmetic (the `(rank - step + N) % N` pattern that trips everyone) |
| | Why the ring is bandwidth-optimal and latency-suboptimal |
| | A reference oracle you'll test the GPU version against |

---

### Stage 3 — Multi-GPU all-reduce (~2–3 weeks) ← **the heart of the project**

Now real: 2–4 GPUs in one machine, one rank per GPU, peer-to-peer transfers,
streams, events.

| Need going in | Gain coming out |
|---|---|
| Stages 1 + 2 | `cudaSetDevice` and the idea of a *current device* as thread state |
| | Peer-to-peer access: `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess` |
| | Direct GPU→GPU copies (`cudaMemcpyPeerAsync`) that never touch the CPU |
| | **CUDA streams** — ordered queues of work, and how to overlap copy with compute |
| | **CUDA events** — correct GPU timing, and cross-stream dependencies |
| | Designing a `Communicator` class: what it owns, what its destructor must free |
| | Error handling in a library (never `exit()` in library code) |
| | Debugging nondeterminism — the hardest skill here |

**Test matrix — write these as you go, not at the end:**

- 2 GPUs and 4 GPUs
- Tiny arrays and 256 MB arrays
- `float` and `double`
- Element counts not divisible by GPU count (the classic ring bug)
- Called repeatedly in a loop (catches leaks and stale state)
- Invalid device IDs → clean error, not a crash
- GPUs without peer access → correct fallback (staging through host)

Always compare against the CPU reference from Stage 2.

---

### Stage 4 — Professional repository (~1 week)

```
miniccl/
├── CMakeLists.txt
├── README.md
├── include/miniccl/{communicator,buffer,types}.hpp
├── src/{communicator.cpp,reduction.cu,ring_allreduce.cu}
├── tests/{reduction_test,allreduce_test}.cu
├── benchmarks/{benchmark_miniccl,benchmark_nccl}.cu
├── scripts/plot_results.py
└── docs/ring_algorithm.md
```

| Need going in | Gain coming out |
|---|---|
| Working code | Public-header vs. implementation separation (API design) |
| | CMake for a real multi-target project with CUDA + tests |
| | GoogleTest |
| | Technical writing: README as the artifact a recruiter actually reads |

The README must answer, above the fold: what problem, what algorithm (with the
diagram), how to build, how correctness was verified, what hardware, what the
numbers are, and **what the limitations are.** For example:

> Currently supports sum reduction for FP32/FP64 across GPUs within a single node.
> Inter-node transport and InfiniBand are not yet supported.

---

### Stage 5 — Benchmark against NCCL (~1 week)

Sweep message sizes: 1 KB, 16 KB, 256 KB, 1 MB, 16 MB, 64 MB, 256 MB. For each,
report average latency, algorithm bandwidth, bus bandwidth, ratio vs. NCCL, and
correctness. Separate warm-up iterations from measured ones.

| Need going in | Gain coming out |
|---|---|
| Stage 3 + 4 | Benchmarking methodology: warm-up, repetition, variance |
| | Algorithm bandwidth vs. **bus bandwidth** (and why the 2(N−1)/N factor matters) |
| | Reading a latency/bandwidth curve and explaining the small-message regime |
| | Profiling with Nsight Systems / Nsight Compute |
| | Honest reporting of results that are worse than the incumbent |

**Non-negotiable:** every number is measured on your hardware. Never write "78% of
NCCL" because it sounds good. Pick the percentage *after* the benchmark runs. A
fabricated number is the one thing that will end an interview instantly, and
interviewers at NVIDIA will know your hardware's roofline better than you do.

---

### Stage 6 — Multi-node (optional, +3–5 weeks)

One process per GPU, TCP transport, rendezvous server for rank discovery,
timeouts, error propagation, graceful handling of a dead rank.

| Gain coming out |
|---|
| Sockets, serialization, partial sends |
| Rank/world-size bootstrapping |
| Distributed failure modes — how one dead rank hangs an entire ring |
| Why TCP is slow relative to RDMA/InfiniBand |

Worth it for NCCL/HPC/distributed-training roles specifically. Optional for
general AI/ML engineering. **Only start this after Stages 0–5 are genuinely done.**

---

## 4. When is it résumé-ready?

All eight, no partial credit:

- [ ] Ring all-reduce working on ≥ 2 GPUs
- [ ] Automated correctness tests
- [ ] Benchmarks from real hardware
- [ ] Comparison against NCCL
- [ ] README with architecture diagram
- [ ] Reproducible CMake build
- [ ] Stated limitations
- [ ] At least one optimization with before/after numbers

---

## 5. Interview questions you must be able to answer

Keep these open while building; answer each one *in writing* when you reach the
stage that teaches it.

**Algorithm.** What is all-reduce and why does distributed training need it?
What are reduce-scatter and all-gather? Why is a ring bandwidth-efficient? How
many communication steps? Why is a tree better for small messages?

**CUDA.** What is a warp? Why does memory coalescing matter? When do you use
shared memory? What does `__syncthreads()` do? Sync vs. async copies? How do
streams overlap communication and computation?

**C++ design.** Why templates? How does RAII prevent GPU memory leaks? How do you
report CUDA errors from a library? What does a `Communicator` own? How did you
design the public API?

**Testing & benchmarking.** How did you verify correctness? How did you avoid
inaccurate GPU timing? Why warm-up iterations? Why is MiniCCL slower than NCCL,
specifically? What would you optimize next?

**Distributed (if Stage 6).** How do ranks discover each other? What happens when
a rank dies? How do you prevent the ring from hanging? Timeouts and partial sends?
Why is TCP slower than RDMA?

---

## 6. The story this project tells

> "I wanted to understand what happens beneath PyTorch DDP. I started by
> optimizing a CUDA reduction kernel through four versions — naive, shared-memory
> tree, warp-shuffle — and benchmarked each against CUB. Then I simulated ring
> all-reduce on the CPU to get the algorithm and chunk indexing provably right.
> Only then did I implement reduce-scatter and all-gather across multiple GPUs
> using asynchronous peer-to-peer transfers and CUDA streams. I validated against
> the CPU reference and NCCL, then profiled to find where synchronization and
> memory transfers were limiting throughput."

Problem → incremental engineering → testing → profiling → optimization. That is
the shape interviewers want, and the stage order above is designed to produce it
as a natural byproduct.

---

## 7. Open question to resolve before Stage 1

**Where will the CUDA code actually run?** This project requires an NVIDIA GPU —
and for Stage 3 onward, **two or more NVIDIA GPUs in one machine**. Apple Silicon
and AMD GPUs cannot run CUDA, and there is no emulator that will give you
meaningful benchmark numbers.

Options to settle now, because it shapes the whole timeline:

- A university HPC cluster or lab machine (usually free; check queue policy)
- Cloud multi-GPU instance — AWS `g4dn.12xlarge`, GCP with 2–4 T4/A10G, Lambda
  Labs, RunPod, Vast.ai (rent by the hour, only when benchmarking)
- Google Colab — fine for Stages 0–1, **single GPU only**, so it cannot carry
  Stage 3
- A personal desktop with 2 NVIDIA cards

Stages 0–2 can be written anywhere (Stage 2 is pure CPU C++). Stage 1 needs one
GPU. Stage 3 needs two. Decide the Stage 3 answer before you get there — a
suddenly-unavailable machine is the most common way this project stalls.

---

## Glossary

| Term | Meaning |
|---|---|
| **Collective** | An operation involving all participants at once (all-reduce, broadcast, all-gather) |
| **Rank** | The ID of one participant, 0..N−1. Here, one rank = one GPU |
| **World size** | Total number of ranks |
| **All-reduce** | Combine a value across all ranks; every rank gets the result |
| **Reduce-scatter** | Combine across ranks; each rank gets one *piece* of the result |
| **All-gather** | Every rank ends up with every rank's piece |
| **Warp** | 32 GPU threads executing in lockstep |
| **Coalescing** | Adjacent threads reading adjacent memory → one wide transaction |
| **Shared memory** | Fast on-chip scratchpad shared by threads in a block |
| **Stream** | An ordered queue of GPU work; separate streams can overlap |
| **P2P** | GPU-to-GPU transfer that bypasses the CPU |
| **NVLink** | NVIDIA's fast direct GPU interconnect (vs. slower PCIe) |
| **Algorithm bandwidth** | `buffer_bytes / time` — what the user perceives |
| **Bus bandwidth** | Actual wire traffic; for ring all-reduce ≈ `alg_bw · 2(N−1)/N` |
| **NCCL** | NVIDIA's production collective library — our reference and benchmark target |
| **CUB** | NVIDIA's library of optimized GPU primitives — Stage 1's benchmark target |
