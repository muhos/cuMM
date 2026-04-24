# CUDA Matrix Multiplication: From Naive to Tensor Cores

A progressive, hand-crafted implementation of single-precision matrix multiplication (SGEMM) in CUDA C++, built from scratch for efficient state-vector operations and AI-driven workloads. The final kernel uses **TF32 Tensor Cores via inline PTX**, asynchronous pipelining, warp-level tiling, and bank-conflict-free shared memory. It achieves 65.19 TF32 TFLOPS on `(4096 x 10240 x 4096)` matrix dimensions, outperforming cuBLAS SGEMM (61.79 TFLOPS) and reaching 87% of cuBLAS Tensor Core throughput (74.58 TFLOPS).

---

## Optimization Journey

Each stage is preserved in source as a stepping stone. The progression below reflects both the development order and the underlying reasoning.

### Stage 0: Naive Kernel
Each thread computes one element of C by iterating over the full K dimension in global memory. Simple but memory-bound: two global loads per FMA, no reuse.

### Stage 1: Shared Memory Tiling
Tiles of A and B are loaded cooperatively into shared memory, reducing global memory traffic by a factor of `TILESIZE`. The kernel is now shared-memory-bound rather than global-memory-bound.

### Stage 2: Double Buffering
While computing on the current tile, the next tile is prefetched into a second shared memory buffer. This overlaps computation and memory access, hiding shared memory latency when the kernel is still memory-bound.

### Stage 3: Register Tiling
Each thread computes an `RM x RN` sub-tile of C, holding intermediate results in registers. Reduces shared memory load instructions by a factor of `RM x RN`, making the kernel compute-bound.

### Stage 4: Vectorized Loads (`float4`)
Global and shared memory loads are widened to 128-bit (`float4` / `LDS.128`). Improves memory throughput and enables better coalescing. Combined with register tiling for a double-buffered register prefetch loop.

### Stage 5: Warp Tiling
The thread block tile is decomposed into warp tiles (`WM x WN`), with each warp responsible for a contiguous region of the output. This improves data locality within the warp and aligns memory access patterns with hardware warp scheduling.

### Stage 6: TF32 Tensor Cores + `cp.async` Pipeline (Final Kernel)
The production kernel: `matrixMul_tiled_db_reg_warp_tc`.

Key techniques:

- **TF32 Tensor Cores via inline PTX**, Uses `wmma.load.a/b.sync.aligned` and `wmma.mma.sync` PTX instructions directly (m16n16k8 shape, TF32 precision, row/col layout). Bypasses the `nvcuda::wmma` C++ wrappers for full control over register mapping.
- **`cp.async` asynchronous pipelining**, Global-to-shared memory copies are issued asynchronously using `cp.async.cg` / `cp.async.ca` PTX with `CP_ASYNC_COMMIT_GROUP` and `CP_ASYNC_WAIT_GROUP(1)`, allowing computation on the current tile to overlap with prefetch of the next. A double-buffered scheme manages the two shared memory stages.
- **Bank-conflict-free shared memory**, Shared memory layouts for A and B use swizzle functions (`SWZ_A`, `SWZ_B`) to permute storage indices, eliminating bank conflicts on `ld.shared` without excessive padding overhead.
- **Warp-level accumulation**, Each warp accumulates a `WM x WN = 32 x 32` output tile across two MMA sub-tiles (`WARP_MMA_M_ITERS x WARP_MMA_N_ITERS = 2 x 2`). Accumulators are kept in registers as named float variables throughout the K-loop.
- **Full unrolling of warp tile loops**, The `WARP_MMA_M_ITERS` and `WARP_MMA_N_ITERS` loops over MMA sub-tiles are fully unrolled (`#pragma unroll`). With all MMA instructions visible simultaneously, the compiler can interleave independent `wmma.mma.sync` operations, hide their latency through instruction-level parallelism (ILP), and assign accumulator registers statically &ndash; reducing register pressure compared to a dynamic loop where the compiler must conservatively spill.
- **Vectorized stores**, Results are written back using inline PTX `st.global.v2.f32` (64-bit vectorized stores) to maximize store throughput.
- **`__restrict__`**, All pointer arguments are marked `__restrict__` to enable the compiler to assume no aliasing and generate better memory access schedules.

---

## Project Structure

```
cuMM/
├── src/
│   ├── cuMM.cu                  # Main entry point and benchmark driver
│   ├── global.cuh               # Matrix dimensions and common includes
│   ├── kernels/
│   │   ├── basic.cuh            # Stage 0: naive kernel
│   │   ├── optimization1.cuh    # Stage 1: shared memory tiling
│   │   ├── optimization2.cuh    # Stage 2: double buffering
│   │   ├── optimization3.cuh    # Stage 3: register tiling
│   │   ├── optimization4.cuh    # Stage 4: vectorized loads (float4)
│   │   ├── optimization5.cuh    # Stage 5: warp tiling
│   │   ├── optimization6.cuh    # Stage 6: TF32 tensor cores + cp.async
│   │   ├── cublas.cuh           # cuBLAS SGEMM and TF32 GemmEx benchmarks
│   │   ├── bench.cuh            # Benchmark runner declaration
│   │   └── bench.cu             # Benchmark runner implementation
│   └── utils/
│       ├── check.h              # GPU error checking + correctness validation
│       ├── helper.h             # Launch macros
│       ├── input.h              # Matrix input handling
│       ├── table.h              # Formatted performance table output
│       └── timer.h              # GPU event-based timing
├── Makefile
└── README.md
```

---

## Prerequisites

- NVIDIA GPU with compute capability 7.0 or later (Volta or newer, required for `cp.async` and TF32 Tensor Cores)
- CUDA Toolkit 12.0 or later

Tested on:
- NVIDIA RTX 4090, CUDA 12.8, Ubuntu 24.04

---

## Build

```bash
make              # CUDA build (default)
```

---

## Usage

```bash
./cuMM
```

Runs the benchmark on randomized matrices of size **4096 x 10240** (A) and **10240 x 4096** (B), producing a **4096 x 4096** output C. Each kernel is timed using CUDA events and validated against the naive baseline.

Optional file I/O:

```bash
./cuMM <input_file> <output_file>
```

Input file format: `M*N` elements of A followed by `N*M` elements of B, whitespace-separated, row-major.

---

## Performance

Benchmarked on **NVIDIA RTX 4090** (82.6 TFLOPS TF32 peak), matrix size 4096 x 10240 x 4096.

```
------[ Performance Evaluation ]==================================================================================
 Kernel (float)           | K-Tile | Shared Mem | Block Size | Grid Size  |  Time (ms) | TFLOPS | Check 
------------------------------------------------------------------------------------------------------------------
 Basic                    |   n/a |         na | (16, 16)   | (256, 256) |      92.24 |   3.72 | na    
 Tiled                    |    16 |     2048 B | (16, 16)   | (256, 256) |      53.72 |   6.40 | PASSED <-- Stage 1
 Tiled                    |    32 |     8192 B | (32, 32)   | (128, 128) |      54.73 |   6.28 | PASSED
 Tiled-DB                 |    16 |     4096 B | (16, 16)   | (256, 256) |      49.52 |   6.94 | PASSED <-- Stage 2
 Tiled-DB                 |    32 |    16384 B | (32, 32)   | (128, 128) |      50.61 |   6.79 | PASSED
 Tiled-DB_Reg             |     8 |    16384 B | (16, 16)   | (32, 32)   |       8.43 |  40.77 | PASSED <-- Stage 3
 Tiled-DB_Reg             |    16 |    32768 B | (16, 16)   | (32, 32)   |       7.91 |  43.42 | PASSED
 Tiled-DB_Reg_Vec         |     8 |     4096 B | (16, 16)   | (32, 32)   |       8.13 |  42.24 | PASSED <-- Stage 4
 Tiled-DB_Reg_Vec         |    16 |     8192 B | (16, 16)   | (32, 32)   |       7.55 |  45.48 | PASSED
 Tiled-DB_Reg_Vec_Warp    |     8 |     4096 B | (256, 1)   | (32, 32)   |       7.89 |  43.53 | PASSED <-- Stage 5
 Tiled-DB_Reg_Vec_Warp    |    16 |     8192 B | (256, 1)   | (32, 32)   |       8.28 |  41.51 | PASSED
 Tiled-DB_Reg_Vec_Warp_TC |     8 |    21504 B | (256, 1)   | (16, 47)   |       5.27 |  65.19 | PASSED <-- Stage 6
 cuBLAS-Warmup            |   n/a |         na | na         | na         |      33.84 |  10.16 | PASSED
 cuBLAS-SGEMM             |   n/a |         na | na         | na         |       5.56 |  61.79 | PASSED
 cuBLAS-Tensor            |   n/a |         na | na         | na         |       4.61 |  74.58 | PASSED
------------------------------------------------------------------------------------------------------------------
```

> **Note:** Run `./cuMM` to populate the table with results on your hardware. The final Tensor Core kernel (`Tiled-Reg-Warp-TC`) is benchmarked alongside cuBLAS SGEMM and cuBLAS TF32 Tensor mode for direct comparison.

---

## Key Implementation Notes

### Why inline PTX instead of `nvcuda::wmma`?
The `wmma` C++ API is convenient but abstracts over the register layout, limiting control over how fragments are loaded relative to shared memory. Using `wmma.load.a/b` and `wmma.mma.sync` PTX directly gives precise control over the lane-to-register mapping (8x4 lane grid for m16n16k8), which is necessary to align with the swizzled shared memory layout and avoid redundant register moves.

### Why `cp.async` instead of manual prefetch?
Manual prefetch with `__syncthreads()` still serializes: the sync forces all threads to wait before computation can proceed. `cp.async` allows the copy to remain in flight (`CP_ASYNC_WAIT_GROUP(1)`; only requiring the *previous* tile to have landed, not the one being issued), enabling true overlap between memory transfer and computation.

### Swizzling to eliminate bank conflicts
Without swizzling, the transposed B layout (`tileB[col][k]`) causes 16-way bank conflicts on `ld.shared` when multiple lanes in a warp access the same bank. `SWZ_B(c) = c ^ ((c & 8) >> 2)` permutes the column index such that consecutive warp lanes hit distinct banks. Similarly, `SWZ_A(r, k) = k ^ (r & 4)` eliminates conflicts in A's row-major layout.

---

## References

- [CUDA C++ Programming Guide (Warp Matrix Functions)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions)
- [PTX ISA (`wmma` instructions)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)
- [PTX ISA (`cp.async`)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
- [CUTLASS Linear Algebra](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
