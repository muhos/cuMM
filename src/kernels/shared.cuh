#pragma once

#include "../cuMM.cuh"

/**
 * Optimization 0: Shared Memory Tiling
 * ------------------------------------
 * Load tiles of A and B into shared memory to reduce global memory accesses.
 * matrixMulHIP kernel is memory-bound due to 2 global loads vs 1 FMA per iteration.
 * By using shared memory, we reduce global memory instructions and turn them into
 * shared memory instructions which have lower latency but still shared memory-bound.
 */
template <typename T, int TILESIZE> 
__global__ 
void matrixMul_tiled(const T *A, const T *B, T *C);

#define BENCHMARK_OPT1_KERNEL(NAME, TYPE, TILE, BX, BY) \
do { \
    dim3 NAME##Block((BX), (BY)); \
    if (NAME##Block.x * NAME##Block.y == 0 || NAME##Block.x * NAME##Block.y > 1024) { \
        std::cerr << "Error: invalid block size: (" << \
            NAME##Block.x << ", " << NAME##Block.y << ")" << std::endl; \
        break; \
    } \
    dim3 NAME##Grid(ROUNDUP(M, NAME##Block.x), ROUNDUP(M, NAME##Block.y)); \
    const int NAME##smemSize = 2 * TILE * TILE * sizeof(TYPE); \
    float NAME##_elapsed = 0.0f; \
    GENERATE_KERNEL_CONFIG(matrixMul_tiled, NAME, TYPE, TILE) \
    table_row(#NAME, M, N, TILE, NAME##smemSize, NAME##Block, NAME##_elapsed, \
        checkMulResults(hC, hC_basic, dimC) ? "PASSED" : "FAILED"); \
} while (0);

