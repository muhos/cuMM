#pragma once

#include "../global.cuh"

/**
 * Optimization 1: Double buffering of shared memory tiles
 * -------------------------------------------------------
 * While computing on one tile, prefetch the next tile into another buffer.
 * This helps hide global memory latency if the kernel is memory-bound which
 * is the case matrixMulHIP_tiled. More shared memory load instructions 'ld.shared'
 * are issued in the TILESIZE loop vs 1 FMA instruction.
 */
template <typename T, int TILESIZE> 
__global__ void matrixMul_tiled_db(const T *A, const T *B, T *C) {

    // Precondition as stated in assumptions.
    assert(TILESIZE > 0 && TILESIZE == blockDim.x && TILESIZE == blockDim.y);

    // Thread indices to access elements within a tile.
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Block indices to fetch tiles from global memory.
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Global row and column indices.
    int row = by * TILESIZE + ty;
    int col = bx * TILESIZE + tx;

    // Shared memory tiles for A and B with double buffering.
    __shared__ float tileA[2][TILESIZE * TILESIZE];
    __shared__ float tileB[2][TILESIZE * TILESIZE];

    // Initialize the accumulation variable for C element.
    // Moved outside the tile loop to accumulate over all tiles.
    T value = 0;

    #undef LOAD_SHARED_TILE_DB
    #define LOAD_SHARED_TILE_DB(TILE, BUFF) \
    { \
        tileA[(BUFF)][ty * TILESIZE + tx] = A[row * N + ((TILE) * TILESIZE + tx)]; \
        tileB[(BUFF)][ty * TILESIZE + tx] = B[((TILE) * TILESIZE + ty) * M + col]; \
    }

    // Load the first tile into buffer 0.
    LOAD_SHARED_TILE_DB(0, 0);
    __syncthreads(); 
    
    const int numTiles = ROUNDUP(N, TILESIZE);

    #pragma unroll 128 // bounded as the compiler takes too long if > 128.
    for (int t = 0; t < numTiles; ++t) {

        int buff = t & 1; // t % 2

        // Prefetch next tile into the other buffer if within bounds.
        if (t + 1 < numTiles)
            LOAD_SHARED_TILE_DB(t + 1, buff ^ 1);

        // Compute local product for current tile and accumulate over value.
        #pragma unroll 16
        for (int k = 0; k < TILESIZE; k++) {
            value += tileA[buff][ty * TILESIZE + k] * tileB[buff][tx + k * TILESIZE];
        }

        __syncthreads();
    }

    // Same as original kernel but moved outside tile loop,
    // since the 'value' is already accumulated over all tiles.
    // Also if this check encapsulates the loading of tiles, it
    // might limit the number of active threads needed to scan
    // global matrices if TILESIZE > M or TILESIZE > N.
    if (row < M && col < M) {
        C[row * M + col] = value;
    }
}

#define BENCHMARK_OPT2_KERNEL(NAME, TYPE, TILE, BX, BY) \
do { \
    dim3 NAME##Block((BX), (BY)); \
    if (NAME##Block.x * NAME##Block.y == 0 || NAME##Block.x * NAME##Block.y > 1024) { \
        std::cerr << "Error: invalid block size: (" << \
            NAME##Block.x << ", " << NAME##Block.y << ")" << std::endl; \
        break; \
    } \
    dim3 NAME##Grid(ROUNDUP(M, NAME##Block.x), ROUNDUP(M, NAME##Block.y)); \
    float NAME##_elapsed = 0.0f; \
    const int NAME##dynSmemSize = 0; \
    GENERATE_KERNEL_CONFIG(matrixMul_tiled_db, NAME, TYPE, TILE) \
    const int NAME##smemSize = 2 * (2 * TILE * TILE) * sizeof(TYPE); \
    std::string KERNELNAME = #NAME; \
    const size_t pos = KERNELNAME.find("_"); \
    if (pos != std::string::npos) KERNELNAME.replace(pos, 1, "-"); \
    table_row(KERNELNAME.c_str(), M, N, TILE, NAME##smemSize, NAME##Block, NAME##_elapsed, \
        checkMulResults(hC, hC_basic, dimC) ? "PASSED" : "FAILED"); \
} while (0);

