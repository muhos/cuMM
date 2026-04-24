#pragma once

#include "../global.cuh"

/**
 * Optimization 3: Register Tiling
 * -------------------------------
 * Each thread computes a tile of C elements and stores them in registers.
 * This significantly reduces shared memory loads by a factor of OPT3_REG_TILESIZE^2.
 * Kernel becomes less memory-bound and more compute-bound.
 */
#define OPT3_BM 128
#define OPT3_BN 128
#define OPT3_REG_TILESIZE 8

template <typename T, int TILESIZE> 
__global__ 
void matrixMul_tiled_db_reg(const T *A, const T *B, T *C) {

    // Thread indices to access elements within a tile.
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Block indices to fetch tiles from global memory.
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Global row and column indices.
    int by_offset = by * OPT3_BM;
    int bx_offset = bx * OPT3_BN;

    // Shared memory tiles for A and B with double buffering.
    __shared__ float tileA[2][OPT3_BM * TILESIZE];
    __shared__ float tileB[2][TILESIZE * OPT3_BN];

    #undef LOAD_SHARED_TILE_DB
    #define LOAD_SHARED_TILE_DB(TILE, BUFF) \
    { \
    for (int x = tx; x < TILESIZE; x += blockDim.x) { /* shared-tile column */ \
        for (int i = 0; i < OPT3_REG_TILESIZE; ++i) { /* register-tile row  */ \
                int tReg = ty * OPT3_REG_TILESIZE + i; \
                int row = by_offset + tReg; \
                tileA[(BUFF)][tReg * TILESIZE + x] = A[row * N + ((TILE) * TILESIZE + x)]; \
            } \
        } \
        for (int y = ty; y < TILESIZE; y += blockDim.y) { /* shared-tile row */ \
            for (int i = 0; i < OPT3_REG_TILESIZE; ++i) { /* register-tile column */ \
                int tReg = tx * OPT3_REG_TILESIZE + i; \
                int col = bx_offset + tReg; \
                tileB[(BUFF)][y * OPT3_BN + tReg] = B[((TILE) * TILESIZE + y) * M + col]; \
            } \
        } \
    }

    LOAD_SHARED_TILE_DB(0, 0);

    // Create the register tile for accumulation in C.
    T value[OPT3_REG_TILESIZE][OPT3_REG_TILESIZE] = { 0 };

    __syncthreads(); 
    
    const int numTiles = ROUNDUP(N, TILESIZE);

    #pragma unroll 64 // bounded as the compiler takes too long if > 128.
    for (int t = 0; t < numTiles; ++t) {

        int buff = t & 1; // t % 2

        if (t + 1 < numTiles)
            LOAD_SHARED_TILE_DB(t + 1, buff ^ 1);

        // Load from shared memory tiles to register tiles and compute.
        // This significantly reduces shared memory accesses.
        T aReg[OPT3_REG_TILESIZE];
        T bReg[OPT3_REG_TILESIZE];
        #pragma unroll
        for (int k = 0; k < TILESIZE; k++) {
            #pragma unroll
            for (int i = 0; i < OPT3_REG_TILESIZE; ++i) {
                int tReg = ty * OPT3_REG_TILESIZE + i;
                aReg[i] = tileA[buff][tReg * TILESIZE + k];
            }
            #pragma unroll
            for (int i = 0; i < OPT3_REG_TILESIZE; ++i) {
                int tReg = tx * OPT3_REG_TILESIZE + i;
                bReg[i] = tileB[buff][k * OPT3_BM + tReg];
            }
            #pragma unroll
            for (int i = 0; i < OPT3_REG_TILESIZE; ++i) {
                for (int j = 0; j < OPT3_REG_TILESIZE; ++j) {
                    value[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write the register tile back to global memory.
    for (int i = 0; i < OPT3_REG_TILESIZE; ++i) {
        int row = by_offset + ty * OPT3_REG_TILESIZE + i;
        for (int j = 0; j < OPT3_REG_TILESIZE; ++j) {
            int col = bx_offset + tx * OPT3_REG_TILESIZE + j;
            if (row < M && col < M)
                C[row * M + col] = value[i][j];
        }
    }
}

#define BENCHMARK_OPT3_KERNEL(NAME, TYPE, TILE, BX, BY) \
do { \
    dim3 NAME##Block((BX), (BY)); \
    if (NAME##Block.x * NAME##Block.y == 0 || NAME##Block.x * NAME##Block.y > 1024) { \
        std::cerr << "Error: invalid block size: (" << \
            NAME##Block.x << ", " << NAME##Block.y << ")" << std::endl; \
        break; \
    } \
    dim3 NAME##Grid(ROUNDUP(M, OPT3_BN), ROUNDUP(M, OPT3_BM)); \
    float NAME##_elapsed = 0.0f; \
    const int NAME##dynSmemSize = 0; \
    GENERATE_KERNEL_CONFIG(matrixMul_tiled_db_reg, NAME, TYPE, TILE) \
    const int NAME##smemSize = 2 * (OPT3_BM * TILE + OPT3_BN * TILE) * sizeof(TYPE); \
    std::string KERNELNAME = #NAME; \
    const size_t pos = KERNELNAME.find("_"); \
    if (pos != std::string::npos) KERNELNAME.replace(pos, 1, "-"); \
        table_row(KERNELNAME.c_str(), M, N, TILE, \
        NAME##smemSize, NAME##Block, NAME##Grid, \
        NAME##_elapsed, \
        checkMulResults(hC, hC_basic, dimC) ? "PASSED" : "FAILED"); \
} while (0);

