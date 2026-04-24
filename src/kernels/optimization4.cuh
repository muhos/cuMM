#pragma once

#include "../global.cuh"

/**
 * Optimization 4: Register Tiling with Vectorization and Double Buffering
 * -----------------------------------------------------------------------
 * Same as above but with vectorized loads and stores to further reduce memory instructions.
 */
#define OPT4_BM 128
#define OPT4_BN 128
#define OPT4_REG_TILESIZE 8
#define OPT4_VEC 4

constexpr int OPT4_REG_TILESIZE_VEC = OPT4_REG_TILESIZE / OPT4_VEC;
constexpr int OPT4_BN_VEC = OPT4_BN / OPT4_VEC;

template <typename T, int TILESIZE> 
__global__ void matrixMul_tiled_db_reg_vec(T *A, T *B, T *C) {

    // Thread indices to access elements within a tile.
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Block indices to fetch tiles from global memory.
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Global row and column indices.
    int by_offset = by * OPT4_BM;
    int bx_offset = bx * OPT4_BN;

    constexpr int OPT4_TILESIZE_VEC = TILESIZE / OPT4_VEC;

    // Shared memory tiles for A and B with double buffering.
    __shared__ float4 tileA[2][OPT4_TILESIZE_VEC * OPT4_BM]; // Transposed.
    __shared__ float4 tileB[2][TILESIZE * OPT4_BN_VEC];

    #undef LOAD_SHARED_TILE_DB
    #define LOAD_SHARED_TILE_DB(TILE, BUFF) \
    { \
        for (int x = tx; x < OPT4_TILESIZE_VEC; x += blockDim.x) { /* we need this loop to cover all columns since threads only cover a subset in both dimensions. */ \
            for (int i = 0; i < OPT4_REG_TILESIZE; ++i) { \
                int tReg = ty * OPT4_REG_TILESIZE + i; \
                int row = by_offset + tReg; \
                tileA[(BUFF)][x * OPT4_BN + tReg] = *reinterpret_cast<float4*>( &A[row * N + ((TILE) * TILESIZE + x * OPT4_VEC)] ); \
            } \
        } \
        for (int y = ty; y < TILESIZE; y += blockDim.y) { /* we need this loop to cover all rows since threads only cover a subset in both dimensions. */ \
            for (int i = 0; i < OPT4_REG_TILESIZE_VEC; ++i) { \
                int tReg = tx * OPT4_REG_TILESIZE + i * OPT4_VEC; \
                int col = bx_offset + tReg; \
                tileB[(BUFF)][y * OPT4_BN_VEC + (tReg / OPT4_VEC)] = *reinterpret_cast<float4*>( &B[((TILE) * TILESIZE + y) * M + col] ); \
            } \
        } \
    }

    LOAD_SHARED_TILE_DB(0, 0);

    // Create the register tile for accumulation in C.
    float4 value[OPT4_REG_TILESIZE][OPT4_REG_TILESIZE_VEC] = { 0 };

    __syncthreads(); 
    
    const int numTiles = ROUNDUP(N, TILESIZE);

    #pragma unroll 64 // bounded as the compiler takes too long if > 128.
    for (int t = 0; t < numTiles; ++t) {

        int buff = t & 1;

        if (t + 1 < numTiles)
            LOAD_SHARED_TILE_DB(t + 1, buff ^ 1);

        // Load from shared memory tiles to register tiles and compute.
        // This significantly reduces shared memory accesses.
        float4 aReg[OPT4_REG_TILESIZE];
        float4 bReg[2][OPT4_REG_TILESIZE_VEC];

        #define PREFETCH_B_REGS(TILE, BUFF) \
        { \
            for (int i = 0; i < OPT4_REG_TILESIZE_VEC; ++i) { \
                int tReg = tx * OPT4_REG_TILESIZE + i * OPT4_VEC; \
                bReg[(BUFF)][i] = tileB[buff][(TILE) * OPT4_BN_VEC + (tReg / OPT4_VEC)]; \
            } \
        }

        PREFETCH_B_REGS(0, 0);

        #pragma unroll
        for (int k4 = 0; k4 < OPT4_TILESIZE_VEC; k4++) {
            #pragma unroll
            for (int i = 0; i < OPT4_REG_TILESIZE; ++i) {
                int tReg = ty * OPT4_REG_TILESIZE + i;
                aReg[i] = tileA[buff][k4 * OPT4_BN + tReg];
            }
            #pragma unroll
            for (int v = 0; v < OPT4_VEC; ++v) {
                int k = k4 * OPT4_VEC + v;
                int bBuff = k & 1;
                if (k + 1 < TILESIZE) {
                    PREFETCH_B_REGS(k + 1, (bBuff ^ 1));
                }
                #pragma unroll
                for (int i = 0; i < OPT4_REG_TILESIZE; ++i) {
                    float a = (v == 0) ? aReg[i].x :
                            (v == 1) ? aReg[i].y :
                            (v == 2) ? aReg[i].z : aReg[i].w;
                    #pragma unroll
                    for (int j = 0; j < OPT4_REG_TILESIZE_VEC; ++j) {
                        value[i][j].x += a * bReg[bBuff][j].x;
                        value[i][j].y += a * bReg[bBuff][j].y;
                        value[i][j].z += a * bReg[bBuff][j].z;
                        value[i][j].w += a * bReg[bBuff][j].w;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write the register tile back to global memory.
    #pragma unroll
    for (int i = 0; i < OPT4_REG_TILESIZE; ++i) {
        int row = by_offset + ty * OPT4_REG_TILESIZE + i;
        #pragma unroll
        for (int j = 0; j < OPT4_REG_TILESIZE_VEC; ++j) {
            int col = bx_offset + tx * OPT4_REG_TILESIZE + j * OPT4_VEC;
            if (row < M && col < M) {
                *reinterpret_cast<float4*>( &C[row * M + col] ) = value[i][j];
            }
        }
    }
}

#define BENCHMARK_OPT4_KERNEL(NAME, TYPE, TILE, BX, BY) \
do { \
    dim3 NAME##Block((BX), (BY)); \
    if (NAME##Block.x * NAME##Block.y == 0 || NAME##Block.x * NAME##Block.y > 1024) { \
        std::cerr << "Error: invalid block size: (" << \
            NAME##Block.x << ", " << NAME##Block.y << ")" << std::endl; \
        break; \
    } \
    dim3 NAME##Grid(ROUNDUP(M, OPT4_BN), ROUNDUP(M, OPT4_BM)); \
    float NAME##_elapsed = 0.0f; \
    const int NAME##dynSmemSize = 0; \
    GENERATE_KERNEL_CONFIG(matrixMul_tiled_db_reg_vec, NAME, TYPE, TILE) \
    const int NAME##smemSize = 2 * ((TILE / OPT4_VEC) * OPT4_BM + TILE * OPT4_BN_VEC) * sizeof(TYPE); \
    std::string KERNELNAME = #NAME; \
    const size_t pos = KERNELNAME.find("_"); \
    if (pos != std::string::npos) KERNELNAME.replace(pos, 1, "-"); \
        table_row(KERNELNAME.c_str(), M, N, TILE, \
        NAME##smemSize, NAME##Block, NAME##Grid, \
        NAME##_elapsed, \
        checkMulResults(hC, hC_basic, dimC) ? "PASSED" : "FAILED"); \
} while (0);

