#pragma once

#include "../global.cuh"

/**
 * Optimization 5: Register Tiling with Vectorization, Double Buffering, and Warp tiling
 * -------------------------------------------------------------------------------------
 * Same as above but with warp-level tiling for improved locality and ILP.
 */
#define OPT5_BM 128
#define OPT5_BN 128
#define OPT5_WM 32
#define OPT5_WN 64
#define OPT5_RM 4
#define OPT5_RN 4
constexpr int OPT5_NUM_WARP_TILES_M = OPT5_BM / OPT5_WM;
constexpr int OPT5_NUM_WARP_TILES_N = OPT5_BN / OPT5_WN;
constexpr int OPT5_NUM_WARP_TILES = OPT5_NUM_WARP_TILES_M * OPT5_NUM_WARP_TILES_N;
constexpr int OPT5_VEC = 4;
static_assert((OPT5_BN % OPT5_VEC) == 0);
static_assert((OPT5_WN % OPT5_VEC) == 0);
static_assert((OPT5_RN % OPT5_VEC) == 0);
constexpr int OPT5_BN_VEC = OPT5_BN / OPT5_VEC;
constexpr int OPT5_RN_VEC = OPT5_RN / OPT5_VEC;
constexpr int OPT5_NUM_THREADS_M = 4; // tunable.
constexpr int OPT5_NUM_THREADS_N = 8; // tunable.
constexpr int OPT5_THREADS_PER_WARP = OPT5_NUM_THREADS_M * OPT5_NUM_THREADS_N;
static_assert(OPT5_THREADS_PER_WARP == 32);
constexpr int OPT5_WARP_STEP_M = OPT5_NUM_THREADS_M * OPT5_RM;    // rows covered per warp-step
constexpr int OPT5_WARP_STEP_N = OPT5_NUM_THREADS_N * OPT5_RN;    // cols covered per warp-step
constexpr int OPT5_WARP_STEP_N_VEC = OPT5_WARP_STEP_N / OPT5_VEC;   // float4 columns per warp-step
static_assert(OPT5_WM % OPT5_WARP_STEP_M == 0);
static_assert(OPT5_WN % OPT5_WARP_STEP_N == 0);
constexpr int OPT5_WARP_M_ITERS = OPT5_WM / OPT5_WARP_STEP_M;
constexpr int OPT5_WARP_N_ITERS = OPT5_WN / OPT5_WARP_STEP_N;

template <typename T, int TILESIZE>
__global__ void matrixMul_tiled_db_reg_warp(const T *A, const T *B, T *C) {

    assert(blockDim.y == 1);

    int tx = threadIdx.x;
    int lane_id = tx & 31;  // tid % 32
    int warp_id = tx >> 5;  // tid / 32

    // Warp tile indices within the shared tile.
    int warp_tile_row = warp_id / OPT5_NUM_WARP_TILES_N;
    int warp_tile_col = warp_id % OPT5_NUM_WARP_TILES_N;

    int warp_offset_row = warp_tile_row * OPT5_WM;
    int warp_offset_col = warp_tile_col * OPT5_WN;
    int warp_offset_col_vec = warp_offset_col / OPT5_VEC;

    // Thread indices within a warp.
    int lane_offset_row = (lane_id / OPT5_NUM_THREADS_N) * OPT5_RM;
    int lane_offset_col = (lane_id % OPT5_NUM_THREADS_N) * OPT5_RN;
    int lane_offset_col_vec = lane_offset_col / OPT5_VEC;

    // Shared memory tiles for A and B with double buffering.
    constexpr int TILESIZE_VEC = TILESIZE / OPT5_VEC;
    __shared__ float4 tileA[2][TILESIZE_VEC * OPT5_BM];      // Transposed: [kvec][row]
    __shared__ float4 tileB[2][TILESIZE * OPT5_BN_VEC];   // [k][col_vec]

    #define VECTORIZE(ARRAY) reinterpret_cast<float4*>(&(ARRAY))
    #define VECTORIZE_const(ARRAY) reinterpret_cast<const float4*>(&(ARRAY))

    #undef LOAD_SHARED_TILE
    #define LOAD_SHARED_TILE(TILE, BUFF) \
    { \
        /* Loads OPT5_BM x TILESIZE into shared memory (transposed into float4) */ \
        for (int i = tx; i < TILESIZE_VEC * OPT5_BM; i += blockDim.x) { \
            int shared_row = i % OPT5_BM; \
            int kvec = i / OPT5_BM; \
            tileA[BUFF][kvec * OPT5_BM + shared_row] = \
                *VECTORIZE_const(A[(by_offset + shared_row) * N + ((TILE) * TILESIZE + kvec * OPT5_VEC)]); \
        } \
        /* Loads TILESIZE x OPT5_BN into shared memory */ \
        for (int i = tx; i < TILESIZE * OPT5_BN_VEC; i += blockDim.x) { \
            int k = i / OPT5_BN_VEC; \
            int shared_col_vec = i % OPT5_BN_VEC; \
            tileB[BUFF][k * OPT5_BN_VEC + shared_col_vec] = \
                *VECTORIZE_const(B[((TILE) * TILESIZE + k) * M + (bx_offset + shared_col_vec * OPT5_VEC)]); \
        } \
    }

    const int numTiles = ROUNDUP(N, TILESIZE);

    for (int by = blockIdx.y; by < ROUNDUP(M, OPT5_BM); by += gridDim.y) {
        for (int bx = blockIdx.x; bx < ROUNDUP(M, OPT5_BN); bx += gridDim.x) {

            int by_offset = by * OPT5_BM;
            int bx_offset = bx * OPT5_BN;

            LOAD_SHARED_TILE(0, 0);

            // Accumulators for all warp-tile iterations.
            float4 value[OPT5_WARP_M_ITERS][OPT5_WARP_N_ITERS][OPT5_RM][OPT5_RN_VEC];

            #pragma unroll
            for (int w_row = 0; w_row < OPT5_WARP_M_ITERS; ++w_row) {
                #pragma unroll
                for (int w_col = 0; w_col < OPT5_WARP_N_ITERS; ++w_col) {
                    #pragma unroll
                    for (int i = 0; i < OPT5_RM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < OPT5_RN_VEC; ++j) {
                            value[w_row][w_col][i][j] = {0.f, 0.f, 0.f, 0.f};
                        }
                    }
                }
            }

            __syncthreads();

            #pragma unroll 64
            for (int t = 0; t < numTiles; ++t) {

                int buff = t & 1;
                if (t + 1 < numTiles)
                    LOAD_SHARED_TILE(t + 1, buff ^ 1);

                float4 aReg[OPT5_RM];
                float4 bReg[2][OPT5_RN_VEC];

                #pragma unroll
                for (int w_row = 0; w_row < OPT5_WARP_M_ITERS; ++w_row) {

                    int shared_row_offset = warp_offset_row + w_row * OPT5_WARP_STEP_M + lane_offset_row;

                    #pragma unroll
                    for (int w_col = 0; w_col < OPT5_WARP_N_ITERS; ++w_col) {

                        int shared_col_offset4 = warp_offset_col_vec + w_col * OPT5_WARP_STEP_N_VEC + lane_offset_col_vec;

                        #undef PREFETCH_B_REGS
                        #define PREFETCH_B_REGS(TILE, BUFF) \
                        { \
                            for (int j = 0; j < OPT5_RN_VEC; ++j) { \
                                bReg[(BUFF)][j] = tileB[buff][(TILE) * OPT5_BN_VEC + (shared_col_offset4 + j)]; \
                            } \
                        }

                        // Prefetch initial B reg-tile.
                        PREFETCH_B_REGS(0, 0);

                        #pragma unroll
                        for (int kvec = 0; kvec < TILESIZE_VEC; kvec++) {

                            // Load A reg-tile.
                            {  
                                #pragma unroll
                                for (int i = 0; i < OPT5_RM; ++i) {
                                    aReg[i] = tileA[buff][kvec * OPT5_BM + (shared_row_offset + i)];
                                }
                            }

                            #pragma unroll
                            for (int v = 0; v < 4; ++v) {
                                int k = kvec * 4 + v;
                                int bBuff = k & 1;

                                if (k + 1 < TILESIZE) {
                                    PREFETCH_B_REGS(k + 1, (bBuff ^ 1));
                                }

                                #pragma unroll
                                for (int i = 0; i < OPT5_RM; ++i) {
                                    float a = (v == 0) ? aReg[i].x :
                                              (v == 1) ? aReg[i].y :
                                              (v == 2) ? aReg[i].z : aReg[i].w;

                                    #pragma unroll
                                    for (int j = 0; j < OPT5_RN_VEC; ++j) {
                                        value[w_row][w_col][i][j].x += a * bReg[bBuff][j].x;
                                        value[w_row][w_col][i][j].y += a * bReg[bBuff][j].y;
                                        value[w_row][w_col][i][j].z += a * bReg[bBuff][j].z;
                                        value[w_row][w_col][i][j].w += a * bReg[bBuff][j].w;
                                    }
                                }
                            }
                        } // kvec
                    } // w_col
                } // w_row

                __syncthreads();
            } // t


            #pragma unroll
            for (int w_row = 0; w_row < OPT5_WARP_M_ITERS; ++w_row) {
                int row_offset = by_offset + warp_offset_row + w_row * OPT5_WARP_STEP_M + lane_offset_row;
                #pragma unroll
                for (int w_col = 0; w_col < OPT5_WARP_N_ITERS; ++w_col) {
                    int col_offset = bx_offset + warp_offset_col + w_col * OPT5_WARP_STEP_N + lane_offset_col;
                    #pragma unroll
                    for (int i = 0; i < OPT5_RM; ++i) {
                        int row = row_offset + i;
                        #pragma unroll
                        for (int j = 0; j < OPT5_RN_VEC; ++j) {
                            int col = col_offset + j * OPT5_VEC;
                            if (row < M && col + 3 < M) {
                                *VECTORIZE(C[row * M + col]) = value[w_row][w_col][i][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

#define BENCHMARK_OPT5_KERNEL(NAME, TYPE, TILE, BX, BY) \
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
    GENERATE_KERNEL_CONFIG(matrixMul_tiled_db_reg_warp, NAME, TYPE, TILE) \
    const int NAME##smemSize = 2 * ((TILE / OPT5_VEC) * OPT5_BM + TILE * OPT5_BN_VEC) * sizeof(TYPE); \
    std::string KERNELNAME = #NAME; \
    const size_t pos = KERNELNAME.find("_"); \
    if (pos != std::string::npos) KERNELNAME.replace(pos, 1, "-"); \
    table_row(KERNELNAME.c_str(), M, N, TILE, NAME##smemSize, NAME##Block, NAME##_elapsed, \
        checkMulResults(hC, hC_basic, dimC) ? "PASSED" : "FAILED"); \
} while (0);

