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
constexpr int OPT5_BN_4 = OPT5_BN / OPT5_VEC;
constexpr int OPT5_RN_4 = OPT5_RN / OPT5_VEC;
constexpr int OPT5_NUM_THREADS_M = 4; // tunable.
constexpr int OPT5_NUM_THREADS_N = 8; // tunable.
constexpr int OPT5_THREADS_PER_WARP = OPT5_NUM_THREADS_M * OPT5_NUM_THREADS_N;
static_assert(OPT5_THREADS_PER_WARP == 32);
constexpr int OPT5_WARP_STEP_M = OPT5_NUM_THREADS_M * OPT5_RM;    // rows covered per warp-step
constexpr int OPT5_WARP_STEP_N = OPT5_NUM_THREADS_N * OPT5_RN;    // cols covered per warp-step
constexpr int OPT5_WARP_STEP_N_4 = OPT5_WARP_STEP_N / OPT5_VEC;   // float4 columns per warp-step
static_assert(OPT5_WM % OPT5_WARP_STEP_M == 0);
static_assert(OPT5_WN % OPT5_WARP_STEP_N == 0);
constexpr int OPT5_WARP_M_ITERS = OPT5_WM / OPT5_WARP_STEP_M;
constexpr int OPT5_WARP_N_ITERS = OPT5_WN / OPT5_WARP_STEP_N;

template <typename T, int TILESIZE>
__global__ void matrixMulHIP_tiled_db_reg_warp(const T *A, const T *B, T *C) {

    assert(blockDim.y == 1);

    int tx = threadIdx.x;
    int lane_id = tx & 31;  // tid % 32
    int warp_id = tx >> 5;  // tid / 32

    // Warp tile indices within the shared tile.
    int warp_tile_row = warp_id / NUM_WARP_TILES_N;
    int warp_tile_col = warp_id % NUM_WARP_TILES_N;

    int warp_offset_row = warp_tile_row * WM;
    int warp_offset_col = warp_tile_col * WN;
    int warp_offset_col4 = warp_offset_col / VEC;

    // Thread indices within a warp.
    int lane_offset_row = (lane_id / NUM_THREADS_N) * RM;
    int lane_offset_col = (lane_id % NUM_THREADS_N) * RN;
    int lane_offset_col4 = lane_offset_col / VEC;

    // Shared memory tiles for A and B with double buffering.
    constexpr int TILESIZE4 = TILESIZE / 4;
    __shared__ float4 tileA[2][TILESIZE4 * (BM + PAD_A)];      // Transposed: [k4][row]
    __shared__ float4 tileB[2][TILESIZE * (BN_4 + PAD_B)];     // [k][col4]

    #define VECTORIZE(ARRAY) reinterpret_cast<float4*>(&(ARRAY))
    #define VECTORIZE_const(ARRAY) reinterpret_cast<const float4*>(&(ARRAY))

    #undef LOAD_SHARED_TILE
    #define LOAD_SHARED_TILE(TILE, BUFF) \
    { \
        /* Loads BM x TILESIZE into shared memory (transposed into float4) */ \
        for (int i = tx; i < TILESIZE4 * BM; i += blockDim.x) { \
            int shared_row = i % BM; \
            int k4 = i / BM; \
            tileA[BUFF][k4 * (BM + PAD_A) + shared_row] = \
                *VECTORIZE_const(A[(by_offset + shared_row) * N + ((TILE) * TILESIZE + k4 * VEC)]); \
        } \
        /* Loads TILESIZE x BN into shared memory */ \
        for (int i = tx; i < TILESIZE * BN_4; i += blockDim.x) { \
            int k = i / BN_4; \
            int shared_col4 = i % BN_4; \
            tileB[BUFF][k * (BN_4 + PAD_B) + shared_col4] = \
                *VECTORIZE_const(B[((TILE) * TILESIZE + k) * M + (bx_offset + shared_col4 * 4)]); \
        } \
    }

    const int numTiles = ROUNDUP(N, TILESIZE);

    for (int by = blockIdx.y; by < ROUNDUP(M, BM); by += gridDim.y) {
        for (int bx = blockIdx.x; bx < ROUNDUP(M, BN); bx += gridDim.x) {

            int by_offset = by * BM;
            int bx_offset = bx * BN;

            LOAD_SHARED_TILE(0, 0);

            // Accumulators for all warp-tile iterations.
            float4 value[WARP_M_ITERS][WARP_N_ITERS][RM][RN_4];

            #pragma unroll
            for (int w_row = 0; w_row < WARP_M_ITERS; ++w_row) {
                #pragma unroll
                for (int w_col = 0; w_col < WARP_N_ITERS; ++w_col) {
                    #pragma unroll
                    for (int i = 0; i < RM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < RN_4; ++j) {
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

                float4 aReg[RM];
                float4 bReg[2][RN_4];

                #pragma unroll
                for (int w_row = 0; w_row < WARP_M_ITERS; ++w_row) {

                    int shared_row_offset = warp_offset_row + w_row * WARP_STEP_M + lane_offset_row;

                    #pragma unroll
                    for (int w_col = 0; w_col < WARP_N_ITERS; ++w_col) {

                        int shared_col_offset4 = warp_offset_col4 + w_col * WARP_STEP_N_4 + lane_offset_col4;

                        #undef PREFETCH_B_REGS
                        #define PREFETCH_B_REGS(TILE, BUFF) \
                        { \
                            for (int j = 0; j < RN_4; ++j) { \
                                bReg[(BUFF)][j] = tileB[buff][(TILE) * (BN_4 + PAD_B) + (shared_col_offset4 + j)]; \
                            } \
                        }

                        // Prefetch initial B reg-tile.
                        PREFETCH_B_REGS(0, 0);

                        #pragma unroll
                        for (int k4 = 0; k4 < TILESIZE4; k4++) {

                            // Load A reg-tile.
                            {  
                                #pragma unroll
                                for (int i = 0; i < RM; ++i) {
                                    aReg[i] = tileA[buff][k4 * (BM + PAD_A) + (shared_row_offset + i)];
                                }
                            }

                            #pragma unroll
                            for (int v = 0; v < 4; ++v) {
                                int k = k4 * 4 + v;
                                int bBuff = k & 1;

                                if (k + 1 < TILESIZE) {
                                    PREFETCH_B_REGS(k + 1, (bBuff ^ 1));
                                }

                                #pragma unroll
                                for (int i = 0; i < RM; ++i) {
                                    float a = (v == 0) ? aReg[i].x :
                                              (v == 1) ? aReg[i].y :
                                              (v == 2) ? aReg[i].z : aReg[i].w;

                                    #pragma unroll
                                    for (int j = 0; j < RN_4; ++j) {
                                        value[w_row][w_col][i][j].x += a * bReg[bBuff][j].x;
                                        value[w_row][w_col][i][j].y += a * bReg[bBuff][j].y;
                                        value[w_row][w_col][i][j].z += a * bReg[bBuff][j].z;
                                        value[w_row][w_col][i][j].w += a * bReg[bBuff][j].w;
                                    }
                                }
                            }
                        } // k4
                    } // w_col
                } // w_row

                __syncthreads();
            } // t


            #pragma unroll
            for (int w_row = 0; w_row < WARP_M_ITERS; ++w_row) {
                int row_offset = by_offset + warp_offset_row + w_row * WARP_STEP_M + lane_offset_row;
                #pragma unroll
                for (int w_col = 0; w_col < WARP_N_ITERS; ++w_col) {
                    int col_offset = bx_offset + warp_offset_col + w_col * WARP_STEP_N + lane_offset_col;
                    #pragma unroll
                    for (int i = 0; i < RM; ++i) {
                        int row = row_offset + i;
                        #pragma unroll
                        for (int j = 0; j < RN_4; ++j) {
                            int col = col_offset + j * VEC;
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
    GENERATE_KERNEL_CONFIG(matrixMul_tiled_db_reg_vec, NAME, TYPE, TILE) \
    const int NAME##smemSize = 2 * ((TILE / OPT5_VEC) * OPT5_BM + TILE * OPT5_BN_VEC) * sizeof(TYPE); \
    std::string KERNELNAME = #NAME; \
    const size_t pos = KERNELNAME.find("_"); \
    if (pos != std::string::npos) KERNELNAME.replace(pos, 1, "-"); \
    table_row(KERNELNAME.c_str(), M, N, TILE, NAME##smemSize, NAME##Block, NAME##_elapsed, \
        checkMulResults(hC, hC_basic, dimC) ? "PASSED" : "FAILED"); \
} while (0);

