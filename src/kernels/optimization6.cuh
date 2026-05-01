#pragma once

#include "../global.cuh"
#include <mma.h>
using namespace nvcuda;

/**
 * Optimization 6: Tensor Core with Register Tiling, Vectorization, Warp tiling, and Pipelining
 * --------------------------------------------------------------------------------------------
 * Significantly more complex but achieves much higher performance by utilizing Tensor Cores 
 * with WMMA instructions, while also employing register tiling, vectorized memory access, 
 * warp-level tiling, and pipelining with asynchronous shared memory loads.
 * Tiling is fully unrolled, and the code is structured to maximize ILP and to minimize register pressure.
 * Shared memory is bank conflict free for both A and B tiles based on index swizzling (fine-tuned
 * based on feedback from Nsight compute).
 */

// Block tile (scalar units).
#define BM 128
#define BN 64

// Warp tile (scalar units).
#define WM 32
#define WN 32

constexpr int NUM_WARP_TILES_M = BM / WM;
constexpr int NUM_WARP_TILES_N = BN / WN;
constexpr int NUM_WARP_TILES = NUM_WARP_TILES_M * NUM_WARP_TILES_N;

constexpr int VEC = 4;

// WMMA tile sizes (TF32)
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 8;

constexpr int WARP_MMA_M_ITERS = WM / MMA_M;
constexpr int WARP_MMA_N_ITERS = WN / MMA_N;

#define WMMA_LOAD_A_M16N16K8_ROW(a0,a1,a2,a3, ptr, ld)                       \
    asm volatile(                                                                 \
        "wmma.load.a.sync.aligned.row.m16n16k8.tf32 {%0,%1,%2,%3}, [%4], %5;\n"   \
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)                                 \
        : "l"((unsigned long long)(ptr)), "r"(ld)                                \
    )

#define WMMA_LOAD_B_M16N16K8_COL(b0,b1,b2,b3, ptr, ld)                       \
    asm volatile(                                                                 \
        "wmma.load.b.sync.aligned.col.m16n16k8.tf32 {%0,%1,%2,%3}, [%4], %5;\n"   \
        : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)                                 \
        : "l"((unsigned long long)(ptr)), "r"(ld)                                \
    )

#define WMMA_LOAD_B_M16N16K8_ROW(b0,b1,b2,b3, ptr, ld)                       \
    asm volatile(                                                                 \
        "wmma.load.b.sync.aligned.row.m16n16k8.tf32 {%0,%1,%2,%3}, [%4], %5;\n"   \
        : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)                                 \
        : "l"(ptr), "r"(ld)                                \
    )

#define WMMA_MMA_M16N16K8_ROWCOL(                                     \
    d0,d1,d2,d3,d4,d5,d6,d7,                                                    \
    a0,a1,a2,a3,                                                                \
    b0,b1,b2,b3)                                                                \
    asm volatile(                                                               \
      "wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32 "               \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, "                                             \
      "{%8,%9,%10,%11}, "                                                      \
      "{%12,%13,%14,%15}, "                                                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                           \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7) \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),                                    \
        "r"(b0), "r"(b1), "r"(b2), "r"(b3)                                     \
    )

#define WMMA_MMA_M16N16K8_ROWROW(                                     \
    d0,d1,d2,d3,d4,d5,d6,d7,                                                    \
    a0,a1,a2,a3,                                                                \
    b0,b1,b2,b3)                                                                \
    asm volatile(                                                               \
      "wmma.mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32 "               \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, "                                             \
      "{%8,%9,%10,%11}, "                                                      \
      "{%12,%13,%14,%15}, "                                                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                           \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7) \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),                                    \
        "r"(b0), "r"(b1), "r"(b2), "r"(b3)                                     \
    )

#define STG64(ptr, a, b) \
    asm volatile("st.global.v2.f32 [%0], {%1,%2};" :: \
        "l"(ptr), "f"(a), "f"(b) : "memory")

#define STS128(ptr, a,b,c,d) \
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: \
        "r"(ptr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory")

__device__ __forceinline__ int SWZ_A(const int& r, const int& k) {
    return k ^ (r & 4);
}

__device__ __forceinline__ int SWZ_B(const int& c) {
    return c ^ ((c & 8) >> 2);
}

#define LDS_F32(out_f, smem_u32) \
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(out_f) : "r"(smem_u32));

#define CVT_TF32(out_r, in_f) \
{ \
    unsigned int temp = __float_as_int(in_f); \
    temp &= 0xFFFFE000; /* Zero out the lower 13 bits */ \
    out_r = temp; \
}

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

constexpr int PIPE_DEPTH = 3;

template <typename T, int TILESIZE>
__global__ 
void matrixMul_tiled_db_reg_warp_tc(const T* __restrict__ A,
                                                  const T* __restrict__ B,
                                                  float* __restrict__ C) {
    assert(blockDim.y == 1);

    int tx      = threadIdx.x;
    int warp_id = tx >> 5;
    int warp_tile_row = warp_id / NUM_WARP_TILES_N; // NUM_WARP_TILES_M = 128/32 = 4 warp-tile rows
    int warp_tile_col = warp_id % NUM_WARP_TILES_N; // NUM_WARP_TILES_N = 64/32 = 2 warp-tile cols
    int warp_offset_row = warp_tile_row * WM;       // 0, 32, 64, 96
    int warp_offset_col = warp_tile_col * WN;       // 0, 32

    constexpr int LDB = TILESIZE + 4;               // padding by 4 achieved 0 load bank conflicts for tileB

    __shared__ float tileA[PIPE_DEPTH][BM * TILESIZE];       // row-major: [row][k]
    __shared__ float tileB[PIPE_DEPTH][BN * LDB];            // transposed: [col][k] so WMMA can load col-major

    constexpr int TILESIZE4 = TILESIZE / 4;
    constexpr int BN4 = BN / 4;

    #define VECTORIZE(ARRAY) reinterpret_cast<float4*>(&(ARRAY))
    #define VECTORIZE_const(ARRAY) reinterpret_cast<const float4*>(&(ARRAY))

    const int numTiles = ROUNDUP(N, TILESIZE);

    for (int by = blockIdx.y, yblocks = ROUNDUP(M, BM); by < yblocks; by += gridDim.y) {
        for (int bx = blockIdx.x, xblocks = ROUNDUP(M, BN); bx < xblocks; bx += gridDim.x) {

            int by_offset = by * BM;
            int bx_offset = bx * BN;

            auto load_tile = [&](const int& TILE, const int& BUFF)
            {
                for (int i = tx; i < BM * TILESIZE4; i += blockDim.x) { 
                    int r = i / TILESIZE4; 
                    int k4 = (i % TILESIZE4) * VEC; 
                    unsigned smemA = (unsigned)__cvta_generic_to_shared(&tileA[BUFF][r * TILESIZE + SWZ_A(r, k4)]); 
                    CP_ASYNC_CG(smemA, VECTORIZE_const(A[(by_offset + r) * N + ((TILE) * TILESIZE + k4)]), 16); 
                }
                for (int i = tx; i < BN4 * TILESIZE; i += blockDim.x) {
                    int k = i % TILESIZE;
                    int c4 = (i / TILESIZE) * VEC; 
                    const float4& b = *VECTORIZE_const(B[((TILE) * TILESIZE + k) * M + (bx_offset + c4)]); 
                    unsigned smemB_0 = (unsigned)__cvta_generic_to_shared(&tileB[BUFF][SWZ_B(c4) * LDB + k]); 
                    unsigned smemB_1 = (unsigned)__cvta_generic_to_shared(&tileB[BUFF][SWZ_B(c4 + 1) * LDB + k]); 
                    unsigned smemB_2 = (unsigned)__cvta_generic_to_shared(&tileB[BUFF][SWZ_B(c4 + 2) * LDB + k]); 
                    unsigned smemB_3 = (unsigned)__cvta_generic_to_shared(&tileB[BUFF][SWZ_B(c4 + 3) * LDB + k]); 
                    CP_ASYNC_CA(smemB_0, &b.x, 4); 
                    CP_ASYNC_CA(smemB_1, &b.y, 4); 
                    CP_ASYNC_CA(smemB_2, &b.z, 4); 
                    CP_ASYNC_CA(smemB_3, &b.w, 4); 
                }
            };

            for (int preload = 0; preload < PIPE_DEPTH - 1 && preload < numTiles; ++preload) {
                load_tile(preload, preload);
                CP_ASYNC_COMMIT_GROUP();
            }

            CP_ASYNC_WAIT_GROUP(PIPE_DEPTH - 2); // tile 0 ready, tile 1 is in flight

            // Per-warp accumulators for this warp tile (supports WM/WN > 16)
            float c00_0=0.f, c00_1=0.f, c00_2=0.f, c00_3=0.f, c00_4=0.f, c00_5=0.f, c00_6=0.f, c00_7=0.f;
            float c01_0=0.f, c01_1=0.f, c01_2=0.f, c01_3=0.f, c01_4=0.f, c01_5=0.f, c01_6=0.f, c01_7=0.f;
            float c10_0=0.f, c10_1=0.f, c10_2=0.f, c10_3=0.f, c10_4=0.f, c10_5=0.f, c10_6=0.f, c10_7=0.f;
            float c11_0=0.f, c11_1=0.f, c11_2=0.f, c11_3=0.f, c11_4=0.f, c11_5=0.f, c11_6=0.f, c11_7=0.f;

            // Main K-tiling loop
            for (int t = 0; t < numTiles; ++t) {

                __syncthreads();

                const int next_t = t + PIPE_DEPTH - 1; // the tile index that will be loaded next after this iteration
                if (next_t < numTiles) {
                    const int next_buff = next_t % PIPE_DEPTH; // modulo for ping-pong buffering
                    load_tile(next_t, next_buff);
                    CP_ASYNC_COMMIT_GROUP();
                    CP_ASYNC_WAIT_GROUP(PIPE_DEPTH - 2); // 1 tile is in flight (pending)
                }
                else {
                    CP_ASYNC_WAIT_GROUP(0); // for last tile
                }

                __syncthreads();

                const int buff = t % PIPE_DEPTH;

                // For TF32 m16n16k8, the A tile A[row][k] is 16 (rows) x 8 (k) = 128 scalars.
                // A warp has 32 lanes, and in this mapping each lane provides 4 scalars,
                // so the warp covers 32 x 4 = 128 scalars = the whole A tile.
                //
                // We view the warp lanes as an 8 x 4 grid (after reversing the wmma.load.a):
                //   lane_row = lane >> 2  -> 8 groups (0..7) selecting the "row" within [0..7]
                //   lane_col = lane &  3  -> 4 elements (0..3) selecting the "k-col" within [0..3]
                //
                // Each lane then covers a (2 x 2) patch of the A tile by taking:
                //   rows: { lane_row, lane_row + 8 }      (covers 16 rows total)
                //   k:    { lane_col, lane_col + 4 }      (covers 8 k total)
                // giving 2 x 2 = 4 scalars per lane.

                // For A, we use the same mapping as WMMA require B to be transposed to
                // to B[col][k]. That way we keep columns continguous in B over k. Thus, same 
                // lane indices are reused for B access.
                int lane = tx & 31;
                int lane_row   = lane >> 2;   // 0..7
                int lane_col   = lane & 3;    // 0..3

                // Since each block is decomposed into warp tiles 
                // of WM x WN, and each warp computes one warp tile,
                // we need to offset TF32 tiles by warp_offset_col{0,1}
                // in B and warp_offset_row{0,1,2,3} in A.

                // Load B first to be shared with As
                // Forst MMA tile columns (first half of the warp tile)
                // Since WN=32 and MMA-tile is 16 columns wide
                uint32_t b0r0, b0r1, b0r2, b0r3;
                {
                    int bc0 = warp_offset_col + lane_row;
                    int bc1 = bc0 + 8;
                    int bc0s = SWZ_B(bc0);
                    int bc1s = SWZ_B(bc1);

                    unsigned smB0   = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc0s * LDB + lane_col]);
                    unsigned smB1   = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc0s * LDB + (lane_col + 4)]);
                    unsigned smB2   = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc1s * LDB + lane_col]);
                    unsigned smB3   = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc1s * LDB + (lane_col + 4)]);

                    float bf0, bf1, bf2, bf3;
                    LDS_F32(bf0, smB0);
                    LDS_F32(bf1, smB1);
                    LDS_F32(bf2, smB2);
                    LDS_F32(bf3, smB3);

                    CVT_TF32(b0r0, bf0);
                    CVT_TF32(b0r1, bf1);
                    CVT_TF32(b0r2, bf2);
                    CVT_TF32(b0r3, bf3);
                }
                // Second MMA tile columns (second half of the warp tile)
                uint32_t b1r0, b1r1, b1r2, b1r3;
                {
                    int bc0_1 = warp_offset_col + MMA_N + lane_row;
                    int bc1_1 = bc0_1 + 8;
                    int bc0_1s = SWZ_B(bc0_1);
                    int bc1_1s = SWZ_B(bc1_1);

                    unsigned smB0_1 = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc0_1s * LDB + lane_col]);
                    unsigned smB1_1 = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc0_1s * LDB + (lane_col + 4)]);
                    unsigned smB2_1 = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc1_1s * LDB + lane_col]);
                    unsigned smB3_1 = (unsigned)__cvta_generic_to_shared(&tileB[buff][bc1_1s * LDB + (lane_col + 4)]);

                    float bf0_1, bf1_1, bf2_1, bf3_1;
                    LDS_F32(bf0_1, smB0_1);
                    LDS_F32(bf1_1, smB1_1);
                    LDS_F32(bf2_1, smB2_1);
                    LDS_F32(bf3_1, smB3_1);

                    CVT_TF32(b1r0, bf0_1);
                    CVT_TF32(b1r1, bf1_1);
                    CVT_TF32(b1r2, bf2_1);
                    CVT_TF32(b1r3, bf3_1);
                }

                // A0B0 A0B1
                {
                    // First MMA tile rows (first half of the warp tile, WM = 32)
                    int ar0 = warp_offset_row + lane_row;
                    int ar1 = ar0 + 8;
                    int k0  = SWZ_A(ar0, lane_col);
                    int k4  = SWZ_A(ar0, lane_col + 4);
                    int k0b = SWZ_A(ar1, lane_col);
                    int k4b = SWZ_A(ar1, lane_col + 4);
                    unsigned smA0 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar0 * TILESIZE + k0]);
                    unsigned smA1 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar1 * TILESIZE + k0b]);
                    unsigned smA2 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar0 * TILESIZE + k4]);
                    unsigned smA3 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar1 * TILESIZE + k4b]);

                    float af0, af1, af2, af3;
                    LDS_F32(af0, smA0);
                    LDS_F32(af1, smA1);
                    LDS_F32(af2, smA2);
                    LDS_F32(af3, smA3);

                    uint32_t a0r0, a0r1, a0r2, a0r3;
                    CVT_TF32(a0r0, af0);
                    CVT_TF32(a0r1, af1);
                    CVT_TF32(a0r2, af2);
                    CVT_TF32(a0r3, af3);

                    WMMA_MMA_M16N16K8_ROWCOL(
                        c00_0, c00_1, c00_2, c00_3, c00_4, c00_5, c00_6, c00_7,
                        a0r0, a0r1, a0r2, a0r3,
                        b0r0, b0r1, b0r2, b0r3
                    );

                    WMMA_MMA_M16N16K8_ROWCOL(
                        c01_0, c01_1, c01_2, c01_3, c01_4, c01_5, c01_6, c01_7,
                        a0r0, a0r1, a0r2, a0r3,
                        b1r0, b1r1, b1r2, b1r3
                    );
                }
                // A1B0 A1B1
                {
                    // Second MMA tile rows (second half of the warp tile)
                    int ar0_1 = warp_offset_row + MMA_M + lane_row;
                    int ar1_1 = warp_offset_row + MMA_M + lane_row + 8;
                    int k0  = SWZ_A(ar0_1, lane_col);
                    int k4  = SWZ_A(ar0_1, lane_col + 4);
                    int k0b = SWZ_A(ar1_1, lane_col);
                    int k4b = SWZ_A(ar1_1, lane_col + 4);
                    unsigned smA0_1 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar0_1 * TILESIZE + k0]);
                    unsigned smA1_1 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar1_1 * TILESIZE + k0b]);
                    unsigned smA2_1 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar0_1 * TILESIZE + k4]);
                    unsigned smA3_1 = (unsigned)__cvta_generic_to_shared(&tileA[buff][ar1_1 * TILESIZE + k4b]);

                    float af0_1, af1_1, af2_1, af3_1;
                    LDS_F32(af0_1, smA0_1);
                    LDS_F32(af1_1, smA1_1);
                    LDS_F32(af2_1, smA2_1);
                    LDS_F32(af3_1, smA3_1);

                    uint32_t a1r0, a1r1, a1r2, a1r3;
                    CVT_TF32(a1r0, af0_1);
                    CVT_TF32(a1r1, af1_1);
                    CVT_TF32(a1r2, af2_1);
                    CVT_TF32(a1r3, af3_1);

                    WMMA_MMA_M16N16K8_ROWCOL(
                        c10_0, c10_1, c10_2, c10_3, c10_4, c10_5, c10_6, c10_7,
                        a1r0, a1r1, a1r2, a1r3,
                        b0r0, b0r1, b0r2, b0r3
                    );

                    WMMA_MMA_M16N16K8_ROWCOL(
                        c11_0, c11_1, c11_2, c11_3, c11_4, c11_5, c11_6, c11_7,
                        a1r0, a1r1, a1r2, a1r3,
                        b1r0, b1r1, b1r2, b1r3
                    );
                }
            }

            // Store results
            // For m16n16k8, the accumulator fragment is distributed across the warp,
            // so that each lane owns 8 output scalars (regs c00_0..c00_7). 
            // Those 8 scalars correspond to a 2 x 4 micro-tile inside the 16 x 16
            // access as (note that columns are read in pairs)
            //           2 rows: r and r + 8
            //           4 cols: c, c + 1, c + 8, c + 9
            { 
                int lane = tx & 31;
                int lane_row = lane >> 2;          // 0..7
                int lane_col = (lane & 3) << 1;    // 0,2,4,6
                // << 2: * sizeof(float)
                uintptr_t lane_index = (uintptr_t)(lane_row * M + lane_col) << 2;
                uintptr_t C0 = lane_index + (uintptr_t)(C + (by_offset + warp_offset_row) * M + (bx_offset + warp_offset_col));
                constexpr uintptr_t row_8 = (8 * M) << 2;
                constexpr uintptr_t col_8 = 8 << 2;
                constexpr uintptr_t row_16= (16 * M) << 2;
                constexpr uintptr_t col_16= 16 << 2;
                // First quadrant of the warp tile (col + 0)
                STG64(C0,                 c00_0, c00_1); // row r,   cols c0,c0+1
                STG64(C0 + row_8,         c00_2, c00_3); // row r+8, cols c0,c0+1
                STG64(C0 + col_8,         c00_4, c00_5); // row r,   cols c0+8,c0+9
                STG64(C0 + row_8 + col_8, c00_6, c00_7); // row r+8, cols c0+8,c0+9
                // Second quadrant of the warp tile (col + 16)
                STG64(C0 + col_16,                  c01_0, c01_1);
                STG64(C0 + col_16  + row_8,         c01_2, c01_3);
                STG64(C0 + col_16  + col_8,         c01_4, c01_5);
                STG64(C0 + col_16  + row_8 + col_8, c01_6, c01_7);
                // Third quadrant of the warp tile (row + 16, col + 0)
                STG64(C0 + row_16,                 c10_0, c10_1);
                STG64(C0 + row_16 + row_8,         c10_2, c10_3);
                STG64(C0 + row_16 + col_8,         c10_4, c10_5);
                STG64(C0 + row_16 + row_8 + col_8, c10_6, c10_7);
                // Fourth quadrant of the warp tile (row + 16, col + 16)
                STG64(C0 + row_16 + col_16,                 c11_0, c11_1);
                STG64(C0 + row_16 + col_16 + row_8,         c11_2, c11_3);
                STG64(C0 + row_16 + col_16 + col_8,         c11_4, c11_5);
                STG64(C0 + row_16 + col_16 + row_8 + col_8, c11_6, c11_7);
            }
        } //bx
    } // by
}

#define BENCHMARK_OPT6_KERNEL(NAME, TYPE, TILE, BX, BY, GX, GY) \
float NAME##_elapsed = 0.0f; \
do { \
    dim3 NAME##Block((BX), (BY)); \
    if (NAME##Block.x * NAME##Block.y == 0 || NAME##Block.x * NAME##Block.y > 1024) { \
        std::cerr << "Error: invalid block size: (" << \
            NAME##Block.x << ", " << NAME##Block.y << ")" << std::endl; \
        break; \
    } \
    dim3 NAME##Grid(GX, GY); \
    const int NAME##dynSmemSize = 0; \
    GENERATE_KERNEL_CONFIG(matrixMul_tiled_db_reg_warp_tc, NAME, TYPE, TILE) \
    const int NAME##smemSize = PIPE_DEPTH * (BM * TILE + BN * (TILE + 4)) * sizeof(TYPE); \
    std::string KERNELNAME = #NAME; \
    const size_t pos = KERNELNAME.find("_"); \
    if (pos != std::string::npos) KERNELNAME.replace(pos, 1, "-"); \
    table_row(KERNELNAME.c_str(), M, N, TILE, \
        NAME##smemSize, NAME##Block, NAME##Grid, \
        NAME##_elapsed, \
        checkMulResults(hC, hC_basic, dimC, 1e-3) ? "PASSED" : "FAILED"); \
} while (0);

