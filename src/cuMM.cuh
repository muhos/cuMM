#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

// For random number generation.
#include <random>

// Error and correctness handling.
#include "utils/check.h"

// For GPU timer (based on events).
#include "utils/timer.h"

// For performance table formatting.
#include "utils/table.h"

// For helper macros/functions.
#include "utils/helper.h"

#define M 4096  // Number of rows in Matrix A and C
#define N 10240 // Number of columns in Matrix B and C


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

/**
 * Optimization 1: Double buffering of shared memory tiles
 * -------------------------------------------------------
 * While computing on one tile, prefetch the next tile into another buffer.
 * This helps hide global memory latency if the kernel is memory-bound which
 * is the case matrixMulHIP_tiled. More shared memory load instructions 'ld.shared'
 * are issued in the TILESIZE loop vs 1 FMA instruction.
 */
template <typename T, int TILESIZE> 
__global__ 
void matrixMul_tiled_db(const T *A, const T *B, T *C);

/**
 * Optimization 2: Register Tiling
 * -------------------------------
 * Each thread computes a tile of C elements and stores them in registers.
 * This significantly reduces shared memory loads by a factor of REG_TILESIZE^2.
 * Kernel becomes less memory-bound and more compute-bound.
 */
#define OPT2_BM 128
#define OPT2_BN 128
#define OPT2_REG_TILESIZE 8

template <typename T, int TILESIZE> 
__global__ 
void matrixMulHIP_tiled_db_reg(const T *A, const T *B, T *C);

/**
 * Optimization 3: Register Tiling with Vectorization and Double Buffering
 * -----------------------------------------------------------------------
 * Same as above but with vectorized loads and stores to further reduce memory instructions.
 */
#define OPT3_BM 128
#define OPT3_BN 128
#define OPT3_REG_TILESIZE 8
#define OPT3_VSIZE 4

template <typename T, int TILESIZE> 
__global__
void matrixMul_tiled_db_reg_vec(T *A, T *B, T *C);

/**
 * Optimization 4: Register Tiling with Vectorization, Double Buffering, and Warp tiling
 * -------------------------------------------------------------------------------------
 * Same as above but with warp-level tiling for improved locality and ILP.
 */
#define OPT4_BM 128
#define OPT4_BN 128
#define OPT4_WM 32
#define OPT4_WN 64
#define OPT4_RM 4
#define OPT4_RN 4
constexpr int OPT4_NUM_WARP_TILES_M = OPT4_BM / OPT4_WM;
constexpr int OPT4_NUM_WARP_TILES_N = OPT4_BN / OPT4_WN;
constexpr int OPT4_NUM_WARP_TILES = OPT4_NUM_WARP_TILES_M * OPT4_NUM_WARP_TILES_N;
constexpr int OPT4_VEC = 4;
static_assert((OPT4_BN % OPT4_VEC) == 0);
static_assert((OPT4_WN % OPT4_VEC) == 0);
static_assert((OPT4_RN % OPT4_VEC) == 0);
constexpr int OPT4_BN_4 = OPT4_BN / OPT4_VEC;
constexpr int OPT4_RN_4 = OPT4_RN / OPT4_VEC;
constexpr int OPT4_NUM_THREADS_M = 4; // tunable.
constexpr int OPT4_NUM_THREADS_N = 8; // tunable.
constexpr int OPT4_THREADS_PER_WARP = OPT4_NUM_THREADS_M * OPT4_NUM_THREADS_N;
static_assert(OPT4_THREADS_PER_WARP == 32);
constexpr int OPT4_WARP_STEP_M = OPT4_NUM_THREADS_M * OPT4_RM;    // rows covered per warp-step
constexpr int OPT4_WARP_STEP_N = OPT4_NUM_THREADS_N * OPT4_RN;    // cols covered per warp-step
constexpr int OPT4_WARP_STEP_N_4 = OPT4_WARP_STEP_N / OPT4_VEC;   // float4 columns per warp-step
static_assert(OPT4_WM % OPT4_WARP_STEP_M == 0);
static_assert(OPT4_WN % OPT4_WARP_STEP_N == 0);
constexpr int OPT4_WARP_M_ITERS = OPT4_WM / OPT4_WARP_STEP_M;
constexpr int OPT4_WARP_N_ITERS = OPT4_WN / OPT4_WARP_STEP_N;

template <typename T, int TILESIZE>
__global__ 
void matrixMul_tiled_db_reg_vec_warp(const T *A, const T *B, T *C);

