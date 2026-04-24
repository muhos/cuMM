#include "shared.cuh"

template <typename T, int TILESIZE> 
__global__ 
void matrixMul_tiled(const T *A, const T *B, T *C) {

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

    // Shared memory tiles for A and B
    __shared__ T tileA[TILESIZE * TILESIZE];
    __shared__ T tileB[TILESIZE * TILESIZE];

    // Initialize the accumulation variable for C element.
    // Moved outside the tile loop to accumulate over all tiles.
    T value = 0;
    
    // This loop functions as k-loop inside but it accumulates across tiles,
    // to cover the entire width of A and height of B. Without it, we would
    // be computing the inner product of just one tile of A and B.
    #pragma unroll 128 // bounded as the compiler takes too long if > 128.
    for (int t = 0; t < ROUNDUP(N, TILESIZE); ++t) {

        // Load elements of A and B into shared memory tiles.
        // (by * TILESIZE) * N: index of row-tile in A.
        // (t * TILESIZE):      index of column-tile in A.
        // (bx * TILESIZE):     index of column-tile in B.
        // (t * TILESIZE) * M:  index of row-tile in B.
        // ty * N + tx:         offset within the tile for A.
        // ty * M + tx:         offset within the tile for B.
        // ty * TILESIZE + tx:  offset within shared memory tile.
        // A index: (by * TILE_SIZE) * N + (t * TILE_SIZE)     + ty * N + tx
        // B index: (bx * TILE_SIZE)     + (t * TILE_SIZE) * M + ty * M + tx
        // These indices are simplified below.
        tileA[ty * TILESIZE + tx] = A[row * N + (t * TILESIZE + tx)];
        tileB[ty * TILESIZE + tx] = B[(t * TILESIZE + ty) * M + col];
    
        // Synchronize to ensure all threads loaded their elements into shared memory.
        __syncthreads();
    
        // Compute local product for current tile and accumulate over value.
        #pragma unroll 32
        for (int k = 0; k < TILESIZE; ++k) {
            value += tileA[ty * TILESIZE + k] * tileB[tx + k * TILESIZE];
        }
    
        // Synchronize again to ensure threads read from shared memory.
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