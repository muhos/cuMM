#include "bench.cuh"
#include "basic.cuh"
#include "optimization1.cuh"
#include "optimization2.cuh"
#include "optimization3.cuh"

void run_benchmarks(
    float* hA, 
    float* hB, 
    float* hC, 
    float* hC_basic, 
    float* dA, 
    float* dB, 
    float* dC, 
    int dimC, 
    gpuTimer& timer) 
{
    const int sizeC = dimC * sizeof(float);
    table_header("Performance Evaluation");
    BENCHMARK_BASIC_KERNEL(Basic);
    for (int kTileSize : {16, 32}) {
        BENCHMARK_OPT1_KERNEL(Tiled, float, 
            // k-tile size
            kTileSize, 
            // block size (x, y)
            kTileSize, kTileSize 
        );
    }
    for (int kTileSize : {16, 32}) {
        BENCHMARK_OPT2_KERNEL(Tiled_DB, float, 
            // k-tile size
            kTileSize,
            // block size (x, y)
            kTileSize, kTileSize 
        );
    }
    for (int kTileSize : {8, 16}) {
        BENCHMARK_OPT3_KERNEL(Tiled_DB_Reg, float, 
            kTileSize, // k-tile size
            // block size (x, y)
            OPT3_BM / OPT3_REG_TILESIZE, OPT3_BN / OPT3_REG_TILESIZE
        );
    }
}
