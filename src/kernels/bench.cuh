#include "../utils/table.h"
#include "../utils/timer.h"
#include "../utils/check.h"
#include "../utils/helper.h"

void run_benchmarks(
    float* hA, 
    float* hB, 
    float* hC, 
    float* hC_basic, 
    float* dA, 
    float* dB, 
    float* dC, 
    int dimC, 
    gpuTimer& timer);