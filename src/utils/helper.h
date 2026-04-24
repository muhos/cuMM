
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

constexpr int NRUNS = 10;

#define ROUNDUP(x, y) (((x) + (y) - 1) / (y))

#define launchKernel(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...) \
do {                                                                            \
    kernelName<<<numBlocks, numThreads, memPerBlock, streamId>>>(__VA_ARGS__);  \
} while (0)

#define MeasureTime(ELAPSED, TIMER, KERNEL) \
do { \
    KERNEL; /* warm up */ \
    float TOTAL = 0.0; \
    for (int i = 0; i < NRUNS; ++i) { \
        TIMER.start(); \
        KERNEL; \
        TIMER.stop(); \
        TOTAL += TIMER.elapsed(); \
    } \
    ELAPSED = TOTAL / float(NRUNS); \
} while (0)

#define LAUNCH_TEMPLATE_KERNEL(KERNEL, NAME, TYPE, TILE) \
{ \
    checkErrors(cudaMemset(dC, 0, sizeC), "cudaMemset C"); \
    MeasureTime(NAME##_elapsed, timer, \
        launchKernel((KERNEL<TYPE, TILE>), \
            NAME##Grid, NAME##Block, NAME##dynSmemSize, 0, (TYPE*)dA, (TYPE*)dB, dC)); \
}

#define GENERATE_KERNEL_CONFIG(KERNEL, NAME, TYPE, TILE) \
{ \
    switch (TILE) { \
        case 4:  LAUNCH_TEMPLATE_KERNEL(KERNEL, NAME, TYPE, 4); break; \
        case 8:  LAUNCH_TEMPLATE_KERNEL(KERNEL, NAME, TYPE, 8); break; \
        case 16:  LAUNCH_TEMPLATE_KERNEL(KERNEL, NAME, TYPE, 16); break; \
    } \
    checkErrors(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy C"); \
}
