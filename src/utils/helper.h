
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define ROUNDUP(x, y) (((x) + (y) - 1) / (y))

#define launchKernel(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...) \
do {                                                                            \
    kernelName<<<numBlocks, numThreads, memPerBlock, streamId>>>(__VA_ARGS__);  \
} while (0)

#define MeasureTime(ELAPSED, TIMER, KERNEL) \
do { \
    TIMER.start(); \
    KERNEL; \
    TIMER.stop(); \
    ELAPSED = TIMER.elapsed(); \
} while (0)
