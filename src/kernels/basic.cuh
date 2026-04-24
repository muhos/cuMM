#pragma once

#include "../global.cuh"

/**
 * Kernel to perform matrix multiplication C = A × B
 * A is of size M x N, B is N x M, and C is M x M
 * Each thread computes one element of matrix C.
 */
__global__ 
void matrixMul(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < M) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * M + col];
        }
        C[row * M + col] = value;
    }
}


#define BENCHMARK_BASIC_KERNEL(NAME) \
do { \
    dim3 NAME##Block(16, 16); \
    dim3 NAME##Grid(ROUNDUP(M, NAME##Block.x), ROUNDUP(M, NAME##Block.y)); \
    float basic_elapsed = 0.0f; \
    MeasureTime(basic_elapsed, timer, launchKernel(matrixMul, NAME##Grid, NAME##Block, 0, 0, dA, dB, dC)); \
    checkErrors(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy C"); \
    table_row(#NAME, M, N, 0, 0, NAME##Block, basic_elapsed, "na"); \
    checkErrors(cudaMemcpy(hC_basic, dC, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy C to C_basic"); \
} while (0)
