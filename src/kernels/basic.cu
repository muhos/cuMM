#include "../cuMM.cuh"

__global__ 
void matrixMulHIP(float *A, float *B, float *C) {
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