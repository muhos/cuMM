#pragma once

#include "../global.cuh"
#include <cublas_v2.h>

#define CUBLAS_DEFAULT checkCUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha, dB, M, dA, N, &beta, dC, M), "SGEMM")
#define CUBLAS_TENSOR  checkCUBLAS( \
        cublasGemmEx( \
            handle, \
            CUBLAS_OP_N, CUBLAS_OP_N, \
            M, M, N, \
            &alpha, \
            dB, CUDA_R_32F, M, \
            dA, CUDA_R_32F, N, \
            &beta, \
            dC, CUDA_R_32F, M, \
            CUBLAS_COMPUTE_32F_FAST_TF32, \
            CUBLAS_GEMM_DEFAULT_TENSOR_OP \
        ), \
        "cublasGemmEx TensorCore" \
    )

#define BENCHMARK_CUBLAS(CALL, NAME) \
do { \
    checkErrors(cudaMemset(dC, 0, sizeC), "cudaMemset C (cuBLAS)"); \
    float total_elapsed = 0.0f, cublas_elapsed = 0.0f; \
    for (int i = 0; i < 1; i++) { \
        float alpha = 1.0f, beta  = 0.0f; \
        timer.start(); \
        CALL; \
        checkErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize (cuBLAS)"); \
        timer.stop(); \
        total_elapsed += timer.elapsed(); \
    } \
    cublas_elapsed = total_elapsed / NRUNS; \
    checkErrors(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy C"); \
    std::string KERNELNAME = #NAME; \
    const size_t pos = KERNELNAME.find("_"); \
    if (pos != std::string::npos) KERNELNAME.replace(pos, 1, "-"); \
    table_row(KERNELNAME.c_str(), M, N, 0, 0, dim3(0,0), dim3(0,0), cublas_elapsed, \
        checkMulResults(hC, hC_basic, M * M, 1e-3) ? "PASSED" : "FAILED"); \
} while (0);
