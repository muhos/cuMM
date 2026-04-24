#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <cmath>
#include <iostream>

#define KBYTE (1 << 10)
#define MBYTE (1 << 20)
#define GBYTE (1 << 30)

// Function to check HIP errors after API calls.
inline void checkErrors(const cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "Error: " << message << " : " << cudaGetErrorString(error) <<
        std::endl;
        exit(error);
    }
}

// Function to check matrix dimensions.
template <typename T>
inline bool checkMatrixDimensions(const T& M, const T& N) {
    if (M <= 0 || N <= 0) {
        std::cerr << "Error: Matrix dimensions must be positive (M: " << M << ", N: " << N << ")" << std::endl;
        return false;
    }
    const size_t max_size = static_cast<size_t>(std::numeric_limits<T>::max());
    if (static_cast<size_t>(M) * static_cast<size_t>(N) > max_size) {
        std::cerr << "Error: Matrix size overflow for dimensions (M: " << M << ", N: " << N << ")" << std::endl;
        return false;
    }
    return true;
}

// Function to compare two matrices up to some accuracy and verify correctness.
template <typename T>
inline bool checkMulResults(
    const T* hC,
    const T* hC_ref,
    int size,
    double rtol = 5e-4,
    double atol = 1e-2
) {
    for (int idx = 0; idx < size; ++idx) {
        const double x = static_cast<double>(hC[idx]);
        const double y = static_cast<double>(hC_ref[idx]);
        const double diff = fabs(x - y);
        const double tol  = atol + rtol * fabs(y);

        if (diff > tol) {
            std::cerr << "Mismatch at " << idx
                      << ": GPU " << x << ", REF " << y
                      << " | diff " << diff << " tol " << tol
                      << " (rel " << (diff / (fabs(y) + 1e-30)) << ")"
                      << std::endl;
            return false;
        }
    }
    return true;
}

inline void checkCUBLAS(const cublasStatus_t& status, const char* message) {
    std::string status_str = "Error: " + std::string(message) + " failed due to: ";
    if (status != CUBLAS_STATUS_SUCCESS) {
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                status_str += "CUBLAS not initialized";
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                status_str += "CUBLAS allocation failed";
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                status_str += "CUBLAS invalid value";
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                status_str += "CUBLAS architecture mismatch";
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                status_str += "CUBLAS mapping error";
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                status_str += "CUBLAS execution failed";
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                status_str += "CUBLAS internal error";
                break;
            default:
                status_str += "Unknown CUBLAS error";
        }
        std::cerr << status_str << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to check GPU availability and properties.
inline int checkGPUAvailability() {
    int devCount = 0;
    checkErrors(cudaGetDeviceCount(&devCount), "Unable to get GPU device count");
    if (!devCount) {
        std::cerr << "Error: No GPU device found" << std::endl;
        return 0;
    }
    std::cout << "Number of GPU devices available: " << devCount << std::endl;
    cudaDeviceProp devProp;
    checkErrors(cudaGetDeviceProperties(&devProp, 0), "Unable to get GPU device properties");
    std::cout << "Device " << 0 << ": " << devProp.name << std::endl;
    std::cout << "  Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "  Total global memory: " << devProp.totalGlobalMem / MBYTE << " MB" << std::endl;
    std::cout << "  Shared memory per block: " << devProp.sharedMemPerBlock / KBYTE << " KB" << std::endl;
    std::cout << "  Warp size: " << devProp.warpSize << std::endl;
    std::cout << "  Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
    return devCount;
}

inline bool checkGPUMemoryAvailability(const size_t& required_mem) {
    size_t free_mem = 0, total_mem = 0;
    checkErrors(cudaMemGetInfo(&free_mem, &total_mem), "Unable to get GPU memory info");
    if (!total_mem) {
        std::cerr << "Error: Unable to determine GPU memory" << std::endl;
        return false;
    }
    if (free_mem < required_mem) {
        std::cerr << "Error: Not enough GPU memory. Available: " << free_mem / MBYTE << " MB"
                  << ", Required: " << required_mem / MBYTE << " MB" << std::endl;
        return false;
    }
    std::cout << "GPU memory available: " << free_mem / MBYTE << " MB"
              << " / " << total_mem / MBYTE << " MB" << std::endl;
    return true;
}


