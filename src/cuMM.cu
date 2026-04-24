#include "utils/input.h"
#include "kernels/bench.cuh"
#include "global.cuh"

int main(int argc, char* argv[]) {

    header("cuMM: Fast Matrix Multiplication on GPUs");

    // If 2 arguments are provided, the first is input file and second is output file.
    // If 1 argument is provided, the first is output file and input matrices are randomized.
    // If no arguments are provided, input matrices are randomized but result is not written to output file.
    if (argc > 3) {
        printUsage(argv[0]);
        return 1;
    }

    if (!checkMatrixDimensions(M, N))
        return EXIT_FAILURE;

    if (!checkGPUAvailability())
        return EXIT_FAILURE;

    gpuTimer timer;

    const int dimA = M * N;
    const int dimB = N * M;
    const int dimC = M * M;
    const int sizeA = dimA * sizeof(float);
    const int sizeB = dimB * sizeof(float);
    const int sizeC = dimC * sizeof(float);

    std::cout << "Allocating host memory of (" << (sizeA + sizeB + 2 * sizeC) / (1024 * 1024) << " MB)..";

    float *hA = new float[dimA];
    float *hB = new float[dimB];
    float *hC = new float[dimC];

    // For testing correctness of optimized kernels.
    float* hC_basic = new float[dimC];

    std::cout << " done." <<  std::endl;

    // Read matrices A and B from file iff an input file is provided
    // If file does not exist std::ifstream or std::ofstream will fail.
    if (initializeInput(hA, hB, dimA, dimB, argc, argv)) return EXIT_FAILURE;

    const size_t memBytes = sizeA + sizeB + sizeC;

    if (!checkGPUMemoryAvailability(memBytes)) {
        if (hA != nullptr) free(hA);
        if (hB != nullptr) free(hB);
        if (hC != nullptr) free(hC);
        if (hC_basic != nullptr) free(hC_basic);
        return EXIT_FAILURE;
    }

    const size_t memMB = (sizeA + sizeB + sizeC) / MBYTE;
    std::cout << "Allocating device memory of (" << memMB << " MB) and sending input matrices to device..";

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    checkErrors(cudaMalloc(&dA, sizeA), "cudaMalloc for A");
    checkErrors(cudaMalloc(&dB, sizeB), "cudaMalloc for B");
    checkErrors(cudaMalloc(&dC, sizeC), "cudaMalloc for C");

    checkErrors(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkErrors(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice), "cudaMemcpy B");

    std::cout << " done." << std::endl;

    run_benchmarks(hA, hB, hC, hC_basic, dA, dB, dC, dimC, timer);

    std::cout << "Cleaning up..";

    if (hA != nullptr) free(hA);
    if (hB != nullptr) free(hB);
    if (hC != nullptr) free(hC);
    if (hC_basic != nullptr) free(hC_basic);
    if (dA != nullptr) checkErrors(cudaFree(dA), "Free A");
    if (dB != nullptr) checkErrors(cudaFree(dB), "Free B");
    if (dC != nullptr) checkErrors(cudaFree(dC), "Free C");

    std::cout << " done." << std::endl;

    return 0;
}