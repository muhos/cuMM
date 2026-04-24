#include "kernels/basic.cuh"
#include "kernels/shared.cuh"

int main(int argc, char* argv[]) {

    header("cuMM: Fast Matrix Multiplication on GPUs");

    // If 2 arguments are provided, the first is input file and second is output file.
    // If 1 argument is provided, the first is output file and input matrices are randomized.
    // If no arguments are provided, input matrices are randomized but result is not written to output file.
    if (argc > 3) {
        std::cerr << "Usage: " << argv[0] << " [<input_file>] [<output_file>]\n";
        std::cerr << "If input file is not provided, input matrices are randomized.\n";
        std::cerr << "If output file is not provided, result will not be written.\n";
        std::cerr << "Otherwise provide an input file with this format:\n";
        std::cerr << "- Contains (M * N) elements of matrix A (row-major order)\n";
        std::cerr << "- Followed by (N * M) elements of matrix B (row-major order)\n";
        std::cerr << "- All values are whitespace-separated (space or newline)\n";
        std::cerr << "\nExample for small matrices (M=2, N=3):\n";
        std::cerr << " 1 2 3 4 5 6 # A: 2x3 matrix (flattened)\n";
        std::cerr << " 7 8 9 10 11 12 # B: 3x2 matrix (flattened)\n";
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
    if (argc == 3) {
        const char* inputFileName = argv[1];
        std::ifstream inputFile(inputFileName);
        if (!inputFile) {
            std::cerr << "Failed to open input file." << std::endl;
            return 1;
        }

        for (int i = 0; i < dimA; ++i) {
            if (!(inputFile >> hA[i])) {
                std::cerr << "Error reading matrix A from file." << std::endl;
                return 1;
            }
        }

        for (int i = 0; i < dimB; ++i) {
            if (!(inputFile >> hB[i])) {
                std::cerr << "Error reading matrix B from file." << std::endl;
                return 1;
            }
        }

        inputFile.close();
    }
    else {
        std::cout << "Generating random input matrices A and B..";
        constexpr int SEED = 2025;
        std::mt19937 generator(SEED);
        const int RANGE = M;
        for (int i = 0; i < dimA; ++i) {
            hA[i] = static_cast<float>(generator() % RANGE);
            hB[i] = static_cast<float>(generator() % RANGE);
        }
        std::cout << " done." << std::endl;
    }

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

    table_header("Performance Evaluation");

    BENCHMARK_BASIC_KERNEL(Basic);

    for (int tileSize = 16; tileSize <= 32; tileSize *= 2) {
        BENCHMARK_OPT1_KERNEL(Tiled, float, tileSize, tileSize, tileSize);
    }

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