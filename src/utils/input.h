#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

inline 
void printUsage(const char* name) {
    std::cerr << "Usage: " << name << " [<input_file>] [<output_file>]\n";
    std::cerr << "If input file is not provided, input matrices are randomized.\n";
    std::cerr << "If output file is not provided, result will not be written.\n";
    std::cerr << "Otherwise provide an input file with this format:\n";
    std::cerr << "- Contains (M * N) elements of matrix A (row-major order)\n";
    std::cerr << "- Followed by (N * M) elements of matrix B (row-major order)\n";
    std::cerr << "- All values are whitespace-separated (space or newline)\n";
    std::cerr << "\nExample for small matrices (M=2, N=3):\n";
    std::cerr << " 1 2 3 4 5 6 # A: 2x3 matrix (flattened)\n";
    std::cerr << " 7 8 9 10 11 12 # B: 3x2 matrix (flattened)\n";
}

inline 
int initializeInput(float* hA, float* hB, 
                    const int& dimA, const int& dimB, 
                    const int& argc, char** argv) 
{
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
        const int RANGE = 100;
        for (int i = 0; i < dimA; ++i) {
            hA[i] = static_cast<float>(generator() % RANGE);
            hB[i] = static_cast<float>(generator() % RANGE);
        }
        std::cout << " done." << std::endl;
    }

    return 0;
}