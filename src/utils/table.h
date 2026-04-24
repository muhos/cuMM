#pragma once

#include <iostream>
#include <iomanip>
#include <string>

#define RULER_WIDTH 110

inline void table_ruler(const int& width, const char& ch = '-', const bool& newline = true) {
    for (int i = 0; i < width; ++i) std::cout << ch;
    if (newline) std::cout << std::endl;
}

inline void header(const char* title) {
    const int start_width = 6;
    table_ruler(start_width, '-', false);
    const std::string head = "[ " + std::string(title) + " ]";
    const int width = head.length();
    std::cout << head;
    if (RULER_WIDTH >= (width + start_width))
        table_ruler(RULER_WIDTH - (width + start_width), '=');
}

// Constants for maximum lengths of table columns.
#define KERNEL_NAME_MAX_LEN 20
#define MATRIX_HEIGHT_MAX_LEN 6
#define MATRIX_WIDTH_MAX_LEN 6
#define TILE_SIZE_MAX_LEN 10
#define SHARED_SIZE_MAX_LEN 10
#define BLOCK_SIZE_MAX_LEN 10
#define TIME_MS_MAX_LEN 10
#define GFLOPS_MAX_LEN 6
#define CHECK_MAX_LEN 6

// Function to print a table header.
inline void table_header(const char* title) {
    header(title);
    std::cout   << " " << std::setw(KERNEL_NAME_MAX_LEN) << std::left << "Kernel (float)" << " | "
                << std::setw(MATRIX_HEIGHT_MAX_LEN) << std::right << "M" << " | "
                << std::setw(MATRIX_WIDTH_MAX_LEN) << std::right << "N" << " | "
                << std::setw(TILE_SIZE_MAX_LEN) << std::right << "K-Tile Size" << " | "
                << std::setw(SHARED_SIZE_MAX_LEN) << std::right << "Shared Mem" << " | "
                << std::setw(BLOCK_SIZE_MAX_LEN) << std::left << "Block Size" << " | "
                << std::setw(TIME_MS_MAX_LEN) << std::right << "Time (ms)" << " | "
                << std::setw(GFLOPS_MAX_LEN) << std::right << "TFLOPS" << " | "
                << std::setw(CHECK_MAX_LEN) << std::left << "Check" << std::endl;
    table_ruler(RULER_WIDTH);
}

// Function to print a table row with performance results.
inline void table_row(const char* kernel, 
                        const int& M, const int& N, 
                        const int& tile_size, 
                        const int& shared_size,
                        const dim3& block, 
                        const float& time_ms, 
                        const char* check = "na") 
{
    const std::string blockstr = block.x ? "(" + std::to_string(block.x) + ", " + std::to_string(block.y) + ")" : "na";
    const std::string tile_size_str = tile_size ? std::to_string(tile_size) : "n/a";
    const std::string shared_size_str = shared_size ? std::to_string(shared_size) + " B" : "na";
    // two operations (add + mul).
    const double tflops = (2.0 * M * M * N) / (time_ms * 1e9);
    std::cout   << " " << std::setw(KERNEL_NAME_MAX_LEN) << std::left << kernel << " | "
                << std::setw(MATRIX_HEIGHT_MAX_LEN) << std::right << M << " | "
                << std::setw(MATRIX_WIDTH_MAX_LEN) << std::right << N << " | "
                << std::setw(TILE_SIZE_MAX_LEN) << std::right << tile_size_str << " | "
                << std::setw(SHARED_SIZE_MAX_LEN) << std::right << shared_size_str << " | "
                << std::setw(BLOCK_SIZE_MAX_LEN) << std::left << blockstr << " | "
                << std::setw(TIME_MS_MAX_LEN) << std::right << std::fixed << std::setprecision(2) << time_ms << " | "
                << std::setw(GFLOPS_MAX_LEN) << std::right << std::fixed << std::setprecision(2) << tflops << " | "
                << std::setw(CHECK_MAX_LEN) << std::left << check << std::endl;
}
