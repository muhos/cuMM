#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <string>

#define M 4096  // Number of rows in Matrix A and C
#define N 10240 // Number of columns in Matrix B and C
