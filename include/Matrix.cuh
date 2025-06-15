#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda_runtime.h>

// 检查 CUDA 错误
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void matrixMulKernel(double* A, double* B, double* result, size_t A_size, size_t B_size, size_t result_size);


#endif // MATRIX_MUL_CUH