#include "matrix_mul.cuh"
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void matrixMulKernel(double* A, double* B, double* result, size_t A_size, size_t B_size, size_t result_size) 
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程在矩阵范围内
    if (row < A_size && col < B_size) {
        double sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        result[row * N + col] = sum;
    }
}

// 矩阵乘法函数
void matrixMultiply(double* A, double* B, double* result, size_t A_size, size_t B_size, size_t result_size) 
{

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));


    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError()); 
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}







template<temple T>
void apMem(T* dataPtr, int size)
{
    cudaMalloc()
}