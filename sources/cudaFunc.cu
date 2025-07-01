#include "CudaKernel/cudaFunc.cuh"
// #include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>



// 数组除一个相同datasize的数组
__global__ void Arrays_divide_arrays_Kernel(double* arr, const double n, const size_t _datasize)
{
    // 每个线程计算四个
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int smallIndex = 0; smallIndex < 4; ++smallIndex)
    {   
        size_t a_index = index * 4 + smallIndex;
        if (a_index < _datasize)
        {
            arr[a_index] = arr[a_index] / n;
        }
    }
}
void __cdecl Common::Arrays_divide_arrays(double* src, const double n, size_t _datasize)
{
    const int BlockthreadNum = 512;
    const int grid = (_datasize / (512 * 4)) + 1;

    dim3 blockSize(BlockthreadNum);
    dim3 gridSize(grid);
    Arrays_divide_arrays_Kernel<<<gridSize, blockSize>>>(src, n, _datasize);
    cudaDeviceSynchronize();
}





// 神经网络层误差的传递，
__global__ void AverageNextloss_kernel(double* loss, double* dst, const size_t batch_size, const size_t output_num)
{
    // 一个线程计算一个loss
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < output_num)
    {
        for (int b = 0; b < batch_size; ++b)
        {
            dst[index] += loss[b*output_num + index];
        }
        // printf("%f\n", dst[index]);
    }
}
void __cdecl FullconnecttionKernelFunc::AverageNextloss(double* loss, double* dst, const size_t batch_size, const size_t output_num)
{
    const int BlockthreadNum = 512;
    const int grid = (output_num / (512 * 4)) + 1;
    dim3 blockSize(BlockthreadNum);
    dim3 gridSize(grid);

    AverageNextloss_kernel<<<gridSize, blockSize>>>(loss, dst, batch_size, output_num);
    cudaDeviceSynchronize();
}







// 更新参数所作的操作
__global__ void updata_weight_kernel(double* w, double* grad, const size_t _datasize, const double lr)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < _datasize)
    {
        w[index] -= grad[index] * lr;
        // printf("w is %f, grad is %f\n", w[index], grad[index]);
    }
}
void __cdecl Common::update_weight(double* w, double* grad, const size_t _datasize, const double lr)
{
    const int BlockthreadNum = 512;
    const int grid = (_datasize / (512)) + 1;
    dim3 blockSize(BlockthreadNum);
    dim3 gridSize(grid);
    updata_weight_kernel<<<gridSize, blockSize>>>(w, grad, _datasize, lr);
    cudaDeviceSynchronize();
}






// 数组每一个元素加一个数
__global__ void batch_ouput_add_b_kernel(double* batch_output, double* _b, const size_t batch_size, const size_t output_num)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < output_num)
    {
        for (int b = 0; b < batch_size; ++b)
        {
            batch_output[b*output_num + index] += _b[index];
        }
    }
}
void FullconnecttionKernelFunc::batch_ouput_add_b(double* batch_output, double* _b, const size_t batch_size, const size_t output_num)
{
    const int BlockthreadNum = 512;
    const int grid = (output_num / (512)) + 1;
    dim3 blockSize(BlockthreadNum);
    dim3 gridSize(grid);
    batch_ouput_add_b_kernel<<<gridSize, blockSize>>>(batch_output, _b, batch_size, output_num);
    cudaDeviceSynchronize();
}
