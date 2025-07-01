#ifndef _CUDAFUNC_CUH_
#define _CUDAFUNC_CUH_


// __global__ void divideArraysSelfKernel(double* src, const double n, size_t _datasize);

namespace Common
{
    void Arrays_divide_arrays(double* arr, const double n, const size_t _datasize);
    void update_weight(double* w, double* grad, const size_t _datasize, const double lr);
};

namespace FullconnecttionKernelFunc
{
    // 神经元的误差传到上一层
    void AverageNextloss(double* loss, double* dst, const size_t batch_size, const size_t output_num);

    // 前向传播计算完毕时把矩阵相乘的结果每行加上bias
    void batch_ouput_add_b(double* batch_output, double* _b, const size_t batch_size, const size_t output_num);
};
#endif