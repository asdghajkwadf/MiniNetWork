#ifndef _FULLCONNECTIONLAYER_H_
#define _FULLCONNECTIONLAYER_H_


#include "Layer.h"
#include "Onion.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

// 检查 CUDA 错误的辅助宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 检查 cuBLAS 错误的辅助宏
#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d code=%d \"%s\"\n", \
                    __FILE__, __LINE__, status, #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


class FullconnectionLayer : public Layer
{
public:
    
    FullconnectionLayer(size_t output_num);
    ~FullconnectionLayer();
    
    
    void trainForword(Onion& batch_input) override;
    void trainBackword(Onion& loss) override;
    void _forword(Onion& input) override;
    void initMatrix(Layer* lastLayer, dataWhere where) override;


    void* getWeight();

    size_t input_num = 0;
    size_t output_num = 0;

private:

    cublasHandle_t cuBLAShandle = nullptr;

    size_t in_rows = 0;
    size_t in_cols = 0;

    // CPU
    void _CPUZeroGrad();
    void _CPUupdate();
    void _CPUforword();
    void clac_loss(Onion& batch_output);
    void _CPUclac_gradient(Onion& nextLayerBatchLoss);


    // GPU
    void _GPUforword();
    void _GPUZeroGrad();
    void _GPUclac_gradient(Onion& nextLayerBatchLoss);
    void _GPUupdate();



    void initGradient();
    void initWeight();


    Onion _w;
    Onion _b;

    Onion _w_grad;
    Onion _b_grad;
};

#include "inl/FullconnectionLayer.hpp"

#endif