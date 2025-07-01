#include "FullconnectionLayer.h"
#include "Onion.h"
#include "Layer.h"
#include "ViewLayer.h"
#include "ReluLayer.h"
#include "SoftMaxLayer.h"
#include <iostream>
#include "Timer/Timer.h"
#include "CudaKernel/cudaFunc.cuh"

FullconnectionLayer::FullconnectionLayer(size_t output_num)
{
    Layer::layerType = FullConnectionLayerENUM;
    this->output_num = output_num;

    CHECK_CUBLAS_ERROR(cublasCreate(&cuBLAShandle));
}

FullconnectionLayer::~FullconnectionLayer()
{
    // if (_w != nullptr)
    // {
    //     delete _w;
    // }

    // if (_b != nullptr)
    // {
    //     delete _b;
    // }

    // if (_w_grad != nullptr)
    // {
    //     delete _w_grad;
    // }

    // if (_b_grad != nullptr)
    // {
    //     delete _b_grad;
    // }
}

void FullconnectionLayer::_forword(Onion& input)
{
    for (size_t out = 0; out < output_num; out++)
    {
        double _sum = 0;
        for (size_t in = 0; in < input_num; in++)
        {
            _sum += _w[out*input_num + in] * input[in];
        }
        Layer::output[out] = _sum + _b[out];
    }
}

void FullconnectionLayer::trainForword(Onion& batch_input)
{
    Layer::batch_input.CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUforword();
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUforword();
    }
}


void FullconnectionLayer::trainBackword(Onion& loss)
{
    if (Layer::datawhere == dataWhere::CPU)
    {
        Timer t(this);
        _CPUZeroGrad();
        _CPUclac_gradient(loss);
        _CPUupdate();
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _w_grad.toGPU();
        _b_grad.toGPU();
        _w.toGPU();
        _b.toGPU();
        Layer::_loss.toGPU();
        Layer::batch_input.toGPU();
        loss.toGPU();


        _GPUZeroGrad();
        _GPUclac_gradient(loss);
        _GPUupdate();



        _w_grad.toCPU();
        _b_grad.toCPU();
        _w.toCPU();
        _b.toCPU();
        Layer::_loss.toCPU();
        Layer::batch_input.toCPU();
        loss.toCPU();
    }
}

void FullconnectionLayer::_GPUZeroGrad()
{
    Layer::_loss.setAllData(0);
    _w_grad.setAllData(0);
    _b_grad.setAllData(0);
}

void FullconnectionLayer::_GPUclac_gradient(Onion& nextLayerBatchLoss)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    // 计算w的梯度
    cublasStatus_t status = cublasDgemm(
        cuBLAShandle,
        CUBLAS_OP_N,      
        CUBLAS_OP_T,   
        output_num,     
        input_num,          
        Layer::batch_size,     
        &alpha,
        nextLayerBatchLoss.getdataPtr(),       
        output_num,     
        Layer::batch_input.getdataPtr(),            
        input_num,   
        &beta,
        _w_grad.getdataPtr(),       
        output_num     
    );

    status = cublasDgemm(
        cuBLAShandle,
        CUBLAS_OP_T,   
        CUBLAS_OP_N,   
        batch_size,   
        input_num,    
        output_num,    
        &alpha,
        _w.getdataPtr(),         
        output_num,   
        nextLayerBatchLoss.getdataPtr(),        
        output_num,    
        &beta,
        Layer::_loss.getdataPtr(),         
        input_num    
    );


    // 计算b的梯度
    FullconnecttionKernelFunc::AverageNextloss(nextLayerBatchLoss.getdataPtr(), _b_grad.getdataPtr(), Layer::batch_size, output_num);

    // 由于上面的矩阵相乘是把所有梯度都加上去了，要除平均梯度
    _w_grad.__divide__(Layer::batch_size);
    _b_grad.__divide__(Layer::batch_size);

}

void FullconnectionLayer::_GPUupdate()
{
    Common::update_weight(_w.getdataPtr(), _w_grad.getdataPtr(), _w.Size(), Layer::lr);
    Common::update_weight(_b.getdataPtr(), _b_grad.getdataPtr(), _b.Size(), Layer::lr);
}

void FullconnectionLayer::_GPUforword()
{
    _w.toGPU();
    _b.toGPU();
    Layer::batch_output.toGPU();
    Layer::batch_input.toGPU();

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t status = cublasDgemm(
        cuBLAShandle,
        CUBLAS_OP_N,        // Transpose weight matrix (for row-major data)
        CUBLAS_OP_N,        // Transpose input matrix (for row-major data)
        output_num,     // Rows of resulting matrix (output^T)
        Layer::batch_size,          // Columns of resulting matrix (output^T)
        input_num,      // Common dimension
        &alpha,
        _w.getdataPtr(),          // Weight matrix (device pointer)
        output_num,     // Leading dimension of weight matrix (original rows)
        Layer::batch_input.getdataPtr(),            // Input matrix (device pointer)
        input_num,     // Leading dimension of input matrix (original rows)
        &beta,
        Layer::batch_output.getdataPtr(),           // Output matrix (device pointer)
        output_num     // Leading dimension of output matrix
    );

    FullconnecttionKernelFunc::batch_ouput_add_b(Layer::batch_output.getdataPtr(), 
                                                _b.getdataPtr(),
                                                Layer::batch_size,
                                                output_num
                                            );

    _w.toCPU();
    _b.toCPU();
    Layer::batch_output.toCPU();
    Layer::batch_input.toCPU();
}














void FullconnectionLayer::_CPUZeroGrad()
{
    _w_grad.setAllData(0);
    _b_grad.setAllData(0);
    Layer::_loss.setAllData(0);
}

void FullconnectionLayer::_CPUclac_gradient(Onion& nextLayerBatchLoss)
{
    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t out = 0; out < output_num; ++out)
        {
            for (size_t in = 0; in < input_num; ++in)  
            {
                Layer::_loss[b*input_num + in] += nextLayerBatchLoss[b*output_num + out] * _w[in*output_num + out];    
                _w_grad[in*output_num + out] += nextLayerBatchLoss[b*output_num + out] * Layer::batch_input[b*input_num + in] / Layer::batch_size;
            }
            _b_grad[out] += nextLayerBatchLoss[b*output_num + out] / Layer::batch_size;
        }
    }


    // for (size_t b = 0; b < Layer::batch_size; ++b)
    // {
    //     for (size_t r = 0; r < output_num; ++r)
    //     {
    //         bGradPtr[r] += nextLayerBatchLossPtr[b*output_num + r] / Layer::batch_size;
    //     } 
    // }



    // for (size_t b = 0; b < Layer::batch_size; ++b)
    // {
    //     for (size_t r = 0; r < output_num; ++r)
    //     {
    //         for (size_t c = 0; c < input_num; ++c)
    //         {
    //             wGradPtr[r*input_num + c] += nextLayerBatchLossPtr[b*output_num + r] * batchinputPtr[b*input_num + c] / Layer::batch_size;
    //         }
    //     }
    // }

}

void FullconnectionLayer::_CPUupdate()
{
    for (size_t b = 0; b < output_num; ++b)
    {
        _b[b] -= _b_grad[b] * Layer::lr;
    }

    for (size_t r = 0; r < output_num; ++r)
    {
        for (size_t c = 0; c < input_num; ++c)
        {
            _w[r*input_num + c] -= _w_grad[r*input_num + c] * Layer::lr;
        }
    }
}

void FullconnectionLayer::_CPUforword()
{   
    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t out = 0; out < output_num; out++)
        {
            double _sum = 0;
            for (size_t in = 0; in < input_num; in++)
            {
                _sum += Layer::batch_input[b*input_num + in] * _w[in*output_num + out];
            }
            Layer::batch_output[b*output_num + out] = _sum + _b[out];
        }
    }
}




void FullconnectionLayer::initGradient()
{
    OnionShape _w_grad_Shape = {output_num, input_num};
    OnionShape _b_grad_Shape = {output_num};
    _w_grad.initOnion(_w_grad_Shape, Layer::datawhere);
    _b_grad.initOnion(_b_grad_Shape, Layer::datawhere);
}

void FullconnectionLayer::initWeight()
{
    OnionShape w_Shape = {output_num, input_num};
    OnionShape b_Shape = {output_num};
    _w.initOnion(w_Shape, Layer::datawhere);
    _b.initOnion(b_Shape, Layer::datawhere);
    _w.initdata(-0.5, 0.5);
    _b.setAllData(0);
}

void FullconnectionLayer::initMatrix(Layer* lastLayer, dataWhere where)
{
    Layer::batch_size = lastLayer->batch_size;
    Layer::datawhere = where;
    if (lastLayer->layerType == LayerType::ViewLayerENUM)
    {
        ViewLayer* v = static_cast<ViewLayer*>(lastLayer);
        this->input_num = v->output_num;
    }
    else if (lastLayer->layerType == LayerType::FullConnectionLayerENUM)
    {
        FullconnectionLayer* fc = static_cast<FullconnectionLayer*>(lastLayer);
        this->input_num = fc->output_num;
    }
    else if (lastLayer->layerType == LayerType::ReluLayerENUM)
    {
        ReluLayer* r = static_cast<ReluLayer*>(lastLayer);
        this->input_num = r->input_num;
    }
    else
    {
        throw "ouccr some error, fuck ! ";
    }




    if (Layer::modelType == ModelType::Train)
    {
        OnionShape batchoutputShape = {Layer::batch_size, output_num};
        OnionShape batchinputShape = {Layer::batch_size, input_num};

        OnionShape lossShape = {Layer::batch_size, input_num};

        Layer::batch_input.initOnion(batchinputShape, Layer::datawhere);
        Layer::batch_output.initOnion(batchoutputShape, Layer::datawhere);
        Layer::_loss.initOnion(lossShape, Layer::datawhere);

        initWeight();
        initGradient();
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {input_num};
        OnionShape outputShape = {output_num};

        Layer::input.initOnion(inputShape, Layer::datawhere);
        Layer::output.initOnion(outputShape, Layer::datawhere);
    }
}

void* FullconnectionLayer::getWeight()
{
    return nullptr;
}
