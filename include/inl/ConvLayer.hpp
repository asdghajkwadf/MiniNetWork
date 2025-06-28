#include "ConvLayer.h"
#include "Onion.h"


#include "Layer.h"
#include "PoolLayer.h"
#include "StartLayer.h"

#include <iostream>
#include "Timer/Timer.h"

ConvLayer::ConvLayer(size_t step, size_t in_channel, size_t out_channel, size_t kernel_r, size_t kernel_c)
{
    Layer::layerType = LayerType::ConvolutionLayerENUM;    
    this->step = step;
    this->in_channel = in_channel;
    this->out_channel = out_channel;

    this->kernel_num = out_channel / in_channel;
}

ConvLayer::~ConvLayer()
{   
    // if (_w != nullptr)
    // {
    //     delete _w;
    // }

    // if (_b != nullptr)
    // {
    //     delete _b;
    // }
}

size_t ConvLayer::getoutRows() const
{
    return out_rows;
}

size_t ConvLayer::getoutCols() const
{
    return out_cols;
}

void ConvLayer::initMatrix(Layer* lastLayer)
{   
    Layer::batch_size = lastLayer->batch_size;
    if (lastLayer->layerType == LayerType::StartLayerENUM)
    {
        StartLayer* al = static_cast<StartLayer*>(lastLayer);
        in_rows = al->out_rows;
        in_cols = al->out_cols;
    }

    else if (lastLayer->layerType == LayerType::MaxPoolingLayerENUM)
    {
        MaxPoolLayer* p = static_cast<MaxPoolLayer*>(lastLayer);
        this->in_rows = p->out_rows;
        this->in_cols = p->out_cols;
        this->in_channel = p->channel;
        
        Layer::batch_size = lastLayer->batch_size;

    }
    else if (lastLayer->layerType == LayerType::ConvolutionLayerENUM)
    {
        throw "fuck you";
    }



    this->_r_times = in_rows - kernel_r + 1;
    this->_c_times = in_cols - kernel_c + 1;

    if (this->_r_times % step != 0) 
    {
        throw "step不匹配!";
    }
    if (this->_c_times % step != 0) 
    {
        throw "step不匹配!";
    }

    this->out_rows = this->_r_times;
    this->out_cols = this->_c_times;

    if (Layer::modelType == ModelType::Train)
    {
        OnionShape batchoutputShape = {Layer::batch_size, out_channel, out_rows, out_cols};
        OnionShape lossShape = {Layer::batch_size, in_channel, in_rows, in_cols};
        OnionShape batchinputShape = {Layer::batch_size, in_channel, in_rows, in_cols};

        Layer::batch_input.initOnion(batchinputShape, Layer::datawhere);
        Layer::batch_output.initOnion(batchoutputShape, Layer::datawhere);
        Layer::_loss.initOnion(lossShape, Layer::datawhere);

        initWeight();
        initGradient();
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {in_channel, in_rows, in_cols};
        OnionShape outputShape = {out_channel, out_rows, out_cols};

        Layer::input.initOnion(inputShape, Layer::datawhere);
        Layer::output.initOnion(outputShape, Layer::datawhere);
    }
}

void* ConvLayer::getWeight()
{
    return nullptr;
}

void ConvLayer::setKernelSize(size_t r, size_t c)
{
    this->kernel_r = r;
    this->kernel_c = c;
}

void ConvLayer::_forword(Onion& input)
{
    for (size_t in_c = 0; in_c < this->in_channel; ++in_c)
    {
        for (size_t kernel_i = 0; kernel_i < this->kernel_num; ++kernel_i)
        {
            for (size_t r = 0; r < this->_r_times; ++r)
            {
                for (size_t c = 0; c < this->_c_times; ++c)
                {
                    double sum = 0;
                    for (size_t k_r = 0; k_r < this->kernel_r; ++k_r)
                    {
                        for (size_t k_c = 0; k_c < this->kernel_c; ++k_c)
                        {
                            size_t inindex = in_c*in_rows*in_cols + (r*step + k_r)*in_cols + (c*step + k_c);
                            sum += input[inindex] * _w[kernel_i*kernel_r*kernel_c + k_r*kernel_c + k_c];
                        }
                    }
                    size_t outindex = in_c*kernel_num*out_rows*out_cols + kernel_i*out_rows*out_cols + r*out_cols + c;

                    Layer::output[outindex] = sum + _b[kernel_i];
                }
            }  
        }
    }
}

void ConvLayer::trainForword(Onion& batch_input)
{
    Layer::batch_input.CopyData(batch_input);
    if (datawhere == dataWhere::CPU)
    {
        _CPUforword();
    }
    else if (datawhere == dataWhere::GPU)
    {
        _GPUforword();
    }
}

void ConvLayer::trainBackword(Onion& loss)
{   
    Timer t(this);
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUZeroGrad();
        _CPUclac_gradient(loss);
        _CPUupdate();
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUZeroGrad();
        _GPUclac_gradient(loss);
    }
}

void ConvLayer::_CPUZeroGrad()
{
    Layer::_loss.setAllData(0);
    _w_grad.setAllData(0);
    _b_grad.setAllData(0);
}

void ConvLayer::_CPUclac_gradient(Onion& loss)
{
    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t in_c = 0; in_c < this->in_channel; ++in_c)
        {
            for (size_t kernel_i = 0; kernel_i < this->kernel_num; ++kernel_i)
            {
                for (size_t r = 0; r < this->_r_times; ++r)
                {
                    for (size_t c = 0; c < this->_c_times; ++c)
                    {
                        size_t lossindex = b*out_channel*out_rows*out_cols + in_c*kernel_num*out_rows*out_cols + kernel_i*out_rows*out_cols + r*out_cols + c;
                        for (size_t k_r = 0; k_r < this->kernel_r; ++k_r)
                        {
                            for (size_t k_c = 0; k_c < this->kernel_c; ++k_c)
                            {
                                size_t inindex = b*in_channel*in_rows*in_cols + in_c*in_rows*in_cols + (r*step + k_r)*in_cols + (c*step + k_c);
                                _w_grad[k_r*kernel_c + k_c] += Layer::batch_input[inindex] * loss[lossindex] / Layer::batch_size;
                                Layer::_loss[inindex] = loss[lossindex] * _w[kernel_i*kernel_r*kernel_c + k_r*kernel_c + k_c];
                            }
                        }
                        // std::std::cout << lossindex << std::endl;
                        _b_grad[kernel_i] += loss[lossindex] / Layer::batch_size;
                    }
                }  
            }
        }
    }

    // for (size_t k = 0; k < kernel_num; ++k)
    // {
    //     bGradPtr[k] /= Layer::batch_size;
    //     for (size_t k_r = 0; k_r < kernel_r; ++k_r)
    //     {
    //         for (size_t k_c = 0; k_c < kernel_c; ++k_c)
    //         {
    //             wGradPtr[k*kernel_r*kernel_c + k_r*kernel_c + k_c] /= Layer::batch_size;
    //         }
    //     }
    // } 
}

void ConvLayer::_CPUupdate()
{
    for (size_t k = 0; k < kernel_num; ++k)
    {
        _b[k] -= _b[k] * Layer::lr;
        for (size_t k_r = 0; k_r < kernel_r; ++k_r)
        {
            for (size_t k_c = 0; k_c < kernel_c; ++k_c)
            {
                _w[k*kernel_r*kernel_c + k_r*kernel_c + k_c] -= _w_grad[k*kernel_r*kernel_c + k_r*kernel_c + k_c] * Layer::lr;
            }
        }
    }
}

void ConvLayer::_GPUupdate()
{

}

void ConvLayer::_GPUZeroGrad()
{

}

void ConvLayer::_GPUclac_gradient(Onion& loss)
{

}

void ConvLayer::_CPUforword()
{
    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t in_c = 0; in_c < this->in_channel; ++in_c)
        {
            for (size_t kernel_i = 0; kernel_i < this->kernel_num; ++kernel_i)
            {
                for (size_t r = 0; r < this->_r_times; ++r)
                {
                    for (size_t c = 0; c < this->_c_times; ++c)
                    {
                        double sum = 0;
                        for (size_t k_r = 0; k_r < this->kernel_r; ++k_r)
                        {
                            for (size_t k_c = 0; k_c < this->kernel_c; ++k_c)
                            {
                                size_t inindex = b*in_channel*in_rows*in_cols + in_c*in_rows*in_cols + (r*step + k_r)*in_cols + (c*step + k_c);
                                // std::std::cout << inindex << std::endl;
                                sum += Layer::batch_input[inindex] * _w[kernel_i*kernel_r*kernel_c + k_r*kernel_c + k_c];
                            }
                        }
                        size_t outindex = b*in_channel*kernel_num*out_rows*out_cols + in_c*kernel_num*out_rows*out_cols + kernel_i*out_rows*out_cols + r*out_cols + c;
                        Layer::batch_output[outindex] = sum + _b[kernel_i];
                    }
                }  
            }
        }
    }
}

void ConvLayer::_GPUforword()
{

}

void ConvLayer::initWeight()
{   
    OnionShape wShape = {kernel_num, kernel_r, kernel_c};
    _w.initOnion(wShape, Layer::datawhere);
    _w.initdata(-0.5, 0.5);

    OnionShape bShape = {kernel_num};
    _b.initOnion(bShape, Layer::datawhere);
    _b.setAllData(0);
}

void ConvLayer::initGradient()
{
    OnionShape wGradShape = {kernel_num, kernel_r, kernel_c};
    OnionShape bGradShape = {kernel_num};

    _w_grad.initOnion(wGradShape, Layer::datawhere);
    _b_grad.initOnion(bGradShape, Layer::datawhere);

    _w_grad.setAllData(0);
    _b_grad.setAllData(0);
}