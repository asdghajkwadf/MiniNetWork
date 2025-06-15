#include "Conv.h"
#include "Onion.h"
#include "layer.h"
#include "Pool.h"
#include "Start.h"
#include <iostream>

ConvLayer::ConvLayer(int step, int in_channel, int out_channel, int kernel_r, int kernel_c)
{
    Layer::layerType = LayerType::ConvolutionLayer;    
    this->step = step;
    this->in_channel = in_channel;
    this->out_channel = out_channel;

    this->kernel_num = out_channel / in_channel;
}

ConvLayer::~ConvLayer()
{   
    if (_w != nullptr)
    {
        delete _w;
    }

    if (_b != nullptr)
    {
        delete _b;
    }
}

int ConvLayer::getoutRows() const
{
    return out_rows;
}

int ConvLayer::getoutCols() const
{
    return out_cols;
}

void ConvLayer::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::StartingLayer)
    {
        StartLayer* sl = static_cast<StartLayer*>(lastLayer);
        this->in_rows = sl->rows;
        this->in_cols = sl->cols;
        this->in_channel = sl->channel;
    }
    else
    {
        if (lastLayer->layerType == LayerType::PoolingLayer)
        {
            PoolLayer* p = static_cast<PoolLayer*>(lastLayer);
            this->in_rows = p->out_rows;
            this->in_cols = p->out_cols;
            this->in_channel = p->channel;
        }
        else if (lastLayer->layerType == LayerType::ConvolutionLayer)
        {
            throw "fuck you";
        }
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
        vector<int> batchoutputShape = {Layer::batch_size, out_channel, out_rows, out_cols};
        vector<int> lossShape = {Layer::batch_size, in_channel, in_rows, in_cols};
        vector<int> batchinputShape = {Layer::batch_size, in_channel, in_rows, in_cols};

        Layer::batch_input = new Onion(batchinputShape, Layer::datawhere);
        Layer::batch_output = new Onion(batchoutputShape, Layer::datawhere);
        Layer::_loss = new Onion(lossShape, Layer::datawhere);

        initWeight();
        initGradient();
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        vector<int> inputShape = {in_channel, in_rows, in_cols};
        vector<int> outputShape = {out_channel, out_rows, out_cols};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }
}

void* ConvLayer::getWeight()
{
    return nullptr;
}

void ConvLayer::setKernelSize(int r, int c)
{
    this->kernel_r = r;
    this->kernel_c = c;
}

void ConvLayer::_forword(Onion* input)
{
    double* _wPtr = _w->getdataPtr();
    double* _bPtr = _b->getdataPtr();

    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();
    for (int in_c = 0; in_c < this->in_channel; ++in_c)
    {
        for (int kernel_i = 0; kernel_i < this->kernel_num; ++kernel_i)
        {
            for (int r = 0; r < this->_r_times; ++r)
            {
                for (int c = 0; c < this->_c_times; ++c)
                {
                    double sum = 0;
                    for (int k_r = 0; k_r < this->kernel_r; ++k_r)
                    {
                        for (int k_c = 0; k_c < this->kernel_c; ++k_c)
                        {
                            int inindex = in_c*in_rows*in_cols + (r*step + k_r)*in_cols + (c*step + k_c);
                            // std::cout << inindex << std::endl;
                            sum += inputPtr[inindex] * _wPtr[kernel_i*kernel_r*kernel_c + k_r*kernel_c + k_c];
                            // if (inindex >= 900)
                            // {
                            //     std::cout << inindex << std::endl;
                            // }
                        }
                    }
                    int outindex = in_c*kernel_num*out_rows*out_cols + kernel_i*out_rows*out_cols + r*out_cols + c;

                    outputPtr[outindex] = sum + _bPtr[kernel_i];
                }
            }  
        }
    }
}

void ConvLayer::trainForword(Onion* batch_input)
{
    Layer::batch_input->CopyData(batch_input);
    if (datawhere == dataWhere::CPU)
    {
        _CPUforword(batch_input);
    }
    else if (datawhere == dataWhere::GPU)
    {
        _GPUforword(batch_input);
    }
}

void ConvLayer::trainBackword(Onion* loss)
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
    Layer::_loss->setAllData(0);
    _w_grad->setAllData(0);
    _b_grad->setAllData(0);
}

void ConvLayer::_CPUclac_gradient(Onion* loss)
{
    double* batchinputPtr = Layer::batch_input->getdataPtr();
    double* lossPtr = loss->getdataPtr();
    double* layerLossPtr = Layer::_loss->getdataPtr();
    double* bGradPtr = _b_grad->getdataPtr();
    double* wGradPtr = _w_grad->getdataPtr();

    double* wPtr = _w->getdataPtr();

    for (int b = 0; b < Layer::batch_size; ++b)
    {
        for (int in_c = 0; in_c < this->in_channel; ++in_c)
        {
            for (int kernel_i = 0; kernel_i < this->kernel_num; ++kernel_i)
            {
                for (int r = 0; r < this->_r_times; ++r)
                {
                    for (int c = 0; c < this->_c_times; ++c)
                    {
                        int lossindex = b*out_channel*out_rows*out_cols + in_c*kernel_num*out_rows*out_cols + kernel_i*out_rows*out_cols + r*out_cols + c;
                        for (int k_r = 0; k_r < this->kernel_r; ++k_r)
                        {
                            for (int k_c = 0; k_c < this->kernel_c; ++k_c)
                            {
                                int inindex = b*in_channel*in_rows*in_cols + in_c*in_rows*in_cols + (r*step + k_r)*in_cols + (c*step + k_c);
                                wGradPtr[k_r*kernel_c + k_c] += batchinputPtr[inindex] * lossPtr[lossindex] / Layer::batch_size;
                                layerLossPtr[inindex] = lossPtr[lossindex] * wPtr[kernel_i*kernel_r*kernel_c + k_r*kernel_c + k_c];
                            }
                        }
                        // std::cout << lossindex << std::endl;
                        bGradPtr[kernel_i] += lossPtr[lossindex] / Layer::batch_size;
                    }
                }  
            }
        }
    }

    // for (int k = 0; k < kernel_num; ++k)
    // {
    //     bGradPtr[k] /= Layer::batch_size;
    //     for (int k_r = 0; k_r < kernel_r; ++k_r)
    //     {
    //         for (int k_c = 0; k_c < kernel_c; ++k_c)
    //         {
    //             wGradPtr[k*kernel_r*kernel_c + k_r*kernel_c + k_c] /= Layer::batch_size;
    //         }
    //     }
    // } 
}

void ConvLayer::_CPUupdate()
{
    double* bGradPtr = _b_grad->getdataPtr();
    double* wGradPtr = _w_grad->getdataPtr();

    double* bPtr = _b->getdataPtr();
    double* wPtr = _w->getdataPtr();

    for (int k = 0; k < kernel_num; ++k)
    {
        bPtr[k] -= bGradPtr[k] * Layer::lr;
        for (int k_r = 0; k_r < kernel_r; ++k_r)
        {
            for (int k_c = 0; k_c < kernel_c; ++k_c)
            {
                wPtr[k*kernel_r*kernel_c + k_r*kernel_c + k_c] -= wGradPtr[k*kernel_r*kernel_c + k_r*kernel_c + k_c] * Layer::lr;
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

void ConvLayer::_GPUclac_gradient(Onion* loss)
{

}

void ConvLayer::_CPUforword(Onion* batch_input)
{
    double* _wPtr = _w->getdataPtr();
    double* _bPtr = _b->getdataPtr();

    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();

    for (int b = 0; b < Layer::batch_size; ++b)
    {
        for (int in_c = 0; in_c < this->in_channel; ++in_c)
        {
            for (int kernel_i = 0; kernel_i < this->kernel_num; ++kernel_i)
            {
                for (int r = 0; r < this->_r_times; ++r)
                {
                    for (int c = 0; c < this->_c_times; ++c)
                    {
                        double sum = 0;
                        for (int k_r = 0; k_r < this->kernel_r; ++k_r)
                        {
                            for (int k_c = 0; k_c < this->kernel_c; ++k_c)
                            {
                                int inindex = b*in_channel*in_rows*in_cols + in_c*in_rows*in_cols + (r*step + k_r)*in_cols + (c*step + k_c);
                                // std::cout << inindex << std::endl;
                                sum += batchinputPtr[inindex] * _wPtr[kernel_i*kernel_r*kernel_c + k_r*kernel_c + k_c];
                            }
                        }
                        int outindex = b*in_channel*kernel_num*out_rows*out_cols + in_c*kernel_num*out_rows*out_cols + kernel_i*out_rows*out_cols + r*out_cols + c;
                        batchoutputPtr[outindex] = sum + _bPtr[kernel_i];
                    }
                }  
            }
        }
    }
}

void ConvLayer::_GPUforword(Onion* batch_input)
{

}

void ConvLayer::initWeight()
{   
    vector<int> wShape = {kernel_num, kernel_r, kernel_c};
    _w = new Onion(wShape, Layer::datawhere);
    _w->initdata(-0.1, 0.1);

    vector<int> bShape = {kernel_num};
    _b = new Onion(bShape, Layer::datawhere);
    _b->setAllData(1);
}

void ConvLayer::initGradient()
{
    vector<int> wGradShape = {kernel_num, kernel_r, kernel_c};
    vector<int> bGradShape = {kernel_num};

    _w_grad = new Onion(wGradShape, Layer::datawhere);
    _b_grad = new Onion(bGradShape, Layer::datawhere);

    _w_grad->setAllData(0);
    _b_grad->setAllData(0);
}