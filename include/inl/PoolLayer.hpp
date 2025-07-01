#include "Layer.h"
#include "PoolLayer.h"
#include "Onion.h"
#include "ConvLayer.h"
#include "Model.h"
#include <iostream>
#include "Timer/Timer.h"


MaxPoolLayer::MaxPoolLayer(size_t pooling_rows, size_t pooling_cols)
{
    this->pooling_rows = pooling_rows;
    this->pooling_cols = pooling_cols;

    Layer::layerType = LayerType::MaxPoolingLayerENUM;
}

MaxPoolLayer::~MaxPoolLayer()
{
    // if (max_index != nullptr)
    // {
    //     delete max_index;
    // }
}

void* MaxPoolLayer::getWeight()
{
    throw "Pooling Layer no Weight , SB";
}

void MaxPoolLayer::initMatrix(Layer* lastLayer, dataWhere where)
{
    Layer::batch_size = lastLayer->batch_size;
    Layer::datawhere = where;
    if (lastLayer->layerType == LayerType::MaxPoolingLayerENUM)
    {
        throw "fuck you !!!";
    }
    else if (lastLayer->layerType == LayerType::ConvolutionLayerENUM)
    {
        ConvLayer* c = static_cast<ConvLayer*>(lastLayer);
        channel = c->out_channel;
        in_rows = c->out_rows;
        in_cols = c->out_cols;
    }

    _r_times = in_rows / 2;
    this->_c_times = in_cols / 2;

    out_rows = _r_times;
    out_cols = _c_times;




    if (Layer::modelType == ModelType::Train)
    {
        OnionShape lossShape = {Layer::batch_size, channel, in_rows, in_cols};
        OnionShape batchoutputShape = {Layer::batch_size, channel, out_rows, out_cols};
        OnionShape batchinputShape = {Layer::batch_size, channel, in_rows, in_cols};

        OnionShape maxindexShape = {Layer::batch_size, channel, out_rows, out_cols};

        Layer::_loss.initOnion(lossShape, Layer::datawhere);
        Layer::batch_output.initOnion(batchoutputShape, Layer::datawhere);
        Layer::batch_input.initOnion(batchinputShape, Layer::datawhere);
        max_index.initOnion(maxindexShape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {channel, in_rows, in_cols};
        OnionShape outputShape = {channel, out_rows, out_cols};

        Layer::input.initOnion(inputShape, Layer::datawhere);
        Layer::output.initOnion(outputShape, Layer::datawhere);
    }

}

void MaxPoolLayer::trainForword(Onion& batch_input)
{
    Layer::batch_input.CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUpooling();
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUpooling();
    }
}

void MaxPoolLayer::_forword(Onion& input)
{
    for (size_t cha = 0; cha < channel; ++cha)
    {
        for (size_t r = 0; r < this->_r_times; ++r)
        {
            for (size_t c = 0; c < this->_c_times; ++c)
            {
                size_t _bigindex = cha*in_rows*in_cols + (r*pooling_rows) * in_cols + (c*pooling_cols);
                double max = input[_bigindex];
                for (size_t p_r = 0; p_r < pooling_rows; ++p_r)
                {
                    for (size_t p_c = 0; p_c < pooling_cols; ++p_c)
                    {
                        size_t inindex = cha*in_rows*in_cols + (r*pooling_rows + p_r) * in_cols + (c*pooling_cols + p_c);
                        if (input[inindex] > max)
                        {
                            max = input[inindex];
                        }
                    }
                }
                Layer::output[cha*out_rows*out_cols + r*out_cols + c] = max;
            }
        }
    }
}

void MaxPoolLayer::trainBackword(Onion& loss)
{
    
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUZeroGrad();
        _CPUclac_gradient(loss);
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUZeroGrad();
        _GPUclac_gradient(loss);
    }
}

void MaxPoolLayer::_CPUZeroGrad()
{
    Layer::_loss.setAllData(0);
}
void MaxPoolLayer::clac_loss(Onion& batch_output)
{

}
void MaxPoolLayer::_CPUclac_gradient(Onion& nextLayerBatchLoss)
{
    for (size_t i = 0; i < max_index.Size(); ++i)
    {
        size_t lossindex = max_index[i];
        Layer::_loss[lossindex] = nextLayerBatchLoss[i];
    }
}



void MaxPoolLayer::_GPUZeroGrad()
{

}
void MaxPoolLayer::_GPUclac_gradient(Onion& nextLayerBatchLoss)
{

}
void MaxPoolLayer::_GPUupdate()
{

}

void MaxPoolLayer::_CPUpooling()
{
    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t cha = 0; cha < channel; ++cha)
        {
            for (size_t r = 0; r < this->_r_times; ++r)
            {
                for (size_t c = 0; c < this->_c_times; ++c)
                {
                    size_t _bigindex = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows) * in_cols + (c*pooling_cols);
                    double max = Layer::batch_input[_bigindex];

                    size_t maxIndex = b*channel*_r_times*_c_times + cha*_r_times*_c_times + r*_c_times + c;
                    max_index[maxIndex] = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows)*in_cols + c*pooling_cols;
                    for (size_t p_r = 0; p_r < pooling_rows; ++p_r)
                    {
                        for (size_t p_c = 0; p_c < pooling_cols; ++p_c)
                        {
                            size_t inindex = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows + p_r) * in_cols + (c*pooling_cols + p_c);
                            if (Layer::batch_input[inindex] > max)
                            {
                                max = Layer::batch_input[inindex];
                                max_index[maxIndex] = (double)(inindex);
                            }
                        }
                    }
                    Layer::batch_output[b*channel*out_rows*out_cols + cha*out_rows*out_cols + r*out_cols + c] = max;
                }
            }
        }
    }    
}

void MaxPoolLayer::_GPUpooling()
{

}

void MaxPoolLayer::initWeight()
{   
    throw "Pooling Layer no Weight , SB";
}


