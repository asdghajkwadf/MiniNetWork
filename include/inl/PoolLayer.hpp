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

    Layer::layerType = LayerType::PoolingLayerENUM;
}

MaxPoolLayer::~MaxPoolLayer()
{
    if (max_index != nullptr)
    {
        delete max_index;
    }
}

void* MaxPoolLayer::getWeight()
{
    throw "Pooling Layer no Weight , SB";
}

void MaxPoolLayer::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::PoolingLayerENUM)
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

        std::vector<size_t> maxindexShape = {Layer::batch_size, channel, out_rows, out_cols};

        Layer::_loss = new Onion(lossShape, Layer::datawhere);
        Layer::batch_output = new Onion(batchoutputShape, Layer::datawhere);
        Layer::batch_input = new Onion(batchinputShape, Layer::datawhere);

        max_index = new Onion(maxindexShape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {channel, in_rows, in_cols};
        OnionShape outputShape = {channel, out_rows, out_cols};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }

}

void MaxPoolLayer::trainForword(Onion* batch_input)
{
    Layer::batch_input->CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUpooling(batch_input);
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUpooling(batch_input);
    }
}

void MaxPoolLayer::_forword(Onion* input)
{
    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();

    for (size_t cha = 0; cha < channel; ++cha)
    {
        for (size_t r = 0; r < this->_r_times; ++r)
        {
            for (size_t c = 0; c < this->_c_times; ++c)
            {
                size_t _bigindex = cha*in_rows*in_cols + (r*pooling_rows) * in_cols + (c*pooling_cols);
                double max = inputPtr[_bigindex];

                size_t maxIndex = cha*_r_times*_c_times + r*_c_times + c;
                for (size_t p_r = 0; p_r < pooling_rows; ++p_r)
                {
                    for (size_t p_c = 0; p_c < pooling_cols; ++p_c)
                    {
                        size_t inindex = cha*in_rows*in_cols + (r*pooling_rows + p_r) * in_cols + (c*pooling_cols + p_c);
                        if (inputPtr[inindex] > max)
                        {
                            max = inputPtr[inindex];
                        }
                    }
                }
                outputPtr[cha*out_rows*out_cols + r*out_cols + c] = max;
            }
        }
    }
    
}

void MaxPoolLayer::trainBackword(Onion* loss)
{
    Timer t(this);
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
    Layer::_loss->setAllData(0);
}
void MaxPoolLayer::clac_loss(Onion* batch_output)
{

}
void MaxPoolLayer::_CPUclac_gradient(Onion* nextLayerBatchLoss)
{
    double* lossPtr = Layer::_loss->getdataPtr();
    double* nextLossPtr = nextLayerBatchLoss->getdataPtr();
    double* maxIndexPtr = this->max_index->getdataPtr();

    for (size_t i = 0; i < max_index->Size(); ++i)
    {
        size_t lossindex = (size_t)maxIndexPtr[i];
        lossPtr[lossindex] = nextLossPtr[i];
    }
}



void MaxPoolLayer::_GPUZeroGrad()
{

}
void MaxPoolLayer::_GPUclac_gradient(Onion* nextLayerBatchLoss)
{

}
void MaxPoolLayer::_GPUupdate()
{

}

void MaxPoolLayer::_CPUpooling(Onion* batch_input)
{
    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();
    double* maxIndexPtr = this->max_index->getdataPtr();

    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t cha = 0; cha < channel; ++cha)
        {
            for (size_t r = 0; r < this->_r_times; ++r)
            {
                for (size_t c = 0; c < this->_c_times; ++c)
                {
                    size_t _bigindex = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows) * in_cols + (c*pooling_cols);
                    double max = batchinputPtr[_bigindex];

                    size_t maxIndex = b*channel*_r_times*_c_times + cha*_r_times*_c_times + r*_c_times + c;
                    maxIndexPtr[maxIndex] = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows)*in_cols + c*pooling_cols;
                    for (size_t p_r = 0; p_r < pooling_rows; ++p_r)
                    {
                        for (size_t p_c = 0; p_c < pooling_cols; ++p_c)
                        {
                            size_t inindex = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows + p_r) * in_cols + (c*pooling_cols + p_c);
                            if (batchinputPtr[inindex] > max)
                            {
                                max = batchinputPtr[inindex];
                                maxIndexPtr[maxIndex] = (double)(inindex);
                            }
                        }
                    }
                    batchoutputPtr[b*channel*out_rows*out_cols + cha*out_rows*out_cols + r*out_cols + c] = max;
                }
            }
        }
    }    
}

void MaxPoolLayer::_GPUpooling(Onion* batch_input)
{

}

void MaxPoolLayer::initWeight()
{   
    throw "Pooling Layer no Weight , SB";
}


