#include "Layer.h"
#include "Pool.h"
#include "Onion.h"
#include "Conv.h"

#include <iostream>



PoolLayer::PoolLayer(PoolType type, int pooling_rows, int pooling_cols)
{
    this->pooling_rows = pooling_rows;
    this->pooling_cols = pooling_cols;

    Layer::layerType = LayerType::PoolingLayer;
}

PoolLayer::~PoolLayer()
{
    if (max_index != nullptr)
    {
        delete max_index;
    }
}

void* PoolLayer::getWeight()
{
    throw "Pooling Layer no Weight , SB";
}

void PoolLayer::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::StartingLayer)
    {
        throw "fuck you !!!";
    }
    else if (lastLayer->layerType == LayerType::PoolingLayer)
    {
        throw "fuck you !!!";
    }
    else if (lastLayer->layerType == LayerType::ConvolutionLayer)
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
        vector<int> lossShape = {Layer::batch_size, channel, in_rows, in_cols};
        vector<int> batchoutputShape = {Layer::batch_size, channel, out_rows, out_cols};
        vector<int> batchinputShape = {Layer::batch_size, channel, in_rows, in_cols};

        vector<int> maxindexShape = {Layer::batch_size, channel, out_rows, out_cols};

        Layer::_loss = new Onion(lossShape, Layer::datawhere);
        Layer::batch_output = new Onion(batchoutputShape, Layer::datawhere);
        Layer::batch_input = new Onion(batchinputShape, Layer::datawhere);

        max_index = new Onion(maxindexShape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        vector<int> inputShape = {channel, in_rows, in_cols};
        vector<int> outputShape = {channel, out_rows, out_cols};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }

}

void PoolLayer::trainForword(Onion* batch_input)
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

void PoolLayer::_forword(Onion* input)
{
    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();

    for (int cha = 0; cha < channel; ++cha)
    {
        for (int r = 0; r < this->_r_times; ++r)
        {
            for (int c = 0; c < this->_c_times; ++c)
            {
                int _bigindex = cha*in_rows*in_cols + (r*pooling_rows) * in_cols + (c*pooling_cols);
                double max = inputPtr[_bigindex];

                int maxIndex = cha*_r_times*_c_times + r*_c_times + c;
                for (int p_r = 0; p_r < pooling_rows; ++p_r)
                {
                    for (int p_c = 0; p_c < pooling_cols; ++p_c)
                    {
                        int inindex = cha*in_rows*in_cols + (r*pooling_rows + p_r) * in_cols + (c*pooling_cols + p_c);
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

void PoolLayer::trainBackword(Onion* loss)
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

void PoolLayer::_CPUZeroGrad()
{
    Layer::_loss->setAllData(0);
}
void PoolLayer::clac_loss(Onion* batch_output)
{

}
void PoolLayer::_CPUclac_gradient(Onion* nextLayerBatchLoss)
{
    double* lossPtr = Layer::_loss->getdataPtr();
    double* nextLossPtr = nextLayerBatchLoss->getdataPtr();
    double* maxIndexPtr = this->max_index->getdataPtr();

    for (int i = 0; i < max_index->Size(); ++i)
    {
        int lossindex = (int)maxIndexPtr[i];
        lossPtr[lossindex] = nextLossPtr[i];
    }
}



void PoolLayer::_GPUZeroGrad()
{

}
void PoolLayer::_GPUclac_gradient(Onion* nextLayerBatchLoss)
{

}
void PoolLayer::_GPUupdate()
{

}

void PoolLayer::_CPUpooling(Onion* batch_input)
{
    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();
    double* maxIndexPtr = this->max_index->getdataPtr();

    for (int b = 0; b < Layer::batch_size; ++b)
    {
        for (int cha = 0; cha < channel; ++cha)
        {
            for (int r = 0; r < this->_r_times; ++r)
            {
                for (int c = 0; c < this->_c_times; ++c)
                {
                    int _bigindex = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows) * in_cols + (c*pooling_cols);
                    double max = batchinputPtr[_bigindex];

                    int maxIndex = b*channel*_r_times*_c_times + cha*_r_times*_c_times + r*_c_times + c;
                    maxIndexPtr[maxIndex] = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows)*in_cols + c*pooling_cols;
                    for (int p_r = 0; p_r < pooling_rows; ++p_r)
                    {
                        for (int p_c = 0; p_c < pooling_cols; ++p_c)
                        {
                            int inindex = b*channel*in_rows*in_cols + cha*in_rows*in_cols + (r*pooling_rows + p_r) * in_cols + (c*pooling_cols + p_c);
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

void PoolLayer::_GPUpooling(Onion* batch_input)
{

}

void PoolLayer::initWeight()
{   
    throw "Pooling Layer no Weight , SB";
}


