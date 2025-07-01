#include "Layer.h"
#include "ViewLayer.h"
#include "Onion.h"
#include "PoolLayer.h"
#include "Timer/Timer.h"

ViewLayer::ViewLayer()
{
    Layer::layerType = LayerType::ViewLayerENUM;
}

ViewLayer::~ViewLayer()
{

}

void ViewLayer::setChannel(size_t channel)
{
    this->in_channel = channel;
}

void ViewLayer::initMatrix(Layer* lastLayer, dataWhere where)
{
    Layer::batch_size = lastLayer->batch_size;
    Layer::datawhere = where;
    if (lastLayer->layerType == LayerType::MaxPoolingLayerENUM)
    {
        MaxPoolLayer* p = static_cast<MaxPoolLayer*>(lastLayer);
        this->in_rows =  p->out_rows;
        this->in_cols = p->out_cols;
        this->in_channel = p->channel;
    }

    this->output_num = in_channel*in_rows*in_cols;




    if (Layer::modelType == ModelType::Train)
    {
        OnionShape batchouputShape = {Layer::batch_size, in_rows*in_cols*in_channel};
        OnionShape lossShape = {Layer::batch_size, output_num};

        batch_output.initOnion(batchouputShape, Layer::datawhere);
        _loss.initOnion(lossShape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {in_channel, in_rows, in_cols};
        OnionShape outputShape = {in_channel*in_rows*in_cols};

        Layer::input.initOnion(inputShape, Layer::datawhere);
        Layer::output.initOnion(outputShape, Layer::datawhere);
    }
}

void ViewLayer::trainForword(Onion& batch_input)
{
    if (Layer::datawhere == dataWhere::CPU)
    {
        // memcpy(batchoutputPtr, batchinputPtr, sizeof(double) * batch_input.Size());
        Layer::batch_output.CopyData(batch_input);
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void ViewLayer::_forword(Onion& input)
{
    if (Layer::datawhere == dataWhere::CPU)
    {
        Layer::output.CopyData(input);
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void ViewLayer::trainBackword(Onion& loss)
{
    
    if (Layer::datawhere == dataWhere::CPU)
    {
        Layer::_loss.CopyData(loss);
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void* ViewLayer::getWeight()
{
    throw "ViewLayer Layer no funking weight";
}

void ViewLayer::initWeight()
{
    throw "ViewLayer Layer no weight";
}
