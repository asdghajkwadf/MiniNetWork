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

void ViewLayer::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

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
    double* batchinputPtr = batch_input.getdataPtr();
    double* batchoutputPtr = batch_output.getdataPtr();

    if (Layer::datawhere == dataWhere::CPU)
    {
        memcpy(batchoutputPtr, batchinputPtr, sizeof(double) * batch_input.Size());
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void ViewLayer::_forword(Onion& input)
{
    double* inputPtr = input.getdataPtr();
    double* outputPtr = Layer::output.getdataPtr();
    if (Layer::datawhere == dataWhere::CPU)
    {
        memcpy(outputPtr, inputPtr, sizeof(double) * input.Size());
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void ViewLayer::trainBackword(Onion& loss)
{
    double* lossPtr = Layer::_loss.getdataPtr();
    double* inlossPtr = loss.getdataPtr();

    Timer t(this);
    if (Layer::datawhere == dataWhere::CPU)
    {
        memcpy(lossPtr, inlossPtr, sizeof(double) * loss.Size());
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
