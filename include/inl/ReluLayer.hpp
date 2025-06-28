#include "ReluLayer.h"
#include "Onion.h"
#include "Layer.h"
#include "ConvLayer.h"
#include "FullconnectionLayer.h"
#include "VIewLayer.h"
#include "Timer/Timer.h"

ReluLayer::ReluLayer()
{
    Layer::layerType = LayerType::ReluLayerENUM;
}

ReluLayer::~ReluLayer()
{
}

void ReluLayer::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::ConvolutionLayerENUM)
    {
        ConvLayer* c = static_cast<ConvLayer*>(lastLayer);
        this->rows = c->out_rows;
        this->cols = c->out_cols;
        this->channel = c->out_channel;
        this->input_num = c->out_rows*c->out_cols*c->out_channel;
    }
    else if (lastLayer->layerType == LayerType::FullConnectionLayerENUM)
    {
        FullconnectionLayer* fc = static_cast<FullconnectionLayer*>(lastLayer);
        this->rows = fc->output_num;
        this->cols = 1;
        this->channel = 1;
        this->input_num = fc->output_num;
    }
    else if (lastLayer->layerType == LayerType::ViewLayerENUM)
    {
        ViewLayer* v = static_cast<ViewLayer*>(lastLayer);
        this->rows = v->output_num;
        this->cols = 1;
        this->channel = 1;
        this->input_num = v->output_num;
    }
    else 
    {
        throw "error ouccr";
    }




    if (Layer::modelType == ModelType::Train)
    {
        OnionShape batchoutput_shape = {Layer::batch_size, input_num};
        OnionShape batchinput_shape = {Layer::batch_size, input_num};    
        OnionShape loss_shape = {Layer::batch_size, input_num};

        Layer::batch_input.initOnion(batchinput_shape, Layer::datawhere);
        Layer::batch_output.initOnion(batchoutput_shape, Layer::datawhere);
        Layer::_loss.initOnion(loss_shape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {input_num};
        OnionShape outputShape = {input_num};

        Layer::input.initOnion(inputShape, Layer::datawhere);
        Layer::output.initOnion(outputShape, Layer::datawhere);
    }
}

void ReluLayer::_forword(Onion& input)
{
    for (size_t i = 0; i < input.Size(); ++i)
    {
        Layer::output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

void ReluLayer::trainForword(Onion& batch_input)
{
    Layer::batch_input.CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        CPUforword();
    }
}

void ReluLayer::trainBackword(Onion& loss)
{
    Timer t(this);
    if (Layer::datawhere == dataWhere::CPU)
    {
        CPUZeroGrad();
        CPUclac_gradient(loss);
    }
    else
    {
        GPUZeroGrad();
        GPUclac_gradient(loss);
    }
}

void ReluLayer::CPUclac_gradient(Onion& loss)
{
    for (size_t i = 0; i < Layer::batch_input.Size(); ++i)
    {
        Layer::_loss[i] = (Layer::batch_input[i] > 0) ? loss[i] : 0;
    }
}

void ReluLayer::CPUforword()
{
    for (size_t i = 0; i < batch_input.Size(); ++i)
    {
        Layer::batch_output[i] = (Layer::batch_input[i] > 0) ? Layer::batch_input[i] : 0;
    }
}

void ReluLayer::CPUZeroGrad()
{
    Layer::_loss.setAllData(0);
}

void ReluLayer::GPUZeroGrad()
{

}

void ReluLayer::GPUclac_gradient(Onion& loss)
{

}

void ReluLayer::initWeight()
{
    
}

void* ReluLayer::getWeight()
{
    return nullptr;
}