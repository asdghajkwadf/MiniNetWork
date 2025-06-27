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

        Layer::batch_input = new Onion(batchinput_shape, Layer::datawhere);
        Layer::batch_output = new Onion(batchoutput_shape, Layer::datawhere);
        Layer::_loss = new Onion(loss_shape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {input_num};
        OnionShape outputShape = {input_num};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }
}

void ReluLayer::_forword(Onion* input)
{
    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();
    for (size_t i = 0; i < input->Size(); ++i)
    {
        outputPtr[i] = (inputPtr[i] > 0) ? inputPtr[i] : 0;
    }
}

void ReluLayer::trainForword(Onion* batch_input)
{
    Layer::batch_input->CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        CPUforword();
    }
}

void ReluLayer::trainBackword(Onion* loss)
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

void ReluLayer::CPUclac_gradient(Onion* loss)
{
    double* nextLossPtr = loss->getdataPtr();
    double* batchinputPtr = Layer::batch_input->getdataPtr();
    double* lossPtr = Layer::_loss->getdataPtr();
    for (size_t i = 0; i < Layer::batch_input->Size(); ++i)
    {
        lossPtr[i] = (batchinputPtr[i] > 0) ? nextLossPtr[i] : 0;
    }
}

void ReluLayer::CPUforword()
{
    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();
    for (size_t i = 0; i < batch_input->Size(); ++i)
    {
        batchoutputPtr[i] = (batchinputPtr[i] > 0) ? batchinputPtr[i] : 0;
    }
}

void ReluLayer::CPUZeroGrad()
{
    Layer::_loss->setAllData(0);
}

void ReluLayer::GPUZeroGrad()
{

}

void ReluLayer::GPUclac_gradient(Onion* loss)
{

}

void ReluLayer::initWeight()
{
    
}

void* ReluLayer::getWeight()
{
    return nullptr;
}