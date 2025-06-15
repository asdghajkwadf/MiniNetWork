#include "Relu.h"
#include "Onion.h"
#include "layer.h"
#include "Conv.h"
#include "Fc.h"
#include "View.h"

Relu::Relu()
{
    Layer::layerType = LayerType::ReluLayer;
}

Relu::~Relu()
{
}

void Relu::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::ConvolutionLayer)
    {
        ConvLayer* c = static_cast<ConvLayer*>(lastLayer);
        this->rows = c->out_rows;
        this->cols = c->out_cols;
        this->channel = c->out_channel;
        this->input_num = c->out_rows*c->out_cols*c->out_channel;
    }
    else if (lastLayer->layerType == LayerType::FullConnectionLayer)
    {
        FullConnection* fc = static_cast<FullConnection*>(lastLayer);
        this->rows = fc->output_num;
        this->cols = 1;
        this->channel = 1;
        this->input_num = fc->output_num;
    }
    else if (lastLayer->layerType == LayerType::ViewLayer)
    {
        View* v = static_cast<View*>(lastLayer);
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
        vector<int> batchoutput_shape = {Layer::batch_size, input_num};
        vector<int> batchinput_shape = {Layer::batch_size, input_num};    
        vector<int> loss_shape = {Layer::batch_size, input_num};

        Layer::batch_input = new Onion(batchinput_shape, Layer::datawhere);
        Layer::batch_output = new Onion(batchoutput_shape, Layer::datawhere);
        Layer::_loss = new Onion(loss_shape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        vector<int> inputShape = {input_num};
        vector<int> outputShape = {input_num};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }
}

void Relu::_forword(Onion* input)
{
    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();
    for (int i = 0; i < input->Size(); ++i)
    {
        outputPtr[i] = (inputPtr[i] > 0) ? inputPtr[i] : 0;
    }
}

void Relu::trainForword(Onion* batch_input)
{
    Layer::batch_input->CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        CPUforword(batch_input);
    }
}

void Relu::trainBackword(Onion* loss)
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

void Relu::CPUclac_gradient(Onion* loss)
{
    double* nextLossPtr = loss->getdataPtr();
    double* batchinputPtr = Layer::batch_input->getdataPtr();
    double* lossPtr = Layer::_loss->getdataPtr();
    for (int i = 0; i < Layer::batch_input->Size(); ++i)
    {
        lossPtr[i] = (batchinputPtr[i] > 0) ? nextLossPtr[i] : 0;
    }
}

void Relu::CPUforword(Onion* batch_input)
{
    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();
    for (int i = 0; i < batch_input->Size(); ++i)
    {
        batchoutputPtr[i] = (batchinputPtr[i] > 0) ? batchinputPtr[i] : 0;
    }
}

void Relu::CPUZeroGrad()
{
    Layer::_loss->setAllData(0);
}

void Relu::GPUZeroGrad()
{

}

void Relu::GPUclac_gradient(Onion* loss)
{

}

void Relu::initWeight()
{
    
}

void* Relu::getWeight()
{
    return nullptr;
}