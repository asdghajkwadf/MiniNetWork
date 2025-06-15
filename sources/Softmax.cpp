#include "Softmax.h"
#include "Onion.h"
#include "layer.h"
#include "Fc.h"

#include <iostream>

SoftMax::SoftMax()
{
    Layer::layerType = SoftmaxLayer;
}

SoftMax::~SoftMax()
{
}

void SoftMax::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    
    if (lastLayer->layerType == LayerType::FullConnectionLayer)
    {
        FullConnection* fc = static_cast<FullConnection*>(lastLayer);
        oneBot_num = fc->output_num;
    }
    else
    {
        throw "fc you";
    }

    if (Layer::modelType == ModelType::Train)
    {
        vector<int> batchoutput_shape = {Layer::batch_size, oneBot_num};
        batch_output = new Onion(batchoutput_shape, Layer::datawhere);
        
        vector<int> loss_shape = {Layer::batch_size, oneBot_num};
        _loss = new Onion(loss_shape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        vector<int> inputShape = {oneBot_num};
        vector<int> outputShape = {oneBot_num};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }
}

void SoftMax::_forword(Onion* input)
{
    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();
    double total = 0;
    for(int i = 0; i < oneBot_num; ++i)
    {
        total += exp(inputPtr[i]);
    }

    for(int i = 0; i < oneBot_num; ++i)
    {
        outputPtr[i] = exp(inputPtr[i]) / total;
    }

    double max = 0;

    ID = 0;
    confiden = outputPtr[0];
    for (int i = 0; i < oneBot_num; ++i)
    {
        if (outputPtr[i] > confiden)
        {
            confiden = outputPtr[i];
            ID = i;
        }
    }
}

result SoftMax::getResult()
{
    result r;
    r.ID = ID;
    r.confiden = confiden;
    return r;
}

void SoftMax::trainForword(Onion* batch_input)
{
    if (Layer::datawhere == dataWhere::CPU)
    {
        CPUforword(batch_input);
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        
    }
}

void SoftMax::trainBackword(Onion* Label)
{
    Timer t(this);
    if (Layer::datawhere == dataWhere::CPU)
    {
        CPUZeroGrad();
        CPUclac_loss(Label);
    }
    else
    {
        GPUZeroGrad();
        GPUclac_loss(Label);
    }
}

void SoftMax::CPUclac_loss(Onion* Label)
{
    double* labelPtr = Label->getdataPtr();
    double* lossPtr = Layer::_loss->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();
    for (int b = 0; b < Layer::batch_size; ++b)
    {
        for (int o = 0; o < oneBot_num; ++o)
        {
            if (o == labelPtr[b])
            {
                lossPtr[b*oneBot_num + o] = batchoutputPtr[b*oneBot_num + o] - 1;
                // std::cout << batchoutputPtr[b*oneBot_num + o] << " " << loss_sum << std::endl;
                loss_sum += -log(batchoutputPtr[b*oneBot_num + o]);
            }
            else 
            {
                lossPtr[b*oneBot_num + o] = (batchoutputPtr[b*oneBot_num + o]);
            }
            // std::cout << batchoutputPtr[b*oneBot_num + o] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl << std::endl;;
}

void SoftMax::CPUforword(Onion* batch_input)
{
    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();
    for (int b = 0; b < Layer::batch_size; ++b)
    {
        double total = 0;
        for(int i = 0; i < oneBot_num; ++i)
        {
            total += exp(batchinputPtr[b*oneBot_num + i]);
        }

        for(int i = 0; i < oneBot_num; ++i)
        {
            batchoutputPtr[b*oneBot_num + i] = exp(batchinputPtr[b*oneBot_num + i]) / total;
            // std::cout << batchinputPtr[b*oneBot_num + i] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl;
}

void SoftMax::CPUZeroGrad()
{
    _loss->setAllData(0);
}

void SoftMax::GPUZeroGrad()
{

}

void  SoftMax::GPUclac_loss(Onion* Label)
{

}

void SoftMax::initWeight()
{
    
}

void* SoftMax::getWeight()
{
    throw "SoftMax no weight";
    return nullptr;
}