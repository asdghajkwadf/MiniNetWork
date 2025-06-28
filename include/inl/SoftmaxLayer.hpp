#include "SoftmaxLayer.h"
#include "Onion.h"
#include "Layer.h"
#include "FullconnectionLayer.h"
#include "Timer/Timer.h"
#include <iostream>

SoftmaxLayer::SoftmaxLayer()
{
    Layer::layerType = SoftmaxLayerENUM;
}

SoftmaxLayer::~SoftmaxLayer()
{
}

void SoftmaxLayer::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    
    if (lastLayer->layerType == LayerType::FullConnectionLayerENUM)
    {
        FullconnectionLayer* fc = static_cast<FullconnectionLayer*>(lastLayer);
        oneBot_num = fc->output_num;
    }
    else
    {
        throw "fc you";
    }

    if (Layer::modelType == ModelType::Train)
    {
        OnionShape batchoutput_shape = {Layer::batch_size, oneBot_num};
        batch_output.initOnion(batchoutput_shape, Layer::datawhere);
        
        OnionShape loss_shape = {Layer::batch_size, oneBot_num};
        _loss.initOnion(loss_shape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {oneBot_num};
        OnionShape outputShape = {oneBot_num};

        Layer::input.initOnion(inputShape, Layer::datawhere);
        Layer::output.initOnion(outputShape, Layer::datawhere);
    }
}

void SoftmaxLayer::_forword(Onion& input)
{
    double* inputPtr = input.getdataPtr();
    double* outputPtr = Layer::output.getdataPtr();
    double total = 0;
    for(size_t i = 0; i < oneBot_num; ++i)
    {
        total += exp(inputPtr[i]);
    }

    for(size_t i = 0; i < oneBot_num; ++i)
    {
        outputPtr[i] = exp(inputPtr[i]) / total;
    }

    double max = 0;

    ID = 0;
    confiden = outputPtr[0];
    for (size_t i = 0; i < oneBot_num; ++i)
    {
        if (outputPtr[i] > confiden)
        {
            confiden = outputPtr[i];
            ID = i;
        }
    }
}

result SoftmaxLayer::getResult()
{
    result r;
    r.ID = ID;
    r.confiden = confiden;
    return r;
}

void SoftmaxLayer::trainForword(Onion& batch_input)
{
    if (Layer::datawhere == dataWhere::CPU)
    {
        CPUforword(batch_input);
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        
    }
}

void SoftmaxLayer::trainBackword(Onion& Label)
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

void SoftmaxLayer::CPUclac_loss(Onion& Label)
{
    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t o = 0; o < oneBot_num; ++o)
        {
            if (o == Label[b])
            {
                Layer::_loss[b*oneBot_num + o] = Layer::batch_output[b*oneBot_num + o] - 1;
                // std::std::cout << batchoutputPtr[b*oneBot_num + o] << " " << loss_sum << std::endl;
                loss_sum += -log(Layer::batch_output[b*oneBot_num + o]);
            }
            else 
            {
                Layer::_loss[b*oneBot_num + o] = (Layer::batch_output[b*oneBot_num + o]);
            }
            // std::std::cout << batchoutputPtr[b*oneBot_num + o] << " ";
        }
        // std::std::cout << std::endl;
    }
    // std::std::cout << std::endl << std::endl;;
}

void SoftmaxLayer::CPUforword(Onion& batch_input)
{
    double* batchinputPtr = batch_input.getdataPtr();
    double* batchoutputPtr = Layer::batch_output.getdataPtr();
    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        double total = 0;
        for(size_t i = 0; i < oneBot_num; ++i)
        {
            total += exp(batchinputPtr[b*oneBot_num + i]);
        }

        for(size_t i = 0; i < oneBot_num; ++i)
        {
            batchoutputPtr[b*oneBot_num + i] = exp(batchinputPtr[b*oneBot_num + i]) / total;
            // std::cout << batchoutputPtr[b*oneBot_num + i] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl;
}

void SoftmaxLayer::CPUZeroGrad()
{
    _loss.setAllData(0);
}

void SoftmaxLayer::GPUZeroGrad()
{

}

void  SoftmaxLayer::GPUclac_loss(Onion& Label)
{

}

void SoftmaxLayer::initWeight()
{
    
}

void* SoftmaxLayer::getWeight()
{
    throw "SoftmaxLayer no weight";
    return nullptr;
}