#include "FullconnectionLayer.h"
#include "Onion.h"
#include "Layer.h"
#include "ViewLayer.h"
#include "ReluLayer.h"
#include "SoftMaxLayer.h"
#include <iostream>
#include "Timer/Timer.h"

FullconnectionLayer::FullconnectionLayer(size_t output_num)
{
    Layer::layerType = FullConnectionLayerENUM;
    this->output_num = output_num;
}

FullconnectionLayer::~FullconnectionLayer()
{
    // if (_w != nullptr)
    // {
    //     delete _w;
    // }

    // if (_b != nullptr)
    // {
    //     delete _b;
    // }

    // if (_w_grad != nullptr)
    // {
    //     delete _w_grad;
    // }

    // if (_b_grad != nullptr)
    // {
    //     delete _b_grad;
    // }
}

void FullconnectionLayer::_forword(Onion& input)
{
    double* inputPtr = input.getdataPtr();
    double* outputPtr = Layer::output.getdataPtr();

    double* wPtr = _w.getdataPtr();
    double* bPtr = _b.getdataPtr();

    for (size_t out = 0; out < output_num; out++)
    {
        double _sum = 0;
        for (size_t in = 0; in < input_num; in++)
        {
            _sum += wPtr[out*input_num + in] * inputPtr[in];
        }
        outputPtr[out] = _sum + bPtr[out];
    }
}

void FullconnectionLayer::trainForword(Onion& batch_input)
{
    Layer::batch_input.CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUforword(batch_input);
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUforword(batch_input);
    }
}

void FullconnectionLayer::trainBackword(Onion& loss)
{
    Timer t(this);
    // double* p = loss.getdataPtr();
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUZeroGrad();
        _CPUclac_gradient(loss);
        _CPUupdate();
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUZeroGrad();
        _GPUclac_gradient(loss);
        _GPUupdate();
    }
}

void FullconnectionLayer::_GPUZeroGrad()
{

}

void FullconnectionLayer::_GPUclac_gradient(Onion& loss)
{

}

void FullconnectionLayer::_GPUupdate()
{

}

void FullconnectionLayer::_GPUforword(Onion& batch_input)
{

}




void FullconnectionLayer::_CPUZeroGrad()
{
    _w_grad.setAllData(0);
    _b_grad.setAllData(0);
    Layer::_loss.setAllData(0);
}

void FullconnectionLayer::_CPUclac_gradient(Onion& nextLayerBatchLoss)
{
    double* nextLayerBatchLossPtr = nextLayerBatchLoss.getdataPtr();
    double* batchinputPtr = Layer::batch_input.getdataPtr();

    double* wPtr = _w.getdataPtr();
    double* bPtr = _b.getdataPtr();

    double* wGradPtr = _w_grad.getdataPtr();
    double* bGradPtr = _b_grad.getdataPtr();

    double* lossPtr = Layer::_loss.getdataPtr();

    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t r = 0; r < output_num; ++r)
        {
            for (size_t c = 0; c < input_num; ++c)  
            {
                lossPtr[b*input_num + c] += nextLayerBatchLossPtr[b*output_num + r] * wPtr[r*input_num + c];    
                wGradPtr[r*input_num + c] += nextLayerBatchLossPtr[b*output_num + r] * batchinputPtr[b*input_num + c] / Layer::batch_size;
            }
            bGradPtr[r] += nextLayerBatchLossPtr[b*output_num + r] / Layer::batch_size;
        }
    }




    // for (size_t b = 0; b < Layer::batch_size; ++b)
    // {
    //     for (size_t r = 0; r < output_num; ++r)
    //     {
    //         bGradPtr[r] += nextLayerBatchLossPtr[b*output_num + r] / Layer::batch_size;
    //     } 
    // }



    // for (size_t b = 0; b < Layer::batch_size; ++b)
    // {
    //     for (size_t r = 0; r < output_num; ++r)
    //     {
    //         for (size_t c = 0; c < input_num; ++c)
    //         {
    //             wGradPtr[r*input_num + c] += nextLayerBatchLossPtr[b*output_num + r] * batchinputPtr[b*input_num + c] / Layer::batch_size;
    //         }
    //     }
    // }

}

void FullconnectionLayer::_CPUupdate()
{
    double* wGradPtr = _w_grad.getdataPtr();
    double* bGradPtr = _b_grad.getdataPtr();

    double* wPtr = _w.getdataPtr();
    double* bPtr = _b.getdataPtr();
    for (size_t b = 0; b < output_num; ++b)
    {
        bPtr[b] -= bGradPtr[b] * Layer::lr;
    }

    for (size_t r = 0; r < output_num; ++r)
    {
        for (size_t c = 0; c < input_num; ++c)
        {
            wPtr[r*input_num + c] -= wGradPtr[r*input_num + c] * Layer::lr;
        }
    }
}

void FullconnectionLayer::_CPUforword(Onion& batch_input)
{   
    double* batchinputPtr = batch_input.getdataPtr();
    double* batchoutputPtr = Layer::batch_output.getdataPtr();

    double* wPtr = _w.getdataPtr();
    double* bPtr = _b.getdataPtr();

    for (size_t b = 0; b < Layer::batch_size; ++b)
    {
        for (size_t out = 0; out < output_num; out++)
        {
            double _sum = 0;
            for (size_t in = 0; in < input_num; in++)
            {
                _sum += wPtr[out*input_num + in] * batchinputPtr[b*input_num + in];
            }
            batchoutputPtr[b*output_num + out] = _sum + bPtr[out];
        }
    }
}






void FullconnectionLayer::initGradient()
{
    OnionShape _w_grad_Shape = {output_num, input_num};
    OnionShape _b_grad_Shape = {output_num};
    _w_grad.initOnion(_w_grad_Shape, Layer::datawhere);
    _b_grad.initOnion(_b_grad_Shape, Layer::datawhere);
}

void FullconnectionLayer::initWeight()
{
    OnionShape w_Shape = {output_num, input_num};
    OnionShape b_Shape = {output_num};
    _w.initOnion(w_Shape, Layer::datawhere);
    _b.initOnion(b_Shape, Layer::datawhere);
    _w.initdata(-0.05, 0.05);
    _b.initdata(-0.05, 0.05);
}

void FullconnectionLayer::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::ViewLayerENUM)
    {
        ViewLayer* v = static_cast<ViewLayer*>(lastLayer);
        this->input_num = v->output_num;
    }
    else if (lastLayer->layerType == LayerType::FullConnectionLayerENUM)
    {
        FullconnectionLayer* fc = static_cast<FullconnectionLayer*>(lastLayer);
        this->input_num = fc->output_num;
    }
    else if (lastLayer->layerType == LayerType::ReluLayerENUM)
    {
        ReluLayer* r = static_cast<ReluLayer*>(lastLayer);
        this->input_num = r->input_num;
    }
    else
    {
        throw "ouccr some error, fuck ! ";
    }




    if (Layer::modelType == ModelType::Train)
    {
        OnionShape batchoutputShape = {Layer::batch_size, output_num};
        OnionShape batchinputShape = {Layer::batch_size, input_num};

        OnionShape lossShape = {Layer::batch_size, input_num};

        Layer::batch_input.initOnion(batchinputShape, Layer::datawhere);
        Layer::batch_output.initOnion(batchoutputShape, Layer::datawhere);
        Layer::_loss.initOnion(lossShape, Layer::datawhere);

        initWeight();
        initGradient();
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        OnionShape inputShape = {input_num};
        OnionShape outputShape = {output_num};

        Layer::input.initOnion(inputShape, Layer::datawhere);
        Layer::output.initOnion(outputShape, Layer::datawhere);
    }
}

void* FullconnectionLayer::getWeight()
{
    return nullptr;
}
