#include "fc.h"
#include "Onion.h"
#include "layer.h"
#include "View.h"
#include "Relu.h"
#include "Softmax.h"

#include <iostream>


FullConnection::FullConnection(int output_num)
{
    Layer::layerType = FullConnectionLayer;
    this->output_num = output_num;
}

FullConnection::~FullConnection()
{
    if (_w != nullptr)
    {
        delete _w;
    }

    if (_b != nullptr)
    {
        delete _b;
    }

    if (_w_grad != nullptr)
    {
        delete _w_grad;
    }

    if (_b_grad != nullptr)
    {
        delete _b_grad;
    }
}

void FullConnection::_forword(Onion* input)
{
    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();

    double* wPtr = _w->getdataPtr();
    double* bPtr = _b->getdataPtr();

    for (int out = 0; out < output_num; out++)
    {
        double _sum = 0;
        for (int in = 0; in < input_num; in++)
        {
            _sum += wPtr[out*input_num + in] * inputPtr[in];
        }
        outputPtr[out] = _sum + bPtr[out];
    }
}

void FullConnection::trainForword(Onion* batch_input)
{
    Layer::batch_input->CopyData(batch_input);
    if (Layer::datawhere == dataWhere::CPU)
    {
        _CPUforword(batch_input);
    }
    else if (Layer::datawhere == dataWhere::GPU)
    {
        _GPUforword(batch_input);
    }
}

void FullConnection::trainBackword(Onion* loss)
{
    Timer t(this);
    // double* p = loss->getdataPtr();
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

void FullConnection::_GPUZeroGrad()
{

}

void FullConnection::_GPUclac_gradient(Onion* loss)
{

}

void FullConnection::_GPUupdate()
{

}

void FullConnection::_GPUforword(Onion* batch_input)
{

}




void FullConnection::_CPUZeroGrad()
{
    _w_grad->setAllData(0);
    _b_grad->setAllData(0);
    Layer::_loss->setAllData(0);
}

void FullConnection::_CPUclac_gradient(Onion* nextLayerBatchLoss)
{
    double* nextLayerBatchLossPtr = nextLayerBatchLoss->getdataPtr();
    double* batchinputPtr = Layer::batch_input->getdataPtr();

    double* wPtr = _w->getdataPtr();
    double* bPtr = _b->getdataPtr();

    double* wGradPtr = _w_grad->getdataPtr();
    double* bGradPtr = _b_grad->getdataPtr();

    double* lossPtr = Layer::_loss->getdataPtr();

    for (int b = 0; b < Layer::batch_size; ++b)
    {
        for (int r = 0; r < output_num; ++r)
        {
            for (int c = 0; c < input_num; ++c)  
            {
                lossPtr[b*input_num + c] += nextLayerBatchLossPtr[b*output_num + r] * wPtr[r*input_num + c];    
                wGradPtr[r*input_num + c] += nextLayerBatchLossPtr[b*output_num + r] * batchinputPtr[b*input_num + c] / Layer::batch_size;
            }
            bGradPtr[r] += nextLayerBatchLossPtr[b*output_num + r] / Layer::batch_size;
        }
    }




    // for (int b = 0; b < Layer::batch_size; ++b)
    // {
    //     for (int r = 0; r < output_num; ++r)
    //     {
    //         bGradPtr[r] += nextLayerBatchLossPtr[b*output_num + r] / Layer::batch_size;
    //     } 
    // }



    // for (int b = 0; b < Layer::batch_size; ++b)
    // {
    //     for (int r = 0; r < output_num; ++r)
    //     {
    //         for (int c = 0; c < input_num; ++c)
    //         {
    //             wGradPtr[r*input_num + c] += nextLayerBatchLossPtr[b*output_num + r] * batchinputPtr[b*input_num + c] / Layer::batch_size;
    //         }
    //     }
    // }

}

void FullConnection::_CPUupdate()
{
    double* wGradPtr = _w_grad->getdataPtr();
    double* bGradPtr = _b_grad->getdataPtr();

    double* wPtr = _w->getdataPtr();
    double* bPtr = _b->getdataPtr();
    for (int b = 0; b < output_num; ++b)
    {
        bPtr[b] -= bGradPtr[b] * Layer::lr;
    }

    for (int r = 0; r < output_num; ++r)
    {
        for (int c = 0; c < input_num; ++c)
        {
            wPtr[r*input_num + c] -= wGradPtr[r*input_num + c] * Layer::lr;
        }
    }
}

void FullConnection::_CPUforword(Onion* batch_input)
{   
    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = Layer::batch_output->getdataPtr();

    double* wPtr = _w->getdataPtr();
    double* bPtr = _b->getdataPtr();

    for (int b = 0; b < Layer::batch_size; ++b)
    {
        for (int out = 0; out < output_num; out++)
        {
            double _sum = 0;
            for (int in = 0; in < input_num; in++)
            {
                _sum += wPtr[out*input_num + in] * batchinputPtr[b*input_num + in];
            }
            batchoutputPtr[b*output_num + out] = _sum + bPtr[out];
        }
    }
}






void FullConnection::initGradient()
{
    vector<int> _w_grad_Shape = {output_num, input_num};
    vector<int> _b_grad_Shape = {output_num};
    _w_grad = new Onion(_w_grad_Shape, Layer::datawhere);
    _b_grad = new Onion(_b_grad_Shape, Layer::datawhere);
}

void FullConnection::initWeight()
{
    vector<int> w_Shape = {output_num, input_num};
    vector<int> b_Shape = {output_num};
    _w = new Onion(w_Shape, Layer::datawhere);
    _b = new Onion(b_Shape, Layer::datawhere);
    _w->initdata(-0.05, 0.05);
    _b->initdata(-0.05, 0.05);
}

void FullConnection::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::ViewLayer)
    {
        View* v = static_cast<View*>(lastLayer);
        this->input_num = v->output_num;
    }
    else if (lastLayer->layerType == LayerType::FullConnectionLayer)
    {
        FullConnection* fc = static_cast<FullConnection*>(lastLayer);
        this->input_num = fc->output_num;
    }
    else if (lastLayer->layerType == LayerType::ReluLayer)
    {
        Relu* r = static_cast<Relu*>(lastLayer);
        this->input_num = r->input_num;
    }
    else
    {
        throw "ouccr some error, fuck ! ";
    }




    if (Layer::modelType == ModelType::Train)
    {
        vector<int> batchoutputShape = {Layer::batch_size, output_num};
        vector<int> batchinputShape = {Layer::batch_size, input_num};

        vector<int> lossShape = {Layer::batch_size, input_num};

        Layer::batch_input = new Onion(batchinputShape, Layer::datawhere);
        Layer::batch_output = new Onion(batchoutputShape, Layer::datawhere);
        Layer::_loss = new Onion(lossShape, Layer::datawhere);

        initWeight();
        initGradient();
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        vector<int> inputShape = {input_num};
        vector<int> outputShape = {output_num};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }
}

void* FullConnection::getWeight()
{
    return nullptr;
}
