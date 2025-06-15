#include "layer.h"
#include "View.h"
#include "Onion.h"
#include "Pool.h"


View::View()
{
    Layer::layerType = LayerType::ViewLayer;
}

View::~View()
{

}

void View::setChannel(int channel)
{
    this->in_channel = channel;
}

void View::initMatrix(Layer* lastLayer)
{
    Layer::batch_size = lastLayer->batch_size;

    if (lastLayer->layerType == LayerType::PoolingLayer)
    {
        PoolLayer* p = static_cast<PoolLayer*>(lastLayer);
        this->in_rows =  p->out_rows;
        this->in_cols = p->out_cols;
        this->in_channel = p->channel;
    }

    this->output_num = in_channel*in_rows*in_cols;




    if (Layer::modelType == ModelType::Train)
    {
        vector<int> batchouputShape = {Layer::batch_size, in_rows*in_cols*in_channel};
        vector<int> lossShape = {Layer::batch_size, output_num};

        batch_output = new Onion(batchouputShape, Layer::datawhere);
        _loss = new Onion(lossShape, Layer::datawhere);
    }
    else if (Layer::modelType == ModelType::Inference)
    {
        vector<int> inputShape = {in_channel, in_rows, in_cols};
        vector<int> outputShape = {in_channel*in_rows*in_cols};

        Layer::input = new Onion(inputShape, Layer::datawhere);
        Layer::output = new Onion(outputShape, Layer::datawhere);
    }
}

void View::trainForword(Onion* batch_input)
{
    double* batchinputPtr = batch_input->getdataPtr();
    double* batchoutputPtr = batch_output->getdataPtr();

    if (Layer::datawhere == dataWhere::CPU)
    {
        memcpy(batchoutputPtr, batchinputPtr, sizeof(double) * batch_input->Size());
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void View::_forword(Onion* input)
{
    double* inputPtr = input->getdataPtr();
    double* outputPtr = Layer::output->getdataPtr();
    if (Layer::datawhere == dataWhere::CPU)
    {
        memcpy(outputPtr, inputPtr, sizeof(double) * input->Size());
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void View::trainBackword(Onion* loss)
{
    double* lossPtr = Layer::_loss->getdataPtr();
    double* inlossPtr = loss->getdataPtr();

    Timer t(this);
    if (Layer::datawhere == dataWhere::CPU)
    {
        memcpy(lossPtr, inlossPtr, sizeof(double) * loss->Size());
    }
    else if (Layer::datawhere = dataWhere::GPU)
    {

    }
}

void View::_view(double*** input, double* output)
{
    for (int cha = 0; cha < this->in_channel; ++cha)
    {
        for (int r = 0; r < this->in_rows; ++r)
        {
            for (int c = 0; c < in_cols; ++c)
            {
                output[cha*r*c + r*c + c] = input[cha][r][c];
            }
        }
    }
}

void* View::getWeight()
{
    throw "View Layer no funking weight";
}

void View::initWeight()
{
    throw "View Layer no weight";
}
