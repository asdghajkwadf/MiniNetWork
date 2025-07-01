#ifndef _STARTLAYER_HPP_
#define _STARTLAYER_HPP_

#include "StartLayer.h"
#include "Layer.h"
#include "DataLoader.h"
#include "Onion.h"
#include "Model.h"

StartLayer::StartLayer(DataLoader* dataLoader) : dataLoader(dataLoader)
{
    Layer::layerType = LayerType::StartLayerENUM;
    this->channel = dataLoader->sample_channel;
    this->out_rows = dataLoader->rows;
    this->out_cols = dataLoader->cols;
}

StartLayer::~StartLayer()
{

}

void StartLayer::setBatchSize(size_t batch_size)
{
    Layer::batch_size = batch_size;
}

void StartLayer::initMatrix(Layer* lastLayer, dataWhere where)
{
    Layer::datawhere = where;
    OnionShape batchoutputShape = {Layer::batch_size, channel, out_rows, out_cols};
    OnionShape lossShape = {Layer::batch_size, channel, out_rows, out_cols};
    OnionShape batchinputShape = {Layer::batch_size, channel, out_rows, out_cols};

    Layer::batch_input.initOnion(batchinputShape, Layer::datawhere);
    Layer::batch_output.initOnion(batchoutputShape, Layer::datawhere);
    Layer::_loss.initOnion(lossShape, Layer::datawhere);

    
    OnionShape outputShape = {channel, out_rows, out_cols};
    Layer::output.initOnion(outputShape, Layer::datawhere);


}


void StartLayer::trainForword(Onion& batch_input)
{
    this->batch_output.CopyData(batch_input);
}

void StartLayer::trainBackword(Onion& loss)
{

}    

void StartLayer::_forword(Onion& input)
{
    Layer::output.CopyData(input);
}


#endif
