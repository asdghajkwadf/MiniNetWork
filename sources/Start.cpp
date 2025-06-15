#include "Start.h"
#include "DataLoader.h"

StartLayer::StartLayer(DataLoader* dataloader) : rows(dataloader->rows), cols(dataloader->cols), channel(dataloader->sample_channel)
{
    this->dataloader = dataloader;
    Layer::layerType = LayerType::StartingLayer;
    Layer::batch_size = dataloader->_Batch_Size();
}

StartLayer::~StartLayer()
{

}

void StartLayer::setDataLoader(DataLoader& dataLoader)
{

}

void StartLayer:: _forword(Onion* input)
{
    Layer::output = input;
}

void StartLayer::trainForword(Onion* batch_input)
{
    Layer::batch_output = batch_input;
    double* p =batch_input->getdataPtr();
}

void StartLayer::trainBackword(Onion* loss)
{
    
}

void StartLayer::initMatrix(Layer* lastLayer)
{

}