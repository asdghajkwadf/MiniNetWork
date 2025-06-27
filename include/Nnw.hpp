#ifndef _NNW_HPP_
#define _NNW_HPP_


#include <algorithm>

#include "Nnw.h"
#include "DataLoader.h"

#include "Layer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "ViewLayer.h"
#include "FullconnectionLayer.h"
#include "SoftmaxLayer.h"
#include "ReluLayer.h"

#include "Model.h"
#include "Batch.h"

NetWork::NetWork(ModelType type)
{
    this->NetWorlType = type;
}

NetWork::~NetWork()
{
    for (auto i : _layer)
    {
        delete i;
    }
}

void NetWork::AddLayer(ModelSet::LayerBase& l)
{
    // static layer_num = 0;
    // if (layer_num == 1)
    // {

    // }
    if (l.layerType == LayerType::ConvolutionLayerENUM)
    {
        auto& al = static_cast<ModelSet::Conv&>(l);
        _layer.push_back(new ConvLayer(al.step, al.in_channel, al.out_channel));
    }
    else if (l.layerType == LayerType::MaxPoolingLayerENUM)
    {
        auto& al = static_cast<ModelSet::MaxPool&>(l);
        _layer.push_back(new MaxPoolLayer(PoolType::Maxpool));
    }
    else if (l.layerType == LayerType::ViewLayerENUM)
    {
        auto& al = static_cast<ModelSet::View&>(l);
        _layer.push_back(new ViewLayer());
    }
    else if (l.layerType == LayerType::FullConnectionLayerENUM)
    {
        auto& al = static_cast<ModelSet::Fullconnection&>(l);
        _layer.push_back(new FullconnectionLayer(al.output_num));
    }
    else if (l.layerType == LayerType::ReluLayerENUM)
    {
        auto& al = static_cast<ModelSet::Relu&>(l);
        _layer.push_back(new ReluLayer());
    }
    else if (l.layerType == LayerType::SoftmaxLayerENUM)
    {
        auto& al = static_cast<ModelSet::SoftMax&>(l);
        _layer.push_back(new SoftmaxLayer());
    }
}

void NetWork::setMpdelType(ModelType mt)
{
    for (auto l : _layer)
    {
        l->modelType = mt;
        l->lr = lr;
    }
}

void NetWork::initLayerMatrix()
{
    for (auto i = 0; i < _layer.size(); ++i)
    {
        if (i == 0)
        {
            StartLayer* s = new StartLayer(dataloader);
            _layer.insert(_layer.begin(), s);
            s->batch_size = _Batch_Size();
            s->initMatrix(nullptr);
        }
        else
        {
            _layer.at(i)->initMatrix(_layer[i-1]);
        }
    }
}

void NetWork::moveData(dataWhere where)
{
    for (auto i : _layer)
    {
        i->datawhere = where;
    }
}

void NetWork::train(size_t epoch, double lr, DataLoader* dataLoader,  size_t batch_size)
{
    Batch_size = batch_size;
    this->dataloader = dataLoader;
    this->lr = lr;
    setMpdelType(ModelType::Train);
    initLayerMatrix();
    initBatch();
    _reverseLayer = _layer; 
    reverse(_reverseLayer.begin(),_reverseLayer.end());

    for (size_t t = 0; t < epoch; ++t)
    {
        static_cast<SoftmaxLayer*>(_layer.back())->loss_sum = 0;
        while (true)
        {
            Batch* batch = getBatch();
            if (!batch->full)
            {
                break;
            }
            Train_forword(batch);
            Train_backword(batch);
        }
        printLoss();
    }
}

void NetWork::test(DataLoader* dataLoader)
{
    setMpdelType(ModelType::Inference);
    initLayerMatrix();

    size_t correctNum = 0;

    std::vector<size_t> sampleShape = {dataLoader->sample_channel, dataLoader->rows, dataLoader->cols};
    Onion* sampleOnion = new Onion(sampleShape, dataWhere::CPU);
    while(true)
    {
        PicSample* p = dataLoader->getTestSample();
        if (p == nullptr)
        {
            break;
        }
        memcpy(sampleOnion->getdataPtr(), p->getData(), sizeof(double) * sampleOnion->Size());
        for (size_t i = 0; i < _layer.size(); ++i)
        {
            if (i == 0)
            {
                _layer[i]->_forword(sampleOnion);
            }
            else 
            {
                _layer[i]->_forword(_layer[i-1]->output);
            }
        }

        SoftmaxLayer* s = static_cast<SoftmaxLayer*>(_layer.back());
        result r = s->getResult();
        if (r.ID == (size_t)(p->getID()))
        {
            correctNum++;
        }
    }
    double CorrectRate = (double)correctNum / (dataLoader->getTestSampleNum());
    std::cout << "correct rate is: " << CorrectRate << std::endl;
}

void NetWork::Train_forword(Batch* batch)
{
    for (size_t i = 0; i < _layer.size(); ++i)
    {
        if (i == 0)
        {
            _layer[i]->trainForword(batch->data);
        }
        else 
        {
            _layer[i]->trainForword(_layer[i-1]->batch_output);
        }
    }
}

void NetWork::Train_backword(Batch* batch)
{
    for (size_t l = 0; l < _reverseLayer.size(); ++l)
    {
        if (l == 0)
        {
            _reverseLayer.at(l)->trainBackword(batch->Label);
        }
        else if (l == _reverseLayer.size()-1)
        {
            return ;
        }
        else
        {
            _reverseLayer.at(l)->trainBackword(_reverseLayer[l-1]->_loss);
        }
    }
}

void NetWork::printLoss()
{
    static size_t times = 0;
    std::cout << times << ": " <<static_cast<SoftmaxLayer*>(_layer.back())->loss_sum << std::endl;
    ++times;
}

void NetWork::forword()
{
    if (NetWorlType == Inference)
    {
        Inference_forword();
    }
}

void NetWork::Inference_forword()
{
}



void NetWork::initBatch()
{
    batch = new Batch();

    std::vector<size_t> batchdataShape = {Batch_size, (dataloader->sample_channel), (dataloader->rows), (dataloader->cols)};
    std::vector<size_t> batchOnebotShape = {Batch_size, (dataloader->class_num)};
    std::vector<size_t> batchLabelShape = {Batch_size};

    batch->data = new Onion(batchdataShape, dataWhere::CPU);
    batch->one_bot = new Onion(batchOnebotShape, dataWhere::CPU);
    batch->Label = new Onion(batchLabelShape, dataWhere::CPU);

    batch->size = Batch_size;
}

Batch* NetWork::getBatch()
{

    // 这司马getBatch写得就是一坨屎
    static auto it = 0;
    
    double* batchDataPtr = batch->data->getdataPtr();
    double* batchOneBotPtr = batch->one_bot->getdataPtr();
    double* batchLabelPtr = batch->Label->getdataPtr();

    for (size_t b = 0; b < Batch_size; ++b)
    {
        batch->batch_index = it / Batch_size;
        if (it < dataloader->_TrainSample->size())
        {
            batch->full = true;

            double* dataPtr = dataloader->_TrainSample->at(it)->getData();   
            size_t index = it*batch->data->Size();
            memcpy(batchDataPtr + b*(dataloader->sample_channel)*(dataloader->rows)*(dataloader->cols), dataPtr, (dataloader->sample_channel)*(dataloader->rows)*(dataloader->cols)*sizeof(double)); 

            double* OnebotPtr = (dataloader->_TrainSample)->at(it)->getOneBot();

            memcpy(batchOneBotPtr + b*(dataloader->class_num), OnebotPtr, (dataloader->class_num)*sizeof(double));   

            double labelPtr = (dataloader->_TrainSample)->at(it)->getID();
            batchLabelPtr[b] = labelPtr;
            ++it;
        }
        else
        {
            batch->full = false;
            it = 0;
            break;
        }
    }
    return batch;
}

size_t NetWork::_Batch_Size()
{
    return Batch_size;
}


#endif