#include "Nnw.h"
#include "layer.h"
#include "DataLoader.h"

#include "Conv.h"
#include "Pool.h"
#include "View.h"
#include "Fc.h"
#include "Softmax.h"
#include "Relu.h"


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

void NetWork::AddLayer(Layer* l)
{
    _layer.push_back(l);
    _reverseLayer.insert(_reverseLayer.begin(), l);
}

void NetWork::setMpdelType(ModelType mt)
{
    for (auto l : _layer)
    {
        l->modelType = mt;
        l->lr = lr;
    }
}

void NetWork::initLayerMatrix(DataLoader* dataloader)
{
    for (auto i = 0; i < _layer.size(); ++i)
    {
        if (i == 0)
        {
            _layer.at(i)->initMatrix(nullptr);
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

void NetWork::train(int epoch, double lr, DataLoader* dataLoader)
{
    this->lr = lr;
    setMpdelType(ModelType::Train);
    initLayerMatrix(dataLoader);
    dataLoader->initBatch();

    this->Batch_size = dataLoader->_Batch_Size();

    for (int t = 0; t < epoch; ++t)
    {
        static_cast<SoftMax*>(_layer.back())->loss_sum = 0;
        while (true)
        {
            Batch* batch = dataLoader->getBatch();
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
    initLayerMatrix(dataLoader);

    int correctNum = 0;

    vector<int> sampleShape = {dataLoader->sample_channel, dataLoader->rows, dataLoader->cols};
    Onion* sampleOnion = new Onion(sampleShape, dataWhere::CPU);
    while(true)
    {
        PicSample* p = dataLoader->getTestSample();
        if (p == nullptr)
        {
            break;
        }
        memcpy(sampleOnion->getdataPtr(), p->getData(), sizeof(double) * sampleOnion->Size());
        for (int i = 0; i < _layer.size(); ++i)
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

        SoftMax* s = static_cast<SoftMax*>(_layer.back());
        result r = s->getResult();
        if (r.ID == (int)(p->getID()))
        {
            correctNum++;
        }
    }
    double CorrectRate = (double)correctNum / (dataLoader->getTestSampleNum());
    std::cout << "correct rate is: " << CorrectRate << endl;
}

void NetWork::Train_forword(Batch* batch)
{
    for (int i = 0; i < _layer.size(); ++i)
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
    for (int l = 0; l < _reverseLayer.size(); ++l)
    {
        if (l == 0)
        {
            _reverseLayer.at(l)->trainBackword(batch->Label);
        }
        else
        {
            _reverseLayer.at(l)->trainBackword(_reverseLayer[l-1]->_loss);
        }
    }
}

void NetWork::printLoss()
{
    std::cout << static_cast<SoftMax*>(_layer.back())->loss_sum << endl;
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