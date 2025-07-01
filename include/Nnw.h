#ifndef _NNW_H_
#define _NNW_H_

#include "DataLoader.h"
#include "Model.h"
#include "Layer/Layer.h"
#include "Onion.h"
#include <vector>
#include "Batch.h"






class NetWork
{
DataLoader* dataloader = nullptr;
Batch* batch = nullptr;  
public:
    NetWork(ModelType type);
    ~NetWork();

    void train(size_t epoch, double lr, DataLoader* dataLoader, size_t batch_size, dataWhere where = dataWhere::CPU);
    void test(DataLoader* dataLoader);

    ModelType NetWorlType = Inference;

    void moveData(dataWhere where);

    void forword();
    void calc_loss(Batch* batch);
    void AddLayer(ModelSet::LayerBase& l);
    void setMpdelType(ModelType mt);

    void printLoss();

    double lr = 0.001;

private:

    void Inference_forword();
    void Train_forword(Batch* batch);
    void Train_backword(Batch* batch);

    size_t Batch_size = 0;
    Batch* getBatch();
    size_t _Batch_Size();
    void initBatch();  

    void initLayerMatrix();
    // double*** input = nullptr;

    dataWhere where = dataWhere::CPU;

    std::vector<Layer*> _reverseLayer;
    std::vector<Layer*> _layer; 
};

#include "Nnw.hpp"

#endif