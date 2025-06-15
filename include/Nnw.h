#ifndef _NNW_H_

#define _NNW_H_

#include "DataLoader.h"
#include "layer.h"
#include "Onion.h"
#include <vector>
using namespace std;


class NetWork
{
    public:
        NetWork(ModelType type);
        ~NetWork();

        void train(int epoch, double lr, DataLoader* dataLoader);
        void test(DataLoader* dataLoader);

        ModelType NetWorlType = Inference;

        void moveData(dataWhere where);

        void forword();
        void calc_loss(Batch* batch);
        void AddLayer(Layer* l);
        void setMpdelType(ModelType mt);

        void printLoss();

        double lr = 0.001;

    private:

        void Inference_forword();
        void Train_forword(Batch* batch);
        void Train_backword(Batch* batch);

        int Batch_size = 0;

        void initLayerMatrix(DataLoader* dataLoader);
        // double*** input = nullptr;

        vector<Layer*> _reverseLayer;
        vector<Layer*> _layer; 
};










#endif