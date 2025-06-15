#include <iostream>
#include <opencv2/opencv.hpp>
#include "DataLoader.h"
#include "PreData.h"
#include "Layer.h"
#include "Nnw.h"
#include "Onion.h"

#include "Conv.h"
#include "Pool.h"
#include "View.h"
#include "Fc.h"
#include "Softmax.h"
#include "ReLU.h"
#include "Start.h"

int main(int)
{

    DataLoader* dataLoader = new DataLoader("D:/DATA/NetWork/data", 16);


    dataLoader->readfile(100);

    PicImporve::Normalization(dataLoader);
    PicImporve::Padding(dataLoader, 1);

    NetWork net = NetWork(Train);

    StartLayer* first = new StartLayer(dataLoader);
    ConvLayer* Conv1 = new ConvLayer(1, 1, 10);
    PoolLayer* pool1 = new PoolLayer(PoolType::Maxpool);
    ConvLayer* Conv2 = new ConvLayer(1, 10, 30);
    PoolLayer* pool2 = new PoolLayer(PoolType::Maxpool);
    View* view = new View();
    FullConnection* fc_1 = new FullConnection(120);
    Relu* fc1_relu = new Relu();
    FullConnection* fc_2 = new FullConnection(84);
    Relu* fc2_relu = new Relu();
    FullConnection* fc_3 = new FullConnection(10);
    SoftMax* sm = new SoftMax();


    
    net.AddLayer(first);
    net.AddLayer(Conv1);
    net.AddLayer(pool1);
    net.AddLayer(Conv2);
    net.AddLayer(pool2);
    net.AddLayer(view);
    net.AddLayer(fc_1);
    net.AddLayer(fc1_relu);
    net.AddLayer(fc_2);
    net.AddLayer(fc2_relu);
    net.AddLayer(fc_3);
    net.AddLayer(sm);

    
    net.train(100, 0.01, dataLoader);
    net.test(dataLoader);
    
    return 0;
}
