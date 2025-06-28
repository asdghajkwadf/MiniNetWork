#include <iostream>
#include "DataLoader.h"
#include "Nnw.h"
#include "PreData.h"
#include "Model.h"

int main()
{

    DataLoader* dataLoader = new DataLoader("D:/DATA/NetWork/data");

    dataLoader->readfile(50);

    PicImporve::Normalization(dataLoader);
    PicImporve::Padding(dataLoader, 1);

    NetWork net = NetWork(ModelType::Train);

    net.AddLayer(ModelSet::Conv(1, 1, 10));
    net.AddLayer(ModelSet::MaxPool());
    net.AddLayer(ModelSet::Conv(1, 10, 20));
    net.AddLayer(ModelSet::MaxPool());
    net.AddLayer(ModelSet::View());
    net.AddLayer(ModelSet::Relu());
    net.AddLayer(ModelSet::Fullconnection(120));
    net.AddLayer(ModelSet::Relu());
    net.AddLayer(ModelSet::Fullconnection(10));
    net.AddLayer(ModelSet::SoftMax());

    net.train(100, 0.1, dataLoader, 4);
    net.test(dataLoader);
    
    return 0;
}
