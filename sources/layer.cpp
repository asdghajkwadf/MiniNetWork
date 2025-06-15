#include "layer.h"
#include "ActiveFunc.h"
#include <iostream>
using namespace std;
#include "Onion.h"
#include <vector>
#include <random>
#include <cmath>



extern string LayerName[8] = {
    "RootLayer",
    "StartingLayer",
    "ConvolutionLayer",
    "PoolingLayer",
    "FullConnectionLayer",
    "ViewLayer",
    "SoftmaxLayer",
    "ReluLayer",
};


Layer::Layer()
{

}

Layer::~Layer()
{
    if (batch_output != nullptr)
    {
        delete batch_output;
    }
    if (batch_input != nullptr)
    {
        delete batch_input;
    }
    if (_loss != nullptr)
    {
        delete _loss;
    }
    if (input != nullptr)
    {
        delete input;
    }
    if (output != nullptr)
    {
        delete output;
    }

    // 输出一个backword平均损耗的时间
    if (COST_TIME != 0)
    {
        int LayerNameIndex = static_cast<int>(this->layerType);
        cout << LayerName[LayerNameIndex] << ": cost " << COST_TIME / callTimes << " ms" << endl;
    }
}
















  







  

