#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include "Layer.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

Layer::Layer()
{

}

Layer::~Layer()
{
    // if (batch_output != nullptr)
    // {
    //     delete batch_output;
    // }
    // if (batch_input != nullptr)
    // {
    //     delete batch_input;
    // }
    // if (_loss != nullptr)
    // {
    //     delete _loss;
    // }
    // if (input != nullptr)
    // {
    //     delete input;
    // }
    // if (output != nullptr)
    // {
    //     delete output;
    // }

    // 输出一个backword平均损耗的时间
    if (COST_TIME != 0)
    {
        size_t LayerNameIndex = static_cast<size_t>(this->layerType);
        std::cout << LayerName[LayerNameIndex] << ": cost " << COST_TIME / callTimes << " ms" << std::endl;
    }
}


#endif