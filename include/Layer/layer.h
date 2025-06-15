#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include "ActiveFunc.h"
#include "Onion.h"

enum LayerType : unsigned char 
{
    RootLayer, 
    StartingLayer,
    ConvolutionLayer, 
    PoolingLayer,
    FullConnectionLayer,
    ViewLayer,
    SoftmaxLayer,
    ReluLayer,
}; 

extern std::string LayerName[8];


// std::vector<std::string> colour {"Blue", "Red", "Orange"};

enum ModelType : unsigned char
{
    Train,
    Inference,
};


class Layer
{
public:

    Onion* batch_output = nullptr;
    Onion* batch_input = nullptr;
    Onion* _loss = nullptr;

    Onion* input = nullptr;
    Onion* output = nullptr;

    int callTimes = 0;
    float COST_TIME = 0;

    Layer();
    ~Layer();

    dataWhere datawhere = dataWhere::CPU;

    // virtual void* getWeight() = 0;

    LayerType layerType = RootLayer;
    ModelType modelType = Inference;

    ActivationFuncType activeFunctype = ActivationFuncType::None;

    // virtual void initTrain(int batch_size) = 0;

    virtual void trainForword(Onion* batch_input) = 0;
    virtual void trainBackword(Onion* loss) = 0;

    virtual void _forword(Onion* input) = 0;

    virtual void initMatrix(Layer* lastLayer) = 0;
    int batch_size = 0;

    double lr = 0.001;

    double ActiveFunc(double input);
    double D_ActiveFunc(double input);

protected:
    // virtual void initWeight() = 0;
    // virtual void clearGrad() = 0;    
};




#endif