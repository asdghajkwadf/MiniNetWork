#ifndef _MODEL_H_
#define _MODEL_H_

enum LayerType : unsigned char 
{
    RootLayerENUM, 
    StartLayerENUM,
    ConvolutionLayerENUM, 
    MaxPoolingLayerENUM,
    FullConnectionLayerENUM,
    ViewLayerENUM,
    SoftmaxLayerENUM,
    ReluLayerENUM,
}; 


enum ModelType : unsigned char
{
    Train,
    Inference,
};

namespace ModelSet
{
    class LayerBase;
    class Start;
    class Conv;
    class MaxPool;
    class View;
    class Relu;
    class Fullconnection;
    class SoftMax;
}

enum PoolType
{
    Maxpool,
    AveragePool,
};


class ModelSet::LayerBase
{
public:
    LayerType layerType = LayerType::RootLayerENUM;
};

class ModelSet::Conv : public LayerBase
{
public:
    size_t in_channel = 0;
    size_t out_channel = 0;
    size_t step = 1;
    Conv(size_t step, size_t in_channel, size_t out_channel, size_t kernel_r = 3, size_t kernel_c = 3)
    : in_channel(in_channel), out_channel(out_channel), step(step)
    {
        layerType = LayerType::ConvolutionLayerENUM;
    }



    // size_t in_rows = 0;
    // size_t in_cols = 0;
    // size_t out_rows = 0;
    // size_t out_cols = 0;

    // size_t _r_times = 0;
    // size_t _c_times = 0;

    // size_t kernel_r = 3;
    // size_t kernel_c = 3;


    // size_t kernel_num = 0;
};

class ModelSet::MaxPool : public LayerBase
{
public:
    MaxPool(size_t pooling_rows = 2, size_t pooling_cols = 2)
    : pooling_rows(pooling_rows), pooling_cols(pooling_cols)
    {
        layerType = LayerType::MaxPoolingLayerENUM;
    }

    size_t channel = 0;

    size_t in_rows = 0;
    size_t in_cols = 0;

    size_t out_rows = 0;
    size_t out_cols = 0;

    size_t _r_times = 0;
    size_t _c_times = 0;

    size_t pooling_rows = 2;
    size_t pooling_cols = 2;
};

class ModelSet::View : public LayerBase
{
public:
    View()
    {
        layerType = LayerType::ViewLayerENUM;
    }

    size_t in_channel = 0;
    size_t in_rows = 0;
    size_t in_cols = 0;

    size_t output_num = 0;
};

class ModelSet::Relu : public LayerBase
{
public:
    Relu()
    {
        layerType = LayerType::ReluLayerENUM;
    }
    double loss_sum = 0;
    size_t input_num = 0;
    size_t rows = 0;
    size_t cols = 0;
    size_t channel = 0;
};

class ModelSet::Fullconnection : public LayerBase
{
public:
    Fullconnection(size_t output_num)
    : output_num(output_num)
    {
        layerType = LayerType::FullConnectionLayerENUM;
    }
    size_t output_num;
    size_t input_num;
};

class ModelSet::SoftMax : public LayerBase
{
public:
    SoftMax()
    {
        layerType = LayerType::SoftmaxLayerENUM;
    }
    size_t input_num = 0;
};

#endif