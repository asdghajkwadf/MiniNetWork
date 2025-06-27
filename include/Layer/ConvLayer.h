#ifndef _CONVLAYER_H_
#define _CONVLAYER_H_

#include "Onion.h"
#include "Layer.h"

class ConvLayer : public Layer
{
public:
    
    ConvLayer(size_t step, size_t in_channel, size_t out_channel, size_t kernel_r = 3, size_t kernel_c = 3);
    ~ConvLayer();

    void* getWeight();
    void setKernelSize(size_t r = 3, size_t c = 3);
    void forword(double*** input, double*** output);

    size_t getoutRows() const;
    size_t getoutCols() const;
    
    void trainForword(Onion* batch_input) override;
    void trainBackword(Onion* loss) override;
    void _forword(Onion* input) override;
    void initMatrix(Layer* lastLayer) override;

    size_t in_channel = 0;
    size_t out_channel = 0;
    size_t in_rows = 0;
    size_t in_cols = 0;
    size_t out_rows = 0;
    size_t out_cols = 0;

private:

    size_t _r_times = 0;
    size_t _c_times = 0;

    size_t kernel_r = 3;
    size_t kernel_c = 3;
    size_t step = 1;

    size_t kernel_num = 0;

    void initWeight();
    void initGradient();
    
    void _CPUforword(Onion* batch_input);
    void _GPUforword(Onion* batch_input);


    void _CPUZeroGrad();
    void _CPUclac_gradient(Onion* loss);
    void _CPUupdate();

    void _GPUZeroGrad();
    void _GPUclac_gradient(Onion* loss);
    void _GPUupdate();

    Onion* _w_grad = nullptr;
    Onion* _b_grad = nullptr;

    Onion* _w = nullptr;
    Onion* _b = nullptr;

};

#include "inl/ConvLayer.hpp"

#endif