#ifndef _RELU_H_
#define _RELU_H_

#include "Layer.h"
#include "Onion.h"

class ReluLayer : public Layer
{
public:
    ReluLayer();
    ~ReluLayer();

    size_t input_num = 0;

    size_t rows = 0;
    size_t cols = 0;
    size_t channel = 0;

    void trainForword(Onion& batch_input) override;
    void trainBackword(Onion& loss) override;
    void _forword(Onion& input) override;
    void initMatrix(Layer* lastLayer, dataWhere where);

    void* getWeight();

private:

    void GPUZeroGrad();
    void GPUclac_gradient(Onion& loss);

    void CPUZeroGrad();
    void CPUforword();
    void CPUclac_gradient(Onion& loss);

    void initWeight();
};

#include "inl/ReluLayer.hpp"

#endif