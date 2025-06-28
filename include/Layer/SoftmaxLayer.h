#ifndef _SOFTMAXLAYER_H_
#define _SOFTMAXLAYER_H_

#include "Onion.h"
#include "Layer.h"

typedef struct result
{

    size_t ID;
    double confiden;

} result;

class SoftmaxLayer : public Layer
{
public:

    SoftmaxLayer();
    ~SoftmaxLayer();

    double loss_sum = 0;

    size_t input_num = 0;

    void trainForword(Onion& batch_input) override;
    void trainBackword(Onion& loss) override;
    void _forword(Onion& input) override;
    void initMatrix(Layer* lastLayer);

    result getResult();


    void* getWeight();

private:

    size_t ID = 0;
    double confiden = 0;

    void GPUZeroGrad();
    void GPUclac_loss(Onion& Label);
    void _GPUupdate();


    void CPUZeroGrad();
    void CPUforword(Onion& batch_input);
    void CPUclac_loss(Onion& Label);

    size_t oneBot_num = 0;
    void initWeight();
};

#include "inl/SoftmaxLayer.hpp"

#endif