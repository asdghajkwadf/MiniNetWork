#include "Layer.h"
#include "Onion.h"

#ifndef _RELU_H_
#define _RELU_H_



class Relu : public Layer
{
public:
    Relu();
    ~Relu();

    double loss_sum = 0;

    int input_num = 0;

    int rows = 0;
    int cols = 0;
    int channel = 0;

    void trainForword(Onion* batch_input) override;
    void trainBackword(Onion* loss) override;

    void _forword(Onion* input) override;

    void initMatrix(Layer* lastLayer);
    void* getWeight();

private:

    void GPUZeroGrad();
    void GPUclac_gradient(Onion* loss);
    void _GPUupdate();


    void CPUZeroGrad();
    void CPUforword(Onion* loss);
    void CPUclac_gradient(Onion* loss);

    void initWeight();
};









#endif