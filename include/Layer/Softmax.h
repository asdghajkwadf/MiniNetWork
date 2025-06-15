#include "Onion.h"
#include "layer.h"


#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

typedef struct result
{

    int ID;
    double confiden;

} result;


class SoftMax : public Layer
{
public:

    SoftMax();
    ~SoftMax();

    double loss_sum = 0;

    int input_num = 0;

    void trainForword(Onion* batch_input) override;
    void trainBackword(Onion* loss) override;

    void _forword(Onion* input) override;

    result getResult();

    void initMatrix(Layer* lastLayer);
    void* getWeight();

private:

    int ID = 0;
    double confiden = 0;

    void GPUZeroGrad();
    void GPUclac_loss(Onion* Label);
    void _GPUupdate();


    void CPUZeroGrad();
    void CPUforword(Onion* batch_input);
    void CPUclac_loss(Onion* Label);

    int oneBot_num = 0;
    void initWeight();
};



#endif