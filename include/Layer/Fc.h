#include "Layer.h"
#include "Onion.h"

#ifndef _FC_H_
#define _FC_H_


class FullConnection : public Layer
{
public:
    
    FullConnection(int output_num);
    ~FullConnection();
    
    
    void trainForword(Onion* batch_input) override;
    void trainBackword(Onion* loss) override; 

    void _forword(Onion* input) override;


    void initMatrix(Layer* lastLayer) override;
    void* getWeight();

    int input_num = 0;
    int output_num = 0;

private:
    int in_rows = 0;
    int in_cols = 0;

    // CPU
    void _CPUZeroGrad();
    void _CPUupdate();
    void _CPUforword(Onion* batch_input);
    void clac_loss(Onion* batch_output);
    void _CPUclac_gradient(Onion* nextLayerBatchLoss);


    // GPU
    void _GPUforword(Onion* batch_input);
    void _GPUZeroGrad();
    void _GPUclac_gradient(Onion* nextLayerBatchLoss);
    void _GPUupdate();



    void initGradient();
    void initWeight();


    Onion* _w = nullptr;
    Onion* _b = nullptr;

    Onion* _w_grad = nullptr;
    Onion* _b_grad = nullptr;
};


#endif