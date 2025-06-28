#ifndef _FULLCONNECTIONLAYER_H_
#define _FULLCONNECTIONLAYER_H_


#include "Layer.h"
#include "Onion.h"

class FullconnectionLayer : public Layer
{
public:
    
    FullconnectionLayer(size_t output_num);
    ~FullconnectionLayer();
    
    
    void trainForword(Onion& batch_input) override;
    void trainBackword(Onion& loss) override;
    void _forword(Onion& input) override;
    void initMatrix(Layer* lastLayer);


    void* getWeight();

    size_t input_num = 0;
    size_t output_num = 0;

private:
    size_t in_rows = 0;
    size_t in_cols = 0;

    // CPU
    void _CPUZeroGrad();
    void _CPUupdate();
    void _CPUforword(Onion& batch_input);
    void clac_loss(Onion& batch_output);
    void _CPUclac_gradient(Onion& nextLayerBatchLoss);


    // GPU
    void _GPUforword(Onion& batch_input);
    void _GPUZeroGrad();
    void _GPUclac_gradient(Onion& nextLayerBatchLoss);
    void _GPUupdate();



    void initGradient();
    void initWeight();


    Onion _w;
    Onion _b;

    Onion _w_grad;
    Onion _b_grad;
};

#include "inl/FullconnectionLayer.hpp"

#endif