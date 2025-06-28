#ifndef _POOLLAYER_H_
#define _POOLLAYER_H_

#include "Layer.h"
#include "Onion.h"
#include "Model.h"


class MaxPoolLayer : public Layer
{
public:
    MaxPoolLayer(size_t pooling_rows = 2, size_t pooling_cols = 2);
    ~MaxPoolLayer();

    void* getWeight();

    void trainForword(Onion& batch_input) override;
    void trainBackword(Onion& loss) override;
    void _forword(Onion& input) override;
    void initMatrix(Layer* lastLayer);

    size_t channel = 0;

    size_t in_rows = 0;
    size_t in_cols = 0;

    size_t out_rows = 0;
    size_t out_cols = 0;

private:

    Onion max_index;

    void initWeight();

    void _CPUpooling();
    void _GPUpooling();

    // CPU
    void _CPUZeroGrad();
    void clac_loss(Onion& batch_output);
    void _CPUclac_gradient(Onion& nextLayerBatchLoss);


    // GPU
    void _GPUZeroGrad();
    void _GPUclac_gradient(Onion& nextLayerBatchLoss);
    void _GPUupdate();

    size_t _r_times = 0;
    size_t _c_times = 0;

    size_t pooling_rows = 2;
    size_t pooling_cols = 2;

};

#include "inl/PoolLayer.hpp"

#endif