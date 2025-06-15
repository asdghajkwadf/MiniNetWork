#include "layer.h"
#include "Onion.h"
#ifndef _POOL_H_
#define _POOL_H_

enum PoolType
{
    Maxpool,
    AveragePool,
};

class PoolLayer : public Layer
{
public:
    PoolLayer(PoolType type, int pooling_rows = 2, int pooling_cols = 2);
    ~PoolLayer();

    void* getWeight();

    void trainForword(Onion* batch_input) override;
    void trainBackword(Onion* loss) override;

    void _forword(Onion* input) override;

    void initMatrix(Layer* lastLayer) override;

    int channel = 0;

    int in_rows = 0;
    int in_cols = 0;

    int out_rows = 0;
    int out_cols = 0;

private:

    Onion* max_index = nullptr;

    void initWeight();

    void _CPUpooling(Onion* batch_input);
    void _GPUpooling(Onion* batch_input);

    // CPU
    void _CPUZeroGrad();
    void clac_loss(Onion* batch_output);
    void _CPUclac_gradient(Onion* nextLayerBatchLoss);


    // GPU
    void _GPUZeroGrad();
    void _GPUclac_gradient(Onion* nextLayerBatchLoss);
    void _GPUupdate();

    int _r_times = 0;
    int _c_times = 0;

    int pooling_rows = 2;
    int pooling_cols = 2;

};


#endif