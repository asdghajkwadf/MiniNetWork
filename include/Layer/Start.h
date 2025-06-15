#ifndef _START_H_
#define _START_H_

#include "layer.h"
#include "Onion.h"
#include "DataLoader.h"

class StartLayer : public Layer
{
DataLoader* dataloader;

public:
    StartLayer(DataLoader* dataloader);
    ~StartLayer();

    void trainForword(Onion* batch_input) override;
    void trainBackword(Onion* loss) override;

    void _forword(Onion* input) override;

    void setDataLoader(DataLoader& dataloader);

    void initMatrix(Layer* lastLayer) override;

    int rows = 0;
    int cols = 0;
    int channel = 0;


};

#endif