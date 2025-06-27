#ifndef _STARTLAYER_H_
#define _STARTLAYER_H_

#include "DataLoader.h"
#include "Layer.h"


class StartLayer : public Layer
{
    DataLoader* dataLoader = nullptr;
public:
    StartLayer(DataLoader* dataLoader);
    ~StartLayer();

    void setBatchSize(size_t batch_size);

    void trainForword(Onion* batch_input) override;
	void trainBackword(Onion* loss) override;  
	void _forword(Onion* input) override; 
	void initMatrix(Layer* lastLayer) override;

    size_t channel = 1;
    size_t out_rows = 0;
    size_t out_cols = 0;

};

#include "inl/StartLayer.hpp"

#endif