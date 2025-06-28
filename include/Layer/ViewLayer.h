#ifndef _VIEWLAYER_H_
#define _VIEWLAYER_H_

#include "Onion.h"
#include "Layer.h"

class ViewLayer : public Layer
{
    public:

        ViewLayer();
        ~ViewLayer();

        size_t in_channel = 0;
        size_t in_rows = 0;
        size_t in_cols = 0;

        size_t output_num = 0;
        void setChannel(size_t channel);

        void trainForword(Onion& batch_input) override;
        void trainBackword(Onion& loss) override;
        void _forword(Onion& input) override;
        void initMatrix(Layer* lastLayer);


        void* getWeight();
        

    private:
        
        void initWeight();
};

#include "inl/ViewLayer.hpp"

#endif