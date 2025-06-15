#include "Onion.h"
#include "layer.h"

#ifndef _VIEW_H_
#define _VIEW_H_

class View : public Layer
{
    public:

        View();
        ~View();

        int in_channel = 0;
        int in_rows = 0;
        int in_cols = 0;

        int output_num = 0;
        void setChannel(int channel);

        void trainForword(Onion* batch_input) override;
        void trainBackword(Onion* loss) override;

        void _forword(Onion* input) override;

        void initMatrix(Layer* lastLayer) override;
        void* getWeight();
        

    private:
        
        void initWeight();
        void _view(double*** input, double* output);


};



#endif