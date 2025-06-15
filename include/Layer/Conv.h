#include "Onion.h"
#include "layer.h"







class ConvLayer : public Layer
{
public:
    
    ConvLayer(int step, int in_channel, int out_channel, int kernel_r = 3, int kernel_c = 3);
    ~ConvLayer();

    void* getWeight();
    void setKernelSize(int r = 3, int c = 3);
    void forword(double*** input, double*** output);

    void initMatrix(Layer* lastLayer) override;  //设置输入矩阵的长宽和配置后续变量

    int getoutRows() const;
    int getoutCols() const;
    
    void trainForword(Onion* batch_input) override;
    void trainBackword(Onion* loss) override;

    void _forword(Onion* input) override;

    int in_channel = 0;
    int out_channel = 0;
    int in_rows = 0;
    int in_cols = 0;
    int out_rows = 0;
    int out_cols = 0;


private:

    int _r_times = 0;
    int _c_times = 0;

    int kernel_r = 3;
    int kernel_c = 3;
    int step = 1;

    int kernel_num = 0;

    void initWeight();
    void initGradient();
    
    void _CPUforword(Onion* batch_input);
    void _GPUforword(Onion* batch_input);


    void _CPUZeroGrad();
    void _CPUclac_gradient(Onion* loss);
    void _CPUupdate();

    void _GPUZeroGrad();
    void _GPUclac_gradient(Onion* loss);
    void _GPUupdate();

    Onion* _w_grad = nullptr;
    Onion* _b_grad = nullptr;

    Onion* _w = nullptr;
    Onion* _b = nullptr;

};