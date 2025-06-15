#ifndef _ACTIVEFUNC_H_

#define _ACTIVEFUNC_H_

enum ActivationFuncType
{
    None,
    ReLU,
};


namespace ActivationFunction
{
    double ReLU(double input);
};



namespace D_ActivationFunction
{
    double DReLU(double input);
};

#endif 