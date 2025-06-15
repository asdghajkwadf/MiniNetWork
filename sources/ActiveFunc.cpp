#include "ActiveFunc.h"


double ActivationFunction::ReLU(double input)
{
    if (input > 0)
    {
        return input;
    }
    else
    {
        return 0;
    }
}
double D_ActivationFunction::DReLU(double input)
{
    if (input > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
