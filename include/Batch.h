#ifndef _BATCH_H_
#define _BATCH_H_


#include "Onion.h"



struct Batch
{
    size_t batch_index = 0;
    Onion* data = nullptr;
    Onion* one_bot = nullptr;
    Onion* Label = nullptr; 
    size_t size = 0;
    bool full = true;
};




#endif