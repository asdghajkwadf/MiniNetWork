#ifndef _BATCH_H_
#define _BATCH_H_


#include "Onion.h"



struct Batch
{
    size_t batch_index = 0;
    Onion data;
    Onion one_bot;
    Onion Label; 
    size_t size = 0;
    bool full = true;
};




#endif