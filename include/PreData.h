#include "DataLoader.h"
#include "Onion.h"

#ifndef _PREDATA_H_

#define _PREDATA_H_


namespace PicImporve
{
    void Normalization(DataLoader* dataLoader);
    void Normalization(Onion* o);

    void Padding(DataLoader* dataLoader, size_t circles);
    void Padding(Onion* dataLoader, size_t circles);
};

#endif