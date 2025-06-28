#include "PreData.h"
#include "DataLoader.h"

void PicImporve::Normalization(DataLoader* dataLoader)
{
    for (auto s_ptr : *(dataLoader->Sample()))
    {
        double* _data = s_ptr->getDataPtr();
        for (size_t channel = 0; channel < s_ptr->getChannelNum(); ++channel)
        {
            for (size_t r = 0; r < s_ptr->rows; ++r)
            {
                for (size_t c = 0; c < s_ptr->cols; ++c)
                {
                    _data[channel*s_ptr->rows*s_ptr->cols + r*s_ptr->rows + c] =  _data[channel*s_ptr->rows*s_ptr->cols + r*s_ptr->rows + c] / 255;
                }
            }
        }

        // for (size_t r = 0; r < s_ptr->rows; ++r)
        // {
        //     for (size_t c = 0; c < s_ptr->cols; ++c)
        //     {
        //         std::cout << _data[r*s_ptr->cols + c] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
}

void PicImporve::Padding(DataLoader* dataLoader, size_t circles)
{
    dataLoader->rows = dataLoader->rows + 2 * circles;
    dataLoader->cols = dataLoader->cols + 2 * circles;

    for (auto s_ptr : *(dataLoader->Sample()))
    {   
        size_t newRows = s_ptr->rows + 2 * circles;
        size_t newCols = s_ptr->cols + 2 * circles;

        // 分配新矩阵内存
        double* newData = new double[s_ptr->getChannelNum()*newRows*newCols]();
        double* _data = s_ptr->getDataPtr();
        for (size_t channel = 0; channel < s_ptr->getChannelNum(); ++channel)
        {
            for (size_t i = 0; i < s_ptr->rows; ++i) 
            {
                for (size_t j = 0; j < s_ptr->cols; ++j) 
                {
                    newData[channel*newRows*newCols + (i + circles)*newCols + (j + circles)] = _data[channel*s_ptr->rows*s_ptr->cols + i*s_ptr->rows + j];
                }
            }
        }

        s_ptr->setDataPtr(newData); // 自动管理内存

        s_ptr->rows = newRows;
        s_ptr->cols = newCols;

        // for (size_t r = 0; r < s_ptr->rows; ++r)
        // {
        //     for (size_t c = 0; c < s_ptr->cols; ++c)
        //     {
        //         std::cout << newData[r*s_ptr->cols + c] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }   
}