#include "Onion.h"
#include <vector>
#include <random>
#include <cmath>

using namespace std;
#include <cuda_runtime.h>

double rand_num(double min, double max)
{
    
    std::random_device rd;  // 获取随机种子
    std::mt19937 gen(rd()); // 梅森旋转算法生成器

    // 定义均匀分布的实数范围
    std::uniform_real_distribution<double> dis(min, max);

    // 生成随机double数
    double r = dis(gen);
    return r;
}



void matrixMul(Onion& A, Onion& B, Onion& result)
{
    if (A.isGPU || B.isGPU || result.isGPU)
    {
        throw "data not all in GPU";
    }

    dim3 threadsPerBlock(); 
    dim3 blocksPerGrid();
}

Onion::Onion(OnionShape& shape, dataWhere where)
{   
    initOnion(shape, where);
}

void Onion::initOnion(OnionShape& shape, dataWhere where)
{
    this->_shape = shape; 
    this->where = where;
    if (where == dataWhere::GPU)
    {
        isGPU = true;
    }

    size_t size = 1;
    for (auto s : _shape)
    {
        size = size * s;
    }
    _datasize = size;


    if (where == dataWhere::CPU)
    {
        createData_CPU();
    }
    else if (where == dataWhere::GPU)
    {
        createData_GPU();
    }
}

Onion::~Onion()
{
    if (_data != nullptr)
    {
        delete _data;
    }
}

void Onion::CopyData(const Onion& onion)
{
    if (onion.Size() != this->_datasize)
    {
        throw "can t  copy the data";
    }
    memcpy(_data, onion.getdataPtr(), sizeof(double) * onion.Size());
}


void Onion::toCPU()
{

}

void Onion::toGPU()
{

}

double& Onion::operator[](const size_t index)
{
    if (_data != nullptr)
    {
        if (index < _datasize)
        {
            return _data[index];
        }
        else 
        {
            throw "out of range";
        }
    }
}

double Onion::operator[](const size_t index) const
{
    if (_data != nullptr)
    {
        if (index < _datasize)
        {
            return _data[index];
        }
        else 
        {
            throw "out of range";
        }
    }
}

inline double Onion::get(const size_t index) const
{
    if (_data != nullptr)
    {
        if (index < _datasize)
        {
            return _data[index];
        }
        else 
        {
            throw "out of range";
        }
    }
}

inline double Onion::set(const size_t index, double data) const
{
    if (_data != nullptr)
    {
        if (index < _datasize)
        {
            return _data[index];
        }
        else 
        {
            throw "out of range";
        }
    }
}

double* Onion::getdataPtr() const
{
    return _data;
}

void Onion::initdata(double min, double max)
{
    if (isGPU)
    {

    }
    else
    {
        for (size_t i = 0; i < _datasize; ++i)
        {
            _data[i] = rand_num(min, max);
        }
    }
}

size_t Onion::Size() const
{
    return _datasize;
}

void Onion::setAllData(double data)
{
    if (isGPU)
    {

    }
    else
    {
        for (size_t i = 0; i < _datasize; ++i)
        {
            _data[i] = 0;
        }
    }
}

void Onion::createData_CPU()
{
    _data = new double[_datasize]();
}

void Onion::createData_GPU()
{
    cudaMalloc(&_data, _datasize * sizeof(double));
}

void Onion::applyGPUMem(size_t size)
{
    if (_data != nullptr)
    {
        throw "have data, you can create a new";
    }
}
