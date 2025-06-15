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

Timer::Timer(Layer* l)
{
    start = std::chrono::high_resolution_clock::now();
    l->callTimes += 1;
    this->l = l;
}

Timer::~Timer()
{
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    float ms = duration.count() * 1000.0f;
    // std::cout << "Timer took " << ms << "ms" << std::endl;
    l->COST_TIME += ms;
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

Onion::Onion() : _data(nullptr)
{

}

Onion::Onion(vector<int>& shape, dataWhere where) : _shape(shape), where(where), _data(nullptr)
{

    if (where == dataWhere::GPU)
    {
        isGPU = true;
    }

    int size = 1;
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

void Onion::CopyData(Onion* onion)
{
    if (onion->Size() != this->_datasize)
    {
        throw "can t  copy the data";
    }
    memcpy(_data, onion->getdataPtr(), sizeof(double) * onion->Size());
}


void Onion::toCPU()
{

}

void Onion::toGPU()
{

}

double Onion::operator[](int index)
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

inline double Onion::get(const unsigned int index) const
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

inline double Onion::set(const unsigned int index, double data) const
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

double* Onion::getdataPtr()
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
        for (int i = 0; i < _datasize; ++i)
        {
            _data[i] = rand_num(min, max);
        }
    }
}

int Onion::Size()
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
        for (int i = 0; i < _datasize; ++i)
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

void Onion::applyGPUMem(int size)
{
    if (_data != nullptr)
    {
        throw "have data, you can create a new";
    }
}
