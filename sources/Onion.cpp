#include "Onion.h"
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include "CudaKernel/cudaFunc.cuh"
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
    if (where != onion.where)
    {
        throw "Copy Onion no in the same devide ! ";
    }
    if (onion.Size() != this->_datasize)
    {
        throw "can t  copy the data, because the datasize no the same";
    }
    if (where == dataWhere::CPU)
    {
        memcpy(_data, onion.getdataPtr(), sizeof(double) * onion.Size());
    }
    else if (where == dataWhere::GPU)
    {
        cudaMemcpy(this->_testGPUdata, onion.getdataPtr(), sizeof(double) * _datasize, cudaMemcpyDeviceToDevice);
    }
}


void Onion::toCPU()
{
    if (where == dataWhere::GPU)
    {
        cudaMemcpy(_getCPUdataPtr(), _getGPUdataPtr(), _datasize * sizeof(double), cudaMemcpyDeviceToHost);
    }
    else 
    {
        throw " funk you ";
    }
    where = dataWhere::CPU;
}

void Onion::toGPU()
{
    if (where == dataWhere::CPU)
    {
        cudaMemcpy(_getGPUdataPtr(), _getCPUdataPtr(), _datasize * sizeof(double), cudaMemcpyHostToDevice);
    }
    else 
    {
        throw " funk you ";
    }
    where = dataWhere::GPU;
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

void Onion::__divide__(const double n) const
{
    if (n == 0)
    {
        throw "fuck you, 你是不是没上过小学，除数不能为零";
    }
    if (where == dataWhere::CPU)
    {
        for (auto i = 0; i < _datasize; ++i)
        {
            _data[i] = _data[i] / n;
        }
    }
    else if (where == dataWhere::GPU)
    {
        Common::Arrays_divide_arrays(_testGPUdata, n, _datasize);
    }
}

// 暂时为开放该接口
void Onion::__add__(const double n) const
{
    if (n == 0)
    {
        throw "fuck you, 加0有什么意义吗";
    }
    if (where == dataWhere::CPU)
    {
        for (auto i = 0; i < _datasize; ++i)
        {
            _data[i] += n;
        }
    }
    else if (where == dataWhere::GPU)
    {
        // Common::Arrays_add_a_number(_testGPUdata, n, _datasize);
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
    if (where == dataWhere::CPU)
    {
        return _data;
    }
    else if (where == dataWhere::GPU)
    {
        return _testGPUdata;
    }
    else 
    {
        throw " ";
    }
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
    if (where == dataWhere::CPU)
    {
        for (size_t i = 0; i < _datasize; ++i)
        {
            _data[i] = 0;
        }
    }
    else if (where == dataWhere::GPU)
    {
        cudaMemset(_testGPUdata, 0, _datasize * sizeof(double));
    }
}

void Onion::createData_CPU()
{
    _data = new double[_datasize]();
}

void Onion::createData_GPU()
{
    cudaMalloc(&_testGPUdata, _datasize * sizeof(double));
}

void Onion::applyGPUMem(size_t size)
{
    if (_data != nullptr)
    {
        throw "have data, you can create a new";
    }
}

double* Onion::_getCPUdataPtr()
{
    if (_data == nullptr)
    {
        createData_CPU();
    }
    return _data;
}

double* Onion::_getGPUdataPtr()
{
    if (_testGPUdata == nullptr)
    {
        createData_GPU();
    }
    return _testGPUdata;
}
