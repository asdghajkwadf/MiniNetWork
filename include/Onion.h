#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include "layer.h"

#ifndef ONION_H
#define ONION_H
// #pragma once 


using namespace std;

double rand_num(double min, double max);

class Layer;
class Timer {
public:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float> duration;

    Layer* l = nullptr;
    Timer(Layer* l);
    ~Timer();
};

enum dataWhere
{
    CPU, 
    GPU
};

typedef vector<int> shape;

class Onion 
{    
dataWhere where;
double* _data = nullptr;           
vector<int> _shape;
int _datasize;

public:
    Onion();
    Onion(vector<int>& shape, dataWhere where); 
    ~Onion();     

    bool isGPU = false;

    double operator[](int index);
    void initdata(double min, double max);
    void setAllData(double data);
    double get(const unsigned int index);
    double set(const unsigned int index, double data);

    double* getdataPtr();

    int Size();

    void CopyData(Onion* onion);

    void toGPU();
    void toCPU();               

private:
    void createData_CPU();
    void createData_GPU();

    void applyGPUMem(int size);

};




void matrixMul(Onion A, Onion B, Onion result);










#endif