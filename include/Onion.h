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

	dataWhere where; // 一个枚举变量，指明这个Onion所存储的数据是在CPU还是在GPU上
	double* _data = nullptr; // 实际的数据指向地址，是一个连续的指针
	vector<int> _shape; // 数据的层数结构（二维或者三维）
	int _datasize; // 数据的长度

public:
    // 没什么比用的构造函数
	Onion(); 
	Onion(vector<int>& shape, dataWhere where); // 构造函数要传入shape和datawhere
	~Onion(); 


	bool isGPU = false; 
	double operator[](int index); // 重载[]操作，方便调试
	void initdata(double min, double max); // 通常用来初始化权重和偏置
	void setAllData(double data); // 通常用来Zerodata
	double get(const unsigned int index) const; //
	double set(const unsigned int index, double data) const; //
	  

	double* getdataPtr(); // 返回指向真实数据的指针
	 

	int Size(); // 返回数据的长度
	void CopyData(Onion* onion); // 将另一个Onion的数据复制到本Onion中
	void toGPU(); //把数据转去GPU
	void toCPU(); //把数据转去CPU
    
private:
	void createData_CPU();
	void createData_GPU();
	void applyGPUMem(int size);
};




void matrixMul(Onion A, Onion B, Onion result);










#endif