#ifndef ONION_H
#define ONION_H

#include <vector>
#include <iostream>


double rand_num(double min, double max);


enum dataWhere
{
    CPU, 
    GPU
};

typedef std::vector<size_t> OnionShape;

class Onion
{

	dataWhere where; // 一个枚举变量，指明这个Onion所存储的数据是在CPU还是在GPU上
	double* _data = nullptr; // 实际的数据指向地址，是一个连续的指针
	std::vector<size_t> _shape; // 数据的层数结构（二维或者三维）
	size_t _datasize; // 数据的长度

public:
    // 没什么比用的构造函数
	Onion(); 
	Onion(std::vector<size_t>& shape, dataWhere where); // 构造函数要传入shape和datawhere
	~Onion(); 


	bool isGPU = false; 
	double operator[](size_t index); // 重载[]操作，方便调试
	void initdata(double min, double max); // 通常用来初始化权重和偏置
	void setAllData(double data); // 通常用来Zerodata
	double get(const size_t index) const; //
	double set(const size_t index, double data) const; //
	  

	double* getdataPtr(); // 返回指向真实数据的指针
	 

	size_t Size(); // 返回数据的长度
	void CopyData(Onion* onion); // 将另一个Onion的数据复制到本Onion中
	void toGPU(); //把数据转去GPU
	void toCPU(); //把数据转去CPU
    
private:
	void createData_CPU();
	void createData_GPU();
	void applyGPUMem(size_t size);
};

void matrixMul(Onion A, Onion B, Onion result);

#endif