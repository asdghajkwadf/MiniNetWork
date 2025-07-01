# MiniNetWork
一个纯手搓的神经网络框架，包含卷积，池化，全连接，展平，ReLU激活，SoftMax归一化等操作的框架，框架内部包括了训练时的批次向前传播，反向传播，和测试的向前，在未来更新之后也可用于加载权重数据进行推理

## 项目目录
**data**: 用于存放数据集，每个子文件夹为一个类别(可以有多个类别)
**include**: 存放框架的头文件(.h)
**sources**:  存放(.cpp)
**main.cpp**:是一个分类手写数字识别神经网络的demo
## Layer.h
一个抽象层，包含了各种操作，包括(TrainForword, TrainBackword, InferenceForword)
所有的操作层(卷积层， 池化层， ReLU激活函数层， SoftMax归一层， View展平层)都继承于Layer
实现对象的多态
```C++
class Layer
{
public:
	// 网络运行时所需要的各层的数据
	Onion& batch_output = nullptr; 
	Onion& batch_input = nullptr;
	Onion& _loss = nullptr;
	Onion& input = nullptr;
	Onion& output = nullptr;

	// 表示该层操作的次数和所用的时间
	size_t callTimes = 0;
	float COST_TIME = 0;
	 
	//没有什么意义的构造函数，因为抽象层
	Layer();
	//删除所有Onion
	~Layer();
	
	// 默认为CPU
	dataWhere datawhere = dataWhere::CPU;
	LayerType layerType = RootLayer;
	ModelType modelType = Inference;    

	// 便于继承和调用
	virtual void trainForword(Onion& batch_input) = 0;
	virtual void trainBackword(Onion& loss) = 0;  
	virtual void _forword(Onion& input) = 0; 
	virtual void initMatrix(Layer* lastLayer, dataWhere where); = 0;
	
	size_t batch_size = 0; 
	double lr = 0.001; 
protected:
};
```
## DataLoader数据加载器
用来存放样本，方便后入训练的使用
```C++
// 存放批次的数据
struct Batch{
	size_t batch_index = 0;
	Onion& data = nullptr;
	Onion& one_bot = nullptr;
	Onion& Label = nullptr;
	size_t size = 0;
	bool full = true;
};


class DataLoader
{
public:
	string root_path = "";
	DataLoader(const string& root_path, size_t batch_size); // batch_size用于规定训练时的批次，可以加快训练榨干GPU
	DataLoader(const DataLoader& obj);
	~DataLoader();  
	
	// 用户可以继承DataLoader自己读取数据，定义了两个虚函数
	virtual void readfile(unsigned size_t limit = 0, bool shuffle = true); // 读取每个类别多少个数据(默认全部读取)
	virtual void splitSample(float rate = 0.75);
	
	// 表示第一层输入的数据维度
	size_t rows = 28; size_t cols = 28;  
	size_t sample_channel = 1;  

	// 获取batch的数据
	// getBatch会将需要用到的样本一整个批次转化为一个Onion
	Batch* getBatch();
	size_t _Batch_Size();

	
	PicSample* getTestSample();
	size_t getTestSampleNum(); 
	std::vector<PicSample*>* Sample();
	
	// 初始化Batch
	void initBatch();  
	void clear();
	size_t class_num = 0;  

private:
	Batch* batch = nullptr;  
	size_t Batch_size = 0; 

	// 找出用户给出数据集目录中的文件
	void find();  
	
	// 存放各种样本的数据信息
	std::vector<string>* all_class = nullptr;  
	std::vector<std::vector<string>*>* _path = nullptr;  
	std::vector<PicSample*>* _sample = nullptr;  
	std::vector<PicSample*>* _TrainSample = nullptr;
	std::vector<PicSample*>* _TestSample = nullptr;  
	size_t sample_num = 0; 
	size_t _one_class_num = 0;
};
```
## 运行中各种数据的处理方式
**Onion&*类似pytorch中的Tensor
可以存储权重，偏置，每一层的误差，原始数据输入等，任何参与计算的数据都要转变为Onion的形式
```C++
class Onion
{
	dataWhere where; // 一个枚举变量，指明这个Onion所存储的数据是在CPU还是在GPU上
	double* _data = nullptr; // 实际的数据指向地址，是一个连续的指针
	OnionShape _shape; // 数据的层数结构（二维或者三维）
	size_t _datasize; // 数据的长度
public:
	Onion(); // 没什么比用的构造函数
	Onion(OnionShape& shape, dataWhere where); // 构造函数要传入shape和datawhere
	~Onion(); 
	bool isGPU = false; 
	double operator[](size_t index); // 重载[]操作，方便调试
	void initdata(double min, double max); // 通常用来初始化权重和偏置
	void setAllData(double data); // 通常用来Zerodata
	double get(const unsigned size_t index) const; //
	double set(const unsigned size_t index, double data) const; //
	  
	double* getdataPtr(); // 返回指向真实数据的指针
	 
	size_t Size(); // 返回数据的长度
	void CopyData(Onion& onion); // 将另一个Onion的数据复制到本Onion中
	void toGPU(); //把数据转去GPU
	void toCPU(); //把数据转去CPU
private:
	void createData_CPU();
	void createData_GPU();
	void applyGPUMem(size_t size);
};
```
## demo分析
main.cpp 一个简单的手写数字识别分类网络， 可以达到92%的准确率
```C++
#include <iostream>

// 数据加载器
#include "DataLoader.h"

// 定义网络结构的类
#include "Nnw.h"

//数据预处理
#include "PreData.h"

# 层操作的头文件
#include "Layer.h"
#include "Onion.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "VIewLayer.h"
#include "FullconnectionLayer.h"
#include "Softmax.h"
#include "ReLU.h"
#include "Start.h"  

size_t main(size_t)
{
	// 创建一个DataLoader加载数据集
	DataLoader* dataLoader = new DataLoader("D:/DATA/NetWork/data", 2);
	dataLoader->readfile(100); // 每个类别读取100张图片去训练

	// 也算作数据增强把
	PicImporve::Normalization(dataLoader); // 把样本数据像素值弄到0-1之间
	PicImporve::Padding(dataLoader, 1); // 在图像的四周在填充一圈0

	// 创建一个网络对象
	NetWork net = NetWork(Train);
	
	// 各种层的加载
	StartLayer* first = new StartLayer(dataLoader);
	ConvLayer* Conv1 = new ConvLayer(1, 1, 10);
	PoolLayer* pool1 = new PoolLayer(PoolType::Maxpool);
	ConvLayer* Conv2 = new ConvLayer(1, 10, 30);
	PoolLayer* pool2 = new PoolLayer(PoolType::Maxpool);
	View* view = new View();
	FullConnection* fc_1 = new FullConnection(120);
	Relu* fc1_relu = new Relu();
	FullConnection* fc_2 = new FullConnection(84);
	Relu* fc2_relu = new Relu();
	FullConnection* fc_3 = new FullConnection(10);
	SoftMax* sm = new SoftMax();

	// 将实例化的层加载到网络对象中
	net.AddLayer(first);
	net.AddLayer(Conv1);
	net.AddLayer(pool1);
	net.AddLayer(Conv2);
	net.AddLayer(pool2);
	net.AddLayer(view);
	net.AddLayer(fc_1);
	net.AddLayer(fc1_relu);
	net.AddLayer(fc_2);
	net.AddLayer(fc2_relu);
	net.AddLayer(fc_3);
	net.AddLayer(sm);

	// 开始训练，会一边打印损失(目前是使用交叉熵)
	net.train(300, 0.01, dataLoader);
	net.test(dataLoader);

	// 完美结束
	return 0;
}
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQzNzI2MTA4Niw0Mzc5MDE2MTldfQ==
-->