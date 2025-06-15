#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include "ActiveFunc.h"
#include "Onion.h"

enum LayerType : unsigned char 
{
    RootLayer, 
    StartingLayer,
    ConvolutionLayer, 
    PoolingLayer,
    FullConnectionLayer,
    ViewLayer,
    SoftmaxLayer,
    ReluLayer,
}; 

extern std::string LayerName[8];


// std::vector<std::string> colour {"Blue", "Red", "Orange"};

enum ModelType : unsigned char
{
    Train,
    Inference,
};


class Layer
{
public:
	// 网络运行时所需要的各层的数据
	Onion* batch_output = nullptr; 
	Onion* batch_input = nullptr;
	Onion* _loss = nullptr;
	Onion* input = nullptr;
	Onion* output = nullptr;

	// 表示该层操作的次数和所用的时间
	int callTimes = 0;
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
	virtual void trainForword(Onion* batch_input) = 0;
	virtual void trainBackword(Onion* loss) = 0;  
	virtual void _forword(Onion* input) = 0; 
	virtual void initMatrix(Layer* lastLayer) = 0;
	
	int batch_size = 0; 
	double lr = 0.001; 

};




#endif