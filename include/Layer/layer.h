#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include "Onion.h"
#include "Model.h"

std::string LayerName[8] = {
    "RootLayer",
    "StartingLayer",
    "ConvolutionLayer",
    "MaxPoolingLayer",
    "FullConnectionLayer",
    "ViewLayer",
    "SoftmaxLayer",
    "ReluLayer",
};

class Layer
{
public:
	// 网络运行时所需要的各层的数据
	Onion batch_output; 
	Onion batch_input;
	Onion _loss;
	Onion input;
	Onion output;

	// 表示该层操作的次数和所用的时间
	size_t callTimes = 0;
	float COST_TIME = 0;
	 
	//没有什么意义的构造函数，因为抽象层
	Layer();
	//删除所有Onion
	~Layer();
	
	// 默认为CPU
	dataWhere datawhere = dataWhere::CPU;
	LayerType layerType = LayerType::RootLayerENUM;
	ModelType modelType = ModelType::Inference;    

	// 便于继承和调用
	virtual void trainForword(Onion& batch_input) = 0;
	virtual void trainBackword(Onion& loss) = 0;  
	virtual void _forword(Onion& input) = 0; 
	virtual void initMatrix(Layer* lastLayer, dataWhere where) = 0;
	
	size_t batch_size = 0; 
	double lr = 0.001; 

};

#include "inl/Layer.hpp"

#endif