#ifndef _DATALOADER_H_
#define _DATALOADER_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Onion.h"

void CopyMem(double* dst, double* src, size_t dst_num, size_t src_num);

// 列出文件夹中所有文件名
std::vector<std::string>* listDirectories(std::string& path);

class PicSample
{
    public:

        PicSample(const std::string path, const size_t ID, const size_t class_num, size_t r, size_t c, cv::ImreadModes mod);
        ~PicSample();

        size_t rows = 0; size_t cols = 0;
        
        double* getData();
        double* getOneBot();
        double getID();
        void readIamge(cv::ImreadModes mod);

        double* getDataPtr();
        void setDataPtr(double* ptr);

        size_t getChannelNum() const;

        size_t channel_num = 0;

        void clearData();

        size_t class_num = 0;

    private:
        void setOneBot(size_t ID);
        
        double ID = 0;
        double* _data = nullptr;
        double* one_bot = nullptr;
        std::string path = "";
  
};






class DataLoader
{
public:
	std::string root_path = "";
	DataLoader(const std::string& root_path); // batch_size用于规定训练时的批次，可以加快训练榨干GPU
	DataLoader(const DataLoader& obj) = delete; // 禁用拷贝
	~DataLoader();  
	
	// 用户可以继承DataLoader自己读取数据，定义了两个虚函数
	virtual void readfile(size_t limit = 0, bool shuffle = true); // 读取每个类别多少个数据(默认全部读取)
	virtual void splitSample(float rate = 0.75);
	
	// 表示第一层输入的数据维度
	size_t rows = 28; size_t cols = 28;  
	size_t sample_channel = 1;  

	// 获取batch的数据
	// getBatch会将需要用到的样本一整个批次转化为一个Onion


	
	PicSample* getTestSample();
	size_t getTestSampleNum(); 
	std::vector<PicSample*>* Sample();
	
	// 初始化Batch

	void clear();
	size_t class_num = 0;  

	// 找出用户给出数据集目录中的文件
	void find();  
	
	// 存放各种样本的数据信息
	std::vector<std::string>* all_class = nullptr;  
	std::vector<std::vector<std::string>*>* _path = nullptr;  
	std::vector<PicSample*>* _sample = nullptr;  
	std::vector<PicSample*>* _TrainSample = nullptr;
	std::vector<PicSample*>* _TestSample = nullptr;  
	size_t sample_num = 0; 
	size_t _one_class_num = 0;
};


#endif