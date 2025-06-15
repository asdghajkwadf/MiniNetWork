#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Onion.h"
using namespace std;


#ifndef _DATALOADER_H_

#define _DATALOADER_H_


void CopyMem(double* dst, double* src, size_t dst_num, size_t src_num);

// 列出文件夹中所有文件名
vector<std::string>* listDirectories(string& path);

class PicSample
{
    public:

        PicSample(const string path, const int ID, const int class_num, int r, int c, cv::ImreadModes mod);
        ~PicSample();

        int rows = 0; int cols = 0;
        
        double* getData();
        double* getOneBot();
        double getID();
        void readIamge(cv::ImreadModes mod);

        double* getDataPtr();
        void setDataPtr(double* ptr);

        int getChannelNum() const;

        int channel_num = 0;

        void clearData();

        int class_num = 0;

    private:
        void setOneBot(int ID);
        
        double ID = 0;
        double* _data = nullptr;
        double* one_bot = nullptr;
        string path = "";
  
};



struct Batch
{
    int batch_index = 0;
    Onion* data = nullptr;
    Onion* one_bot = nullptr;
    Onion* Label = nullptr; 
    int size = 0;
    bool full = true;
};


class DataLoader
{
public:
	string root_path = "";
	DataLoader(const string& root_path, int batch_size); // batch_size用于规定训练时的批次，可以加快训练榨干GPU
	DataLoader(const DataLoader& obj) = delete; // 禁用拷贝
	~DataLoader();  
	
	// 用户可以继承DataLoader自己读取数据，定义了两个虚函数
	virtual void readfile(unsigned int limit = 0, bool shuffle = true); // 读取每个类别多少个数据(默认全部读取)
	virtual void splitSample(float rate = 0.75);
	
	// 表示第一层输入的数据维度
	int rows = 28; int cols = 28;  
	int sample_channel = 1;  

	// 获取batch的数据
	// getBatch会将需要用到的样本一整个批次转化为一个Onion
	Batch* getBatch();
	int _Batch_Size();

	
	PicSample* getTestSample();
	int getTestSampleNum(); 
	vector<PicSample*>* Sample();
	
	// 初始化Batch
	void initBatch();  
	void clear();
	int class_num = 0;  

private:
	Batch* batch = nullptr;  
	int Batch_size = 0; 

	// 找出用户给出数据集目录中的文件
	void find();  
	
	// 存放各种样本的数据信息
	vector<string>* all_class = nullptr;  
	vector<vector<string>*>* _path = nullptr;  
	vector<PicSample*>* _sample = nullptr;  
	vector<PicSample*>* _TrainSample = nullptr;
	vector<PicSample*>* _TestSample = nullptr;  
	int sample_num = 0; 
	int _one_class_num = 0;
};


#endif