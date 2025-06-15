#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
// using namespace cv;
#include "DataLoader.h"
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <windows.h>
#include "Onion.h"


vector<std::string>* listDirectories(string& path) 
{
    vector<string>* directories = new vector<string>();

    // 检查路径是否为空
    if (path.empty()) {
        throw std::invalid_argument("error : path is empty");
    }

    // 构造搜索路径（添加通配符）
    std::string searchPath = path + "\\*";

    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);

    // 检查是否成功打开目录
    if (hFind == INVALID_HANDLE_VALUE) {
        DWORD errorCode = GetLastError();
        throw std::runtime_error("错误：无法访问目录 - " + path + " (错误代码: " + std::to_string(errorCode) + ")");
    }

    // 遍历目录
    do {
        // 跳过 . 和 .. 目录
        if (std::strcmp(findData.cFileName, ".") == 0 || std::strcmp(findData.cFileName, "..") == 0) {
            continue;
        }
        // string _path = path + "/" + findData.cFileName;
        directories->push_back(findData.cFileName);
    } while (FindNextFileA(hFind, &findData));

    // 关闭句柄
    FindClose(hFind);

    // 检查是否为空
    if (directories->empty()) {
        throw std::runtime_error("错误：目录中没有找到文件夹 - " + path);
    }

    return directories;
}

void CopyMem(double* dst, double* src, size_t dst_num, size_t src_num)
{
    if (dst_num != src_num)
    {
        throw "error accur";
    }

    // memcpy(dst + it*batch->data->Size()*sizeof(double), src, batch->data->Size()*sizeof(double));    
}





PicSample::PicSample(const string path, const int ID, const int class_num, int r, int c, cv::ImreadModes mod)
{
    this->path = path;
    this->ID = ID;
    this->rows = r; this->cols = c;
    this->class_num = class_num;
    setOneBot(ID);
    readIamge(mod);
}

PicSample::~PicSample()
{
    clearData();
    delete[] one_bot;
}  

void PicSample::clearData()
{
    if (_data != nullptr)
    {
        delete _data;
    }
    if (one_bot != nullptr)
    {
        delete one_bot;
    }
}

double PicSample::getID()
{
    return this->ID;
}

void PicSample::setDataPtr(double* ptr)
{
    if (_data != nullptr)
    {
        delete _data;
    }
    _data = ptr;

}

double* PicSample::getData()
{
    return _data;
}

int PicSample::getChannelNum() const
{
    return channel_num;
}

double* PicSample::getOneBot()
{
    return one_bot;
}

double* PicSample::getDataPtr()
{
    return _data;
}

void PicSample::setOneBot(int ID)
{
    one_bot = new double[this->class_num]();
    for (int o = 0; o < this->class_num; ++o)
    {
        if (o == ID)
        {
            one_bot[o] = 1;
        }
        else
        {
            one_bot[o] = 0;
        }
    }
}

void PicSample::readIamge(cv::ImreadModes mod)
{

    if (mod == cv::IMREAD_GRAYSCALE)
    {
        channel_num = 1;
    }

    cv::Mat img = cv::imread(path, mod);
    this->rows = img.rows;
    this->cols = img.cols;
    _data = new double[img.rows*img.cols];

    for (int row = 0; row < img.rows; ++row) 
    {
        const uchar* rowPtr = img.ptr<uchar>(row);
        for (int col = 0; col < img.cols; ++col) 
        {
            double pixel = static_cast<double>(rowPtr[col]);
            _data[row*img.rows + col] = pixel;
        }
    }
}











DataLoader::DataLoader(const string& root_path, int batch_size)
{
    this->root_path = root_path;
    this->Batch_size = batch_size;

    find();

    this->Batch_size = batch_size;
}

DataLoader::~DataLoader()
{
    clear();
}

void DataLoader::initBatch()
{
    batch = new Batch();

    vector<int> batchdataShape = {Batch_size, sample_channel, rows, cols};
    vector<int> batchOnebotShape = {Batch_size, class_num};
    vector<int> batchLabelShape = {Batch_size};

    batch->data = new Onion(batchdataShape, dataWhere::CPU);
    batch->one_bot = new Onion(batchOnebotShape, dataWhere::CPU);
    batch->Label = new Onion(batchLabelShape, dataWhere::CPU);

    batch->size = Batch_size;
}

void DataLoader::clear()
{
    delete all_class;

    for (auto p : *(_path))
    {
        delete p;
    }
    delete _path;

    for (auto s : *(_sample))
    {
        if (s != nullptr)
        {
            delete s;
        }
    }
    delete _sample;
}

vector<PicSample*>* DataLoader::Sample()
{
    return _sample;
}

Batch* DataLoader::getBatch()
{

    // 这司马getBatch写得就是一坨屎

    static auto it = 0;
    
    double* batchDataPtr = batch->data->getdataPtr();
    double* batchOneBotPtr = batch->one_bot->getdataPtr();
    double* batchLabelPtr = batch->Label->getdataPtr();

    for (int b = 0; b < Batch_size; ++b)
    {
        batch->batch_index = it / Batch_size;
        if (it < _TrainSample->size())
        {
            batch->full = true;

            double* dataPtr = _TrainSample->at(it)->getData();   
            int index = it*batch->data->Size();
            memcpy(batchDataPtr + b*sample_channel*rows*cols, dataPtr, sample_channel*rows*cols*sizeof(double)); 

            double* OnebotPtr = _TrainSample->at(it)->getOneBot();

            memcpy(batchOneBotPtr + b*class_num, OnebotPtr, class_num*sizeof(double));   

            double labelPtr = _TrainSample->at(it)->getID();
            batchLabelPtr[b] = labelPtr;
            ++it;
        }
        else
        {
            batch->full = false;
            it = 0;
            break;
        }
    }
    return batch;
}

int DataLoader::_Batch_Size()
{
    return Batch_size;
}

PicSample* DataLoader::getTestSample()
{
    static size_t testSampleIndex = 0;
    if (testSampleIndex < _TestSample->size())
    {
        return _TestSample->at(testSampleIndex++);
    }
    else
    {
        return nullptr;
    }
}

int DataLoader::getTestSampleNum()
{
    return _TestSample->size();
}


void DataLoader::find()
{
    
    all_class = listDirectories(root_path);

    _path = new vector<vector<string>*>(static_cast<int>(all_class->size()));
    this->class_num = static_cast<int>(_path->size());
    // 找出目录下所有文件
    for (auto i = 0; i < all_class->size(); ++i)
    {
        string Path = this->root_path + "/" + all_class->at(i);
        vector<string>* one_path = listDirectories(Path);
        (*_path)[i] = one_path;
        this->sample_num = this->sample_num + static_cast<int>(one_path->size());
    }
}

void DataLoader::readfile(unsigned int limit, bool shuffle)
{
    this->sample_num = limit * this->class_num;
    if (limit == 0)
    {
        _sample = new vector<PicSample*>(this->sample_num);
        int count = 0;
        for (int c = 0; c < _path->size(); ++c)
        {
            for (vector<string>::iterator it = ((*_path)[c])->begin(); it != ((*_path)[c])->end(); ++it)
            {
                string path = this->root_path + "/" + all_class->at(c) + "/" + *it;
                (*_sample)[count] = new PicSample(path, c, static_cast<int>(_path->size()), rows, cols, cv::IMREAD_GRAYSCALE);
            }
        }
    }
    else
    {
        _sample = new vector<PicSample*>(this->sample_num);
        int count = 0;
        for (auto c = 0; c < _path->size(); ++c)
        {
            if (limit > _path->at(c)->size())
            {
                throw "读取数量超过样本数量";
            }
            for (unsigned int i = 0; i < limit; ++i)
            {
                string path = this->root_path + "/" + all_class->at(c) + "/" + (*((*_path)[c]))[i];
                (*_sample)[count] = new PicSample(path, c, static_cast<int>(_path->size()), rows, cols, cv::IMREAD_GRAYSCALE);
                count = count + 1;
            }
        }
    }

    if (shuffle)
    {
        std::random_device rd;
        std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
        
        // 打乱 vector
        std::shuffle(_sample->begin(), _sample->end(), gen);
    }

    splitSample();
}

void DataLoader::splitSample(float rate)
{
    int TrainNum = _sample->size() * rate;
    // int TestNum = _sample->size() * (1 - rate);

    if (_TrainSample == nullptr && _TestSample == nullptr)
    {
        _TrainSample = new vector<PicSample*>(); 
        _TestSample = new vector<PicSample*>(); 
    }
    else 
    {
        _TrainSample->clear(); 
        _TestSample->clear();
    }

    for (int s = 0; s < _sample->size(); ++s)
    {
        if (s < TrainNum)
        {
            // std::cout << _sample->at(s) << endl;
            _TrainSample->push_back(_sample->at(s));
        }
        else
        {
            _TestSample->push_back(_sample->at(s));
        }
    }
}