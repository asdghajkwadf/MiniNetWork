#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "DataLoader.h"
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <windows.h>
#include "Onion.h"


std::vector<std::string>* listDirectories(std::string& path) 
{
    std::vector<std::string>* directories = new std::vector<std::string>();

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

    // memcpy(dst + it*batch->data.Size()*sizeof(double), src, batch->data.Size()*sizeof(double));    
}





PicSample::PicSample(const std::string path, const size_t ID, const size_t class_num, size_t r, size_t c, cv::ImreadModes mod)
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

size_t PicSample::getChannelNum() const
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

void PicSample::setOneBot(size_t ID)
{
    one_bot = new double[this->class_num]();
    for (size_t o = 0; o < this->class_num; ++o)
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

    for (size_t row = 0; row < img.rows; ++row) 
    {
        const uchar* rowPtr = img.ptr<uchar>(row);
        for (size_t col = 0; col < img.cols; ++col) 
        {
            double pixel = static_cast<double>(rowPtr[col]);
            _data[row*img.rows + col] = pixel;
        }
    }
}











DataLoader::DataLoader(const std::string& root_path)
{
    this->root_path = root_path;

    find();
}

DataLoader::~DataLoader()
{
    clear();
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

std::vector<PicSample*>* DataLoader::Sample()
{
    return _sample;
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

size_t DataLoader::getTestSampleNum()
{
    return _TestSample->size();
}


void DataLoader::find()
{
    
    all_class = listDirectories(root_path);

    _path = new std::vector<std::vector<std::string>*>(static_cast<size_t>(all_class->size()));
    this->class_num = static_cast<size_t>(_path->size());
    // 找出目录下所有文件
    for (auto i = 0; i < all_class->size(); ++i)
    {
        std::string Path = this->root_path + "/" + all_class->at(i);
        std::vector<std::string>* one_path = listDirectories(Path);
        (*_path)[i] = one_path;
        this->sample_num = this->sample_num + static_cast<size_t>(one_path->size());
    }
}

void DataLoader::readfile(size_t limit, bool shuffle)
{
    this->sample_num = limit * this->class_num;
    if (limit == 0)
    {
        _sample = new std::vector<PicSample*>(this->sample_num);
        size_t count = 0;
        for (size_t c = 0; c < _path->size(); ++c)
        {
            for (std::vector<std::string>::iterator it = ((*_path)[c])->begin(); it != ((*_path)[c])->end(); ++it)
            {
                std::string path = this->root_path + "/" + all_class->at(c) + "/" + *it;
                (*_sample)[count] = new PicSample(path, c, static_cast<size_t>(_path->size()), rows, cols, cv::IMREAD_GRAYSCALE);
            }
        }
    }
    else
    {
        _sample = new std::vector<PicSample*>(this->sample_num);
        size_t count = 0;
        for (auto c = 0; c < _path->size(); ++c)
        {
            if (limit > _path->at(c)->size())
            {
                throw "读取数量超过样本数量";
            }
            for (size_t i = 0; i < limit; ++i)
            {
                std::string path = this->root_path + "/" + all_class->at(c) + "/" + (*((*_path)[c]))[i];
                (*_sample)[count] = new PicSample(path, c, static_cast<size_t>(_path->size()), rows, cols, cv::IMREAD_GRAYSCALE);
                count = count + 1;
            }
        }
    }

    if (shuffle)
    {
        std::random_device rd;
        std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
        
        // 打乱 std::vector
        std::shuffle(_sample->begin(), _sample->end(), gen);
    }

    splitSample();
}

void DataLoader::splitSample(float rate)
{
    size_t TrainNum = _sample->size() * rate;
    // size_t TestNum = _sample->size() * (1 - rate);

    if (_TrainSample == nullptr && _TestSample == nullptr)
    {
        _TrainSample = new std::vector<PicSample*>(); 
        _TestSample = new std::vector<PicSample*>(); 
    }
    else 
    {
        _TrainSample->clear(); 
        _TestSample->clear();
    }

    for (size_t s = 0; s < _sample->size(); ++s)
    {
        if (s < TrainNum)
        {
            // std::std::cout << _sample->at(s) << endl;
            _TrainSample->push_back(_sample->at(s));
        }
        else
        {
            _TestSample->push_back(_sample->at(s));
        }
    }
}