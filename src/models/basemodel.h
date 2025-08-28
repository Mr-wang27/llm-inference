#pragma once
#include <string>
#include <functional>
#include "src/utils/tensor.h"
#include "src/models/common_params.h"
#include "src/memory/allocator/base_allocator.h"    // BaseModel利用baseallocator获取到子类的allocator
#include "src/kernels/cublas_utils.h"


// 回调函数，用于打印当前轮次对话的LLM生成内容
// 声明了一个函数指针类型
using CallBack = std::function<void(int index, const char* GenerateContent)>;

class BaseModel{
public:
    std::string model_name;
    // 必须且所有模型子类都共有的4个数据成员
    cudaStream_t stream;
    cublasWrapper* cublas_wrapper;
    BaseAllocator* allocator;
    cudaDeviceProp* cuda_device_prop;   // cuda设备属性
    BaseModel(cudaStream_t stream,
              cublasWrapper* cublas_wrapper,
              BaseAllocator* allocator,
              cudaDeviceProp* cuda_device_prop = nullptr) : 
        stream(stream), cublas_wrapper(cublas_wrapper), allocator(allocator), cuda_device_prop(cuda_device_prop){};
    
    // 3个纯虚函数API,定义相关接口即功能，每个具体模型子类都需要实现
    virtual void loadTokenizer(std::string file) = 0;
    virtual void loadWeights(std::string file) = 0;
    virtual void loadWeightsFromDummy() = 0;

    // 3个纯虚函数API,定义每轮对话的输入、历史信息和回复API,每个具体模型子类需要实现
    // 根据历史信息和当前输入生成当前轮次的prompt
    virtual std::vector<std::string> MakeInput(const std::string &history, int round, const std::string &input) = 0;

    // 根据当前轮次回复更新到history string
    virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) = 0;

    // 回复内容的返回接口
    virtual std::string Response(const std::vector<std::string>& input, CallBack PrintRes) = 0;
    
};