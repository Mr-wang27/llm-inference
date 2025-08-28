/*
    该文件定义了一些错误检测相关的宏：用于检测CUDA、CUBLAS等错误
    #define CHECK(call)                                     这个宏用于检查cuda运行时的错误
    #define CHECK_CUBLAS(val) check((val), #val, __FILE__, __LINE__) 该宏用于检查CUBLAS的运行时错误，或者cuda的运行时错误，具体看被调用的函数时CUBLAS还是CUDA的运行时函数
    #define DeviceSyncAndCheckCudaError() syncAndCheck(__FILE__, __LINE__)  该宏用于同步CUDA设备然后检测CUDA错误

*/
#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
//(RussWong) note: some macro check to assert for helping us find errors, so that we can 
// find the bugs faster
// 这个宏用于检查cuda运行时的错误11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// 定义两个同名函数，通过函数重载实现对不同类型的错误代码进行检查
static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);       // 将宏的错误代码转换成其含义
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)      // 返回cublas的状态信息
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
/*
    result:接受多中类型的错误码
    func: 函数字符串名称
    file: __FILE__
    line: __LINE__
*/
{
    if (result) {
        throw std::runtime_error(std::string("[TM][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

// 该宏用于检查CUBLAS的运行时错误，或者cuda的运行时错误，具体看被调用的函数时CUBLAS还是CUDA的运行时函数2222222222222222222222222222222222222222222222222222222222222222222222222222222
#define CHECK_CUBLAS(val) check((val), #val, __FILE__, __LINE__)        // #val是字符串化操作符，把宏参数转换成字符串字面量


// CUDA变成中常用的同步与错误检查工具函数
// file = __FILE__, line = __LINE__
// 很多CUDA错误不会立即上报，必须在同步点才能检测到错误，因此需要先同步cuda设备，然后才获取错误码
inline void syncAndCheck(const char* const file, int const line)
{
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[TM][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

//该宏用于同步CUDA设备然后检测CUDA错误333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
#define DeviceSyncAndCheckCudaError() syncAndCheck(__FILE__, __LINE__)


// 这两个函数是为了实现一个自定义的断言机制
[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw std::runtime_error(std::string("[oneLLM][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void llmAssert(bool result, const char* const file, int const line, std::string const& info = "")
{
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

#define LLM_CHECK(val) llmAssert(val, __FILE__, __LINE__)
#define LLM_CHECK_WITH_INFO(val, info)                                                                              \
    do {                                                                                                               \
        bool is_valid_val = (val);                                                                                     \
        if (!is_valid_val) {                                                                                           \
            llmAssert(is_valid_val, __FILE__, __LINE__, (info));                                                    \
        }                                                                                                              \
    } while (0)
