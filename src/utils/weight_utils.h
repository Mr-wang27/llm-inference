// 包装cudamalloc和cudafree
// 用于分配参数的显存空间，因为参数得显存空间在推理过程中是需要长期存在的，因此直接包装一下cudaFree与cudaMalloc即可，无需维护显存内存池
// 对于中间激活值得显存空间通过allocator维护得显存内存池进行管理
#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include "src/utils/macro.h"

// 不使用封装的GPUMalloc与GPUFree也没有关系，直接使用cudaMalloc或者cudaFree()即可
template <typename T>
void GPUMalloc(T** ptr, size_t size);

template <typename T> 
void GPUFree(T* ptr);

template <typename T_OUT, typename T_FILE, bool Enabled = std::is_same<T_OUT, T_FILE>::value>
struct loadWeightFromBin
{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename);
};  // 模板的泛化形式(原型)