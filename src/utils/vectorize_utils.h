// 创建向量化读取与存储的数据类型，用于向量化存取
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T_OUT, typename T_IN>
inline __device__ T_OUT scalar_cast_vec(T_IN val)
{
    return val;
}

template <>
inline __device__ half2 scalar_cast_vec<half2, float>(float val){
    return __float2half2_rn(val);
}

template <>
inline __device__ half2 scalar_cast_vec<half2, half>(half val){
    return __half2half2(val);
    /*
    half2 res;
    res.x = val;
    res.y = val;
    return res;
    */
}

template <>
inline __device__ float4 scalar_cast_vec<float4, float>(float val){
    return make_float4(val, val, val, val);
}

template <>
inline __device__ float2 scalar_cast_vec<float2, float>(float val){
    return make_float2(val, val);
}






// size是静态变量，属于类本身
template<typename T>
struct Vec {
    using Type = T;             // 该向量化数据的数据类型
    static constexpr int size = 0;  // 此处模板设置为0，因为需要依靠模板特化指定        
    // 正对不同的向量化类型，size会有不同
};

// 特化向量化读取的数据类型
// float32类型的向量化读取：float4: 因为一次GPU内存事务(Cache line=128bit) 4*32=128bit
// 因此一个float4刚好128bit
// 其次FP16，一次向量话读取只有2个，只占用32bit。而没有一次占用128bit
// 是因为，对于FP16有专门的CUDA指令可供调用，这些指令一次只能使用两个FP16的数据
template<>
struct Vec<half> {
    using Type = half2;
    static constexpr int size = 2;
};

template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};