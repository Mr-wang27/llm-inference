// 抽象出所有weights的共同特点，作为base_weights,
// 所有weights继承base_weights
#pragma once
#include <vector>
#include <cstdint>
#include <cuda_fp16.h>
enum class WeightType{  // 枚举类，可以避免名字污染，使用枚举类前需要指明作用域：WeightType
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W
};

// 类型萃取Type Traits
template<typename T>
inline WeightType getWeightType()
{
    if(std::is_same<T, float>::value || std::is_same<T, const float>::value){
        return WeightType::FP32_W;
    }
    else if(std::is_same<T, half>::value || std::is_same<T, const half>::value){
        return WeightType::FP16_W;
    }
    else if(std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value){
        return WeightType::INT8_W;
    }
    else{
        return WeightType::UNSUPPORTED_W;
    }
}

template<typename T>
struct BaseWeight{
    std::vector<int> shape;
    WeightType type;
    T* data;
    T* bias;
};

