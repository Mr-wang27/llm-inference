#pragma once
#include "src/weights/base_weights.h"

template <typename T>
struct LLaMAFFNWeights{
    BaseWeight<T> gate; // 独立的gate线性层参数
    BaseWeight<T> up;   // 独立的up线性层参数
    BaseWeight<T> down;
    BaseWeight<T> gateAndup;    // 将gate和up的线性层参数合并
};