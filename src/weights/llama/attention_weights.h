// 计算self attention的weight
#pragma once
#include "src/weights/base_weights.h"
template<typename T>
struct LLaMAattentionWeights {
    BaseWeight<T> q;
    BaseWeight<T> k;
    BaseWeight<T> v;            // 这三个参数暂时还不知道是干啥的, 这三个参数似乎也是生成qkv矩阵的线性层参数，只不过是分开存储的，而下面是将其合并存储的
    BaseWeight<T> qkv;          // 这个参数是从input的数据计算q、k、v三个矩阵的线性层参数
    BaseWeight<T> output;       // 这个是attention最后输出的一个线性层的参数
};