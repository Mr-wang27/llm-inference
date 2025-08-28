#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "src/utils/tensor.h"
#include "src/utils/params.h"


// 这里其实也是支持beam searhc的，只需要在传入这个kernel之前，把batce_size 和 Beam_width维度合并一下即可
template <typename T>
void launchSampling(TensorWrapper<int>* topk_id,          // [bs, k]
                    TensorWrapper<T>* topk_valuse,      // [bs, k]
                    TensorWrapper<int>* seqLen,         // [bs] 每个序列维护的自身的长度，总的token数：历史上下文长度+当前轮次的query长度+当前轮次已经生成的token数
                    TensorWrapper<bool>* is_finished,   // [bs] 每个序列是否已经到达最后的一个token,如果是，代表这个序列不需要进行采样，也不需要进行推理了
                    TensorWrapper<int>* output_id,      // [bs] 输出每个seq在当前轮次的采样结果
                    IntDict& params);                   // 一些相关的参数，用一个数据结构包装起来