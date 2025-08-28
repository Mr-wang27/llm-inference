#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
//[num layers, bs, kv head num, max_seq_len, head size]=>[bs, q head num, max_k_len, head size]
template <typename T>
void launchRepeatKVCache(TensorWrapper<T> *k_cache_src, //[num layers, bs, kv head num, max_seq_len, head size]
                         TensorWrapper<T> *v_cahce_src,
                         TensorWrapper<int> *context_length,    // [batch_size]:当前句子的上下文长度, 就是已缓存的token数量
                         TensorWrapper<int> *layer_id,          // [batch_size]
                         TensorWrapper<T> *k_cache_dst,
                         TensorWrapper<T> *v_cache_dst);