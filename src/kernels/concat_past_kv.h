#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"


// 把kv添加到past_kv后面
// history_length是当前样本（batch_id）在本轮生成之前，已经生成的 token 数量，
// 即该样本的 KV cache 当前已有的 token 总数（即缓存长度）
template <typename T>
void launchConcatKVCache(TensorWrapper<T> *k_src, // form qkv bias and rope
                         TensorWrapper<T> *v_src,
                         TensorWrapper<int> *layer_id,  // layer_offset = layer_id * batch_size * max_seq_len * kv_head_num * head_size
                         TensorWrapper<int> *cur_query_length,  // currrent epoch or locak input leh=gth, [bathc_size]
                         TensorWrapper<int> *history_length,    // [batch_size]: 多轮对话所累积的token长度
                         TensorWrapper<T> *k_dst,
                         TensorWrapper<T> *v_dst);