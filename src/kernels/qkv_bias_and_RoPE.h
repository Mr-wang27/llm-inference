#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/models/llama/llama_params.h"
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/vectorize_utils.h"  // 这个文件这里没有用到


// contex的decoder和self的decoder有点不一样，两者分开实现

/*
padding数据：       11111000    11110000    11111110    11111100
padding_offset:     00000    3333    7777777    888888
input_length:       [5, 4, 7, 6]    当前输入的每个句子的长度
*/
//先是context的decoder
// input: qkv_buf : qkv continouns buf when no padding
// shape = [num_tokens, qkv_head_num, head_size], 因为各句子长度不一，所以不用bs * seqlen表示
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size]
// ps: seqlen = max_q_len here


template <typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T>* q_buf, // output:[batch_size, head_num, seq_len, head_size]
                                           TensorWrapper<T>* k_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<T>* v_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<T>* QKV,   // input: [token_num, head_num + 2 * kv_head_num, head_size],
                                           BaseWeight<T>& qkv,  //   qkv_bias:[qkv_head_num, head_size]         对于Linear层，每一个token都是采用相同的这一个weight和bias进行计算
                                           TensorWrapper<int>* padding_offset,  // [batch_size, seq_len]之前计算的padding_offset
                                           TensorWrapper<int>* history_length,  // 历史上下文长度，多轮对话，新的prompt需要连接之前的上下文
                                           TensorWrapper<int>* input_length,    // 要计算padding,这个是输入的所有句子的长度     actual length of each seq
                                           LLaMAAttentionStaticParams& params);




template<typename T>
void launchRoPE(TensorWrapper<T>* qkv_buf,      // output: [bs, seq_len, qkv_head_num, head_dim]->[num_tokens, qkv_head_num, head_dim]
                TensorWrapper<int>* step,       // 这里这个是
                LLaMAAttentionStaticParams& static_params);