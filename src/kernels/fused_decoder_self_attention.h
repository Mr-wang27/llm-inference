#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/models/llama/llama_params.h" // 这个里面是旋转编码的参数
#include "src/weights/base_weights.h"       // linear bias
#include "src/utils/vectorize_utils.h"

template <typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,  // [bs, (head_num+2*kv_head_num), seq_q_len, head_size]
                            BaseWeight<T>& qkv,      // bias [head_num+2*kv_head_um, head_size]
                            TensorWrapper<int>* layer_id, // [layers]
                            TensorWrapper<T>* k_cache,  // [num_layers, batch_size, kv_head_nums, step, head_size]   这里setp是多轮对话的累积长度，包括当前轮次对话已生成的token，就是step
                            TensorWrapper<T>* v_cache,  // 同上，注意的是这里的max_seq_len其实是预先分配的，并没有实际全部使用，而是以当前的token数量进行填充
                            TensorWrapper<bool>* finished,  // [bs]当前batch生成是否结束
                            TensorWrapper<int>* step,       // [bs]
                            TensorWrapper<T>* mha_output,   // 同q: [bs, head_num, seq_q_len, head_size]
                            LLaMAAttentionStaticParams& static_params);