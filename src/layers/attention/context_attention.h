#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h"
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/kernels/fused_transpose_and_remv_pad.h"
#include "src/kernels/concat_past_kv.h"
#include "src/kernels/repeat_kv.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_params.h"

template <typename T>
class LLaMAContextAttentionLayer{
private:
    // 这些参数是所有llms共享的
    const int head_num;
    const int kv_head_num;
    const int head_size;
    const int hidden_units;
    const int q_head_per_kv;    // for GQA and MQA
    float scale;    // 根号下head_size

    // this params are only saw in llama and aer unchanged
    LLaMAAttentionStaticParams attn_static_params;  // 旋转位置编码相关的一些参数
    cudaStream_t stream;    // CUDA Stream
    BaseAllocator* allocator;   // 前面创建的统一分配器
    // for linear and batchGEMM
    cublasWrapper* cublas_wrapper;

    // input: [num_tokens, hidden_units]
    TensorWrapper<T> *qkv_buf_wo_pad = nullptr; // [num_tokens, qkv_head_num, head_size]
    TensorWrapper<T> *q_buf_w_pad = nullptr;    // [batch_size, head_num, max_q_len, head_size]
    TensorWrapper<T> *k_buf_w_pad = nullptr;    // [batch_size, head_num, max_q_len, head_size]
    TensorWrapper<T> *v_buf_w_pad = nullptr;    // [batch_size, head_num, max_q_len, head_size]
    // max_k_len表示当前batch中，经过了多轮对话所累积的最大上下文长度
    // 这里的cache是head_num, 而不是kv_head_num, 因为需要经过repeatKV
    TensorWrapper<T> *k_cache_buf = nullptr;    // [layer_nums, batch_szie, head_num, max_k_len, head_size]
    TensorWrapper<T> *v_cache_buf = nullptr;    // [layer_nums, batch_szie, head_num, max_k_len, head_size]
    TensorWrapper<T> *qk_buf = nullptr;         // [batch_size, head_num, max_q_len, max_k_len]
    TensorWrapper<T> *qkv_buf_w_pad = nullptr;  // [batch_size, head_num, max_q_len, head_size]
    TensorWrapper<T> *qkv_buf_wo_pad_1 = nullptr;   // [num_tokens, head_num, head_size]

public:
    LLaMAContextAttentionLayer(int head_num,
                               int kv_head_num,
                               int head_szie,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator);   // 构造函数

    LLaMAAttentionStaticParams& GetAttnStaticParams(){
        return attn_static_params;
    }

    void allocForForward(LLaMAAttentionDynParams& params);
    void freeBuf();
    void forward(TensorMap& inputs, TensorMap& outputs, 
                  LLaMAattentionWeights<T>& weights, 
                  LLaMAAttentionDynParams& params, 
                  LLaMAAttentionStaticParams& static_params);
};