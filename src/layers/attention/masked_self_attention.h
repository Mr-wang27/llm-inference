#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h" // 第一个与第四个kernel
#include "src/kernels/attn_softmax_kernel.h"        // 这个kernel并不适用
#include "src/kernels/qkv_bias_and_RoPE.h"  // 第二个kernel
#include "src/kernels/fused_decoder_self_attention.h"   // 第三个kernel
#include "src/utils/tensor.h"
#include "src/models/llama/llama_params.h"  // 静态参数与动态参数
#include "src/utils/macro.h"

// 
template <typename T>
class LLaMASelfAttentionLayer {
private:
    // [num_tokens, hidden_size] --> [num_tokens, qkv_head_num, head_size] ---> [num_tokens, q_head_num, head_size] + 2 *[num_tokens, kv_head_num, head_size]
    // ---> [bs, head_num, seq_q_len, head_size]
    
    // 所有大模型都一样的参数
    const int head_num;
    const int kv_head_num;
    const int head_size;
    const int hidden_size;      // 输入的hidden_size = head_num*head_size
    const int q_hear_per_kv;    // q_head_num / kv_head_num. for GQA与MQA
    float scale;    // sqrt(head_size)

    // 这部分参数是不同大模型可能不同的参数
    LLaMAAttentionStaticParams attn_static_params;
    cudaStream_t stream;
    BaseAllocator* allocator;   // 基类的指针，指向cudaAllocator,利用多态
    // for linear and batchgemm
    cublasWrapper* cublas_wrapper;

    // intermedia buffer
    TensorWrapper<T>* qkv_buf        = nullptr;  // for qkv liner output and rope bias input/output
    TensorWrapper<T>* mha_output     = nullptr;  // mha output, then invoke a linear to attention output

public:
    // 构造函数声明
    LLaMASelfAttentionLayer(int head_num, int kv_head_num, int head_size,
                            LLaMAAttentionStaticParams attn_params,
                            cudaStream_t stream,
                            cublasWrapper* cublas_wrapper,
                            BaseAllocator* allocator);
                            
    // 私有数据成员，只能通过成员函数获取
    LLaMAAttentionStaticParams& GetAttnStaticParams(){
        return attn_static_params;
    }

    void allocForForward(LLaMAAttentionDynParams& params);
    void freeBuf();
    void Forward(TensorMap& input, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params);
};