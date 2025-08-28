#pragma once
#include "src/kernels/build_causal_mask.h"
#include "src/kernels/cal_paddingoffset.h"
#include "src/kernels/fused_addresidual_norm.h"
#include "src/kernels/add_residual.h"  
#include "src/kernels/rmsnorm_kernel.h"
#include "src/layers/attention/context_attention.h"
#include "src/layers/ffn/ffn.h"
// #include "src/weights/llama/llama_weights.h"
#include "src/weights/llama/layer_weights.h"
#include "src/utils/tensor.h"

// layer weights is ready at src/utils/model_utils.h
template <typename T>
class LlamaContextDecoder
{
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int inter_size;
    int num_layer;
    int hidden_units;
    float rmsnorm_eps;

    DataType data_type;
    cudaStream_t stream;
    cublasWrapper *cublas_wrapper;
    BaseAllocator *allocator;


    TensorWrapper<T> *attention_mask;   // 整个推理流程中均存在，每个deocder_layer都要用
    TensorWrapper<int> *padding_offset;
    TensorWrapper<int> *cum_seqlens;
    TensorWrapper<T> *decoder_residual;   // 存储residual
    
    LLaMAContextAttentionLayer<T> *ctxAttn; // 
    LLaMAFFNLayer<T> *ffn;


public:
    LlamaContextDecoder(int head_num,
                        int kv_head_num,
                        int head_size,
                        int inter_size,
                        int num_layer,
                        const LLaMAAttentionStaticParams &attn_params,      // 主要是在attention中需要的超参数，用于初始化LLaMAContextAttentionLayer
                        float rmsnorm_eps,
                        cudaStream_t stream,
                        cublasWrapper *cublas_wrapper,
                        BaseAllocator *allocator) : head_num(head_num),
                                                    kv_head_num(kv_head_num),
                                                    head_size(head_size),
                                                    inter_size(inter_size),
                                                    num_layer(num_layer),
                                                    hidden_units(head_num * head_size),
                                                    rmsnorm_eps(rmsnorm_eps),
                                                    data_type(getTensorType<T>()),
                                                    stream(stream),
                                                    cublas_wrapper(cublas_wrapper),
                                                    allocator(allocator)
{
    // 构造layer
    ctxAttn = new LLaMAContextAttentionLayer<T>(head_num, kv_head_num, head_size, attn_params, stream, cublas_wrapper, allocator);
    
    ffn = new LLaMAFFNLayer<T>(head_num, head_size, inter_size, stream, cublas_wrapper, allocator);

}

    ~LlamaContextDecoder();

    void allocForForward(LLaMAAttentionDynParams &dyn_params);
    void freeBuf();
    void forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight<T>*> &layerWeights, TensorMap &output_tensors, LLaMAAttentionDynParams &dyn_params);

};