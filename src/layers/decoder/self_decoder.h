#pragma once
#include "src/kernels/fused_decoder_self_attention.h"
#include "src/kernels/fused_addresidual_norm.h"
#include "src/kernels/add_residual.h"
#include "src/kernels/rmsnorm_kernel.h"
#include "src/layers/attention/masked_self_attention.h"
#include "src/layers/ffn/ffn.h"
#include "src/weights/llama/layer_weights.h"        // 与它的不一样
#include "src/utils/tensor.h"

// layer weights is ready at the model_utils.h
template <typename T>
class LlamaSelfDecoder
{
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int inter_size;
    int num_layer;
    int hidden_units;
    float rmsnorm_eps;

    cudaStream_t stream;
    cublasWrapper *cublas_wrapper;
    BaseAllocator *allocator;

    TensorWrapper<T> *decoder_residual;
    LLaMASelfAttentionLayer<T> *selfAttn;
    LLaMAFFNLayer<T> *ffn;
    DataType data_type;

public:
    LlamaSelfDecoder(int head_num,
                     int kv_head_num,
                     int head_size,
                     int inter_size,
                     int num_layer,
                     const LLaMAAttentionStaticParams &attn_params,
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
                                                 stream(stream),
                                                 cublas_wrapper(cublas_wrapper),
                                                 allocator(allocator),
                                                 data_type(getTensorType<T>())
{
    selfAttn = new LLaMASelfAttentionLayer<T>(head_num, kv_head_num, head_size,
                                              attn_params, stream, cublas_wrapper, allocator);

    ffn = new LLaMAFFNLayer<T>(head_num, head_size, inter_size, stream, cublas_wrapper, allocator);

}
    ~LlamaSelfDecoder();

    void allocForForward(LLaMAAttentionDynParams& dyn_params);
    void freeBuf();
    void forward(TensorMap &input_tensors, const std::vector<LlamaLayerWeight<T>*> &layerWeights, TensorMap &output_tensors, LLaMAAttentionDynParams &dyn_params);
};