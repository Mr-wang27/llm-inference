#include <math.h>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"
#include "src/layers/attention/context_attention.h"

template <typename T>
LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(
                                int head_num_,
                                int kv_head_num_,
                                int head_size_,
                                LLaMAAttentionStaticParams attn_params_,
                                cudaStream_t stream_,
                                cublasWrapper* cublas_wrapper_,
                                BaseAllocator* allocator_):
    head_num(head_num_),
    kv_head_num(kv_head_num_),
    head_size(head_size_),
    hidden_units(head_num_ * head_size_),
    q_head_per_kv(head_num_ / kv_head_num_),
    scale(float(1 / sqrt(head_size))),
    attn_static_params(attn_params_),
    stream(stream_),
    allocator(allocator_),
    cublas_wrapper(cublas_wrapper_){}



template <typename T>
void LLaMAContextAttentionLayer<T>::allocForForward(LLaMAAttentionDynParams& params)
{
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>();     // 获取数据类型
    [[unused]] const int qkv_head_num = head_num + 2 * kv_head_num;

    // for qkv linear and bias rope
    // launchAddFusedQKVBiasTransposeAndRoPE的输入和输出
    qkv_buf_wo_pad = new TensorWrapper<T>(Device::GPU, type, {num_tokens, head_num, head_size});
    q_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
    k_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});
    v_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});
    // launchConcatKVCache,把kv添加到kv_cache后面,然后再从kv_cache中launchRepeatKVCache，才得到下面的kv_cache_buf
    k_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});
    v_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});
    // linear.h中启动launchLinearStridedBatchGemm计算q@k, 然后启动launchScaleMaskAndSoftmax，在原地计算
    qk_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len});
    //linear.h中启动launchLinearStridedBatchGemm计算qk@v
    qkv_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
    // launchTransposeOutRemovePadding启动去除padding
    qkv_buf_wo_pad_1 = new TensorWrapper<T>(Device::GPU, type, {num_tokens, head_num, head_size});


    // 分配数据指针
    qkv_buf_wo_pad->data = allocator->Malloc(qkv_buf_wo_pad->data, sizeof(T)* num_tokens * head_num * head_size, false);
    q_buf_w_pad->data = allocator->Malloc(q_buf_w_pad->data, sizeof(T) * batch_size * (head_num + 2 * kv_head_num) * max_q_len * head_size, false);
    k_buf_w_pad->data = (T*)q_buf_w_pad->data + (batch_size * head_num * max_q_len * head_size);
    v_buf_w_pad->data = (T*)k_buf_w_pad->data + (batch_size * kv_head_num * max_q_len * head_size);
    k_cache_buf->data = allocator->Malloc(k_cache_buf->data, 2 * sizeof(T) * batch_size * head_num * max_k_len * head_size, false);
    v_cache_buf->data = (T*)k_cache_buf->data + (batch_size * head_num * max_k_len * head_size);
    qk_buf->data      = allocator->Malloc(qk_buf->data, sizeof(T) * batch_size * head_num * max_q_len * max_k_len, false);
    qkv_buf_w_pad->data = allocator->Malloc(qkv_buf_w_pad->data, sizeof(T) * batch_size * head_num * max_q_len * head_size, false);
    qkv_buf_wo_pad_1->data = allocator->Malloc(qkv_buf_wo_pad_1->data, sizeof(T) * num_tokens * head_num * head_size, false);

}

template <typename T>
void LLaMAContextAttentionLayer<T>::freeBuf(){
    allocator->Free(qkv_buf_wo_pad->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(q_buf_w_pad->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(k_cache_buf->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(qk_buf->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(qkv_buf_w_pad->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(qkv_buf_wo_pad_1->data, false);
}



template <typename T>
void LLaMAContextAttentionLayer<T>::forward(TensorMap& inputs, TensorMap& outputs,
                    LLaMAattentionWeights<T>& weights,
                    LLaMAAttentionDynParams& params,
                    LLaMAAttentionStaticParams& static_params
)
{
    // 为中间变量分配内存
    allocForForward(params);

    // 1. qkv linear,计算
    // 住q_hiddenunits是embedding的维度
    // shape: [num_tokens, q_hiddenunits] ---> [q_hiddenunits, hiddenunits]
    Tensor* attention_input = inputs["attention_input"];    // TensorMap用于保存一系列的Tensor
    launchLinearGemm(attention_input->as<T>(), weights.qkv, qkv_buf_wo_pad, cublas_wrapper, false, true);    // 权重矩阵需要trans
    DeviceSyncAndCheckCudaError();

    // 2. qkv bias and rope and padding
    // shape:[num_tokens, hiddenunits] ---> [batch_size, q(kv)_head_num, max_q_len, head_size]
    // qkv bias在llama中不存在
    Tensor* padding_offset = inputs["padding_offset"];
    Tensor* history_length = inputs["history_length"];
    Tensor* input_length = inputs["input_length"];      // 当前batch中，每个句子的长度，
    launchAddFusedQKVBiasTransposeAndRoPE(q_buf_w_pad, k_buf_w_pad, v_buf_w_pad,
        qkv_buf_wo_pad, weights.qkv, padding_offset->as<int>(),history_length->as<int>(),
        input_length->as<int>(), static_params);


#ifdef PERF
    DeviceSyncAndCheckCudaError();
#else
#endif
#ifdef SAVE_DATA
    save_tensor(q_buf_w_pad ,"q_buf_after_rope.bin", layer_id->as<int>()); //{batch_size, head_num, max_q_len, head_size}
#else
#endif

    // 3. concat past kv cache
    // max_q_len is input length with bs = 1
    // shape :{batch_size, kv_head_num, max_q_len, head_size} ---> {num_layer, batch_size, max_seq_len[cumsum_seq_len:cumsum_seq_len+cur_seq_len], hidden_units_(kv_head_num * head_size) }
    Tensor* layer_id = inputs["layer_id"];  // ON CPU，这个Tensor在CPU上
    Tensor* all_k_cache = outputs["all_k_cache"];
    Tensor* all_v_cache = outputs["all_v_cache"];
    launchConcatKVCache(k_buf_w_pad, v_buf_w_pad, layer_id->as<int>(),input_length->as<int>(), history_length->as<int>(),all_k_cache->as<T>(), all_v_cache->as<T>());
    DeviceSyncAndCheckCudaError();


    // 4. MHA/MQA/GQA part, reduce kv cache size to [num_layers, bs, kv_head_num, max_seq_len, head_size]
    // 4.0 kv repeat/broadcast to adapt batch gemm shape requirement([bs, head_num, seq_len, head_size]) if need
    // shape: [num_layer, bs, kv_head_num, max_seq_len, head_size] ---> [bs, q_head_num, max_k_len, head_size]
    Tensor* context_length = inputs["context_length"];
    launchRepeatKVCache(all_k_cache->as<T>(),all_v_cache->as<T>(),context_length->as<int>(), layer_id->as<int>(), k_cache_buf, v_cache_buf);
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(k_cache_buf, "k_buf_after)repeat.bin", layer_id->as<int>());
#else
#endif
    
    // 4.1 qk gemm
    // shape: [bs, q_head_num, max_q_len, head_size] @ [bs, q_head_num, max_k_len, head_size](N*T)--->[bs, max_q_len, max_k_len]
    launchLinearStridedBatchGemm(q_buf_w_pad, k_buf_w_pad, qk_buf, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    // 4.2 scale + mask + softmax
    Tensor* attention_mask = inputs["attention_mask"];
    launchScaleMaskAndSoftmax(qk_buf, attention_mask->as<T>(), qk_buf, scale);
    DeviceSyncAndCheckCudaError();
    // 4.3 qk*v
    // shape: [bs, head_num, max_q_len, max_k_len] ---> [bs, head_num, max_q_len, head_size]
    launchLinearStridedBatchGemm(qk_buf, v_cache_buf, qkv_buf_w_pad, cublas_wrapper, false, false);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(qkv_buf_w_pad, "qkv_buf_after_bmm.bin", layer_id->as<int>());
#else
#endif
    // 4.4 transpose + reshape (shape: [bs, head_num, max_q_len, head_size]----> [bs, max_q_len, head_num, head_size]----> [num_tokens, hidden_units]) + remove padding
    launchTransposeOutRemovePadding(qkv_buf_w_pad, padding_offset->as<int>(), qkv_buf_wo_pad_1);
    DeviceSyncAndCheckCudaError();
    // 4.5 output linear
    // shape: [num_tokens, hidden_units]---->[num_tokens, hidden_units]
    Tensor* attention_output = outputs["attention_output"];
    launchLinearGemm(qkv_buf_wo_pad_1, weights.output, attention_output->as<T>(), cublas_wrapper, false, true);

#ifdef SAVE_DATA
    save_tensor(attention_output->as<T>(), "out_linear_output.bin", layer_id->as<int>());   // [num_tokens, head_num, head_size]
#else
#endif
    DeviceSyncAndCheckCudaError();
    this->freeBuf();
}

template class LLaMAContextAttentionLayer<float>;
template class LLaMAContextAttentionLayer<half>;
