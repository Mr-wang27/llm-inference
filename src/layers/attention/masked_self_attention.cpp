#include <math.h>
#include "src/utils/debug_utils.h"
#include "src/layers/attention/masked_self_attention.h"


template <typename T>
LLaMASelfAttentionLayer<T>::LLaMASelfAttentionLayer(int head_num,
                                                    int kv_head_num,
                                                    int head_size,
                                                    LLaMAAttentionStaticParams attn_params,
                                                    cudaStream_t stream,
                                                    cublasWrapper* cublas_wrapper,
                                                    BaseAllocator* allocator):
    head_num(head_num),                                         // 代表的是q的head_num
    kv_head_num(kv_head_num),
    head_size(head_size),
    hidden_size(head_num * head_size),                  // 这里如果出现问题，需要回来求改，就觉得这里有问题：应该是(head_um + 2*kv_head_num) * head_size
    q_hear_per_kv(head_num / kv_head_num),
    scale( float(1) / sqrt(head_size)),
    attn_static_params(attn_params),
    stream(stream),
    allocator(allocator),
    cublas_wrapper(cublas_wrapper){}


template <typename T>
void LLaMASelfAttentionLayer<T>::allocForForward(LLaMAAttentionDynParams& params){
    int batch_size = params.batch_size;
    [[unused]] int num_tokens = params.num_tokens;
    [[unused]] int max_q_len = params.max_q_len;
    [[unused]] int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    // 在self decoder阶段，seq_len = 1
    // 分配中间数据内存空间
    qkv_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, qkv_head_num, head_size});
    mha_output = new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_size});

    // 分配数据指针
    qkv_buf->data = allocator->Malloc(qkv_buf->data, sizeof(T) * batch_size * qkv_head_num * head_size, false);
    mha_output->data = allocator->Malloc(mha_output->data, sizeof(T) * batch_size * hidden_size, false);

}


template<typename T>
void LLaMASelfAttentionLayer<T>::freeBuf(){
    allocator->Free(qkv_buf->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(mha_output->data, false);
    DeviceSyncAndCheckCudaError();
}

template <typename T>
void LLaMASelfAttentionLayer<T>::Forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params){

    allocForForward(params);

    // 1. qkv linear
    // shape: [bs, 1, q_hidden_size] * [q_hidden_size, hidden_size] = [ba, 1, hidden_size]
    Tensor* attention_input = inputs["attention_input"];
    launchLinearGemm(attention_input->as<T>(), weights.qkv, qkv_buf, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();

    // 2. bias and rope
    // shape: [ba, 1, hidden_size] ---> [ba, 1, hidden_size]
    Tensor* attention_output = outputs["attention_output"];     // 最终的输出
    // kv cache shape = [bs, kv_head_num, max_seq_len, head_size], layer_id在外部进行处理
    Tensor* key_cache       = outputs["all_k_cache"];
    Tensor* value_cache     = outputs["all_v_cache"];
    Tensor* finished        = inputs["finished"];
    Tensor* step = inputs["step"];  // on cpu, step代表的是当前句子的总长度:历史上下文+当前轮次的query+当前轮次已生成的token数。
                                    // 例如： 历史上下文：20个token, 当前轮次query: 8个token, 以及生成了3个token, 此时step = 20 + 8 + 3 = 31。也代表着这在生成第4个token。但是第四个token还没有完成生成
                                    // 而此时在self decoder阶段，输入是一个token, 生成的q、k均为[batch_size, head_num, seq_len(1), head_size]-->[bs, head_num, head_size].因此要对这个q、k进行旋转编码操作，实际上这个token的index是step-1，即在30的位置上
    Tensor* layer_id = inputs["layer_id"];  // on cpu
    launchRoPE(qkv_buf, step->as<int>(), attn_static_params); // 在这里面会解开数据
    DeviceSyncAndCheckCudaError();
    
    // 3. fused masked mha
    launchDecoderMaskedMHA(qkv_buf, weights.qkv, layer_id->as<int>(), key_cache->as<T>(), value_cache->as<T>(), finished->as<bool>(), step->as<int>(), mha_output, attn_static_params);
    DeviceSyncAndCheckCudaError();

    // 4. attention output linear
    launchLinearGemm(mha_output, weights.output, attention_output->as<T>(), cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();

    #ifdef SAVE_DATA
        save_tensor(mha_output, "self_decoder_outlinear_out.bin", layer_id->as<T>());
    #else
    #endif
    this->freeBuf();
}

template class LLaMASelfAttentionLayer<float>;
template class LLaMASelfAttentionLayer<half>;
