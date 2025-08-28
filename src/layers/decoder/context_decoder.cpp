#include <iostream>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"
#include "src/layers/decoder/context_decoder.h"

// in llama, all linear dont have bias

template <typename T>
void LlamaContextDecoder<T>::allocForForward(LLaMAAttentionDynParams& params)
{
    int num_tokens = params.num_tokens;
    int batch_size = params.batch_size;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;

    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();

    decoder_residual = new TensorWrapper<T>(Device::GPU, type, {num_tokens, hidden_units});
    attention_mask = new TensorWrapper<T>(Device::GPU, type_int, {batch_size, max_q_len, max_k_len});
    padding_offset = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, max_q_len});
    cum_seqlens = new TensorWrapper<int>(Device::CPU, type_int, {batch_size+1});
    decoder_residual->data = allocator->Malloc(decoder_residual->data, sizeof(T) * num_tokens * hidden_units, false);
    attention_mask->data = allocator->Malloc(attention_mask->data, sizeof(T) * batch_size * max_q_len * max_k_len, false);
    padding_offset->data = allocator->Malloc(padding_offset->data, sizeof(int) * batch_size * max_q_len, false);
    cum_seqlens->data = allocator->Malloc(cum_seqlens->data, sizeof(int) * (batch_size + 1), false);

}


template<typename T>
void LlamaContextDecoder<T>::freeBuf()
{
    allocator->Free(attention_mask->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(padding_offset->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(cum_seqlens->data, false);
    DeviceSyncAndCheckCudaError();
    // 为什么不free掉decoder_residual->data?
    // 应该是可以free的，
    // allocator->Free(decoder_residual->data);
    // DeviceSyncAndCheckCudaError();
}


template <typename T>
void LlamaContextDecoder<T>::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight<T>*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params)
{
    allocForForward(dyn_params);
    Tensor* seq_lens = input_tensors["input_length"];   // batchsize中每个句子的query长度：[batch_size]
    // 1. calculate padding offset
    // shape:
    // seq_lengths: [batch_size]
    // output cum_seqlens: [batch_size + 1], first element is 0
    // output padding_offset: [batch_size*max_q_len]
    launchCalPaddingoffset(padding_offset, cum_seqlens, seq_lens->as<int>());
    DeviceSyncAndCheckCudaError();

    // 2. build causal mask
    Tensor* context_length = input_tensors["context_length"];       // 历史上下文长度： history + cur_genarate
    launchBuildCausalMasks<T>(attention_mask,   // out
                            seq_lens->as<int>(),    // q, input lesn, [bs]
                            context_length->as<int>()); // k, context lens, [bs]
    DeviceSyncAndCheckCudaError();

    // 3. context attn
    Tensor* history_length = input_tensors["history_length"];   // 
    Tensor* decoder_output = output_tensors["decoder_output"];
    Tensor* all_k_cache = output_tensors["all_k_cache"];
    Tensor* all_v_cache = output_tensors["all_v_cache"];
    DataType type_int = getTensorType<int>();
    DataType type = getTensorType<T>();
    Tensor* layer_id = input_tensors["layer_id"];       // 从这里拿到的layer_id 为0
    Tensor* decoder_input = input_tensors["decoder_input"];     // 就是embedding之后的数据:[num_tokens, embedding_dim]
    LLM_CHECK_WITH_INFO(decoder_input->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(history_length->as<int>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

    TensorMap ctx_attn_inputs{
        {"attention_input", decoder_input},
        {"padding_offset", padding_offset},
        {"history_length", history_length},
        {"input_length", seq_lens},
        {"layer_id", layer_id},
        {"context_length", context_length},
        {"attention_mask", attention_mask}
    };
    TensorMap ctx_attn_output{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };


    for(int layer_id = 0; layer_id < num_layer; layer_id++){
        if(layer_id > 0){
            TensorWrapper<int>* layer = new TensorWrapper<int>(Device::CPU, type_int, {1}, &layer_id);
            ctx_attn_inputs.insert("layer_id", layer);  // 更新ctx_attn_inputs[layer_id]
        }   // 当前的layer层

        // 循环num_layer次
        decoder_input = ctx_attn_inputs["attention_input"];
        launchRMSNorm(decoder_input->as<T>(),   // in&out, [num_tokens, q_hidden_units]
                      decoder_residual,         // [num_tokens, q_hidden_units]
                      layerWeights[layer_id]->attn_norm_weight, // rmsnorm weights, [q_hidden_units]
                      rmsnorm_eps);
        DeviceSyncAndCheckCudaError();
        ctxAttn->forward(ctx_attn_inputs, ctx_attn_output, layerWeights[layer_id]->self_attn_weight, dyn_params, ctxAttn->GetAttnStaticParams());

        // 调用launchFusedAddBiasResidualRMSNorm，将attn的输出与之前的residual相加
        launchFusedAddBiasResidualRMSNorm(decoder_residual,     // RMSNorm的input为residual，与之后的数据相加，均为：[num_tokens, q_hidden_units]
                                          decoder_output->as<T>(),      // in&out from attention output, [num_tokens]
                                          layerWeights[layer_id]->self_attn_weight.output,      // bias [q_hidden_units]
                                          layerWeights[layer_id]->ffn_norm_weight.gamma,        // RMSNormal 的参数: [q_hidden_units]
                                          rmsnorm_eps);
        DeviceSyncAndCheckCudaError();
        #ifdef SAVE_DATA
            save_tensor(decoder_output->as<T>(), "ffn_input.bin", layer_id);
        #else
        #endif

        TensorMap ffn_inputs{
            {"ffn_input", decoder_output}
        };
        TensorMap ffn_outputs{
            {"ffn_output", decoder_output}
        };
        dyn_params.is_ctx = true;   // 表示当前是context decoder, 在ffn阶段，prefill与self decoder阶段的分配数据行为不一致，因此，需要区分
        ffn->forward(ffn_inputs, ffn_outputs, layerWeights[layer_id]->ffn_weight, dyn_params);
        #ifdef SAVE_DATA
            save_tensor(decoder_output->as<T>(), "ffn_output.bin", layer_id);
        #else
        #endif
        // RussWong note:这里的residual为上一个launchFusedAddBiasResidualRMSNorm中加了redisual后的hidden states
        launchAddResidual(decoder_residual, decoder_output->as<T>());
        DeviceSyncAndCheckCudaError();
        
        // 更新attention_input为当前decoder层的输出decoder_output，进行下一层的循环
        ctx_attn_inputs.insert("attention_input", decoder_output);
    }
    freeBuf();              // 还是不明白为什么不free   decoder_residual
    DeviceSyncAndCheckCudaError();
}


template <typename T>
LlamaContextDecoder<T>::~LlamaContextDecoder()
{
    // 其他的中间变量会在forward中进行释放，只有deocder_residual没有被释放
    if(decoder_residual->data != nullptr){
        allocator->Free(decoder_residual->data, false);
    }
}


template class LlamaContextDecoder<float>;  // 实例化模板类
template class LlamaContextDecoder<half>;  // 实例化模板类
