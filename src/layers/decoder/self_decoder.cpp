#include <iostream>
#include "src/utils/macro.h"
#include "src/layers/decoder/self_decoder.h"

template <typename T>
void LlamaSelfDecoder<T>::allocForForward(LLaMAAttentionDynParams &dyn_params)
{
    DataType type = getTensorType<T>();
    int batch_size = dyn_params.batch_size;
    // [batch_size, hidden_units]
    decoder_residual = new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_units});
    decoder_residual->data = allocator->Malloc(decoder_residual->data, sizeof(T) * batch_size * hidden_units, false);
}

template <typename T>
void LlamaSelfDecoder<T>::freeBuf()
{
    if(decoder_residual->data != nullptr){
        allocator->Free(decoder_residual->data, false);
    }
}

template <typename T>
void LlamaSelfDecoder<T>::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight<T>*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams &dyn_params)
{
    allocForForward(dyn_params);

    Tensor* decoder_input = input_tensors["decoder_input"];
    Tensor* decoder_output = output_tensors["decoder_output"];
    Tensor* step = input_tensors["step"];   // 当前self decoder阶段的生成步数
    Tensor* finished = input_tensors["finished"];       // 维护每个序列是否完成生成
    Tensor* all_k_cache = output_tensors["all_k_cache"];
    Tensor* all_v_cache = output_tensors["all_v_cache"];
    Tensor* layer_id = input_tensors["layer_id"];   // 这里传入的是layer_id=0,在for循环中逐步增加
    DataType type_int = getTensorType<int>();

    LLM_CHECK_WITH_INFO(decoder_input->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(decoder_output->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

    TensorMap self_attn_input{
        {"attention_input", decoder_input},
        {"finished", finished},
        {"step", step},
        {"layer_id", layer_id}
    };
    TensorMap self_attn_output{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    for(int layer_id = 0; layer_id < num_layer; layer_id++){
        if(layer_id > 0){
            TensorWrapper<int>* layer = new TensorWrapper<int>(Device::CPU, type_int, {1});
            self_attn_input.insert("layer_id", layer);  // 更新self_attn_input中的layer_id
        }
        decoder_input = self_attn_input["attention_input"];
        launchRMSNorm(decoder_input->as<T>(),   // in&out, [batch_size, q_hidden_units]
                      decoder_residual,
                      layerWeights[layer_id]->attn_norm_weight, // [q_hidden_units]
                      rmsnorm_eps, false);
        DeviceSyncAndCheckCudaError();  
        selfAttn->Forward(self_attn_input, self_attn_output, layerWeights[layer_id]->self_attn_weight, dyn_params);

        launchFusedAddBiasResidualRMSNorm(decoder_residual,         //          [batch_size, hidden_units]
                                          decoder_output->as<T>(),  // in&out   [batch_size, hidden_units]
                                          layerWeights[layer_id]->self_attn_weight.output,  // [hidden_units]
                                          layerWeights[layer_id]->ffn_norm_weight.gamma,          // [hidden_units]
                                          rmsnorm_eps);
        DeviceSyncAndCheckCudaError();  

        TensorMap ffn_inputs{
            {"ffn_inputs", decoder_output}
        };
        TensorMap ffn_outputs{
            {"ffn_outputs", decoder_output}
        };
        ffn->forward(ffn_inputs, ffn_outputs, layerWeights[layer_id]->ffn_weight, dyn_params);

        launchAddResidual(decoder_residual, decoder_output->as<T>(), true);
	    DeviceSyncAndCheckCudaError();
        
        self_attn_input.insert("attention_input", decoder_output);
    }
    // 在context decoder以及self decoder中，都没有释放decoder_residual。我还是觉得应该需要释放
    // 除非是创建一个self/context decoder实例，然后后面每次进行llama推理的时候，都使用该对象进行推理，就可以不不用释放decoder residual,避免重复内存的释放与申请
    // 但是，就算如此，也应该添加一个析构函数，在析构函数中释放掉decoder_residual, 避免内存泄露
}




template <typename T>
LlamaSelfDecoder<T>::~LlamaSelfDecoder()
{
    freeBuf();
}

// 在.cpp中实现模板类或模板函数需要显式实例化
template class LlamaSelfDecoder<float>;
template class LlamaSelfDecoder<half>;
