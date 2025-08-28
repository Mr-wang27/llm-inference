#include <iostream>
#include "src/weights/llama/llama_weights.h"

template <typename T>
LlamaWeight<T>::LlamaWeight(
    int head_num,
    int kv_head_num,
    int head_size,
    int inter_size,
    int vocab_size,
    int num_layer,
    bool attn_bias,
    WeightType weight_type
) : hidden_units(head_num * head_size),
    inter_size(inter_size),
    vocab_size(vocab_size),
    vocab_size_padded(vocab_size),
    num_layers(num_layer),
    weight_type(weight_type)
{
    // 分配内存空间
    llama_layer_weight.reserve(num_layer);
    for(int i = 0; i < num_layer; i++){
        llama_layer_weight.push_back(new LlamaLayerWeight<T>(head_num,
                                                             kv_head_num,
                                                             head_size,
                                                             inter_size,
                                                             weight_type,
                                                             attn_bias));
    }
    GPUMalloc<T>(&out_rmsnorm_weight.gamma, sizeof(T) * hidden_units);
    GPUMalloc<T>(&post_decoder_embedding_weight.data, sizeof(T) * vocab_size * hidden_units);
    GPUMalloc<T>(&pre_decoder_embedding_weight.data, sizeof(T) * vocab_size * hidden_units);
    post_decoder_embedding_weight.shape = {vocab_size , hidden_units};
    pre_decoder_embedding_weight.shape = {vocab_size , hidden_units};
    post_decoder_embedding_weight.type = weight_type;
    pre_decoder_embedding_weight.type = weight_type;
}


template <typename T>
void LlamaWeight<T>::loadWeights(std::string weight_path)
{
    // 注：后面的参数是看前面保存的的bin参数文件名
    loadWeightFromBin<T, float>::internalFunc(out_rmsnorm_weight.gamma, {(size_t)hidden_units}, weight_path + "model.norm.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(post_decoder_embedding_weight.data, {(size_t)vocab_size, (size_t)hidden_units}, weight_path + "lm_head.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(pre_decoder_embedding_weight.data, {(size_t)vocab_size, (size_t)hidden_units}, weight_path + "model.embed_tokens.weight.bin");
    for(int layer = 0; layer < num_layers; layer++){
        // 调用layer_weight中的参数加载函数加载参数
        llama_layer_weight[layer]->loadWeights(weight_path + "model.layers." + std::to_string(layer), weight_type);
    }

}


template <typename T>
void LlamaWeight<T>::loadWeightsFormDummy()
{
    T* d_dummy_out_rmsnorm_weight_gamma;
    T* d_dummy_post_decoder_embedding_weight;
    T* d_dummy_pre_decoder_embedding_weight;
    GPUMalloc<T>(&d_dummy_out_rmsnorm_weight_gamma, sizeof(T) * hidden_units);
    GPUMalloc<T>(&d_dummy_post_decoder_embedding_weight, sizeof(T) * vocab_size * hidden_units);
    GPUMalloc<T>(&d_dummy_pre_decoder_embedding_weight, sizeof(T) * vocab_size * hidden_units);
    T* h_dummy_out_rmsnorm_weight_gamma = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_post_decoder_embedding_weight = (T*)malloc(sizeof(T) * vocab_size * hidden_units);
    T* h_dummy_pre_decoder_embedding_weight = (T*)malloc(sizeof(T) * vocab_size * hidden_units);

    for(int i = 0; i < hidden_units; i++){
        h_dummy_out_rmsnorm_weight_gamma[i] = (T)1.0f;
    }
    for(int i = 0; i < vocab_size * hidden_units; i++){
        h_dummy_post_decoder_embedding_weight[i] = (T)1.0f;
        h_dummy_pre_decoder_embedding_weight[i] = (T)1.0f;
    }

    cudaMemcpy(d_dummy_out_rmsnorm_weight_gamma, h_dummy_out_rmsnorm_weight_gamma, sizeof(T) * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dummy_post_decoder_embedding_weight, h_dummy_post_decoder_embedding_weight, sizeof(T) * vocab_size * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dummy_pre_decoder_embedding_weight, h_dummy_pre_decoder_embedding_weight, sizeof(T) * vocab_size * hidden_units, cudaMemcpyHostToDevice);

    out_rmsnorm_weight.gamma = d_dummy_out_rmsnorm_weight_gamma;
    post_decoder_embedding_weight.data = d_dummy_post_decoder_embedding_weight;
    pre_decoder_embedding_weight.data = d_dummy_pre_decoder_embedding_weight;

    for(int layer = 0; layer < num_layers; layer++){
        llama_layer_weight[layer]->loadWeights();
    }
}

template <typename T>
LlamaWeight<T>::~LlamaWeight()
{
    cudaFree(out_rmsnorm_weight.gamma);
    cudaFree(post_decoder_embedding_weight.data);
    cudaFree(pre_decoder_embedding_weight.data);
    
    for(auto& p : llama_layer_weight){
        delete p;
    }
}

template struct LlamaWeight<float>;
template struct LlamaWeight<half>;
