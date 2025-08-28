#pragma once
#include "src/weights/weight.h"
#include "src/weights/base_weights.h"
#include "src/weights/llama/embedding_weights.h"
#include "src/weights/llama/layer_weights.h"
template <typename T>
struct LlamaWeight : public Weight{
private:
    int hidden_units;
    int inter_size;
    int vocab_size;
    int vocab_size_padded;  // 这个参数是干啥呢？
    int num_layers;
    WeightType weight_type;

public:
    std::vector<LlamaLayerWeight<T>*> llama_layer_weight;   
    LayerNormWeight<T> out_rmsnorm_weight;                  // 最后一个rmsnorm的参数
    EmbeddingWeight<T> post_decoder_embedding_weight;       // 这里的post decoder_embedding_weight就是采样层的最后一个linear层的参数，就是分类头的参数
    EmbeddingWeight<T> pre_decoder_embedding_weight;        // embedding kernel的参数

    LlamaWeight() = default;
    LlamaWeight(
        int head_num,
        int kv_head_num,
        int head_size,
        int inter_size,
        int vocab_size,
        int num_layer,
        bool attn_bias,
        WeightType weight_type
    );

    ~LlamaWeight();
    void loadWeights(std::string weight_path);
    void loadWeightsFormDummy();
};