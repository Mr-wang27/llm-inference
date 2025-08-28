// 为每个层分配权重显存
#pragma once
#include "src/weights/llama/norm_weights.h"
#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/utils/weight_utils.h"

template <typename T>
class LlamaLayerWeight{
private:
    // attention_weights
    int head_num;
    int kv_head_num;
    int head_size;
    // ffn_weights
    int hidden_units;       // head_num * head_size
    int inter_size;
    
    WeightType weight_type;
    bool attn_bias; // 指生成qkv的线性层是否有bias
    int bit_size;


public:
    LlamaLayerWeight() = delete;    // 没有无参数的构造函数
    LlamaLayerWeight(int head_num,
                     int kv_head_num,
                     int head_size,
                     int inter_size,
                     WeightType weight_type,
                     bool attn_bias);
    ~LlamaLayerWeight();

    // 该函数用于从HF文件加载权重
    void loadWeights(std::string weight_path, WeightType weight_type);

    void loadWeights(); // 手动设置参数数值，用于测试空架构，或者用于测试性能，而不验证正确性与精度

    // 四个public 权重参数
    LayerNormWeight<T> attn_norm_weight;
    LayerNormWeight<T> ffn_norm_weight;
    LLaMAattentionWeights<T> self_attn_weight;
    LLaMAFFNWeights<T> ffn_weight;
};