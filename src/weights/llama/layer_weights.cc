#include <random>
#include "src/weights/llama/layer_weights.h"
#include "src/utils/macro.h"

template <typename T>
LlamaLayerWeight<T>::LlamaLayerWeight(int head_num,
                                      int kv_head_num,
                                      int head_size,
                                      int inter_size,
                                      WeightType weight_type,
                                      bool attn_bias)
    : head_num(head_num), kv_head_num(kv_head_num), head_size(head_size), hidden_units(head_num * head_size),
    inter_size(inter_size), weight_type(weight_type), attn_bias(attn_bias)
{
    // 初始化weights的结构并分配相应的显存给weights
    
    // attn_norm_weight在生成qkv之前做，所以输入的shape:[num_tokens, hidden_units(head_num*head_size)]
    CHECK(cudaMalloc((void**)&attn_norm_weight.gamma, sizeof(T) * hidden_units));

    // ffn_norm_weight在self-attn输出后，输入到gate/up linear之前，所以输入的shape:[num_tokens, hidden_units]
    CHECK(cudaMalloc((void**)&ffn_norm_weight, sizeof(T) * hidden_units));

    // self_attn_weight, 只需要分配qkv以及output这两个线性层的权重即可
    self_attn_weight.qkv.type = weight_type;
    self_attn_weight.qkv.shape = {(head_num + 2 * kv_head_num) * head_size, hidden_units};  
    CHECK(cudaMalloc((void**)&self_attn_weight.qkv.data, sizeof(T) * (head_num+2*kv_head_num) * head_size * hidden_units));
    self_attn_weight.output.type = weight_type;
    self_attn_weight.output.shape = {hidden_units, hidden_units}; // [bs, head_num, seq_q_len, head_size] --> [num_tokens, head_num*head_size] ---> [num_tokens, hidden_units] @ [hidden_units, hidden_units]
    CHECK(cudaMalloc((void**)&self_attn_weight.output.data, sizeof(T) * hidden_units * hidden_units));

    if(attn_bias){
        // 分配bias的显存，bias为最后一个维度
        CHECK(cudaMalloc((void**)&self_attn_weight.qkv.bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size));
        CHECK(cudaMalloc((void**)&self_attn_weight.output.bias, sizeof(T) * hidden_units));
    }

    // 将gate linear与up linear的权重拼接为一个大的权重，提升性能
    ffn_weight.gateAndup.type = weight_type;    // [num_tokens, hidden_units] @ [inter_size, hidden_units]
    ffn_weight.down.type = weight_type;
    ffn_weight.gateAndup.shape = {2*inter_size, hidden_units};
    ffn_weight.down.shape = {hidden_units, inter_size};

    CHECK(cudaMalloc((void**)&ffn_weight.gateAndup.data, sizeof(T) * 2 * inter_size * hidden_units));
    CHECK(cudaMalloc((void**)&ffn_weight.down.data, sizeof(T) * hidden_units * hidden_units));
}

template <typename T>
void LlamaLayerWeight<T>::loadWeights(std::string weight_path, WeightType weight_type)
{
    // 第三个参数是文件path
    loadWeightFromBin<T, float>::internalFunc(attn_norm_weight.gamma, {(size_t)hidden_units}, weight_path + ".input_layernorm.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_norm_weight.gamma, {(size_t)hidden_units}, weight_path + ".post_attention_layernorm.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.qkv.data, {(size_t)(head_num + 2 * kv_head_num) * head_size, (size_t)hidden_units}, weight_path + ".self_attn.qkv.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.output.data, {(size_t)(hidden_units), (size_t)hidden_units}, weight_path + ".self_attn.o_proj.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_weight.gateAndup.data, {(size_t)(2 * inter_size), (size_t)hidden_units}, weight_path + ".mlp.gate_up_proj.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_weight.down.data, {(size_t)hidden_units, (size_t)inter_size}, weight_path + ".mlp.down_proj.weight.bin");
    if(attn_bias){
        // llama 中的线性层全部没有bias，权重参数使用py脚本转换的时候，也没有，所以这里的文件参数是暂定的
    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.qkv.bias, {(size_t)(head_num + 2 * kv_head_num) * head_size}, weight_path + ".mlp.down_proj.bias.bin");
    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.output.bias, {(size_t)hidden_units}, weight_path + ".mlp.down_proj.bias.bin");
    }else{
        self_attn_weight.qkv.bias = nullptr;
        self_attn_weight.output.bias = nullptr;
        ffn_weight.down.bias = nullptr;
    }
}


template <typename T>
void LlamaLayerWeight<T>::loadWeights()
{
    // load dummy model/weight API, is used to the time when you want test inference performance only
    T* d_dummy_attn_norm_weight;    // q_hidden_units       
    T* d_dummy_ffn_norm_weight;     // q_hidden_units
    T* d_dummy_qkv_weights;         // [num_tokens, q_hidden_units] @ [(head_num + 2*kv_head_num)*head_size, q_hidden_units]
    // T* d_dummy_qkv_bias;            // (head_num + 2*kv_head_num)*head_size
    T* d_dummy_output_weights;      // [num_tokens, q_hidden_units] @ [q_hidden_units, q_hidden_units]
    T* d_dummy_output_bias;         // q_hidden_units
    T* d_dummy_ffn_down;            // [num_tokens, inter_size] @ [q_hidden_units, inter_size]
    T* d_dummy_ffn_down_bias;       // q_hidden_units
    //   这两个是没有bias的
    T* d_dummy_ffn_gate_up;         // [num_tokens, q_hidden_units] @ [2*inter_size, q_hidden_units]

    CHECK(cudaMalloc((void**)&d_dummy_attn_norm_weight, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_norm_weight, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_qkv_weights, sizeof(T) * (head_num + 2*kv_head_num)*head_size * hidden_units));
    // CHECK(cudaMalloc((void**)&d_dummy_qkv_bias, sizeof(T) * (head_num + 2*kv_head_num)*head_size));
    CHECK(cudaMalloc((void**)&d_dummy_output_weights, sizeof(T) * hidden_units * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_output_bias, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down, sizeof(T) * hidden_units * inter_size));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down_bias, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_gate_up, sizeof(T) * 2*inter_size * hidden_units));

    T* h_dummy_attn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_qkv_weights = (T*)malloc(sizeof(T) * (head_num + 2*kv_head_num)*head_size * hidden_units);
    // T* h_dummy_qkv_bias = (T*)malloc(sizeof(T) * (head_num + 2*kv_head_num)*head_size);
    T* h_dummy_output_weights = (T*)malloc(sizeof(T) * hidden_units * hidden_units);
    T* h_dummy_output_bias = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_down = (T*)malloc(sizeof(T) * hidden_units * inter_size);
    T* h_dummy_ffn_down_bias = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_gate_up = (T*)malloc(sizeof(T) * 2*inter_size * hidden_units);

    // 初始化数据
    for(int i = 0; i < hidden_units; i++){
        h_dummy_attn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_output_bias[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_down_bias[i] = (T)(rand() % 100 / (float)100000);
    }

    for(int i = 0; i < (head_num + 2*kv_head_num)*head_size * hidden_units; i++){
        h_dummy_qkv_weights[i] = (T)(rand() % 100 / (float)100000);
    }
    for(int i = 0; i < hidden_units * hidden_units; i++){
        h_dummy_output_weights[i] = (T)(rand() % 100 / (float)100000);
    }
    for(int i = 0; i < hidden_units * inter_size; i++){
        h_dummy_ffn_down[i] = (T)(rand() % 100 / (float)100000);
    }
    for(int i = 0; i < 2*inter_size * hidden_units; i++){
        h_dummy_ffn_gate_up[i] = (T)(rand() % 100 / (float)100000);
    }
    
    // 将数据拷贝到显存上
    CHECK(cudaMemcpy(d_dummy_attn_norm_weight, h_dummy_attn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_norm_weight, h_dummy_ffn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_qkv_weights, h_dummy_qkv_weights, sizeof(T) * (head_num + 2*kv_head_num)*head_size * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_weights, h_dummy_output_weights, sizeof(T) * hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_bias, h_dummy_output_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down, h_dummy_ffn_down, sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down_bias, h_dummy_ffn_down_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_gate_up, h_dummy_ffn_gate_up, sizeof(T) * 2*inter_size * hidden_units, cudaMemcpyHostToDevice));

    // 将数据指针赋值给BaseWeights<T>
    attn_norm_weight.gamma = d_dummy_attn_norm_weight;
    ffn_norm_weight.gamma = d_dummy_ffn_norm_weight;
    self_attn_weight.qkv.data = d_dummy_qkv_weights;
    self_attn_weight.qkv.bias = nullptr;    // 如果没有需要赋值为nullptr,此时会自动不计算bias
    self_attn_weight.output.data = d_dummy_output_weights;
    self_attn_weight.output.bias = d_dummy_output_bias;
    ffn_weight.down.data = d_dummy_ffn_down;
    ffn_weight.down.bias = d_dummy_ffn_down_bias;
    ffn_weight.gateAndup.data = d_dummy_ffn_gate_up;        // 在kernel里卖弄，这里就没有实现bias的加法

}


template <typename T>
void freeWeights(BaseWeight<T>& weights)
{
    cudaFree(weights.data);
    if(weights.bias != nullptr){
        cudaFree(weights.bias);
    }
    weights.data = nullptr;
    weights.bias = nullptr;
}


template <typename T>
LlamaLayerWeight<T>::~LlamaLayerWeight()
{
    // free norm weights ptr, 归一化的参数不能调用freeWeights进行free
    CHECK(cudaFree(attn_norm_weight.gamma));
    CHECK(cudaFree(ffn_norm_weight.gamma));
    // freee other weights, including data and bias
    freeWeights(self_attn_weight.qkv);
    freeWeights(self_attn_weight.output);
    freeWeights(ffn_weight.gateAndup);
    freeWeights(ffn_weight.down);
}


// 在链接的时候需要模板实例化，生成代码
template class LlamaLayerWeight<float>;
template class LlamaLayerWeight<half>;
