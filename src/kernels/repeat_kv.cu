/*
    特别注意一下这里的max_k_len
    这个max_k_len代表什么含义？
*/

#include "src/kernels/repeat_kv.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <iostream>
// if MQA or GQA, we should use this transpose to broadcast kv head num to q head num
//[num layers, bs, kv head num, max_seq_len, head size]=>[bs, q head num, max_k_len, head size]
// context_length.shape=[bs]
// 在分配的时候，会统一分配一个大的缓存空间用于kv cache
// 然后，这个max_seq_len就是模型支持的最大序列长度，主要就是在这里分配了大量的缓存，因为大部分句子不会到这个长度
// 然后在缓存的写入的时候，就只写入有效的token数，后面的不写入即认为是padding token
// 然后在这里reapeat的时候，因为是进行一轮的批对话，上一轮的批对话的kv cache就缓存在kv  cache中
// 上一轮的批对话所缓存的kv cache的形状是[bs, kv head num, max_seq_len, head size]
// 但是，这里面有效的形状以及用于计算的shape是：[bs, kv_head_num, max_q_len(history_len+cur_q_len), head_size]
// 所以在这一轮的对话时。因为同样需要对批对话中的不同序列长度进行padding,padding到一个当前batch中prompt的最长长度
// 然后进行计算,在repeat之前，会先及将其进行concat到pastKVCache上，此时就是把这个数据写入到[num layers, bs, kv head num, max_seq_len, head size]的对应位置上
// 此时max_k_len就是这一轮的批对话中的最大的query长度加上之前的



// 在写入kv cache的时候，只会保存那些真实的token数量
// 不会保存padding的token数据
template <typename T>
__global__ void repeat_value_cache(T *v_dst,
                                   const T *v_src,
                                   const size_t layer_offset,
                                   const int head_num,  // 这里指的是q_head_num
                                   const int q_head_per_kv, // 每个kv对应多少个q_head
                                   const int head_size,
                                   const int *context_length,   // 当前上下文长度，有效的token数： 真实历史长度+历史生成token数+当前query长度
                                   const int max_k_len,     // 转成max_k_len这个维度，然后后面用来做矩阵乘法
                                   const int max_seq_len)
{
    /*
        dim3 block(head_size);
        dim3 grid((max_k_len * head_size + head_size -1) / head_size, batch_size, head_num)
    */
    const int batch_id = blockIdx.y;    // y维度处理每个batch维度的数据
    const int head_id = blockIdx.z;     // head_id处理q_head_num维度的数据
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // x维度上的全局线程
    // x维度处理的是max_k_len * head_size个数据
    const auto val_src = v_src + layer_offset;
    const auto val_dst = v_dst;
    
    const auto seq_len = context_length[batch_id];  // 当前已累计的上下文长度，有效的上下文长度是history_length + 当前当前轮真实 token 长度（不含 padding）
                                                    // 当前当前轮真实 token 长度（不含 padding） = max_q_len + 一推理的token数                                                 

    // 计算当前线程处理的数据
    const int v_head_size_id = idx % head_size; // head_size维度的位置
    const int v_seq_len_id = idx / head_size;   // seq_len维度的数据， 就是token_id

    if(v_seq_len_id  < seq_len){
        // 这里的seq_len就是这个样本已累积的上下文长度，注意的是这是拼接之后的qkv,也就是：真实历史+真实推理生成+该次生成的输入token数
        // 只处理这些token即可。seq_len <= max_seq_len
        const int64_t src_idx = batch_id * (head_num / q_head_per_kv) * max_seq_len * head_size +
                                (head_id / q_head_per_kv) * max_seq_len * head_size +
                                v_seq_len_id * head_size +
                                v_head_size_id;

        const int64_t dst_idx = batch_id * head_num * max_k_len * head_size +
                                head_id * max_k_len * head_size +
                                v_seq_len_id * head_size +
                                v_head_size_id;
        
        val_dst[dst_idx] = val_src[src_idx];
    }
}




template <typename T>
void launchRepeatKVCache(TensorWrapper<T> *k_cache_src, // //{num_layers, batch_size, kv_head_num, max_seq_len, head_size} ：max_seq_len模型支持的最大上下文长度
                         TensorWrapper<T> *v_cache_src, // //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<int> *context_length,    // // [batch_size] : 当前句子的上下文长度, 就是已缓存的token数量
                         TensorWrapper<int> *layer_id,
                         TensorWrapper<T> *k_cache_dst, //{batch_size, head_num, max_k_len, head_size}  max_k_len当前batch中一个seq的已缓存的token数量
                         TensorWrapper<T> *v_cache_dst)
// max_k_len是历史上下文长度加上当前轮次的query中的最长序列长度： max_k_len = history_length + max_q_len
{
    // {num_layers, batch_size, kv_head_num, max_seq_len, head_size}
    int batch_size = context_length->shape[0];
    int kv_head_num = k_cache_src->shape[2];
    int max_seq_len = k_cache_src->shape[3];
    int head_num = k_cache_dst->shape[1];

    int max_k_len = k_cache_dst->shape[2];
    int head_size = k_cache_dst->shape[3];
    int layer = layer_id->getVal();

    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    int q_head_per_kv = head_num / kv_head_num;
    int blockSize = head_size;
    // int blockSize = 128;
    dim3 block(blockSize);
    dim3 grid((max_k_len * head_size + blockSize -1) / blockSize, batch_size, head_num);

    repeat_value_cache<T><<<grid, block>>>(v_cache_dst->data,
                                           v_cache_src->data,
                                           layer_offset,
                                           head_num,
                                           q_head_per_kv,
                                           head_size,
                                           context_length->data,
                                           max_k_len,
                                           max_seq_len);

    repeat_value_cache<T><<<grid, block>>>(k_cache_dst->data,
                                           k_cache_src->data,
                                           layer_offset,
                                           head_num,
                                           q_head_per_kv,
                                           head_size,
                                           context_length->data,
                                           max_k_len,
                                           max_seq_len);
}




template void launchRepeatKVCache(TensorWrapper<float> *k_cache_src, // //{num_layers, batch_size, kv_head_num, max_seq_len, head_size} ：max_seq_len模型支持的最大上下文长度
                         TensorWrapper<float> *v_cache_src, // //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<int> *context_length,    // // [batch_size] : 当前句子的上下文长度, 就是已缓存的token数量
                         TensorWrapper<int> *layer_id,
                         TensorWrapper<float> *k_cache_dst, //{batch_size, head_num, max_k_len, head_size}  max_k_len当前batch中一个seq的已缓存的token数量
                         TensorWrapper<float> *v_cache_dst);


template void launchRepeatKVCache(TensorWrapper<half> *k_cache_src, // //{num_layers, batch_size, kv_head_num, max_seq_len, head_size} ：max_seq_len模型支持的最大上下文长度
                         TensorWrapper<half> *v_cache_src, // //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<int> *context_length,    // // [batch_size] : 当前句子的上下文长度, 就是已缓存的token数量
                         TensorWrapper<int> *layer_id,
                         TensorWrapper<half> *k_cache_dst, //{batch_size, head_num, max_k_len, head_size}  max_k_len当前batch中一个seq的已缓存的token数量
                         TensorWrapper<half> *v_cache_dst);