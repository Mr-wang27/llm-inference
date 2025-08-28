// k/v shape = [bs, kv_head_num, max_q_len, head_size]
// kv cache shape = [num_layers, bs, kv_head_num, max_seq_len, head_size]
// 或者：[bs, kv_head_num, seq_len[history_len:history_len + max_q_len], head_size]

#include "src/kernels/concat_past_kv.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <iostream>

/*
history_length[batch_id] 表示 当前样本（batch_id）在本轮生成之前，已经生成的 token 数量，
即该样本的 KV cache 当前已有的 token 总数（即缓存长度）。
*/
// 传入的k_dst是整个kv cache的第一个元素指向的位置， 所以要偏移到当前layer的k cache处

// 访存密集型kernel 
template <typename T>
__global__ void append_key_cache(T *k_dst,  // [num_layers, bs, kv_head_num, max_q_len, head_size]
                                 const size_t layer_offset,
                                 const T *k_src,    // [bs, kv_head_num, max_q_len, head_size]
                                 const int kv_head_num,
                                 const int head_size,
                                 const int *cur_query_length,   // [bs] 用于避免越界
                                 const int *history_length, // [bs]
                                 const int max_q_len,
                                 const int max_seq_len) // 当前模型所支持的最大上下文长度
{
    // grid(max_q_len, batch_size, kv_head_num)
    // blockSize = head_size
    // 计算需要从k_src中取出的数据位置
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;
    int token_id = blockIdx.x;  // 当前处理的token

    // 指针偏移到当前layer的k cache
    T *k_cache_dst = k_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];  // 已经缓存的kv cache的token数

    // 判断当前的block处理的token是小于当前的cur_seq_len的，防止越界
    if(token_id < cur_seq_len){
        // 计算写入的位置
        // [batch, kv head num, max_q_len, head size] -> [batch, head num, maxseqlen[cumsum_seq_len:cumsum_seq_len + max q len], head size]
        int src_offset = batch_id * kv_head_num * max_q_len * head_size + 
                         head_id * max_q_len * head_size +
                         token_id * head_size +
                         tid;
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size +
                         head_id * max_seq_len * head_size + 
                         (cumsum_seq_len + token_id) * head_size + 
                         tid;

        // 将数据写入
        k_cache_dst[dst_offset] = k_src[src_offset];
    }
}

template <typename T>
__global__ void append_value_cache(T *v_dst,  // [num_layers, bs, kv_head_num, max_q_len, head_size]
                                 const size_t layer_offset,
                                 const T *v_src,    // [bs, kv_head_num, max_q_len, head_size]
                                 const int kv_head_num,
                                 const int head_size,
                                 const int *cur_query_length,   // [bs] 用于避免越界
                                 const int *history_length, // [bs]
                                 const int max_q_len,
                                 const int max_seq_len) // 当前模型所支持的最大上下文长度
{
    // 与append_key_cache类似，但是要注意跳转到v cache
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;
    int token_id = blockIdx.x;

    // 将指针偏移到当前layer的v cache
    T *v_cache_dst = v_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];
    if(token_id < cur_seq_len){
        int src_offset = batch_id * kv_head_num * max_q_len * head_size +
                         head_id * max_q_len * head_size +
                         token_id * head_size +
                         tid;
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size +
                         head_id * max_seq_len * head_size + 
                         (cumsum_seq_len + token_id) * head_size +
                        tid;
        
        v_cache_dst[dst_offset] = v_src[src_offset];
    }
}










template <typename T>
void launchConcatKVCache(TensorWrapper<T> *k_src,   // from qkv bias and rope [batch_size. kv_head_num, max_q_len, head_size]
                         TensorWrapper<T> *v_src,
                         TensorWrapper<int> *layer_id,  // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                         TensorWrapper<int> *cur_query_length,  //current epoch or local input length,[batchsize]
                         TensorWrapper<int> *history_length, // [batch_size]
                         TensorWrapper<T> *k_dst,
                         TensorWrapper<T> *v_dst)
{
    /*
        k/v shape = [bs, kv_head_num, max_q_len, head_size]
        kv cache shape = [num_layers, bs, kv_head_num, max_seq_len, head_size]
    */
    int batch_size = k_src->shape[0];
    int max_seq_len = k_dst->shape[3];
    int kv_head_num = k_dst->shape[2];
    int max_q_len = k_src->shape[2];
    int head_size = k_src->shape[3];
    int blockSize = head_size;
    int layer = layer_id->getVal();    // 获取第一个数据： layer_id[0]
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    dim3 grid(max_q_len, batch_size, kv_head_num);  // self_decoder的时候，max_q_len=1
    append_key_cache<T><<<grid, blockSize>>>(k_dst->data,
                                             layer_offset,
                                             k_src->data,
                                             kv_head_num,
                                             head_size,
                                             cur_query_length->data,
                                             history_length->data,
                                             max_q_len,
                                             max_seq_len);

    append_key_cache<T><<<grid, blockSize>>>(v_dst->data,
                                             layer_offset,
                                             v_src->data,
                                             kv_head_num,
                                             head_size,
                                             cur_query_length->data,
                                             history_length->data,
                                             max_q_len,
                                             max_seq_len);
}


template void launchConcatKVCache(TensorWrapper<float> *k_src,   // from qkv bias and rope [batch_size. kv_head_num, max_q_len, head_size]
                         TensorWrapper<float> *v_src,
                         TensorWrapper<int> *layer_id,  // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                         TensorWrapper<int> *cur_query_length,  //current epoch or local input length,[batchsize]
                         TensorWrapper<int> *history_length, // [batch_size]
                         TensorWrapper<float> *k_dst,
                         TensorWrapper<float> *v_dst);

template void launchConcatKVCache(TensorWrapper<half> *k_src,   // from qkv bias and rope [batch_size. kv_head_num, max_q_len, head_size]
                         TensorWrapper<half> *v_src,
                         TensorWrapper<int> *layer_id,  // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                         TensorWrapper<int> *cur_query_length,  //current epoch or local input length,[batchsize]
                         TensorWrapper<int> *history_length, // [batch_size]
                         TensorWrapper<half> *k_dst,
                         TensorWrapper<half> *v_dst);