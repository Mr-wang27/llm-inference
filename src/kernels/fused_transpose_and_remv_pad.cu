#include <iostream>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_transpose_and_remv_pad.h"

// [bs, head_nums, seq_len, head_size] --> [bs, seeq_len, head_nums, head_size]  -->[num_tokens, head_nums, head_size]

// padding_offset.shape = [num_tokens]
/*
11111000    11110000    11111110    11111100
11110000
11111110
11111100

padding offset: 00000    3333    7777777    888888
*/

template <typename T>
__global__ void fused_transpose_reshape_remv_pad(T *src,//  [bs,head nums,seqlen,head size]
                                                 T *dst,//  [num tokens,head nums,head size]
                                                 const int num_tokens,
                                                 const int batch_size,
                                                 const int seq_len,
                                                 const int head_num,
                                                 const int head_size,
                                                 const int *padding_offset/*[num_tokens]*/)
{
    // [bs, head_nums, seq_len, head_size] --> [bs, eeq_len, head_nums, head_size]  -->[num_tokens, head_nums, head_size]
    // 只需要计算出读写的数据对于偏移即可
    /*
        dim3 grid(num_tokens);
        dim3 block(std::min(1024, head_num*head_size));
    */
    int token_id = blockIdx.x;
    int batch_id = (token_id + padding_offset[token_id]) / seq_len;
    int seq_id = (token_id + padding_offset[token_id]) % seq_len;

    // 计算偏移
    int dst_offset = token_id * head_num * head_size;   // 偏移到对应的[head_num, head_size]位置
    int src_offset = batch_id * head_num * seq_len * head_size + seq_id * head_size;  // 偏移到对应的[特定, head_nums, 特定,head_size]位置


    for(int i = threadIdx.x; i < head_num * head_size; i += blockDim.x ){
        int head_id = i / head_size;
        int head_size_id = i % head_size;
        dst[dst_offset + i] = src[src_offset + head_id * seq_len * head_size + head_size_id];
    }
}


template <typename T>
void launchTransposeOutRemovePadding(TensorWrapper<T> *qkv_buf_w_pad,   //  [bs,head nums,seqlen,head size]
                                     TensorWrapper<int> *padding_offset,  //  [num_tokens]    注意，padding_offset真正上并不是numtokens，只是只有前num_tokens有用
                                     TensorWrapper<T> *qkv_buf_wo_pad_1)    // [num tokens,head nums,head size]
{
    // 获取属性
    int batch_size = qkv_buf_w_pad->shape[0];
    int num_tokens = qkv_buf_wo_pad_1->shape[0];
    int seq_len = qkv_buf_w_pad->shape[2];
    int head_num = qkv_buf_w_pad->shape[1];
    int head_size = qkv_buf_w_pad->shape[3];

    // 分配kernel
    // 一个block处理一个token,
    dim3 grid(num_tokens);
    dim3 block(std::min(1024, head_num*head_size));
    // 启动kernel
    fused_transpose_reshape_remv_pad<T><<<grid, block>>>(qkv_buf_w_pad->data,
                                        qkv_buf_wo_pad_1->data,
                                        num_tokens,
                                        batch_size,
                                        seq_len,
                                        head_num,
                                        head_size,
                                        padding_offset->data);
}

template void launchTransposeOutRemovePadding(TensorWrapper<float> *qkv_buf_w_pad,   //  [bs,head nums,seqlen,head size]
                                              TensorWrapper<int> *padding_offset,  //  [num_tokens]    注意，padding_offset真正上并不是numtokens，只是只有前num_tokens有用
                                              TensorWrapper<float> *qkv_buf_wo_pad_1);
                        
template void launchTransposeOutRemovePadding(TensorWrapper<half> *qkv_buf_w_pad,   //  [bs,head nums,seqlen,head size]
                                              TensorWrapper<int> *padding_offset,  //  [num_tokens]    注意，padding_offset真正上并不是numtokens，只是只有前num_tokens有用
                                              TensorWrapper<half> *qkv_buf_wo_pad_1);