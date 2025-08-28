#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"               
#include "src/weights/llama/norm_weights.h" // RMSNorm权重[q_hidden_dim]
#include "src/utils/vectorize_utils.h"  // 做向量化 
#include "src/weights/base_weights.h"

template <typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num_tokens, hidden_units]
                                        TensorWrapper<T> *residual,
                                        TensorWrapper<T> *decoder_out,  // [num_tokens, hidden_units]
                                        BaseWeight<T> &norm,    // bias [hidden_units]每个输出维度一个bias
                                        T* scale,   // RMSNorm weights  [hidden_units]
                                        float eps);


/*
    residual 是最初的数据
    decoder_out是做完attn之后的那个输出linear的输出结果，但是linear没有加bias
    所以这个算子需要先将bias加上
    decoder_out + bias + residual
    然后在做RMSNorm
    同时利用residual再次返回记录输入RMSNorm的数据作为新的residual
    所以：做的事情就是：
    residual = decoder_out + bias + residual
    RMSNorm(residual)
*/

/*
    linear层：
    toech.nn(head_size, hidden_dim);
    输入数据：[batch_size, seq_len, head_size]
    weights:  [hidden_dim, head_size]
    bias:     [hidden_dim]
    输出数据：[batch_size, seq_len, hidden_dim]

    RMSNorm:
    输入数据：  [batch_size, seq_len, hidden_dim]
    scale:      [hidden_dim]
    输出数据：  [batch_size, seq_len, hidden_dim]

*/