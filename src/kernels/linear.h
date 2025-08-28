// 原本每个层都具有自己的参数，需要在src/weight/llama里面定义这个层的参数
// 但是因为线性层的参数比较简单，可以直接用BaseWeight进行表示


#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>  // 文件输入输出流库，用来实现 文件读写操作
#include "src/kernels/cublas_utils.h"
#include "src/weights/base_weights.h"
#include "src/utils/tensor.h"
#include "src/utils/macro.h"

// TODO: 当将来启用 int8 / int4 的 weight-only（权重量化）模式时，可以新增一个模板类型参数 T2 来表示权重的类型。
template<typename T>
void launchLinearGemm(TensorWrapper<T>* input,      // [num_tokens, q_hideen_unit]
                      BaseWeight<T>& weight,     // [hidden_size, q_hidden_unit] ,为原始权重，未转置的
                      TensorWrapper<T>* output,     // [num_tokens, hidden_size]
                      cublasWrapper* cublas_wrapper,
                      bool trans_a = false, // 这里的tans_a指代的是input
                      bool trans_b = false);    // 这里的trans_b指代的是weight
// 因为是为转置的，所以我们计算的时候需要将权重矩阵转置，我们将权重矩阵设为矩阵A。    trans_b=true.因为矩阵weight要转置
// weight = A;  input = B
// 所以传入的参数 trans_b 需要为true


// 用于qk^T 以及attn与v的相乘
template<typename T>                                        // QK^T, 同样要trans_b=true,因为input2要转置                  // attn V
void launchLinearStridedBatchGemm(TensorWrapper<T>* input1,  // Q:[bs, heads, seq_len, head_size]                 // Attn:    [bs, heads, seq_len, seq_len] 
                                  TensorWrapper<T>* input2,  // K:[bs, heads, seq_len, head_size]                 // V:       [bs, heads, seq_len, head_size]
                                  TensorWrapper<T>* output,  // [bs, heads, seq_len, seq_len]                        // [bs, heads, seq_len, head_size]=[bs*seq_len, heads*head_size]=[num_token, hidden_size]
                                  cublasWrapper* cublas_wrapper,
                                  bool trans_a = false,      // 这里的trans_a指代的是input1是否需要转置
                                  bool trans_b = false       // trans_b指代的是input2是否需要转置
);
