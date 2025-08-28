#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

template <typename T>
void launchScaleMaskAndSoftmax(TensorWrapper<T> *qk,    // (batch_size, head_num, seq_q_len, seq_k_len)
                               TensorWrapper<T> *mask,  // (bacth_size, seq_q_len, seq_k_len)
                               TensorWrapper<T> *attn_score,    // (batch_size, head_num, seq_q_len, seq_k_len)
                               float scale);