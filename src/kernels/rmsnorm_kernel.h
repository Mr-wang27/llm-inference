#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/weights/llama/norm_weights.h"
#include "src/utils/vectorize_utils.h"
// 该kernel需要使用头文件全部写在.h文件中，然后再由.cu文件include


template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out,  // [num tokens, hidden_units]
                    TensorWrapper<T>* decoder_residual,
                    LayerNormWeight<T>& attn_norm_weight,   // RMSNorm weights:[hidden_unnits]
                    float eps,  // RMSNorm eps 
                    bool is_last = false
                    );
// 该kernel原地进行计算。
// 直接再decoder_out中输出结果。即输入与输出都是decoder_out



