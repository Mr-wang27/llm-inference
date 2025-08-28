#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

template <typename T>
void launchAddResidual(     // residual.shape = [num_tokens, hidden_units]  batch_size=num_tokens
    TensorWrapper<T>* residual,
    TensorWrapper<T>* decoder_out,  // [num_tokens, hidden_units]
    bool is_print=false
);