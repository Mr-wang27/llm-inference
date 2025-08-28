#include <stdio.h>
#include "src/kernels/add_residual.h"
#include "src/utils/cuda_debug_utils.cuh"

// 该kernel用于每个decoder的ffn后


template <typename T>
__global__ void AddResidual(
    T* residual,
    T* decoder_output,
    int num_tokens,
    int hidden_units
)
{
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t*>(decoder_output + batch_id * hidden_units);    // 偏移到这个block需要处理的数据的位置
    Vec_t *rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);
    for(int index = tid; index < hidden_units / vec_size; index += blockDim.x){
        // float4
        dout[index].x += rsd[index].x;
        dout[index].y += rsd[index].y;
        dout[index].w += rsd[index].w;
        dout[index].z += rsd[index].z;
    }
}


template <>
__global__ void AddResidual(
    half* residual,
    half* decoder_output,
    int num_tokens,
    int hidden_units
)
{
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t*>(decoder_output + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t*>(decoder_output + batch_id * hidden_units);

    for(int index = tid; tid < hidden_units / vec_size; tid += blockDim.x){
        // half2
        dout[index] = __hadd2(dout[index], rsd[index]);
    }
}





template <typename T>
void launchAddResidual(
    TensorWrapper<T> *residual, // [num_tokens, hidden_units]       num_tokens = batch_size
    TensorWrapper<T>* decoder_out,  // [num_tokens, hidden_units]
    bool is_print
)
{
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    dim3 grid(batch_size);  // 一般而言，batch_size不会很大，但是如果是在prifll阶段，num_tokens可能很大
    dim3 block(256);
    AddResidual<T><<<grid, block>>>(residual->data, decoder_out->data, batch_size, hidden_units);
#ifdef PRINT_DATA
    if(is_print){
        print_data<<<1, 1>>>(deocder_out->data);
    }
#else
#endif
}


template void launchAddResidual(
    TensorWrapper<float> *residual, // [num_tokens, hidden_units]       num_tokens = batch_size
    TensorWrapper<float>* decoder_out,  // [num_tokens, hidden_units]
    bool is_print
);

template void launchAddResidual(
    TensorWrapper<half> *residual, // [num_tokens, hidden_units]       num_tokens = batch_size
    TensorWrapper<half>* decoder_out,  // [num_tokens, hidden_units]
    bool is_print
);