#include <iostream>
#include "src/kernels/act_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include "src/utils/macro.h"

template <typename T>
__device__ __forceinline__ T silu(const T& in){
    // 计算一个数据的silu(x) = x * sigmoid(x)
    return (T) ( (float)in / (1 + expf((float) -in)) );
}

template <>
__device__ __forceinline__ half2 silu<half2>(const half2& in){
    return make_half2( __float2half(silu<float>((float)in.x)), __float2half(silu<float>((float)in.y)) );
}

template <typename T>
__global__ void silu_and_mul_kernel(T* out,     // out:     [bs, intermedia size]
                                    T* input,   // input:   [bs, 2, intermedia size]
                                    const int intermedia_size)
{
    // 注意的是，input这种的前一半内容是gate linear的输出，后一半内容是up linear的输出
    // 一个线程处理一个输出得到的数据，这个线程需要从input中把两个部分的数据都取出来，然后做计算，然后再存回out中即可
    const int batch_id = blockIdx.x;
    for(int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x){
        int dst_id = batch_id * intermedia_size + idx;
        int gate_id = batch_id * 2 * intermedia_size + idx;
        int up_id = batch_id * 2 * intermedia_size + 1 * intermedia_size + idx;
        out[dst_id] = silu<T>(input[gate_id]) * input[up_id];
    }
}

template <>
__global__ void silu_and_mul_kernel<half>(half* out,
                                          half* input,
                                          const int intermedia_size)
{
    // 这里要用到向量化的读写
    const int batch_id = blockIdx.x;
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    // 把指针转为向量化类型
    Vec_t *vec_out = reinterpret_cast<Vec_t*>(out);
    Vec_t *vec_input = reinterpret_cast<Vec_t*>(input);
    for(int idx = threadIdx.x; idx < intermedia_size / vec_size; idx += blockDim.x){
        // 计算偏移
        int dst_id = batch_id * intermedia_size + idx * vec_size;
        int gate_id = batch_id * 2 * intermedia_size + idx * vec_size;
        int up_id = batch_id * 2 * intermedia_size + 1 * intermedia_size + idx * vec_size;
        vec_out[dst_id] = __hmul2(silu<half2>(vec_input[gate_id]), vec_input[up_id]);
    }
}



template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out){
    // input:   [bs, 2, intermedia size]        这个kernel是self和context公用的，所以bs可以是num_tokens, input是gate和up两个线性层的输出进行concate
    // out:     [bs, intermedia size]
    // input中的前一半是gate,后一半是up的结果
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);    // 检查是否以input这种数据格式进行输入
    int intermedia_size = input->shape[2];
    dim3 grid(batch_size);
    dim3 block(256);    // 固定线程分配
    silu_and_mul_kernel<T><<<grid, block>>>(out->data,
                                            input->data,
                                            intermedia_size);
#ifdef PRINT_DATA
    printf("act kernel top2 result:\n");
    print_data<<<1,1>>>(out->data);
#else
#endif
}


template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* out);
template void launchAct(TensorWrapper<half>* input, TensorWrapper<half>* out);