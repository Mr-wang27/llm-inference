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

    residual.shape =        [num_tokens, hidden_units]
    decoder_out.shape =     [num_tokens, hidden_units]
    RMSNorm:scale.shape =   [hidden_unitws]
*/
#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_addresidual_norm.h"

// 先实现block reduce sum
template <typename T>
__device__ T warpReduceSum(T val)
{
    for(int i = 32 / 2; i > 0; i >>=1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
}

template <typename T>
__device__ T blockReduceSum(T val)
{
    int tid = threadIdx.x;
    int warp_nums = (blockDim.x +32 - 1) / 32;
    int warp_id = threadIdx.x / 32;
    int lean_id = threadIdx.x % 32;

    // 直接调用warpReduceSum做reduce,然后需要将数据存在shared memory上
    static __shared__ T warp_sum[64];
    val = warpReduceSum<T>(val);
    if(lean_id == 0){
        warp_sum[warp_id] = val;
    }
    __syncthreads();
    // 准备数据再做一次warp reduce
    T sum = tid < warp_nums ? warp_sum[warp_id] : (T)0.0f;
    sum = warpReduceSum<T>(sum);
    return sum;
}









// 1.this kernel is used after self attention in every layer
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
template <typename T>
__global__ void FusedAddBiasResidualRMSNorm(T* residual, // residual.shape = [num tokens, hidden_units]
                                            T* decoder_out, // [num_tokens, hidden_units]
                                            /*optional*/const T* bias,  // [hidden_units]
                                            const T* scale, // [hidden_units]   RMSNorm weights
                                            float eps,  // RMSNorm eps
                                            int num_tokens,
                                            int hidden_units)
{
    /*
        dim3 grid(num_tokens);
        dim3 block(hidden_units / vec_size);
    */
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int token_id = blockIdx.x;
    int tid = threadIdx.x;

    // 做向量化的操作，先把指针转化成向量指针
    Vec_t *vec_rsd = reinterpret_cast<Vec_t*>(residual + token_id * hidden_units);
    const Vec_t *vec_scale = reinterpret_cast<const Vec_t*>(scale);
    Vec_t *vec_deocder_out = reinterpret_cast<Vec_t*>(decoder_out + token_id * hidden_units);
    const Vec_t *vec_bias;
    if(bias != nullptr){
        vec_bias = reinterpret_cast<const Vec_t*>(bias);
    }

    T thread_accm = static_cast<T>(0);
    Vec_t res;
    // 1. decoder_out + bias + residual,
    for(int i = tid; i < hidden_units/vec_size; i += blockDim.x){
        if(bias != nullptr){
            res.x = vec_deocder_out[i].x + vec_bias[i].x + vec_rsd[i].x;
            res.y = vec_deocder_out[i].y + vec_bias[i].y + vec_rsd[i].y;
            res.z = vec_deocder_out[i].z + vec_bias[i].z + vec_rsd[i].z;
            res.w = vec_deocder_out[i].w + vec_bias[i].w + vec_rsd[i].w;
            vec_rsd[i] = res;
        }
        else{
            res.x = vec_deocder_out[i].x + vec_rsd[i].x;
            res.y = vec_deocder_out[i].y + vec_rsd[i].y;
            res.z = vec_deocder_out[i].z + vec_rsd[i].z;
            res.w = vec_deocder_out[i].w + vec_rsd[i].w;
            vec_rsd[i] = res;
        }
    // 2. 从vec_rsd中的数据累积线程的平方和
        thread_accm += res.x * res.x;
        thread_accm += res.y * res.y;
        thread_accm += res.z * res.z;
        thread_accm += res.w * res.w;
    }
    // 3. 对thread_accm做block reduce
    T block_sum = blockReduceSum(thread_accm);
    // 4. 计算RMSNorm的分母， 这个分母需要放在shared memory上，供这个block的所有线程使用
    __shared__ float inv_fenmu;
    if(tid == 0){
        inv_fenmu = rsqrt(block_sum / hidden_units + eps);
    }
    __syncthreads();

    // 5. 计算RMSNorm,然后写入到vec_decoder_out中
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x){
        vec_deocder_out[i].x = vec_scale[i].x * vec_rsd[i].x * inv_fenmu;
        vec_deocder_out[i].y = vec_scale[i].y * vec_rsd[i].y * inv_fenmu;
        vec_deocder_out[i].z = vec_scale[i].z * vec_rsd[i].z * inv_fenmu;
        vec_deocder_out[i].w = vec_scale[i].w * vec_rsd[i].w * inv_fenmu;
    }
}



template <>
__global__ void FusedAddBiasResidualRMSNorm(half* residual, // residual.shape = [num tokens, hidden_units]
                                            half* decoder_out, // [num_tokens, hidden_units]
                                            /*optional*/const half* bias,  // [hidden_units]
                                            const half* scale, // [hidden_units]   RMSNorm weights
                                            float eps,  // RMSNorm eps
                                            int num_tokens,
                                            int hidden_units)
{
    /*
        dim3 grid(num_tokens);
        dim3 block(hidden_units / vec_size);
    */
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int token_id = blockIdx.x;
    int tid = threadIdx.x;

    // 做向量化的操作，先把指针转化成向量指针
    Vec_t *vec_rsd = reinterpret_cast<Vec_t*>(residual + token_id * hidden_units);
    const Vec_t *vec_scale = reinterpret_cast<const Vec_t*>(scale);
    Vec_t *vec_deocder_out = reinterpret_cast<Vec_t*>(decoder_out + token_id * hidden_units);
    const Vec_t *vec_bias;
    if(bias != nullptr){
        vec_bias = reinterpret_cast<const Vec_t*>(bias);
    }

    // float thread_accm = static_cast<half>(0);
    float thread_accm = 0.0f;
    Vec_t res;
    // 1. decoder_out + bias + residual,
    for(int i = tid; i < hidden_units/vec_size; i += blockDim.x){
        if(bias != nullptr){
            res = __hadd2(__hadd2(vec_deocder_out[i], vec_bias[i]), vec_rsd[i]);
            vec_rsd[i] = res;

        }
        else{
            res = __hadd2(vec_deocder_out[i], vec_rsd[i]);
            vec_rsd[i] = res;
        }
    // 2. 从vec_rsd中的数据累积线程的平方和
        thread_accm += __half2float(res.x) * __half2float(res.x) + __half2float(res.y) * __half2float(res.y);
    }
    // 3. 对thread_accm做block reduce
    float block_sum = blockReduceSum(thread_accm);
    // 4. 计算RMSNorm的分母， 这个分母需要放在shared memory上，供这个block的所有线程使用
    __shared__ Vec_t inv_fenmu; // 因为后面要做向量化的乘法，所以这里需要把inv_fenmu做成half2
    if(tid == 0){
        inv_fenmu = scalar_cast_vec<Vec_t>(__float2half(rsqrt(block_sum / hidden_units + eps)));
    }
    __syncthreads();

    // 5. 计算RMSNorm,然后写入到vec_decoder_out中
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x){
        vec_deocder_out[i] = __hmul2(__hmul2(vec_scale[i], vec_rsd[i]), inv_fenmu);
    }
}


template <typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num_tokens, hidden_units]
                                        TensorWrapper<T> *residual,
                                        TensorWrapper<T> *decoder_out,  // [num_tokens, hidden_units]
                                        BaseWeight<T> &norm,    // bias [hidden_units]每个输出维度一个bias
                                        T* scale,   // RMSNorm weights  [hidden_units]
                                        float eps)
{
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    T *bias = norm.bias;
    T *gamma = scale;
    int vec_size = Vec<T>::size;    // 获取向量化数据的size
    int num_threads = hidden_units / vec_size;
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    // dim3 block(std::min(num_threads, 1024));
    FusedAddBiasResidualRMSNorm<T><<<grid, block>>>(residual->data,
                                                    decoder_out->data,
                                                    bias,
                                                    gamma,
                                                    eps,
                                                    num_tokens,
                                                    hidden_units);
#ifdef PRINT_DATA
    printf("fused addres norm kernel top2 result:\n");
    print_data<<<1,1>>>(decoder_out->data);     // decoder_out是传出的数据
#else
#endif
}



template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num_tokens, hidden_units]
                                                TensorWrapper<float> *residual,
                                                TensorWrapper<float> *decoder_out,  // [num_tokens, hidden_units]
                                                BaseWeight<float> &norm,    // bias [hidden_units]每个输出维度一个bias
                                                float* scale,   // RMSNorm weights  [hidden_units]
                                                float eps);


template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num_tokens, hidden_units]
                                                TensorWrapper<half> *residual,
                                                TensorWrapper<half> *decoder_out,  // [num_tokens, hidden_units]
                                                BaseWeight<half> &norm,    // bias [hidden_units]每个输出维度一个bias
                                                half* scale,   // RMSNorm weights  [hidden_units]
                                                float eps);