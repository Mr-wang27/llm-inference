#include "src/kernels/attn_softmax_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <float.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
// attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
// qk,                 (batch_size, head_num, q_length, k_length), QK^T.
// attention_mask,     (batch_size, q_length, k_length), attention mask.


// block reduce
template <typename T>
struct SumOp{   // 这就是一个functor,仿函数
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename T>
struct MaxOp{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return max(a, b);   // <cuda_runtime.h>
    }   
};

// 模板模板参数， 这里的typename T 是Reduction使用的模板参数： ReductionOp<T>
template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T warpReduce(T val)
{
    for (int mask = 32 / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


template <typename T>
__inline__ __device__ T warpReduceMax(T val)
{
    for (int offset = 32 / 2; offset > 0; offset /= 2)
    {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}




//
template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T blockReduce(T val)
{
    // 在warpreduce的基础上在做一次warp reduce
    // 一般blockDim.x <= 1024,因此，我们考虑1024/32=32.刚好够一个warp
    /*
        1. 对每个warp进行reducee---->计算warp nums, warp id
        2. 对每个warp进行reduce,然后讲每个warp的reduce存在shared memroy上
        3. 为下一次warp reduce做准备，给每个线程分配需要进行reduce的数据，然后进行最后一次reduce
    */

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lean_id = tid % 32;
    int warp_nums = (blockDim.x + 32 - 1) / 32;
    static __shared__ T warp[32];
    val = warpReduce<ReductionOp, T>(val);
    if(lean_id == 0){
        warp[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warp[warp_id] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}


template <typename T, int NUMS_PER_THREAD_PER_ROW>
__global__ void ScaleMaskAndSoftmax_float(T *attn_score,
                                          T *qk,
                                          T *mask,
                                          int batch_size,
                                          int head_nums,
                                          int q_len,
                                          int k_len,
                                          float scale)  // sacle = q/sqrt(head_size)
{
/*
    注意：这里这个kernel是context的kernel
    dim3 grid(q_length, batch_size, head_nums);
    dim3 block(k_length/vec_size / NUMS_PER_THREAD_PER_ROW)     NUMS_PER_THREAD_PER_ROW是每个线程需要处理的次数，如果是half2,NUMS_PER_THREAD_PER_ROW=4，那这个线程就要处理4个half2，也就是8个元素
    需要注意的是，这里处理的4个half2是按照网格步进行处理的，也就是按照stride = blockDim.x进行处理
    attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    attention_mask,     (batch_size, q_length, k_length), attention mask.
*/
    // 先将数据从qk以及mask中取出来
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;

    // 过滤线程，因为线程处理的数据量是大于等于k_length的，  // 避免访问越界
    if(tid >= k_len){
        return;
    }
    __shared__ float inv_sum, s_max;
    // 这里是通用一点的写法，主要是为了扩展性， 如果在分配grid的x维度的block数量小于q_length的时候，这个kernel也能够处理
    for(int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x){
        int qk_offset = 0;
        int mask_offset = 0;
        T qk_data = static_cast<T>(0);
        T mask_data = static_cast<T>(0);
        T data[NUMS_PER_THREAD_PER_ROW];
        T thread_max = FLT_MIN; // float 的最小值
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){
            // 处理一次数据
            // 计算偏移
            qk_offset =     batch_id * head_nums * q_len * k_len +
                            head_id * q_len * k_len +
                            row_start * k_len +
                            col_start * blockDim.x + tid;
            mask_offset =   batch_id * q_len * k_len +
                            row_start * k_len +
                            col_start * blockDim.x + tid;
            // 将数据取出来
            qk_data = qk[qk_offset];
            mask_data = mask[mask_offset];
            // 计算scale以及mask后的数据
            data[col_start] = scale * qk_data + (1 - mask_data) * (-10000.0f);
            // 求最大值
            thread_max = fmax(data[col_start], thread_max);
        }

        // 在这一行上求最大值---->求所有线程的最大值
        T max_val = blockReduce<MaxOp, T>(thread_max);
        // 将最大值取出来，放在shared memory上
        if(threadIdx.x == 0){
            s_max = max_val;    // 取出这一行的max
        }
        __syncthreads();
        // 计算这一行的exp(xi - max)的sum reduce
        // 因为一个线程要处理次数据，所以在blockreduce前，需要把线程的数据进行加和处理
        T thread_sum = 0.;
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){
            // 将这一线程处理的的数据做计算exp(xi - max)相加，然后放在local memory上
            // qk_offset = batch_id * head_nums * q_len * k_len +
            //             head_id * q_len * k_len +
            //             row_start * k_len +
            //             col_start * blockDim.x + tid;
            data[col_start] = expf(data[col_start] - s_max);    // expf: cuda_runtime.h的函数       注意：此时data中存储的数据是e^(x_1 - max),也就是softmax的分子
            thread_sum += data[col_start];
        }
        // 下面就进行blockSumReduce
        T sum = blockReduce<SumOp, T>(thread_sum);
        // 取出sum。然后计算inv_sum
        if(threadIdx.x == 0){
            inv_sum = 1 / (sum + 1e-6f);
        }
        __syncthreads();

        // 计算softmax,利用inv_sum,结果存在attn_score：  (batch_size, head_num, q_length, k_length)
        // qk:x/inv_sum
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){
            // 计算qk_offset的偏移
            qk_offset = batch_id * head_nums * q_len * k_len +
            head_id * q_len * k_len +
            row_start * k_len +
            col_start * blockDim.x + tid;
            // 写入位置与读取位置是一样的
            attn_score[qk_offset] = data[col_start] * inv_sum;  // 分子/分母
        }
    }
}

// CUDA 不支持部分特化的 __global__ 函数，所以只能重新写一个模板函数来执行half
// cant partial specialize in func
template <typename T_half, int NUMS_PER_THREAD_PER_ROW> // 传入的时候：T_half=half
__global__ void ScaleMaskAndSoftmax_half(T_half *attn_score,
                                         T_half *qk,
                                         T_half *mask,
                                         int batch_size,
                                         int head_nums,
                                         int q_len,
                                         int k_len,
                                         float scale)
{
/*
注意：这里这个kernel是context的kernel
dim3 grid(q_length, batch_size, head_nums);
dim3 block(k_length/vec_size / NUMS_PER_THREAD_PER_ROW)     NUMS_PER_THREAD_PER_ROW是每个线程需要处理的次数，如果是half2,NUMS_PER_THREAD_PER_ROW=4，那这个线程就要处理4个half2，也就是8个元素
需要注意的是，这里处理的4个half2是按照网格步进行处理的，也就是按照stride = blockDim.x进行处理
attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
qk,                 (batch_size, head_num, q_length, k_length), QK^T.
attention_mask,     (batch_size, q_length, k_length), attention mask.
*/
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int vec_size = Vec<T_half>::size;   // 向量化的数据量
    int tid = threadIdx.x;
    using Vec_t = typename Vec<T_half>::Type;   // 向量化类型： half2

    // 将数据指针转为向量化指针
    Vec_t* attn_score_vec = reinterpret_cast<Vec_t*>(attn_score);
    Vec_t* qk_buf_vec = reinterpret_cast<Vec_t*>(qk);
    Vec_t* attn_mask_vec = reinterpret_cast<Vec_t*>(mask);
    // 在计算的过程中，需要使用half2的intrinsic做向量化的计算
    // 因此将涉及到的标量转为向量： scale,mask,one
    Vec_t ONE = scalar_cast_vec<Vec_t>(__float2half(1.0f));    // 将一个half转为一个half2:(1,1)
    Vec_t NEG_INF = scalar_cast_vec<Vec_t>(__float2half(-10000.0f));
    Vec_t scale_vec = scalar_cast_vec<Vec_t>(__float2half(scale));

    __shared__ float inv_sum, s_max;    // 计算得到的分母以及最大值，需要放在shared_memory上，供这一个block(行)的数据使用
    if(threadIdx.x * vec_size >= k_len){    // 过滤线程
        return;
    }

    for(int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x){
        int qk_offset = 0;
        int mask_offset = 0;
        Vec_t qk_data;
        Vec_t mask_data;
        float thread_max = FLT_MIN;
        Vec_t data[NUMS_PER_THREAD_PER_ROW];
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){
            // 计算偏移
            qk_offset = batch_id * head_nums * q_len * (k_len / vec_size) +
                        head_id * q_len * (k_len  / vec_size) +
                        row_start * (k_len / vec_size) +
                        col_start * blockDim.x + tid;
            mask_offset = batch_id * q_len * (k_len / vec_size) +
                          row_start * (k_len / vec_size) +
                          col_start * blockDim.x + tid;
            qk_data = qk_buf_vec[qk_offset];
            mask_data = attn_mask_vec[mask_offset];
            // 计算：qk*scale + (1-mask) * inf;
            Vec_t mask_vec_reg = __hmul2(__hsub2(ONE, mask_data), NEG_INF);
            data[col_start] = __hadd2(__hmul2(qk_data, scale_vec), mask_vec_reg);
            // 求一下当前线程的最大值，
            thread_max = fmax(fmax((float)data[col_start].x, (float)data[col_start].y), (float)thread_max);
        }

        // 做warp max reduce
        float max_val = blockReduce<MaxOp, float>(thread_max);
        // 取出最大值，放在shared memory上
        if(tid == 0){
            s_max = max_val;
        }
        __syncthreads();

        // 计算分母： sum: e^(x_i - s_max)
        // 先计算出每个线程处理的数据的和，然后再做block reduce
        float thread_sum = 0.0f;
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){
            // qk_offset = batch_id * head_nums * q_len * (k_len / vec_size) +
            //             head_id * q_len * (k_len  / vec_size) +
            //             row_start * (k_len / vec_size) +
            //             col_start * blockDim.x + tid;
            // qk_data = ;
            Vec_t tmp = __hsub2(data[col_start], scalar_cast_vec<Vec_t>(s_max));
            data[col_start] = h2exp(tmp);                                                               // 分子
            thread_sum += (float)(__hadd(data[col_start].x, data[col_start].y));
        }
        // 计算行和
        float sum = blockReduce<SumOp, float>(thread_sum);
        if(tid == 0){
            inv_sum = 1 / (sum + 1e-6f);    // 分母
        }
        __syncthreads();
        // 计算分子/分母，然后写回到attn_mask_vec中，向量化的写
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){
            qk_offset = batch_id * head_nums * q_len * (k_len / vec_size) +
            head_id * q_len * (k_len  / vec_size) +
            row_start * (k_len / vec_size) +
            col_start * blockDim.x + tid;

            attn_score_vec[qk_offset] = __hmul2(data[col_start], scalar_cast_vec<Vec_t>(inv_sum));
        }
    }
}








// \是预处理器的换行连接符，在使用多行宏定义的时候，就需要换行连接符。因为宏定义默认只能识别单行的内容
// ##dtype 是 C/C++ 宏中的一种标记粘贴运算符（token pasting operator），也叫 宏连接符。它的作用是：
// 将宏参数 dtype 和前后的标记连接成一个新的标记（标识符）。
// 宏定义
#define LAUNCH_SOFTMAX(dtype, vec_size)                                                                                                     \
    if(block.x > 2048 && block.x < 4096){                                                                                                   \
        constexpr int NUMS_PER_THREAD_PER_ROW = 4;                                                                                          \
        block.x /= 4 * vec_size;                                                                                                            \
        block.x = (block.x + 32 -1) / 32 * 32;                                                                                              \
        assert(block.x <= 1024);                                                                                                            \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data,                             \
                                                                                     (dtype *)qk->data,                                     \
                                                                                     (dtype *)mask->data,                                   \
                                                                                     batch_size,                                            \
                                                                                     head_nums,                                             \
                                                                                     q_length,                                              \
                                                                                     k_length,                                              \
                                                                                     scale);                                                \
    }                                                                                                                                       \
    else if(block.x > 1024){                                                                                                                \
        constexpr int NUMS_PER_THREAD_PER_ROW = 2;                                                                                          \
        block.x /= 2 * vec_size;                                                                                                            \
        block.x = (block.x + 32 -1) / 32 * 32;                                                                                              \
        assert(block.x <= 1024);                                                                                                            \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data,                             \
                                                                                     (dtype *)qk->data,                                     \
                                                                                     (dtype *)mask->data,                                   \
                                                                                     batch_size,                                            \
                                                                                     head_nums,                                             \
                                                                                     q_length,                                              \
                                                                                     k_length,                                              \
                                                                                     scale);                                                \
    }                                                                                                                                       \
    else{                                                                                                                                   \
        constexpr int NUMS_PER_THREAD_PER_ROW = 1;                                                                                          \
        block.x /= vec_size;                                                                                                                \
        assert(block.x <= 1024);                                                                                                            \
                ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data,                     \
                                                                                     (dtype *)qk->data,                                     \
                                                                                     (dtype *)mask->data,                                   \
                                                                                     batch_size,                                            \
                                                                                     head_nums,                                             \
                                                                                     q_length,                                              \
                                                                                     k_length,                                              \
                                                                                     scale);                                                \
    }                                                                                                                                       \

template <typename T>
void launchScaleMaskAndSoftmax(TensorWrapper<T> *qk,
                               TensorWrapper<T> *mask,
                               TensorWrapper<T> *attn_score,
                               float scale)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    int q_length = qk->shape[2];
    int batch_size = qk->shape[0];
    int head_nums = qk->shape[1];
    int k_length = qk->shape[3];
    bool is_half = sizeof(T) == 2;  // 判断当前数据是否是half，如果是，在后面启动kernel的时候有差异
    // 因为half要使用half2进行向量化处理
    // 因为这里使用了half2进行向量化处理，所以就要保证数据量能够被2整除，也就是k_length % 2 == 0
    // 如果不想进行检查，可以通过padding,讲数据padding到能够被整除的长度，然后再传入一个padding参数来维护真实数据长度进行计算即可
    if (is_half){
        LLM_CHECK_WITH_INFO(k_length % 2 == 0, "Currently, K_len should be divided by 2 under half type!");
    }

    // 还是分配每个block处理一个token即k_length个数据。
    dim3 grid(q_length, batch_size, head_nums);
    dim3 block((k_length + 32 -1) / 32 * 32);   // 对齐32个线程， 确保线程数能够cover数据量。
    // 但是如果随着对话的进行，k_length长度会逐渐增加，可能超过1024，因此就需要避免分配的线程数量超过1024，导致性能下降
    // 所以就需要在这里对block的的分配进行调整
    // 同时，对于half采用half2进行向量化
    // 对于float可以采用float4进行向量化，但是这里没有使用float的向量化
    // 这里使用宏展开调整代码
    if(is_half){
        LAUNCH_SOFTMAX(half,2); // 2是向量化的长度
    }
    else{
        LAUNCH_SOFTMAX(float, 1);
    }

#ifdef PRINT_DATA
    printf("attn softmax kernel top2 result:\n");
    print_data<<<1, 1>>>(attn_score->data);
#else
#endif
}




template void launchScaleMaskAndSoftmax(TensorWrapper<float> *qk,
                               TensorWrapper<float> *mask,
                               TensorWrapper<float> *attn_score,
                               float scale);

template void launchScaleMaskAndSoftmax(TensorWrapper<half> *qk,
                               TensorWrapper<half> *mask,
                               TensorWrapper<half> *attn_score,
                               float scale);