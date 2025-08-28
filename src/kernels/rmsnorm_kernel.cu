#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/rmsnorm_kernel.h"


// 先写warp level reduce操作，该操作基本上写法是固定的，使用shfl指令进行编写，性能更高
// __shfl_xor_sync(0xffffffff, val, i):表示所有线程(0xffffffff)均参与shfl.
// 同时将该线程后面的第i个线程的数据val返回给当前线程。
// 所以在一个warp上进行reduce操作，就只需要第一次i=16,将32个数据reduce成16个数据，并且存放在0，1，2，3..., 15位置上
// 然后再进行第二次i=8,将16个数据reduce成8个数据，存放在0-7的位置上
// 第三次rrduce:i=4, reduce成4个数据，存放再0-3上
// 第四次reduce:i=2, reduce成2个数据，存放在0-1上
// 第五次reduce:i=1, reduce成1个数据，最终的结果，存放在0号位置上
template<typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){   // i>>=1表示i右移一位，表示除以2
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 只有leandid=0的线程返回的才是正真的reduce结果
}
// 需要注意的是，warpReduceSum只管算，不管用，确实warp中的每个thread都会返回一个值
// 使用那个值由调用者决定。
// 这样写不会存在warp divergence,更高效一些。因为GPU是SIMD


// 然后写blockReduceSum。该写法也基本固定
// note:！！ 当blocksize<32的时候，使用blockDim.x/32 获取warp数量是错误的。应该进一
template<typename T>
__device__ T blockReduceSum(T val){
    // 1. 将所有数据先在warp level进行reduce,然后将每个warp的结果存在shared memory上
    // 2. 再次调用warp level reduce计算最终的结果，得到block的reduce结果
    int tid = threadIdx.x;  // block内的thread id
    int wid = tid / 32;
    int leanid = tid % 32;
    int warpnum = (blockDim.x + 32 -1) / 32;
    // 调用warp level reduce, 然后将数据存在shared memory上
    static __shared__ T warpsum[64];    // 这个数值的设定，应该是warpnum。但是static __shared__设置内存的时候，需要常量。而warpnum不是常量，所以不能使用warpnum进行设置
                                        // 而设置这个数据：如果一个block中分配了1024个线程，一个warp会处理32个数据，所以只需要1024/32=32个内存即可。因此分配32即可
                                        // 然而，如果blockDim.x=2048,就需要0248/32=64个内存位置。这里分配64个位置就是为了避免block中线程数量超过1024而出错
    val = warpReduceSum(val);
    // 将数据写入share memory中，供二次启动warpReduceSum使用
    if(leanid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();
    
    // 再次调用warpReduceSum计算block redue的值
    // 构造调用warpReduceSum计算所需的数据。参与warp reducde的一个warp中的线程需要拿到各自的数据
    T sum = tid < warpnum ? warpsum[tid] : (T)0;    // 因为如果tid>warpnum，在后面取数据的时候可能会越界
    // 其次warpnum之后的数据全部填充0，确保在warp数量不是32的整数倍时，进行warp reduce sum, 数据不会被污染
    sum = warpReduceSum(sum);
    return sum;     // 同样是只有block中的0号线程返回的是正确的结果。
}



// 下面开始实现RMSNormal kernel
// 该kernel用于每个decoder layer的开始和32个decoder layer之后
// 分配的线程数量为(head size / 4) 或者(head size / 2)
// 该kernel中，每个block处理一个token,
// 每个block中的线程处理该token中的所有数据
template<typename T>
__global__ void RMSNorm(T* decoder_out, // [num tokens, q_hidden_units]
                        T* decoder_residual,    // 存储原始数据，用于residual
                        T* scale,       // [q_hidden_unit], RMSNorm weights
                        float eps,      // RMSNorm eps
                        int num_tokens,
                        int hidden_units)
{
    int vec_size = Vec<T>::size;    // 获取向量化的数据量
    using Vec_t = typename Vec<T>::Type;    // 表示向量化中每个数据的数据类型
    float thread_sum = 0.0f;
    // 将指针转换成向量化的数据指针，以便进行向量化的读取
    // 每个block处理一个token, 将每个block中的0号线程的指针指向对应行的第一个元素
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);        
    Vec_t* rsd = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
    // 向量化的从global memory加载数据，同时计算每个线程所加载的数的平方和
    // block-step loop，确保少量线程也可以处理大量的数据
    // 因为一个block处理一个token的数据
    for(int idx = threadIdx.x; idx < hidden_units/vec_size; idx += blockDim.x){
        Vec_t vec = dout[idx];  // 读取一个float4数据
        rsd[idx] = vec; // 写入一个float4数据
        // 标量计算每个数据的平方和
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;
    }
    // 直接计算这个block的reduce sum, 计算一个block的reduce sum,需要block中的所有线程参与
    thread_sum = blockReduceSum<float>(thread_sum);
    // 然后计算平方和的均值的倒数，然后将该结果保存在shared memory上，
    // 因为RMSNorm是对token进行的归一化，而一个token由一个block进行处理。并且每一个token中的数据都要使用这个数据。所以正好存在可以存在shared memory上，以供使用
    // 不能放在线程的私有寄存器上面。其次需要注意的是，blockReduceSum仍然是只有block内线程id为0的数据是正确的结果
    __shared__ float inv_mean;    // 每个token使用的是同一个分母:inv_mean
    if(threadIdx.x == 0){
        inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
    }
    __syncthreads();

    // 然后将原始数据中的每一个数据全部乘以inv_mean再乘以sacle.
    // 需要注意的是scale: [hidden_units]。 每个token都使用这个数据sacle。同时每个token中的每个特征数据使用自己的scale
    // scale也可以进行向量化的读取
    Vec_t* s = reinterpret_cast<Vec_t*>(scale); // 这里需要确保RMSNorm参数类型与输入的数据参数类型一致。要么都是FP32要么都是FP16
    for(int idx = threadIdx.x; idx < hidden_units/vec_size; idx += blockDim.x){
        Vec_t out = dout[idx]; // 引入这个临时变量是为了编译器优化
        
        dout[idx].x = out.x * inv_mean * s[idx].x;
        dout[idx].y = out.y * inv_mean * s[idx].y;
        dout[idx].z = out.z * inv_mean * s[idx].z;
        dout[idx].w = out.w * inv_mean * s[idx].w;
    } 
}   // 结束。



// 特化FP16版本的kernel
template<>
__global__ void RMSNorm(half* decoder_out, // [num tokens, q_hidden_units]
                        half* decoder_residual,    // 存储原始数据，用于residual
                        half* scale,       // [q_hidden_unit], RMSNorm weights
                        float eps,      // RMSNorm eps
                        int num_tokens,
                        int hidden_units)
{
    int vec_size = Vec<half>::size;    // 获取向量化的数据量
    using Vec_t = typename Vec<half>::Type;    // 表示向量化中每个数据的数据类型
    float thread_sum = 0.0f;
    // 将指针转换成向量化的数据指针，以便进行向量化的读取
    // 每个block处理一个token, 将每个block中的0号线程的指针指向对应行的第一个元素
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);        
    Vec_t* rsd;
    if(decoder_residual != nullptr){    
        rsd = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
    }
    // 向量化的从global memory加载数据，同时计算每个线程所加载的数的平方和
    // block-step loop，确保少量线程也可以处理大量的数据
    // 因为一个block处理一个token的数据
    for(int idx = threadIdx.x; idx < hidden_units/vec_size; idx += blockDim.x){
        Vec_t vec = dout[idx];  // 读取一个__half2数据
        if(decoder_residual != nullptr){    
            rsd[idx] = vec; // 写入一个__half2数据
        }
        // 标量计算每个数据的平方和
        thread_sum += __half2float(vec.x) * __half2float(vec.x);    // 对于FP16的计算，是有专门的FP16计算指令可以使用，这里没有这么做
        thread_sum += __half2float(vec.y) * __half2float(vec.y);    // 而是转成FP16进行计算，然后存在thread_sun中
    }
    // 直接计算这个block的reduce sum, 计算一个block的reduce sum,需要block中的所有线程参与
    thread_sum = blockReduceSum<float>(thread_sum);
    // 然后计算平方和的均值的倒数，然后将该结果保存在shared memory上，
    // 因为RMSNorm是对token进行的归一化，而一个token由一个block进行处理。并且每一个token中的数据都要使用这个数据。所以正好存在可以存在shared memory上，以供使用
    // 不能放在线程的私有寄存器上面。其次需要注意的是，blockReduceSum仍然是只有block内线程id为0的数据是正确的结果
    __shared__ float inv_mean;    // 每个token使用的是同一个分母:inv_mean
    if(threadIdx.x == 0){
        inv_mean = rsqrtf(float(thread_sum / hidden_units) + eps);
    }
    __syncthreads();

    // 然后将原始数据中的每一个数据全部乘以inv_mean再乘以sacle.
    // 需要注意的是scale: [hidden_units]。 每个token都使用这个数据sacle。同时每个token中的每个特征数据使用自己的scale
    // scale也可以进行向量化的读取
    Vec_t* s = reinterpret_cast<Vec_t*>(scale); // 这里需要确保RMSNorm参数类型与输入的数据参数类型一致。要么都是FP32要么都是FP16
    for(int idx = threadIdx.x; idx < hidden_units/vec_size; idx += blockDim.x){
        Vec_t out = dout[idx]; // 引入这个临时变量是为了编译器优化
        
        dout[idx].x = __float2half(__half2float(out.x) * inv_mean) * s[idx].x;      // s是half
        dout[idx].y = __float2half(__half2float(out.y) * inv_mean) * s[idx].y;      // out是half, inv_mean是float32
    } 
}   // 结束。




// 启动kernel
template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out,  // [num_tokens, hidden_units]
                    TensorWrapper<T>* decoder_residual,
                    LayerNormWeight<T>& attn_norm_weight, // RMSNorm weights
                    float eps,  // RMSNorm eps
                    bool is_last    // for print last rmsnorm output to debug
                    )
{
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    // 获取调用kernel的类型的size, 这个size是
    // int vec_size = Vec<T>::size;
    // 分配线程
    int num_threads = hidden_units / 4; // 这里是4096/4=1024
    T* rsd = decoder_residual->data;
    dim3 grid(num_tokens);  // 一个block处理一个token
    dim3 block(num_threads);    // 处理一个token的线程数
    
    // 启动kernel
    RMSNorm<T><<<grid, block>>>(decoder_out->data,
                                rsd,
                                attn_norm_weight.gamma,
                                eps,
                                num_tokens,
                                hidden_units);
#ifdef PRINT_DATA
    printf("rmsnorm kernel top2 result:\n");
    print_data<<<1,1>>>(decoder_out->data); // 打印kernel上的数据
#else
#endif
}



// 实例化模板FP32\FP16
template void launchRMSNorm( TensorWrapper<float>* decoder_out,  // [num_tokens, hidden_units]
                    TensorWrapper<float>* decoder_residual,
                    LayerNormWeight<float>& attn_norm_weight, // RMSNorm weights
                    float eps,  // RMSNorm eps
                    bool is_last    // for print last rmsnorm output to debug
                    );



template void launchRMSNorm( TensorWrapper<half>* decoder_out,  // [num_tokens, hidden_units]
    TensorWrapper<half>* decoder_residual,
    LayerNormWeight<half>& attn_norm_weight, // RMSNorm weights
    float eps,  // RMSNorm eps
    bool is_last    // for print last rmsnorm output to debug
    );