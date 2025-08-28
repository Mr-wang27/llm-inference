# src/utils/tensor.h

1. template<typename T>        
    TensorWrapper<T>* as()

为什么要把Tensor类转换成tensorWrapper类呢？

2. static const std::unordered_map<Device, std::string> devicetring{       // "懒汉式" 单例常量表，该变量是一个， 在第一次调用该函数的时候构建，然后
            {CPU, "CPU"}, {CPU_PINNED, "CPU_PINNED"}, {GPU, "GPU"}};



# 不同模型的结构不同，其对于的权重形状啊，数据类型啊等等都不一样。
所以每个模型需要单独定义自己的每一个层的权重
不同的层的权重也不一样
所以正对每种不一样的层也需要定义一些自己的权重数据结构

所以在**weights**目录下设置**llama**文件夹，表示是llama这个模型的权重文件，该文件夹下存储的时llama模型的不同层的**权重数据结构**
因此定义一个头文件，在头文件中定义**模板权重类**

然后具体的kernel实现在**src/kernel**
在该文件夹下，每个kernel有两个文件组成，一个头文件，一个.cu文件。因为cpp文件不能出现cuda的语法，所以具体的kernel实现，以及kernel启动的语句都写在.cu文件中，然后在头文件中声明启动kernel的函数
之后当其他文件需要调用这个kernel的时候，只需要include这个头文件即可，

因为在编译的时候，会将.cu文件也进行编译，然后其他文件include了这个.cu文件对应的.h文件，那么在链接的阶段，就会去找到这个启动kernel的函数的定义，就会找到这个.cu文件编译的结果，然后进行链接，这样就可以完成整个过程。


# kernel中的文件结构安排
/*
    真正的原因在于：
        1. 模板函数调用处必须能看到模板定义，否则不会生成代码。
        2. cuda语法不能出现在.cpp文件中

    如果将kernel启动的模板函数定义在.cu文件(kernel.cu)中,然后使用一个头文件声明kernel启动函数: kernel.h
    那么当a.cpp文件#inlcue "kernel.h"的时候，因为模板函数的定义在kernel.cu中，并没有在kernel.h中，而且模板函数调用处必须能看到模板定义，才会生成代码
    而这个情况,a.cpp看不到模板函数的定义，所以不会生成代码，所以在链接的时候，需要链接kernel启动函数的时候，就会出现来链接错误，不能找到kernel函数启动的函数符号

    所以就必须在.cu文件中，显示的对kernel启动函数进行显示的实例化

    

    另一个方案：就是将kernel启动的模板函数定义在kernel.h中，kernel.cu文件中只定义kernel。但是这个方案是行不通的。
    原因在于:cuda语法不能出现在.cpp文件中。
    当a.cpp文件#include "kernel.h"，此时kernel启动的模板函数就会被复制到a.cpp文件中，而kernel.h中有kernel启动函数，该函数会用到cuda语法
    此时就会报错，因为cpp文件中不能出现cuda语法。

    所以只能按照最上面的方式进行，将kernel定义以及kernel启动函数定义在.cu文件中，然后在.cu文件中显示实例化模板
    然后再用.h文件声明kernel启动函数
    其他文件需要使用该kernel的时候，只需要include这个.h文件即可
*/




# 前面的linear kernel
做的是融合的QKVGemm
就是QKV通常有多头的，然后把多头的数据融合起来一起计算，一次性把所有头的数据全部计算出来




# RoPE


# allocate的设计
1. 为什么不直接使用mallc cudamalloc free cudafree?
    因为malloc和cudamalloc分配内存和显存的时候，一般情况下都需要通过系统调用（brk(),mmap()）进入OS内核态，然后再进行内存的分配。如果直接使用malloc或者cudamalloc的话，就会导致频繁的内存、显存申请，导致性能降低。
    free也是同理
    其次，由于在推理的场景下，大内存的申请与小内存的申请都会经常存在，此时在CPU端，大内存会通过mmap这个系统调用在匿名文件映射区分配内存，free的时候会直接进行free。这就是上面的一个示例情况。然后小内存的申请，会走brk()系统调用，然后在堆上进行内存分配，此时如果进行频繁的free与malloc就会导致堆上存在大量的内存碎片。
    所以我们设计这个allocte的目的有二：
    - 减少频繁malloc与free的次数
    - 减少内存碎片

1. Pytorch allocatro
    - Block: ptr, size, is_allocated, prev, next, 双向链表
    - Blockpool: set<block>
    - Malloc:
        - Find buf from pool(set)
        - if not, try to merge free blocks to a big block;(把一些block合并成一个大的block)
        - if not, try to release free blocks, then search if meet requirements(释放一些不需要的内存块，然后再看是否满足要求)
        - if not, try to split big block to clean up memory fragment(把一些分配的大的block进行切分，因为一些大的block被分配之后，可能并不会用到这个block的全部内存，此时就把这些没有用到的内存split下来，清除内存碎片)
        - if not, cudamalloc(最后，还是没有，就cudamalloc)



2. Our allocater
    - Block: ptr, size, is_allocated
    - Blockpool: map<int, vector < block >>
    - UnifyMalloc:  bigBlockPoolForBigBuf,  smallBlockPoolForSmallBuf
        - Find buf from pool(block vector)
        - if not, cudamalloc
    - UnifyFree:
        - Clear memory fragment (清理内存碎片， 其实就是用size记录一下每个小内存块的大小，然后当很多个小内存块的大小总和达到了一定的阈值之后，再统一归还给os)
        - if find ptr in blockpool, Set is_allocated=false, not return to os
        - if not, cudaFree
    
    这里将在CPU端分配内存与在GPU端分配显存利用allocator进行统一管理，
    UnifyMalloc与UnifyFree是暴露给外部的接口

    在Blockpool中，使用的是map<int, vector< block >>进行管理。key为int类型，表示的是不同的设备device,适用于多卡的场景，每个卡的内存需要适用单独的内存pool进行管理。

    同时在推理场景下，经常会涉及到大内存与小内存的分配，所以，这里创建了两个内存pool: bigBlockPoolForBigBuf与smallBlockPoolForSmallBuf

3. bigBlockPoolForBigBuf与smallBlockPoolForSmallBuf
    - bigBlockPoolForBigBuf的内存分配是不容易产生内存碎片的。
    - smallBlockPoolForSmallBuf的内存分配才是容易产生内存碎片的。所以对于小块内存的申请单独采用一个内存pool进行管理。目的是为了收集这些内存碎片的大小，以便在UnifyFree的时候，可以对这些内存碎片进行清理

4. 在调用了UnifyMalloc的时候，先到内存pool中去寻找是否有这次所申请的这么大的block，如果找到了就直接把这个block中的内存分配给它。如果没有找到，就调用malloc或cudamalloc，通过OS进行内存分配

5. 这个allocated并没有涉及一个CPU上的内存pool,因为在推理的场景下，一般用到CPU上的内存分配场景并不多，
    也就是：用户输入一个句子，然后申请一块buffer，就结束了。后面还需要再CPU上申请的内存就是一些小的参数。因此没有专门为CPU设计内存pool。而是直接适用malloc和free。
    当然，这也是封装再Unifymalloc和UnifyFree里面的。只是没有为CPU设计内存pool

    而GPU的显存就需要进行管理了，因为一次推理需要启动许多kernel，每个kernel都需要进行显存的分配，有大有小，因此需要进行管理。

6. 在分配内存，也就是在unifyMalloc()中，分配内存的时候，我们会将内存进行32bytes的对齐，目的是为了确保在kernel运行的时候的float4能够正确的进行向量化的读写！
   
   对于 cudaMalloc 本身，它返回的指针地址已经是 256 字节对齐的，所以通常没必要再额外手动对齐。





#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_decoder_self_attention.h"


template <typename T>
__device__ T warpReduceSum(T val){
    for(int mask = 16; mask > 0; mask >>= 1){  // 右移一位等于除以2    
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__device__ T warpReduceMax(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <typename T>
__device__ T blockReduceSum(T val)
{
    int tid = threadIdx.x;
    int warp_nums = (blockDim.x + 31) / 32;
    int warp_id = tid / 32;
    int lean_id = tid % 32;
    static __shared__ T warpsum[64];    
    val = warpReduceSum(val);   
    if(lean_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum(warp_val);
}


template <typename T>
__device__ T blockReduceMax(T val)
{
    int tid = threadIdx.x;
    int warp_nums = (blockDim.x + 31)/ 32;
    int warp_id = tid / 32;
    int lean_id = tid % 32;
    static __shared__ T warpsum[64];    
    val = warpReduceMax(val);   
    if(lean_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceMax(warp_val);
}



template <typename T>
__global__ void masked_MHA_kernel(T* q, // q; input vec [bs, q num heads, 1, head size]
                                  T* k, // k; input vec [bs, kv num heads, 1, head size]
                                  T* v, // v; input vec [bs, num heads, 1, head size]
                                  T* qkv_bias,  // [(q num heads + 2*kv_head_nums), head size]
                                  T* k_cache,   // k_cache; output,[bs, kv num heads, max_seq_len, head size] from prompt phase
                                  T* v_cache,   // v_cache; output,[bs, num heads, max_seq_len, head size] from prompt phase
                                  T* mha_output,    // q; input vec [bs, q num heads, 1, head size]
                                  const int batch_size,
                                  const int head_num,
                                  const int kv_head_num,
                                  const int max_seq_len,
                                  const int head_size,
                                  const int step        
)
{

    int tid = threadIdx.x;
    int head_id = blockIdx.x % head_num;
    int batch_id = blockIdx.x / head_num;
    int q_head_id = head_id;
    int q_batch_id = batch_id;
    int kv_head_id = q_batch_id / (head_num / kv_head_num);     
    int kv_batch_id = batch_id;
    int q_batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset_vec = q_batch_id * q_batch_stride + q_head_id * head_stride + tid;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    int cache_offset = kv_batch_id * max_seq_len * kv_head_num * head_size +
                       kv_head_id * max_seq_len * head_size +                                 
                       tid;

    float scale = rsqrt(float(head_size));



    const int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    Vec_t* vec_q_mem = reinterpret_cast<Vec_t*>(q);
    Vec_t* vec_k_mem = reinterpret_cast<Vec_t*>(k);
    Vec_t* vec_k_cache_mem = reinterpret_cast<Vec_t*>(k_cache);


    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    Vec_t vec_q = zero_f4;
    Vec_t vec_k = zero_f4; // 用于存储取出来的向量
    if(tid < head_size / vec_size){ // 除法的运算比乘法慢，这里可以改成乘法， 确保不会越界
        vec_q = vec_q_mem[q_offset_vec];
        vec_k = vec_k_mem[k_offset_vec];
    }
    extern __shared__ char sqk[];   // 一个char一个字节，4个char就是一个float
    T* sq_scalar = reinterpret_cast<T*>(sqk);
    T* attn_score = reinterpret_cast<T*>(sq_scalar + head_size);

    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);
    Vec_t vec_qk = zero_f4;
    for(int iter = 0; iter < step; iter++){ 
        if(tid < head_size / vec_size){
            Vec_t vec_k_for_qk;
            if(iter < step-1){
                vec_k_for_qk = vec_k_cache_mem[cache_offset + iter * head_size];
            }
            else if(iter == step - 1){  
                vec_k_cache_mem[cache_offset + iter * head_size] = vec_k;
                vec_k_for_qk = vec_k;
            }
            vec_qk.x = vec_q.x * vec_k_for_qk.x * scale_f4.x;
            vec_qk.y = vec_q.y * vec_k_for_qk.y * scale_f4.y;
            vec_qk.z = vec_q.z * vec_k_for_qk.z * scale_f4.z;
            vec_qk.w = vec_q.w * vec_k_for_qk.w * scale_f4.w;
        }

        T qk_acc = vec_qk.x + vec_qk.y + vec_qk.z + vec_qk.w;
        T logits = blockReduceSum(qk_acc);
        if(tid == 0){
            attn_score[iter] = logits;     
        }
        __syncthreads();
    }

    T local_logits = tid < step ? (T)attn_score[tid] : 0;
    __shared__ float row_max, fenmu;    
    T block_max = blockReduceMax(local_logits);
    if(tid == 0){
        row_max = block_max;
    }
    __syncthreads();
    T fenzi = expf(local_logits - row_max);

    T block_fenmu = blockReduceSum(fenzi);
    if(tid == 0){
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads();
    if(tid < step){
        attn_score[tid] = (T)(fenzi / fenmu);
    }
    __syncthreads();


    Vec_t* vec_v_mem = reinterpret_cast<Vec_t*>(v);     // v; input vec [bs, num heads, 1, head size]
    Vec_t* vec_v_cache_mem = reinterpret_cast<Vec_t*>(v_cache);
    Vec_t vec_v = vec_v_mem[k_offset_vec];
    if(tid < head_size / vec_size){ 
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f); 
        for(int iter = 0; iter < step; iter++){
            Vec_t vec_v_for_qkv = vec_v_cache_mem[cache_offset + iter * head_size];
            if(iter == step-1){ 
                vec_v_cache_mem[cache_offset + iter * head_size] = vec_v;
                vec_v_for_qkv = vec_v;   
            }
            O.x += vec_v_for_qkv.x * attn_score[iter];
            O.y += vec_v_for_qkv.y * attn_score[iter];
            O.z += vec_v_for_qkv.z * attn_score[iter];
            O.w += vec_v_for_qkv.w * attn_score[iter];
        }
        reinterpret_cast<Vec_t*>(mha_output)[q_offset_vec] = O;
    }
}


template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,      
                            BaseWeight<T>& qkv,         
                            TensorWrapper<int>* layer_id,     
                            TensorWrapper<T>* k_cache,     
                            TensorWrapper<T>* v_cache,      
                            TensorWrapper<bool>* finished,  
                            TensorWrapper<int>* step,       
                            TensorWrapper<T>* mha_output,  
                            LLaMAAttentionStaticParams& static_params)  
{
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int head_num = qkv_head_num - 2 * kv_head_num;
    const int max_seq_len = k_cache->shape[3];
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();        
    const int layer = layer_id->getVal();
    const int layer_offset = layer * batch_size * max_seq_len * head_size;
    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);  
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + (batch_size * head_num * head_size);
    T* v = qkv_data + (batch_size * (head_num + kv_head_num) * head_size);

    int rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int max_position_embeddings = static_params.max_position_embeddings;
    bool use_dynamic_ntk = static_params.use_dynamic_ntk;

    dim3 grid(head_num * batch_size);
    dim3 block(head_size);  
    masked_MHA_kernel<T><<<grid, block, smem_size_bytes>>>(q,
                                                           k,
                                                           v,
                                                           qkv.bias,
                                                           k_cache->data + layer_offset,
                                                           v_cache->data + layer_offset,
                                                           mha_output->data,
                                                           batch_size,
                                                           head_num,
                                                           kv_head_num,
                                                           max_seq_len,
                                                           head_size,
                                                           cur_step);
#ifdef PRINt_DATA
    printf("fused decoder self attn kernel top2 result:]n");
    print_data<<<1, 1>>>(mha_out->data, true);
#else
#endif
}





template void launchDecoderMaskedMHA(TensorWrapper<float>* qkv_buf,      //qkv; input vec [bs, (q_num_heads+2*kv_head_num), 1, head_size]
                            BaseWeight<float>& qkv,          // qkv_bias: input vec [(q_nums_head+2*kv_head_num), head_size]
                            TensorWrapper<int>* layer_id,     // [layers]
                            TensorWrapper<float>* k_cache,      // [nums_layers, bs, kv_head_num, max_seq_len, head_size]  这里的step是当前缓存的总tokens
                            TensorWrapper<float>* v_cache,      // 
                            TensorWrapper<bool>* finished,  // [bs]用于记录每个seq是否完成推理
                            TensorWrapper<int>* step,       // 当前的step数,
                            TensorWrapper<float>* mha_output,   // 同q [bs, q_head_num, 1, head_size]
                            LLaMAAttentionStaticParams& static_params);  // RoPE的相关配置参数

