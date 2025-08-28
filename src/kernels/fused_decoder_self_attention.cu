#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_decoder_self_attention.h"

/*
bias + RoPE算子由于是0-63的数据进行配对旋转的，所以无法进行向量化的读写操作，因此这里没办法融合这两个算子

这个算子融合的是:从RoPE的输出开始一直到乘以矩阵v结束
*/
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
    static __shared__ T warpsum[64];    // 这里使用static是因为这个函数是device端的，需要再编译器进行确定大小
    val = warpReduceSum(val);   // 计算这个block中的每个warp的sum
    if(lean_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    // 准备下一次warp reducec的数据
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
    static __shared__ T warpsum[64];    // 这里使用static是因为这个函数是device端的，需要再编译器进行确定大小
    val = warpReduceMax(val);   // 计算这个block中的每个warp的sum
    if(lean_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    // 准备下一次warp reducec的数据
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceMax(warp_val);
}



/*
// block and thread allocation
// 1 block -> head size，后续可改进为1 warp -> 1 head size or 1 block -> multi head size
// 1 grid -> bs * num heads
// q; input vec [bs, q num heads, 1, head size]
// k; input vec [bs, kv num heads, 1, head size]
// v; input vec [bs, num heads, 1, head size]
// k_cache; output,[bs, kv num heads, max_seq_len, head size] from prompt phase
// v_cache; output,[bs, num heads, max_seq_len, head size] from prompt phase
// mha_output; output vec [bs, q num heads, 1, head size]

*/
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
                                  const int step        // 是所有轮次的总step长度，kv cache的总长度
)
{
/*
    ConcatPastKVCache--->Broadcast--->QK gemv--->Scale--->Softamx--->QK*v gemv
    dim3 grid(head_num * batch_size);
    dim3 block(head_size);  // 从输出的角度来分配：[bs, head_num, 1, head_size]
    先进行第一步:计算qk gemv的数据偏移
*/
    int tid = threadIdx.x;
    int head_id = blockIdx.x % head_num;
    int batch_id = blockIdx.x / head_num;
    // 计算q,k的数据偏移
    int q_head_id = head_id;
    int q_batch_id = batch_id;
    int kv_head_id = q_batch_id / (head_num / kv_head_num);     // k与v是一致的
    int kv_batch_id = batch_id;
    // 计算对应的stride
    int q_batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    // 计算offseet
    int q_offset_vec = q_batch_id * q_batch_stride + q_head_id * head_stride + tid;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    // 计算k cache的偏移， kv的cache也是一致的[bs, kv num heads, max_seq_len, head size]        注意：layer_id在launch之前已经处理了偏移
    int cache_offset = kv_batch_id * max_seq_len * kv_head_num * head_size +
                       kv_head_id * max_seq_len * head_size +                                 // 没有计算max_seq_len维度的偏移，也就是seq_id。因为这个id，将在后面循环的时候才能获取到
                       tid;

    // int setp_stride = head_size;
    float scale = rsqrt(float(head_size));



    // 将qk指针以及kv cache指针转化为向量化的数据类型
    const int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    Vec_t* vec_q_mem = reinterpret_cast<Vec_t*>(q);
    Vec_t* vec_k_mem = reinterpret_cast<Vec_t*>(k);
    Vec_t* vec_k_cache_mem = reinterpret_cast<Vec_t*>(k_cache);

/*
    一个线程处理一个head_size的数据，也就是一个线程从q的head_size维度取出一个向量化的数据
    然后进行标量计算
    其中每个标量乘以k中的一行并求和得到一个qk gemv的结果
    然后再做scale

    这里需要注意的一点就是，
        1.因为这里是一个block来处理一个head_size的数据，所以这把q[head_size]放在shared memory
            可以让这个block中的所有线程进行复用。虽然每个线程取得数据不一样，但是可以提高缓存命中率
        2. 这里是一个block要计算得到一个qk gemv得结果，所以q[head_size]会和k[step, head_size]做乘法
            然后各自累加。所以这里是一个block需要处理step次dot和累加，然后结果放在shared memory上，以便之后做softmax（一个block）
        
        3. 这里如果后续改进，可以让一个block来处理多行k的乘法，此时就可以复用到q
        ....这里其实是有疑惑的，觉得就算是这样，也不会有复用。因为数据q就在本地的寄存器里面

    这里实现的是naive版本的，在后续优化中，可以用多个block来并行的计算这step步乘法
*/
    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    // 取出数据
    Vec_t vec_q = zero_f4;
    Vec_t vec_k = zero_f4; // 用于存储取出来的向量
    if(tid < head_size / vec_size){ // 除法的运算比乘法慢，这里可以改成乘法， 确保不会越界
        vec_q = vec_q_mem[q_offset_vec];
        vec_k = vec_k_mem[k_offset_vec];
    }
    // 每个线程执行运算，并将数据保存在dynamic shared memory上，供后面做reduce
    extern __shared__ char sqk[];   // 一个char一个字节，4个char就是一个float
    T* sq_scalar = reinterpret_cast<T*>(sqk);
    T* attn_score = reinterpret_cast<T*>(sq_scalar + head_size);

    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);
    Vec_t vec_qk = zero_f4;
    for(int iter = 0; iter < step; iter++){ // 一个block执行step次计算，得到输出的haed_size个数据
        if(tid < head_size / vec_size){// 只有这些线程需要处理数据
            Vec_t vec_k_for_qk;
            if(iter < step-1){
                vec_k_for_qk = vec_k_cache_mem[cache_offset + iter * head_size];
            }
            else if(iter == step - 1){  // 更新k cache 同时取vec_k_for_qk从k_vec中
                vec_k_cache_mem[cache_offset + iter * head_size] = vec_k;
                vec_k_for_qk = vec_k;
            }
            // 开始计算
            vec_qk.x = vec_q.x * vec_k_for_qk.x * scale_f4.x;
            vec_qk.y = vec_q.y * vec_k_for_qk.y * scale_f4.y;
            vec_qk.z = vec_q.z * vec_k_for_qk.z * scale_f4.z;
            vec_qk.w = vec_q.w * vec_k_for_qk.w * scale_f4.w;
        }// 这里只对需要处理的线程数据做了计算，还有一些线程没有处理计算，所以需要将这些线程赋初始值为0,否则，在后面做reduce的时候会因为垃圾值而影响结果
        // 可以在这里使用else {}来初始化这些值。但是这样会照成warp divergence。
        // 所以我们还有的方案就是在声明变量的时候对这些变量赋初始值。

        // 接下来需要将这个线程处理的数据求和，然后再做block的reduce，就得到一个两个向量点乘的结果
        T qk_acc = vec_qk.x + vec_qk.y + vec_qk.z + vec_qk.w;
        // 求和的记过要存在dynamic shared memory上
        T logits = blockReduceSum(qk_acc);
        if(tid == 0){
            attn_score[iter] = logits;     
        }
        __syncthreads();
    }
/*
    qk gemv计算结束
    vec_q, vec_k中存储的数据是从q,k中读取的向量化诗句
    attn_score是softmax的输入，现在attn_score里面存的数据还是logits
    attn_score:     [bs, num_heads, 1, stpe]
    接下来就要计算softmax了
*/
    // 先计算attn_score的max
    // 只有step个线程需要参与reduec，其余线程的值需要赋值为0，否则可能影响reduce结果
    T local_logits = tid < step ? (T)attn_score[tid] : 0;
    __shared__ float row_max, fenmu;    // 分母与max需要放在shared上
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
    // 计算softmax并写回
    if(tid < step){
        attn_score[tid] = (T)(fenzi / fenmu);
    }
    __syncthreads();

/*
    下面开始做atten_score * v
    attn_score是在shared_memory上，[bs, head_num, 1, step]
    v是在global memory上; [bs, head_num, 1, head_size]  step是这个token的生成
    v_cache是在global_memory上： [bs, head_num, max_seq_len, head_szie]：   这里面缓存了step-1个数据，也即是0~(step-2)

    计算的思路有前面的略有区别:
        因为这里的attn_score和 v_cache+v的维度正好是可以一行乘以一列进行计算的
        但是呢，v_cache+v的内存数据排布式行排布的，然后我们按照一行乘以一列进行计算的话
        就是会以列来访问数据。但是GPU在加载数据的时候，是按照行来缓存数据的
        一次GPU的内存事务访问只能加载128bit的数据，也就是一次内存事务可以取一个float4
        在加载的时候，会先去L2 Cache中找数据，如果没有，就是Cache miss，此时就会去global memory
        中找数据，然后将数据缓存到L2 Cache中，然后再将数据给L1 Cache,然后GPU才能获取到数据
        但是这一次取得flaot4数据只有一个float是我们需要的，所以这一次内存事务的利用率只有25%。
        然后下一次访问下一列的数据的时候，还是会出现cache miss，所以又回去global拿数据
        所以这样就会出现大量的cache miss
    
    因此，我们为了避免大量的cache miss，我们就不能按照一行乘以一列的思路来计算

    所以计算思路是：   v: [a, b, c, d]
                       m:  [e   i   M
                            f   j   N
                            g   k   o
                            h   l   p]
                    v @ m =   [a*e + b*f + c*g + d*h,
                             a*i + b*j + c*k + d*l,
                             a*m + b*N + c*o + d*p]
                    由此可以发现：可以先用向量的第一个元素乘以矩阵的第一行得到中间结果1
                                    再用向量的第二个元素乘以矩阵的第二行得到中间结果2
                                    再用向量的第三个元素乘以矩阵的第三行得到中间结果3
                                    再用向量的第四个元素乘以矩阵的第四行得到中间结果4
                                    最后再将中间结果全部相加即可 得到GEMV
    下面开始做atten_score * v
    attn_score是在shared_memory上的标量指针，[bs, head_num, 1, step]
    v是在global memory上; [bs, head_num, 1, head_size]  step是这个token的生成
    v_cache是在global_memory上： [bs, head_num, max_seq_len, head_szie]：   这里面缓存了step-1个数据，也即是0~(step-2)
*/

    // 转化v与attn_socre指针
    // 不对shared memory做向量化的读写，因为可能发生bank conflict而且需要确保数据对齐
    // Vec_t* vec_attn_score = reinterpret_cast<Vec_t*>(attn_score);
    Vec_t* vec_v_mem = reinterpret_cast<Vec_t*>(v);     // v; input vec [bs, num heads, 1, head size]
    Vec_t* vec_v_cache_mem = reinterpret_cast<Vec_t*>(v_cache);
    Vec_t vec_v = vec_v_mem[k_offset_vec];
    if(tid < head_size / vec_size){ // 同样是只有这些数据需要处理
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f);  // 初始化，作为中间结果
        // 每个线程处理一行(head_size)中的vec_size个数据.所有参与处理的线程就能够处理完一行(head_size)个数据
        // 然后这样就保证了数据是按行来访问计算的，只有tid=0或大于cache line的线程在获取数据的时候，会cache miss
        // 然后cache miss的线程之后会加载cache,然后这个线程附近的线程都不会cache miss
        for(int iter = 0; iter < step; iter++){
            Vec_t vec_v_for_qkv = vec_v_cache_mem[cache_offset + iter * head_size];
            if(iter == step-1){ // 更新v cache
                vec_v_cache_mem[cache_offset + iter * head_size] = vec_v;
                vec_v_for_qkv = vec_v;    // k的offset和v一致
            }
            // 计算
            O.x += vec_v_for_qkv.x * attn_score[iter];
            O.y += vec_v_for_qkv.y * attn_score[iter];
            O.z += vec_v_for_qkv.z * attn_score[iter];
            O.w += vec_v_for_qkv.w * attn_score[iter];
        }
        // 写回数据    mha_output,    // q; input vec [bs, q num heads, 1, head size]
        reinterpret_cast<Vec_t*>(mha_output)[q_offset_vec] = O;
    }
}




template <>
__global__ void masked_MHA_kernel(half* q, // q; input vec [bs, q num heads, 1, head size]
                                  half* k, // k; input vec [bs, kv num heads, 1, head size]
                                  half* v, // v; input vec [bs, num heads, 1, head size]
                                  half* qkv_bias,  // [(q num heads + 2*kv_head_nums), head size]
                                  half* k_cache,   // k_cache; output,[bs, kv num heads, max_seq_len, head size] from prompt phase
                                  half* v_cache,   // v_cache; output,[bs, num heads, max_seq_len, head size] from prompt phase
                                  half* mha_output,    // q; input vec [bs, q num heads, 1, head size]
                                  const int batch_size,
                                  const int head_num,
                                  const int kv_head_num,
                                  const int max_seq_len,
                                  const int head_size,
                                  const int step        // 是所有轮次的总step长度，kv cache的总长度
)
{
    // 暂时没有实现
}






template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,      //qkv; input vec [bs, (q_num_heads+2*kv_head_num), head_size]
                            BaseWeight<T>& qkv,          // qkv_bias: input vec [(q_nums_head+2*kv_head_num), head_size]
                            TensorWrapper<int>* layer_id,     // [layers]
                            TensorWrapper<T>* k_cache,      // [nums_layers, bs, kv_head_num, max_seq_len, head_size]  这里的step是当前缓存的总tokens
                            TensorWrapper<T>* v_cache,      // 
                            TensorWrapper<bool>* finished,  // [bs]用于记录每个seq是否完成推理
                            TensorWrapper<int>* step,       // 当前的step数,
                            TensorWrapper<T>* mha_output,   // 同q [bs, q_head_num, 1, head_size]
                            LLaMAAttentionStaticParams& static_params)  // RoPE的相关配置参数
{
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int head_num = qkv_head_num - 2 * kv_head_num;
    const int max_seq_len = k_cache->shape[3];
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();        // 这里需要注意一下，获取的是当前step数
    const int layer = layer_id->getVal();
    const int layer_offset = layer * batch_size * max_seq_len * head_size;
    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);  // 这里主要是用来存q和q*k^T之后的logits
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + (batch_size * head_num * head_size);
    T* v = qkv_data + (batch_size * (head_num + kv_head_num) * head_size);

    int rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int max_position_embeddings = static_params.max_position_embeddings;
    bool use_dynamic_ntk = static_params.use_dynamic_ntk;

    dim3 grid(head_num * batch_size);
    dim3 block(head_size);  // 从输出的角度来分配：[bs, head_num, 1, head_size]
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


template void launchDecoderMaskedMHA(TensorWrapper<half>* qkv_buf,      //qkv; input vec [bs, (q_num_heads+2*kv_head_num), 1, head_size]
                            BaseWeight<half>& qkv,          // qkv_bias: input vec [(q_nums_head+2*kv_head_num), head_size]
                            TensorWrapper<int>* layer_id,     // [layers]
                            TensorWrapper<half>* k_cache,      // [nums_layers, bs, kv_head_num, max_seq_len, head_size]  这里的step是当前缓存的总tokens
                            TensorWrapper<half>* v_cache,      // 
                            TensorWrapper<bool>* finished,  // [bs]用于记录每个seq是否完成推理
                            TensorWrapper<int>* step,       // 当前的step数,
                            TensorWrapper<half>* mha_output,   // 同q [bs, q_head_num, 1, head_size]
                            LLaMAAttentionStaticParams& static_params);  // RoPE的相关配置参数