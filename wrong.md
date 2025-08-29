# 
embedding kernel 精度？
**qkv_bias_and_RoPE精度？。**





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
    if(tid < head_size / vec_size){ 
        vec_q = vec_q_mem[q_offset_vec];
        vec_k = vec_k_mem[k_offset_vec];
    }
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
        }
        T qk_acc = vec_qk.x + vec_qk.y + vec_qk.z + vec_qk.w;
        T logits = blockReduceSum(qk_acc);
        if(tid == 0){
            attn_score[iter] = logits;     
        }
        __syncthreads();
    }
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
    Vec_t* vec_v_mem = reinterpret_cast<Vec_t*>(v);     // v; input vec [bs, num heads, 1, head size]
    Vec_t* vec_v_cache_mem = reinterpret_cast<Vec_t*>(v_cache);
    Vec_t vec_v = vec_v_mem[k_offset_vec];
    if(tid < head_size / vec_size){ // 同样是只有这些数据需要处理
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f);  // 初始化，作为中间结果
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
