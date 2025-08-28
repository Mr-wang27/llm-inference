// launchAddFusedQKVBiasTransposeAndRoPE kernel can be used in prompt phase and launchRoPE kernel is used in token generation phase
// 1.add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
// QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].

// 2.For q and k, apply RoPE, then send to attention.

// 3.rebuild padding to do mha      llama使用的是MHA，没有使用GAQ,MQA

// input: qkv_buf : qkv continouns buf when no padding  
// shape = [num_tokens, qkv_head_num, head_size],
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size]
// ps: seqlen = max_q_len here


//output        q:[batch_size, seq_len, head_num, head_size]
//output        k:[batch_size, seq_len, kv_head_num, head_size]
//output        v:[batch_size, seq_len, kv_head_num, head_size]
//              QKV:[token_num, qkv_head_num, head_size]
//              qkv_bias:[qkv_head_num, head_size]
#include <math.h>
#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/qkv_bias_and_RoPE.h"

// RoPE:
// 计算RoPE的cos_sin
// freq指代的是mθ
inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    // zid 表示的是分母
    // rot_embed_dim = head_szie： 表示需要旋转的维度。
    // 每个freq值对应于zid = head size 维度上0 2 4 6...64计算
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim); // mθ
    return {cos(inv_freq), sin(inv_freq)};
}


inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    // coef = [cosmθ， sinmθ]
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}


template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf,
                                                   T* k_buf,
                                                   T* v_buf,
                                                   T* QKV,
                                                   const T *qkv_bias,   // 在llama2中的数据没有bias,所以可以不用该数据
                                                   const int *padding_offset,
                                                   const int *history_length,
                                                   const int *input_length, // actual length of each seq
                                                   const int batch_size,
                                                   const int seq_len,       // max_seq_len to pad to
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotary_embedding_base, // 10000
                                                   int max_position_embeddings, // 默认为2048
                                                   bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
{
    // 在llama中，head_size=128,旋转的位置是两两配对进行旋转的，但是在llama中，是第0号元素于第64号元素进行配对，第1号于第65号元素进行配对的
    // 所以不能进行向量化的存取
    int token_id = blockIdx.x;  // 传入的grid(num_token, head_size)
    int head_id = blockIdx.y;
                                // 传入的block(head_size)
    int tid = threadIdx.x;  // 在该blokc中的线程id
    int token_padding_offset = padding_offset[token_id];        // padding_offset:[batch_size, seq_len]---> batch_size*seq_len

    // 在没有padding的QK上进行RoPE,然后在写入的时候再进行padding和transpose
    // 1. 准备重建： 计算这个token在padding后的数据位置。然后在存储的时候进行padding
    // q:[batch_size, seq_len, head_num, head_size]
    // k:[batch_size, seq_len, kv_head_num, head_size]
    int dst_token_id = token_id + token_padding_offset; //padding_offset这个是

    int batch_id = dst_token_id / seq_len;      // 在处理的token的batch_id
    int local_token_id= dst_token_id % seq_len; // 这个seq中的第几个token

    // 2. bias add
    int qkv_head_num = head_num + kv_head_num;
        // 把q,k,v的数据从QKV中的对应位置取出来 QKV:[token_num, qkv_head_num, head_size]
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid;
    int k_id = token_id * qkv_head_num * head_size + head_num * head_size + head_id * head_size + tid;
    int v_id = token_id * qkv_head_num * head_size + head_num * head_size + kv_head_num * head_size + head_id * head_size + tid;
    //     // 然后把数据加上对应的bias，再写回到原来的数据上
    //     // 但是llama2中并没有在生成qkv的过程中使用bias,因此，这里并没有实际上的使用qkv_bias来加
    // float q = QKV[q_id];
    // float k = QKV[k_id];
    // float v = QKV[v_id];    // 取出v的数据
    // // qkv_bias:[qkv_head_num, head_size]
    // // 还要把qkv对应的bias位置取出来
    // int q_bias_id = head_id * head_size + tid;
    // int k_bias_id = head_num * head_size + head_id * head_size + tid;
    // int v_bias_id = head_num * head_size + kv_head_num * head_size + head_id * head_size + tid;
    // float q_bias = qkv_bias[q_bias_id];
    // float k_bias = qkv_bias[k_bias_id];
    // float v_bias = qkv_bias[v_bias_id];
    // 计算数据写回q,k,v的位置， 这里计算的是padding之后写回q:[batch_size, head_num, seq_len, head_size]
    int dst_q_id = batch_id * seq_len* head_num * head_size + 
                    head_id * seq_len * head_size +
                    local_token_id * head_size + tid;
    int dst_kv_id = batch_id * seq_len * head_num * head_size +
                    head_id * seq_len * head_size +
                    local_token_id * head_size +tid;    // 这里是将q,k,v写入到各自的buf里面，而且策略是一个blokc处理qkv三个的一个head_size的数据
    // if(head_id < kv_head_num){
    //     // GQA、MQA的kv_head_num < head_num.因此要判断head_id是否小于kv_head_num，防止越界
    //     // 将数据写入到QKV原位置上，以便后面做RoPE
    //     QKV[q_id] = k + k_bias;
    //     QKV[k_id] = v + v_bias;
    // }
    // // 将数据写入到q
    // QKV[v_id] = q + q_bias;

    // 3. RoPE
    const int cur_seq_history_len = history_length[batch_id];   // 这个样本的历史长度
    const int context_length = cur_seq_history_len + input_length[batch_id];
    // 多轮对话下要结合history length求得全局的cos 和 sin
    const int timestep = cur_seq_history_len + token_id;    // 这个token的位置，就是m
    // timestep为cos(m * theta)中的m
    if(tid >= rotary_embedding_dim / 2){        // 每个线程处理两个数据，ratoary_embedding_dim = head_size
        return ;    // 过滤线程，rotary_embedding_dim = head_size
    }   // tid = [0,1,2,...,63]

    float2 cos_sin = GetRoPEfreq(tid*2, rotary_embedding_dim, rotary_embedding_base, timestep);
    float2 q_rotate = GetRoPEres(QKV[q_id], QKV[q_id + head_size / 2], cos_sin);
    float2 k_rotate = GetRoPEres(QKV[k_id], QKV[k_id + head_size / 2], cos_sin);

    // RoPe编码的结果写回，这里没有做转置，是因为在linear kernel中，我们默认的输入是
    // q.shape=k.shape: [batch_size, head_num, seq_len, head_size]
    // 在linear中调用gemm的时候，控制传入的参数使其转置
    q_buf[dst_kv_id] = q_rotate.x;
    q_buf[dst_kv_id + head_size / 2] = q_rotate.y;
    if(head_id < kv_head_num){
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}                       

// 特化FP16算子
template <>
__global__ void add_fusedQKV_bias_transpose_kernel(half *q_buf,
                                                   half *k_buf,
                                                   half *v_buf,
                                                   half *QKV,
                                                   const half *qkv_bias,    // 在调用的时候需要传入该参数，但是传入一个全零参数即可
                                                   const int *padding_offset,   // created before qkv linear
                                                   const int *history_length,
                                                   const int *input_length, // actual length of each seq
                                                   const int batch_size,
                                                   const int seq_len,   // max_seq_len to pad
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotatry_embedding_base,
                                                   int max_position_embeddings,
                                                   bool use_dynamic_ntk /*placeholdeer for ntk RoPE*/)
{
    /*
        一个block处理qkv的head_size, 一个线程处理head_size中的一个数据
        output        q:[batch_size, head_num, seq_len,  head_size]
        output        k:[batch_size, kv_head_num, seq_len, head_size]
        output        v:[batch_size, kv_head_num, seq_len, head_size]
                    QKV:[token_num, qkv_head_num, head_size]
                    qkv_bias:[qkv_head_num, head_size]
        计算出该线程处理的数据位置
            计算位置的时候，要注意input_length

        计算出该线程写回的位置
            计算位置的时候，要注意padding_offset:[batch_size, seq_len]---> batch_size*seq_len
        将bias加到该位置上
        计算RoPE，并添加到该位置上
    */
    int token_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];

    // 计算该线程写回的位置
    int dst_token_id = token_id + token_padding_offset;
    int batch_id = dst_token_id / seq_len;  // seq_len is max_seq_len to pad
    int local_token_id = dst_token_id % seq_len;    // 是写回位置的locak_token_id
    
    // 取出数据
    int qkv_head_num = head_num + 2 * kv_head_num;
    int q_id = token_id * qkv_head_num * head_size + head_id + head_size + tid;
    int k_id = token_id * qkv_head_num * head_size + head_id + head_size + tid + head_num * head_size;
    int v_id = token_id * qkv_head_num * head_size + head_id + head_size + tid + head_num * head_size + kv_head_num * head_size;

    // // 取出bias的数据位置
    // int q_bias_id = head_id * head_size + tid;
    // int k_bias_id = head_id * head_size + tid + head_num * head_size;
    // int k_bias_id = head_id * head_size + tid + head_num * head_size + kv_head_num * head_size;

    // // 计算数据写回的位置， 这里计算两个数据，一个是写回到QKV中，一个是写回到q,k,v中
    // int QKV_dst_q_id = q_id;    // QKV的位置就从哪儿取出来，就写回哪儿
    int dst_q_id = batch_id * head_num * seq_len * head_size +
                    head_id * seq_len * head_size +
                    local_token_id * head_size + tid;
    int dst_kv_id = batch_size * kv_head_num + seq_len * head_size +
                    head_id * seq_len * head_size +
                    local_token_id * head_size + tid;

    // 我们这里将数据写回到QKV的位置上，方便后写做RoPE,然后再写回到dst_q_id与dst_k_id的位置上

    // QKV[q_id] += __hadd(QKV[q_id] + qkv_bias[q_bias_id]);
    // if(head_id < kv_head_num){
    //     // for GQA and MQA
    //     QKV[k_id] = __hadd(QKV[k_id], qkv_bias[k_bias_id]);
    //     QKV[v_id] = __hadd(QKV[v_id], qkv_bias[v_bias_id]);
    // }

    // 计算RoPE
    const int cur_seq_history_len = history_length[batch_id];
    const int context_length = cur_seq_history_len + input_length[batch_id];
    const int timestep = cur_seq_history_len + local_token_id;
    // 过滤线程，只有rotary_embedding_dim / 2 的线程参与计算RoPE
    if(tid >= rotary_embedding_dim / 2){
        return;
    }

    float2 cos_sin = GetRoPEfreq(tid*2, rotary_embedding_dim, rotatry_embedding_base, timestep);
    float2 q_rotate = GetRoPEres(__half2float(QKV[q_id]), __half2float(QKV[q_id + head_size / 2]), cos_sin);
    float2 k_rotate = GetRoPEres(__half2float(QKV[k_id]), __half2float(QKV[k_id + head_size / 2]), cos_sin);

    // 将数据写回
    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;
    if(head_id < kv_head_num){
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}





template<typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T> *q_buf, // output:[batch_size, head_num, seq_len, head_size]
                                           TensorWrapper<T> *k_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<T> *v_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<T> *QKV,   // input: [token_num, head_num + 2 * kv_head_num, head_size],
                                           BaseWeight<T>    &qkv,   //   qkv_bias:[qkv_head_num, head_size]
                                           TensorWrapper<int> *padding_offset,  // [batch_size, seq_len] 之前计算的padding_offset
                                           TensorWrapper<int> *history_length,  // [batch_size] 历史上下文长度，多轮对话，新的prompt需要连接之前的上下文
                                           TensorWrapper<int> *input_length,    // [batch_size] 要计算padding,这个是输入的所有句子的长度
                                           LLaMAAttentionStaticParams &params)
{
    int token_num = QKV->shape[0];  
    int qkv_head_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];
    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];
    int kv_head_num = k_buf->shape[1];

    dim3 grid(token_num, head_num); // 分配二维的block来处理四维数据。然后第二个维度为head_num是因为q的头数是一定大于等于kv_head_nnum的，所以可以保证覆盖所有数据
                                    // 用一个block来处理这个token的qkv中的一个头的数据，也就是一个block实际处理的是q的[head_size],k的[head_size],v的[head_size]
    dim3 block(head_size);  // 一个block可以处理一个head_size
    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(q_buf->data,
                                                           k_buf->data,
                                                           v_buf->data,
                                                           QKV->data,
                                                           qkv.bias,
                                                           padding_offset->data,
                                                           history_length->data,
                                                           input_length->data,
                                                           batch_size,
                                                           seq_len,
                                                           token_num,
                                                           head_num,
                                                           kv_head_num,
                                                           head_size,
                                                           params.rotary_embedding_dim,
                                                           params.rotary_embedding_base,
                                                           params.max_position_embeddings,
                                                           params.use_dynamic_ntk);
#ifdef PRINT_DATA
    printf("qkv bias and rope ctx kernel top2 result:\n");
    print_data<<<1,1>>>(q_buf->data);
#else
#endif
}



// 实例化模板

template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<float> *q_buf, // output:[batch_size, head_num, seq_len, head_size]
                                           TensorWrapper<float> *k_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<float> *v_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<float> *QKV,   // input: [token_num, head_num + 2 * kv_head_num, head_size],
                                           BaseWeight<float>    &qkv,   //   qkv_bias:[qkv_head_num, head_size]
                                           TensorWrapper<int> *padding_offset,  // [batch_size, seq_len] 之前计算的padding_offset
                                           TensorWrapper<int> *history_length,  // [batch_size] 历史上下文长度，多轮对话，新的prompt需要连接之前的上下文
                                           TensorWrapper<int> *input_length,    // [batch_size] 要计算padding,这个是输入的所有句子的长度
                                           LLaMAAttentionStaticParams &params);


template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<half> *q_buf, // output:[batch_size, head_num, seq_len, head_size]
                                           TensorWrapper<half> *k_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<half> *v_buf, // output:[batch_size, kv_head_num, seq_len, head_size]
                                           TensorWrapper<half> *QKV,   // input: [token_num, head_num + 2 * kv_head_num, head_size],
                                           BaseWeight<half>    &qkv,   //   qkv_bias:[qkv_head_num, head_size]
                                           TensorWrapper<int> *padding_offset,  // [batch_size, seq_len] 之前计算的padding_offset
                                           TensorWrapper<int> *history_length,  // [batch_size] 历史上下文长度，多轮对话，新的prompt需要连接之前的上下文
                                           TensorWrapper<int> *input_length,    // [batch_size] 要计算padding,这个是输入的所有句子的长度
                                           LLaMAAttentionStaticParams &params);



// self decoder阶段调用该kernel, 上面的是context decoder调用
template<typename T>
__global__ void rope_kernel_for_self_decoder(T *q,
                                             T *k,
                                             const int batch_size,
                                             const int head_num,    // q的head_num
                                             const int kv_head_num,
                                             const int head_size,
                                             const int step,    // 直接传入上面求得的timestep
                                             int rotary_embedding_dim,
                                             float rotary_embedding_base)
{
    /*    
        dim3 grid(head_num, batch_size); // 这里注意分配的维度
        dim3 block(head_size);
        qkv_buf: [bs, qkv_head_num, head_size]
        q\k:[bs, q_head_num, head_size]  seq_len=1：所以省略了.原本应该是[bs, q_head_num, seq_len, head_size]
        写回的数据也是这个
        */
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;

    int kv_head_id = q_head_id / (head_num / kv_head_num);    // for MQA\GQA
    int kv_batch_id = q_batch_id; // 这两个对MHA来说就是相等的，但是对MHA,GQA来说就不是

    int tid = threadIdx.x;

    int batch_stride = head_num * head_size;
    int head_stride = head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid; //就是q_id
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;  // k_id，取出k的位置

    // 不做bias，直接做RoPE， 过滤线程
    if(tid >= rotary_embedding_dim / 2){
        return;
    }

    // RoPE
    float k_reg = k[k_offset];
    float k_rotate_reg = k[k_offset + head_size / 2];
    float q_reg = q[q_offset];
    float q_rotate_reg = q[q_offset + head_size / 2];
    // 这里传入的step-1需要注意
    // 因为step代表的是当前句子的总长度:历史上下文+当前轮次的query+当前轮次已生成的token数。
    // 例如： 历史上下文：20个token, 当前轮次query: 8个token, 以及生成了3个token, 此时step = 20 + 8 + 3 = 31。也代表着这在生成第4个token。但是第四个token还没有完成生成
    // 而此时在self decoder阶段，输入是一个token, 生成的q、k均为[batch_size, head_num, seq_len(1), head_size]-->[bs, head_num, head_size].因此要对这个q、k进行旋转编码操作，实际上这个token的index是step-1，即在30的位置上
    float2 cos_sin = GetRoPEfreq(tid*2, rotary_embedding_dim, rotary_embedding_base, step-1);
    float2 q_rotate = GetRoPEres(q_reg, q_rotate_reg, cos_sin);
    // float2 k_rotate = GetRoPEres(k_reg, k_rotate_reg, cos_sin);
    float2 k_rotate = make_float2(0, 0);
    k_rotate.x = cos_sin.x * k_reg - cos_sin.y * k_rotate_reg;
    k_rotate.y = cos_sin.x * k_rotate_reg + cos_sin.y * k_reg;

    // 写回
    q[q_offset] = q_rotate.x;
    q[q_offset + head_size / 2] = q_rotate.y;
    k[k_offset] = k_rotate.x;
    k[k_offset + head_size / 2] = k_rotate.y;
}

// 没有实现half版本的
template<>
__global__ void rope_kernel_for_self_decoder(half *q,
                                             half *k,
                                             const int batch_size,
                                             const int head_num,    // q的head_num
                                             const int kv_head_num,
                                             const int head_size,
                                             const int step,    // 直接传入上面求得的timestep
                                             int rotary_embedding_dim,
                                             float rotary_embedding_base)
{


}









// 这个self decoder只适用于llama2
// 这里qkv_buf把seq_len的维度省略了，因为seq_len = 1
// qkv_buf: [batch_size, qkv_head_num, head_size]
template<typename T>
void launchRoPE(TensorWrapper<T> *qkv_buf,  // [batch_size, 1, 3*head_num, head_size]
                TensorWrapper<int> *step,     // 考虑了多轮上下文的timestep
                LLaMAAttentionStaticParams& static_params)
{
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int head_num = qkv_head_num / 3;    // 只适用于MHA
    const int kv_head_num = head_num;
    const int head_size = qkv_buf->shape[2];

    // 这里不做检查
    // LLM_CHECK(batch_size == 1);                     // 这里也是，硬编码写死了，只适用于llama2
    // LLM_CHECK(qkv_head_num == 96);
    // LLM_CHECK(head_size == 128);

    // 获取setp, 因为在self decoder中，需要用step来衡量这个句子长度，衡量seq_len这个维度，
    // 所以这里要传入一个setp tensorwrapper。比如step可以是：（1， 3， 13， 20...）: 表示当前的历史的句子的长度  
    const int cur_step = step->getVal();
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;

    int rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int max_position_embeddings = static_params.max_position_embeddings;
    dim3 grid(head_num, batch_size); // 这里注意分配的维度
    dim3 block(head_size);
    rope_kernel_for_self_decoder<T><<<grid, block>>>(q, // [bs, q_head_num, head_size]  seq_len=1：所以省略了
                                                     k,
                                                     batch_size,
                                                     head_num,      // q的head_num
                                                     kv_head_num,
                                                     head_size,
                                                     cur_step,
                                                     rotary_embedding_dim,
                                                     rotary_embedding_base);
#ifdef PRINT_DATA
    printf("qkv bias and rope self kernel top2 result: \n");
    print_data<<<1, 1>>>(q);
#else
#endif
}




template void launchRoPE(TensorWrapper<float> *qkv_buf,  // [batch_size, 1, 3*head_num, head_size]
                TensorWrapper<int> *step,     // 考虑了多轮上下文的timestep
                LLaMAAttentionStaticParams& static_params);


template void launchRoPE(TensorWrapper<half> *qkv_buf,  // [batch_size, 1, 3*head_num, head_size]
                TensorWrapper<int> *step,     // 考虑了多轮上下文的timestep
                LLaMAAttentionStaticParams& static_params);