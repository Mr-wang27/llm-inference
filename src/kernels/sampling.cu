#include <iostream>
#include "src/kernels/sampling.h"

/*

    将topK的输出，需要注意的是，这里输出的topK是模型的logist
    按照常理来说，需要先将这些logist计算出softmax值之后再进行topk的选取
    但是softmax是递增的函数，因此logist值大的，其softmax值也大
    所以先进topk选出最大的k个，同样能够找到对应的id。
    而且这样调换顺序，可以让softmax少计算很多数值。因为如果先进行softmax
    那么就要计算vocab_size个数据的softmax
    而现在只需要计算topK个数据的softamx。推理性能更佳

    
    按理说对着topK个数据的采样，应该也要计算出softmax然后再进行采样
    但是这里是对着topK个数据的采样添加了一定的随机性。通过添加这样的随机性，可以
    让模型的输出更加多样化
    因此这里的采样策略其实是：
        将topK个数据的logist计算出softmax的分子，然后以softmax的分母作为基准
        通过随机数与分母相乘，得到一个数据，看当前这个数据落在哪一个softmax分子上，我们就采样那个分子
        这里如果分子大的数据，其softmax数值也会大一些，然后在这里采样的过程中，采样到它的概率也会相应的大一些
        因此是合理的


*/


template <typename T>
__global__ void SamplingKernel(int* topk_id,    // [bs, K]
                               T* topk_values,  // [bs, k]
                               int* output_id,  // [bs]
                               int* seqlen,     // [bs]
                               bool* is_finished,   // [bs]
                               int K,
                               int rand_num,    // step， 种子
                               int end_id,
                               int vocab_size)
{
/*
    每个block处理一个bs的数据
    dim3 grid(batch_size);  // 从数据的输入角度分配
    dim3 block(K);
*/

    // 如果bs中的某一个数据已经finished了，就没必要进行下面的采样了
    if(is_finished[blockIdx.x]){
        return;
    }
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    int batch_offset = batch_id * K;
    int data_offset = batch_id * K + tid;
    T max = topk_values[batch_id];      // 最大的数据在topK的第一位
    topk_values[data_offset] = expf(topk_values[data_offset] - max);    // inplace

    // 接下来要求一个sum作为softmax分母
    __shared__ float sum;
    sum = (T)0.0f;
    __shared__ float threadhold; // 采样的阈值， threadhold = sum * rand_num;
    if(tid == 0){   // 每个block的0号线程进行求和sum，串行的block reduce
        for(int idx = 0; idx < K; idx++){
            sum += (float)topk_values[batch_offset + idx];
        }

        // 进行随机化的采样
        curandState_t state;    
        // curand_init API only support ulonglong data type
        curand_init( (unsigned long long)rand_num, (unsigned long long)batch_id, (unsigned long long)0, &state);
        threadhold = (float)curand_uniform(&state) * sum;   // 一个block的随机数

        // 初始化一个基础的output_id,防止随机数偏移，没有进入到下面的for循环中进行采样
        // 初始化为最大值的id
        output_id[batch_id] = topk_id[batch_id * K];
        for(int i = 0; i < K; i++){
            threadhold = threadhold - (float)topk_values[batch_offset + i];
            if(threadhold <= 0){
                output_id[batch_id] = topk_id[batch_offset + i] % vocab_size;   // 这里对vocab_size求余是一种防御性的写法，照理说是不会大于vocab_size的
                break;
            }
        }
        // 更新seqlen和is_finished
        seqlen[batch_id] = is_finished[batch_id] ? seqlen[batch_id] : seqlen[batch_id]+1;
        is_finished[batch_id] = (output_id[batch_id] == end_id);
    }
}




template <typename T> 
void launchSampling(TensorWrapper<int>* topk_id,  // [bs, K]
                    TensorWrapper<T>* topk_values,  // [bs, K]
                    TensorWrapper<int>* seqlen, // [bs]
                    TensorWrapper<bool>* is_finished,   // [bs]
                    TensorWrapper<int>* output_id,  // [bs]
                    IntDict& params)
{
    int batch_size = topk_id->shape[0];
    int K = topk_id->shape[1];
    int vocab_size = params["vocab_size"];
    int step = params["step"];                      // 当前推理的toekn步数，将这个数据作为随机数生成的种子
    int end_id = params["end_id"];                  // 生成的终止token_id

    dim3 grid(batch_size);                          // 从数据的输入角度分配
    dim3 block(K);                                  // K is samall, so directly allocate K threads is enough
    SamplingKernel<T><<<grid, block>>>(topk_id->data,
                                       topk_values->data,
                                       output_id->data,
                                       seqlen->data,
                                       is_finished->data,
                                       K,
                                       step,
                                       end_id,
                                       vocab_size);
}



template void launchSampling(TensorWrapper<int>* topk_id,  // [bs, K]
                    TensorWrapper<float>* topk_values,  // [bs, K]
                    TensorWrapper<int>* seqlen, // [bs]
                    TensorWrapper<bool>* is_finished,   // [bs]
                    TensorWrapper<int>* output_id,  // [bs]
                    IntDict& params);



template void launchSampling(TensorWrapper<int>* topk_id,  // [bs, K]
                    TensorWrapper<half>* topk_values,  // [bs, K]
                    TensorWrapper<int>* seqlen, // [bs]
                    TensorWrapper<bool>* is_finished,   // [bs]
                    TensorWrapper<int>* output_id,  // [bs]
                    IntDict& params);