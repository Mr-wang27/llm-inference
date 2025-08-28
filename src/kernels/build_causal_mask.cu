#include "src/kernels/build_causal_mask.h"

// 我觉得这里也没有考虑到pastKV
template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(T* mask,// [bs, max_q_lens, max_k_lens]
                                                        const int* q_lens,  // [bs] 当前轮次对话中每个seq的query长度
                                                        const int* k_lens, // [bs]  每个seq中的hitory长度，也就是pastKVcache长度+当前轮次对话的query长度
                                                        int max_q_len,  // max(q_lens)
                                                        int max_k_len)  // max(k_lens)
{
    // 每个block处理一个样本，也就是一个(max_q_lens, max_k_lens)
    // 然后每个block处理的数据写入到mask[bs, max_q_lens, max_k_lens]对应的位置上
    int tid = threadIdx.x;
    int qlen = q_lens[blockIdx.x]; // 行数，q
    int klen = k_lens[blockIdx.x]; // 列数，k
    
    // mask是指向的[bs, max_q_lens, max_k_lens]中的第一个元素
    // 每个block处理其中一个，因此需要将mask指针在各个bkcok中指向该block处理的那部分数据的开头
    mask += blockIdx.x * max_q_len * max_k_len;

    // 确保block中的线程数量能够覆盖掉所有的数据，需要使用block内的线程数为stride进行循环
    int offset = threadIdx.x;
    while(offset < max_q_len*max_k_len){    // 处理[max_q_len. max_k_len]
        // 确定当前线程处理的行号
        int q = offset / max_k_len;
        // 确定当前线程处理的列号
        int k = offset % max_k_len;
        // 确定在(q_len, k_len)这个矩阵上为1的位置。mask句子为[max_q_len, max_k_len].其余的位置为padding的位置，是需要被屏蔽的
        bool is_one = q < qlen && k < klen && k <= q + (klen - qlen) && k >= klen-qlen;
        // k >= klen - qlen	防止 query 看到 padding 或残留数据
        // 因为在第一次对话的时候，也是有bs这个维度进行计算，同时query也会进行padding
        // 此时query padding到max_q_len的时候，然后计算生成Q,K的时候
        // 也就会生成Q(bs,max_q_len,hidden_units), K(bs, max_q_len, hidden_units)
        // 此时在k中，也会存在padding的部分，也就是K转置后的后面几列(或者转置前的后几行)
        // 所以在存储K Cache的时候，这部分内容也是回存储的，也就是padding的K Cache也是回存储的
        // 所以为了避免看到这些padding的残留数据，需要k > klen-qlen
        

        // 将bool类型的is_one转为T类型，写入到对应的[max_q_len, max_k_len]
        mask[offset] = static_cast<T>(is_one);

        // stride
        offset += blockDim.x;
    }
}



template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask, // [bs, max_q_lens, max_k_lens]
                            TensorWrapper<int>* q_lens, // [bs]
                            TensorWrapper<int>* k_lens // [bs]        
)
{
    int batch_size = mask->shape[0];
    int max_q_len = mask->shape[1];
    int max_k_len = mask->shape[2];
    BuildCausalMasksConsideringContextPastKV<T><<<batch_size, 256>>>(mask->data, q_lens->data, k_lens->data, max_q_len, max_k_len);
}


template void launchBuildCausalMasks(TensorWrapper<float>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);

template void launchBuildCausalMasks(TensorWrapper<half>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);