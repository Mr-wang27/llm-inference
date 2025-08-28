#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/macro.h"

template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask, // [bs, max_q_lens, max_k_lens]
                            TensorWrapper<int>* q_lens, // [bs]
                            TensorWrapper<int>* k_lens // [bs]        
);




/*
q_lens维度是[bs]:
    举例：当前batch中有3个句子（bs=3）
        3个句子本次要padding生成5个token(有padding,比如原本分别生成3、4、5个token)
        所以q_lens为[3, 5, 4]

k_lens:当前batch中每个样本的key的总长度(context+query),维度是[bs]
    举例：第0个句子的上下文是10 tokens,第一个是8 tokens, 第二个是 12 tokens
        则，k_lens = [10+3, 8+5, 12+4]=[13, 13, 16] 
        后面加的数字是本次需要生成的key.与本次的query是对应的。即past context + current query

mask:最终生成的Mask, shape=[bs, max_q_len, max_k_len]
    max_q_len是所有样本中最大的q_len, mask_q_len=5
    max_k_len是所有样本中最大的k_len, mask_k_len=16
    mask.shape=[3, 5, 16]

这个kernel做的目标：
    对每个样本(bs=blockIdx.x)
        生成它的[q_len, k_len]的causal mask
        把它填进[max_q_len, max_k_len]里，（超出的部分是padding）

推理时的数据示例：
    bs = 2;
    样本1：q_len=3, k_len=7(历史有4个token,本次要生成3个)
    样本2：q_len=5, k_len=9(历史用4个token,本次要生成5个)
    则：q_len = [3, 5]
        k_len = [7, 9]
        max_q_len = 5;
        max_k_len = 9
        mask形状为 [2, 5, 9]
    mask对应关系：
        样本1 （blockIdx.x=0）
            q_lens[0] = 3, k_lens[0] = 7
            mask的第0块是[5, 9] 大小的矩阵(padding后的形状)
            只填有效的前[3, 7], 其余的是padding(未来的数据也在[3,7中]， 这里[3,4]是历史，需要全部能够看见，[3,4:7]是本次的atten,是一个下三角的cause mask)
            causal mask规则：
                Past KV:位置k in [0,4)是历史（context）
                当前query: 位置k in [4, 7)
                query索引 q in [0,3)
                q 只能看到 <= q 的query, 以及前面的context
        样本2：同理
    

还存在一个问题，每个bs中的k_lens长度是不同的，为什么也要padding到max_k_len。那么在计算attn的时候,也需要将K Cache取出来，然后做padding吗？
首先，K Cache存储的是所有历史token的k数据，维度是[bs, max_total_seq_len, head_dim]
这里的max_total_seq_len是当前bs中，最长的上下文的长度。对于那些没有这么长的上下文长度的样本，也会申请这么多的空间，只不过有一些空间没有用
这样做的目的是为了后面使用带batch维度的GEMM.
然后，怎么判断每个样本那些是有效的toekn(历史token+当前生成的token)呢?   依靠的就是这里每个k_len(历史token+当前query)


前面计算paddingoffset似乎也与这里相关


*/


