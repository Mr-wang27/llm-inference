# 1.EmbeddingFuncotr
第一个kernel
template <typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [token num]
                          TensorWrapper<T>* output,         // FP32 [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T>* embed_table); // FP32 [vocal_size, hidden_size]

- 输入的维度是[token num]
    最初的输入的句子的维度就是一维的，所有history进行拼接成一个长句子，输入到tokenizer中进行数据id转换，得到[token num]数据的信息
    但是在最初的时候，通过round(对话轮数)记录了有多少条句子，也就是bs维度。然后没条句子也有记录每条句子的输出长度。这个长度具体是怎么获取的还不知道
    但是这些信息是能够知道的
- 所以输出的维度信息是[token_num, q_hidden_size]

# 2. Cal_padding_offset
// shape:
    //seq_lengths:[batch size]
    //cum_seqlens:[batch size + 1],first ele is 0
    //padding_offset:[batch size * max q len]
// note: the point is to calc padding offset and cum offset
void launchCalPaddingoffset(TensorWrapper<int>* padding_offset,        // 输出
                            TensorWrapper<int>* cum_seqlens,        // 输出：累积句子长度
                            TensorWarpper<int>* input_lens)     // 输入: 每个句子的长度

__gloabl__ void CalPaddingoffset(int*           padding_offset,         // padding_offset:[batch_size * max_q_len],但是只有前token_nums个数据被填充了
                                 int*           cum_seqlens,
                                 const int*     input_lengths,
                                 const int      batch_size,
                                 const int      max_q_len)
这个kernel就是做padding前计算每个token的padding offset。用于后面去除padding计算



# 3. RMSNormal
1. 输入的维度是[num_tokens q_hidden_unit]，输出也是这个维度
2. RMSNormal的缩放维度是[q_hidden_unit ], 因为RMSNormal是对最后一个维度进行归一化，对每个token实行归一化。
3. RMSNornal的计算流程：
    - 先求一个token的所有数据的平方
    - 然后求所有数据平方后的和
    - 然后求均值
    - 然后均值加上epsilon
    - 然后求平方根的倒数
    - 然后于这个token的原数据相乘，在和scale相乘

3. 实现RMSNormal的思路是：
    - 每个block处理一个token，因此block数量为num_tokens
    - 用(q_hidden_unit/4, 1024)个线程来处理每个token. 这里的1024是因为很多GPU的线程数超过1024之后，就不支持了或者性能不好
    - 然后在每个block上处理的时候，会涉及到reduce操作，所以用warp与block的reduce
    - 

# 4.BUildMask kernel
为什么需要mask呢？
- 首先在训练阶段，我们知道，输入的是一整个完整的label。需要使用mask将后面计算的attn给mask掉。
- 然后在自回归推理的时候，因为输入的是一个token进行推理，输入的这一个token是句子开始的那个token,所以在推理的时候，
    天然的就看不到后面生成的token。所以在自回归推理的时候，就不需要mask
- 但是呢，还有一种全量推理：context decoder。输入的不是一个句子开始的token,而是用户输入的一段话，也就是一个prompt
    此时，在推理的时候，就需要在计算的过程中，就和训练类似，需要让前面的token无法看到未来的token进行计算。
    因此也需要一个mask来避免看到未来的token.
    所以，全量推理在生成第一个token的时候，需要用mask来做这样的一件事情。然后再第二个token生成以及之后的token生成的时候，就不需要
    这样mask了，就和前面的自回归推理类似了。



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