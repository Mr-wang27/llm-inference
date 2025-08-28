#pragma once

struct LLaMAAttentionStaticParams {
    int   rotary_embedding_dim;     // RoPE应用维度，一般是head_dim/2
    float rotary_embedding_base;    // base，一般是10000
    int   max_position_embeddings;  // 最大位置，决定RoPE的缓存表最大支持多长序列
    bool  use_dynamic_ntk; // for dyn scaling rope      是否开始动态NTK(用于扩展RoPE支持更长序列)
};
/*
    静态参数：这些都是RoPE机制本身的固定配置，在模型训练、初始化、导出时就已经确定。基本不会在单次推理过程中改变
    特点：  全局唯一
            初始化一次即可
            跨batch共享
            静态缓存、可以离线准备RoPE角度表


use_dynamic_ntk： 原版的RoPE的理论最大序列长度受限于max_position_embedding(2048\4096)
                    如果推理是希望支持更长的上下文(如：32K. 128K)等，RoPE就不支持
                    动态NTK方法(NTK aware RoPE,参考Linear Scaling RoPE)是一种通过动态缩放角频率的方法，让RoPE支持更长的序列
                
                具体思路：在推理时，如果KV Cache操作原设计长度，就开启use_dynamic_ntk，动态调整频率

    
*/




// (RussWong)note: llama类模型里面动态改变的变量, 注意非全部必需
struct LLaMAAttentionDynParams {
    int batch_size;     // 当前推理的batch大小                                  在self decoder阶段，只有batch_size
    int num_tokens;     // 当前输入的token数(尤其streaming时重要)               在self decoder阶段，num_tokens为0       prefill阶段有num_tokens, 也有 batch_size
    int max_q_len;      // 本这次query的最大长度
    int max_k_len;      // 累积的KV cache最大长度
    int num_layers;     // 总层数、某些实现中需要知道
    bool is_ctx = false;    // 当前是否处于上下文拼接(ctx阶段)
};
/*
    动态参数：这是每次推理、每个batch实际的场景配置
            因为推理过程中的batch size、序列长度、层数、是否在做上下文扩展(Cache模式)都是动态变化的

    特点：  每个batch动态变化
            推理是按需设置
            决定RoPE使用的实际长度、batch处理维度等



     is_ctx — 标记是否处于上下文拼接阶段 (Context阶段)
        场景
            在自回归推理阶段分两种情况：

            初始上下文拼接阶段（is_ctx = true）

            一次性输入多个 token（比如用户 prompt）。

            Q、K、V 都是 num_tokens = prompt_len。

            需要对整段做 attention（比如从 0 到 prompt_len-1 的所有位置）。

            后续单步自回归阶段（is_ctx = false）

            只输入新生成的一个 token。

            Q 是 [batch_size, 1, head_dim]

            K、V 使用历史缓存，不重新计算历史 K、V，只计算当前。

            作用
            is_ctx 是一个标志位，告诉引擎当前是否是 prompt 拼接阶段：

            true：直接走 full attention，KV 全部 freshly 计算。

            false：只计算当前 token 的 K、V，并与历史 KV cache 做增量推理
*/



/*
    总体流程：
    就是说推理引擎初始化的时候利用静态参数计算一个全局的RoPE表，
    然后在后面推理生成第一个token以及后续的token的过程中利用通带参数决定取RoPE表中的那些元素对QK句子做旋转编码。
    这里的RoPE表示是计算的Cos\Sin结果，即cos(mθ)与sin(mθ)
    RoPE表的维度：(max_position_embeddings, rotary_embedding_dim)
                    这里max_position_embeddings起始就是max_seq_len(最长的序列长度)
                    rotary_embedding_dim就是每个头的head_dim//2
    存在于显存中常驻(FP16或FP32)

    在推理引擎初始化或者第一次需要使用RoPE的时候进行初始化。
    然后初始化得到RoPE表常驻在内存中，推理时：
        第一步推理(第一批示token)：
            batch_szie=2, num_token=1(第一个token), max_q_len=1, max_k_len=1
            取RoPE表中第0行的cos/sin计算旋转之后的q，k
        
        第50步推理的时候：
            batch_size=2, num_tokens=1(新生成的token), max_q_len=1, max_k_len=50(上下文已经缓存了49个token)
            从RoPE表中取出第49行的cos/sin，对Q(第50个token)和历史K(第0-49个token)按位置进行旋转






    */











//     #pragma once
// // 创建两个结构体，分别存储计算RoPE的静态参数于用于推理时的动态参数
// struct LLaMAAttentionStaticParams {
//     int rotary_embedding_dim;   // RoPE应用维度，一般时head_dim//2
//     int rotary_embedding_base;  // 一般为10000
//     int max_position_embeddings;    // // 序列的最大上下文长度，决定RoPE的缓存表最大支持多长序列
//     bool use_dynamic_ntk;
// };


// // llama类模型里面动态改变的变量，注意非全部必须
// struct LLaMAAttentionDynParams{
//     int batch_size; // 当前推理的batch大小
//     int num_tokens; // 当前输入的token数（尤其streaming时重要）
//     int max_q_len;  // 本次query的最大长度
//     int max_k_len;  // 累积的KV cache最大长度
//     int num_layers; // 总层数，某些实现中需要
//     bool is_ctx = false;    // 当前是否处于上下文拼接(ctx)阶段
// };


/*
    静态参数：这些都是RoPE机制本身的固定配置，在模型训练、初始化、导出时就已经确定。基本不会在单次推理过程中改变
    特点：  全局唯一
            初始化一次即可
            跨batch共享
            静态缓存、可以离线准备RoPE角度表
*/


/*
    动态参数：这是每次推理、每个batch实际的场景配置
            因为推理过程中的batch size、序列长度、层数、是否在做上下文扩展(Cache模式)都是动态变化的

    特点：  每个batch动态变化
            推理是按需设置
            决定RoPE使用的实际长度、batch处理维度等
*/



/*
    总体流程：
    就是说推理引擎初始化的时候利用静态参数计算一个全局的RoPE表，
    然后在后面推理生成第一个token以及后续的token的过程中利用通带参数决定取RoPE表中的那些元素对QK句子做旋转编码。
    这里的RoPE表示是计算的Cos\Sin结果，即cos(mθ)与sin(mθ)
    RoPE表的维度：(max_position_embeddings, rotary_embedding_dim)
                    这里max_position_embeddings起始就是max_seq_len(最长的序列长度)
                    rotary_embedding_dim就是每个头的head_dim//2
    存在于显存中常驻(FP16或FP32)

    在推理引擎初始化或者第一次需要使用RoPE的时候进行初始化。
    然后初始化得到RoPE表常驻在内存中，推理时：
        第一步推理(第一批示token)：
            batch_szie=2, num_token=1(第一个token), max_q_len=1, max_k_len=1
            取RoPE表中第0行的cos/sin计算旋转之后的q，k
        
        第50步推理的时候：
            batch_size=2, num_tokens=1(新生成的token), max_q_len=1, max_k_len=50(上下文已经缓存了49个token)
            从RoPE表中取出第49行的cos/sin，对Q(第50个token)和历史K(第0-49个token)按位置进行旋转






    */