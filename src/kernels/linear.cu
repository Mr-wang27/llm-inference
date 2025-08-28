#include <iostream>
#include <fstream>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/linear.h"
/*
    // TODO: when abstracted weight class, replace T with class
    意思是如果未来，将抽象化(优化)权重类(weight class)时，将模板参数T替换为一个具体的类类型
    当前的代码中，模板类型为T，表示input、weight、output的类型。
    未来可以将weight抽象为一个更复杂的类，这个类不仅仅时一个包含数据的容器，
    可能还会包含一些操作、方法、属性(如初始化、更新等)，用于处理权重矩阵。
    此时将模板类替换为相关的类类型即可
    比如量化的权重，可能为了精度，在给出权重参数的时候，在推理时进行反量化权重推理，
    就需要对权重进行操作
*/

// all matmul cases:
// ctx qkv lienar: [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits] = {num_tokens, qkv_head_num,  head_size}
// ctx attn output linear: {num_tokens, head_num, head_size} * {q hidden units, q hidden units} = {num_tokens, q hidden units}
// self qkv linear: [bs, q hidden units] * [qhiddenunits, hiddenunits] = {bs, qkv_head_num,  head_size}}
// self attn output linear: {batch_size, q hidden_units} * [qhiddenunits, qhiddenunits] = [bs, q hiddenunits]
// lmhead linear: [bs, q hidden units] * [vocab size, q hiden units], need transpose B
// gate:[bs/token nums, q hidden units] * [q hidden units, inter size] = [bs/token nums, inter size]
// up:[bs/token nums, q hidden units] * [q hidden units, inter size] = [bs/token nums, inter size]
// fusedGateUpGemm: [bs/token nums, q hidden units] * [q hidden units, 2 * inter size] = [bs/token nums, 2, inter size]     在后面讲weight的时候，会将gate\up的linear权重再python层面拼接起来
// 这里将两个weight进行合并计算，除了能够提高矩阵乘的计算效率之外，还有内存方面的考虑。
/*
    前半部分是gate的权重，后半部分是up的权重
    如果不将其拼接乘一个，那么输出就会对应有两个，那么在申请空间的时候，就需要申请两个空间来存储这两个数据，而空间的申请如果直接在主机端用cudamalloc分配或者
    ，此时是需要通过OS切换到内核态对空间进行申请的。此时就需要走两次内核态到用户态的切换来完成空间申请，因此这是一笔开销
    如果合并了，那么就只需要申请一次内存即可完成
    另外，在项目的后期会自己区实现一个分配器
    另外就是后面SiLu的计算特性也是刚好支持将这两个数据拼接在一起的
    因为SiLu,需要计算[bs/token nums, 1:2, inter size]的数据全部计算sigmoid然后乘以自身
    然后再
*/
// down:[bs/token nums, inter size] * [q hidden units, inter size] = [bs/token nums, q hidden units]

template<typename T>
void launchLinearGemm(TensorWrapper<T> *input,
                      BaseWeight<T> &weight,
                      TensorWrapper<T> *output,
                      cublasWrapper *cublas_wrapper,
                      bool trans_a,
                      bool trans_b)
{
    // 不转置A,B的情况
    int Am = weight.shape[1];    // qh    A: weight   [hiddenunits, qhiddenunits]
    int Ak = weight.shape[0];    // h
    int Bk = input->shape[1];    // qh或qkv head nunm*heda size   B: input    [num_tokens, qhiddenunits]         [bs/token nums, qkv head nunm, heda size]
    int Bn = input->shape[0];    // nt  
    int Cm = output->shape[1];   // h 或qkv head nunm*heda size   C: output    [num_tokens, hiddenunits]          [bs/token nums, qkv head nunm, heda size]
    int Cn = output->shape[0];   // nt  
    // for ctx attn adn self attn linear, assume [bs/token nums, qkv head nunm, heda size]
    // for gate & up linear, assume weight.shape=[hidden,2*intersize], output.shape=[bs, 2, inter size]
    // 如果传入的是三维的数据，即输入的input：[bs/token nums, qkv head nunm, heda size]
    // 那么输出的仍然是二维矩阵，仍然要以二维矩阵来计算，[bs/token nums, qkv head nunm*heda size]变成这样来计算
    Cm = output != nullptr && output->shape.size()==3 ? output->shape[1] * output->shape[2] : output->shape[1];
    Bk = input != nullptr && input->shape.size() == 3 ? input->shape[1] * input->shape[2] : input->shape[1];

    int lda = Am;
    int ldb = Bk;
    int ldc = Cm;

    // 判断矩阵是否需要转置： for lmhead linear and ffn all lieanrs
    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    if(!trans_a && !trans_b){
        // 均不转置
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }
    if(trans_b){
        // 矩阵A需要转置，此时Am,Ak需要对换
        Ak = weight.shape[1];   // qh
        Am = weight.shape[0];   // h
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }
    if(trans_a){
        // 矩阵B需要转置，此时Bk,Bn需要对换
        int tmp = Bk;
        Bk = Bn;
        Bn = tmp;
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }

    // 数据准备完毕，下面开始调库
    cublas_wrapper->Gemm(   transA,
                            transB,
                            Am,
                            Cn,
                            Bk,
                            weight.data,
                            lda,
                            input->data,
                            ldb,
                            output->data,
                            ldc,
                            1.0f,
                            0.0f);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(outpyt->data);
#else
#endif
}

    // 准备好调用cublas_warpper->Gemm需要的参数即可
    // input:   [num_tokens, q_hidden_units]
    // weight:  [hidden_size, q_hidden_units]
    // output:  [num_tokens, hidden_size]
    // 计算该矩阵相乘，我们计算的是y = input * weight^T;
    // 因此需要将weight进行转置，然后再进行矩阵乘法
    // 所以:C=output: [num_tokens, hidden_size]
    // OP(A) = OP(weight):           [q_hidden_units, hidden_size]
    // OP(B) = OP(input) = input:    [num_tokens, q_hidden_units]
    // C=output:                     [num_tokens, hidden_size]
    // 由于cublas中对二维数据的内存排布是"col-major(列为主)"的方式存储，而我们的数据是"row-major"
    // 并且CUBlas文档上所说的矩阵A，B，C均是以"col-major"的矩阵，而我们的矩阵是"row-major"
    // 所以在计算m,n,k,lda,ldb,ldc时，需要将我们的OP(A),OP(B),C转置后再看
    // C_(m*n) = alpha * OP(A)_(m*k) * OP(B)_(k,n) + beta * C_(m,n)
    // [hidden_size, num_tokens] = alpha * [hidden_size, q_hidden_units] * [q_hidden_units, num_tokens] + beta * [hidden_size, num_tokens]
    // 所以m=hidden_size,   n=num_tokens, k=q_hidden_units




template <typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T> *input1,
                                  TensorWrapper<T> *input2,
                                  TensorWrapper<T> *output,
                                  cublasWrapper *cublas_wrapper,
                                  bool trans_a, // 表示input1是否需要转置
                                  bool trans_b) // 表示input2是否需要转置
{
    // 要计算：attn = Q * K^T， 等价于在这里计算attn^T = K * Q^T
    //                                          因此A作为矩阵K, B作为矩阵Q
    //        然后需要转置矩阵K,就需要传入trans_b = True
    // Q:   input1: [bs, q_heads, q_seq_len, head_dim]
    // K:   input2: [bs, k_heads, k_seq_len, head_dim]
    // output: [bs, heads, q_seq_len, k_seq_len]
    // 将矩阵B作为input1, 矩阵A作为input2       此时需要转置矩阵A
    // Q->input1->B
    // k->input2->A 要转置，trans_b=true

    // 要计算 C = Attn * V, 等价于计算C^T = V^T * Attn^T            // 内存排布式列
    //                              所以要按这个顺序去喂数据：A-->V   B-->Attn
    // Attn * V     (k_heads 等于 v_head)   (qkv_seq_len都相等)
    // Attn:    input1: [bs, q_heads, q_seq_len, k_seq_len]
    // V:       input2: [bs, v_heads, v_seq_len, head_dim]   (如果kv_heads小于q_head，要在这个维度上做广播操作，知道与q_head相等)
    // output: [bs, q_heads, q_seq_len, head_dim] -----> [bs, q_seq_len, q_heads, head_dim] -----> [bs*q_seq_len, q_heads*head_dim] ----> [num tokens, q_hiddenunits]
    // 将矩阵Attn作为input1传入    矩阵V作为input2传入，
    // Attn--->input1----->B
    // V------>input2----->A        均不需要转置
    
/*
    总结：
        1.  先看row-major的时候，需要计算的公式：attn = Q * K^T
        2.  由于cublas是col-major，所以需要对计算得到的结果attn进行转置，才能得到我们期望的row-major的结果
                所以需要我们手动对结果进行转置： attn^T = (Q * K^T)^T = K * Q^T 即等价我们计算K*Q^T就可以得到row-major的结果，也不需要我们手动转置
        3. 由此确定实际需要给cubals API的数据形式： K * Q^T     --> K 为矩阵A，Q为矩阵B
        4. 然后回到原始的公式中，attn = Q * K^T， 得知无论是row-major还是col-major,计算时都要转置矩阵K
        5. 由此得知需要转置矩阵K，对应下来就是要转置矩阵A。然后对应看input1传入的是K还是Q
            如果input1传入的是Q,trans_a=false,因为Q不需要转置， input2传入的是K,则trans_b=true。因为K要转置
        6. 然后确定mnk: API文档中是：C_(m,n) = OP(A)_(m,k) * OP(B)_(k,n)
                所以我们得到OP(A),OP(B),C的维度
                [q_seq_len, k_seq_len] = [head_dim, k_seq_len] * [q_seq_len, head_dim]
                然后将OP(A),OP(B),C转置一下，得到col-major在内存中的排布
                [k_seq_len, q_seq_len] = [k_seq_len, head_dim] * [head_dim, q_seq_len]
                此时就对应文档中的(m,n) = (m, k) * (k, n)了
        7. 然后再看数据的形式，从原始shape中把数据取出来就可以了
        8. ld数据就是A\B\C的列。 注意，不是OP(A\B\C)
*/              


    // 不需要转置的情况:
    int Am = input2->shape[3];  // head_dim     // K: input2:  [bs, q_heads, q_seq_len, head_dim]       Attn：  input1(B): [bs, q_heads, q_seq_len, k_seq_len]  // Am=head_dim
    int Ak = input2->shape[2];  // q_seq_len    // Q: input1:  [bs, q_heads, q_seq_len, head_dim]       v:      input2(A): [bs, v_heads, v_seq_len, head_dim]   // Ak=v_seq_len
    int Bk = input1->shape[3];  // head_dim                                                             output:            [bs, q_heads, q_seq_len, head_dim]   // Bk=k_seq_len
    int Bn = input1->shape[2];  // q_seq_len                                                                                                                    // Bn=q_seq_len
    int Cm = output->shape[3];  // k_seq_len    // output:  [bs, heads, q_seq_len, k_seq_len]                                                                   // Cm=head_dim
    int Cn = output->shape[2];  // q_seq_len                                                                                                                    // Cn=q_seq_len

    int lda = Am;
    int ldb = Bk;
    int ldc = Cm;
    int64_t strideA = Am * Ak;
    int64_t strideB = Bk * Bn;
    int64_t strideC = Cm * Cm;
    int batchCount = input1->shape[0] * input1->shape[1];

    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    if(!trans_a && trans_b){
        // 均不转置的情况
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }
    else if(trans_a){
        // input1需要转置，也就是矩阵B需要转置，就需要交换矩阵的km的维度
        int Bk = input1->shape[2];  // head_dim     
        int Bn = input1->shape[3];  // q_seq_len
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }
    else if(trans_b){
        // input2需要转置，也就是矩阵A需要转置，就需要交换矩阵的km的维度
        int Am = input2->shape[2];  // head_dim    
        int Ak = input2->shape[3];  // q_seq_len 
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }
    cublas_wrapper->stridedBatchedGemm(transA,
                                       transB,
                                       Am,
                                       Cn,
                                       Bk,
                                       input2->data,
                                       lda,
                                       strideA,
                                       input1->data,
                                       ldb,
                                       strideB,
                                       output->data,
                                       ldc,
                                       strideC,
                                       batchCount,
                                       1.0f,
                                       0.0f);
#ifdef PRINT_DATA
        print_data<<<1, 1>>>(output->data);
#else
#endif
}





template void launchLinearGemm(TensorWrapper<float> *input,
                      BaseWeight<float> &weight,
                      TensorWrapper<float> *output,
                      cublasWrapper *cublas_wrapper,
                      bool trans_a,
                      bool trans_b);

template void launchLinearGemm(TensorWrapper<half> *input,
                      BaseWeight<half> &weight,
                      TensorWrapper<half> *output,
                      cublasWrapper *cublas_wrapper,
                      bool trans_a,
                      bool trans_b);


template void launchLinearStridedBatchGemm(TensorWrapper<float> *input1,
                                  TensorWrapper<float> *input2,
                                  TensorWrapper<float> *output,
                                  cublasWrapper *cublas_wrapper,
                                  bool trans_a, 
                                  bool trans_b);
                                

template void launchLinearStridedBatchGemm(TensorWrapper<half> *input1,
                                  TensorWrapper<half> *input2,
                                  TensorWrapper<half> *output,
                                  cublasWrapper *cublas_wrapper,
                                  bool trans_a, 
                                  bool trans_b);