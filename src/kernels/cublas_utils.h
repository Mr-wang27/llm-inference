// 包装需要使用到的cublas相关的功能
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include "src/utils/macro.h"        // 检查cublas调用是否成功

// 1. cuBLAS API：必须先把矩阵分配到 GPU 显存中，填充数据，调用 cuBLAS 函数序列，最后再把结果从 GPU 拷贝回 Host（CPU）端。
// 2. cuBLASXt API：可以直接在 Host（CPU）上保存数据，API 会自动管理数据分发和显存拷贝。
// 3. cuBLASLt API：是一个专注于 GEMM（矩阵乘法）的轻量级库，提供了新的灵活 API。
//      它增强了矩阵数据布局、输入类型、计算类型的灵活性，同时允许通过参数编程选择不同的算法实现与启发式策略。

class cublasWrapper{
    private:
        cublasHandle_t  cublas_handle_;
        cublasLtHandle_t cublaslt_handle_;  // 调用APi需要handle，这是固定的

        cudaDataType_t Atype_;    // 调用cublas的api的时候，需要传递类型。所以需要在这里指定类型
        cudaDataType_t Btype_;      // 数据精度表示的是数据存储在显存中精度
        cudaDataType_t Ctype_;
        cudaDataType_t computeType_;    // 计算精度表示的是从内存中取出来的数据进行运算时的运算精度。
        // 如果两者类型不一致，取出来的数据会先转换乘计算精度然后在进行计算，计算完毕存回内存的时候，会再次转回数据存储的精度进行存储

    public:
        cublasWrapper(cublasHandle_t cublas_hadle,
                      cublasLtHandle_t cublaslt_handle);
                      // BaseAllocator* allocator); enable it when we use cublasLt API
        
        ~cublasWrapper();

        void setFP32GemmConfig();
        void setFP16GemmConfig();
        
        // 包装对矩阵乘法的调用
        // 这里是二维的矩阵乘，也就是普通的线性层，就是input:[num_tokens, q_hidden_units]   weight:[hidden_size, q_hidden_units]
        // (注意这里给出的权重是原始权重，没有转置的权重)
        // C = alpha * OP(A) OP(B) + beta * C
        void Gemm(cublasOperation_t transa,     // cublas的操作，是否需要转置矩阵a
                cublasOperation_t transb,       // 同上
                const int       m,              // 结果矩阵C的行数, 或者也是OP(A)的第一个维度
                const int       n,              // 结果矩阵C的列数，或者也是OP(B)的第二个维度
                const int       k,              // A/B矩阵相乘的"内部维度"，或者是OP(A)的第二个维度，OP(B)的第一个维度
                const void*     A,              // 矩阵A的数据
                const int       lda,            // 矩阵A的leading dimension(A每一列元素之间的跨度，即列数)
                const void*     B,              // 矩阵B的数据
                const int       ldb,            // 同理
                void*           C,              // C指针是输出，注意不能是const
                const int       ldc,
                float           alpha,          // scale因子
                float           beta);          // scale因子

        // for qk*v and q*k
        // q*k : [batch_size, num_heads, seq_len, head_dim] * [batch_size, num_heads, seq_len, head_dim]
        // q*k 中，k需要转置
        // qk*v： [batch_size, num_heads, seq_len, seq_len] * [batch_size, num_heads, seq_len, head_dim]
        // 
        void stridedBatchedGemm(cublasOperation_t    transa,         // 同上
                                cublasOperation_t   transb,
                                const int           m,
                                const int           n,
                                const int           k,
                                const void*         A,
                                const int           lda,
                                const int64_t       strideA,        // 这个是因为存在batch维度，其数据是以元素个数为stride.也就是seq_len*head_dim。这么多个元素为一个矩阵乘
                                const void*         B,
                                const int           ldb,
                                const int64_t       strideB,
                                void*               C,
                                const int           ldc,
                                const int64_t       strideC,
                                const int           batchCount,     // 这个数据计算batch有多少个，也就是要进行多少次矩阵乘：batch_size*num_heads
                                float               f_alpha,
                                float               f_beta);
};