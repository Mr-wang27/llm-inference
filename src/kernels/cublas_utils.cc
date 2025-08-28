#include "src/kernels/cublas_utils.h"
#include <iostream>
// cublas gemm和stridedbatchgemm调库的写法，比较固定
cublasWrapper::cublasWrapper(cublasHandle_t cublas_hadle,
                            cublasLtHandle_t cublaslt_handle)
    : cublas_handle_(cublas_hadle), cublaslt_handle_(cublaslt_handle)
{}


cublasWrapper::~cublasWrapper()
{}

// 在初始化 cublas 包装器之后，在模型示例的 main 函数中调用。
void cublasWrapper::setFP32GemmConfig()
{
    Atype_          = CUDA_R_32F;  // 表示是FP32
    Btype_          = CUDA_R_32F;  // 表示是FP32
    Ctype_          = CUDA_R_32F;  // 表示是FP32
    computeType_    = CUDA_R_32F;  // 表示是FP32
}

void cublasWrapper::setFP16GemmConfig()
{
    Atype_          = CUDA_R_16F;  // 表示是FP32
    Btype_          = CUDA_R_16F;  // 表示是FP32
    Ctype_          = CUDA_R_16F;  // 表示是FP32
    computeType_    = CUDA_R_16F;  // 表示是FP32
}


// 调用Gemm
void cublasWrapper::Gemm(cublasOperation_t transa,     // cublas的操作，是否需要转置矩阵a
                        cublasOperation_t transb,       // 同上
                        const int       m,              // 结果矩阵C的行数, 或者也是OP(A)的第一个维度
                        const int       n,              // 结果矩阵C的列数，或者也是OP(B)的第二个维度
                        const int       k,              // A/B矩阵相乘的"内部维度"，或者是OP(A)的第二个维度，OP(B)的第一个维度
                        const void*     A,              // 矩阵A的数据
                        const int       lda,            // 矩阵A的leading dimension(A每一列元素之间的跨度，即列数)
                        const void*     B,              // 矩阵B的数据
                        const int       ldb,            // 同理
                        void*           C,
                        const int       ldc,
                        float           f_alpha = 1.0f,          // scale因子
                        float           f_beta = 0.0f)           // scale因子
// 在实现的时候添加默认参数，但是在声明的时候可以不写明默认参数
{
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;    // 确定计算类型
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&(h_alpha)) : reinterpret_cast<void*>(&(f_alpha));
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&(h_beta )) : reinterpret_cast<void*>(&(f_beta ));
    CHECK_CUBLAS(cublasGemmEx(  cublas_handle_,
                                transa,
                                transb,
                                m,
                                n,
                                k,
                                alpha,
                                A,
                                Atype_,
                                lda,
                                B,
                                Btype_,
                                ldb,
                                beta,
                                C,
                                Ctype_,
                                ldc,
                                computeType_,
                                CUBLAS_GEMM_DEFAULT));// 选择的 gemm 算法
}



// 调用strideBatchedGemm()
void cublasWrapper::stridedBatchedGemm(  cublasOperation_t    transa,         // 同上
                                        cublasOperation_t   transb,
                                        const int           m,
                                        const int           n,
                                        const int           k,
                                        const void*         A,
                                        const int           lda,
                                        const int64_t       strideA,        // 这个是因为存在batch维度
                                        const void*         B,
                                        const int           ldb,
                                        const int64_t       strideB,
                                        void*               C,
                                        const int           ldc,
                                        const int64_t       strideC,
                                        const int           batchCount,
                                        float               f_alpha = 1.0f,
                                        float               f_beta = 0.0f)
{
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;   // 确定当前CUBlasWrapper中的数据的数据类型
    // 这里当计算类型为fp16的时候，我们仍然可以传入fp32的数据，因为cublas会在内部将数据转为half.
    // 所以我们这里是将其转为void*即可。
    // 而如果数据是fp32,那么alpha本身也为fp32,就转为const void*, 不允许在cublas内部转换数据类型
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&(f_alpha)) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&(f_beta)) : reinterpret_cast<const void*>(&f_beta);

    // 开始调用
    CHECK_CUBLAS(cublasGemmStridedBatchedEx( cublas_handle_,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            Atype_,
                                            lda,
                                            strideA,
                                            B,
                                            Btype_,
                                            ldb,
                                            strideB,
                                            beta,
                                            C,
                                            Ctype_,
                                            ldc,
                                            strideC,
                                            batchCount,
                                            computeType_,
                                            CUBLAS_GEMM_DEFAULT)); // 选择GEMM算法
}



