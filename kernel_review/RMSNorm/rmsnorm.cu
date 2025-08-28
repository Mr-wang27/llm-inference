#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/*
    RMSNorm计算公式：
        一行数据： x1, x2, x3, ... ..., x_d
            第一步： 计算reduce sum: x_i平方
            第二步： 计算： x_i/ 根号下((reduce_sum / d)+eps)
            第三步： 计算： gamma * （x_i/ 根号下((reduce_sum / d)+eps)）

    输入数据格式：
        d_in/d_out          [num_tokens, hidden_dim]        注：输入输出都采用该空间
        gamma               [num_tokens]
        eps                 10e-6
        residual            [num_tokens, hidden_dim]        注：residual,保存归一化之前的数据

    并行计算思路：
        一个block处理一个token的数据
        一个block中的线程处理这个token中的多个数据

        block(512)
        grid(num_tokens)

    并行计算步骤:
        第一步： 先对一个token中的所有数据做block reduce sum, 然后将数据放在shared memory上
        第二步： 将数据再从shared memory上取出来，计算RMS(x),然后再将数据写回到shared memory上
        第三步： 将数据再从shared memory上取出来，计算RMSNorm(x), 然后再将数据写入到d_out中
*/

template <typename T>
__device__ T warpReduceSum(T val){
    for(int i = 16; i > 0; i >>= 1){
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;     // 只有lean_id = 0的线程返回的才是正确的结果
}

/*
    第一步： block中的所有线程，以32为顺序分成多个warp, 然后每个warp做warp reduce sum, 将所有warp reduce的结果放在share memory上
    第二步： 将shared memory上的数据取出来，放在前32号线程上，然后第0号warp再做依次reduce sum, 即可得到这个block reduce sum 的结果
*/
template <typename T>
__device__ T blockReduceSum(T val){
    // int gtid = blockIdx.x * blockDim.x + threadIdx.x;  block中不能使用gtid
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int leanId = threadIdx.x % 32;
    int warpNum = (blockDim.x + 32 - 1) / 32;      // 计算warp数量
    // __shared__ T warpSum[warpNum];       warpNum不是编译器常量，不能作为分配shared memory的参数
    __shared__ T warpSum[64];   // 或者32 

    // 第一步： 让block中的所有线程进行warp reduce sum
    T warp_val = warpReduceSum(val);
    // 第二步： 将warp的结果放在shared memory上
    if(leanId == 0){
        warpSum[warpId] = warp_val;
    }
    __syncthreads();
    // 第三步： 将数据从shared memory上取出来，给到0-31号线程。让0号warp在做依次warp reduce sum
    // 需要注意的是，只需要将warpSum中的数据存放在前面的线程中，非warpSum中的数据需要传递0，避免在第二次warp reduce sum 的时候污染数据
    warp_val = tid < warpNum ? warpSum[tid] : (T)0;
    // 第四步： 做第二次warp sum reduce
    T block_val = warpReduceSum(warp_val);
    return block_val;
}


    // 输入数据格式：
    //     d_in/d_out          [num_tokens, hidden_dim]        注：输入输出都采用该空间
    //     gamma               [num_tokens]
    //     eps                 10e-6
    //     residual            [num_tokens, hidden_dim]        注：residual,保存归一化之前的数据

    // 并行计算步骤:
        // 第一步： 先对一个token中的所有数据做block reduce sum, 然后将数据放在shared memory上
        // 第二步： 将数据再从shared memory上取出来，计算RMS(x),然后再将数据写回到shared memory上
        // 第三步： 将数据再从shared memory上取出来，计算RMSNorm(x), 然后再将数据写入到d_out中
template<typename T>
__global__ void rmsnorm_kernel(T* d_out, T* gamma, float eps, T* residual, int num_tokens, int hidden_dim){
    // 每个blcok处理矩阵的一行，依旧是一个token 
    for(int blockIndex = blockIdx.x; blockIndex < num_tokens; blockIndex += gridDim.x){ // 每个block处理一个token
        T val=0;
        T d_val = 0;
        // 第一步： 先计算线程的reduce sum
        for(int index = threadIdx.x; index < hidden_dim; index += blockDim.x){
            // 在里面处理一个block的数据
            val += d_out[blockIndex * hidden_dim + index] * d_out[blockIndex * hidden_dim + index];   
        }
        // 第二步： 计算block reduce sum, 然后再写入到shared memory上的时候，计算RMS(x)
        __shared__ T inv_rms;
        T block_val = blockReduceSum(val);
        if(threadIdx.x == 0){   // 正确的数据在tid=0号线程上
            inv_rms = rsqrt(block_val / hidden_dim + static_cast<T>(eps));    // 这里将会计算均方根的倒数，目的是在后面使用乘法计算提高效率
        }
        __syncthreads();
        // 第三步： 计算RMSNorm(x), 同时在写入到d_out中的时候，d_out中的结果写入到residual中
        for(int index = threadIdx.x; index < hidden_dim; index += blockDim.x){
            val = d_out[blockIndex * hidden_dim + index];
            residual[blockIndex * hidden_dim + index] = val;
            d_out[blockIndex * hidden_dim + index] = gamma[blockIndex] * val * inv_rms;
        }
    }
}
// 该kernel由两次从全局内存中读取数据：d_out[blockIndex * hidden_dim + index]
// 会产生较大的内存带宽压力
// 所以可以考虑使用shared memory来优化，先将一行hidden_dim个数据加载到shared memory上，然后再在block的for循环内计算
// 第二次需要d_out的数据的时候，就可以直接从shared memory上读取
// 如果hidden_dim较大，不能一次放到shared memory上，就需要采用tiling(分块)的方法： 即每次只加载一部分数据到shared memory上进行处理




// --------------------- CPU 参考函数 ---------------------
void rmsnorm_cpu(const std::vector<float>& in, const std::vector<float>& gamma,
                 float eps, std::vector<float>& out, std::vector<float>& residual,
                 int num_tokens, int hidden_dim) {
    out = in; // 初始复制
    for (int i = 0; i < num_tokens; ++i) {
        float sum_of_squares = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            float val = out[i * hidden_dim + j];
            sum_of_squares += val * val;
            residual[i * hidden_dim + j] = val; // 保存归一化前的数据
        }

        float inv_rms = 1.0f / std::sqrt(sum_of_squares / hidden_dim + eps);
        float gamma_val = gamma[i];

        for (int j = 0; j < hidden_dim; ++j) {
            out[i * hidden_dim + j] = out[i * hidden_dim + j] * gamma_val * inv_rms;
        }
    }
}

// --------------------- 主测试函数 ---------------------
int main() {
    // 1. 设置参数
    const int num_tokens = 8192;
    const int hidden_dim = 4096;
    const float eps = 1e-6f;
    const int total_size = num_tokens * hidden_dim;

    // 2. 主机端内存分配和数据初始化
    std::vector<float> d_in_host(total_size);
    std::vector<float> d_out_host(total_size);
    std::vector<float> gamma_host(num_tokens);
    std::vector<float> residual_host(total_size);
    std::vector<float> cpu_out_host(total_size);
    std::vector<float> cpu_residual_host(total_size);

    for (int i = 0; i < total_size; ++i) {
        d_in_host[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f; // 随机初始化
    }
    for (int i = 0; i < num_tokens; ++i) {
        gamma_host[i] = 1.0f; // 简化验证，gamma设为1
    }

    // 3. GPU 端内存分配和数据传输
    float *d_in, *d_out, *d_gamma, *d_residual;
    CUDA_CHECK(cudaMalloc(&d_in, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, num_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, total_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, d_in_host.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, d_in_host.data(), total_size * sizeof(float), cudaMemcpyHostToDevice)); // d_out = d_in
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma_host.data(), num_tokens * sizeof(float), cudaMemcpyHostToDevice));

    // 4. 定义CUDA启动配置
    int threadsPerBlock = 512;
    int blocksPerGrid = 1024; // 保证有足够的块来处理所有token
    // if (blocksPerGrid > num_tokens) {
    //     blocksPerGrid = num_tokens;
    // }
    
    // 5. 使用cudaEvent测量时间
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 6. 启动内核
    CUDA_CHECK(cudaEventRecord(start, 0));
    rmsnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_gamma, eps, d_residual, num_tokens, hidden_dim);
    CUDA_CHECK(cudaGetLastError()); // 检查内核启动错误
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    std::cout << "Kernel execution time: " << kernel_time_ms << " ms" << std::endl;

    // 7. 将结果从GPU拷贝回CPU
    CUDA_CHECK(cudaMemcpy(d_out_host.data(), d_out, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(residual_host.data(), d_residual, total_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 8. 调用CPU参考函数进行计算
    rmsnorm_cpu(d_in_host, gamma_host, eps, cpu_out_host, cpu_residual_host, num_tokens, hidden_dim);

    // 9. 验证结果
    bool passed = true;
    const float tolerance = 1e-4f;
    for (int i = 0; i < total_size; ++i) {
        if (std::abs(d_out_host[i] - cpu_out_host[i]) > tolerance) {
            std::cerr << "Mismatch at d_out[" << i << "]: GPU=" << d_out_host[i] << ", CPU=" << cpu_out_host[i] << std::endl;
            passed = false;
        }
        if (std::abs(residual_host[i] - cpu_residual_host[i]) > tolerance) {
            std::cerr << "Mismatch at residual[" << i << "]: GPU=" << residual_host[i] << ", CPU=" << cpu_residual_host[i] << std::endl;
            passed = false;
        }
        if (!passed) break;
    }

    if (passed) {
        std::cout << "Test passed successfully!" << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }

    // 10. 清理GPU资源
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}