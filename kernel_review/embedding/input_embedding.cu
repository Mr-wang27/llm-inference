#include<stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

/*
输入数据：
    d_in:               input.shape = [num_tokens]
    d_out:              output.shape = [num_tokens, hidden_dim]
    embedding_tabel:    embeddiing_table.shape = [vocab_size, hidden_dim]

并行思路：
    block(256)
    grid(2048)
    
    将线程按照d_in中的行号依次从embedding_table中取出数据放到d_out中
    每hidden_dim个线程取input中对应一行的数据

    */

template <typename T>
__global__ void embedding(int* d_in, T* d_out, T* embedding_table, int hidden_dim, int num_tokens){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int index = gtid; index < num_tokens * hidden_dim; index += stride){
        int id = d_in[index / hidden_dim];         // 获取到每hidden_dim个线程所需要从embedding_table中取得数据行号
        d_out[index] = embedding_table[id * hidden_dim + index % hidden_dim];     // gtid % hidden_dim是该线程对应的列号
    }
}



// ------------------ CPU Baseline ------------------
void embedding_cpu(const int* in, float* out, const float* embedding_table,
                   int hidden_dim, int num_tokens) {
    for (int t = 0; t < num_tokens; t++) {
        int id = in[t];
        for (int j = 0; j < hidden_dim; j++) {
            out[t * hidden_dim + j] = embedding_table[id * hidden_dim + j];
        }
    }
}

// ------------------ Main ------------------
int main() {
    // 参数
    int vocab_size = 1000;   // 词表大小
    int hidden_dim = 128;    // embedding dim
    int num_tokens = 64;     // 输入 token 数量

    // 分配 host 内存
    int* h_in = new int[num_tokens];
    float* h_embedding_table = new float[vocab_size * hidden_dim];
    float* h_out_gpu = new float[num_tokens * hidden_dim];
    float* h_out_cpu = new float[num_tokens * hidden_dim];

    // 初始化输入 token ids
    for (int i = 0; i < num_tokens; i++) {
        h_in[i] = i % vocab_size;
    }

    // 初始化 embedding_table
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            h_embedding_table[i * hidden_dim + j] = i * 0.001f + j * 0.01f;
        }
    }

    // 分配 device 内存
    int* d_in;
    float* d_out;
    float* d_embedding_table;
    cudaMalloc((void**)&d_in, num_tokens * sizeof(int));
    cudaMalloc((void**)&d_out, num_tokens * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_embedding_table, vocab_size * hidden_dim * sizeof(float));

    // 拷贝数据 H2D
    cudaMemcpy(d_in, h_in, num_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_embedding_table, h_embedding_table, vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    // 设置 CUDA Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (num_tokens * hidden_dim + blockSize - 1) / blockSize;

    // ------------------ GPU Kernel ------------------
    cudaEventRecord(start);
    embedding<<<gridSize, blockSize>>>(d_in, d_out, d_embedding_table, hidden_dim, num_tokens);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // GPU 结果拷贝回 host
    cudaMemcpy(h_out_gpu, d_out, num_tokens * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // ------------------ CPU Baseline ------------------
    embedding_cpu(h_in, h_out_cpu, h_embedding_table, hidden_dim, num_tokens);

    // ------------------ 结果对比 ------------------
    int errors = 0;
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
        if (fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-5) {
            if (errors < 10) { // 只打印前10个错误
                printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_out_gpu[i], h_out_cpu[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("✅ GPU result matches CPU result!\n");
    } else {
        printf("❌ Found %d mismatches!\n", errors);
    }

    // ------------------ 打印部分结果 ------------------
    printf("Kernel execution time: %f ms\n", milliseconds);
    printf("Output sample:\n");
    for (int i = 0; i < 3; i++) {
        printf("token %d (id=%d): ", i, h_in[i]);
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_out_gpu[i * hidden_dim + j]);
        }
        printf("...\n");
    }

    // ------------------ 内存释放 ------------------
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_embedding_table);
    delete[] h_in;
    delete[] h_embedding_table;
    delete[] h_out_gpu;
    delete[] h_out_cpu;

    return 0;
}