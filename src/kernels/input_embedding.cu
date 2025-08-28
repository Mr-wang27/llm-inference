#include<stdio.h>
#include "src/kernels/input_embedding.h"
#include "src/utils/cuda_debug_utils.cuh"
template<typename T>
__global__ void embeddingFunctor(const int* input_ids,
                T* output,
                const T* embed_table,
                const int max_context_token_num,
                const int hidden_size)
{
    // kernel实现逻辑：分配每个线程从embde_table中取出一个数据，然后添加到output中
    // 就需要从全局线程id定位到这个线程需要处理embed_table中的哪一个数据
    // 也就是需要定位到embed_table中的行号和列号
    // 行号是在input_ids中的，但是input_ids中的一个数据，对应embed——table中hiddensize个数据
    // 因此行号： id = input_ids[index / hidden_size]
    // 列号： id*hidden_size + index % hidden_size
    // 其次，因为启动kernel分配的线程数量是一定的，因此需要用循环来保证数据全部处理完毕
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // 需要注意，分配的grid以及block都是一维的
    while(index < max_context_token_num * hidden_size){
        int id = input_ids[index / hidden_size];    // 获取当前线程需要处理的行号
        output[index] = embed_table[id * hidden_size + (index % hidden_size)];  // 获取列号，然后写入
        // 需要注意的是，虽然output以及embed_table是二维tensor,但在内存上的分布就是一维的，所以用一维模拟二维的数据位置即可
        index += gridDim.x * blockDim.x;
    }
}



template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [token num]
                          TensorWrapper<T>* output,       // FP32 [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T>* embed_table   // FP32 [vocal_size, hidden_size]
                        ){
    const int blockSize = 256;
    const int max_context_token_num = output->shape[0]; // token num
    const int hidden_size = output->shape[1];
    const int gridSize = 2048;
    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output.");
    embeddingFunctor<T><<<gridSize, blockSize>>>(input_ids->data,
                                                output->data,
                                                embed_table->data,
                                                max_context_token_num,
                                                hidden_size);
                          
// 调试宏，如果需要打印数据来调试kernel,就需要在编译的时候打开这个宏
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}


// 实例化模板
template void launchInputEmbedding(TensorWrapper<int>* input_ids,
                                   TensorWrapper<float>* output,
                                   EmbeddingWeight<float>* embed_table);

template void launchInputEmbedding(TensorWrapper<int>* input_ids,
                                   TensorWrapper<half>* output,
                                   EmbeddingWeight<half>* embed_table);