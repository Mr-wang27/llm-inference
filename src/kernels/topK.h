#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template <typename T, int K>
struct topK
{
    T val[K];
    int id[K];

    __device__ void init(){ // 初始化
        for(int i = 0; i < K; i++){
            id[i] = -1;     // 对应数据的id
            val[i] = 1e-20; // topK个数据
        }
        // 数据的排布由大到小
    }

    __device__ void insertHeap(T data, int data_id){
        float v = (float)val[K-1];  // 维护的topK的最后一个数据，也是最小的一个数据
        if(id[K-1] == -1 || (float)data > v){
            // topK个数据还有空余位置或者当前数据比最小的数据大，需要进行替换添加到topK序列中
            id[K-1] = data_id;
            val[K-1] = data;    // 替换数据
        }
        // 仅仅需要一轮冒泡排序(插入新元素的重排)，因为此时除了最后一个新元素，其他元素都是有序的
        for(int i = K - 2; i >= 0; i--){
            if(val[i+1] > val[i] || id[i] == -1){// 出现一个更大的数据或者topK有剩余位置
                T tmp = val[i];
                val[i] = val[i+1];  // 向上冒泡
                val[i+1] = tmp;
                // id数据也要跟着一起交换
                int tmp_id = id[i];
                id[i] = id[i+1];
                id[i+1] = tmp_id;
            }
        }
    }

    __device__ void merge(const topK<T, K>& other){
        for(int i =0; i < K; i++){
            insertHeap(other.val[i], other.id[i]);
        }
    }
};



template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T>* probs,               // 模型层的输出：[bs, beam_width, vocab_size]
                             TensorWrapper<int>* topk_ids,          // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<T>* topK_values,         // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<int>* final_topk_ids,    // 第二轮topK的数据：[bs, beam_width, k]
                             TensorWrapper<T>* final_topk_values);// 第二轮topK的数据：[bs, beam_width, k]

                             // 需要注意的是这里的BlockPerBeam维度