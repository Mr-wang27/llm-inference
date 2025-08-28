#include <float.h>
#include <cuda.h>
#include <iostream>
#include "src/kernels/topK.h"
#include <cub/cub.cuh>

// 创建一个topK functor
// Note: a,b两个topK reduce输出一个topK
// 也就是进行reduce的操作，不同于之前的sum,max。因为sum、max都不用复杂的操作
template<typename T, int K>
__device__ topK<T, K> reduce_functor(const topK<T, K>& a, const topK<T, K>& b)
{
    topK<T, K> res = a;
    for(int i = 0; i < K; i++){     // 把topK b中的数据依次插入到topK a中，从而实现两个topK之间的操作
        res.insertHeap(b.val[i], b.id[i]);
    }
    return res;
}



// 参数需要从模板参数中传进来，因为这些数据需要是编译期常量
// blockSize = 256
template <typename T, int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round1(const T* probs, const int vocab_size,
                                   int* topK_ids, T* topK_values)
{
/*
    gridsize:bs * beamwidth * BlockPerBeam 
    blocksize: 256
    shape infer: [bs, beamwidth, vocab size] => [bs, beamwidth, BlockPerBeam, K]
*/

    // 创建BlockReduce<>

    // 第一步：先把一些线程相关的索引写出来，还有数据偏移相关的索引
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    // int gid = block_id * blockDim.x + tid;  // 全局线程id
    int row_id = block_id / BlockPerBeam;   // [bsxbm, BlockPerBeam*k]
    int block_lean_id = block_id % BlockPerBeam;   // 一行的数据(vocab_size)中，这个block在这一行中的block_id
    // int block_nums = vocab_size / BlockPerBeam;

    topK<T, K> thread_topK;    // 256个线程处理的所有数据，对这些数据做reduce topK
    thread_topK.init();

    // thread local reduce
/*
    这里的实现，并不是真正意义上的将vocab_size个数据分段乘BlockPerBeam个小段，然后每个block处理其中的一个小段上的数据。即不是从数据的角度去考虑
    而是从线程的角度去考虑，总共由BlockPerBeam*blockSzie个线程(8*256)。然后有vocab_size个数据
    因此将这些全部的线程排布在数据上，一次for循环，所有线程(8*256)执行一次，就取出前8*256个数据，
    然后利用tid控制每个线程在其block中的偏移，
    利用block_lean_id*blockSize控制该block是这一行数据中的第几个block上的数据，然后偏移到这个block对应的数据上面

*/
    int row_offset = row_id * vocab_size;    // 数据偏移到一行vocab_size
    for(int data_id = tid + block_lean_id * blockSize; data_id < vocab_size; data_id += BlockPerBeam * blockSize){
        // 取出数据，然后将数据插入到该线程的topK中，进行topK运算
        int data_offset = row_offset + data_id;
        T data = probs[data_offset];
        thread_topK.insertHeap(data, data_id);  // 注意，这里这个data_id应该是一行上面的数据偏移，因为计算的就是一行上面的topK
    }
    // 接下来需要做blockReduce, 对每个线程的thread_topK做reduce
    // 使用cub库实现blockreduce
    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storages;
    topK<T, K> block_topK = blockreduce(temp_storages).Reduce(thread_topK, reduce_functor<T, K>);

    if(tid==0){
        // 一个线程写回数据
        for(int k_offset = 0; k_offset < K; k_offset++){
            topK_values[row_id * BlockPerBeam * K + block_lean_id * K + k_offset] = block_topK.val[k_offset];
            topK_ids[row_id * BlockPerBeam * K + block_lean_id * K + k_offset] = block_topK.id[k_offset];
        }
    }


// 下面是手动reduce的写法，把每个线程的topK写入到shared memory上，然后用一个线程reduce这些所有的topK,最后写入即可
// 可以使用warp_level reduce来提升性能    
//     // 每个线程将自己的 topK 结果写入 shared memory
//      __shared__ topK<T, K> shared_topks[blockSize];  // 每个线程一个 topK
//     shared_topks[tid] = thread_topK;

//     __syncthreads();  // 确保写入完成

//     // 线程0负责合并所有线程的topK结果,并写入结果
//     if (tid == 0) {
//         topK<T, K> final_topK;
//         final_topK.init();
//         for (int i = 0; i < blockSize; ++i) {
//             final_topK.merge(shared_topks[i]);      // 将所有线程的topK进行merge， 得到final_topK
//         }
// // [bs, beamwidth, BlockPerBeam, K]
//         for (int i = 0; i < K; ++i) {
//             topK_ids[row_id * BlockPerBeam * K + block_lean_id * K + i] = final_topK.id[i];
//             topK_values[row_id * BlockPerBeam * K + block_lean_id * K + i] = final_topK.val[i];
//         }
//     }

}


// 一个blcok处理BlockPerBeam * K 个数据。(8*5=40)
// shape infer: [bs, beamwidth, BlockPerBeam, K] => [bs, beamwidth, K]
// ids是beam width * vocab size中的全局word id
// gridSize = bs*bm
// blockSize = 256
template <typename T, int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round2(const int* topK_ids, const T* topK_values,
                                   int* final_topK_ids, T* final_topK_values)
{
    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // int gid = blockIdx.x * blockDim.x + tid;
    int row_id = bid;
    topK<T, K> thread_topK;
    thread_topK.init();

    // thread local reduce
    for(int i = tid; i < BlockPerBeam * K; i += blockDim.x){
        int data_offset = row_id * BlockPerBeam * K + i;
        T data = topK_values[data_offset];
        int data_id = topK_ids[data_offset];
        thread_topK.insertHeap(data, data_id);
    }

    // block reduce
    topK<T, K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor<T, K>);

    // 写回数据
    if(tid == 0){
        for(int k_offset = 0; k_offset < K; k_offset++){
            final_topK_values[row_id * K + k_offset] = block_topK.val[k_offset];
            final_topK_ids[row_id * K + k_offset] = block_topK.id[k_offset];
        }
    }
}





template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T>* probs,               // 模型层的输出：[bs, beam_width, vocab_size]--->[bs*bw, vocabsize] 在传入这个kernel之前做处理
                             TensorWrapper<int>* topk_ids,          // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<T>* topK_values,         // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<int>* final_topk_ids,    // 第二轮topK的数据：[bs, beam_width, k]
                             TensorWrapper<T>* final_topk_values)// 第二轮topK的数据：[bs, beam_width, k]
{
    // support both beamserach and sampling topk by integrate beamwidth into batchsize
    // we get variable bsxbw = bs*bw, the probs shape is [bs*bw, vocabsize]
    int bsxbm = probs->shape[0];
    int vocab_size = probs->shape[1];
    constexpr int BlockPerBeam = 8;
    // constexpr int beamwidth = 1;    // 这里使用的是greddy策略，所以beamwidth为1
    constexpr int K = 5;
    // 输出空间的 buffer size
    // int topK_val_buf_szie = bsxbm * BlockPerBeam * K;
    // int topK_ids_buf_size = bsxbm * BlockPerBeam * K;
    // int final_topK_val_buf_size = bsxbm * K;

    T* topK_vals = topK_values->data;
    int* topK_ids = topk_ids->data;
    T* final_topK_vals = final_topk_values->data;
    int* final_topK_ids = final_topk_ids->data;
    // prepare launch
    // TODO: add GPUconfig API to easily get GPU config, ep: maxblocknums
    // GPUConfig config;
    // int maxBlockNums = config.getMaxBlockNums();
    // TODO: how to alloc block nums more flexable according to shape
    //constexpr int BlockPerBeam = 8;
    int maxBlockNums = 1024;
    int BlockNums1 = std::min(bsxbm * BlockPerBeam, maxBlockNums);
    int BlockNums2 = std::min(bsxbm, maxBlockNums);
    dim3 grid_round1(BlockNums1);
    dim3 blcok_round1(256);
    dim3 grid_round2(BlockNums2);
    dim3 blcok_round2(256);

    topK_kernel_round1<T, K, 256, BlockPerBeam>
                        <<<grid_round1, blcok_round1>>>(probs->data, vocab_size,
                                                        topK_ids, topK_vals);
                                    
    topK_kernel_round2<T, K, 256, BlockPerBeam>
                        <<<grid_round2, blcok_round2>>>(topK_ids, topK_vals,
                                                        final_topK_ids, final_topK_vals);

}


template void launchTopKforBeamSearch(TensorWrapper<float>* probs,               // 模型层的输出：[bs, beam_width, vocab_size]--->[bs*bw, vocabsize] 在传入这个kernel之前做处理
                             TensorWrapper<int>* topk_ids,          // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<float>* topK_values,         // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<int>* final_topk_ids,    // 第二轮topK的数据：[bs, beam_width, k]
                             TensorWrapper<float>* final_topk_values);// 第二轮topK的数据：[bs, beam_width, k]


template void launchTopKforBeamSearch(TensorWrapper<half>* probs,               // 模型层的输出：[bs, beam_width, vocab_size]--->[bs*bw, vocabsize] 在传入这个kernel之前做处理
                             TensorWrapper<int>* topk_ids,          // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<half>* topK_values,         // 第一轮topk的数据：[bs, bead_width, BlockPerBeam, k]
                             TensorWrapper<int>* final_topk_ids,    // 第二轮topK的数据：[bs, beam_width, k]
                             TensorWrapper<half>* final_topk_values);// 第二轮topK的数据：[bs, beam_width, k]


/*
    这个topK的整体实现思路：
    
    第一轮topK分配batch_size*beam_width*BlockPerBeam个block来处理[bs*bw, vocabsize]个数据
    然后对应的，就是一个BlockPerBeam个block来处理vocabsize个数据
    然后对应的就是，这vocab_size个数据就是一个beam的数据，所以我们将这个维度命名为BlockPerBeam
    所以一个Beam这里分配了8个block来处理。每个block分配256个线程
    但是block之间的线程是不能进行同步的，数据也不能进行同步.因为一个kernel启动的block是可能分配到不同的SM上
    因此block之间难以进行同步
    所以我们这里就是使用了8个blcok来将vocab_size的数据分段(vocab_szie很大，llama2是32000)，分成8段
    然后每个block处理一段数据，得到这个block的topK数据
    所以经过第一轮的处理之后，就会得到batch_size*beam_width*BlockPerBeam个topK的数据
    然后进行输出，输出维度就是[batch_size, beam_width, BlockPerBeam, K];
    
    然后第二轮数据就只需要将这些数据作为输入，然后进行第二轮的reduce
    这一轮的reduce的输出是[batch_size, beam_width, k]
    因此我们分配batch_size*beam_width个block来处理，也就是每个block处理BlockPerBeam*K个数据，然后维护一个topK，并输出k个数据
    因此分配256个线程处理BlockPerBeam*K个数据。这256个线程每个线程维护一个topK，然后所有线程再做一个topk。得到最终的topk
    再写入对应位置即可


    前面遗漏的一个问题就是怎么在一个block里面去处理得到topK的数据
    看第一轮：分配了256个线程处理vocab_size/BlockPerBeam个数据，也就是256个线程处理32000/8=4000个数据
        这里处理的策略就是每个线程本地维护一个topK。因为每个线程需要处理: 4000/256=15.625,即前160个线程需要处理16个数据，后面的线程需要处理15个数据
        然后，每个线程所处理的数据中维护一个topK，然后即可得到每个线程的topK数据
        最后再将每个线程的topK数据做一个block级别的reduce topK操作，即可得到最终的一个block_topK数据.
*/