#include "src/kernels/cal_paddingoffset.h"
// shape：
    // seq_lengths:[batch_size]
    // cum_seqlens:[batch size + 1], first element is 0
    // padding_offset:[batch_size * max_q_len]
// note: the point is to calc padding offset and cum offset
// TODO: we first use serial algo, then can enhance to CUDA Scan algo



/*

11111000    11110000    11111110    11111100
11110000
11111110
11111100
batch_size=4
max_q_len=8
input_lengths:[5,4,7,6]
cum_seqlens：累积的句子长度就是：[0, 5, 9, 16, 22]
paddingoffset:
00000    3333    7777777    888888
3333
7777777
888888
*/
// __global__ void CalPaddingoffset(int*           padding_offset,         // padding_offset:[batch_size * max_q_len],但是只有前token_nums个数据被填充了
//                                  int*           cum_seqlens,
//                                  const int*     input_lengths,
//                                  const int      batch_size,
//                                  const int      max_q_len){
//     int ind = 0;
//     int cum_offset = 0;
//     int total_seqlen = 0;
//     for(int b = 0; b < batch_size; b++){
//         int seqlen = input_lengths[b];  // 当前序列实际长度
//         // 将上一次的结果存进cum_seqlens与padding_offset
//         cum_seqlens[b] = total_seqlen;
//         // 一个序列中，一行中的每一个token具有相同的offset
//         for(int i = 0; i < seqlen; ++i){
//             padding_offset[ind] = cum_offset;
//             ind++;
//         }
//         cum_offset += max_q_len - seqlen;
//         total_seqlen += seqlen;
//     }
//     cum_seqlens[batch_size] = total_seqlen;
// }


__global__ void CalPaddingoffset(int*         padding_offset,               // 输出每个数据，包括padding token的padding offset.
                                int*         cum_seqlens,                   // 输出一个一维数组，第一个元素位0，后面的元素位累积句子和
                                const int*   input_lengths, //actual input lens,    是一个一维数组，存储每个序列的长度
                                const int    batch_size,
                                const int    max_q_len) {
    int ind = 0;
    int cum_offset = 0;
    int total_seqlen = 0;
    for(int b = 0; b < batch_size; b++) {
        int seqlen = input_lengths[b];

        cum_seqlens[b] = total_seqlen;
        // each token in one seq has same cum offset
        for (int i = 0; i < seqlen; i++) {
            padding_offset[ind] = cum_offset;
            ind++;
        }
        cum_offset += max_q_len - seqlen;
        total_seqlen += seqlen;
    }
    cum_seqlens[batch_size] = total_seqlen;
}

void launchCalPaddingoffset(TensorWrapper<int>* padding_offset, // [batch_size, max_q_len]
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_lengths) // actual input lens
{
    const int batch_size = padding_offset->shape[0];
    const int max_q_len = padding_offset->shape[1];
    LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0], "input lengths numbers should equal to padding offset bs dim");
    LLM_CHECK_WITH_INFO(batch_size == cum_seqlens->shape[0] - 1, "cum seqlen numbers should equal to padding offset bs dim + 1!");
    CalPaddingoffset<<<1, 1>>>(
        padding_offset->data, cum_seqlens->data, input_lengths->data, batch_size, max_q_len
    );

}