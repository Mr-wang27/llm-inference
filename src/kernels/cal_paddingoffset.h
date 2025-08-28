// 计算paddingoffset

#pragma once 
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/macro.h"
#include "src/utils/tensor.h"

// shape:
    //seq_lengths:[batch size]
    //cum_seqlens:[batch size + 1],first ele is 0
    //padding_offset:[batch size * max q len]
// note: the point is to calc padding offset and cum offset
void launchCalPaddingoffset(TensorWrapper<int>* padding_offset,        // 输出
                            TensorWrapper<int>* cum_seqlens,        // 输出：累积句子长度
                            TensorWrapper<int>* input_lens);     // 输入: 每个句子的长度



/*
例子：下面这个是batch=4(有4个句子)，最大seq_len=7的数据，有效的token用1表示，padding 的token用0 表示
11111000    11110000    11111110    11111100
11110000
11111110
11111100

padding_offset:
cum_seqlens：累积的句子长度就是：[0, 5, 9, 16, 22]， 通过这个数组可以判断每个序列的起始位置：第一个句子的索引[0,5), 第二个句子[5,9), 第三个句子[9, 16),第四个句子[16,22)
因此在embedding那个kernel中，输入的input_ids可以直接将原始的维度[batch_size, seq_len]进行flatten，编程(num_tonkes)维度，然后进行embedding

这里计算每个数据的padding offset,也就是针对上面的有效token,计算当前位置的数据应该在应该在后面的第几个位置处
比如第一行中的前5个token的offset均为0.因为前面没有padding token。
第二行中前4个token，前面有3个padding token，因此这4个token的offset=3
第三行的前7个token的padding offset = 7
第四行的前6个token的padding offset = 8.
所以，起始这个padding offset计算的就是每个有效token的padding token的前缀和。
所以这个padding offset为：
00000    3333    7777777    888888
33333333
77777777
88888888


然后再后面remove padding的时候，就可以利用这个padding offset，将padding之后的数据：[batch_size, seq_len] 拉平成一维数组A，然后padding offset数组也拉平为一维数组B，
然后padding offset代表的就是当前位置的元素应该在原数组的后面第几个位置处, 所以将这 A[index] = A[index + paddingOffset[index]] 按照index进行遍历 for(int index > 0; index < padding_offset.size(); index++)
就可以将数据还原为[num tokens]。
注意的是这样遍历之后，原数据数据维度为[batch_size*seq_len]。 但是只有前面的num tokens是有效后，后面的token都是无效的。
*/
