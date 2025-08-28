// RMSNormal 权重，只有一个Scale权重，shape=[q_hidden_unit]
// q_hedden_unit=q_heads*hidden_size

// 为RMSNoraml参数创建参数的数据结构
#pragma once
template<typename T>
struct LayerNormWeight{
    T* gamma;
};