// 定义embedding层的的weights的数据结构
// 继承自抽象基类BaseWeight
// 由于embedding层的weights较为简单，baseWeight可以进行表示，因此没有添加额外的内容，直接继承即可

#pragma once
#include "src/weights/base_weights.h"
template<typename T>
struct EmbeddingWeight: public BaseWeight<T>{

};