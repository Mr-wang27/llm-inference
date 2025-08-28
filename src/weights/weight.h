// 定义了一个抽象类 Weight，它的作用通常是作为接口基类，用于不同类型的权重类的统一调用。
# pragma once
#include <string>
struct Weight{
    virtual void loadWeights(std::string weight_path) = 0;  // 纯虚函数
};


// 加载不同权重类的统一接口