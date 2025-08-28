#pragma once
#include<vector>
#include<unordered_map>
#include<string>
#include<algorithm>
#include<numeric>
#include<sstream>
#include<iostream>
#include<cuda_fp16.h>
#include "src/utils/string_utils.h"
#include "src/utils/macro.h"

// 先创建两个枚举类型用来表示数据类型与设备类型
enum Device{
    CPU_PINNED, // CPU_Pinned表示OS锁定的内存区域，不会进行分页，CPU与GPU可以通过DMA直接在这个内存区域进行数据交换。提高CPU与GPU之间数据交换效率
    CPU,
    GPU
};

enum DataType{
    FP32,
    FP16,   // half
    INT8,   // int8_t
    INT32,
    BOOL,
    BYTES,  // 表示char类型，也就是一个字节，8位
    UNSUPPORTED
};


// 创建模板函数返回数据类型(类型萃取？)
template<typename T>
DataType getTensorType()
{
    if(std::is_same<T, float>::value || std::is_same<T, const float>::value){
        return DataType::FP32;
    }
    else if(std::is_same<T, half>::value || std::is_same<T, const half>::value){
        return DataType::FP16;
    }
    else if(std::is_same<T, int>::value || std::is_same<T, const int>::value){
        return DataType::INT32;
    }
    else if(std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value){
        return DataType::INT8;
    }
    else if(std::is_same<T, bool>::value || std::is_same<T, const bool>::value){
        return DataType::BOOL;
    }
    else if(std::is_same<T, char>::value || std::is_same<T, const char>::value){
        return DataType::BYTES;
    }
    else{
        return DataType::UNSUPPORTED;
    }
}


// 前置声明模板类TensorWrapper; tensor的数据指针封装在这个模板类中
template<typename T>
class TensorWrapper;

// Tensor不是一个模板类
struct Tensor{
    Device              location;   // 设备
    DataType            dtype;  
    std::vector<int>    shape;

    // 构造函数
    Tensor() = default; // 显示生成默认构造函数
    Tensor(Device location_, DataType dtype_, std::vector<int> shape_)
        : location(location_), dtype(dtype_), shape(shape_){}

    // 计算该tensor的元素数量
    virtual int size() const{
        if(shape.size() == 0){
            // TODO: add an reminder info
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>()); // 返回tensor中的元素总量。第三个参数是初始化数值，第四个参数是functor,用于自定义的计算函数
    }

    // 将Tensor类型指向转换成TensorWrapper<T>类型
    template<typename T>    // 模板函数
    TensorWrapper<T>* as(){
        // 使用static_cast向下转换的时候，只有确保基类指针指向的是TensorWrapper这个基类的时候才是安全的
        return static_cast<TensorWrapper<T>*>(this);
        // 创建的时候： Tensor<T>* newTensor = TensorWarpper<T>(location, dtype, shape);
    }

    // 返回当前设备的字符串
    std::string DeviceString() const{
        // 使用懒汉式的单例模式，在第一次调用该函数的时候，生成一个静态局部成员变量，该变量因为是静态的，可以存在于整个程序运行生命周期
        // 并且静态变量是属于类的，而非对象的
        static const std::unordered_map<Device, std::string> devicetring{
            {CPU, "CPU"}, {CPU_PINNED, "CPU_PINNED"}, {GPU, "GPU"}
        };
        return devicetring.at(location);
    }

    // 返回当前tensor的所有相关信息，以string的形式输出
    virtual std::string toString() const{
        std::string device_str = DeviceString();    // 获取设备字符串
        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"}, {INT32, "INT32"}, {FP16, "Fp16"}, {FP32, "FP32"},
        };
        return fmtstr("Tensor[where=%s, type=%s, shape= %s]",
            device_str.c_str(),
            type_to_string.at(dtype).c_str(),
            vec2str(shape).c_str());
    }
};



template<typename T>
class TensorWrapper: public Tensor{
public:
    T* data;    // 数据指针
    TensorWrapper(Device location, DataType dtype, std::vector<int> shape)
        : Tensor(location, dtype, shape){}  // 调用基类的构造函数
    
    TensorWrapper(Device location, DataType dtype, std::vector<int> shape, T* data)
        : Tensor(location, dtype, shape), data(data){
            DataType in_dtype = getTensorType<T>(); // 检查传入的数据类型是否与声明的Tensor类型一致
            LLM_CHECK_WITH_INFO(in_dtype == dtype, "when build TensorWarpper, the passed in data type should be same as dtype in params");
        }


    // friend bool operator==(Tensor& t1, Tensor& t2);

    // 派生类需要实现虚函数:计算tensor size
    virtual int size() const {
        if(data == nullptr || shape.size() == 0){
            // TODO: add an reminder info
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }

    // 获取数值。 注意，获取数据前要检查数据在那个设备上，只有在CPU上的时候才能获取
    inline T getVal(int id) const {
        // TODO: need some boundaary and device check
        LLM_CHECK(location == CPU); // 如果不在CPU上，则会抛出异常
        return data[id];
    }// only available on CPU by []

    // 不传入参数的时候，返回第一个数值
    inline T getVal() const {
        // TODO: add type check, this is very important, because we often naturally access GPU data, which is wrong
        // for example, I am in transpose kernel to use layer_id->getVal<int>(), which is wrong
        LLM_CHECK(location == CPU);
        return getVal(0);
    }

    // 获取数据指针
    inline T* getPtr() const{
        // TODO: need some boundary check
        return (T*)data;
    }

    // 获取数据指针往后偏移offset之后的指针指向位置
    inline T* getPtrByOffset(int offset) const {
        // TODO: need some boundary check
        return (T*)data + offset;
    }

    // for debug
    virtual std::string toString() const{
        // 继承自基类的函数
        std::string device_str = DeviceString();    // 获取设备字符串

        // 子类中的类型字符串
        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"}, {FP16, "FP16"}, {FP32, "FP32"},
        };

        return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                device_str.c_str(),
                type_to_string.at(dtype).c_str(),
                vec2str(shape).c_str(),
                data);
    }
};





//I cant check if the data pointer in TensorWrapper is nullptr, because the val in tensormap is tensor*
//so I must check the data pointer using LLM_CHECK_WITH_INFO before insert into tensormap.
// 前面利用TensorWrapper来包裹数据指针，是为了实现泛型，可以使用一个容器来存储不同数据类型的tensor
// 因为在一个网络中，可能存在INT8量化后的权重参数，也可能存在FP32未量化的权重参数，要统一进行管理
// 初始化的时候，使用Tensor类指针指向TensorWrapper即可。
struct TensorMap{
    std::unordered_map<std::string, Tensor*> tensor_map_;

    TensorMap() = default;
    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map){
        for(auto& pair : tensor_map){
            if(isValid(pair.second)){               // isValid是类成员函数
                insert(pair.first, pair.second);    // 构造函数中可以调用类成员函数
            }
            else{
                // std::cout << "this is not a valid tensor, skip to insert into tensormap" << std::endl;
                LLM_CHECK_WITH_INFO(isValid(pair.second), fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
            }
        }
    }
    // 拷贝构造函数
    TensorMap(const std::unordered_map<std::string, Tensor*>& tensor_map){
        // C++ 11 traverse
        // for(auto& kv : tensor_map){}
        // C++ 98 traverse
        for(auto it = tensor_map.begin(); it != tensor_map.end(); it++){
            if(isValid(it->second)){
                insert(it->first, it->second);
            }
            else{
                // TODO：add a reminder info
                LLM_CHECK_WITH_INFO(isValid(it->second), fmtstr("%s is not a valid tensor, skipping insert into TensorMap", it->first.c_str()));
            }
        }
    }

    ~TensorMap(){
        tensor_map_.clear(); // 调用成员函数
    }


    // 返回存储的Tensor<T> 个数
    inline size_t size() const {
        return tensor_map_.size();
    }

    // 判断是否存在某一个key对应的tensor
    inline bool isExist(const std::string& key) const{
        return tensor_map_.find(key) != tensor_map_.end();
    }

    // 判断该tensor是否有效：即该tensor是否为空
    inline bool isValid(const Tensor* tensor){
        return tensor->size() > 0;
    }

    // 增
    inline void insert(const std::string& key, Tensor* value){
        // TODO: add a check to check key is unique and value is valid
        // tensor_map_.insert({key, value});
        tensor_map_[key] = value;
    }

    inline void insert(const std::pair<std::string, Tensor*> p){
        tensor_map_.insert(p);
    }

    // 删

    // 改

    // 查
    inline Tensor* at(const std::string& key){
        // TODO: add a check to check key is existed
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                                key.c_str(),
                                                vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor* operator[](const std::string& key){
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                                key.c_str(),
                                                vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }
    
    // TODO: Now cant use get* function in TensorMap struct, cause the value is Tensor*, not TensorWrapper,need to enhance

    // for debug
    std::vector<std::string> keys() const
    {
        std::vector<std::string> key_names;
        for(auto& kv : tensor_map_){
            key_names.push_back(kv.first);
        }
        return key_names;
    }

    // 打印出tensormap中的所有key
    std::string toString()
    {
        std::stringstream ss;
        ss << "{";
        std::vector<std::string> key_names = keys();
        for(size_t i = 0; i < tensor_map_.size(); ++i){
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if(i < tensor_map_.size()-1){
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();    // 返回字符串
    }
};