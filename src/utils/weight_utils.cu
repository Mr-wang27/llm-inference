#include "src/utils/weight_utils.h"

// 转换类型
template <typename T_OUT, typename T_IN>
inline __device__ T_OUT type_cast(T_IN val){
    return val;
}
template <>
inline __device__ float type_cast(half val){
    return __half2float(val);
}
template <>
inline __device__ half type_cast(float val){
    return __float2half(val);
}

// 分配权重显存
template <typename T>
void GPUMalloc(T** ptr, size_t size)
{
    LLM_CHECK_WITH_INFO(size >= (size_t)0, "Ask cudaMalloc size " + std::to_string(size) + "< 0 is invalid.");
    CHECK(cudaMalloc((void**)(ptr), sizeof(T) * size));
}
template void GPUMalloc(float** ptr, size_t size);
template void GPUMalloc(half** ptr, size_t size);

template <typename T>
void GPUFree(T* ptr)
{
    if(ptr != nullptr){
        CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
}
template void GPUFree(float* ptr);
template void GPUFree(half* ptr);

// 将权重从CPU加载到GPU
template <typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size)
{
    CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}
template void cudaH2Dcpy(float* tgt, const float* src, const size_t size);
template void cudaH2Dcpy(half* tgt, const half* src, const size_t size);

// kernel 函数，转换数据类型
template <typename T_IN, typename T_OUT>
__global__ void type_conversion(T_OUT* dst, const T_IN* src, const int size)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread_nums = blockDim.x * gridDim.x;
    for(int index = gtid; index < size; index += total_thread_nums){
        dst[index] = type_cast<T_OUT, T_IN>(src[index]);
        // type_cast<T_OUT>(src[index]),        这样省略模板参数T_IN也可以，这样写就相当于让编译器自动推导T_IN的类型
    }
}
// launch函数
template <typename T_IN, typename T_OUT>
void cuda_type_conversion(T_OUT* dst, const T_IN*src, const int size)
{
    // 这里的size是数据量
    dim3 grid(128);
    dim3 block(128);
    type_conversion<T_IN, T_OUT><<<grid, block, 0, 0>>>(dst, src, size);
}
template void cuda_type_conversion(float* dst, const half* src, const int size);
template void cuda_type_conversion(half* dst, const float* src, const int size);
/*
在 CUDA C++ 中，当你使用 <<<grid, block, shared_mem_size, stream>>> 语法启动一个核函数（kernel）时，括号里的参数有特定的含义。你提到的 0, 0 对应于最后两个参数：

第三个参数（0）：shared_mem_size
这个参数指定了为每个块（block）动态分配的共享内存（shared memory）的大小，单位是字节（bytes）。

共享内存是一种高速、低延迟的片上内存，可以在同一个线程块内的线程之间共享数据。它比全局内存（global memory）快得多，非常适合用于线程块内部的数据共享和同步。
如果你不需要动态分配共享内存（所有共享内存都在编译时声明，或者根本不使用共享内存），那么这个参数就设置为 0，表示不需要额外的动态共享内存。
第四个参数（0）：stream
这个参数指定了核函数将在哪个 **CUDA 流（CUDA stream）**中执行。

CUDA 流是设备上操作（如核函数启动、内存拷贝）的序列。同一个流中的操作会按照它们被添加的顺序依次执行。
不同的流可以并发执行（如果硬件资源允许），从而实现异步操作和重叠计算与数据传输。
如果你将此参数设置为 0（或者 nullptr），则表示核函数将在**默认流（default stream）**中执行。默认流是一个特殊的流，它与主机（CPU）的同步行为不同于非默认流。在大多数简单的 CUDA 程序中，使用默认流就足够了。
*/



// from FT code
// loads data from binary file. If it succeeds, returns a non-empty (shape size) vector. If loading fails or
// the product of the elements in shape is 0, this function will return an empty vector.
template <typename T>
std::vector<T> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    if(shape.size() > 2){
        printf("[ERROR]: shape should have less than two dims \n");
    }
    size_t dim0 = shape[0], dim1 = 1;
    if(shape.size() == 2){
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if(size == 0){
        std::cout << "shape is zero, skip loading weight from file: " << filename << std::endl;
        return std::vector<T>();
    }


/*

    声明一个名为 in 的 std::ifstream 类型的对象。
    std::ios::in: 这是一个文件打开模式标志，表示以**输入（读取）**模式打开文件。
    std::ios::binary: 这是一个文件打开模式标志，表示以二进制模式打开文件 
    | (按位或运算符): 这个运算符将两个或多个文件打开模式标志组合在一起。在这里，它表示同时使用“输入模式”和“二进制模式”来打开文件。
*/
    std::vector<T> host_array(size);        // size个数据的数组
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(!in.is_open()){// 文件未成功打开
        std::cout << "file: " << filename << "cannot be opened, loading model failes!" << std::endl; 
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size; // sizeof(T):返回T类型的bytes数， sizeof(float)返回4
    in.seekg(0, in.end);
    std::streampos fileSize = in.tellg();   // 获取当前读取位置，即文件大小
                                            // std::streampos一般是size_t或long long。
    if(fileSize != loaded_data_size){
        std::cout << "file: " << filename << "file size is not macth shape size, loading model failes!" << std::endl;
        return std::vector<T>();
    }

    in.seekg(0, in.beg);    // 将文件流的读取指针指向到文件的第0个字节处
    std::cout << "Read" << std::to_string(loaded_data_size) << " bytes from " << filename << std::endl;
    in.read((char*)host_array.data(), loaded_data_size);    // 因为是以二进制格式读取文件，一个字节一个字节进行读取，因此需要将vector的原始指针转换未char*再读取    

    size_t in_get_size = in.gcount();   // 这行代码用于获取上一次未格式化输入操作（如 read(), get(), getline() 等）从输入流中实际读取的字符（或字节）数量。
    if(in_get_size != loaded_data_size){
        std::cout << "file: " << filename << "loaded data size is not match shape size, loading model failes!" << std::endl;
        return std::vector<T>();    
    }
    in.close();
    // if we succeed, return an arry with values.
    return host_array;
}


// template <typename T_OUT, typename T_FILE, bool same>
// struct loadWeightFromBin{
//     // 通用实现为空，定义接口，只是用特化版本
// };       原型定义在头文件中

template <typename T_OUT, typename T_FILE>
struct loadWeightFromBin<T_OUT, T_FILE, true>{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename){
        // 从文件中加载数据，放在host_arry数组中
        std::vector<T_FILE> host_array = loadWeightFromBinHelper<T_FILE>(shape, filename);
        if(host_array.empty()){
            return; // 没有加载到数据
        }
        cudaH2Dcpy(ptr, host_array.data(), host_array.size());  // host_array.data()拿到原始指针
        return;
    }
};


template <typename T_OUT, typename T_FILE>
struct loadWeightFromBin<T_OUT, T_FILE, false>{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename){
        std::vector<T_FILE> host_array = loadWeightFromBinHelper<T_FILE>(shape, filename);
        if(host_array.empty()){
            return ;
        }
        T_FILE* ptr_tmp;
        GPUMalloc(&ptr_tmp, host_array.size());
        cudaH2Dcpy(ptr_tmp, host_array.data(), host_array.size());
        // 启动kernel，做element-wise的数据类型转换
        cuda_type_conversion(ptr, ptr_tmp, host_array.size());
        GPUFree(ptr_tmp);
        return ;
    }
};


template struct loadWeightFromBin<float, float, true>;
template struct loadWeightFromBin<half, half, true>;
template struct loadWeightFromBin<float, half, false>;
template struct loadWeightFromBin<half, float, false>;