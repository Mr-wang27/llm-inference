#pragma once
#include <cuda.h>
#include <cuda_runtime.h>


// allocator的基类，定义一些接口
class BaseAllocator
{
public:
    /*
        只有基类的析构函数是虚函数的时候，在使用多态(基类指针指向派生类)的时候，当我们
        需要释放这个基类指针是，才能调用到派生类的析构函数进行释放，否则将直接调用到基类的析构函数
        导致派生类中的一些内存可能无法释放，从而导致内存泄露
    */
    virtual ~BaseAllocator(){}; // 基类的析构函数需要定义为虚函数。

    // unified interface for all derived allocator to alloc buffer
    template <typename T>
    T* Malloc(T* ptr, size_t size, bool is_host){
        return (T*)UnifyMalloc((void*)ptr, size, is_host);  // 返回一个T*指针， UnifyMalloc返回的是一个void指针
    }
    // 纯虚函数
    virtual void* UnifyMalloc(void* ptr, size_t size, bool is_host = false) = 0;

    template <typename T>
    void Free(T* ptr, bool is_host){
        UnifyFree((void*)ptr, is_host);
    }
    virtual void UnifyFree(void* ptr, bool is_host) = 0;

};