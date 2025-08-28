/*
1. 一个结构体block： ptr, size, is_allocated，
2. 使用一个std::map<int, block>来作为内存pool管理
3. 分为bigBlock和SmallBlock来管理
4. 同样map也有两个内存pool来管理着两种block
5. 实现抽象类的接口
*/

#pragma once
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include "src/memory/allocator/base_allocator.h"
#include "src/utils/macro.h"
#include <mutex>    // 引入互斥锁头文件


// I use Bytes to printf buffer size msg, because sometime I allocate <1KB buffer, which causes that display 0KB
struct CudaBigBlock{
    void* data;
    size_t size;
    bool is_allocated;
    CudaBigBlock() = default;   // 默认构造函数
    CudaBigBlock(void* data_, size_t size_, bool is_allocated_)
        : data(data_), size(size_), is_allocated(is_allocated_){}
};

struct CudaSmallBlock{
    void* data;
    size_t size;
    bool is_allocated;
    CudaSmallBlock() = default;
    CudaSmallBlock(void* data_, size_t size_, bool is_allocated_)
        : data(data_), size(size_), is_allocated(is_allocated_){}
};


class CudaAllocator: public BaseAllocator{
private:
    // {device_id, block}
    std::map<int, std::vector<CudaSmallBlock>> CudaSmallBlocksMap;
    std::map<int, std::vector<CudaBigBlock>> CudaBigBlocksMap;
    std::map<int, size_t> FreeSize; // 当前free了的内存字节数，但是还没有return 给OS的内存空间
    // 在使用[]第一次插入一个map中不存在的key时，会自动创建这个key,并将其初始化为0
    size_t total_allocated_size = 0;
    int current_dev_id;
    // std::mutex 不是可重入的（non-recursive）。
    // 因此，如果一个线程在第一次持有该锁之后，如果再次请求这个锁，就会出现死锁
    // 另一种场景就是，线程A在函数a中，持有了mutex锁，然后在函数a中调用了函数b，然后函数b中又去持有这同一个mutex锁，就会导致"递归加锁"。而mutex是不可重入的，因此也会出现死锁
    // 解决这种场景的方案就是使用std::recursive_mutex，递归锁
/*
std::recursive_mutex 与 std::mutex 一样，也是一种可以被上锁的对象，但是和 std::mutex 不同的是，std::recursive_mutex 允许同一个线程对互斥量多次上锁（即递归上锁），
来获得对互斥量对象的多层所有权，std::recursive_mutex 释放互斥量时需要调用与该锁层次深度相同次数的 unlock()，可理解为 lock() 次数和 unlock() 次数相同，
除此之外，std::recursive_mutex 的特性和 std::mutex 大致相同。

例如函数a需要获取锁mutex，函数b也需要获取锁mutex，同时函数a中还会调用函数b。如果使用std::mutex必然会造成死锁。但是使用std::recursive_mutex就可以解决这个问题。
*/
    // std::mutex mutex_;   // 定义互斥锁成员变量，用于保护所有共享资源的读写     
    // 因此这里需要使用递归锁 
    std::recursive_mutex mutex_;

public:
    // 如果基类中有默认构造函数，则派生类会自动调用基类的默认构造函数(也就是无参的构造函数)
    // 如果基类中没有默认构造函数，就需要在派生类中手动显示的调用基类的构造函数
    CudaAllocator(){
        cudaGetDevice(&current_dev_id); // 获取当前使用的cuda设备id                 // 这里， 如果在多张卡上使用这个allocator的话，需要在对于的cuda设备上声明并调用这个构造函数，然后就可以在不同的卡上使用相同的这一个allocator了。一个alloctor统一管理所有cuda 设备
    }

    ~CudaAllocator(){   // 释放这个分配器需要将该分配器中的内存pool全部释放
        // 析构时释放所有GPU内存，需要加锁防止并发访问导致崩溃
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        for(auto& it : CudaSmallBlocksMap){
            std::vector<CudaSmallBlock>& cudaSmallBlocks = it.second;
            for(int i = 0; i < cudaSmallBlocks.size(); i++){
                cudaFree(cudaSmallBlocks[i].data);
            }

            std::vector<CudaBigBlock>& CudaBigBlocks = CudaBigBlocksMap[it.first];   // 获取到这个设备的cudaBigBlocks pool
            for(int i = 0; i < CudaBigBlocks.size(); i++){
                cudaFree(CudaBigBlocks[i].data);
            }
        }
    }

    bool TryReleaseBigBlocksAndRetry(void** new_ptr, size_t size){
        // 执行到此处的话，前面已经确定了活跃的设备了，因此这里可以直接使用current_dev_id
        // 策略就是遍历该cuda 设备的大块内存空间，然后从前往后遍历，做累加，只要累加到了足够size的大块内存，
        // 就停止free,然后重新尝试分配
        
        // 该函数会修改大块内存的管理结构，需要加锁保证修改一致性
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        auto& CudaBigBlocks = CudaBigBlocksMap[current_dev_id];
        std::unordered_set<int> free_index;    // 记录需要free的big block索引
        for(int i = 0; i < CudaBigBlocks.size(); i++){
            if(size < 0){
                break;
            }
            if(!CudaBigBlocks[i].is_allocated){
                free_index.insert(i);
                size -= CudaBigBlocks[i].size;
            }
        }
        // 释放索引中的big block, 并将其从CudaBigBlocks中删除
        for(int i : free_index){
            CHECK(cudaFree(CudaBigBlocks[i].data));
        }
        // 替换新的CudaBigBlocks
        std::vector<CudaBigBlock> temp;
        for(int i = 0; i < CudaBigBlocks.size(); i++){
            if(free_index.find(i) == free_index.end()){ // 不是被free的big block
                temp.push_back(CudaBigBlocks[i]);
            }
        }
        CudaBigBlocks.clear();
        CudaBigBlocks = temp;

        // 再次尝试重新分配内存
        // 传进来的是为进行32位对齐的size
        int size_32 = (size + 31)/32 * 32;
        cudaError_t err = cudaMalloc(new_ptr, size_32);
        if(err == cudaSuccess){
            CHECK(cudaMemset(*new_ptr, 0, size_32));    // new_ptr是一个二级指针，而cudaMemset需要的是一级指针，所以这里要解引用依次
            CudaBigBlocks.push_back(CudaBigBlock(*new_ptr, size_32, true)); // 加入到big block pool 中
            total_allocated_size += size_32;
            return true;
        }
        return false;
    }



    // 写基类的暴露的纯虚函数接口
    // 这里这个ptr指针的传入是没有意义的
    void* UnifyMalloc(void* ptr, size_t size, bool is_host){
        // 接受一个指针，然后将这个指针指向赋予的空间上，
        // 1. host malloc
        if(is_host){
            ptr = malloc(size);
            memset(ptr, 0, size);
            return ptr;
        }

        // GPU 内存分配前加锁，防止多个线程同时修改内部管理结构
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        // 2. big buf, 先去big blokcs里面找空闲的(free出来且未归还给OS)
        // 设置阈值为1024*1024Bytes=1M的内存未Big buffer
        if(size > 1024*1024){
            // 去CudaBigBlocksMap里面找符合要求的block id
            cudaGetDevice(&current_dev_id);     // 获取到当前活跃的设备
            auto& CudaBigBlocks = CudaBigBlocksMap[current_dev_id];
            int block_id = -1;
            for(int i = 0; i < CudaBigBlocks.size(); i++){
                if(CudaBigBlocks[i].size >= size && CudaBigBlocks[i].is_allocated == false && CudaBigBlocks[i].size - size < 1024*1024){ // 不能比需要的内存大太多，这里要求小于1024*1024
                    if(block_id == -1 || CudaBigBlocks[i].size < CudaBigBlocks[block_id].size)
                        block_id = i;   // 只有还没有找到一个满足要求的block或者找到了下一个满足要求，且内存更小的block时，才更新block_id
                }
            }
            if(block_id != -1){
                CudaBigBlocks[block_id].is_allocated = true;
                return CudaBigBlocks[block_id].data;        // 返回这个指针
            }
            else{   // 没有找到一个符合条件的，就cudaMalloc
                void* new_ptr;
                // 在分配的时候做32位的数据对齐
                int size_32 = (size + 31)/32*32;
                cudaError_t err = cudaMalloc(&new_ptr, size_32);
                if(err == cudaSuccess){
                    CHECK(cudaMemset(new_ptr, 0, size_32));
                    total_allocated_size += size_32;
                    // 将这个指针加入到BigBlockPool中
                    CudaBigBlocks.push_back(CudaBigBlock(new_ptr, size_32, true));
                    return new_ptr;
                }
                else{
                    // 尝试释放空闲的大块，然后再申请一次
                    if(TryReleaseBigBlocksAndRetry(&new_ptr, size)){
                        return new_ptr;
                    }else{
                        std::cerr << "CudaAllocator: Failed to allocate big buffer even after releasing blocks.\n";
                        CHECK(err);   // 报错退出
                    }
                }

            }
        }
        // 3. small buf, 先去smallblocks里面找空闲的,   small_buff <= 1024*1024
        cudaGetDevice(&current_dev_id);     // 获取到当前活跃的设备
        auto& CudaSmallBlocks = CudaSmallBlocksMap[current_dev_id];
        int block_id = -1;
        for(int i = 0; i < CudaSmallBlocks.size(); i++){    // 只要找到符合要求的block就可以了
            if(CudaSmallBlocks[i].size >= size && CudaSmallBlocks[i].is_allocated == false){
                if(block_id == -1 || CudaSmallBlocks[i].size < CudaSmallBlocks[block_id].size){     // 每次找空间都要比哪里完整个pool
                    block_id = i;
                }
            }
        }
        if(block_id != -1){
            CudaSmallBlocks[block_id].is_allocated = true;
            // 这里，在第一次分配小块内存的时候，pool里面是空的，此时减去这个数值将会导致FreeSize[current_dev_id] < 0,但是这个值应该时大于等于0的。因此需要保护该值
            if(FreeSize[current_dev_id] > CudaSmallBlocks[block_id].size){     // 记录当前free而且没有分配出去，且还没有返回给os的小内存，大小
                FreeSize[current_dev_id] -= CudaSmallBlocks[block_id].size;
            }
            else{
                FreeSize[current_dev_id] = 0;
            }
            return CudaSmallBlocks[block_id].data;
        }
        // 4. 没有找到空闲的再去cudaMalloc并插入block pool
        else{
            // cudaMalloc, 同样要做对齐
            void* new_ptr = (void*)ptr;
            int size_32 = (size+31)/32 * 32;
            CHECK(cudaMalloc(&new_ptr, size_32));
            CHECK(cudaMemset(new_ptr, 0, size_32));
            total_allocated_size += size_32;
            // 将小内存块加入到CudaSmallBlocksMap中
            CudaSmallBlocks.push_back(CudaSmallBlock(new_ptr, size_32, true));
            return new_ptr;
        }
    }

    void UnifyFree(void* ptr, bool is_host)
    {
        // 如果是空指针，直接返回
        if(ptr == nullptr){
            return ;
        }
        // 如果是host的内存指针，直接free
        if(is_host){
            free(ptr);
            return;
        }

        // 释放GPU内存时加锁，防止多个线程同时访问和修改共享结构
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        // 2.清理内存碎片，就是当累积的smallbuf超出1G时， 清理那些未分配出去的smallblocks,当未分配出去的内存超过1G=1024*1024*1024Bytes时，将其返回给OS
        // 需要检查每个设备的FreeSize是否到达了需要清理的阈值
        for(auto& item : CudaSmallBlocksMap){       // item： std::pair <int, std::vector<CudaSmallBlock> >
            // 先检查SmallBlocks
            if(FreeSize[item.first] > 1024*1024*1024){
                // 清理
                std::vector<CudaSmallBlock> temp;
                for(int i = 0; i < item.second.size(); i++){
                    if(!item.second[i].is_allocated){
                        // 释放
                        cudaSetDevice(item.first);
                        cudaFree(item.second[i].data);
                    }
                    else{
                        // 正在使用的block,需要重新存储起来
                        temp.push_back(item.second[i]);
                    }
                }
                // 将temp重新给CudaSmallBlocksMap
                item.second.clear();
                item.second = temp;
                FreeSize[item.first] = 0;
            }
        }
        // 找到待free的block位置，将其设置为false.
        // 大小block都不归还给OS,除非没有在大或小的block里面找到待free的指针
        // 大的block不归还：他这里的想法时大的block一般不那么容易造成外部内存碎片
        // 所以就申请了之后直接自己拿来使用分配
        // 但是呢大的buffer容易造成内部内存碎片，就是在big buffer第一次进行free后，回到我们的big block pool
        // 之后，下一次有申请内存的请求，就从这个big block pool中找适合的block
        // 此时找到的block可能就会把需要的内存大一些，这样就造成了内存碎片
        // 但是在前面big block的分配策略中，采取的时只有大于我们申请的内存空间，且大于的内存空间小于1M（1024*1024）时
        // 才是满足我们条件的block，此时才会分配这个block出去
        // 所以其实这里产生的内部内存碎片最多就1M
        // 但是这里还有一个新的问题，就是大的block不还给OS,然后又来了一个新的big 内存申请
        // 此时在big block pool中没有找到合适的，OS也没有多余的(随着程序运行，OS上将GPU内存全部被分配了)
        // 此时就会出现内存分配失败了
        // 然而事实是，在big block big中是由能够覆盖这次内存请求的空间的，只是因为没有还给os
        // 而且要求不能大于申请内存1M的要求，导致没有分配内存成功
        // 所以这里对于big block pool内存的管理，需要添加一个：
        // 当big 内存请求从big block pool与os请求均失败的时候
        // 就查找 big block pool中的空闲内存，将空闲内存进行合并知道满足此处内存请求的申请
        // 然后再把这些block还给OS
        // 然后再次尝试从OS申请内存
        // 这部分内容需要放在UnifyMalloc中实现

        // 3.找到待free的buffer的位置，设is_allocated = false，大小block都不归还到OS，除非没有在大小block里面找到待free的ptr
        // 其实这里不用做每个设备的遍历，来查找待free的指针。因为cudafree与cudamalloc是需要在对应的同一个cuda设备上进行的
        // 也就是在那个cuda设备上进行malloc,指针就会存在这个设备的block pool上
        // free也只需要找这个设备对应的即可
        // 因为这个CudaAllocator是需要在每个使用的cuda设备上进行初始化，才能使用的
        // 然后你在那个设备上释放内存，就只能释放这个cuda设备的内存，不能在这个cuda设备上释放别的cuda设备的内存。
        // 这样做也不安全
        // 所以这里其实只要遍历这个cuda设备上的block pool 就好了
        
        cudaGetDevice(&current_dev_id);     // 获取到当前活跃的设备    
        auto& cudaSmallBlocks = CudaSmallBlocksMap[current_dev_id];
        for(int i = 0; i < cudaSmallBlocks.size(); i++){
            if(cudaSmallBlocks[i].data == ptr){
                cudaSmallBlocks[i].is_allocated = false;
                FreeSize[current_dev_id] += cudaSmallBlocks[i].size;
                return;
            }
        }
        // 如果没有在small block pool中遍历找到，就到big pool里面找
        auto& cudaBigBlocks = CudaBigBlocksMap[current_dev_id];
        for(int i = 0; i < cudaBigBlocks.size(); i++){
            if(cudaBigBlocks[i].data == ptr){
                cudaBigBlocks[i].is_allocated = false;
                return;
            }
        }
        // 如果没有在block里面找到这个指针，就直接free,但这是不安全的

        cudaFree(ptr);  
    }
};
