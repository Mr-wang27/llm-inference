#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/weights/llama/embedding_weights.h"
template <typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // (batchsize, seq_len)                 INT [token num]
                          TensorWrapper<T>* output,         // (batchsize, seq_len, hidden_dim)     FP32 [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T>* embed_table); // 权重参数(vocabsize, hidden_dim)      FP32 [vocal_size, hidden_size]

















// 由于CUDA语法不能出现在CPP文件中
// 所以需要把CUDA kernel卸载.cu文件中
// 然后由于cpp文件需要调用该CUDA kernel，所以需要kernel启动函数
// 所以只好将kernel启动函数也写在.cu中
// 然后再添加一个.h文件，用于声明该kernel的启动函数
// 所以其他cpp文件需要调用该kernel执行任务的时候，只需要include这个头文件即可
// 最后，由于kernel需要支持泛型，因此写的是模板函数
// 所以在.cu文件中需要将所写的kernel启动函数的模板显示实例化

// 因为头文件中只是定义了kernel的启动函数，并没有调用，然后kernel的启动函数是模板函数
// 因此在真正调用该kernel启动函数前，并不会实例化模板，因此include这个头文件的cpp文件在链接的时候，就会去找这个kernel启动函数
// 但是因为没有调用这个kernel启动函数，编译器也就不会实例化这个模板函数，所以连接的时候，这个cpp文件就找不到这个kernel启动函数的符号
// 导致链接错误。
// 所以需要显示的实例化模板，在.cu文件中。


/*
    真正的原因在于：
        1. 模板函数调用处必须能看到模板定义，否则不会生成代码。
        2. cuda语法不能出现在.cpp文件中

    如果将kernel启动的模板函数定义在.cu文件(kernel.cu)中,然后使用一个头文件声明kernel启动函数: kernel.h
    那么当a.cpp文件#inlcue "kernel.h"的时候，因为模板函数的定义在kernel.cu中，并没有在kernel.h中，而且模板函数调用处必须能看到模板定义，才会生成代码
    而这个情况,a.cpp看不到模板函数的定义，所以不会生成代码，所以在链接的时候，需要链接kernel启动函数的时候，就会出现来链接错误，不能找到kernel函数启动的函数符号

    所以就必须在.cu文件中，显示的对kernel启动函数进行显示的实例化

    

    另一个方案：就是将kernel启动的模板函数定义在kernel.h中，kernel.cu文件中只定义kernel。但是这个方案是行不通的。
    原因在于:cuda语法不能出现在.cpp文件中。
    当a.cpp文件#include "kernel.h"，此时kernel启动的模板函数就会被复制到a.cpp文件中，而kernel.h中有kernel启动函数，该函数会用到cuda语法
    此时就会报错，因为cpp文件中不能出现cuda语法。

    所以只能按照最上面的方式进行，将kernel定义以及kernel启动函数定义在.cu文件中，然后在.cu文件中显示实例化模板
    然后再用.h文件声明kernel启动函数
    其他文件需要使用该kernel的时候，只需要include这个.h文件即可
*/