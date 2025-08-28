#include <iostream>
#include "src/layers/ffn/ffn.h"
#include "src/utils/debug_utils.h"

template <typename T>
LLaMAFFNLayer<T>::LLaMAFFNLayer(int head_num,
                                int head_size,
                                int inter_size,
                                cudaStream_t stream,
                                cublasWrapper* cublas_wrapper,
                                BaseAllocator* allocator)
    : head_num(head_num), head_size(head_size), inter_size(inter_size), hidden_units(head_num * head_size),stream(stream), cublas_wrapper(cublas_wrapper), allocator(allocator){}



template <typename T>
void LLaMAFFNLayer<T>::allocForForward(LLaMAAttentionDynParams& params)
{
    // 用于prefill阶段
    // 分配显存
    int num_tokens = params.num_tokens;
    DataType type = getTensorType<T>();
    SwiGLU_input = new TensorWrapper<T>(Device::GPU, type, {num_tokens, 2, inter_size});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * num_tokens * 2 * inter_size, false);
    down_proj_input = new TensorWrapper<T>(Device::GPU, type, {num_tokens, inter_size});
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * num_tokens * inter_size, false);
}


template <typename T>
void LLaMAFFNLayer<T>::allocForForward(int batch_size)
{
    // 用于self decoder 阶段
    DataType type = getTensorType<T>();
    SwiGLU_input = new TensorWrapper<T>(Device::GPU, type, {batch_size, 2, inter_size});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * batch_size * 2 * inter_size, false);
    down_proj_input = new TensorWrapper<T>(Device::GPU, type, {batch_size, inter_size});
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * batch_size * inter_size, false);
}


template <typename T>
void LLaMAFFNLayer<T>::freeBuf()
{
    allocator->Free(SwiGLU_input->data, false);
    DeviceSyncAndCheckCudaError();
    allocator->Free(down_proj_input->data, false);
    DeviceSyncAndCheckCudaError();
}


template <typename T>
void LLaMAFFNLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAFFNWeights<T>& weights, LLaMAAttentionDynParams& params)
{
    if(params.num_tokens == 0){     // self decoder阶段
        allocForForward(params.batch_size);
    }
    else{       // prefill阶段
        allocForForward(params);
    }

    Tensor* ffn_input = inputs["ffn_input"];
    Tensor* ffn_output = outputs["ffn_output"];
    count += 1;     // used to record layer index currently, 如果现在是第一次调用，这个layer,则count=0
    bool is_ctx = params.is_ctx;    // 是否是prefill阶段
#ifdef SAVE_DATA
    save_tensor(ffn_inputs->as<T>(), "ffn_input.bin", count);
#else
#endif

    // 1. fusedGateUp proj
    launchLinearGemm(ffn_input->as<T>(), weights.gateAndup, SwiGLU_input, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(SwiGLU_input, "SwiGLU_input.bin", count);
#else
#endif

    // 2. SwiGLU
    launchAct(SwiGLU_input, down_proj_input);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(down_proj_input, "down_proj_input.bin", count);
#else
#endif

    // 3. down proj
    launchLinearGemm(down_proj_input, weights.down, ffn_output->as<T>(), cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    
    this->freeBuf();
}

template class LLaMAFFNLayer<float>;
template class LLaMAFFNLayer<half>;
