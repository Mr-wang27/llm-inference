#include "src/models/llama/llama.h"



template <typename T>
void Llama<T>::allocateCPUBuffer(int batch_size)
{
    // allocator在BaseModel中继承，但是现在还没有实现BaseModel
    int seq_length = 13;    // 当前input_ids的长度
    h_input_ids_buf_ = allocator->Malloc(h_input_ids_buf_, sizeof(int) * seq_length, true);
    h_input_length_buf_ = allocator->Malloc(h_input_length_buf_, sizeof(int) * batch_size, true);
    h_history_length_buf_ = allocator->Malloc(h_history_length_buf_, sizeof(int) * batch_size, true);
    h_context_length_buf_ = allocator->Malloc(h_context_length_buf_, sizeof(int) * batch_size, true);
    h_sequence_lengths_ = allocator->Malloc(h_sequence_lengths_, sizeof(int) * batch_size, true);       // 这个其实并没有使用
    h_finished_buf_ = allocator->Malloc(h_finished_buf_, sizeof(bool) * batch_size, true);
    h_output_ids_ = allocator->Malloc(h_output_ids_, sizeof(int) * batch_size, true);
    // 初始化finished为false(0)
    for(int i = 0; i < batch_size; i++){
        h_finished_buf_[i] = false;
    }
}

// alloc GPU buffer
template <typename T>
void Llama<T>::allocateGPUBuffer(int batch_size)
{
    int num_tokens = 13;    // 硬编码，用于测试, 如
    DataType data_type = getTensorType<T>();
    DataType type_int = getTensorType<int>();
    WeightType weight_type = getWeightType<T>();
    // 初始化需要的一些GPU上面的中间buffer， 
    // prefill阶段, 构造初始信息
    input_ids = new TensorWrapper<int>(Device::GPU, type_int, {num_tokens});
    input_length = new TensorWrapper<int>(Device::GPU, type_int, {batch_size});
    history_length = new TensorWrapper<int>(Device::GPU, type_int, {batch_size});
    context_length = new TensorWrapper<int>(Device::GPU, type_int, {batch_size});   // 这三个是context decoder阶段使用

    // step为当前句子的总长度
    // 这里不支持continuse batching,仅支持Dynamic batching，而且没有做PD分离,所以只需要一个step或者layer的数据来维护当前batch中每个序列的step与layer
    step = new TensorWrapper<int>(Device::CPU, type_int, {1});  // Slef deocder 阶段使用
    layer = new TensorWrapper<int>(Device::CPU, type_int, {1}, &layer_id);  // 赋予了数据指针

    // 下面要先将input_ids经过embedding kernel,变成context_decoder的输入
    context_decoder_input = new TensorWrapper<T>(Device::GPU, data_type, {num_tokens, hidden_units});
    // 然后得到context_decoder 的输出 （context_decoder = ContextAttn + FFN + AddResidual）
    context_decoder_output = new TensorWrapper<T>(Device::GPU, data_type, {num_tokens , hidden_units});
    // 然后要经过rmsnorl kernel, 该kernel在llama类中，所以需要在该类中加载权重, 但是所有的权重在llama_weights中已经进行加载，所以这里只需要将llama_weights中加载的数据转移到这里来使用
    output_rmsnorm_weight = new TensorWrapper<T>(Device::GPU, data_type, {hidden_units}, llama_weights->out_rmsnorm_weight.gamma);
    // 加载了最后一个rmsnorm的权重之后，需要经过该rmsnom kernel, 需要一个用于占位的residual以及该kernel的输出，即lmhead的输入
    unused_residual = new TensorWrapper<T>(Device::GPU, data_type, {num_tokens, hidden_units});
    // 这里需要注意的是，无论是在prefill阶段还是在self decoder阶段，需要的都只是最后一个token的预测输出，因此后面的linear层(分类器)我们也只需要德得到最后一个token的分类结果即可。其他的可以丢弃
    context_decoder_lmhead_input = new TensorWrapper<T>(Device::GPU, data_type, {batch_size, hidden_units});
    // 然后经过这个lmhead分类头的输出即为模型的logits输出，也作为topK的输入
    probs = new TensorWrapper<T>(Device::GPU, data_type, {batch_size, vocab_size});
    // 下面就需要经过topK以及Sampling kernel
    // 需要注意的是tokpK kernel 是利用的两次启动kernel来完成topK排序
    topk_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, beamwidth, BlockPerBeam, K});
    topk_val = new TensorWrapper<T>(Device::GPU, data_type, {batch_size, beamwidth, BlockPerBeam, K});
    final_topk_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, beamwidth, K});
    final_topk_val = new TensorWrapper<T>(Device::GPU, data_type, {batch_size, beamwidth, K});
    // 下面是smapleing kernel需要的参数等，上面的final_topk_id和final_topk_val作为sampling的输入
    sequence_lengths = new TensorWrapper<int>(Device::GPU, type_int, {batch_size});
    is_finished = new TensorWrapper<bool>(Device::GPU, type_int, {batch_size});
    token_ids = new TensorWrapper<int>(Device::GPU, type_int, {batch_size});

    // 上面是一个完整的context decoder流程
    // 下面将进行self decoder的流程，有一些context decoder和self decoder阶段一致的buffer
    // input_ids:[batch_size] ---经过embedding kernel---> [batch_size, hidden_units] 作为slef decoder的输入
    // self decoder阶段的输入，需要经过rmsnorm
    decoder_input = new TensorWrapper<T>(Device::GPU, data_type, {batch_size, hidden_units});
    decoder_output = new TensorWrapper<T>(Device::GPU, data_type, {batch_size, hidden_units});
    // 然后要经过最后一个rmsnorm kernel， 此时该kernel的参数与context 阶段一致
    // 输入与输出也一致，输入为：decoder_output， 输出为：context_decoder_lmhead_input
    // 然后后面的topK与Sampling 与context阶段一致

    // 最后还有kv cache
    all_k_cache = new TensorWrapper<T>(Device::GPU, data_type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size});
    all_v_cache = new TensorWrapper<T>(Device::GPU, data_type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size});


    // 下面为上诉的一些需要许配分数据空间的TensorWrapper分配数据指针
    input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * num_tokens, false);
    input_length->data = allocator->Malloc(input_length->data, sizeof(int) * batch_size, false);
    history_length->data = allocator->Malloc(history_length->data, sizeof(T) * batch_size, false);
    context_length->data = allocator->Malloc(context_length->data, sizeof(T) * batch_size, false);
    // step的数据每轮对话的数据都不一样，在response中进行数据指针的指定
    step->data = allocator->Malloc(step->data, sizeof(int) * 1, true);
    context_decoder_input->data = allocator->Malloc(context_decoder_input->data, sizeof(T) * num_tokens * hidden_units, false);
    context_decoder_output->data = allocator->Malloc(context_decoder_output->data, sizeof(T) * num_tokens * hidden_units, false);
    unused_residual->data = allocator->Malloc(unused_residual->data, sizeof(T) * num_tokens * hidden_units, false);
    context_decoder_lmhead_input->data = allocator->Malloc(context_decoder_lmhead_input->data, sizeof(T) * batch_size * hidden_units, false);
    probs->data = allocator->Malloc(probs->data, sizeof(T) * batch_size * vocab_size, false);
    topk_id->data = allocator->Malloc(topk_id->data, sizeof(int) * batch_size * beamwidth * BlockPerBeam * K, false);
    topk_val->data = allocator->Malloc(topk_val->data, sizeof(T) * batch_size * beamwidth * BlockPerBeam * K, false);
    final_topk_id->data = allocator->Malloc(final_topk_id->data, sizeof(int) * batch_size * beamwidth * K, false);
    final_topk_val->data = allocator->Malloc(final_topk_val->data, sizeof(T) * batch_size * beamwidth * K, false);
    sequence_lengths->data = allocator->Malloc(sequence_lengths->data, sizeof(T) * batch_size, false);
    is_finished->data = allocator->Malloc(is_finished->data, sizeof(T) * batch_size, false);
    token_ids->data = allocator->Malloc(token_ids->data, sizeof(int) * batch_size, false);
    decoder_input->data = allocator->Malloc(decoder_input->data, sizeof(T) * batch_size * hidden_units, false);
    decoder_output->data = allocator->Malloc(decoder_output->data, sizeof(T) * batch_size * hidden_units, false);
    all_k_cache->data = allocator->Malloc(all_k_cache->data, sizeof(T) * num_layers * batch_size * kv_head_num * max_seq_len * head_size, false);
    all_v_cache->data = allocator->Malloc(all_v_cache->data, sizeof(T) * num_layers * batch_size * kv_head_num * max_seq_len * head_size, false);

}


// free CPU and GPU buffer
template <typename T>
void Llama<T>::free()
{
    allocator->Free(h_input_ids_buf_, true);
    allocator->Free(h_input_length_buf_, true);
    allocator->Free(h_history_length_buf_, true);
    allocator->Free(h_context_length_buf_, true);
    allocator->Free(h_sequence_lengths_, true);
    allocator->Free(h_finished_buf_, true);
    allocator->Free(h_output_ids_, true);
    DeviceSyncAndCheckCudaError();

    allocator->Free(input_ids->data, false);
    allocator->Free(input_length->data, false);
    allocator->Free(history_length->data, false);
    allocator->Free(context_length->data, false);
    allocator->Free(step->data, false);
    allocator->Free(context_decoder_input->data, false);
    allocator->Free(context_decoder_output->data, false);
    allocator->Free(unused_residual->data, false);
    allocator->Free(context_decoder_lmhead_input->data, false);
    allocator->Free(probs->data, false);
    allocator->Free(topk_id->data, false);
    allocator->Free(topk_val->data, false);
    allocator->Free(final_topk_id->data, false);
    allocator->Free(final_topk_val->data, false);
    allocator->Free(sequence_lengths->data, false);
    allocator->Free(is_finished->data, false);
    allocator->Free(token_ids->data, false);
    allocator->Free(decoder_input->data, false);
    allocator->Free(decoder_output->data, false);
    allocator->Free(all_k_cache->data, false);
    allocator->Free(all_v_cache->data, false);
    DeviceSyncAndCheckCudaError();
    // layer 中的数据指针式局部的，会在外部自动销毁
    // output_rmsnorm_weight的指针是从llama_weights中赋值过来的，使用的是llama_weights中的指针，
    // 该指针空间的释放在llama_weights中进行
    
}

template <typename T>
std::vector<std::string> Llama<T>::MakeInput(const std::string &history, int round, const std::string &input) // 根据历史信息和当前输入生成promp
{
    // 返回total_sequence、history、input
    std::vector<std::string> res = {(round == 0 ? "" : history) + input, history, input};
    return res;
}

// 根据第round轮的结果制作history
template <typename T>
std::string Llama<T>::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output)   // 更具当前轮次生成的token更新hstory
{
    return (round == 0 ? prompt : history) + input + output;
}

template <typename T>
int Llama<T>::MakeOutput()
{
    // nothing tod 
}




template <typename T>
void Llama<T>::InitializeForContextDecoder(IntDict &int_params_first_token)
{
    // 这里也是只支持batch_size = 1 的推理,
    // 这里直接从llama里面拿到batch_size,是不对的，但是因为是只支持单batch推理。在llama类中设置为了1，因此这里直接从llama类中那batch_size
    int num_tokens = 0;
    for(int index = 0; index < batch_size; index++){ // 这里应该从attn_dyn_params里面那道batch_size
        h_input_length_buf_[index] = int_params_first_token["cur_input_length"];    // 如果batch_size不为1，这个函数的传入参数也应该是std::vector<IntDict>
        num_tokens += h_input_length_buf_[index];
        h_history_length_buf_[index] = int_params_first_token["history_length"];
        h_context_length_buf_[index] = int_params_first_token["context_length"];
    }
    CHECK(cudaMemcpy(input_ids->data, h_input_ids_buf_, sizeof(int) * num_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(input_length->data, h_input_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(history_length->data, h_history_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(context_length->data, h_context_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(is_finished->data, h_finished_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));    // h_finished_buf_在分配cpu buffer的时候，就进行了初始化，初始化全为false
}


template <typename T>
void Llama<T>::InitializeForSelfDecoder()
{
    // 为什么不需要在这里维护变量？因为在self阶段，并不需要维护cur_input_length， cur_input_length， cur_input_length，
    // cur_input_length： 因为slef decoder不需要做padding, 所以无需维度当前句子的长度
    // history_length： self decoder不需要进行causal mask
    // context_length。。。
    // 只需要维护step即可， 但是step在外面以及进行初始化，并且每轮对话时，会在response中进行初始化为当前句子长度，然后再推理的过程中进行维护，因此，并不需要在这里进行初始化
    // nothing to do now
}

template <typename T>
void Llama<T>::inputEmbedding(TensorWrapper<int>* input_ids, TensorWrapper<T>* decoder_input)
{
    // 启动Embedding kernel
    launchInputEmbedding<T>(input_ids, decoder_input, &(llama_weights->pre_decoder_embedding_weight));
    DeviceSyncAndCheckCudaError();
}

// LMHeadAndTopKSample

template <typename T>
int Llama<T>::LMHeadAndTopKSample(TensorMap& decoder_outputs)
{
    Tensor* decoder_output = decoder_outputs["decoder_output"];
    if(index == 0){
        TensorWrapper<T> *decoder_output_tensorwarpper = decoder_output->as<T>();   // [num_tokens, hidden_units]
        auto input_length = decoder_output_tensorwarpper->shape[0];
        auto hidden_units = decoder_output_tensorwarpper->shape[1];
        // fetch last token to handle context decoder sampling
        auto ptr = decoder_output_tensorwarpper->data + (input_length - 1) * hidden_units;
        // 指向最后一个token
        context_decoder_lmhead_input->data = ptr;   // [1, hidden_units]    * [vocab_size, hidden_units]
        // 启动linear gemm
        launchLinearGemm(context_decoder_lmhead_input,  // [1, hidden_units] for context/self decoder
                         llama_weights->post_decoder_embedding_weight,  // [vocab_size, hidden_units]
                         probs,
                         cublas_wrapper,
                         false,
                         true);
        DeviceSyncAndCheckCudaError();
    }
    else{
        // for self decoder
        // 不需要偏移到最后一个token位置上
        launchLinearGemm(decoder_output->as<T>(),
                         llama_weights->post_decoder_embedding_weight,
                         probs,
                         cublas_wrapper,
                         false,
                         true);
        DeviceSyncAndCheckCudaError();
    }

    // 获取到模型的logits: probs;       [batch_size, vocab_size]
    // 然后进行topk以及采样操作
    launchTopKforBeamSearch(probs, topk_id, topk_val, final_topk_id, final_topk_val);
    DeviceSyncAndCheckCudaError();
    // 准备采样层的参数
    int_params_of_sample.insert({"step", step->getVal()});
    int_params_of_sample.insert({"vocab_size", vocab_size});
    int_params_of_sample.insert({"end_id", eos_token_id});
    
// 要维护该数据sequence_lengths
    launchSampling(final_topk_id,       // in   [batch_size, K]
                   final_topk_val,      // in [batch_size, K]
                   sequence_lengths,    // out, +1  [batch_size]
                   is_finished,         // [batch_size]
                   token_ids,
                   int_params_of_sample);
    DeviceSyncAndCheckCudaError();

    // 将结果拷贝到cpu上
    CHECK(cudaMemcpy(h_output_ids_, token_ids->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));
    return h_output_ids_[0];    // 因为这里只支持单batch推理，所以只返回一个，否则需要返回一个vector

}



template <typename T>
int Llama<T>::firstTokenGen(LLaMAAttentionDynParams &dparams, IntDict &int_params_first_token)
{
    // llama的前向推理
    InitializeForContextDecoder(int_params_first_token);
    // 将input_ids送入embedding kernel
    inputEmbedding(input_ids, context_decoder_input);   // 输入input_ids, 输出：contxet_decoder_input
    LLM_CHECK_WITH_INFO(context_decoder_input->data != nullptr, "GPU context decoder input data is not initialized");
    LLM_CHECK_WITH_INFO(history_length->data != nullptr, "GPU history_length data is not initialized");
    LLM_CHECK_WITH_INFO(input_length->data != nullptr, "GPU input_length data is not initialized");
    LLM_CHECK_WITH_INFO(context_length->data != nullptr, "GPU context_length data is not initialized");
    LLM_CHECK_WITH_INFO(output_rmsnorm_weight->data != nullptr, "GPU output_rmsnorm_weight data is not initialized");

    // 构造TensorMap,做context decoder的前向
    TensorMap decoder_inputs{
        {"input_length", input_length},
        {"history_length", history_length},
        {"context_length", context_length},
        {"layer_id", layer},
        {"decoder_input", context_decoder_input}
        // {"output_norm_weight", output_rmsnorm_weight}        // 这个参数不需要放在context decoder的input里面
    };
    // output buffer and input buffer are shared to reuse buffer between layers
    // I dont rewrite Tensor's copy constructor, default shallow copy, that can share buffer, which is I want
    TensorMap decoder_outputs{
        {"decoder_output", context_decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    context_decoder->forward(decoder_inputs, llama_weights->llama_layer_weight, decoder_outputs, dparams);

    // 然后要经过output rmsnorm
    Tensor* decoder_output = decoder_outputs["decoder_output"];
    // decoder_output:  in&out, [bs, q_hidden_units]
    launchRMSNorm(decoder_output->as<T>(), unused_residual, llama_weights->out_rmsnorm_weight, rmsnorm_eps, true);

    save_tensor(decoder_output->as<T>(), "decoder_norm_out.bin");
    DeviceSyncAndCheckCudaError();
    // LMHeadAndTopKSample其实应该是只需decoder_outputs中的decoder_output
    int res_token_id = LMHeadAndTopKSample(decoder_outputs);
    return res_token_id;

}

template <typename T>
int Llama<T>::continueTokenGen(LLaMAAttentionDynParams &dparams)
{
    InitializeForSelfDecoder(); // nothing to do 
    inputEmbedding(input_ids, decoder_input);   // input_ids：[batch_size]  decoder_input:[batch_size, hidden_units]
    // 下面就要为self decoder构造TensorMap
    TensorMap decoder_inputs{
        {"decoder_input", decoder_input},
        {"step", step},
        {"finished", is_finished},
        {"layer_id", layer}
    };
    // (RussWong) note: 最开始是context decoder里面RoPE输出的k和v写到kv cache
    // (RussWong) note: self decoder之后每一个step都会输出kv到kv cache, 需要保证kv cache是llama class的成员, 这样就可以保证同步更新
    TensorMap decoder_outputs{
        {"decoder_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };
    self_decoder->forward(decoder_inputs, llama_weights->llama_layer_weight, decoder_outputs, dparams);

    // 然后需要经过最后一个rmsnorm
    Tensor* decoder_output = decoder_outputs["decoder_output"];
    launchRMSNorm(decoder_output->as<T>(),  // [batch_size, hidden_units]
                  unused_residual,          // [batch_size, hidden_units]
                  llama_weights->out_rmsnorm_weight,    // [hidden_units]
                  rmsnorm_eps,
                  true);
    DeviceSyncAndCheckCudaError();
    int res_token_id = LMHeadAndTopKSample(decoder_outputs);
    return res_token_id;
}



// 单轮对话， batch_size = 1
// 返回所有轮次总共的input、总共的input中的history部分、总共input中的当前轮次input部分
/*
    输入参数: input有三个string: 
        input[0]: total_input: 历史轮次的上下文历史+当前轮次的query
        input[1]: history_context: 历史轮次的上下文历史
        input[2]: cur_query: 当前轮次的query
*/
// 目前该函数返回的只是当前轮次生成的string
template <typename T>
std::string Llama<T>::Response(const std::vector<std::string>& input, CallBack PrintRes)
{
    //std::vector<int> res = tokenizer.Encode(input[2]);
    // from transformers import AutoTokenizer
    // tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer_folder")
    // prompt = "Hey, are you conscious? Can you talk to me?"
    // input_ids = tokenizer(prompt, return_tensors="pt")
    // 下行的input token ids暂时是通过以上4行huggingface python api而得，修复了tokenzier.Encode之后再用以上第5行替换    std::vector<int> res = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973};
    std::vector<int> res = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973};      // 当前轮次的input
    std::string history_str = input[1];
    std::vector<int> history_input_ids;
    if(!history_str.empty()){
        // 首轮对话的history_str为空
        history_input_ids = tokenizer.Encode(history_str);  // 每轮对话调用Response都要调用tokenizer对历史上下文进行编码
    }
    std::string total_str = input[0];
    std::vector<int> context_ids;
    if(!total_str.empty()){
        // context_ids = tokenizer.Encode(total_str);
        context_ids = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973};
    }
    for(int i = 0; i < res.size(); i++){
        h_input_ids_buf_[i] = res[i];   // 将当前轮次的query转移到cpu的input_ids_buf上
    }
    // 准备相关的数据
    int res_token_id;    // 这个是result token，即生成的token_id
    int context_length = context_ids.size();    // 历史上下文+当前轮次的query
    int history_length = history_input_ids.size();  // 历史上下文长度
    int cur_input_length = res.size();  // 当前轮次的query长度

    IntDict int_params_first_token; // 生成第一个token需要的参数，也就是context decoder阶段需要的参数，存储在字典中
    int_params_first_token["context_length"] = context_length;
    int_params_first_token["history_length"] = history_length;
    int_params_first_token["cur_input_length"] = cur_input_length;

    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 1; // 目前仅仅支持单batch推理
    // attn_dyn_params.is_ctx = flase;
    attn_dyn_params.max_k_len = context_length;// 当前batch中的历史上下文+当前query的长度的最大长度，在context decoder阶段使用，用于kv cache中
    attn_dyn_params.max_q_len = cur_input_length;   // 当前batch中，最长的query长度
    attn_dyn_params.num_layers = num_layers;
    attn_dyn_params.num_tokens = cur_input_length;
    *(step->data) = context_length; // 初始化step中数据指针的数据
    *(sequence_lengths->data) = context_length;        // step和sequence_lengths所维护的内容是一致的， 在Sampling kernel中sequence_lengths用来维护序列当前序列长度(history+cur_query+genarated_token), step在Sampling kernel中用处是作为随机种子生成随机数，
    // 然后step的数据在CPU上，维护在token生成后的cpu上的response函数中
    // sequence_lengths在kernel中进行维护。     两者的数据在各自维护完成之后应该是一致的
    
    std::string resString = ""; // 存储结果，为当前轮次对话生成的所有token string
    while(index < output_token_limit){
        // 在首轮对话的中，kv_cache是空的，在kernel的推理过程中，会逐步往kv cache 中添加kv cache
        // 因为是单batch推理，所以得到的res_token_id就只有一个int， 因此如果生成结束，则直接退出
        if(index == 0){
            res_token_id = firstTokenGen(attn_dyn_params, int_params_first_token);  // 生成第一个token_id
        }
        else{
            res_token_id = continueTokenGen(attn_dyn_params);
            if(res_token_id == eos_token_id){
                break;
            }
        }
        *(step->data) = *(step->data) + 1;  // step增加1

        std::string genString = tokenizer.Decode({res_token_id}).c_str();   // 将std::string转为c字符串
        resString += genString; // 存储生成结果字符串
        PrintRes(index, genString.c_str()); // 传入index指明生成的token index
        if(index == 0){
            // 如果是prefill阶段的生成结束，需要将input_ids所分配的内存进行削减
            // 因为在prefill阶段为input_ids 分配的内存空间是[num_tokens(batch_size*seq_len)],
            // 但是在self decoder阶段的input_ids只需要[batch_size]即可,因此，在self decoder阶段，需要重新分配input_ids以及其相关的shape等信息
            // tmp便是用于下一个self decoder阶段的生成结果
            TensorWrapper<int> tmp = TensorWrapper<int>(Device::CPU, getTensorType<T>(), {1}, &res_token_id);
            LLM_CHECK(tmp.shape != input_ids->shape);
            LLM_CHECK(tmp.dtype == input_ids->dtype);
            LLM_CHECK(tmp.location != input_ids->location);
            allocator->Free(input_ids->data, false);
            input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * 1, false);
            input_ids->shape = {1}; // 更新shape
            CHECK(cudaMemcpy(input_ids->data, tmp.data, sizeof(int) * 1, cudaMemcpyHostToDevice));
        }
        else{
            CHECK(cudaMemcpy(input_ids->data, &res_token_id, sizeof(T) * 1, cudaMemcpyHostToDevice));
        }
        index++;
    }
    PrintRes(-1, resString.c_str());
    return resString;
    
    // std::vector<std::string> result;
    // result.push_back("");
    // std::string res_context = input[0] + resString; // resullt[1]
    // result.push_back(res_context);
    // result.push_back("");
    // result[0]在make_input中得到
    // result[2]也是在make_input中得到
}