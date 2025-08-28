// #pragma once
#include "src/models/basemodel.h"
#include "src/models/llama/llama_params.h"
#include "src/weights/llama/llama_weights.h"
#include "src/layers/decoder/context_decoder.h"
#include "src/layers/decoder/self_decoder.h"
#include "src/kernels/input_embedding.h"
#include "src/kernels/topK.h"
#include "src/kernels/sampling.h"
#include "src/models/tokenizer.h"
#include "src/utils/debug_utils.h"
#include "src/utils/params.h"

template <typename T>
class Llama: public BaseModel{
private:
    const int head_num;
    const int kv_head_num;
    const int head_size;
    const int inter_size;
    const int num_layers;
    int vocab_size;
    int vocab_size_padded;  // 不清楚这个参数是干啥的
    float rmsnorm_eps = 1e-5;
    const int hidden_units;
    // 模型在训练阶段便限定了max_seq_len，但是在推理的过程中，可以通过RoPE等相对位置编码的扩展支持更长的max_seq_len,但是越往后，模型效果可能会越差，幻觉增多
    // 另外一点，就是推理时，受限于推理硬件，越长的max_seq_len,需要越多的kv cache,因此，硬件显存会限制推理时的max_seq_len，因此有时需要进行调整
    const int max_seq_len;  //  self defined:最大支持的上下文长度
    int output_token_limit = 256;   // self defined;    一次query推理最大输出的token数
    
    int pad_token_id = 0;   // 依据模型在训练阶段的配置，也就是所采用的tokenizer
    int bos_token_id = 1;   // (begin of sequence)这个token代表：序列开始的标志
    int eos_token_id = 2;   // (end of squence)

    int layer_id = 0;       // 初始化为0
    int batch_size = 1;     // 目前只支持单batch推理
    int beamwidth = 1;      // beam search 在topK kernel中是支持的，因此如果beamwidth=1，表示greedy采样策略
    int BlockPerBeam = 0;   // needed by topK kernel

    int index = 0;  // 用于记录当前query的token生成step,index=0表示正在生成第一个token的推理中，此时是prefill阶段，index > 0表示在self decoder阶段
    std::string prompt = "";    // self defined or not; 即内置的一些prompt

    Tokenizer tokenizer;    // 
    LlamaWeight<T>* llama_weights;
    LlamaSelfDecoder<T>* self_decoder;
    LlamaContextDecoder<T>* context_decoder;

    int K = 4;  // K of topK sort
    TensorWrapper<int>* step;   // self decoder需要：历史上下文长度+当前轮次query长度+当前推理阶段已生成的token数
    TensorWrapper<T>* output_rmsnorm_weight;    // 最后一层decoder layer的ffn之后，还有一个rmsnorm
    TensorWrapper<int>* layer;  // self/context decoder均需要
    TensorWrapper<T>* context_decoder_input;
    TensorWrapper<T>* context_decoder_output;
    TensorWrapper<T>* context_decoder_lmhead_input; // 这个lmhead_input是context_decode_output经过rmsnorm之后的数据
    TensorWrapper<T>* decoder_input;    // self decoder阶段
    TensorWrapper<T>* decoder_output;   // self decoder阶段

    TensorWrapper<int>* input_ids;      // 传入给Embedding kernel的输入数据
    TensorWrapper<int>* input_length;   // 句子长度
    TensorWrapper<int>* history_length; // 历史对话长度
    TensorWrapper<int>* context_length; // 总的上下文长度： history_length + input_length， 用于context_decoder
                                        // 注意：这里的input_length是cur_query_len, 不包括已生成的token数
    TensorWrapper<T>* all_k_cache;
    TensorWrapper<T>* all_v_cache; 
    TensorWrapper<T>* unused_residual;  // 因为rmsnorm kernal在实现的时候，是融合了add residual算子的。但是在最后一层的rmsnorm中并不需要residual,因此这里需要一个占位的residual,只需要将这个residual初始化为0即可

    // used by sampling
    IntDict int_params_of_sample;       // 采样层需要的int参数， end_token, vocab_size
    TensorWrapper<T>* probs;            // 模型层的输出，也就是采样层的linear输出：[batch_szie, beamwidth, vocab_size]
    TensorWrapper<int>* token_ids;      // 新生成的token ids
    TensorWrapper<int>* sequence_lengths;   // 记录在生成过程中的当前sequence length，     sequence_length = context_length + 已生成的token数    // 这个其实并没有使用 
    TensorWrapper<bool>* is_finished;
    TensorWrapper<int>* topk_id;
    TensorWrapper<T>* topk_val;
    TensorWrapper<int>* final_topk_id;
    TensorWrapper<T>* final_topk_val;

    // pinned or not pinned CPU buffers
    int* h_input_ids_buf_{};    // 初始化指针为nullptr
    int* h_input_length_buf_{};
    int* h_history_length_buf_{};
    int* h_context_length_buf_{};   // context_length = history_length + input_length
    int* h_sequence_lengths_{};     // sequence_length = context_length + 已生成的token数    // 这个其实并没有使用
    bool* h_finished_buf_{};
    int* h_output_ids_{};   // 生成的token_ids


public:
    Llama() = default;
    Llama(int head_num,
          int kv_head_num,
          int head_size,
          int inter_size,
          int num_layers,
          int vocab_size,
          const LLaMAAttentionStaticParams& attn_static_params,
          int max_seq_len,
          cudaStream_t stream,
          cublasWrapper* cublas_wrapper,
          BaseAllocator* allocator,
          cudaDeviceProp* cuda_device_prop):    // 先调用基类的构造函数
        BaseModel(stream, cublas_wrapper, allocator, cuda_device_prop),
        head_num(head_num),
        kv_head_num(kv_head_num),
        head_size(head_size),
        inter_size(inter_size),
        num_layers(num_layers),
        vocab_size(vocab_size),
        vocab_size_padded(vocab_size),
        hidden_units(head_num * head_size),
        max_seq_len(max_seq_len)
    {
    /*
        初始化相关权重以及层
        Tokenizer tokenizer;    // 
        LlamaWeight<T>* llama_weights;
        LlamaSelfDecoder<T>* self_decoder;
        LlamaContextDecoder<T>* context_decoder;
    */
        int_params_of_sample.insert({"vocab_size", vocab_size});
        int_params_of_sample.insert({"end_id", eos_token_id});
        layer = new TensorWrapper<int>(CPU, DataType::INT32, {1}, &layer_id);      // TensorWrapper封装的layer,用于传入kernel的launch函数获取layer
                                                                                    // 会维护layer中的数据layer_id
        llama_weights = new LlamaWeight<T>(head_num,
                                        kv_head_num,
                                        head_size,
                                        inter_size,
                                        vocab_size,
                                        num_layers,
                                        /*attn_bias*/false,
                                        getWeightType<T>());

        self_decoder = new LlamaSelfDecoder<T>(head_num,
                                            kv_head_num,
                                            head_size,
                                            inter_size,
                                            num_layers,
                                            attn_static_params,
                                            rmsnorm_eps,
                                            stream,
                                            cublas_wrapper,
                                            allocator);

        context_decoder = new LlamaContextDecoder<T>(head_num,
                                                    kv_head_num,
                                                    head_size,
                                                    inter_size,
                                                    num_layers,
                                                    attn_static_params,
                                                    rmsnorm_eps,
                                                    stream,
                                                    cublas_wrapper,
                                                    allocator);

        // 只需要分配buffer即可，buffer中的数据在kernel运行的时候会进行填充，也就是这些buffer是预先分配出来，然后给kernel使用，用以存储计算的中间结果或者结果
        // 至于模型的权重，需要手动调用loadWeights加载
        // 这里只要创建的这个模型实例存在，就一直保持这些分配的buffer存在，以及初始化这些层的时候所分配的显存、内存空间一直存在
        // 因为之哟啊进行推理，就会需要这么多内存空间，避免每次推理的时候都要先进行显存空间的分配
        allocateCPUBuffer(1);   // batch_size = 1
        allocateGPUBuffer(1);
    }


    ~Llama(){
        this->free();
    }

    void loadTokenizer(std::string file){
        tokenizer.Initialize(file);
    }

    void loadWeights(std::string file){
        llama_weights->loadWeights(file);
    }

    void loadWeightsFromDummy(){
        llama_weights->loadWeightsFormDummy();
    }

    void allocateCPUBuffer(int max_batch_size);
    void allocateGPUBuffer(int max_batch_size);
    
    void free();


    // 其实因为这里只支持单batch(batch_size=1)的推理，因此这里的这些std::vector中，其实都只有一个string
    // history: 历史上下文       round：轮次        input：当前轮次的query
    std::vector<std::string> MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

    std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output);   // 更具当前轮次生成的token更新hstory

    // 单次请求响应
    std::string Response(const std::vector<std::string>& input, CallBack PrintRes);

    int MakeOutput();   // nothing to do

    // 下面是llama前处理与后处理的一些kernel
    void inputEmbedding(TensorWrapper<int>* input_ids, TensorWrapper<T>* decoder_input);
    void InitializeForContextDecoder(IntDict& int_params_first_token);  // 在Responseh中构造该参数，以便每轮对话维护推理过程中动态的参数(句子长度，历史上下文长度、kv cache等信息)
    int firstTokenGen(LLaMAAttentionDynParams& dparams, IntDict& int_params_first_token);
    void InitializeForSelfDecoder();        // 不需要做什么
    int continueTokenGen(LLaMAAttentionDynParams& dparams);
    int LMHeadAndTopKSample(TensorMap& decoder_outputs);

};

template class Llama<float>;