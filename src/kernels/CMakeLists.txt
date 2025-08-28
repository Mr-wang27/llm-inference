# 该目录将所有的kernel全部编译成静态库
# 语句	                                作用	                                    是否必须
# CUDA_SEPARABLE_COMPILATION	            支持 device 代码分离编译	                有 kernel/device function 时建议开
# POSITION_INDEPENDENT_CODE	            生成 PIC 位置无关代码	                    静态库将来链接到 so 时必须开
# CUDA_RESOLVE_DEVICE_SYMBOLS	            解决 device 符号链接问题	                有多个 cu 文件 / 库时建议开

# 常规写法：在编译kernel成静态库的时候，需要将下面的这些开关打开
add_library(embeddingFunctor STATIC input_embedding.cu)
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_library(cal_paddingoffset STATIC cal_paddingoffset.cu)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cal_paddingoffset PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(rmsnorm STATIC rmsnorm_kernel.cu)
set_property(TARGET rmsnorm PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET rmsnorm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rmsnorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_library(build_casual_mask STATIC build_causal_mask.cu)
set_property(TARGET build_casual_mask PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET build_casual_mask PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET build_casual_mask PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_library(cublasWrapper STATIC cublas_utils.cc)
# set_property(TARGET cublasWrapper PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_library(linear STATIC linear.cu)
set_property(TARGET linear PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET linear PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET linear PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
# linear 需要使用cublasWrapper，所以在创建该静态库的时候，要把cublasWrapper链接上
target_link_libraries(linear PUBLIC -lcudart -lcublas cublasWrapper)



add_library(qkv_bias_and_rope STATIC qkv_bias_and_RoPE.cu)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET qkv_bias_and_rope PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_library(concat_kv STATIC concat_past_kv.cu)
set_property(TARGET concat_kv PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET concat_kv PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET concat_kv PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(repeat_kv STATIC repeat_kv.cu)
set_property(TARGET repeat_kv PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET repeat_kv PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET repeat_kv PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(mask_softmax STATIC attn_softmax_kernel.cu)
set_property(TARGET mask_softmax PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET mask_softmax PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET mask_softmax PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(fused_transpose_and_remv_pad STATIC fused_transpose_and_remv_pad.cu)
set_property(TARGET fused_transpose_and_remv_pad PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET fused_transpose_and_remv_pad PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET fused_transpose_and_remv_pad PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(fused_addresidual_norm STATIC fused_addresidual_norm.cu)
set_property(TARGET fused_transpose_and_remv_pad PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET fused_transpose_and_remv_pad PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET fused_transpose_and_remv_pad PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_library(act STATIC act_kernel.cu)
set_property(TARGET act PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET act PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET act PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(fused_decoder_self_attention STATIC fused_decoder_self_attention.cu)
set_property(TARGET fused_decoder_self_attention PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET fused_decoder_self_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET fused_decoder_self_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(topk STATIC topK.cu)
set_property(TARGET topk PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET topk PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET topk PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)




add_library(sampling STATIC sampling.cu)
set_property(TARGET sampling PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET sampling PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)



add_library(add_residual STATIC add_residual.cu)
set_property(TARGET add_residual PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET add_residual PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET add_residual PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)