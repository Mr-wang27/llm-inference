# 定义一个静态库
add_library(LlamaCtxAttn STATIC context_attention.cpp)
set_property(TARGET LlamaCtxAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaCtxAttn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaCtxAttn PUBLIC
#                             -lcudart
                             -lcudadevrt
                             qkv_bias_and_rope
                             concat_kv
#                             cublasWrapper
                             linear
                             fused_transpose_and_remv_pad
                             repeat_kv
                             mask_softmax)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(LlamaCtxAttn PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()


# # 核心：使用生成器表达式设置编译选项和定义
# target_compile_options(LlamaCtxAttn PRIVATE
#     $<$<CONFIG:Debug>:-O0;-g> # Debug配置：无优化，带调试信息
#     $<$<CONFIG:Release>:-O3> # Release配置：O3优化
#     $<$<CONFIG:RelWithDebInfo>:-O2;-g> # RelWithDebInfo配置：O2优化，带调试信息
#     $<$<CONFIG:MinSizeRel>:-Os> # MinSizeRel配置：最小体积优化
# )

# target_compile_definitions(LlamaCtxAttn PRIVATE
#     $<$<CONFIG:Debug>:DEBUG_MODE> # Debug配置下定义 DEBUG_MODE
#     $<$<CONFIG:Release>:NDEBUG> # Release配置下定义 NDEBUG (禁用assert)
#     $<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:NDEBUG> # 其他优化配置也禁用assert
# )

# 如果库本身需要CUDA支持
# find_package(CUDA REQUIRED)
# target_link_libraries(MyStaticLib PRIVATE CUDA::cudart) # 假设库内部使用了CUDA运行时



add_library(LlamaselfAttn STATIC masked_self_attention.cpp)
set_property(TARGET LlamaselfAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaselfAttn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaselfAttn PUBLIC
                             -lcudart
                             -lcudadevrt
                             fused_decoder_self_attention
                             qkv_bias_and_rope
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(LlamaselfAttn PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()
# # 核心：使用生成器表达式设置编译选项和定义
# target_compile_options(LlamaselfAttn PRIVATE
#     $<$<CONFIG:Debug>:-O0;-g> # Debug配置：无优化，带调试信息
#     $<$<CONFIG:Release>:-O3> # Release配置：O3优化
#     $<$<CONFIG:RelWithDebInfo>:-O2;-g> # RelWithDebInfo配置：O2优化，带调试信息
#     $<$<CONFIG:MinSizeRel>:-Os> # MinSizeRel配置：最小体积优化
# )

# target_compile_definitions(LlamaselfAttn PRIVATE
#     $<$<CONFIG:Debug>:DEBUG_MODE> # Debug配置下定义 DEBUG_MODE
#     $<$<CONFIG:Release>:NDEBUG> # Release配置下定义 NDEBUG (禁用assert)
#     $<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:NDEBUG> # 其他优化配置也禁用assert
# )
